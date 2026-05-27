"""Robot feasibility check for surface-cover trajectories.

Two-phase pipeline:

  Phase 1 — Weighted A* over per-waypoint IK pools
    Build a per-waypoint pool of (q_arm, quat_wxyz) by running analytical IK
    at every (roll, pitch, yaw) triple in the cartesian product of
    cfg.roll_search_offsets_deg (rotation about tool X, default [0.0]),
    cfg.pitch_search_offsets_deg (rotation about tool Y, default [0.0])
    and cfg.yaw_search_offsets_deg (rotation about tool Z, default
    18 offsets at 10° spacing covering one half-circle, including 0° =
    the planner-assigned canonical quat). Dedup pool entries by rounding
    q_arm to 1e-3 rad. The A* state is (layer, ik_idx); edge cost
    is the weighted L2 arm-joint distance using JOINT_DIST_WEIGHTS =
    [2,2,2,1,1,1,1]. Two speed-ups:
      • L_inf joint-jump prune: skip edges with max|q_next-q_cur| >
        cfg.jump_threshold before pushing.
      • Weighted heuristic: f = g + W·(layers_remaining)·BASE_COST, with
        W = cfg.heuristic_weight (>1 narrows the search). Aggressive early
        stop: the first node popped at the final layer wins.
    Only endpoint self-collision is checked in Phase 1; swept collisions
    between adjacent IK picks fall through to Phase 2, which slerp-densifies
    at `ee_speed / hz` step and runs endpoint collision per sub-step.
    On failure, the deepest layer reached is reported.

  Phase 2 — densification (two methods, dispatched on cfg.densify_method)
    Both methods walk consecutive (selected_pos, selected_quat) pairs and
    SLERP-densify at step = ee_speed / hz with continuity-preferred IK
    (sort by ||q_arm - cur_arm|| ascending, early-exit on first
    collision-free). They differ in how the time grid is built:

      "slerp" (legacy) — t_global accumulates as path-time (Σ seg_dist/N)
        and emit one QPoint per sub-step. Joint velocity tracks the
        Jacobian; joint acceleration is unbounded at user-waypoint
        boundaries. Phase 4 later restamps t_global = i · dt.

      "topp_smooth" (default) — collect the fine-grid joint path without
        QPoint emission, retime it with `topp.topp_smooth` (analytic
        forward-backward Bobrow/Pham on β = (ds/dt)² with corner-accel
        caps + iterative tightening so vel + accel are both feasible by
        construction), then resample onto a uniform `1 / hz` grid via
        `topp.resample_to_uniform_grid`. Emitted QPoints carry control-
        time `t_global = k / hz` directly and a smooth joint trajectory;
        Phase 4 becomes a no-op when resample.joint_vel_limits matches
        joint_vel_limits and Phase 5's topp_retime collapses to identity.

    Cartesian deviation is validated per sub-step for both methods. Joint
    velocities are checked on the full densified trajectory.

Torso (rail + waist) and the off-arm are frozen at `initial_q` throughout.
Designed to run inside the SudoDeploy docker container, where ampl,
wbc_py, and RoboSkill are already installed.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# `ampl_motion-0.0.2` has a `NDArray[np.bool]` type hint that crashes on
# NumPy ≥ 1.20. Re-add the alias before the wheel is imported. Module-level
# attribute assignment bypasses NumPy's deprecation `__getattr__`.
import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]


HBMP_COMPONENT_DOF_INDICES: dict[str, list[int]] = {
    "rail": [0],
    "waist": [1],
    "left_arm": list(range(2, 9)),
    "right_arm": list(range(9, 16)),
}

# Per-arm-DOF weights for the arm-joint-distance metric used by the
# continuity-preferred IK selectors. Joints 1–3 (proximal/shoulder) carry a
# 2× penalty so the search prefers solutions that move joints 4–7
# (elbow + wrist) over joints 1–3 when both are available. Applied as
# weighted L2: dist = sqrt(sum(w * (q_a - q_b)**2)).
JOINT_DIST_WEIGHTS: np.ndarray = np.array(
    [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64
)


def _weighted_arm_dist(q_a: np.ndarray, q_b: np.ndarray) -> float:
    diff = np.asarray(q_a, dtype=np.float64) - np.asarray(q_b, dtype=np.float64)
    return float(np.sqrt(np.sum(JOINT_DIST_WEIGHTS * diff * diff)))


# ---- ampl_motion sidecar (collision-only) -------------------------------
#
# HBMP's `Robot_T2DA2.check_self_collision()` uses an internal sphere model
# with its own ignore-pair list that surface_cover_planner historically
# relied on. The optimizer subpackage (`optimize/`) audits using
# `ampl_motion`'s convex-mesh check (`colcvx` from `ampl_motion-0.0.2`),
# and the two disagree — HBMP says collision-free for trajectories
# `colcvx` flags as 40/40 colliding (link geometry & ignore-pair lists
# are subtly different). To make the planner and optimizer agree, we
# route every collision query in this file through
# `ampl_motion.collision_free_cvx` instead of the HBMP layer. `Robot_T2DA2`
# is kept solely for IK/FK — its kinematics are unchanged.
#
# The checker is a module-level singleton populated by `_setup_collision_checker`
# at the top of `check_trajectory` and consumed by `_is_self_colliding`
# (the single replacement for `agent.update_col_self() +
# agent.check_self_collision() != ColGroup.NONE`).
_CHECK_MAGIC_DISTANCE = 0.02
_collision_checker: dict[str, Any] = {
    "kin": None,           # ampl.KinHB11 — kinematics (FK, IK, Jacobian).
                            # Used both for pushing poses into colcvx and
                            # as the planner's IK / FK source. Replaces the
                            # prior `Robot_T2DA2` (which went away with the
                            # full swap to ampl_motion).
    "colcvx": None,         # ampl.ColConvex — convex-mesh collision scene.
    "link_to_id": None,     # dict[str, int]
    "frames": None,         # list[str] — kin.get_actuated_frames(), cached.
    "wall": None,           # tuple[list[float], list[float], list[float]]
                            # (x_bounds, y_bounds, z_bounds) for the
                            # diagnostic printout. The actual workspace
                            # check happens inside colcvx via
                            # configure_workspace; this dict entry is for
                            # `_diag_cause`'s error message only.
    "magic": _CHECK_MAGIC_DISTANCE,
    "robot_name": "hb11",
    # ---- attached-tool + world-obstacle bookkeeping ----
    # Populated when `cfg.attached_tool` is non-null. The TCP frame enum
    # (`kin.Frames.left_link_tactile_center` or right) is looked up at
    # setup; the 4x4 local offset (tool relative to the gripper TCP) is
    # stored as a numpy matrix so `_is_self_colliding` can compose
    # `T_world_tool = T_world_gripper @ local_offset` per query and push
    # the 7-vec into every `attached_tool[/sub_name]` colcvx entity.
    "tool_entity": [],               # list[str] of colcvx entity names
    "tool_tcp_frame": None,          # ampl HBFrames enum value
    "tool_local_offset_4x4": None,   # (4, 4) float64 in TCP frame
    # World obstacle names ever registered (just for diagnostics — their
    # poses are set once at add() time and never updated).
    "world_obstacle_names": [],
    # Every entity this module registers into `colcvx` (attached_tool +
    # world_obstacles, including any sub-hull entity names produced by
    # the multi-hull decomposition path). Used by `_fmt_entity` to
    # decide whether to suffix `#component` — robot links registered by
    # `ampl_motion.create_robot_system` are single-component and stay
    # bare.
    "user_added_entities": set(),
    # ---- Phase-1 collision-pair histogram ----
    # Every `_endpoint_collides` hit harvests its full pair list (via
    # `_self_collision_pairs`) and increments one bucket per pair so the
    # final stats table answers "which entity pair rejected most of my
    # Phase-1 IK candidates?". Counter keys are canonicalized as sorted
    # 2-tuples `(min(a,b), max(a,b))` so a/b ordering from
    # `colcvx.process_collisions()` doesn't split a single physical pair
    # across two buckets. Reset inside `_setup_collision_checker` so
    # re-runs of `check_trajectory` start from zero.
    "collision_pair_counts": Counter(),
    "n_failing_endpoint_checks": 0,
}


def _load_convex_hulls(path: str) -> list[tuple[str, "ampl.VSCHf"]]:
    """Load a mesh file as a list of `(sub_name, ampl.VSCHf)` pairs.

    A multi-shard convex decomposition is packed in one `.obj` as
    multiple top-level `o hull_N` (or `g`) entries. Each entry becomes
    its own VSCHf so the broad-phase narrow-phase iteration sees the
    cavity between shards. `trimesh.load(..., force="mesh")` would
    concatenate the Scene into a single Trimesh and dedupe shared
    boundary vertices, collapsing the decomposition into one connected
    component — exactly the failure mode the `mesh.convex_hull` warning
    used to call out. Plain single-mesh OBJs return a one-element list
    with `sub_name=""` so callers stay uniform.
    """
    import trimesh

    loaded = trimesh.load(path, force=None, process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return [("", _trimesh_to_vschf(loaded))]
    if not isinstance(loaded, trimesh.Scene):
        raise RuntimeError(
            f"unsupported asset type {type(loaded).__name__} at {path}"
        )
    pairs: list[tuple[str, "ampl.VSCHf"]] = []
    for sub_name, geom in loaded.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        pairs.append((sub_name, _trimesh_to_vschf(geom)))
    if not pairs:
        raise RuntimeError(f"no Trimesh geometries in Scene at {path}")
    return pairs


def _trimesh_to_vschf(mesh: "trimesh.Trimesh") -> "ampl.VSCHf":
    import ampl

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    return ampl.VSCHf_from_mesh((verts, faces))


def _pose_dict_to_4x4(pose_dict: dict | None) -> np.ndarray:
    """Parse a `{translation, rotation_mode, quat|euler_xyz_deg|matrix}` block
    into a 4x4 rigid transform. Same schema the existing `mesh.transform`
    and `tcp.tool_offset` YAML blocks use; identity when empty.
    """
    if not pose_dict:
        return np.eye(4, dtype=np.float64)
    from scipy.spatial.transform import Rotation as R

    translation = np.asarray(
        pose_dict.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64
    )
    mode = pose_dict.get("rotation_mode", "quat")
    if mode == "quat":
        q_wxyz = np.asarray(
            pose_dict.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64
        )
        rot = R.from_quat(
            np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        ).as_matrix()
    elif mode == "euler_xyz_deg":
        rpy = np.asarray(
            pose_dict.get("euler_xyz_deg", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        rot = R.from_euler("XYZ", rpy, degrees=True).as_matrix()
    elif mode == "matrix":
        rot = np.asarray(
            pose_dict.get("matrix", np.eye(3).tolist()), dtype=np.float64
        )
        if rot.shape != (3, 3):
            raise ValueError(f"pose.matrix must be 3x3, got {rot.shape}")
    else:
        raise ValueError(f"unknown rotation_mode: {mode!r}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = translation
    return T


def _matrix_to_pose7(matrix_4x4: np.ndarray) -> np.ndarray:
    """Convert a 4x4 rigid transform to `[qw, qx, qy, qz, tx, ty, tz]`
    float32, the layout `colcvx.add` / `colcvx.update_pose` expect
    (per `ampl/_utils/col.py:79`). `ampl.Tf(matrix)` requires F-order
    float64 (see `ampl/_core/__init__.pyi:1276`), so we force the
    layout here even if the caller passed a C-order matrix.
    """
    import ampl

    m = np.asfortranarray(np.asarray(matrix_4x4, dtype=np.float64))
    return np.asarray(ampl.Tf(m).as_array(), dtype=np.float32)


def _default_tool_ignore_links(link_to_id: dict, arm: str) -> list[str]:
    """Default `ignore_links` for an attached tool: the last three arm
    links of the active arm (5/6/7 — the wrist that physically grips the
    tool). Any tool mesh registered against the gripper will naturally
    overlap those links; without this default every Phase-1 IK pool
    entry would self-collide.
    """
    prefix = "left-" if arm == "left_arm" else "right-"
    return [
        name
        for name in link_to_id
        if name.startswith(prefix)
        and any(suffix in name for suffix in ("arm_5", "arm_6", "arm_7", "hand", "tactile"))
    ]


def _setup_collision_checker(cfg: "RobotCheckConfig", robot_name: str = "hb11") -> None:
    """Build the `ampl_motion` mesh-collision sidecar.

    Called once at `check_trajectory` entry; populates the module-level
    `_collision_checker` dict. Idempotent — re-calling with the same args
    rebuilds the scene (cheap; the wheel ships hb11 geometry as JSON).
    """
    import ampl_motion as am

    (
        _meta,
        kin,
        colcvx,
        _colsph,
        _wbc,
        link_to_id,
        ignores,
        _cvx_data,
        _sph_data,
    ) = am.create_robot_system(robot_name)
    am.update_ignore(colcvx, link_to_id, ignores)

    # Workspace box. The HBMP layer used `agent.set_wall(x, y, z)`; the
    # ampl_motion equivalent is `configure_workspace(col, link_to_id,
    # includes, min3, max3)` where `includes` is the link names to check
    # against the bounding box.
    #
    # HBMP only checks the *arm* links against the wall — the torso sits
    # below the configured z_min=0.6 m (the surface-cover working plane)
    # and would otherwise fail every pose. To match HBMP's behavior we
    # restrict `includes` to link names that start with `left-` or
    # `right-` (the two arm chains, including hand/tactile frames).
    # `base_link` and the torso links are excluded.
    if cfg.wall:
        min3 = [
            float(cfg.wall.get("x", [-math.inf, math.inf])[0]),
            float(cfg.wall.get("y", [-math.inf, math.inf])[0]),
            float(cfg.wall.get("z", [-math.inf, math.inf])[0]),
        ]
        max3 = [
            float(cfg.wall.get("x", [-math.inf, math.inf])[1]),
            float(cfg.wall.get("y", [-math.inf, math.inf])[1]),
            float(cfg.wall.get("z", [-math.inf, math.inf])[1]),
        ]
        includes = [
            name for name in link_to_id
            if name.startswith("left-") or name.startswith("right-")
        ]
        am.configure_workspace(colcvx, link_to_id, includes, min3=min3, max3=max3)

    _collision_checker["kin"] = kin
    _collision_checker["colcvx"] = colcvx
    _collision_checker["link_to_id"] = link_to_id
    _collision_checker["frames"] = list(kin.get_actuated_frames())
    _collision_checker["robot_name"] = robot_name
    # Reset attached-tool / world-obstacle slots in case of re-setup
    # (idempotent w.r.t. fresh cfg, but bookkeeping has to clear).
    # `tool_entity` is a list of colcvx entity names because the attached
    # tool may be a convex decomposition (one entity per sub-hull).
    _collision_checker["tool_entity"] = []
    _collision_checker["tool_tcp_frame"] = None
    _collision_checker["tool_local_offset_4x4"] = None
    _collision_checker["world_obstacle_names"] = []
    _collision_checker["user_added_entities"] = set()
    _collision_checker["collision_pair_counts"] = Counter()
    _collision_checker["n_failing_endpoint_checks"] = 0
    # Stash the wall bounds for `_diag_cause`'s failure message. The colcvx
    # workspace was already configured above; this is the printout source
    # only.
    if cfg.wall:
        _collision_checker["wall"] = (
            list(cfg.wall.get("x", [-math.inf, math.inf])),
            list(cfg.wall.get("y", [-math.inf, math.inf])),
            list(cfg.wall.get("z", [-math.inf, math.inf])),
        )
    else:
        _collision_checker["wall"] = (
            [-math.inf, math.inf],
            [-math.inf, math.inf],
            [-math.inf, math.inf],
        )

    # ---- Register optional attached tool and world obstacles ----
    # Both share the same `colcvx.add(entity_id, pose, cvh_group)` API
    # and live alongside robot link entities in the same scene. The
    # broad-phase `colcvx.process_collisions()` already iterates over
    # every registered entity-pair (minus `disabled_collision_pairs`),
    # so no further wiring is needed: the existing narrow-phase loop in
    # `_is_self_colliding` picks them up automatically.
    if cfg.attached_tool:
        tool_path = cfg.attached_tool["path"]
        local_offset_4x4 = _pose_dict_to_4x4(cfg.attached_tool.get("local_offset"))
        ignore_links = cfg.attached_tool.get("ignore_links")
        if ignore_links is None:
            ignore_links = _default_tool_ignore_links(link_to_id, cfg.arm)
        sub_hulls = _load_convex_hulls(tool_path)
        tool_entity_names: list[str] = []
        for sub_name, cvh in sub_hulls:
            entity = f"attached_tool/{sub_name}" if sub_name else "attached_tool"
            colcvx.add(entity, cvh_group=cvh, local_pc=None)
            for link_name in ignore_links:
                if link_name in link_to_id:
                    colcvx.disable_collision(entity, link_name)
            tool_entity_names.append(entity)
            _collision_checker["user_added_entities"].add(entity)
        tcp_frame = (
            kin.Frames.left_link_tactile_center
            if cfg.arm == "left_arm"
            else kin.Frames.right_link_tactile_center
        )
        _collision_checker["tool_entity"] = tool_entity_names
        _collision_checker["tool_tcp_frame"] = tcp_frame
        _collision_checker["tool_local_offset_4x4"] = local_offset_4x4
        print(
            f"[robot_check] attached tool '{Path(tool_path).name}' "
            f"-> TCP({cfg.arm}); {len(sub_hulls)} sub-hull(s); "
            f"ignoring {len(ignore_links)} link pair(s)"
        )

    for obs in cfg.world_obstacles or []:
        name = obs["name"]
        path = obs["path"]
        if name == "attached_tool" or name in link_to_id:
            raise ValueError(
                f"world_obstacle name {name!r} conflicts with an existing "
                f"colcvx entity (robot link or attached_tool)"
            )
        sub_hulls = _load_convex_hulls(path)
        pose7 = _matrix_to_pose7(_pose_dict_to_4x4(obs))
        for sub_name, cvh in sub_hulls:
            entity = f"{name}/{sub_name}" if sub_name else name
            if entity != name and (entity in link_to_id):
                raise ValueError(
                    f"world_obstacle sub-entity {entity!r} conflicts with an "
                    f"existing colcvx entity"
                )
            colcvx.add(entity, pose=pose7, cvh_group=cvh, local_pc=None)
            _collision_checker["world_obstacle_names"].append(entity)
            _collision_checker["user_added_entities"].add(entity)
        print(
            f"[robot_check] world obstacle '{name}' from "
            f"'{Path(path).name}'; {len(sub_hulls)} sub-hull(s); at xyz="
            f"({pose7[4]:.3f}, {pose7[5]:.3f}, {pose7[6]:.3f})"
        )


def _push_scene_poses(q_full: np.ndarray) -> None:
    """Run FK for `q_full` and push every entity pose into the colcvx scene.

    Updates poses for: (1) the 16 robot link entities (from
    `kin.get_fk_actuated_frame()`), and (2) every `attached_tool` sub-
    hull entity, composed as `T_world_gripper @ tool_local_offset`.
    World obstacles aren't touched — their poses were set once at
    `add()` time and never change.
    """
    kin = _collision_checker["kin"]
    colcvx = _collision_checker["colcvx"]
    frames = _collision_checker["frames"]
    kin.update_kin(q_full, None, False)
    fk = kin.get_fk_actuated_frame()  # (16, 7) — pose per actuated frame
    for i, name in enumerate(frames):
        colcvx.update_pose(name, fk[i])

    tool_entities = _collision_checker["tool_entity"]
    if tool_entities:
        tcp_frame = _collision_checker["tool_tcp_frame"]
        local_4x4 = _collision_checker["tool_local_offset_4x4"]
        tcp_4x4 = np.asarray(kin.get_fk(tcp_frame).matrix, dtype=np.float64)
        world_4x4 = tcp_4x4 @ local_4x4
        pose7 = _matrix_to_pose7(world_4x4)
        for entity in tool_entities:
            colcvx.update_pose(entity, pose7)


def _is_self_colliding(q_full: np.ndarray) -> bool:
    """Mesh-based binary collision check via `ampl_motion.colcvx`.

    Returns `True` when any registered entity-pair (robot self-collision,
    robot vs attached tool, robot vs world obstacle, tool vs world
    obstacle) closer than `_CHECK_MAGIC_DISTANCE` (2 cm), OR when any
    arm link violates the workspace box.

    `q_full` is the 16-DOF [rail, waist, left_arm(7), right_arm(7)] vector.
    """
    import ampl

    if _collision_checker["kin"] is None:
        raise RuntimeError(
            "_collision_checker is not initialized — call "
            "_setup_collision_checker(cfg) first (check_trajectory does this)"
        )
    _push_scene_poses(q_full)
    colcvx = _collision_checker["colcvx"]
    if not colcvx.is_workspace_safe():
        return True
    magic = _collision_checker["magic"]
    for ent_a, comp_a, ent_b, comp_b in colcvx.process_collisions():
        verts_a = colcvx.get_world_vertices(ent_a, comp_a)
        verts_b = colcvx.get_world_vertices(ent_b, comp_b)
        if ampl.collision_check_cvx_cvx(verts_a, verts_b) < magic:
            return True
    return False


def _self_collision_pairs(
    q_full: np.ndarray,
) -> list[tuple[str, str, float]]:
    """List every (entity_a, entity_b, distance) pair within the magic
    margin. Diagnostic counterpart to `_is_self_colliding`. Entities
    include robot links, the attached tool, and any world obstacles
    registered at setup. Distances are mesh-mesh from
    `ampl.collision_check_cvx_cvx`. Component indices are folded into
    the entity strings (`f"{entity}#{component}"` when a single entity
    holds multiple sub-hulls from a convex decomposition; bare entity
    name when the VSCHf is single-component) so multi-hull granularity
    survives without changing the tuple shape every call site depends on.
    """
    import ampl

    if _collision_checker["kin"] is None:
        return []
    _push_scene_poses(q_full)
    colcvx = _collision_checker["colcvx"]
    magic = _collision_checker["magic"]
    out: list[tuple[str, str, float]] = []
    for ent_a, comp_a, ent_b, comp_b in colcvx.process_collisions():
        verts_a = colcvx.get_world_vertices(ent_a, comp_a)
        verts_b = colcvx.get_world_vertices(ent_b, comp_b)
        d = float(ampl.collision_check_cvx_cvx(verts_a, verts_b))
        if d < magic:
            out.append((_fmt_entity(ent_a, comp_a), _fmt_entity(ent_b, comp_b), d))
    return out


def _fmt_entity(entity: Any, component: Any) -> str:
    """`f"{entity}#{component}"` when the entity was registered by this
    module (`attached_tool[/...]` or any `world_obstacles[*].name[/...]`)
    — these wrap an `ampl.VSCHf` that may hold a multi-shard convex
    decomposition, so the component index is the only way to tell which
    sub-hull collided. Robot-link entities (added internally by
    `ampl_motion.create_robot_system`) are single-component, so they
    print bare without a spurious `#0` suffix.
    """
    ent_str = str(entity)
    user_added: set = _collision_checker.get("user_added_entities") or set()
    if ent_str in user_added:
        return f"{ent_str}#{component}"
    return ent_str


@dataclass
class RobotCheckConfig:
    enabled: bool = False
    arm: str = "right_arm"  # left_arm | right_arm
    hz: float = 30.0
    ee_speed: float = 0.05  # m/s
    min_waypoints: int = 2
    nb_redundant_search: int = 512
    max_cartesian_deviation: float = 0.002
    ignore_rotation: bool = False  # currently unused (kept for parity with skill)
    hbmp_arm_left: str = "hb11_left"
    hbmp_arm_right: str = "hb11_right"
    hbmp_torso: str = "hb11_torso"
    wall: dict[str, list[float]] = field(default_factory=dict)
    # Optional convex mesh held by the active arm's gripper. The pose
    # follows the TCP frame (`*_link_tactile_center`) plus an optional
    # local offset (4x4 in TCP frame). Schema:
    #   {path: str,
    #    local_offset: {translation, rotation_mode, quat|euler_xyz_deg|matrix},
    #    ignore_links: list[str] | None}
    # When `ignore_links` is None / omitted, the last three arm links of
    # `arm` (arm_5/6/7 + hand/tactile) are auto-ignored — the wrist
    # physically holds the tool. The mesh is loaded via trimesh; pass a
    # pre-decomposed convex shard (the `mesh.convex_hull` defensive
    # wrap is cheap but only catches single-piece geometry).
    attached_tool: dict[str, Any] | None = None
    # Optional list of fixed-world convex obstacles checked alongside the
    # robot's self-collision. Each entry is:
    #   {name: str, path: str, translation: [x,y,z],
    #    rotation_mode: 'quat'|'euler_xyz_deg'|'matrix',
    #    quat: [w,x,y,z], euler_xyz_deg: [r,p,y], matrix: [[..],[..],[..]]}
    # Pose is set once at setup and never updated. `name` must be unique
    # within the colcvx scene (must not collide with `attached_tool` or
    # any of the robot's link names).
    world_obstacles: list[dict[str, Any]] = field(default_factory=list)
    initial_q: list[float] | None = None
    joint_vel_limits: list[float] | None = None  # length-7 (arm DOFs)
    # Yaw offsets (deg) around tool Z that seed the per-waypoint IK pool the
    # A* search expands over. Each entry rotates the planner-assigned quat
    # locally around tool Z (= -surface normal in align_z_to_normal mode), so
    # the orientation tolerance follows the surface normal regardless of the
    # waypoint's base yaw. The 0.0 entry is the canonical / planner-assigned
    # quat; non-zero entries widen the IK pool. Default = 18 values at 10°
    # spacing covering one half-circle (0–170°); for non-Z-symmetric tools
    # override with a full-circle list.
    yaw_search_offsets_deg: list[float] | None = None
    # Pitch offsets (deg) around tool Y. Mirrors the yaw axis above: each entry
    # tilts the canonical quat by `dy` deg around the tool's local Y, taken
    # cartesian-product with the yaw and roll offsets to seed the IK pool.
    # Useful for tools whose contact point is offset from the tool axis (a
    # small pitch changes the contact angle without leaving the surface).
    # Default = [0], i.e. no pitch search; widen via YAML when needed (e.g.
    # [-10, 0, 10]).
    pitch_search_offsets_deg: list[float] | None = None
    # Roll offsets (deg) around tool X. Same intrinsic-XYZ-Euler family as the
    # yaw and pitch axes above; each entry rolls the canonical quat by `dr`
    # deg around the tool's local X. Taken cartesian-product with the
    # pitch/yaw lists to seed the IK pool. Default = [0], i.e. roll is held
    # at zero (legacy behaviour); widen via YAML when the tool's contact is
    # roll-tolerant (e.g. a brush whose orientation about its long axis can
    # be relaxed).
    roll_search_offsets_deg: list[float] | None = None
    # Phase 1 search strategy:
    #   "astar"        — weighted A* with endpoint-only collision (uses
    #                    heuristic_weight, heuristic_base_cost,
    #                    jump_threshold). Default; current behaviour. Swept
    #                    collisions between adjacent IK picks are caught by
    #                    Phase 2's dense slerp endpoint checks.
    #   "kshortest_dp" — iterative-mask DP. Pre-caches per-layer edge-cost
    #                    matrices, runs a single-best forward DP each
    #                    iteration (with previously-masked nodes forced to
    #                    inf), reconstructs the shortest path, runs the
    #                    endpoint collision check, masks each colliding
    #                    node, and re-runs until the path is clean or
    #                    max_iterations is reached. Uses jump_threshold,
    #                    max_iterations; ignores heuristic_*.
    search_method: str = "astar"
    # Iteration cap for the iterative-mask DP loop (kshortest_dp only).
    # Each iteration: single-best forward DP → reconstruct path →
    # collision-check → mask colliding nodes → retry. The loop stops
    # early when the path is collision-free, fails when the DP becomes
    # infeasible (mask removed every parent at some layer), or fails
    # when this cap is reached. Raise when scenes have many local
    # collisions; lower for tighter wall-clock budgets.
    max_iterations: int = 20
    # Weighted A* knobs (Phase 1 graph search; consumed when search_method == "astar").
    heuristic_weight: float = 5.0          # W: scales the to-go heuristic; W>1 → greedy
    heuristic_base_cost: float = 0.1       # per-remaining-layer constant (rad)
    jump_threshold: float = 1.5            # L_inf joint-jump prune (rad) between layers
    # Soft acceleration cost added to the kshortest_dp edge cost. The DP
    # edge cost becomes `weighted_L2(Δq) + accel_weight · weighted_L2(q_next
    # − 2·q_cur + q_grandparent)`, with the same JOINT_DIST_WEIGHTS reused
    # on the second-order term so dist and accel stay in matching units.
    # The grandparent is looked up lazily as `parent[L-1][i]` (the previous
    # layer's already-computed argmin), so the DP state stays (L, j) and
    # per-layer work stays O(P²) — not globally optimal under the new cost,
    # but cheap. `0.0` (default) is byte-equivalent to the dist-only DP.
    # Ignored by the A* search (`search_method == "astar"`).
    accel_weight: float = 0.0
    # When True, Phase 1 anchors the IK chain at `initial_q[arm_indices]`:
    # layer-0 candidates farther than `jump_threshold` (L∞) from the initial
    # arm pose are pruned, the surviving candidates seed the search with
    # cost = weighted-L2 distance from the anchor, and an empty post-prune
    # layer-0 pool raises `RuntimeError` (the deployment precondition "initial
    # pose is close to first waypoint" was violated; silent recovery would
    # mislead). Both `astar` and `kshortest_dp` honour the flag. Default
    # False = legacy behaviour: layer 0 starts at cost 0 with every
    # collision-free IK accepted, and `q_arm[0]` is chosen purely by chain
    # cost (may be far from `initial_q`).
    keep_init: bool = False
    # Per-joint preference filter applied to each Phase 1 IK pool entry
    # AFTER convergence + dedup, BEFORE the A* / DP search sees the pool.
    # Keys are arm-DOF indices (0..6 → j0..j6 of the 7-DOF arm). Each value
    # is a `(lo, hi)` tuple in radians; either bound may be None for
    # one-sided filtering. Example: {1: (0.0, None)} drops every candidate
    # with q_arm[1] < 0. None / empty (default) disables the filter. Pools
    # emptied by the filter surface as the same `kind="ik"` failure as
    # IK-convergence-empty pools, with an extended cause string reporting
    # how many candidates were dropped. The IK-grid PNG is unaffected — it
    # still reflects raw IK convergence. Phase 3's `_build_refine_pool`
    # does NOT consult this filter.
    joint_preference_filters: dict[int, tuple[float | None, float | None]] | None = None
    # Rigid transform from gripper TCP frame to user-defined tool tip frame.
    # When set, waypoint poses are interpreted as tool-tip poses; IK targets
    # are computed via T_world_gripper = T_world_tool @ inv(tool_offset), and
    # achieved tool position is computed via FK(gripper) @ tool_offset.
    tool_offset: np.ndarray | None = None  # (4, 4) or None == identity
    # When True, the full per-waypoint IK pool (every (q_arm, quat_wxyz) the
    # Phase 1 search expanded over, before A* selected one) is attached to
    # CheckResult.ik_pools so plan.py can serialize it into the output JSON.
    save_all_ik_results: bool = False
    # ---------------- Phase 3: jerk-region refinement ----------------
    # Phase 3 detects sub-steps in the densified Phase 2 q_trajectory whose
    # L_inf joint jump |Δq_arm|∞ exceeds `refine_jump_threshold` (rad), groups
    # them into anchored regions (padded by `refine_region_buffer` on each
    # side), and runs a per-region weighted A* over a relaxed pool of
    # perturbed targets to smooth the joint path. The relaxation budget is:
    #   • yaw   ∈ refine_yaw_offsets_deg   (deg, around tool Z, perturbing
    #     each step's slerp-canonical quat)
    #   • pitch ∈ refine_pitch_offsets_deg (deg, around tool Y, same)
    #   • xy    ∈ refine_xy_offsets_m       (m, in the surface PlaneFrame
    #     u/v axes — supplied via plane_u_axis / plane_v_axis below)
    # Per-region failures keep the Phase 2 result intact for that region.
    refine_enabled: bool = False
    refine_jump_threshold: float = 0.1
    refine_region_buffer: int = 2
    refine_yaw_offsets_deg: list[float] | None = None
    refine_pitch_offsets_deg: list[float] | None = None
    refine_xy_offsets_m: list[tuple[float, float]] | None = None
    refine_nb_redundant_search: int = 8
    # Surface PlaneFrame axes used to offset target positions in xy. Required
    # when refine_enabled=True. PlanResult.plane.u_axis / v_axis (3,) each.
    plane_u_axis: np.ndarray | None = None
    plane_v_axis: np.ndarray | None = None
    # ---------------- Phase 4: joint-velocity resampling ----------------
    # Walk the post-Phase-3 q_trajectory and splice linearly-interpolated
    # joint-space samples into adjacent pairs that violate joint_vel_limits.
    # Trajectory length grows; control rate stays at cfg.hz. After Phase 4,
    # `t_global` switches semantics from path-time to control-rate time.
    resample_enabled: bool = False
    # When True, every inserted sample is checked against self/wall collision
    # via _endpoint_collides; collisions are logged as `kind="resample_collision"`
    # failures but the point is kept (so the trajectory stays contiguous).
    resample_check_collision: bool = False
    # Safety cap: abort the resample (leaving q_trajectory untouched) and emit
    # `kind="resample_overflow"` if the planned post-resample length exceeds
    # max_expansion * len(q_trajectory). Protects against misconfigured limits.
    resample_max_expansion: int = 1000
    # Per-joint speed cap (rad/s) used by Phase 4 when picking insert counts.
    # Length-7 (arm DOFs). When None, falls back to top-level `joint_vel_limits`
    # (so the resampler target equals the velocity-check limit). Set this to
    # a tighter list (e.g. [0.1]*7) to slow Phase 4's output below the
    # hardware limit — useful when you want the velocity check at the
    # hardware limit but the executed trajectory smoother.
    resample_joint_vel_limits: list[float] | None = None
    # ---------------- Phase 2 densification method ----------------
    # `slerp`        — legacy: uniform-arc-length SLERP + per-step continuity
    #                  IK at `step = ee_speed / hz`. `t_global` is path-time
    #                  (Σ seg_dist / N) until Phase 4 restamps it.
    # `topp_smooth`  — default: same SLERP + per-step IK pipeline at the same
    #                  geometric resolution, but the resulting joint path is
    #                  retimed with an analytic forward-backward TOPP pass
    #                  (vel + accel feasible by construction) and then
    #                  resampled onto a uniform `1 / hz` control grid.
    #                  `t_global = k / hz` directly (control time). Requires
    #                  `densify_joint_accel_limits`.
    densify_method: str = "topp_smooth"
    # Per-joint acceleration cap (rad/s²). Length-7 (arm DOFs). Required when
    # densify_method == "topp_smooth"; ignored by `slerp`. The shape contract
    # matches `joint_vel_limits` — a scalar broadcasts to length-7 via the
    # plan.py loader, and a non-7 sequence raises at load time.
    densify_joint_accel_limits: list[float] | None = None
    # Fractional slack below `joint_vel_limits` / `densify_joint_accel_limits`
    # used by the TOPP pass. Mirrors `optimize/retime.py:topp_retime`'s
    # `vel_headroom`. Default 5%.
    densify_vel_headroom: float = 0.05
    densify_accel_headroom: float = 0.05
    # Per-segment time floor inside TOPP (seconds). Sub-`1 / hz` so TOPP can
    # outpace control rate where slack exists; the uniform-grid resampler
    # then drops back to `1 / hz` for emission.
    densify_dt_min: float = 0.005
    # Pass-2 timing strategy inside `_phase2_topp_smooth_densify`.
    #   "topp_smooth" — forward-backward TOPP with `densify_dt_min` floor and
    #                   discrete corner-accel cap (default; planner Phase 2).
    #   "envelope"    — per-segment dt = max(t_vel, t_accel); no floor, no
    #                   forward-backward smoothing. Used by local_dp's
    #                   re-densify to match its `fastest` DP edge cost.
    densify_timing: str = "topp_smooth"


@dataclass
class QPoint:
    t_global: float
    user_wp_index: int
    sub_index: int
    target_position: np.ndarray
    target_quat_wxyz: np.ndarray
    achieved_position: np.ndarray
    achieved_quat_wxyz: np.ndarray | None
    cartesian_deviation: float
    q_full: np.ndarray | None
    q_rail: float | None
    q_waist: float | None
    q_left_arm: list[float] | None
    q_right_arm: list[float] | None
    ok: bool
    failure_cause: str | None


@dataclass
class CheckResult:
    ok: bool
    n_q_waypoints: int
    total_dist: float
    dt: float
    worst_joint_speed_ratio: float
    worst_dof: int
    joint_speed_ok: bool
    q_trajectory: list[QPoint]
    failures: list[dict]
    # Populated only when cfg.save_all_ik_results is True. One list per user
    # waypoint, each entry a list of (q_arm, quat_wxyz) tuples enumerated by
    # Phase 1's IK pool builder before A* picked one.
    ik_pools: list[list[tuple[np.ndarray, np.ndarray]]] | None = None
    # Always populated when Phase 1a ran (even on early-bail / Phase 1b
    # failure). One flat list[bool] per user waypoint: every IK call's
    # convergence status across the cartesian product (pitch × yaw ×
    # sorted-branch × redundant). Drives the IK-grid PNG renderer; includes
    # pre-collision-filter / pre-dedup outcomes (so it shows the *raw* IK
    # landscape, not the deduped pool). Length is `len(waypoints)` on a clean
    # Phase 1, or `failed_idx + 1` when Phase 1a bailed at an empty pool.
    ik_pool_statuses: list[list[bool]] | None = None
    # Populated when Phase 1 produced a complete chain (both phases). Length =
    # N_waypoints; each entry is the flat ik_pool_statuses index that Phase 1
    # selected at that waypoint, used to overlay the path on the IK-grid PNG.
    ik_selected_flat_indices: list[int] | None = None
    # Always populated when Phase 1a ran. Per-waypoint flat-shaped 3-state
    # list aligned with `ik_pool_statuses`: 0 = untested (search never
    # probed this dedup'd pool entry), 1 = tested + collision-free,
    # 2 = tested + self-collided. Built lazily from whatever
    # `_endpoint_collides` calls Phase 1's search actually made — no extra
    # collision-check budget. Feeds the per-waypoint collision-free count
    # rendered in *_collision_free.png.
    ik_pool_collision_checked: list[list[int]] | None = None
    # Populated only when cfg.refine_enabled is True. Summary stats for the
    # Phase 3 refinement pass: counts and pre/post jump magnitudes.
    refine_stats: dict | None = None
    # Populated only when cfg.resample_enabled is True. Summary stats for
    # the Phase 4 joint-velocity resampling pass: pair counts, insert count,
    # ratios pre/post, expansion factor.
    resample_stats: dict | None = None
    # Populated whenever cfg.densify_method == "topp_smooth" ran (whether or
    # not it succeeded). Summary stats for the Phase 2 TOPP-smoothness
    # densification: vel/accel ratios on the retimed grid, sample counts,
    # total path time. None when the legacy `slerp` method was used.
    densify_stats: dict | None = None
    # Phase-1 endpoint collision histogram. Keys are `"entity_a||entity_b"`
    # strings (canonicalized alphabetic order, "||" delimiter that colcvx
    # entity names never contain), values are the count of Phase-1 IK
    # candidates whose `_endpoint_collides` hit that specific pair. Order
    # is insertion-order = descending by count. None when no Phase-1
    # endpoint collisions were recorded (clean Phase 1 or Phase 1 skipped).
    collision_pair_stats: dict[str, int] | None = None


def _quat_wxyz_to_rotmat(q_wxyz: Sequence[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    q = np.asarray(q_wxyz, dtype=np.float64)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    return R.from_quat(q_xyzw).as_matrix().astype(np.float64)


def _rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    q_xyzw = R.from_matrix(rot).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def _build_target_tf(position: np.ndarray, quat_wxyz: np.ndarray):
    from ampl import Tf

    tf44 = np.eye(4, dtype=np.float64)
    tf44[:3, :3] = _quat_wxyz_to_rotmat(quat_wxyz)
    tf44[:3, 3] = np.asarray(position, dtype=np.float64)
    return Tf(tf44)


def _tool_tf_to_gripper_tf(tf_tool, tool_offset_inv_4x4: np.ndarray | None):
    """Convert a world-frame tool-tip Tf to a world-frame gripper-TCP Tf.

    T_world_gripper = T_world_tool @ inv(T_offset). Returns tf_tool unchanged
    when tool_offset_inv_4x4 is None.
    """
    if tool_offset_inv_4x4 is None:
        return tf_tool
    from ampl import Tf

    return Tf(np.asarray(tf_tool.matrix, dtype=np.float64) @ tool_offset_inv_4x4)


def _gripper_fk_to_tool_position(
    gripper_fk_4x4: np.ndarray, tool_offset_4x4: np.ndarray | None
) -> np.ndarray:
    """Apply T_offset to a gripper FK matrix and return the tool-tip world (3,)."""
    if tool_offset_4x4 is None:
        return np.asarray(gripper_fk_4x4[:3, 3], dtype=np.float64).copy()
    return (np.asarray(gripper_fk_4x4, dtype=np.float64) @ tool_offset_4x4)[:3, 3].copy()


def _gripper_fk_to_tool_quat_wxyz(
    gripper_fk_4x4: np.ndarray, tool_offset_4x4: np.ndarray | None
) -> np.ndarray:
    """Apply T_offset to a gripper FK matrix and return the tool-tip wxyz quat."""
    if tool_offset_4x4 is None:
        rot = np.asarray(gripper_fk_4x4[:3, :3], dtype=np.float64)
    else:
        rot = (np.asarray(gripper_fk_4x4, dtype=np.float64) @ tool_offset_4x4)[:3, :3]
    return _rotmat_to_quat_wxyz(rot)


def _quat_rpy_perturb(
    q_wxyz: np.ndarray, deg_offsets_xyz: tuple[float, float, float]
) -> np.ndarray:
    """Apply a small intrinsic XYZ-Euler offset (deg) to a wxyz quaternion."""
    from scipy.spatial.transform import Rotation as R

    base_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    offset = R.from_euler("XYZ", list(deg_offsets_xyz), degrees=True)
    new = R.from_quat(base_xyzw) * offset
    new_xyzw = new.as_quat()
    return np.array(
        [new_xyzw[3], new_xyzw[0], new_xyzw[1], new_xyzw[2]], dtype=np.float64
    )


def _new_diag(branch_totals: dict[str, list[int]]) -> dict:
    """Return a fresh diag dict matching `_diag_cause`'s expected shape."""
    return {
        "branches": {k: tuple(v) for k, v in branch_totals.items()},
        "best_col_group": None,
        "best_col_parts": [],
        "best_col_dist": float("inf"),
    }


def _accumulate_collision_diag(
    diag: dict, agent: Any, q_full: np.ndarray, q_arm: np.ndarray, cur_arm: np.ndarray
) -> None:
    """Track the nearest-by-joint-distance failing IK candidate for the
    error message. Replaces HBMP's `ColGroup` breakdown with the
    `(link_a, link_b, distance_m)` list from `ampl_motion.colcvx`.
    """
    del agent  # unused — pair info now comes from the colcvx scene
    cand_dist = float(np.linalg.norm(q_arm - cur_arm))
    if cand_dist < diag["best_col_dist"]:
        diag["best_col_dist"] = cand_dist
        pairs = _self_collision_pairs(q_full)
        diag["best_col_group"] = None
        diag["best_col_parts"] = [f"{a}-{b}" for a, b, _d in pairs[:5]]
        diag["best_col_pairs"] = pairs  # full (a, b, dist_m) for the JSON


def _pick_min_dist_collision_free_arm_ik(
    agent: Any,
    cur: np.ndarray,
    tf_target: Any,
    frame: Any,
    arm_indices: list[int],
    nb_redundant_search: int,
) -> tuple[np.ndarray | None, dict]:
    """Sort status-ok IK results by ||q_arm - cur_arm|| ascending; return first collision-free.

    Continuity-preferred selector. Used by both phases: in Phase 1 `cur_arm`
    is the previous waypoint's chosen arm-q (or `initial_q[arm_indices]` at
    the seed); in Phase 2 it is the running arm pose at each slerp sub-step.
    """
    agent.update_kin(cur)
    ik_result = agent.get_ik(tf_target, frame, nb_redundant_search=nb_redundant_search)
    cur_arm = cur[arm_indices]

    candidates: list[tuple[float, np.ndarray, str]] = []
    branch_totals: dict[str, list[int]] = {}
    for which_ik, (status_batch, q_batch) in ik_result.items():
        n_total = len(status_batch)
        n_ok = 0
        for i in range(n_total):
            if not status_batch[i]:
                continue
            n_ok += 1
            q_arm = np.asarray(q_batch[i], dtype=np.float64)
            candidates.append((_weighted_arm_dist(q_arm, cur_arm), q_arm, str(which_ik)))
        branch_totals[str(which_ik)] = [n_total, n_ok, 0]

    candidates.sort(key=lambda c: c[0])
    diag = _new_diag(branch_totals)

    for _dist, q_arm, which_ik in candidates:
        candidate = cur.copy()
        candidate[arm_indices] = q_arm
        # Keep agent.update_kin in sync — downstream FK/IK calls in the
        # planner read from agent's internal state. The collision verdict
        # itself comes from _is_self_colliding (ampl_motion colcvx).
        agent.update_kin(candidate)
        if not _is_self_colliding(candidate):
            n_total, n_ok, _ = branch_totals[which_ik]
            branch_totals[which_ik] = [n_total, n_ok, 1]
            diag["branches"] = {k: tuple(v) for k, v in branch_totals.items()}
            return q_arm.copy(), diag
        _accumulate_collision_diag(diag, agent, candidate, q_arm, cur_arm)

    return None, diag


def _diag_cause(diag: dict, agent: Any = None) -> str:
    """Render the failure cause for `_pick_min_dist_collision_free_arm_ik`.

    `agent` is kept as a keyword-only optional for backwards compatibility
    with existing call sites; the wall bounds come from the
    `_collision_checker` dict now that `Robot_T2DA2.wall()` is gone.
    """
    del agent  # unused — wall info comes from _collision_checker
    total_ok = sum(b[1] for b in diag["branches"].values())
    total_cf = sum(b[2] for b in diag["branches"].values())
    if total_ok == 0:
        return "analytical IK returned NO status-ok solutions (target likely unreachable for arm-only — torso/rail frozen)"
    if total_cf == 0:
        wall = _collision_checker.get("wall") or (
            [-math.inf, math.inf], [-math.inf, math.inf], [-math.inf, math.inf]
        )
        parts_str = ", ".join(diag["best_col_parts"]) or str(diag["best_col_group"])
        return (
            f"all {total_ok} status-ok IK solutions collide "
            f"(nearest at dq={diag['best_col_dist']:.3f} rad, parts=[{parts_str}]); "
            f"wall bounds: x={list(wall[0])}, y={list(wall[1])}, z={list(wall[2])}"
        )
    return "unknown"


def _progress_bar(i: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + "?" * width + "]"
    frac = max(0.0, min(1.0, i / float(total)))
    fill = int(round(frac * width))
    return "[" + "#" * fill + "-" * (width - fill) + "]"


def _print_progress(label: str, i: int, total: int, suffix: str = "", end: str = "") -> None:
    import sys

    bar = _progress_bar(i, total)
    pct = (100.0 * i / total) if total > 0 else 0.0
    msg = f"\r[robot_check] {label} {bar} {i}/{total} ({pct:5.1f}%)"
    if suffix:
        msg += f" {suffix}"
    sys.stderr.write(msg)
    if end:
        sys.stderr.write(end)
    sys.stderr.flush()


def _finalize_collision_pair_stats(top_n: int = 20) -> dict[str, int] | None:
    """Snapshot the Phase-1 endpoint-collision histogram into a JSON-safe
    dict (descending count) and print a sorted stats table to stderr.

    Returns None when no Phase-1 endpoint collisions were recorded (clean
    Phase 1 or Phase 1 never ran), so the caller can pass it straight to
    `CheckResult(collision_pair_stats=...)`.

    Multiple pairs can collide at the same `q_full`, so the sum of counts
    can exceed `n_failing_endpoint_checks` — the `share` column shows hits
    per failing candidate (can exceed 100% in dense multi-pair scenes;
    that's the signal that lots of constraints are stacking up).
    """
    counts: Counter = _collision_checker.get("collision_pair_counts") or Counter()
    total = int(_collision_checker.get("n_failing_endpoint_checks") or 0)
    if not counts:
        return None
    print("[robot_check] === Phase 1 collision pair stats ===")
    print(
        f"[robot_check]   {total} failing endpoint candidates; "
        f"{len(counts)} unique colliding pairs"
    )
    print("[robot_check]   rank   count  share   entity_a / entity_b")
    sum_top = 0
    for rank, ((a, b), n) in enumerate(counts.most_common(top_n), 1):
        share = 100.0 * n / total if total else 0.0
        sum_top += n
        print(f"[robot_check]   {rank:>4d}  {n:>6d}  {share:>5.1f}%  {a}  ↔  {b}")
    if len(counts) > top_n:
        tail = sum(counts.values()) - sum_top
        print(
            f"[robot_check]   ... + {len(counts) - top_n} more pairs "
            f"({tail} hits)"
        )
    return {f"{a}||{b}": int(n) for (a, b), n in counts.most_common()}


def _build_ik_pool(
    agent: Any,
    initial_q: np.ndarray,
    frame: Any,
    wp: Any,
    yaw_offsets_deg: list[float],
    pitch_offsets_deg: list[float],
    roll_offsets_deg: list[float],
    tool_offset_inv_4x4: np.ndarray | None,
    nb_redundant_search: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict, list[bool], list[int]]:
    """Build the (q_arm, quat_wxyz) pool A* expands at one waypoint.

    For each (roll, pitch, yaw) triple in the cartesian product of the
    supplied offsets — roll = local rotation about tool X (deg), pitch =
    local rotation about tool Y (deg), yaw = local rotation about tool Z
    (deg) — perturb the planner-assigned canonical quat, run analytical IK
    at the perturbed pose, and collect every status-ok solution. Pool
    entries are deduped by rounding q_arm to 1e-3 rad. Collision is NOT
    filtered here — the A* edge check + endpoint check do it.

    Also returns:
      • `ik_status` — flat list[bool] of every IK call's convergence status
        across the cartesian product (roll × pitch × yaw × sorted branch ×
        redundant), in deterministic order. Drives the IK-grid renderer.
      • `pool_flat_indices` — for each (deduped) pool entry, the flat
        ik_status index of the IK call that first produced it. Used to
        overlay the A*-selected path on the IK grid.

    Returns (pool, diag, ik_status, pool_flat_indices); diag mirrors
    `_new_diag` for failure reporting when the pool ends up empty.
    """
    canonical_quat = np.asarray(wp.quaternion_wxyz, dtype=np.float64).copy()
    position = np.asarray(wp.position, dtype=np.float64)

    pool: list[tuple[np.ndarray, np.ndarray]] = []
    pool_flat_indices: list[int] = []
    seen: set[tuple[int, ...]] = set()
    branch_totals: dict[str, list[int]] = {}
    ik_status: list[bool] = []

    for roll_deg in roll_offsets_deg:
        roll_deg_f = float(roll_deg)
        for pitch_deg in pitch_offsets_deg:
            pitch_deg_f = float(pitch_deg)
            for yaw_deg in yaw_offsets_deg:
                yaw_deg_f = float(yaw_deg)
                if (
                    abs(roll_deg_f) <= 1e-9
                    and abs(pitch_deg_f) <= 1e-9
                    and abs(yaw_deg_f) <= 1e-9
                ):
                    quat_used = canonical_quat
                else:
                    quat_used = _quat_rpy_perturb(
                        canonical_quat, (roll_deg_f, pitch_deg_f, yaw_deg_f)
                    )
                tf_target = _tool_tf_to_gripper_tf(
                    _build_target_tf(position, quat_used), tool_offset_inv_4x4
                )
                agent.update_kin(initial_q)
                ik_result = agent.get_ik(
                    tf_target, frame, nb_redundant_search=nb_redundant_search
                )
                # Sort branch keys so the flat ik_status index is stable
                # across waypoints regardless of dict iteration order.
                for which_ik in sorted(ik_result.keys(), key=lambda k: str(k)):
                    status_batch, q_batch = ik_result[which_ik]
                    key = (
                        f"r{roll_deg_f:+.0f}_p{pitch_deg_f:+.0f}"
                        f"_y{yaw_deg_f:+.0f}_{which_ik}"
                    )
                    n_total = len(status_batch)
                    n_ok = 0
                    for i in range(n_total):
                        flat_idx = len(ik_status)
                        ik_status.append(bool(status_batch[i]))
                        if not status_batch[i]:
                            continue
                        n_ok += 1
                        q_arm = np.asarray(q_batch[i], dtype=np.float64).copy()
                        qkey = tuple(int(round(v * 1000.0)) for v in q_arm)
                        if qkey in seen:
                            continue
                        seen.add(qkey)
                        pool.append((q_arm, quat_used.copy()))
                        pool_flat_indices.append(flat_idx)
                    branch_totals[key] = [n_total, n_ok, 0]

    return pool, _new_diag(branch_totals), ik_status, pool_flat_indices


def _filter_pool_by_joint_preference(
    pool: list[tuple[np.ndarray, np.ndarray]],
    pool_flat_indices: list[int],
    filters: dict[int, tuple[float | None, float | None]] | None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[int], int]:
    """Drop Phase 1 IK pool entries that violate per-joint bounds.

    Applied after `_build_ik_pool` and before the A* / DP search sees the
    pool. Each (q_arm, quat) entry is kept iff `q_arm[j] ∈ [lo, hi]` for
    every (j, (lo, hi)) in `filters`. Either bound may be None for a
    one-sided cap. `filters=None`/empty is a fast pass-through.

    Returns (kept_pool, kept_flat_indices, n_dropped). The flat_indices
    list is kept paired with the pool so the IK-grid PNG overlay keeps
    pointing at the right rows.
    """
    if not filters:
        return pool, pool_flat_indices, 0
    kept: list[tuple[np.ndarray, np.ndarray]] = []
    kept_idx: list[int] = []
    for (q_arm, quat), flat_idx in zip(pool, pool_flat_indices):
        ok = True
        for j, (lo, hi) in filters.items():
            v = float(q_arm[j])
            if lo is not None and v < lo:
                ok = False
                break
            if hi is not None and v > hi:
                ok = False
                break
        if ok:
            kept.append((q_arm, quat))
            kept_idx.append(flat_idx)
    return kept, kept_idx, len(pool) - len(kept)


def _endpoint_collides(
    agent: Any,
    base_q: np.ndarray,
    arm_indices: list[int],
    q_arm: np.ndarray,
) -> bool:
    """One-shot self-collision check at a single arm config.

    Routes through `_is_self_colliding` (ampl_motion's convex-mesh `colcvx`)
    instead of `agent.check_self_collision()` so the planner and the
    optimizer subpackage's `n_collisions_after` audit return the same
    verdict on the same `q_full`. On a hit, enumerates the full colliding
    pair list (via `_self_collision_pairs`) and increments
    `_collision_checker["collision_pair_counts"]` so the end-of-run stats
    table can name the dominant culprits. Cost is only paid on collisions
    — collision-free candidates skip the second pass.
    """
    del agent  # unused — collision now goes through _is_self_colliding
    candidate = base_q.copy()
    candidate[arm_indices] = q_arm
    if not _is_self_colliding(candidate):
        return False
    counts: Counter = _collision_checker["collision_pair_counts"]
    for ent_a, ent_b, _d in _self_collision_pairs(candidate):
        counts[tuple(sorted((ent_a, ent_b)))] += 1
    _collision_checker["n_failing_endpoint_checks"] += 1
    return True


def _build_ik_pool_collision_checked(
    endpoint_cache: dict[tuple[int, int], bool],
    pool_flat_indices_per_wp: list[list[int]],
    pool_statuses: list[list[bool]],
) -> list[list[int]]:
    """Convert a sparse Phase 1 endpoint-collision cache to per-waypoint
    flat-shaped lists aligned with `pool_statuses`.

    Cell values:
      0 = untested (Phase 1's search never visited this dedup'd pool entry)
      1 = tested + collision-free
      2 = tested + self-collided
    """
    out: list[list[int]] = [[0] * len(statuses) for statuses in pool_statuses]
    for (L_idx, j_idx), collided in endpoint_cache.items():
        if not (0 <= L_idx < len(pool_flat_indices_per_wp)):
            continue
        flat_list = pool_flat_indices_per_wp[L_idx]
        if not (0 <= j_idx < len(flat_list)):
            continue
        flat_idx = flat_list[j_idx]
        if not (0 <= flat_idx < len(out[L_idx])):
            continue
        out[L_idx][flat_idx] = 2 if collided else 1
    return out


def _astar_phase1(
    agent: Any,
    cfg: "RobotCheckConfig",
    waypoints: list,
    initial_q: np.ndarray,
    arm_indices: list[int],
    frame: Any,
    tool_offset_inv_4x4: np.ndarray | None,
    yaw_offsets_deg: list[float],
    pitch_offsets_deg: list[float],
    roll_offsets_deg: list[float],
) -> tuple[
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[dict],
    list[list[tuple[np.ndarray, np.ndarray]]],
    list[list[bool]],
    list[int] | None,
    list[list[int]],
]:
    """Weighted A* over per-waypoint IK pools.

    Returns (selected_q_arms, selected_quats, failures, pools, pool_statuses,
    selected_flat_indices, ik_pool_collision_checked). On failure the first
    two and the sixth are None and `failures` contains a single dict with
    the deepest layer reached and an `ik` cause string. `pools` is always
    returned (may be partial if pool-building itself bailed out).
    `pool_statuses` is the per-waypoint flat list[bool] of every IK call's
    convergence status across the cartesian product — drives the IK-grid
    PNG renderer. `selected_flat_indices` is one int per waypoint: the flat
    ik_status row of the candidate A* picked, used to overlay the A* path
    on the grid. `ik_pool_collision_checked` is a flat-shaped 3-state list
    (0=untested, 1=clean, 2=collided) collected lazily from whatever
    `_endpoint_collides` calls the search actually made — A* only checks
    pool[0] (seed), so this is sparser for the astar backend than for
    kshortest_dp.
    """
    import heapq

    failures: list[dict] = []
    N = len(waypoints)

    # 1. Pre-compute IK pools for every waypoint.
    n_yaw = len(yaw_offsets_deg)
    n_pitch = len(pitch_offsets_deg)
    n_roll = len(roll_offsets_deg)
    print(
        f"[robot_check] Phase 1a: building IK pools "
        f"({N} waypoints x {n_roll} roll x {n_pitch} pitch x {n_yaw} yaw = "
        f"{n_roll * n_pitch * n_yaw} orientation candidates)"
    )
    pools: list[list[tuple[np.ndarray, np.ndarray]]] = []
    pool_statuses: list[list[bool]] = []
    pool_flat_indices_per_wp: list[list[int]] = []
    pool_sum = 0
    total_filtered = 0
    for i, wp in enumerate(waypoints):
        pool, diag, ik_status, pool_flat_indices = _build_ik_pool(
            agent, initial_q, frame, wp,
            yaw_offsets_deg, pitch_offsets_deg, roll_offsets_deg,
            tool_offset_inv_4x4, cfg.nb_redundant_search,
        )
        pool, pool_flat_indices, n_filtered = _filter_pool_by_joint_preference(
            pool, pool_flat_indices, cfg.joint_preference_filters,
        )
        total_filtered += n_filtered
        pool_statuses.append(ik_status)
        pool_flat_indices_per_wp.append(pool_flat_indices)
        if not pool:
            cause = (
                f"empty IK pool at waypoint {i} across "
                f"{n_roll} roll x {n_pitch} pitch x {n_yaw} yaw offsets"
            )
            if n_filtered > 0:
                cause += (
                    f" (joint preference filter dropped "
                    f"{n_filtered} candidate(s))"
                )
            cause += ": " + _diag_cause(diag, agent)
            failures.append({
                "user_wp_index": i,
                "user_wp_name": getattr(wp, "name", None),
                "sub_index": 0,
                "kind": "ik",
                "cause": cause,
                "branches": diag["branches"],
                "target_position": [float(x) for x in wp.position],
                "target_quat_wxyz": [float(x) for x in wp.quaternion_wxyz],
            })
        pools.append(pool)
        pool_sum += len(pool)
        _print_progress(
            "Phase 1a: IK pools  ", i + 1, N,
            suffix=f"avg={pool_sum / (i + 1):.0f} candidates",
        )
    _print_progress(
        "Phase 1a: IK pools  ", N, N,
        suffix=f"avg={pool_sum / max(N, 1):.0f} candidates", end="\n",
    )
    if cfg.joint_preference_filters and total_filtered > 0:
        print(
            f"[robot_check] joint preference filter dropped "
            f"{total_filtered} IK candidate(s) across {N} waypoint(s)"
        )

    # Phase 1a sweep is complete for every sparse waypoint (no per-waypoint
    # early-quit), so the IK landscape is fully populated. Bail before the
    # A* search runs when any pool came up empty — search exhaustion would
    # surface the same infeasibility but with a less informative message
    # than the per-waypoint `kind="ik"` entries already in `failures`.
    # Lazy collision-verdict cache: (L_idx, j_idx) -> collided_bool. Only
    # nodes the search actually visits land here; the snapshot at the
    # function's exit feeds the per-waypoint collision-free count rendered
    # in *_collision_free.png.
    endpoint_cache: dict[tuple[int, int], bool] = {}

    n_empty = sum(1 for p in pools if not p)
    if n_empty > 0:
        print(
            f"[robot_check] Phase 1a complete: {n_empty}/{N} waypoints "
            f"have empty IK pools — bailing before A* search"
        )
        return (
            None, None, failures, pools, pool_statuses, None,
            _build_ik_pool_collision_checked(
                endpoint_cache, pool_flat_indices_per_wp, pool_statuses,
            ),
        )

    W = float(cfg.heuristic_weight)
    BASE = float(cfg.heuristic_base_cost)
    JUMP = float(cfg.jump_threshold)

    # 2. Seed open list with every collision-free entry of pools[0]. Under
    #    `keep_init`, treat `initial_q[arm_indices]` as a synthetic layer -1:
    #    L_inf-prune layer-0 candidates outside `jump_threshold` of the
    #    initial arm pose and seed the kept ones with cost = weighted-L2
    #    distance from the anchor. Empty post-prune pool is a precondition
    #    violation and raises (vs the legacy soft-failure path below).
    open_list: list[tuple] = []
    parents: dict[tuple[int, int], tuple[int, int] | None] = {}
    closed: set[tuple[int, int]] = set()
    deepest_reached = -1
    counter = 0  # heap tiebreaker

    q_init_arm = initial_q[arm_indices]
    for ik_idx, (q_arm, _quat) in enumerate(pools[0]):
        collided = _endpoint_collides(agent, initial_q, arm_indices, q_arm)
        endpoint_cache[(0, ik_idx)] = collided
        if collided:
            continue
        if cfg.keep_init:
            if float(np.max(np.abs(q_arm - q_init_arm))) > JUMP:
                continue
            g0 = _weighted_arm_dist(q_init_arm, q_arm)
        else:
            g0 = 0.0
        h = W * (N - 1) * BASE
        heapq.heappush(open_list, (g0 + h, g0, 0, ik_idx, counter, None))
        counter += 1
        parents[(0, ik_idx)] = None

    if cfg.keep_init and not open_list:
        raise RuntimeError(
            f"keep_init: no IK candidate at waypoint 0 is within "
            f"jump_threshold={JUMP:.3f} rad (L_inf) of "
            f"initial_q[arm_indices] (pool size = {len(pools[0])}). The "
            f"deployment precondition 'initial pose is close to first "
            f"waypoint' was violated; either move the robot closer to wp_0 "
            f"or widen robot.search.jump_threshold."
        )

    if not open_list:
        failures.append({
            "user_wp_index": 0,
            "user_wp_name": getattr(waypoints[0], "name", None),
            "sub_index": 0,
            "kind": "ik",
            "cause": (
                f"every IK candidate at waypoint 0 self-collides "
                f"(pool size = {len(pools[0])})"
            ),
            "target_position": [float(x) for x in waypoints[0].position],
            "target_quat_wxyz": [float(x) for x in waypoints[0].quaternion_wxyz],
        })
        return (
            None, None, failures, pools, pool_statuses, None,
            _build_ik_pool_collision_checked(
                endpoint_cache, pool_flat_indices_per_wp, pool_statuses,
            ),
        )

    # 3. A* main loop.
    print(f"[robot_check] Phase 1b: A* search (W={W}, jump={JUMP} rad)")
    pop_count = 0
    while open_list:
        f_score, g_score, layer, ik_idx, _ctr, parent = heapq.heappop(open_list)
        pop_count += 1
        state = (layer, ik_idx)
        if state in closed:
            continue

        closed.add(state)
        if layer > deepest_reached:
            deepest_reached = layer
            _print_progress(
                "Phase 1b: A* search ", deepest_reached + 1, N,
                suffix=f"popped={pop_count}",
            )

        if layer == N - 1:
            _print_progress(
                "Phase 1b: A* search ", N, N,
                suffix=f"popped={pop_count}",
                end="\n",
            )
            # Reconstruct: walk parent chain.
            chain: list[tuple[int, int]] = []
            cur_state: tuple[int, int] | None = state
            while cur_state is not None:
                chain.append(cur_state)
                cur_state = parents[cur_state]
            chain.reverse()
            selected_q_arms = [pools[L][k][0].copy() for L, k in chain]
            selected_quats = [pools[L][k][1].copy() for L, k in chain]
            selected_flat_indices = [
                pool_flat_indices_per_wp[L][k] for L, k in chain
            ]
            return (
                selected_q_arms, selected_quats, [], pools, pool_statuses,
                selected_flat_indices,
                _build_ik_pool_collision_checked(
                    endpoint_cache, pool_flat_indices_per_wp, pool_statuses,
                ),
            )

        # Expand to next layer.
        q_cur = pools[layer][ik_idx][0]
        next_layer = layer + 1
        layers_remaining = (N - 1) - next_layer
        for next_idx, (q_next, _quat_next) in enumerate(pools[next_layer]):
            if (next_layer, next_idx) in closed:
                continue
            if float(np.max(np.abs(q_next - q_cur))) > JUMP:
                continue
            edge_cost = _weighted_arm_dist(q_cur, q_next)
            new_g = g_score + edge_cost
            new_f = new_g + W * layers_remaining * BASE
            heapq.heappush(
                open_list, (new_f, new_g, next_layer, next_idx, counter, state)
            )
            counter += 1
            parents.setdefault((next_layer, next_idx), state)

    # Open list drained without reaching the goal.
    _print_progress(
        "Phase 1b: A* search ", max(deepest_reached + 1, 0), N,
        suffix=f"popped={pop_count} EXHAUSTED",
        end="\n",
    )
    deepest_wp = waypoints[max(deepest_reached, 0)]
    failures.append({
        "user_wp_index": max(deepest_reached + 1, 0),
        "user_wp_name": getattr(
            waypoints[min(deepest_reached + 1, N - 1)], "name", None
        ),
        "sub_index": 0,
        "kind": "ik",
        "cause": (
            f"A* exhausted before reaching final waypoint; deepest layer "
            f"reached = {deepest_reached} of {N - 1}. "
            f"Likely causes: IK pool too thin (raise nb_redundant_search "
            f"or widen yaw_search_offsets_deg / pitch_search_offsets_deg / "
            f"roll_search_offsets_deg), "
            f"jump_threshold too "
            f"tight (current {JUMP:.2f} rad), or genuine self-collision "
            f"chokepoint between waypoints {deepest_reached} and "
            f"{deepest_reached + 1}"
        ),
        "deepest_layer_reached": int(deepest_reached),
        "target_position": [float(x) for x in deepest_wp.position],
        "target_quat_wxyz": [float(x) for x in deepest_wp.quaternion_wxyz],
    })
    return (
        None, None, failures, pools, pool_statuses, None,
        _build_ik_pool_collision_checked(
            endpoint_cache, pool_flat_indices_per_wp, pool_statuses,
        ),
    )


def _kshortest_dp_phase1(
    agent: Any,
    cfg: "RobotCheckConfig",
    waypoints: list,
    initial_q: np.ndarray,
    arm_indices: list[int],
    frame: Any,
    tool_offset_inv_4x4: np.ndarray | None,
    yaw_offsets_deg: list[float],
    pitch_offsets_deg: list[float],
    roll_offsets_deg: list[float],
) -> tuple[
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    list[dict],
    list[list[tuple[np.ndarray, np.ndarray]]],
    list[list[bool]],
    list[int] | None,
    list[list[int]],
]:
    """Iterative-mask DP over per-waypoint IK pools.

    Pre-caches the per-layer edge-cost matrix once (a (P_next, P_prev) numpy
    array of weighted-L2 joint distances; entries that fail the L_inf
    `jump_threshold` are set to inf). Then iterates:
      1. Forward single-best DP over the cached edges with a persistent
         per-(L, j) `mask` boolean. Masked nodes are forced to cost=inf
         so they cannot be selected.
      2. Reconstruct the single shortest path by walking parent pointers
         from `argmin cost[N-1]`.
      3. Run `_endpoint_collides` on each node of the reconstructed path
         (swept collisions between adjacent IK picks are caught later by
         Phase 2's dense slerp endpoint checks, not here).
      4. On endpoint failure at (L, j) → `mask[L][j] = True`.
      5. Stop when no failures remain, or when the DP becomes infeasible
         (mask removed every parent at some layer), or when
         `cfg.max_iterations` is reached.

    Returns the same 7-tuple as `_astar_phase1`. The 7th element is the
    flat-shaped `ik_pool_collision_checked` (0/1/2 per cell) built from
    the iterative-mask DP's `endpoint_cache`; kshortest_dp populates this
    richly across iterations (every node on every reconstructed path),
    so this backend's grid is far denser than A*'s.

    Knobs consumed:
      cfg.jump_threshold              hard edge filter on L_inf |Δq_arm|
      cfg.max_iterations              iteration cap for the mask loop
      cfg.nb_redundant_search         pool builder (unchanged from A* path)

    `cfg.heuristic_weight` / `cfg.heuristic_base_cost` are ignored.
    """
    failures: list[dict] = []
    N = len(waypoints)

    # 1. Pre-compute IK pools for every waypoint (mirrors _astar_phase1).
    n_yaw = len(yaw_offsets_deg)
    n_pitch = len(pitch_offsets_deg)
    n_roll = len(roll_offsets_deg)
    print(
        f"[robot_check] Phase 1a: building IK pools "
        f"({N} waypoints x {n_roll} roll x {n_pitch} pitch x {n_yaw} yaw = "
        f"{n_roll * n_pitch * n_yaw} orientation candidates)"
    )
    pools: list[list[tuple[np.ndarray, np.ndarray]]] = []
    pool_statuses: list[list[bool]] = []
    pool_flat_indices_per_wp: list[list[int]] = []
    pool_sum = 0
    total_filtered = 0
    for i, wp in enumerate(waypoints):
        pool, diag, ik_status, pool_flat_indices = _build_ik_pool(
            agent, initial_q, frame, wp,
            yaw_offsets_deg, pitch_offsets_deg, roll_offsets_deg,
            tool_offset_inv_4x4, cfg.nb_redundant_search,
        )
        pool, pool_flat_indices, n_filtered = _filter_pool_by_joint_preference(
            pool, pool_flat_indices, cfg.joint_preference_filters,
        )
        total_filtered += n_filtered
        pool_statuses.append(ik_status)
        pool_flat_indices_per_wp.append(pool_flat_indices)
        if not pool:
            cause = (
                f"empty IK pool at waypoint {i} across "
                f"{n_roll} roll x {n_pitch} pitch x {n_yaw} yaw offsets"
            )
            if n_filtered > 0:
                cause += (
                    f" (joint preference filter dropped "
                    f"{n_filtered} candidate(s))"
                )
            cause += ": " + _diag_cause(diag, agent)
            failures.append({
                "user_wp_index": i,
                "user_wp_name": getattr(wp, "name", None),
                "sub_index": 0,
                "kind": "ik",
                "cause": cause,
                "branches": diag["branches"],
                "target_position": [float(x) for x in wp.position],
                "target_quat_wxyz": [float(x) for x in wp.quaternion_wxyz],
            })
        pools.append(pool)
        pool_sum += len(pool)
        _print_progress(
            "Phase 1a: IK pools  ", i + 1, N,
            suffix=f"avg={pool_sum / (i + 1):.0f} candidates",
        )
    _print_progress(
        "Phase 1a: IK pools  ", N, N,
        suffix=f"avg={pool_sum / max(N, 1):.0f} candidates", end="\n",
    )
    if cfg.joint_preference_filters and total_filtered > 0:
        print(
            f"[robot_check] joint preference filter dropped "
            f"{total_filtered} IK candidate(s) across {N} waypoint(s)"
        )

    # Phase 1a sweep is complete for every sparse waypoint (no per-waypoint
    # early-quit), so the IK landscape is fully populated. Bail before the
    # iterative-mask DP runs when any pool came up empty — the edge-cost
    # precompute below would crash on np.stack of an empty pool, and the
    # per-waypoint `kind="ik"` entries already in `failures` carry the
    # diagnosis.
    n_empty = sum(1 for p in pools if not p)
    if n_empty > 0:
        print(
            f"[robot_check] Phase 1a complete: {n_empty}/{N} waypoints "
            f"have empty IK pools — bailing before iterative-mask DP"
        )
        return (
            None, None, failures, pools, pool_statuses, None,
            # No collision checks ran yet; pass an empty cache so the
            # caller still gets a same-shape (all-untested) matrix.
            _build_ik_pool_collision_checked(
                {}, pool_flat_indices_per_wp, pool_statuses,
            ),
        )

    MAX_ITER = int(cfg.max_iterations)
    JUMP = float(cfg.jump_threshold)
    ACCEL_W = float(cfg.accel_weight)

    # 2. Pre-cache per-layer edge-cost matrices once. edge_costs[L-1] is
    #    the (P[L], P[L-1]) matrix of weighted-L2 distances from each
    #    pool entry at layer L-1 to each at layer L; entries that fail
    #    the L_inf `jump_threshold` are set to inf. Pool geometry doesn't
    #    change across iterations, so this is computed once.
    print(
        f"[robot_check] Phase 1b: iterative-mask DP "
        f"(max_iter={MAX_ITER}, jump={JUMP:.2f} rad, "
        f"accel_w={ACCEL_W:.2f})"
    )
    weights_arr = np.asarray(JOINT_DIST_WEIGHTS, dtype=np.float64)
    pools_arr: list[np.ndarray] = [
        np.stack([entry[0] for entry in pools[L]], axis=0).astype(
            np.float64, copy=False
        )
        for L in range(N)
    ]
    edge_cost_mats: list[np.ndarray] = []  # edge_cost_mats[L-1]: (P[L], P[L-1])
    for L in range(1, N):
        prev_qs = pools_arr[L - 1]
        next_qs = pools_arr[L]
        diffs = next_qs[:, None, :] - prev_qs[None, :, :]
        abs_diffs = np.abs(diffs)
        e = np.sqrt((abs_diffs * abs_diffs) @ weights_arr)
        e = np.where(abs_diffs.max(axis=-1) <= JUMP, e, np.inf)
        edge_cost_mats.append(e)

    # 3. Iterative-mask loop. mask[L][j] = True means (L, j) is excluded
    #    from the DP (the IK candidate previously collided as an endpoint).
    pool_sizes = [arr.shape[0] for arr in pools_arr]
    mask: list[np.ndarray] = [
        np.zeros(P_L, dtype=bool) for P_L in pool_sizes
    ]

    # Layer-0 base cost. Legacy: every candidate enters at 0. `keep_init`:
    # L_inf-prune layer-0 entries outside JUMP of `initial_q[arm_indices]`
    # (set their base cost to inf) and seed the rest with weighted-L2
    # distance from the anchor. Computed once — pool geometry doesn't
    # change across iterations. Empty post-prune layer-0 pool is a
    # precondition violation and raises (matches `_astar_phase1`).
    if cfg.keep_init:
        q_init_arm = initial_q[arm_indices]
        diff0 = np.abs(pools_arr[0] - q_init_arm[None, :])  # (P[0], 7)
        reachable0 = diff0.max(axis=-1) <= JUMP
        base_cost0 = np.where(
            reachable0,
            np.sqrt((diff0 * diff0) @ weights_arr),
            np.inf,
        )
        if not np.isfinite(base_cost0).any():
            raise RuntimeError(
                f"keep_init: no IK candidate at waypoint 0 is within "
                f"jump_threshold={JUMP:.3f} rad (L_inf) of "
                f"initial_q[arm_indices] (pool size = {pool_sizes[0]}). "
                f"The deployment precondition 'initial pose is close to "
                f"first waypoint' was violated; either move the robot "
                f"closer to wp_0 or widen robot.search.jump_threshold."
            )
    else:
        base_cost0 = np.zeros(pool_sizes[0], dtype=np.float64)
    total_endpoint_masks = 0
    last_endpoint_failures = 0

    # Cache endpoint collision verdicts across iterations. Pool geometry is
    # fixed, so `(L, j)` keys the check exactly — without the cache, a path
    # that reuses nodes from a previous iteration would re-run the same
    # collision queries. Most nodes pass and stay unmasked, so reuse is the
    # common case. True = collides, False = clean.
    endpoint_cache: dict[tuple[int, int], bool] = {}
    total_new_endpoint_checks = 0
    total_dp_time = 0.0
    total_check_time = 0.0

    for it in range(MAX_ITER):
        # 3a. Forward single-best DP using the cached edge_cost_mats and
        #     the current mask. cost / parent are per-layer numpy arrays.
        t_dp_start = time.perf_counter()
        cost: list[np.ndarray] = [
            np.full(P_L, np.inf, dtype=np.float64) for P_L in pool_sizes
        ]
        parent: list[np.ndarray] = [
            np.full(P_L, -1, dtype=np.int32) for P_L in pool_sizes
        ]
        cost[0][:] = base_cost0
        cost[0][mask[0]] = np.inf
        for L in range(1, N):
            # cands[j, i] = cost[L-1][i] + edge_cost_mats[L-1][j, i]
            cands = cost[L - 1][None, :] + edge_cost_mats[L - 1]
            if ACCEL_W > 0.0 and L >= 2:
                # Lazy grandparent: for each parent candidate i at L-1,
                # treat parent[L-1][i] (already computed earlier in this
                # forward pass) as i's grandparent. The DP stays at state
                # (L, j) and per-layer work stays O(P²); the price is
                # global optimality under the new cost. -1 sentinel means
                # cost[L-1][i] was inf (unreachable), so the row already
                # falls out of the argmin — zero its accel contribution
                # to avoid an OOB gather and a meaningless cost spike.
                gp_idx = parent[L - 1]                       # (P[L-1],)
                safe_gp = np.where(gp_idx >= 0, gp_idx, 0)
                q_gp = pools_arr[L - 2][safe_gp]             # (P[L-1], 7)
                q_par = pools_arr[L - 1]                     # (P[L-1], 7)
                q_cur = pools_arr[L]                         # (P[L], 7)
                accel = (
                    q_cur[:, None, :]
                    - 2.0 * q_par[None, :, :]
                    + q_gp[None, :, :]
                )
                accel_cost = np.sqrt((accel * accel) @ weights_arr)
                no_gp = gp_idx < 0
                if no_gp.any():
                    accel_cost[:, no_gp] = 0.0
                cands = cands + ACCEL_W * accel_cost
            parent[L] = np.argmin(cands, axis=1).astype(np.int32)
            cost[L] = np.take_along_axis(
                cands, parent[L][:, None], axis=1
            ).reshape(-1)
            cost[L][mask[L]] = np.inf

        # 3b. DP infeasible? Find the deepest reachable layer for diagnostics.
        if not np.isfinite(cost[N - 1]).any():
            t_dp = time.perf_counter() - t_dp_start
            total_dp_time += t_dp
            deepest = -1
            for L in range(N - 1, -1, -1):
                if np.isfinite(cost[L]).any():
                    deepest = L
                    break
            _print_progress(
                "Phase 1b: iter-mask DP ", it + 1, MAX_ITER,
                suffix=(
                    f"INFEASIBLE: deepest reachable={deepest}/"
                    f"{N - 1} masked={total_endpoint_masks} "
                    f"t_dp={t_dp:.3f}s total(dp/check)="
                    f"{total_dp_time:.3f}/{total_check_time:.3f}s"
                ),
                end="\n",
            )
            deepest_wp = waypoints[max(deepest, 0)]
            failures.append({
                "user_wp_index": int(min(deepest + 1, N - 1)),
                "user_wp_name": getattr(
                    waypoints[min(deepest + 1, N - 1)], "name", None
                ),
                "sub_index": 0,
                "kind": "ik",
                "cause": (
                    f"iterative-mask DP infeasible at iteration {it + 1}; "
                    f"deepest layer reached = {deepest} of {N - 1}. "
                    f"Total endpoint nodes masked: {total_endpoint_masks}. "
                    f"Likely causes: IK pool too thin (raise "
                    f"nb_redundant_search or widen "
                    f"yaw_search_offsets_deg / pitch_search_offsets_deg / "
                    f"roll_search_offsets_deg), "
                    f"jump_threshold too tight (current {JUMP:.2f} rad), "
                    f"or genuine collision chokepoint near layer "
                    f"{deepest + 1}"
                ),
                "deepest_layer_reached": int(deepest),
                "iterations_run": int(it + 1),
                "target_position": [float(x) for x in deepest_wp.position],
                "target_quat_wxyz": [float(x) for x in deepest_wp.quaternion_wxyz],
            })
            return (
                None, None, failures, pools, pool_statuses, None,
                _build_ik_pool_collision_checked(
                    endpoint_cache, pool_flat_indices_per_wp, pool_statuses,
                ),
            )

        # 3c. Reconstruct shortest path by walking parent pointers.
        end_j = int(np.argmin(cost[N - 1]))
        path_cost = float(cost[N - 1][end_j])
        chain: list[tuple[int, int]] = [(N - 1, end_j)]
        cur_L, cur_j = N - 1, end_j
        while cur_L > 0:
            par = int(parent[cur_L][cur_j])
            chain.append((cur_L - 1, par))
            cur_L, cur_j = cur_L - 1, par
        chain.reverse()

        t_dp = time.perf_counter() - t_dp_start
        total_dp_time += t_dp

        # 3d. Endpoint collision check the path; collect failures, mask once.
        #     Consult the per-iteration cache first — pool geometry is fixed
        #     so the same `(L, j)` gives the same verdict every iteration.
        t_check_start = time.perf_counter()
        new_endpoint_failures: list[tuple[int, int]] = []
        new_endpoint_checks = 0
        for L_idx, j_idx in chain:
            ep_key = (L_idx, j_idx)
            ep_hit = endpoint_cache.get(ep_key)
            if ep_hit is None:
                ep_hit = _endpoint_collides(
                    agent, initial_q, arm_indices, pools[L_idx][j_idx][0]
                )
                endpoint_cache[ep_key] = ep_hit
                new_endpoint_checks += 1
            if ep_hit:
                new_endpoint_failures.append((L_idx, j_idx))

        t_check = time.perf_counter() - t_check_start
        total_check_time += t_check
        total_new_endpoint_checks += new_endpoint_checks
        last_endpoint_failures = len(new_endpoint_failures)

        if not new_endpoint_failures:
            # Clean path. Build outputs in the same shape as _astar_phase1.
            _print_progress(
                "Phase 1b: iter-mask DP ", it + 1, MAX_ITER,
                suffix=(
                    f"OK iter={it + 1} cost={path_cost:.3f} "
                    f"masked={total_endpoint_masks} "
                    f"new_checks(ep)={new_endpoint_checks} "
                    f"total_new(ep)={total_new_endpoint_checks} "
                    f"t(dp/check)={t_dp:.3f}/{t_check:.3f}s "
                    f"total(dp/check)={total_dp_time:.3f}/"
                    f"{total_check_time:.3f}s"
                ),
                end="\n",
            )
            selected_q_arms = [
                pools[L_idx][j_idx][0].copy() for L_idx, j_idx in chain
            ]
            selected_quats = [
                pools[L_idx][j_idx][1].copy() for L_idx, j_idx in chain
            ]
            selected_flat_indices = [
                pool_flat_indices_per_wp[L_idx][j_idx]
                for L_idx, j_idx in chain
            ]
            return (
                selected_q_arms, selected_quats, [], pools, pool_statuses,
                selected_flat_indices,
                _build_ik_pool_collision_checked(
                    endpoint_cache, pool_flat_indices_per_wp, pool_statuses,
                ),
            )

        # 3e. Mask new endpoint failures.
        for L_idx, j_idx in new_endpoint_failures:
            if not mask[L_idx][j_idx]:
                mask[L_idx][j_idx] = True
                total_endpoint_masks += 1

        _print_progress(
            "Phase 1b: iter-mask DP ", it + 1, MAX_ITER,
            suffix=(
                f"cost={path_cost:.3f} bad_ep={last_endpoint_failures} "
                f"masked={total_endpoint_masks} "
                f"new_checks(ep)={new_endpoint_checks} "
                f"total_new(ep)={total_new_endpoint_checks} "
                f"t(dp/check)={t_dp:.3f}/{t_check:.3f}s "
                f"total(dp/check)={total_dp_time:.3f}/"
                f"{total_check_time:.3f}s"
            ),
        )

    # Iteration cap exceeded.
    _print_progress(
        "Phase 1b: iter-mask DP ", MAX_ITER, MAX_ITER,
        suffix=(
            f"ITER_CAP last bad_ep={last_endpoint_failures} "
            f"masked={total_endpoint_masks} "
            f"total_new(ep)={total_new_endpoint_checks} "
            f"total(dp/check)={total_dp_time:.3f}/"
            f"{total_check_time:.3f}s"
        ),
        end="\n",
    )
    failures.append({
        "user_wp_index": -1,
        "user_wp_name": None,
        "sub_index": 0,
        "kind": "collision",
        "cause": (
            f"iterative-mask DP exhausted max_iterations={MAX_ITER}. "
            f"Last iteration still had {last_endpoint_failures} endpoint "
            f"collisions; total endpoint nodes masked across all "
            f"iterations: {total_endpoint_masks}. "
            f"Suggested remedies: raise max_iterations, raise "
            f"nb_redundant_search, widen "
            f"yaw_search_offsets_deg / pitch_search_offsets_deg / "
            f"roll_search_offsets_deg, "
            f"or fall back to search_method: astar."
        ),
        "iterations_run": int(MAX_ITER),
        "n_endpoint_masked": int(total_endpoint_masks),
    })
    return (
        None, None, failures, pools, pool_statuses, None,
        _build_ik_pool_collision_checked(
            endpoint_cache, pool_flat_indices_per_wp, pool_statuses,
        ),
    )


# ============================================================================
# Phase 3: jerk-region refinement (greedy per-step continuity IK)
# ============================================================================
#
# After Phase 2 produces a densified q_trajectory, walk it to find sub-step
# pairs where |Δq_arm|∞ exceeds `cfg.refine_jump_threshold` (default 0.1 rad).
# Group flagged indices into anchored regions (padded by `region_buffer`).
# For each region, walk interior steps left-to-right; at each step build a
# relaxed pool of perturbed targets:
#   • yaw   ±5° at 1° spacing  (around tool Z)
#   • pitch ±5° at 1° spacing  (around tool Y)
#   • xy    ±1 cm at 0.2 cm spacing in the surface PlaneFrame (u, v)
# Pick the pool entry that minimises weighted L2 |Δq_arm| from the previous
# step's q_arm and is collision-free (sort then early-exit, per IK-selection
# convention). When the pool is empty or every entry collides, keep that
# step's Phase 2 q_arm and continue. Phase 3 mutates q_trajectory in place
# and returns a stats dict for the JSON metadata.


def _per_step_arm_q_array(q_trajectory: list, arm_key: str) -> np.ndarray:
    """Stack per-step arm-q across the trajectory; rows where q_full is None
    (Phase 2 step failures) are filled with NaN so they don't corrupt jump
    detection or anchor selection."""
    rows: list[list[float]] = []
    for qp in q_trajectory:
        v = getattr(qp, arm_key)
        rows.append(list(v) if v is not None else [float("nan")] * 7)
    return np.asarray(rows, dtype=np.float64)


def _detect_jerk_regions(
    q_trajectory: list,
    arm_key: str,
    threshold: float,
    buffer: int,
) -> list[tuple[int, int]]:
    """Return (lo, hi) anchor index pairs around runs of high-Δq sub-steps.

    Detection: a *jerk index* i is one where max|q_arm[i] - q_arm[i-1]| >
    threshold. Consecutive jerk indices are coalesced; each run is padded by
    `buffer` indices on each side and overlapping pads are merged. The lo / hi
    indices are the anchor steps (Phase 2 q_arm preserved); interior =
    indices in (lo, hi) exclusive. NaN rows (Phase 2 failures) cannot serve
    as anchors, so regions where lo or hi has NaN q_arm are dropped.
    """
    arr = _per_step_arm_q_array(q_trajectory, arm_key)
    N = arr.shape[0]
    if N < 2:
        return []
    diff = np.abs(np.diff(arr, axis=0))                       # (N-1, 7)
    linf = np.nanmax(diff, axis=1)                             # (N-1,)
    flagged = np.where(linf > threshold)[0] + 1                # destination indices
    if flagged.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    a = b = int(flagged[0])
    for idx in flagged[1:]:
        idx = int(idx)
        if idx == b + 1:
            b = idx
        else:
            runs.append((a, b))
            a = b = idx
    runs.append((a, b))

    padded: list[tuple[int, int]] = []
    for lo, hi in runs:
        plo = max(0, lo - buffer - 1)  # lo is destination of jump → source is lo-1
        phi = min(N - 1, hi + buffer)
        padded.append((plo, phi))

    merged: list[tuple[int, int]] = []
    for lo, hi in padded:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    valid: list[tuple[int, int]] = []
    for lo, hi in merged:
        if hi - lo < 2:
            continue  # need at least one interior step
        if np.any(np.isnan(arr[lo])) or np.any(np.isnan(arr[hi])):
            continue
        valid.append((lo, hi))
    return valid


def _build_refine_pool(
    agent: Any,
    base_q: np.ndarray,
    frame: Any,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    yaw_offsets_deg: list[float],
    pitch_offsets_deg: list[float],
    xy_offsets_m: list[tuple[float, float]],
    plane_u_axis: np.ndarray,
    plane_v_axis: np.ndarray,
    tool_offset_inv_4x4: np.ndarray | None,
    nb_redundant_search: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    """Build the relaxed (q_arm, pos_used, quat_used) pool for one interior step.

    Iterates the cartesian product of yaw × pitch × xy offsets, perturbs the
    target pose, runs analytical IK, and dedups by q_arm rounded to 1e-3 rad.
    Collision is NOT filtered here — A* edge / endpoint checks do it. Returns
    (pool, n_ik_calls) where n_ik_calls counts perturbed-pose IK invocations.
    """
    pool: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    n_ik_calls = 0

    for pitch_deg in pitch_offsets_deg:
        pitch_f = float(pitch_deg)
        for yaw_deg in yaw_offsets_deg:
            yaw_f = float(yaw_deg)
            if abs(pitch_f) <= 1e-9 and abs(yaw_f) <= 1e-9:
                quat_used = target_quat.copy()
            else:
                quat_used = _quat_rpy_perturb(target_quat, (0.0, pitch_f, yaw_f))
            for du, dv in xy_offsets_m:
                pos_used = (
                    target_pos
                    + float(du) * plane_u_axis
                    + float(dv) * plane_v_axis
                ).astype(np.float64)
                tf_target = _tool_tf_to_gripper_tf(
                    _build_target_tf(pos_used, quat_used), tool_offset_inv_4x4
                )
                agent.update_kin(base_q)
                ik_result = agent.get_ik(
                    tf_target, frame, nb_redundant_search=nb_redundant_search
                )
                n_ik_calls += 1
                for _which_ik, (status_batch, q_batch) in ik_result.items():
                    for i in range(len(status_batch)):
                        if not status_batch[i]:
                            continue
                        q_arm = np.asarray(q_batch[i], dtype=np.float64).copy()
                        qkey = tuple(int(round(float(v_) * 1000.0)) for v_ in q_arm)
                        if qkey in seen:
                            continue
                        seen.add(qkey)
                        pool.append((q_arm, pos_used.copy(), quat_used.copy()))
    return pool, n_ik_calls


def _pick_best_from_refine_pool(
    agent: Any,
    base_q: np.ndarray,
    arm_indices: list[int],
    cur_arm: np.ndarray,
    pool: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> int | None:
    """Sort pool by weighted dist to cur_arm, return first collision-free idx.

    Continuity-preferred selector matching the Phase 2 IK pattern: weighted
    L2 distance with `JOINT_DIST_WEIGHTS`, sort ascending, early-exit on
    first collision-free candidate. Returns None when every candidate
    self-collides (or the pool is empty).
    """
    if not pool:
        return None
    candidates = sorted(
        (
            (_weighted_arm_dist(q_arm, cur_arm), idx)
            for idx, (q_arm, _pos, _quat) in enumerate(pool)
        ),
        key=lambda x: x[0],
    )
    for _dist, idx in candidates:
        q_arm = pool[idx][0]
        if not _endpoint_collides(agent, base_q, arm_indices, q_arm):
            return idx
    return None


def _default_xy_offsets_m() -> list[tuple[float, float]]:
    """Default ±1 cm grid at 0.2 cm spacing → 11×11 = 121 (du, dv) pairs."""
    cm_steps = np.arange(-0.010, 0.0101, 0.002)
    return [(float(du), float(dv)) for du in cm_steps for dv in cm_steps]


def _phase3_refine(
    *,
    agent: Any,
    cfg: "RobotCheckConfig",
    q_trajectory: list,
    arm_key: str,
    arm_indices: list[int],
    initial_q: np.ndarray,
    frame: Any,
    tool_offset: np.ndarray | None,
    tool_offset_inv: np.ndarray | None,
    plane_u_axis: np.ndarray,
    plane_v_axis: np.ndarray,
) -> dict:
    """Run Phase 3 jerk-region detection + greedy continuity refinement.

    Mutates `q_trajectory` in place, replacing each interior step whose
    relaxed pool offers a collision-free q_arm. When a step's pool is empty
    or every entry self-collides, the Phase 2 entry is kept (counted into
    `n_steps_kept`). Returns a stats dict for JSON serialization.
    """
    yaw_offs = (
        [float(d) for d in range(-5, 6)]
        if cfg.refine_yaw_offsets_deg is None
        else [float(o) for o in cfg.refine_yaw_offsets_deg]
    )
    pitch_offs = (
        [float(d) for d in range(-5, 6)]
        if cfg.refine_pitch_offsets_deg is None
        else [float(o) for o in cfg.refine_pitch_offsets_deg]
    )
    xy_offs = (
        _default_xy_offsets_m()
        if cfg.refine_xy_offsets_m is None
        else [(float(a), float(b)) for a, b in cfg.refine_xy_offsets_m]
    )

    arr_before = _per_step_arm_q_array(q_trajectory, arm_key)
    if arr_before.shape[0] < 2:
        return {
            "n_jerk_regions": 0, "n_steps_refined": 0, "n_steps_kept": 0,
            "max_jump_inf_before": 0.0, "max_jump_inf_after": 0.0,
            "total_ik_calls": 0,
        }
    diff_before = np.abs(np.diff(arr_before, axis=0))
    max_jump_before = (
        float(np.nanmax(diff_before)) if diff_before.size and np.isfinite(diff_before).any() else 0.0
    )

    regions = _detect_jerk_regions(
        q_trajectory, arm_key, cfg.refine_jump_threshold, cfg.refine_region_buffer
    )
    if not regions:
        print(
            f"[robot_check] Phase 3: no jerk regions "
            f"(threshold={cfg.refine_jump_threshold:.3f} rad, "
            f"max|Δq|∞={max_jump_before:.3f} rad)"
        )
        return {
            "n_jerk_regions": 0, "n_steps_refined": 0, "n_steps_kept": 0,
            "max_jump_inf_before": max_jump_before,
            "max_jump_inf_after": max_jump_before,
            "total_ik_calls": 0,
        }

    print(
        f"[robot_check] Phase 3: refining {len(regions)} jerk region(s) "
        f"(yaw={len(yaw_offs)} x pitch={len(pitch_offs)} x xy={len(xy_offs)} = "
        f"{len(yaw_offs) * len(pitch_offs) * len(xy_offs)} perturbed poses/step, "
        f"redundant={cfg.refine_nb_redundant_search})"
    )

    n_steps_refined = 0
    n_steps_kept = 0
    total_ik_calls = 0

    for r_idx, (lo, hi) in enumerate(regions):
        interior = list(range(lo + 1, hi))
        if not interior:
            continue
        cur_arm = arr_before[lo].copy()  # start anchor's q_arm
        label = f"Phase 3 r{r_idx + 1:02d}/{len(regions):02d}"

        for k, i in enumerate(interior):
            qp = q_trajectory[i]
            tgt_pos = np.asarray(qp.target_position, dtype=np.float64)
            tgt_quat = np.asarray(qp.target_quat_wxyz, dtype=np.float64)
            pool, n_calls = _build_refine_pool(
                agent, initial_q, frame, tgt_pos, tgt_quat,
                yaw_offs, pitch_offs, xy_offs,
                plane_u_axis, plane_v_axis, tool_offset_inv,
                cfg.refine_nb_redundant_search,
            )
            total_ik_calls += n_calls

            best_idx = _pick_best_from_refine_pool(
                agent, initial_q, arm_indices, cur_arm, pool
            )
            if best_idx is None:
                # Pool empty or every entry collides — keep Phase 2 entry and
                # advance cur_arm to its q_arm so the next step still tracks
                # continuity. If Phase 2 entry has no q_arm (was a step
                # failure), leave cur_arm unchanged.
                n_steps_kept += 1
                phase2_q = arr_before[i]
                if not np.any(np.isnan(phase2_q)):
                    cur_arm = phase2_q
                _print_progress(
                    label, k + 1, len(interior),
                    suffix=(
                        f"interior={lo}..{hi} pool={len(pool)} "
                        f"ik={total_ik_calls} (kept phase 2)"
                    ),
                )
                continue

            q_arm_new, pos_used, quat_used = pool[best_idx]
            new_q_full = initial_q.copy()
            new_q_full[arm_indices] = q_arm_new
            agent.update_kin(new_q_full)
            fk_matrix = agent.get_fk(frame).matrix
            achieved = _gripper_fk_to_tool_position(fk_matrix, tool_offset)
            achieved_quat = _gripper_fk_to_tool_quat_wxyz(fk_matrix, tool_offset)
            dev = float(np.linalg.norm(achieved - pos_used))
            old = q_trajectory[i]
            q_trajectory[i] = _make_qpoint(
                t_global=old.t_global,
                user_wp_index=old.user_wp_index,
                sub_index=old.sub_index,
                target_position=pos_used,
                target_quat_wxyz=quat_used,
                achieved_position=achieved,
                achieved_quat_wxyz=achieved_quat,
                cartesian_deviation=dev,
                q_full=new_q_full,
                ok=True,
                failure_cause=None,
            )
            cur_arm = q_arm_new
            n_steps_refined += 1
            _print_progress(
                label, k + 1, len(interior),
                suffix=(
                    f"interior={lo}..{hi} pool={len(pool)} "
                    f"ik={total_ik_calls}"
                ),
            )
        _print_progress(
            label, len(interior), len(interior),
            suffix=f"interior={lo}..{hi} done", end="\n",
        )

    arr_after = _per_step_arm_q_array(q_trajectory, arm_key)
    diff_after = np.abs(np.diff(arr_after, axis=0))
    max_jump_after = (
        float(np.nanmax(diff_after)) if diff_after.size and np.isfinite(diff_after).any() else 0.0
    )

    print(
        f"[robot_check] Phase 3: refined {n_steps_refined} step(s), kept "
        f"{n_steps_kept} step(s) across {len(regions)} region(s); "
        f"max|Δq|∞ {max_jump_before:.3f} → {max_jump_after:.3f} rad; "
        f"IK calls={total_ik_calls}"
    )

    return {
        "n_jerk_regions": int(len(regions)),
        "n_steps_refined": int(n_steps_refined),
        "n_steps_kept": int(n_steps_kept),
        "max_jump_inf_before": max_jump_before,
        "max_jump_inf_after": max_jump_after,
        "total_ik_calls": int(total_ik_calls),
    }


def _make_qpoint(
    *,
    t_global: float,
    user_wp_index: int,
    sub_index: int,
    target_position: np.ndarray,
    target_quat_wxyz: np.ndarray,
    achieved_position: np.ndarray,
    achieved_quat_wxyz: np.ndarray | None,
    cartesian_deviation: float,
    q_full: np.ndarray | None,
    ok: bool,
    failure_cause: str | None,
) -> QPoint:
    if q_full is not None:
        q_rail = float(q_full[0])
        q_waist = float(q_full[1])
        q_left = [float(x) for x in q_full[2:9]]
        q_right = [float(x) for x in q_full[9:16]]
    else:
        q_rail = q_waist = None
        q_left = q_right = None
    return QPoint(
        t_global=float(t_global),
        user_wp_index=int(user_wp_index),
        sub_index=int(sub_index),
        target_position=np.asarray(target_position, dtype=np.float64),
        target_quat_wxyz=np.asarray(target_quat_wxyz, dtype=np.float64),
        achieved_position=np.asarray(achieved_position, dtype=np.float64),
        achieved_quat_wxyz=(
            None
            if achieved_quat_wxyz is None
            else np.asarray(achieved_quat_wxyz, dtype=np.float64)
        ),
        cartesian_deviation=float(cartesian_deviation),
        q_full=None if q_full is None else np.asarray(q_full, dtype=np.float64).copy(),
        q_rail=q_rail,
        q_waist=q_waist,
        q_left_arm=q_left,
        q_right_arm=q_right,
        ok=bool(ok),
        failure_cause=failure_cause,
    )


def _phase4_resample(
    q_trajectory: list[QPoint],
    *,
    dt: float,
    joint_vel_limits: np.ndarray,
    arm_indices: list[int],
    agent: Any,
    initial_q: np.ndarray,
    frame: Any,
    tool_offset: np.ndarray | None,
    failures: list[dict],
    check_collision: bool = False,
    max_expansion: int = 1000,
) -> dict:
    """Splice linearly-interpolated joint-space samples into adjacent pairs that
    violate joint_vel_limits, so every interval respects the per-joint cap at
    control rate `1/dt`. Endpoints (and pairs already under the cap) are
    untouched.

    Achieved fields on inserts come from FK on the lerped q_full so
    cartesian_deviation reflects real tracking error. Target_quat is slerped
    via the same Tf path Phase 2 uses, so target_position+target_quat trace
    the same screw motion the planner originally emitted.

    `t_global` semantics shift: the function re-stamps EVERY entry to
    `i * dt` after splicing — pre-Phase-4 it was path-time
    (cumulative seg_dist / ee_speed), post-Phase-4 it is control-rate time
    (the assumption the velocity check at np.diff/dt already encodes).

    Mutates `q_trajectory` in place via slice-assign so callers' references
    stay live. Appends failure dicts (kinds: resample_skipped /
    resample_collision / resample_overflow). Returns a stats dict.
    """
    n_pairs_processed = 0
    n_pairs_split = 0
    n_inserted = 0
    n_pairs_skipped_no_q = 0
    n_collision_inserts = 0
    max_ratio_before = 0.0
    len_before = len(q_trajectory)

    if len_before < 2:
        return {
            "n_pairs_processed": 0,
            "n_pairs_split": 0,
            "n_inserted": 0,
            "n_pairs_skipped_no_q": 0,
            "n_collision_inserts": 0,
            "max_ratio_before": 0.0,
            "max_ratio_after": 0.0,
            "len_before": len_before,
            "len_after": len_before,
            "expansion_factor": 1.0,
        }

    # Precompute per-pair n_sub so we can short-circuit on overflow without
    # constructing any QPoints. (Cheap: just vector arithmetic.)
    plan_inserts: list[int] = []
    for i in range(len_before - 1):
        a = q_trajectory[i]
        b = q_trajectory[i + 1]
        if a.q_full is None or b.q_full is None:
            plan_inserts.append(0)
            continue
        dq = np.asarray(b.q_full, dtype=np.float64) - np.asarray(a.q_full, dtype=np.float64)
        worst = 0.0
        for k, dof in enumerate(arm_indices):
            lim = float(joint_vel_limits[k])
            if lim <= 0.0:
                continue
            r = float(abs(dq[dof])) / (dt * lim)
            if r > worst:
                worst = r
        if worst > max_ratio_before:
            max_ratio_before = worst
        if worst <= 1.0:
            plan_inserts.append(0)
        else:
            plan_inserts.append(int(math.ceil(worst)) - 1)

    projected_len = len_before + sum(plan_inserts)
    if projected_len > max_expansion * len_before:
        failures.append({
            "kind": "resample_overflow",
            "cause": (
                f"projected len {projected_len} > max_expansion ({max_expansion}) * "
                f"len_before ({len_before}); leaving q_trajectory untouched"
            ),
            "projected_len": int(projected_len),
            "len_before": int(len_before),
            "max_ratio_before": float(max_ratio_before),
        })
        return {
            "n_pairs_processed": int(len_before - 1),
            "n_pairs_split": 0,
            "n_inserted": 0,
            "n_pairs_skipped_no_q": 0,
            "n_collision_inserts": 0,
            "max_ratio_before": float(max_ratio_before),
            "max_ratio_after": float(max_ratio_before),
            "len_before": int(len_before),
            "len_after": int(len_before),
            "expansion_factor": 1.0,
            "joint_vel_limits": [float(x) for x in joint_vel_limits],
            "overflow": True,
        }

    new_traj: list[QPoint] = []
    cur_full = initial_q.copy()
    for i in range(len_before - 1):
        a = q_trajectory[i]
        b = q_trajectory[i + 1]
        new_traj.append(a)
        n_pairs_processed += 1
        n_sub_inserts = plan_inserts[i]
        if n_sub_inserts <= 0:
            if a.q_full is None or b.q_full is None:
                n_pairs_skipped_no_q += 1
                failures.append({
                    "kind": "resample_skipped",
                    "cause": "endpoint q_full is None (Phase 2 IK failure); pair left unchanged",
                    "left_index": int(i),
                    "right_index": int(i + 1),
                })
            continue

        n_pairs_split += 1
        n_sub = n_sub_inserts + 1
        q_a = np.asarray(a.q_full, dtype=np.float64)
        q_b = np.asarray(b.q_full, dtype=np.float64)
        pos_a = np.asarray(a.target_position, dtype=np.float64)
        pos_b = np.asarray(b.target_position, dtype=np.float64)
        tf_a = _build_target_tf(pos_a, np.asarray(a.target_quat_wxyz, dtype=np.float64))
        tf_b = _build_target_tf(pos_b, np.asarray(b.target_quat_wxyz, dtype=np.float64))

        for j in range(1, n_sub):
            t = j / float(n_sub)
            q_full_t = (1.0 - t) * q_a + t * q_b
            tf_t = tf_a.slerp(float(t), tf_b)
            target_pos_t = tf_t.matrix[:3, 3].astype(np.float64).copy()
            target_quat_t = _rotmat_to_quat_wxyz(tf_t.matrix[:3, :3])

            cur_full[:] = q_full_t
            agent.update_kin(cur_full)
            fk_matrix = agent.get_fk(frame).matrix
            achieved_t = _gripper_fk_to_tool_position(fk_matrix, tool_offset)
            achieved_quat_t = _gripper_fk_to_tool_quat_wxyz(fk_matrix, tool_offset)
            dev_t = float(np.linalg.norm(achieved_t - target_pos_t))

            insert_failure_cause: str | None = None
            if check_collision:
                arm_q_t = q_full_t[arm_indices]
                if _endpoint_collides(agent, initial_q, arm_indices, arm_q_t):
                    n_collision_inserts += 1
                    insert_failure_cause = (
                        f"resample_collision at lerp t={t:.4f} between "
                        f"trajectory indices {i} and {i + 1}"
                    )
                    failures.append({
                        "kind": "resample_collision",
                        "cause": insert_failure_cause,
                        "left_index": int(i),
                        "right_index": int(i + 1),
                        "lerp_t": float(t),
                    })

            new_traj.append(_make_qpoint(
                t_global=0.0,        # re-stamped after splice
                user_wp_index=a.user_wp_index,
                sub_index=0,         # re-stamped after splice
                target_position=target_pos_t,
                target_quat_wxyz=target_quat_t,
                achieved_position=achieved_t,
                achieved_quat_wxyz=achieved_quat_t,
                cartesian_deviation=dev_t,
                q_full=q_full_t,
                ok=insert_failure_cause is None,
                failure_cause=insert_failure_cause,
            ))
            n_inserted += 1

    new_traj.append(q_trajectory[-1])

    # Re-stamp t_global = i * dt and sub_index sequentially within each
    # user_wp_index group (human-readable JSON).
    sub_counter: dict[int, int] = {}
    for idx, qp in enumerate(new_traj):
        qp.t_global = float(idx) * float(dt)
        wp_i = int(qp.user_wp_index)
        sub_counter[wp_i] = sub_counter.get(wp_i, -1) + 1
        qp.sub_index = sub_counter[wp_i]

    # Compute max_ratio_after over the newly spliced trajectory.
    max_ratio_after = 0.0
    for idx in range(len(new_traj) - 1):
        a = new_traj[idx]
        b = new_traj[idx + 1]
        if a.q_full is None or b.q_full is None:
            continue
        dq = np.asarray(b.q_full, dtype=np.float64) - np.asarray(a.q_full, dtype=np.float64)
        for k, dof in enumerate(arm_indices):
            lim = float(joint_vel_limits[k])
            if lim <= 0.0:
                continue
            r = float(abs(dq[dof])) / (dt * lim)
            if r > max_ratio_after:
                max_ratio_after = r

    q_trajectory[:] = new_traj
    len_after = len(q_trajectory)

    print(
        f"[robot_check] Phase 4: resampled — split {n_pairs_split} pair(s), "
        f"inserted {n_inserted} step(s); "
        f"max ratio {max_ratio_before:.3f} → {max_ratio_after:.3f}; "
        f"len {len_before} → {len_after} ({len_after / max(len_before, 1):.2f}x)"
    )

    return {
        "n_pairs_processed": int(n_pairs_processed),
        "n_pairs_split": int(n_pairs_split),
        "n_inserted": int(n_inserted),
        "n_pairs_skipped_no_q": int(n_pairs_skipped_no_q),
        "n_collision_inserts": int(n_collision_inserts),
        "max_ratio_before": float(max_ratio_before),
        "max_ratio_after": float(max_ratio_after),
        "len_before": int(len_before),
        "len_after": int(len_after),
        "expansion_factor": float(len_after) / float(max(len_before, 1)),
        "joint_vel_limits": [float(x) for x in joint_vel_limits],
    }


def _phase2_topp_smooth_densify(
    *,
    agent: Any,
    cfg: "RobotCheckConfig",
    waypoints: list,
    selected_quats: list,
    cur: np.ndarray,
    seed_achieved_pos: np.ndarray,
    seed_achieved_quat: np.ndarray,
    seed_dev: float,
    seed_cause: str | None,
    arm_indices: list[int],
    frame: Any,
    tool_offset: np.ndarray | None,
    tool_offset_inv: np.ndarray | None,
    joint_vel_limits: np.ndarray,
    step: float,
    q_trajectory: list,
    failures: list,
) -> dict:
    """Three-pass TOPP-smoothness Phase 2.

    Pass 1 — geometric SLERP + per-step continuity IK at the same fine grid
    the legacy `slerp` method uses; samples are accumulated without
    QPoint emission. An IK miss short-circuits with a sentinel-fail
    QPoint (matches the legacy contract) and skips Passes 2 / 3.

    Pass 2 — analytic forward-backward TOPP on the collected joint path
    (`topp.topp_smooth`) returning per-segment dt that satisfies vel +
    accel caps with the configured headroom.

    Pass 3 — resample on a uniform `1 / cfg.hz` grid (`topp.resample_to_uniform_grid`),
    interpolate `q_full` (arm DOFs only — non-arm are frozen along the
    path), linear-interp `target_position`, slerp `target_quat`. Per
    dense sample: FK → cartesian-deviation gate, self-collision gate
    (linear-interp between two collision-free fine-grid samples is
    usually collision-free at small Δq but we still verify).

    Mutates `q_trajectory` and `failures` in place; returns a stats dict
    that lands in `metadata.robot_check.densify`.
    """
    from topp import topp_smooth, envelope_smooth, resample_to_uniform_grid, slerp_wxyz

    timing = str(getattr(cfg, "densify_timing", "topp_smooth"))
    if timing not in ("topp_smooth", "envelope"):
        raise ValueError(
            f"densify_timing must be 'topp_smooth' or 'envelope'; got {timing!r}"
        )

    if cfg.densify_joint_accel_limits is None:
        raise ValueError(
            "robot.densify.joint_accel_limits is required when "
            "densify.method == 'topp_smooth'"
        )
    a_max = np.asarray(cfg.densify_joint_accel_limits, dtype=np.float64)
    if a_max.shape == ():
        a_max = np.full(7, float(a_max), dtype=np.float64)
    if a_max.shape != (7,):
        raise ValueError(
            f"robot.densify.joint_accel_limits must broadcast to length 7; "
            f"got shape {a_max.shape}"
        )
    v_max = np.asarray(joint_vel_limits, dtype=np.float64)

    # ----- Pass 1: collect fine-grid samples -----
    wp0 = waypoints[0]
    q_full_path: list[np.ndarray] = [cur.copy()]
    target_pos_path: list[np.ndarray] = [np.asarray(wp0.position, dtype=np.float64)]
    target_quat_path: list[np.ndarray] = [np.asarray(selected_quats[0], dtype=np.float64)]
    user_wp_index_path: list[int] = [0]
    sub_index_in_seg_path: list[int] = [0]

    n_segments = len(waypoints) - 1
    print(f"[robot_check] Phase 2: TOPP densification ({n_segments} segments)")
    total_dist = 0.0
    n_step_failures = 0
    failed = False

    def _emit_seed_qpoint() -> None:
        if q_trajectory:
            return
        q_trajectory.append(_make_qpoint(
            t_global=0.0,
            user_wp_index=0,
            sub_index=0,
            target_position=target_pos_path[0],
            target_quat_wxyz=target_quat_path[0],
            achieved_position=seed_achieved_pos,
            achieved_quat_wxyz=seed_achieved_quat,
            cartesian_deviation=seed_dev,
            q_full=cur,
            ok=seed_cause is None,
            failure_cause=seed_cause,
        ))

    for seg_idx in range(n_segments):
        _print_progress(
            "Phase 2:  collect   ", seg_idx, n_segments,
            suffix=f"samples={len(q_full_path)} fails={n_step_failures}",
        )
        wp_a = waypoints[seg_idx]
        wp_b = waypoints[seg_idx + 1]
        quat_a = selected_quats[seg_idx]
        quat_b = selected_quats[seg_idx + 1]
        tf_from = _build_target_tf(wp_a.position, quat_a)
        tf_to = _build_target_tf(wp_b.position, quat_b)

        seg_start = np.asarray(wp_a.position, dtype=np.float64)
        seg_end = np.asarray(wp_b.position, dtype=np.float64)
        seg_dist = float(np.linalg.norm(seg_end - seg_start))
        total_dist += seg_dist
        N = max(cfg.min_waypoints, int(math.ceil(seg_dist / max(step, 1e-9))))
        ts = np.linspace(0.0, 1.0, N + 1)[1:]

        for sub_idx, t in enumerate(ts):
            tf_s = tf_from.slerp(float(t), tf_to)
            target_pos = tf_s.matrix[:3, 3].astype(np.float64).copy()
            target_quat = _rotmat_to_quat_wxyz(tf_s.matrix[:3, :3])
            tf_s_gripper = _tool_tf_to_gripper_tf(tf_s, tool_offset_inv)

            arm_q, diag = _pick_min_dist_collision_free_arm_ik(
                agent, cur, tf_s_gripper, frame, arm_indices, cfg.nb_redundant_search
            )
            if arm_q is None:
                cause = _diag_cause(diag, agent)
                failures.append({
                    "user_wp_index": seg_idx + 1,
                    "user_wp_name": getattr(wp_b, "name", None),
                    "sub_index": sub_idx + 1,
                    "kind": "ik",
                    "cause": cause,
                    "branches": diag["branches"],
                    "target_position": target_pos.tolist(),
                    "target_quat_wxyz": target_quat.tolist(),
                    "t_slerp": float(t),
                })
                _emit_seed_qpoint()
                q_trajectory.append(_make_qpoint(
                    t_global=0.0,
                    user_wp_index=seg_idx + 1,
                    sub_index=sub_idx + 1,
                    target_position=target_pos,
                    target_quat_wxyz=target_quat,
                    achieved_position=np.zeros(3, dtype=np.float64),
                    achieved_quat_wxyz=None,
                    cartesian_deviation=float("nan"),
                    q_full=None,
                    ok=False,
                    failure_cause=cause,
                ))
                n_step_failures += 1
                failed = True
                break

            cur[arm_indices] = arm_q
            agent.update_kin(cur)
            q_full_path.append(cur.copy())
            target_pos_path.append(target_pos)
            target_quat_path.append(target_quat)
            user_wp_index_path.append(seg_idx + 1)
            sub_index_in_seg_path.append(sub_idx + 1)
        if failed:
            break

    _print_progress(
        "Phase 2:  collect   ", n_segments, n_segments,
        suffix=f"samples={len(q_full_path)} fails={n_step_failures}", end="\n",
    )

    if failed:
        return {
            "method": "topp_smooth",
            "timing": timing,
            "failed": True,
            "total_dist": float(total_dist),
            "n_step_failures": int(n_step_failures),
            "n_fine_samples": int(len(q_full_path)),
            "n_dense_samples": int(len(q_trajectory)),
            "topp_total_time": None,
            "dt_min_used": (
                None if timing == "envelope" else float(cfg.densify_dt_min)
            ),
            "max_vel_ratio": None,
            "max_accel_ratio": None,
            "n_collision_failures": 0,
            "n_dev_failures": 0,
        }

    # ----- Pass 2: TOPP retime -----
    q_full_arr = np.array(q_full_path, dtype=np.float64)          # (M+1, 16)
    q_arm_arr = q_full_arr[:, arm_indices]                          # (M+1, 7)
    target_pos_arr = np.array(target_pos_path, dtype=np.float64)
    target_quat_arr = np.array(target_quat_path, dtype=np.float64)

    if timing == "envelope":
        dt_segs = envelope_smooth(q_arm=q_arm_arr, v_max=v_max, a_max=a_max)
    else:
        dt_segs = topp_smooth(
            q_arm=q_arm_arr,
            v_max=v_max,
            a_max=a_max,
            dt_min=float(cfg.densify_dt_min),
            vel_headroom=float(cfg.densify_vel_headroom),
            accel_headroom=float(cfg.densify_accel_headroom),
        )
    topp_total_time = float(dt_segs.sum())

    if dt_segs.size > 0:
        vel = np.abs(np.diff(q_arm_arr, axis=0)) / dt_segs[:, None]
        max_vel_ratio = float((vel / v_max[None, :]).max())
    else:
        max_vel_ratio = 0.0
    if dt_segs.size >= 2:
        velocities = np.diff(q_arm_arr, axis=0) / dt_segs[:, None]   # (M, 7)
        dvel = np.diff(velocities, axis=0)
        seg_pair_dt = (dt_segs[:-1] + dt_segs[1:]) / 2.0
        accel = np.abs(dvel) / seg_pair_dt[:, None]
        max_accel_ratio = float((accel / a_max[None, :]).max())
    else:
        max_accel_ratio = 0.0

    # ----- Pass 3: uniform resample + emit QPoints -----
    target_dt = 1.0 / float(cfg.hz)
    t_dense, parent, alpha = resample_to_uniform_grid(dt_segs, target_dt)
    n_dense = int(t_dense.size)

    print(
        f"[robot_check] Phase 2: {timing} retime + resample "
        f"({q_arm_arr.shape[0]} -> {n_dense} samples, "
        f"total_time {topp_total_time:.3f}s, vel_ratio {max_vel_ratio:.3f}, "
        f"accel_ratio {max_accel_ratio:.3f})"
    )

    n_collision_failures = 0
    n_dev_failures = 0

    for k in range(n_dense):
        i = int(parent[k])
        a = float(alpha[k])
        q_full_k = q_full_arr[i].copy()
        q_full_k[arm_indices] = (
            (1.0 - a) * q_full_arr[i, arm_indices]
            + a * q_full_arr[i + 1, arm_indices]
        )
        target_pos_k = (1.0 - a) * target_pos_arr[i] + a * target_pos_arr[i + 1]
        target_quat_k = slerp_wxyz(target_quat_arr[i], target_quat_arr[i + 1], a)

        # Carry user_wp / sub_index from the closer endpoint of the fine-grid
        # segment; consumers that need exact mapping can still recover the
        # source segment from the t_global / target_position fields.
        if a >= 0.5:
            uwi = int(user_wp_index_path[i + 1])
            sii = int(sub_index_in_seg_path[i + 1])
            wp_name = getattr(waypoints[uwi], "name", None) if uwi < len(waypoints) else None
        else:
            uwi = int(user_wp_index_path[i])
            sii = int(sub_index_in_seg_path[i])
            wp_name = getattr(waypoints[uwi], "name", None) if uwi < len(waypoints) else None

        agent.update_kin(q_full_k, None, False)
        fk_matrix = agent.get_fk(frame).matrix
        achieved = _gripper_fk_to_tool_position(fk_matrix, tool_offset)
        achieved_quat = _gripper_fk_to_tool_quat_wxyz(fk_matrix, tool_offset)
        dev = float(np.linalg.norm(achieved - target_pos_k))
        cause: str | None = None
        if dev > cfg.max_cartesian_deviation:
            cause = (
                f"Cartesian deviation {dev:.4f}m exceeds "
                f"{cfg.max_cartesian_deviation:.4f}m"
            )
            failures.append({
                "user_wp_index": uwi,
                "user_wp_name": wp_name,
                "sub_index": sii,
                "kind": "densify_deviation",
                "cause": cause,
                "deviation": dev,
                "target_position": target_pos_k.tolist(),
                "target_quat_wxyz": target_quat_k.tolist(),
                "achieved_position": achieved.tolist(),
            })
            n_dev_failures += 1
        if _is_self_colliding(q_full_k):
            col_cause = "self-collision on TOPP-resampled densify sample"
            failures.append({
                "user_wp_index": uwi,
                "user_wp_name": wp_name,
                "sub_index": sii,
                "kind": "densify_collision",
                "cause": col_cause,
                "target_position": target_pos_k.tolist(),
                "target_quat_wxyz": target_quat_k.tolist(),
            })
            if cause is None:
                cause = col_cause
            n_collision_failures += 1

        q_trajectory.append(_make_qpoint(
            t_global=k * target_dt,
            user_wp_index=uwi,
            sub_index=sii,
            target_position=target_pos_k,
            target_quat_wxyz=target_quat_k,
            achieved_position=achieved,
            achieved_quat_wxyz=achieved_quat,
            cartesian_deviation=dev,
            q_full=q_full_k,
            ok=cause is None,
            failure_cause=cause,
        ))

    n_step_failures += n_collision_failures + n_dev_failures

    return {
        "method": "topp_smooth",
        "timing": timing,
        "failed": False,
        "total_dist": float(total_dist),
        "n_step_failures": int(n_step_failures),
        "n_fine_samples": int(q_arm_arr.shape[0]),
        "n_dense_samples": int(n_dense),
        "topp_total_time": float(topp_total_time),
        "dt_min_used": (
            None if timing == "envelope" else float(cfg.densify_dt_min)
        ),
        "vel_headroom": (
            None if timing == "envelope" else float(cfg.densify_vel_headroom)
        ),
        "accel_headroom": (
            None if timing == "envelope" else float(cfg.densify_accel_headroom)
        ),
        "max_vel_ratio": float(max_vel_ratio),
        "max_accel_ratio": float(max_accel_ratio),
        "n_collision_failures": int(n_collision_failures),
        "n_dev_failures": int(n_dev_failures),
        "joint_vel_limits": [float(x) for x in v_max],
        "joint_accel_limits": [float(x) for x in a_max],
    }


def check_trajectory(waypoints: list, cfg: RobotCheckConfig) -> CheckResult:
    """Check a list of TCP waypoints (objects with .position and .quaternion_wxyz)."""
    if cfg.arm not in ("left_arm", "right_arm"):
        raise ValueError(f"cfg.arm must be 'left_arm' or 'right_arm', got {cfg.arm!r}")
    if cfg.initial_q is None:
        raise ValueError("cfg.initial_q is required (16-DOF starting state)")
    initial_q = np.asarray(cfg.initial_q, dtype=np.float64)
    if initial_q.shape != (16,):
        raise ValueError(f"cfg.initial_q must have length 16, got {initial_q.shape}")

    arm_indices = HBMP_COMPONENT_DOF_INDICES[cfg.arm]

    # Build the ampl_motion kin + colcvx sidecar. Replaces the prior
    # `Robot_T2DA2(...)` construction — KinHB11 now handles FK, IK, and is
    # the source of poses the colcvx scene reads. The cfg.hbmp_arm_left /
    # right / torso fields are accepted for back-compat but ignored: the
    # wheel embeds the hb11 robot identity.
    print(
        f"[robot_check] building ampl_motion kin + colcvx (robot=hb11, "
        f"magic_distance={_CHECK_MAGIC_DISTANCE} m)"
    )
    _setup_collision_checker(cfg)

    # Alias `agent` to the KinHB11 instance the sidecar holds. The rest of
    # this function still uses the `agent.*` naming — only the underlying
    # object changed.
    agent = _collision_checker["kin"]
    frame = (
        agent.Frames.left_link_tactile_center
        if cfg.arm == "left_arm"
        else agent.Frames.right_link_tactile_center
    )
    agent.update_kin(initial_q, None, False)

    tool_offset = (
        None if cfg.tool_offset is None else np.asarray(cfg.tool_offset, dtype=np.float64)
    )
    if tool_offset is not None and tool_offset.shape != (4, 4):
        raise ValueError(f"cfg.tool_offset must be 4x4, got {tool_offset.shape}")
    tool_offset_inv = None if tool_offset is None else np.linalg.inv(tool_offset)

    if cfg.joint_vel_limits is None:
        joint_vel_limits = np.array([1.0] * 7, dtype=np.float64)
    else:
        # Accept scalar (broadcast to 7) or length-7 sequence. YAML's
        # `joint_vel_limits: 1` is the common shorthand for "1 rad/s on
        # every joint".
        raw = cfg.joint_vel_limits
        if isinstance(raw, (int, float)):
            joint_vel_limits = np.full(7, float(raw), dtype=np.float64)
        else:
            joint_vel_limits = np.asarray(raw, dtype=np.float64)
            if joint_vel_limits.shape == ():
                joint_vel_limits = np.full(7, float(joint_vel_limits), dtype=np.float64)
            if joint_vel_limits.shape != (7,):
                raise ValueError(
                    f"cfg.joint_vel_limits must be scalar or length-7 sequence, "
                    f"got shape {joint_vel_limits.shape}"
                )

    dt = 1.0 / float(cfg.hz)
    step = float(cfg.ee_speed) * dt

    # Yaw offsets (deg) seeding the per-waypoint IK pool the A* search expands
    # over. Each entry rotates the canonical quat locally around tool Z (=
    # -surface_normal in align_z_to_normal mode), so the orientation tolerance
    # follows the surface normal regardless of the waypoint's base yaw. Default
    # is 18 values at 10° spacing covering one half-circle (0–170°).
    if cfg.yaw_search_offsets_deg is None:
        yaw_offsets_deg = [float(d) for d in range(0, 180, 10)]
    else:
        yaw_offsets_deg = [float(o) for o in cfg.yaw_search_offsets_deg]
    if not yaw_offsets_deg:
        raise ValueError("cfg.yaw_search_offsets_deg must be non-empty")

    # Pitch offsets (deg) around tool Y — taken cartesian-product with the yaw
    # and roll axes. Default = [0.0] (no pitch search); widen via YAML when
    # needed.
    if cfg.pitch_search_offsets_deg is None:
        pitch_offsets_deg = [0.0]
    else:
        pitch_offsets_deg = [float(o) for o in cfg.pitch_search_offsets_deg]
    if not pitch_offsets_deg:
        raise ValueError("cfg.pitch_search_offsets_deg must be non-empty")

    # Roll offsets (deg) around tool X — taken cartesian-product with the yaw
    # and pitch axes. Default = [0.0] (no roll search; legacy behaviour);
    # widen via YAML when the tool's contact is roll-tolerant.
    if cfg.roll_search_offsets_deg is None:
        roll_offsets_deg = [0.0]
    else:
        roll_offsets_deg = [float(o) for o in cfg.roll_search_offsets_deg]
    if not roll_offsets_deg:
        raise ValueError("cfg.roll_search_offsets_deg must be non-empty")

    failures: list[dict] = []

    if len(waypoints) == 0:
        return CheckResult(
            ok=False, n_q_waypoints=0, total_dist=0.0, dt=dt,
            worst_joint_speed_ratio=0.0, worst_dof=-1, joint_speed_ok=True,
            q_trajectory=[], failures=[{"cause": "empty waypoint list"}],
        )

    # ---------- Phase 1: search over per-waypoint IK pools ----------
    if cfg.search_method == "astar":
        phase1_fn = _astar_phase1
    elif cfg.search_method == "kshortest_dp":
        phase1_fn = _kshortest_dp_phase1
    else:
        raise ValueError(
            f"Unknown cfg.search_method: {cfg.search_method!r}; "
            f"expected 'astar' or 'kshortest_dp'"
        )
    print(f"[robot_check] Phase 1 strategy: {cfg.search_method}")
    (
        selected_q_arms,
        selected_quats,
        phase1_failures,
        ik_pools,
        ik_pool_statuses,
        ik_selected_flat_indices,
        ik_pool_collision_checked,
    ) = phase1_fn(
        agent, cfg, waypoints, initial_q, arm_indices, frame,
        tool_offset_inv, yaw_offsets_deg, pitch_offsets_deg, roll_offsets_deg,
    )
    # `ik_pools` (the per-waypoint (q_arm, quat_wxyz) tuples) is always
    # retained on CheckResult so plan.py can compute per-waypoint pool
    # sizes for the main 2D PNG without re-reading the sidecar.
    # `save_all_ik_results` is consumed downstream in
    # plan.write_trajectory_json to gate the heavy sidecar payload only;
    # the lightweight Phase 1a outputs (per-candidate convergence bools +
    # Phase 1b's selected indices + collision verdicts) always pass through.
    saved_pools = ik_pools
    saved_pool_statuses = ik_pool_statuses
    saved_selected_indices = ik_selected_flat_indices
    saved_collision_checked = ik_pool_collision_checked
    if selected_q_arms is None:
        return CheckResult(
            ok=False, n_q_waypoints=0, total_dist=0.0, dt=dt,
            worst_joint_speed_ratio=0.0, worst_dof=-1, joint_speed_ok=True,
            q_trajectory=[], failures=phase1_failures, ik_pools=saved_pools,
            ik_pool_statuses=saved_pool_statuses,
            ik_selected_flat_indices=saved_selected_indices,
            ik_pool_collision_checked=saved_collision_checked,
            collision_pair_stats=_finalize_collision_pair_stats(),
        )

    # Write the chosen orientations back so the JSON output and Phase 2 slerp
    # pass agree on each waypoint's quaternion.
    for i, wp in enumerate(waypoints):
        wp.quaternion_wxyz = selected_quats[i].copy()
    wp0 = waypoints[0]

    # ---------- Phase 2: densification ----------
    # Two methods, dispatched on cfg.densify_method:
    #   - `slerp`        : legacy uniform-arc-length SLERP + per-step IK,
    #                       `t_global` = path-time (Σ seg_dist / N).
    #   - `topp_smooth`  : same SLERP + per-step IK at the same fine grid,
    #                       followed by an analytic forward-backward TOPP
    #                       retime (`topp.topp_smooth`) and a uniform-1/hz
    #                       resample, so the emitted QPoints carry
    #                       control-time `t_global = k / hz` and a
    #                       vel + accel feasible joint trajectory by
    #                       construction.
    agent.update_kin(initial_q, None, False)

    cur = initial_q.copy()
    cur[arm_indices] = selected_q_arms[0]
    agent.update_kin(cur, None, False)

    q_trajectory: list[QPoint] = []
    total_dist = 0.0

    fk0_matrix = agent.get_fk(frame).matrix
    achieved0 = _gripper_fk_to_tool_position(fk0_matrix, tool_offset)
    achieved0_quat = _gripper_fk_to_tool_quat_wxyz(fk0_matrix, tool_offset)
    desired0 = np.asarray(wp0.position, dtype=np.float64)
    dev0 = float(np.linalg.norm(achieved0 - desired0))
    cause0: str | None = None
    if dev0 > cfg.max_cartesian_deviation:
        cause0 = (
            f"Cartesian deviation {dev0:.4f}m exceeds {cfg.max_cartesian_deviation:.4f}m"
        )
        failures.append({
            "user_wp_index": 0,
            "user_wp_name": getattr(wp0, "name", None),
            "sub_index": 0,
            "kind": "deviation",
            "cause": cause0,
            "deviation": dev0,
            "target_position": desired0.tolist(),
            "target_quat_wxyz": [float(x) for x in selected_quats[0]],
            "achieved_position": achieved0.tolist(),
        })

    densify_stats: dict | None = None

    if cfg.densify_method == "topp_smooth":
        densify_stats = _phase2_topp_smooth_densify(
            agent=agent,
            cfg=cfg,
            waypoints=waypoints,
            selected_quats=selected_quats,
            cur=cur,
            seed_achieved_pos=achieved0,
            seed_achieved_quat=achieved0_quat,
            seed_dev=dev0,
            seed_cause=cause0,
            arm_indices=arm_indices,
            frame=frame,
            tool_offset=tool_offset,
            tool_offset_inv=tool_offset_inv,
            joint_vel_limits=joint_vel_limits,
            step=step,
            q_trajectory=q_trajectory,
            failures=failures,
        )
        total_dist = float(densify_stats.get("total_dist") or 0.0)
        n_step_failures = int(densify_stats.get("n_step_failures") or 0)
    elif cfg.densify_method == "slerp":
        q_trajectory.append(_make_qpoint(
            t_global=0.0,
            user_wp_index=0,
            sub_index=0,
            target_position=desired0,
            target_quat_wxyz=selected_quats[0],
            achieved_position=achieved0,
            achieved_quat_wxyz=achieved0_quat,
            cartesian_deviation=dev0,
            q_full=cur,
            ok=cause0 is None,
            failure_cause=cause0,
        ))

        t_global = 0.0
        n_segments = len(waypoints) - 1
        print(f"[robot_check] Phase 2: slerp densification ({n_segments} segments)")
        n_step_failures = 0
        for seg_idx in range(len(waypoints) - 1):
            _print_progress(
                "Phase 2:  densify   ", seg_idx, n_segments,
                suffix=f"q_traj={len(q_trajectory)} fails={n_step_failures}",
            )
            wp_a = waypoints[seg_idx]
            wp_b = waypoints[seg_idx + 1]
            quat_a = selected_quats[seg_idx]
            quat_b = selected_quats[seg_idx + 1]
            tf_from = _build_target_tf(wp_a.position, quat_a)
            tf_to = _build_target_tf(wp_b.position, quat_b)

            seg_start = np.asarray(wp_a.position, dtype=np.float64)
            seg_end = np.asarray(wp_b.position, dtype=np.float64)
            seg_dist = float(np.linalg.norm(seg_end - seg_start))
            total_dist += seg_dist
            N = max(cfg.min_waypoints, int(math.ceil(seg_dist / max(step, 1e-9))))
            ts = np.linspace(0.0, 1.0, N + 1)[1:]

            for sub_idx, t in enumerate(ts):
                tf_s = tf_from.slerp(float(t), tf_to)
                target_pos = tf_s.matrix[:3, 3].astype(np.float64).copy()
                target_quat = _rotmat_to_quat_wxyz(tf_s.matrix[:3, :3])

                tf_s_gripper = _tool_tf_to_gripper_tf(tf_s, tool_offset_inv)

                t_global += seg_dist / float(N)

                arm_q, diag = _pick_min_dist_collision_free_arm_ik(
                    agent, cur, tf_s_gripper, frame, arm_indices, cfg.nb_redundant_search
                )
                if arm_q is None:
                    cause = _diag_cause(diag, agent)
                    failures.append({
                        "user_wp_index": seg_idx + 1,
                        "user_wp_name": getattr(wp_b, "name", None),
                        "sub_index": sub_idx + 1,
                        "kind": "ik",
                        "cause": cause,
                        "branches": diag["branches"],
                        "target_position": target_pos.tolist(),
                        "target_quat_wxyz": target_quat.tolist(),
                        "t_slerp": float(t),
                    })
                    q_trajectory.append(_make_qpoint(
                        t_global=t_global,
                        user_wp_index=seg_idx + 1,
                        sub_index=sub_idx + 1,
                        target_position=target_pos,
                        target_quat_wxyz=target_quat,
                        achieved_position=np.zeros(3, dtype=np.float64),
                        achieved_quat_wxyz=None,
                        cartesian_deviation=float("nan"),
                        q_full=None,
                        ok=False,
                        failure_cause=cause,
                    ))
                    n_step_failures += 1
                    continue

                cur[arm_indices] = arm_q
                agent.update_kin(cur)
                fk_matrix = agent.get_fk(frame).matrix
                achieved = _gripper_fk_to_tool_position(fk_matrix, tool_offset)
                achieved_quat = _gripper_fk_to_tool_quat_wxyz(fk_matrix, tool_offset)
                dev = float(np.linalg.norm(achieved - target_pos))
                cause = None
                if dev > cfg.max_cartesian_deviation:
                    cause = (
                        f"Cartesian deviation {dev:.4f}m exceeds "
                        f"{cfg.max_cartesian_deviation:.4f}m"
                    )
                    failures.append({
                        "user_wp_index": seg_idx + 1,
                        "user_wp_name": getattr(wp_b, "name", None),
                        "sub_index": sub_idx + 1,
                        "kind": "deviation",
                        "cause": cause,
                        "deviation": dev,
                        "target_position": target_pos.tolist(),
                        "target_quat_wxyz": target_quat.tolist(),
                        "achieved_position": achieved.tolist(),
                    })
                    n_step_failures += 1

                q_trajectory.append(_make_qpoint(
                    t_global=t_global,
                    user_wp_index=seg_idx + 1,
                    sub_index=sub_idx + 1,
                    target_position=target_pos,
                    target_quat_wxyz=target_quat,
                    achieved_position=achieved,
                    achieved_quat_wxyz=achieved_quat,
                    cartesian_deviation=dev,
                    q_full=cur,
                    ok=cause is None,
                    failure_cause=cause,
                ))

        _print_progress(
            "Phase 2:  densify   ", n_segments, n_segments,
            suffix=f"q_traj={len(q_trajectory)} fails={n_step_failures}",
            end="\n",
        )
    else:
        raise ValueError(
            f"cfg.densify_method must be 'topp_smooth' or 'slerp'; got "
            f"{cfg.densify_method!r}"
        )

    # ---------- Phase 3: jerk-region refinement ----------
    refine_stats: dict | None = None
    if cfg.refine_enabled:
        if cfg.plane_u_axis is None or cfg.plane_v_axis is None:
            raise ValueError(
                "cfg.refine_enabled=True requires cfg.plane_u_axis and "
                "cfg.plane_v_axis (3,) — populate from PlanResult.plane in plan.py"
            )
        plane_u = np.asarray(cfg.plane_u_axis, dtype=np.float64)
        plane_v = np.asarray(cfg.plane_v_axis, dtype=np.float64)
        if plane_u.shape != (3,) or plane_v.shape != (3,):
            raise ValueError(
                f"cfg.plane_u_axis/plane_v_axis must be shape (3,); got "
                f"{plane_u.shape} / {plane_v.shape}"
            )
        refine_stats = _phase3_refine(
            agent=agent,
            cfg=cfg,
            q_trajectory=q_trajectory,
            arm_key=("q_left_arm" if cfg.arm == "left_arm" else "q_right_arm"),
            arm_indices=arm_indices,
            initial_q=initial_q,
            frame=frame,
            tool_offset=tool_offset,
            tool_offset_inv=tool_offset_inv,
            plane_u_axis=plane_u,
            plane_v_axis=plane_v,
        )

    # ---------- Phase 4: joint-velocity resampling ----------
    resample_stats: dict | None = None
    if cfg.resample_enabled:
        if cfg.resample_joint_vel_limits is None:
            resample_limits = joint_vel_limits
        else:
            resample_limits = np.asarray(cfg.resample_joint_vel_limits, dtype=np.float64)
            if resample_limits.shape != (7,):
                raise ValueError(
                    f"cfg.resample_joint_vel_limits must have length 7 (arm DOFs), "
                    f"got {resample_limits.shape}"
                )
        resample_stats = _phase4_resample(
            q_trajectory,
            dt=dt,
            joint_vel_limits=resample_limits,
            arm_indices=arm_indices,
            agent=agent,
            initial_q=initial_q,
            frame=frame,
            tool_offset=tool_offset,
            failures=failures,
            check_collision=cfg.resample_check_collision,
            max_expansion=cfg.resample_max_expansion,
        )

    # Joint-speed analysis (run on the post-refine + post-resample q_trajectory).
    positions_full_for_speed: list[np.ndarray] = [
        qp.q_full for qp in q_trajectory if qp.q_full is not None
    ]
    worst_dof = -1
    worst_ratio = 0.0
    joint_speed_ok = True
    if len(positions_full_for_speed) >= 2:
        positions_arr = np.stack(positions_full_for_speed, axis=0)
        dq = np.diff(positions_arr, axis=0) / dt
        for k, dof in enumerate(arm_indices):
            lim = float(joint_vel_limits[k])
            if lim <= 0:
                continue
            ratio = float(np.max(np.abs(dq[:, dof])) / lim)
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_dof = dof
        if worst_ratio > 1.0:
            joint_speed_ok = False
            failures.append({
                "kind": "joint_speed",
                "cause": (
                    f"arm dof {worst_dof} exceeds limit by {worst_ratio:.2f}x "
                    f"(ee_speed={cfg.ee_speed} m/s, dt={dt:.4f}s)"
                ),
                "worst_dof": worst_dof,
                "worst_ratio": worst_ratio,
            })

    ok = joint_speed_ok and not failures

    return CheckResult(
        ok=ok,
        n_q_waypoints=len(q_trajectory),
        total_dist=total_dist,
        dt=dt,
        worst_joint_speed_ratio=worst_ratio,
        worst_dof=worst_dof,
        joint_speed_ok=joint_speed_ok,
        q_trajectory=q_trajectory,
        failures=failures,
        ik_pools=saved_pools,
        ik_pool_statuses=saved_pool_statuses,
        ik_selected_flat_indices=saved_selected_indices,
        ik_pool_collision_checked=saved_collision_checked,
        refine_stats=refine_stats,
        resample_stats=resample_stats,
        densify_stats=densify_stats,
        collision_pair_stats=_finalize_collision_pair_stats(),
    )
