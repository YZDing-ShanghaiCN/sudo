"""Densify an existing optimizer JSON to a target control rate.

Doesn't re-run the optimization. Reads the coarse `q_trajectory`, linear-
interps `q_arm` onto a uniform 1/hz grid in joint space, re-evaluates FK
and self-collision at every dense step (when AMPL is available), and
writes a `*_dense_<hz>hz.json` sidecar.

Usage:
    python -m tools.surface_cover_planner.optimize.densify_cli \\
        --input outputs/<name>_<ts>.json --hz 20
    python -m tools.surface_cover_planner.optimize.densify_cli \\
        --input outputs/<name>_<ts>.json --hz 20 \\
        --config configs/config_autorun.yaml      # optional, for wall AABB
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from .densify import densify_joint_space, densify_targets

# geometry.py lives at the top level of tools/surface_cover_planner/; add
# that directory to sys.path so the import works whether this CLI is run as
# `python -m tools.surface_cover_planner.optimize.densify_cli` (parent on
# path already) or directly from inside the planner directory.
import sys as _sys
_PLANNER_DIR = str(Path(__file__).resolve().parent.parent)
if _PLANNER_DIR not in _sys.path:
    _sys.path.insert(0, _PLANNER_DIR)
from geometry import tool_pose_to_tactile_pose  # noqa: E402


def _load_config_wall(config_path: str | None) -> dict | None:
    """Read robot.wall out of a YAML config (for collision checks)."""
    if not config_path:
        return None
    import yaml

    p = Path(config_path)
    if not p.exists():
        return None
    with open(p) as f:
        cfg = yaml.safe_load(f)
    return ((cfg.get("robot") or {}).get("wall"))


def _build_dense_entry(
    *,
    i: int,
    dense,
    dense_q_full: np.ndarray,
    dense_targets_pos: np.ndarray,
    dense_targets_quat: np.ndarray,
    dense_pos: np.ndarray,
    dense_quat: np.ndarray,
    cart_dev: np.ndarray,
    dense_coll: np.ndarray,
    collision_evaluated: bool,
    tool_offset_4x4: np.ndarray,
) -> dict:
    target_pos_i = dense_targets_pos[i]
    target_quat_i = dense_targets_quat[i]
    achieved_pos_i = None if np.isnan(dense_pos[i]).any() else dense_pos[i]
    achieved_quat_i = None if np.isnan(dense_quat[i]).any() else dense_quat[i]

    target_tactile_pos, target_tactile_quat = tool_pose_to_tactile_pose(
        target_pos_i, target_quat_i, tool_offset_4x4
    )
    if achieved_pos_i is None:
        achieved_tactile_pos = None
        achieved_tactile_quat = None
    else:
        achieved_tactile_pos, achieved_tactile_quat = tool_pose_to_tactile_pose(
            achieved_pos_i, achieved_quat_i, tool_offset_4x4
        )

    return {
        "t_global": float(dense.t_global[i]),
        "q_arm": [float(x) for x in dense.q_arm[i]],
        "q_full": (None if np.isnan(dense_q_full[i]).any() else [float(x) for x in dense_q_full[i]]),
        "target_position": [float(x) for x in target_pos_i],
        "target_quat_wxyz": [float(x) for x in target_quat_i],
        "target_tactile_position": [float(x) for x in target_tactile_pos],
        "target_tactile_quat_wxyz": (
            None if target_tactile_quat is None else [float(x) for x in target_tactile_quat]
        ),
        "achieved_position": (None if achieved_pos_i is None else [float(x) for x in achieved_pos_i]),
        "achieved_quat_wxyz": (None if achieved_quat_i is None else [float(x) for x in achieved_quat_i]),
        "achieved_tactile_position": (
            None if achieved_tactile_pos is None else [float(x) for x in achieved_tactile_pos]
        ),
        "achieved_tactile_quat_wxyz": (
            None if achieved_tactile_quat is None else [float(x) for x in achieved_tactile_quat]
        ),
        "cartesian_deviation": (None if np.isnan(cart_dev[i]) else float(cart_dev[i])),
        "self_collision": (bool(dense_coll[i]) if collision_evaluated else None),
        "parent_segment": int(dense.parent_segment[i]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Densify an existing optimizer JSON.")
    parser.add_argument("--input", required=True, help="optimizer JSON path")
    parser.add_argument("--hz", type=float, required=True, help="target control rate (Hz)")
    parser.add_argument("--output", default=None, help="output JSON path (default: <input>_dense_<hz>hz.json)")
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config that produced the input (optional; supplies wall AABB for collision checks)",
    )
    parser.add_argument(
        "--no-collision",
        action="store_true",
        help="skip self-collision re-evaluation (useful when AMPL / RoboSkill aren't installed)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    with open(input_path) as f:
        data = json.load(f)

    qt = data.get("q_trajectory") or []
    if len(qt) < 2:
        raise ValueError(f"input {input_path} has no usable q_trajectory (n={len(qt)})")

    q_arm = np.asarray([qp["q_arm"] for qp in qt], dtype=np.float64)
    target_pos = np.asarray([qp["target_position"] for qp in qt], dtype=np.float64)
    target_quat = np.asarray([qp["target_quat_wxyz"] for qp in qt], dtype=np.float64)
    t_global = np.asarray([qp["t_global"] for qp in qt], dtype=np.float64)
    dt = np.diff(t_global)
    if (dt <= 0).any():
        raise ValueError(f"input {input_path} has non-monotonic t_global; can't densify")

    md = data.get("metadata") or {}
    arm = md.get("arm", "left_arm")
    tool_offset = np.asarray(md.get("tool_offset_4x4", np.eye(4).tolist()), dtype=np.float64)
    # initial_q reconstructed from the first dense q_full so torso/off-arm
    # match what the optimizer actually planned against.
    initial_q = [float(x) for x in qt[0]["q_full"]]

    target_dt = 1.0 / float(args.hz)
    dense = densify_joint_space(q_arm, dt, target_dt)
    dense_targets_pos, dense_targets_quat = densify_targets(target_pos, target_quat, dt, target_dt)
    n_dense = dense.q_arm.shape[0]
    print(f"[densify_cli] N={q_arm.shape[0]} -> N_dense={n_dense} at {args.hz} Hz "
          f"(target_dt={target_dt:.5f}s)")

    # Optional FK + collision re-evaluation via AMPL.
    dense_pos = np.empty((n_dense, 3), dtype=np.float64)
    dense_quat = np.empty((n_dense, 4), dtype=np.float64)
    dense_coll = np.zeros(n_dense, dtype=bool)
    dense_q_full = np.empty((n_dense, 16), dtype=np.float64)
    collision_evaluated = False
    try:
        from .robot_model import RobotConfig, RobotModel

        wall = _load_config_wall(args.config)
        rcfg = RobotConfig(
            arm=arm,
            initial_q=initial_q,
            wall=wall,
            tool_offset_4x4=tool_offset if not np.allclose(tool_offset, np.eye(4)) else None,
            joint_vel_limits=[1.0] * 7,
        )
        model = RobotModel(rcfg)
        do_collision = not args.no_collision
        for i in range(n_dense):
            p, qw = model.fk_pose(dense.q_arm[i])
            dense_pos[i] = p
            dense_quat[i] = qw
            dense_q_full[i] = model.assemble_q_full(dense.q_arm[i])
            if do_collision:
                dense_coll[i] = model.self_collides(dense.q_arm[i])
        collision_evaluated = do_collision
        cart_dev = np.linalg.norm(dense_pos - dense_targets_pos, axis=1)
        v_max = model.v_max
        vel_ratios = np.abs(np.diff(dense.q_arm, axis=0)) / (
            dense.dt[:, None] * v_max[None, :] + 1e-12
        )
    except Exception as e:  # noqa: BLE001
        # No AMPL -> fall back to FK-less output (q only). Cart_dev / collision
        # left as NaN / False; consumer can re-run inside docker if needed.
        print(f"[densify_cli] FK / collision unavailable: {e}")
        dense_pos[:] = np.nan
        dense_quat[:] = np.nan
        dense_q_full[:] = np.nan
        n_arm = q_arm.shape[1]
        dense_q_full[:, :n_arm] = dense.q_arm  # best-effort placeholder
        cart_dev = np.full(n_dense, np.nan)
        vel_ratios = np.abs(np.diff(dense.q_arm, axis=0)) / (dense.dt[:, None] * 1.0 + 1e-12)

    dense_stats = {
        "target_hz": float(args.hz),
        "target_dt": float(target_dt),
        "n_dense": int(n_dense),
        "total_time": float(dense.t_global[-1]),
        "max_cart_dev": (None if np.isnan(cart_dev).all() else float(np.nanmax(cart_dev))),
        "max_joint_vel_ratio": float(vel_ratios.max()),
        "n_collisions": (int(dense_coll.sum()) if collision_evaluated else None),
        "collision_evaluated": collision_evaluated,
    }
    print(f"[densify_cli] stats: {dense_stats}")

    payload = {
        "metadata": {
            "source_trajopt_json": str(input_path.name),
            "config_path": args.config,
            "generated_at_iso": _dt.datetime.now().isoformat(timespec="seconds"),
            "arm": arm,
            "tool_offset_4x4": tool_offset.tolist(),
            "densify_stats": dense_stats,
        },
        "q_trajectory": [
            _build_dense_entry(
                i=i,
                dense=dense,
                dense_q_full=dense_q_full,
                dense_targets_pos=dense_targets_pos,
                dense_targets_quat=dense_targets_quat,
                dense_pos=dense_pos,
                dense_quat=dense_quat,
                cart_dev=cart_dev,
                dense_coll=dense_coll,
                collision_evaluated=collision_evaluated,
                tool_offset_4x4=tool_offset,
            )
            for i in range(n_dense)
        ],
    }

    if args.output:
        out_path = Path(args.output)
    else:
        hz_tag = f"{int(args.hz)}hz" if float(args.hz).is_integer() else f"{args.hz}hz"
        out_path = input_path.with_name(input_path.stem + f"_dense_{hz_tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=True)
    print(f"wrote {n_dense} dense steps -> {out_path}")


if __name__ == "__main__":
    main()
