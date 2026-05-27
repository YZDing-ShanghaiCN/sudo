# surface_cover_planner

Boustrophedon (back-and-forth lawnmower) surface-coverage planner. Given a mesh
and a polygonal region on/near its surface, it generates an ordered list of TCP
poses (in the robot/world frame) that the robot can execute to cover the
surface, and a standalone visualizer for the result.

## Layout

```
surface_cover_planner/
├── plan.py              # entrypoint: YAML config -> trajectory JSON (+ optional optimizer)
├── visualize.py         # standalone visualizer (matplotlib | open3d)
├── boustrophedon.py     # core algorithm
├── geometry.py          # plane / pose / rotation utilities
├── robot_check.py       # Phase 1-4: A* IK + slerp densify + jerk refine + vel resample
├── configs/             # YAML configs (config_example.yaml is canonical)
├── objects/             # input meshes (.obj / .glb / etc., loaded via trimesh)
├── outputs/             # generated trajectory JSONs and visualizer PNGs
├── optimize/            # Phase 5 — trajectory-optimization subpackage (in-tree)
│   ├── solver.py            # SCO (IPOPT default, trust-constr fallback)
│   ├── robot_model.py       # ampl_motion FK + SDF
│   ├── retime.py            # TOPP-style pre-retime
│   ├── io.py                # WarmStart + JSON loader/writer
│   ├── densify.py / densify_cli.py
│   ├── visualize.py         # optimizer-specific PNGs
│   └── ampl_motion-0.0.2-py3-none-any.whl
└── .venv/               # Python 3.10 venv (trimesh, open3d, scipy, matplotlib)
```

The Phase 5 optimizer used to live in `sandbox/zyu/trajopt/`; it's been
folded in so a single `python plan.py --config <yaml>` can run the planner
alone, the optimizer alone (set `input.warmstart_json` in the YAML), or
both end-to-end (set `optimize.enabled: true`). See
`configs/config_optimize_only.yaml` for the optimizer-only entry point.

To enable IPOPT inside the SudoDeploy container, install `cyipopt` once
per container:

```bash
apt-get update && apt-get install -y \
    coinor-libipopt-dev pkg-config libblas-dev liblapack-dev
pip install cyipopt
# And the vendored ampl_motion wheel (required for any optimizer run):
pip install --force-reinstall --no-deps \
    tools/surface_cover_planner/optimize/ampl_motion-0.0.2-py3-none-any.whl
```

`cyipopt` is optional — when it isn't installed, the solver silently
falls back to scipy's `trust-constr`.

## Setup

The tool ships with its own Python 3.10 virtual environment because `open3d`
does not yet provide wheels for the SudoDeploy main `.venv` (Python 3.14).

If `.venv/` is missing, recreate it with:

```bash
cd SudoDeploy/tools/surface_cover_planner
uv venv --python python3.10 .venv
uv pip install --python .venv/bin/python trimesh open3d scipy matplotlib numpy pyyaml
```

## Usage

`plan.py` is a single CLI that runs in one of three modes, chosen by the
YAML — no extra flags. Two routing knobs in the config decide:

| Config | Mode | What runs |
|--------|------|-----------|
| `optimize.enabled: false` (or section omitted) | **Planner only** | Boustrophedon raster → Phases 1–4 of `robot_check` (when `robot.enabled: true`) → planner JSON. |
| `optimize.enabled: true`, no `input.warmstart_json` | **End-to-end** | Planner above, then `WarmStart.from_planner_result(...)` hands the result in-memory to `optimize.solve()`. Two output JSONs side by side: `<name>_<ts>.json` + `<name>_<ts>_optimized_<ts>.json`. |
| `input.warmstart_json: <path>` | **Optimizer only** | Planner skipped. `optimize.load_warmstart()` reads the JSON; `optimize.solve()` writes a single optimizer JSON. |

### Mode 1 — Planner only (geometry / IK feasibility)

Geometry-only flow runs in the host `.venv/`. The robot feasibility check
(`robot.enabled: true`) needs the SudoDeploy docker container because
`ampl` + `ampl_motion` are container-only wheels.

```bash
# Host venv — geometry only (robot.enabled must be false in the config).
cd SudoDeploy/tools/surface_cover_planner
.venv/bin/python plan.py --config configs/config_example.yaml

# Docker — robot feasibility check (Phases 1–4 of robot_check).
cd SudoDeploy/docker
docker-compose run --rm --service-ports sudodeploy_full_dev bash
# inside the container:
cd /home/deployuser/NewDeploy/SudoDeploy/tools/surface_cover_planner
python plan.py --config configs/config_example.yaml
python plan.py --config configs/config_infeasible_search.yaml
```

Output (planner JSON):

- `metadata.robot_check`: `{ok, n_q_waypoints, total_dist, dt,
  worst_joint_speed_ratio, worst_dof, joint_speed_ok, failures[]}`.
- Top-level `q_trajectory[]` — one entry per densified slerp step with the
  full 16-DOF state, the desired/achieved TCP pose, and an
  `ok` / `failure_cause` pair. Failures still produce a trailing
  `q_full: null` step so partial trajectories are inspectable.

The process exits non-zero when the check fails; the JSON is written first.

### Mode 2 — End-to-end (planner + optimizer in one process)

Set `optimize.enabled: true` in your existing planner config. Requires
`robot.enabled: true` (the optimizer needs a `q_trajectory` to refine).
The planner result is handed to the optimizer in-memory — no temp
warm-start JSON is written between phases.

```bash
# Inside the container.
cd /home/deployuser/NewDeploy/SudoDeploy/tools/surface_cover_planner

# One-time per fresh container:
pip install --force-reinstall --no-deps \
    optimize/ampl_motion-0.0.2-py3-none-any.whl
# Optional (recommended) — IPOPT solver. Falls back to trust-constr if absent.
apt-get update && apt-get install -y \
    coinor-libipopt-dev pkg-config libblas-dev liblapack-dev
pip install cyipopt

# Canonical end-to-end recipe — kshortest_dp search + IPOPT optimizer:
python plan.py --config configs/config_complete.yaml
# Or use any planner config with `optimize.enabled: true` flipped on:
python plan.py --config configs/config_example.yaml
# Writes (both modes):
#   outputs/<stem>_<ts>.json                       # planner output + q_trajectory
#   outputs/<stem>_<ts>_optimized_<ts>.json        # optimizer output (solve_stats + refined q_arm/dt)
#   outputs/<stem>_<ts>_optimized_<ts>_{iter,qtraj,cart,dist}.png
```

`configs/config_complete.yaml` is the canonical end-to-end recipe: it
merges the planner setup from `config_infeasible_search.yaml` (kshortest_dp
Phase 1 + slerp densification) with the optimizer tuning from
`config_autorun.yaml` (tight trust radius, high `track_pos`, IPOPT, post-
solve densification). Phase 4 resampling is intentionally off so the
optimizer's TOPP pre-retime + per-segment `dt` handle the velocity
scheduling.

### Mode 3 — Optimizer only (refine an existing planner JSON)

Same CLI, different YAML — point `input.warmstart_json` at a previously
generated planner JSON (must contain `q_trajectory` from a
`robot.enabled: true` run). Useful when sweeping solver knobs or
re-running the optimizer on a known-good warm-start.

```bash
# Inside the container.
cd /home/deployuser/NewDeploy/SudoDeploy/tools/surface_cover_planner
python plan.py --config configs/config_optimize_only.yaml
# Or the canonical autorun warm-start at the lab-default decimate stride:
python plan.py --config configs/config_autorun.yaml
```

Optimizer JSON adds `solve_stats` (iter history, final cart/jerk/vel
summaries, `d_safe_used`, `min_pair_distance_after`) and replaces the
planner's `q_trajectory` with the refined `(q_arm, dt)` schedule. See the
`Output JSON schema` section for the field list.

### Visualizing the planner output (standalone)

`visualize.py` reads only the mesh + the planner JSON, so any historical
trajectory can be re-rendered without re-running the planner.

```bash
cd SudoDeploy/tools/surface_cover_planner

# Matplotlib — saves PNG and opens a window.
.venv/bin/python visualize.py \
    --mesh objects/table.glb \
    --trajectory outputs/example_trajectory.json \
    --out outputs/example_trajectory.png

# PNG only (CI-safe, no window).
.venv/bin/python visualize.py \
    --mesh objects/table.glb \
    --trajectory outputs/example_trajectory.json \
    --out outputs/example_trajectory.png --no-show

# Interactive open3d viewer (mouse-rotatable 3D).
.venv/bin/python visualize.py \
    --mesh objects/table.glb \
    --trajectory outputs/example_trajectory.json \
    --interactive
```

Optimizer-mode PNGs (`*_iter.png`, `*_qtraj.png`, `*_cart.png`,
`*_dist.png`) are rendered automatically alongside the optimizer JSON.

## Config schema

See `config_example.yaml` for the full schema with comments. Sections:

- **`mesh`**: `mesh_path`, optional `flip_normals`, and a rigid `transform`
  (`translation` + rotation as `quat` / `euler_xyz_deg` / `matrix`, plus
  uniform `scale`). The transform is applied to the mesh BEFORE any geometry
  work; the polygon is specified in the post-transform world frame.
- **`region.polygon`**: list of >=3 coplanar 3D points in world frame. The
  plane is taken directly from `polygon[0..2]` — no SVD fit, points are
  assumed coplanar.
- **`pattern`**: `step_over`, `point_spacing`, `sweep_axis` (`u` or `v` in the
  plane's local 2D frame; `u` is the first polygon edge), `start_corner`,
  `serpentine`.
- **`robot`** (optional, default `enabled: false`): feasibility-check
  parameters used when running inside docker. See
  `config_example.yaml` for the full schema. Notable fields:
  - `arm`: `left_arm` | `right_arm` — only this arm's 7 DOFs are planned.
  - `initial_q`: REQUIRED 16-DOF starting state
    `[rail, waist, left_arm(7), right_arm(7)]` (no implicit default).
  - `wall.{x,y,z}`: AABB passed to `hbmp_agent.set_wall()`.
  - `joint_vel_limits`: optional length-7 arm-DOF speed cap (rad/s); default
    `[1.0]*7`.
  - `yaw_search_offsets_deg`: optional list of yaw offsets (deg) around
    tool Z (= -surface normal) used by step 3 of the greedy walk (see
    "Algorithm"). The 0.0 (identity) entry is dropped automatically.
    Default `[-1.0, 0.0, 1.0]` (2 yaw candidates per failed waypoint).
  - `pitch_search_offsets_deg`: optional list of pitch offsets (deg) around
    tool Y. Cartesian-product with the yaw and roll lists; default `[0]`.
  - `roll_search_offsets_deg`: optional list of roll offsets (deg) around
    tool X. Cartesian-product with the yaw and pitch lists; default `[0]`
    (legacy behaviour). Widen for tools whose contact is roll-tolerant.
- **`tcp`**:
  - `rotation_mode: align_z_to_normal` — tool Z aligns with the inverted
    outward surface normal (Z points INTO the surface). Yaw around tool Z is
    set by `yaw_deg`, measured from `yaw_reference_axis` projected onto the
    plane perpendicular to tool Z. The `robot_check` may overwrite each
    waypoint's quaternion with a small perturbation when the canonical
    pose is in collision (see "Algorithm" below).
  - `rotation_mode: fixed_quat` — every waypoint uses `fixed_quat` (`[w,x,y,z]`).
  - `standoff` — signed scalar; positive moves the TCP away from the surface
    along the outward normal.
  - `tool_offset` (optional) — rigid transform from the gripper TCP frame
    (`FRAME_TACTILE_L/R`) to a user-defined tool tip frame. Same field layout
    as `mesh.transform` minus `scale`: `translation` plus rotation as `quat` /
    `euler_xyz_deg` / `matrix`. Semantics: waypoint poses are interpreted as
    *tool tip* poses; `robot_check` applies `inv(tool_offset)` when building
    IK targets and applies `tool_offset` when reporting achieved/Cartesian
    deviation. The output JSON's per-waypoint `position`/`quaternion` remain
    tool-tip poses; `metadata.tool_offset_4x4` carries the 4x4 so consumers
    can reconstruct gripper poses if needed. Default = identity.

## Output JSON schema

```json
{
  "metadata": {
    "obj_path": "...",
    "config_path": "...",
    "n_waypoints": 1234,
    "generated_at_iso": "2026-04-29T...",
    "mesh_transform_matrix_4x4": [[...], [...], [...], [0,0,0,1]],
    "tool_offset_4x4": [[...], [...], [...], [0,0,0,1]],
    "plane_origin": [x, y, z],
    "plane_normal": [nx, ny, nz],
    "plane_u_axis": [...],
    "plane_v_axis": [...],
    "polygon_xyz": [[x, y, z], ...]
  },
  "waypoints": [
    {
      "name": "wp_0000",
      "position": [x, y, z],
      "quaternion": [w, x, y, z],
      "tactile_position": [x, y, z],
      "tactile_quat_wxyz": [w, x, y, z],
      "surface_point": [x, y, z],
      "surface_normal": [nx, ny, nz],
      "pass_index": 0,
      "point_in_pass": 0
    }
  ]
}
```

Quaternion convention: `[w, x, y, z]` (sapien / SudoDeploy convention).
Position units: meters.

`position` / `quaternion` are the tool-tip (TCP) pose in world frame.
`tactile_position` / `tactile_quat_wxyz` are the matching pose at the
`<arm>_link_tactile_center` kinematic frame, derived as
`T_world_tactile = T_world_tool @ inv(tool_offset_4x4)`. When
`tool_offset_4x4` is identity the two are equal.

When `robot.enabled: true`, the `q_trajectory` entries gain the same
pairing: `target_position` / `target_quat_wxyz` and
`achieved_position` / `achieved_quat_wxyz` describe the tool tip, and
`target_tactile_position` / `target_tactile_quat_wxyz` /
`achieved_tactile_position` / `achieved_tactile_quat_wxyz` describe the
matching tactile-center poses. The optimizer JSON
(`*_optimized_<ts>.json`) and the densify-CLI output
(`*_dense_<hz>hz.json`) carry the same `*_tactile_*` keys.

When `robot.enabled: true`, the per-waypoint `quaternion` may differ from the
canonical aligned pose by a yaw around tool Z (= -surface normal). The
discrete yaw offsets come from `robot.yaw_search_offsets_deg` (default
`[-1.0, 0.0, 1.0]`). See "Algorithm" below for the full greedy-walk
description.

## Algorithm

The planner is plane-only: the polygon defines the surface. The mesh is used
only by `visualize.py`; `plan.py` does not load it.

1. Build a plane frame directly from the first three polygon vertices:
   `origin = polygon[0]`, `normal = (p1-p0) x (p2-p0)`, `u_axis = p1-p0`,
   `v_axis = normal x u_axis`.
2. Project polygon to local 2D `(u, v)`; rasterize parallel lines along
   `sweep_axis` with `point_spacing` along the line and `step_over` between
   lines; serpentine if requested.
3. Clip line samples to the polygon interior using
   `matplotlib.path.Path.contains_points`.
4. Unproject 2D samples back to 3D on the polygon plane. Each sample becomes a
   waypoint with `surface_point` on the plane and `surface_normal = plane.normal`
   (negated when `mesh.flip_normals: true`).
5. For each waypoint, compute the TCP pose (position offset by `standoff` along
   the outward normal; rotation per `tcp.rotation_mode`).
6. Write `metadata` + `waypoints` to the configured JSON path. `mesh.transform`
   is recorded in `mesh_transform_matrix_4x4` so the visualizer can position
   the visualization mesh consistently.

The visualizer reads only the mesh and the trajectory JSON, so any historical
trajectory can be re-rendered without re-running the planner.

### `robot_check` two-phase trajectory check

When `robot.enabled: true`, `check_trajectory` runs a two-phase pipeline.

**Phase 1 — Weighted A* over per-waypoint IK pools:**

For every waypoint, build a pool of `(q_arm, quat_wxyz)` candidates by
running analytical IK at every `(roll, pitch, yaw)` triple in the
cartesian product of `robot.roll_search_offsets_deg` (rotation about
tool X, default `[0]`), `robot.pitch_search_offsets_deg` (rotation about
tool Y, default `[0]`), and `robot.yaw_search_offsets_deg` (rotation
about tool Z, default = 18 values at 10° spacing covering one half-
circle, including 0° = the planner-assigned canonical quat). Each triple
perturbs the canonical quat locally — roll around tool X, pitch around
tool Y, yaw around tool Z (= `-surface_normal` in `align_z_to_normal`
mode) — so orientation
tolerance follows the surface frame regardless of the waypoint's base
yaw. Pool entries are deduped by rounding `q_arm` to 1e-3 rad.
Collisions are *not* filtered at pool-build time. Total IK calls per
waypoint scale as `len(yaw) · len(pitch) · nb_redundant_search`, so
widen the pitch axis sparingly.

The A* state is `(layer_index, ik_idx)`. Edge cost between consecutive
layers is the **weighted arm-joint distance** `sqrt(sum(w * Δq²))` with
`w = JOINT_DIST_WEIGHTS = [2, 2, 2, 1, 1, 1, 1]` (joints 1–3 carry a 2×
penalty so the search prefers moving 4–7). Phase 1 only checks
endpoint self-collision per IK candidate; swept collisions between
adjacent IK picks fall through to Phase 2, which slerp-densifies at
`ee_speed / hz` step and runs endpoint collision per sub-step at much
finer resolution. The search uses two speed-ups:

1. **L_inf joint-jump prune.** Edges with `max|q_next - q_cur| >
   robot.search.jump_threshold` (default 1.5 rad) are skipped before
   ever entering the heap, which kills joint-flip transitions in O(1).
2. **Weighted heuristic.** `f = g + W · (layers_remaining) ·
   heuristic_base_cost` where `W = robot.search.heuristic_weight`
   (default 5.0). `W=1` is admissible (optimal but slow); `W>>1` makes
   the search aggressively greedy toward the goal layer. The first node
   popped at the final layer wins (early stop, no global-optimum
   guarantee — a deliberate tradeoff for speed and feasibility recovery
   that the greedy walk cannot do).

If A* exhausts the open list without reaching the final layer, the
failure record names the deepest layer reached and suggests likely
causes (thin pool, tight `jump_threshold`, genuine collision
chokepoint).

**Alternative — iterative-mask DP (`robot.search.search_method:
kshortest_dp`):** the same per-waypoint IK pools, the same
`jump_threshold` hard edge filter, but no collision queries during the
forward DP itself. Phase 1 pre-caches the per-layer edge-cost matrix
once (pool geometry is fixed; only the mask changes), then loops up to
`max_iterations` times: a **single-best forward DP** using the cached
edges + a persistent per-`(layer, ik_idx)` boolean mask, reconstruct the
shortest path, then run `_endpoint_collides` on each node of the path.
Endpoint failures mask the failing node. The loop stops early on the
first clean path, fails with `kind="ik"` if the DP becomes infeasible
(the mask removed every parent at some layer), or fails with
`kind="collision"` when `max_iterations` is reached. Each iteration is
~one `argmin` per layer over a small numpy matrix — orders of magnitude
faster than the K-best variant it replaced. The strategy is selected
via `robot.search.search_method` (`astar` | `kshortest_dp`); the
heuristic knobs are ignored under `kshortest_dp`.

**Phase 2 — slerp densification:**

After every waypoint has a committed `(q_arm, quat)`, walk consecutive
selected poses, slerp-densify at step `= ee_speed / hz`, and re-IK each
sub-step with a continuity-preferred selector (sort by
`||q_arm - cur_arm||` ascending, early-exit on first collision-free).
Cartesian deviation is checked at every sub-step; arm joint velocities are
checked on the full densified trajectory against `joint_vel_limits`.
