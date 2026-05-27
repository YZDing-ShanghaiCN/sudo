# CLAUDE.md — surface_cover_planner

> **Maintenance:** keep this file current when you change the algorithm,
> config schema, output JSON shape, or pipeline wiring. The README has the
> full schema/usage; this file is the *strategic* picture for fast onboarding.

## Purpose

Boustrophedon (back-and-forth) coverage planner for a polygonal region on a
flat surface. Produces an ordered list of TCP poses in world frame. When
`robot.enabled: true`, also runs a robot feasibility check (analytical IK +
collisions + speed limits) and writes the resulting joint trajectory.

When `optimize.enabled: true` (Phase 5), the planner's q-trajectory is
handed in-memory to the `optimize/` subpackage — a sequential convex
optimization over `(q_arm, dt)` that minimizes total time + joint jerk
while honoring cartesian tracking + joint limits + self-collision as
**hard** constraints. The same CLI also runs the optimizer standalone
when `input.warmstart_json` is set in the YAML (the planner phases are
then skipped).

## Stack

- Python 3.10 inside `.venv/` (host venv) for geometry-only flows.
- Docker (SudoDeploy `sudodeploy_full_dev`) required for `robot.enabled:
  true` — depends on `ampl-0.0.32` (host `whls/`) and
  `ampl_motion-0.0.2-py3-none-any.whl` (vendored under
  `optimize/` in this tool). **No RoboSkill / wbc_py / `Robot_T2DA2`
  dependency**. The planner was fully swapped onto `ampl_motion.KinHB11`
  for IK / FK + `ampl_motion.colcvx` for self-collision and workspace
  checks. The standard SudoDeploy compose bootstrap installs `ampl`;
  one extra step per fresh container installs the vendored wheel:
  `pip install --force-reinstall --no-deps
  tools/surface_cover_planner/optimize/ampl_motion-0.0.2-py3-none-any.whl`.
- Deps: numpy, scipy, matplotlib, pyyaml, trimesh, open3d. The Phase 5
  optimizer additionally benefits from `cyipopt` (IPOPT solver); when it
  isn't installed, the optimizer silently falls back to scipy's
  `trust-constr`.

**Kinematics + collision backend** (since the full swap):

- **IK / FK** — `ampl.KinHB11` built once at `check_trajectory` startup
  via `_setup_collision_checker(cfg)` and aliased to the local `agent`
  variable. `kin.get_ik(tf, frame, nb_redundant_search=N)` returns
  `dict[which_ik, (status_batch, q_batch)]` — same shape the prior
  `Robot_T2DA2.get_ik` did, just with default `which_iks=[4, 5, 6, 7]`
  (vs HBMP's `[5, 7]`). `kin.get_fk(frame).matrix` is identical.
- **Self-collision + workspace** — `ampl_motion.colcvx` (convex mesh
  check via `ampl.collision_check_cvx_cvx`) routed through
  `_is_self_colliding(q_full)` (the single entry point for every binary
  collision query in this file). Workspace bounds go to
  `colcvx.configure_workspace(...)` at setup, restricted to `left-*` /
  `right-*` link names (matches HBMP's arm-only `set_wall` behavior).
  Pair ignores come from `ampl_motion.create_default_ignore_pairs("hb11",
  ...)`.
- **Diagnostic wall printout** — `_collision_checker["wall"]` holds a
  `(x_bounds, y_bounds, z_bounds)` triple copied from `cfg.wall` so
  `_diag_cause` can render its failure message without needing
  `agent.wall()`.
- **Attached tool + world obstacles** (`robot.attached_tool` /
  `robot.world_obstacles` YAML blocks). Optional convex meshes added to
  the same `colcvx` scene at the bottom of `_setup_collision_checker`:
  - **Multi-shard OBJ inputs.** `_load_convex_hulls(path)` preserves
    every top-level `o hull_N` / `g` entry in a multi-object `.obj` as
    its own VSCHf. The earlier `force="mesh"` loader concatenated the
    Scene into one Trimesh and trimesh's vertex-merge pass dedupe'd
    shared boundary verts, collapsing the decomposition into one
    convex region (a hollow cage became a solid box that swallowed the
    workspace). Each sub-hull is registered as its own colcvx entity
    named `f"{entry_name}/{sub_name}"` (or just `entry_name` when the
    OBJ is a plain single-mesh).
  - `attached_tool` — `_collision_checker["tool_entity"]` holds a list
    of sub-entity names. Each sub-hull pose is recomputed inside
    `_push_scene_poses(q_full)` per query as
    `T_world_gripper @ local_offset`
    (`kin.get_fk(tactile_center).matrix @ local_4x4`) and pushed via
    `colcvx.update_pose(entity, Tf(...).as_array())` for every entity
    in the list. `ignore_links` (auto-default = `arm_5/6/7 +
    hand/tactile` of the active arm) is applied to every sub-entity
    via `colcvx.disable_collision`.
  - `world_obstacles` — list of `{name, path, translation,
    rotation_mode, quat|euler_xyz_deg|matrix}` entries. Multi-shard
    OBJ entries fan out into `f"{name}/{sub_name}"` colcvx entities
    sharing the entry's pose. Pose is set once at `colcvx.add(...)`
    and never updated.
  Both flow through the existing
  `process_collisions() → collision_check_cvx_cvx < magic` loop, so
  every Phase 1-4 collision query (`_is_self_colliding`,
  `_self_collision_pairs`) automatically picks them up — no per-phase
  plumbing. **The Phase-5 optimizer (`optimize/`) does NOT see these
  meshes**: `solve_local_dp` and `optimize.solve` rebuild the
  collision checker with a wall-only `RobotCheckConfig` and use
  `ampl_motion`'s sphere-BVH for its hard SDF constraint. The
  optimizer can therefore produce trajectories that collide with the
  attached tool / world obstacles; if you need parity, also plumb
  `attached_tool` / `world_obstacles` through `RobotConfig` in
  `optimize/robot_model.py` and add sphere proxies to the SDF
  constraint.

This makes the planner and the `optimize/` subpackage's
`solve_stats.n_collisions_after` audit return the same verdict on the
same `q_full` — prior to the swap they disagreed by ~40 collisions per
autorun warmstart because HBMP's internal sphere model and
`ampl_motion`'s convex-mesh model used different ignore-pair lists.

## Layout

```
surface_cover_planner/
├── plan.py            # entrypoint: YAML → JSON (+ optional robot_check + optimize + PNGs)
├── boustrophedon.py   # raster generation, corner rounding, pose synthesis
├── geometry.py        # plane / pose / rotation utilities
├── robot_check.py     # Phase 1 A* over IK pools + Phase 2 SLERP+IK densification (legacy slerp / TOPP-smoothness) + Phase 3 jerk-region refinement + Phase 4 joint-velocity resampling
├── topp.py            # analytic forward-backward TOPP (Bobrow/Pham) + uniform-dt resampler used by Phase 2's topp_smooth method
├── visualize.py       # standalone (u,v) PNG render + 3D matplotlib + open3d
├── configs/           # YAML configs (config_example.yaml is canonical)
├── objects/           # input meshes
├── outputs/           # generated JSONs + PNGs (timestamp-suffixed)
└── optimize/          # Phase 5 — trajectory optimizer (was sandbox/zyu/trajopt)
    ├── __init__.py        # public API: solve, WarmStart, SolverOpts, ...
    ├── solver.py          # SCO solver (cost / hard constraints / IPOPT or trust-constr)
    ├── robot_model.py     # ampl_motion KinHB11 + SphereBVHScene wrapper (FK, J_pos, SDF)
    ├── retime.py          # TOPP-style pre-retime of warm-start
    ├── io.py              # WarmStart bundle + JSON loader / writer (in-memory & disk paths)
    ├── densify.py         # post-solve joint-space densification + cart projection
    ├── densify_cli.py     # standalone CLI for densifying an existing optimizer JSON
    ├── visualize.py       # iter-history / qtraj / cart-deviation / min-dist PNGs
    └── ampl_motion-0.0.2-py3-none-any.whl  # vendored sphere-BVH SDF + grads
```

## Pipeline (data flow)

```
YAML config
   │
   ▼
plan.plan_boustrophedon()
   ├─ plane frame from polygon[0..2]               (geometry.plane_from_polygon)
   ├─ raster grid in (u,v), polygon-clip            (boustrophedon._generate_uv_grid)
   ├─ corner rounding (serpentine 180° → arc)       (boustrophedon._round_uv_corners)
   ├─ unproject (u,v) → world xyz                   (geometry.unproject_from_plane_uv)
   └─ build TCP poses                               (geometry.build_tcp_pose)
   │   → list[Waypoint]  with kind ∈ {"row", "turn"}
   ▼
robot_check.check_trajectory()  (only if enabled)
   ├─ Phase 1: build per-wp IK pool, weighted A*       → one (q_arm, quat) per user wp
   ├─ Phase 2: SLERP + continuity-IK densification → optional TOPP retime + uniform-1/hz resample → q_trajectory list[QPoint]
   │            (method ∈ {topp_smooth (default), slerp})
   ├─ Phase 3: jerk-region refinement                  → q_trajectory mutated in place
   │    (only if cfg.refine_enabled — detect |Δq_arm|∞ > refine.jump_threshold runs,
   │     per-step greedy IK over yaw/pitch/xy-relaxed pool with continuity preference)
   └─ Phase 4: joint-velocity resampling                → q_trajectory mutated in place
        (only if cfg.resample_enabled — splice linearly-interpolated joint-space
         samples into pairs that violate joint_vel_limits at control rate 1/dt;
         t_global re-stamped to i*dt — semantic shift from path-time to control-time)
   │
   ▼
plan.write_trajectory_json()      → outputs/<name>_<ts>.json
plan → visualize.render_png()     → outputs/<name>_<ts>.png            (always)
plan → visualize.render_joint_jump_png() → outputs/<name>_<ts>_joint_jump.png  (only if robot_check ran)
   │
   ▼  (only if `optimize.enabled: true` AND robot_check ran)
optimize.WarmStart.from_planner_result(check_result, ...)  # in-memory; no
                                                            # intermediate JSON
optimize.retime.topp_retime(...)         # vel-feasible at iter 0
optimize.solve(...)                       # IPOPT (default) | trust-constr
optimize.write_optimizer_json(...)       → outputs/<name>_<ts>_optimized_<ts>.json
optimize.densify_joint_space(...)        → outputs/<name>_<ts>_optimized_<ts>_densified.json
                                            (only if `output.densify_dt` is set;
                                             re-evaluates FK + collision per dense step)
optimize.visualize.render_*_png(...)     → iter / qtraj / cart / dist PNGs
```

Optimizer-only invocation: same CLI, set `input.warmstart_json` in the
YAML and skip everything above `optimize.solve()`. The WarmStart bundle
is then built via `optimize.load_warmstart(json_path)`.

## Key data structures

- `boustrophedon.Waypoint` — `name`, `position` (3,), `quaternion_wxyz` (4,),
  `surface_point`, `surface_normal`, `pass_index`, `point_in_pass`, `kind`
  (`"row"` or `"turn"`).
- `boustrophedon.PlanResult` — `waypoints`, `plane`, `polygon_xyz`,
  `mesh_transform_4x4`, `tool_offset_4x4`.
- `robot_check.QPoint` — per densified slerp step. Has `target_position`,
  `target_quat_wxyz`, `q_full` (16-DOF), `q_left_arm` / `q_right_arm`, etc.
- `robot_check.CheckResult` — `q_trajectory: list[QPoint]`, `failures`,
  `worst_joint_speed_ratio`, optionally `ik_pools`, optionally
  `refine_stats` (Phase 3 summary: counts + max|Δq|∞ before/after),
  optionally `resample_stats` (Phase 4 summary: pair counts, insert
  count, ratios pre/post, expansion factor).

## Strategies (the things that shape the code)

### Boustrophedon raster
- `sweep_axis` rows, stacked at `step_over` apart.
- Polygon clip via `matplotlib.path.Path.contains_points` — boundary
  inclusion is implementation-dependent, so end-points may shift by
  `point_spacing`. Don't depend on exact polygon-edge alignment.
- A single line that crosses the polygon multiple times splits into
  multiple passes (`pass_index` increments).

### Corner rounding (serpentine 180° → tangent-continuous arc)
- Detection (`_is_turn_pair`): consecutive passes whose end/start tangents
  are anti-parallel (cosθ < −0.9), displacement magnitude ≈ `step_over`,
  and displacement perpendicular to the tangent (rejects polygon-induced
  splits whose displacement is *along* the line).
- Geometry: half-circle of radius **derived per-pair from the actual chord
  between trimmed-row endpoints** (`chord_len / 2`). The raster's
  `np.linspace(stack_min, stack_max, n_lines)` divides the polygon range
  evenly across `n_lines - 1` intervals, which only matches `step_over`
  when the range divides evenly — for off-by-mm polygons the actual row
  spacing differs, and a hard-coded `radius = step_over/2` used to make
  `_half_circle_arc` fall back to a straight chord (visible as sharp
  ~90° corners at row ends). Per-pair derivation fits the arc to whatever
  spacing landed, so the chord-equality check never trips. `trim` controls
  how far back each row is cut; `trim ≥ step_over/2` keeps the arc inside
  the polygon (peak just touches the edge when `trim = step_over/2`).
  `pattern.turn.radius` is therefore advisory — logged if set but never
  overrides the geometric value.
- Sampling: `num_points` (default 8) — exact count of interior arc
  samples, evenly spaced at angles in (0, π); overrides `arc_spacing`
  (default `point_spacing/2`) when set. Arc samples carry `kind="turn"`
  and live in their own `pass_index` (rows = 0,2,4…, arcs = 1,3,5…) so
  downstream `pass_idx.max()+1` arithmetic still works.
- Fallbacks (each prints a one-line warning, uses sharp turn for that
  pair): row would shrink below `arc_spacing` after trim; arc samples
  fall outside polygon (concavity); `_is_turn_pair` rejects.
- Iterative disable loop converges in O(n²) over passes — fine for
  realistic n.

### TCP pose synthesis
- `align_z_to_normal`: tool Z = `-surface_normal_outward` (Z points INTO
  the surface). Yaw around tool Z is set by `tcp.yaw_deg` from
  `tcp.yaw_reference_axis` projected into the plane.
- `fixed_quat`: every waypoint uses the same quat verbatim.
- `tool_offset` (optional 4×4) — waypoints describe the *tool tip* pose;
  `robot_check` applies `inv(tool_offset)` when building IK targets.

### Phase 1 IK selection (`robot_check`)
- Pool: cartesian product of `roll_search_offsets_deg` ×
  `pitch_search_offsets_deg` × `yaw_search_offsets_deg` × redundant IK
  samples. Each axis perturbs the canonical TCP quat by an intrinsic
  XYZ-Euler rotation in the tool frame (roll about tool X, pitch about
  tool Y, yaw about tool Z); defaults are `[0]` for roll/pitch and 18
  values at 10° spacing for yaw. Dedup at 1e-3 rad.
- **Phase 1a sweeps every sparse waypoint unconditionally.** Both search
  strategies build the IK pool for all N user waypoints before any search
  runs — no per-waypoint early-quit on the first empty pool. Empty pools
  are recorded as `kind="ik"` failures, and after the full sweep a single
  post-sweep bail returns `selected_q_arms=None` if any pool came up
  empty (skipping Phase 1b before A*'s open-list-empty surface or the
  DP's `np.stack` on an empty pool). Net effect: `ik_pool_statuses`
  always covers all N waypoints, so `*_ik_grid.png` and the main PNG's
  pool-size colorbar always show the complete IK landscape — exactly
  what the user needs when diagnosing infeasibility.
- **Endpoint-only collision in Phase 1.** Both search strategies check
  `_endpoint_collides` per IK candidate they visit; **no swept-collision
  check on edges**. The historical lazy `_edge_collides` interpolation
  was dropped — its job (catching joint-space sweeps between adjacent
  IK picks that endpoint-pass individually) falls through to Phase 2,
  which slerp-densifies at `ee_speed / hz` step and runs
  `_endpoint_collides` per sub-step at much finer resolution. The
  failure mode is later (Phase 2 reports `kind="collision"`) rather
  than self-healing within Phase 1. Both backends also cache every
  `_endpoint_collides` verdict in an `endpoint_cache: dict[(L, j), bool]`
  that's returned through `_build_ik_pool_collision_checked` and
  persisted in the sidecar — feeds the per-waypoint collision-free count
  in `_collision_free.png`. astar only seeds-checks pool[0] (no
  per-expansion check), so its cache is sparse; kshortest_dp fills its
  cache across iterations.
- **Per-joint preference filter** (`robot.search.joint_preference_filters`,
  default null). Applied after `_build_ik_pool` returns and before the
  Phase 1a empty-pool bail; drops every IK-converged, deduped pool entry
  whose `q_arm[j_idx]` falls outside `[lo, hi]` for any configured
  `j_idx ∈ {0..6}`. Either bound may be null for one-sided filtering
  (e.g. `{1: [0.0, null]}` drops every candidate with `q_arm[1] < 0`).
  The IK-grid PNG (`*_ik_grid.png`) is unaffected — it still reflects
  raw IK convergence, since the filter is logically a Phase-1 preference,
  not an IK outcome. Pools emptied by the filter surface as the existing
  `kind="ik"` failure with the cause string extended to report how many
  candidates the filter dropped. The flat-index pairing
  (`pool_flat_indices`) is filtered in lockstep so the IK-grid PNG's red
  "Phase 1 path" overlay still points at the correct rows. Phase 3's
  `_build_refine_pool` does NOT consult this filter — relax via
  `refine_yaw_offsets_deg` / `refine_xy_offsets_m` instead.
- Two interchangeable search strategies, picked by `cfg.search_method`
  (`robot.search.search_method` in YAML):
  - **`astar` (default)** — weighted A* over (layer, ik_idx). Edge cost =
    weighted arm-joint distance with `JOINT_DIST_WEIGHTS = [2,2,2,1,1,1,1]`
    (penalize joints 1–3). **L∞ jump prune** at `jump_threshold` rad.
    Greedy with `heuristic_weight` (default 5).
  - **`kshortest_dp`** — iterative-mask DP. Pre-caches the per-layer
    edge-cost matrix once (the `(P[L], P[L-1])` weighted-L2 + jump-mask
    matrix; pool geometry doesn't change across iterations, only the
    mask does), then loops up to `max_iterations` times: forward
    **single-best** DP using the cached edges + a persistent per-`(L, j)`
    boolean mask → reconstruct the shortest path → run
    `_endpoint_collides` on each node of the path. **Endpoint failure**
    at `(L, j)` masks that node. The loop stops early when the path is
    clean, fails with `kind="ik"` if the DP becomes infeasible (mask
    removed every parent at some layer), or fails with
    `kind="collision"` when `max_iterations` is reached.
    Ignores `heuristic_weight` / `heuristic_base_cost`.
    **Acceleration cost (`robot.search.accel_weight`, default 0.0)** —
    when non-zero, the per-edge cost gains a second-order term
    `accel_weight · weighted_L2(q_next − 2·q_par + q_grandparent)` (same
    `JOINT_DIST_WEIGHTS` reused so dist + accel share units). The
    grandparent is looked up lazily as `parent[L-1][i]` (the previous
    layer's already-computed argmin), so the DP state stays `(L, j)` and
    per-layer work stays O(P²); the cost is **not** globally optimal
    under this approximation but penalises IK-branch reversals the
    dist-only cost ignores. `0.0` is byte-equivalent to the dist-only
    code path (the accel block is gated off). A* ignores this knob.
    **Endpoint collision cache** — verdicts keyed by `(L, j)` persist
    across iterations. Pool geometry is fixed, so the same node gives
    the same verdict on every visit; without the cache, any node that
    previously passed would be re-checked whenever it reappeared in
    the reconstructed path. The cache is also returned out of Phase 1
    (via `_build_ik_pool_collision_checked`) and persisted in the
    sidecar as `ik_pool_collision_checked` to feed
    `_collision_free.png` — no extra collision sweep runs. The per-iter progress suffix prints
    `new_checks(ep)=… total_new(ep)=…` so cache pressure (how much
    wasted work the cache is avoiding) is visible from the console.
    **Per-iter timings** — `t(dp/check)=A/B s total(dp/check)=A'/B' s`
    is also printed so the DP forward pass (`A`: cached edge-cost
    matmul + reconstruction) can be compared against the collision
    phase (`B`: cache lookups + cache-miss `_endpoint_collides` calls).
    In practice `t_check` dominates whenever cache pressure is low
    (early iterations) and shrinks as the cache fills.
- Both strategies preserve the same Phase 1 output contract: one
  `(q_arm, quat)` per waypoint plus `ik_selected_flat_indices` for the
  IK-grid PNG overlay; downstream Phase 2 / 3 / 4 don't care which ran.
- **`robot.search.keep_init` (default false).** When true, Phase 1
  anchors the chain at `initial_q[arm_indices]`: layer-0 candidates
  outside `jump_threshold` (L∞) of the initial arm pose are pruned,
  the surviving candidates seed the search with cost = weighted-L2
  distance from the anchor (so `q_arm[0]` becomes the cheapest
  reachable IK rather than the lowest-chain-cost IK), and an empty
  post-prune layer-0 pool raises `RuntimeError` — a precondition
  violation, not a soft `kind="ik"` failure, since the deployment
  contract ("initial pose is close to wp_0") was wrong and silent
  recovery would mislead. Both A* and kshortest_dp honour the flag
  with identical semantics (L∞ prune + weighted-L2 seed). Phase 2
  still densifies starting from `selected_q_arms[0]` (the
  initial→wp_0 hop is the caller's responsibility); the threshold
  check is what justifies skipping that densification.

### Phase 2 densification
- **Geometric pass (both methods).** Slerp at `ee_speed / hz` step in TCP
  space (using `Tf.slerp()`); per sub-step run continuity-preferred IK
  (sort by `||q_arm − cur||`, early-exit on first collision-free) — see
  [feedback_ik_selection.md](../../../.claude/projects/-home-yuzeren-sudo-ws-SudoDeploy/memory/feedback_ik_selection.md).
- **Two methods**, dispatched on `cfg.densify_method`
  (`robot.densify.method` in YAML):
  - **`topp_smooth` (default)** — three passes implemented in
    `_phase2_topp_smooth_densify`. Pass 1 runs the geometric SLERP + IK
    loop and **accumulates the fine-grid joint path without QPoint
    emission**. Pass 2 calls `topp.topp_smooth(q_arm, v_max, a_max,
    dt_min, vel_headroom, accel_headroom)` — a classical analytic
    forward-backward TOPP (Bobrow / Pham) over `β = (ds/dt)²` with
    `Δs = 1` per segment, **extended with a corner-acceleration cap**
    (interior samples k cap `β ≤ a_eff[j] / |Δq[k] − Δq[k-1]|` per DOF,
    propagated symmetrically to neighbours k-1 / k+1 so the locally-
    constant-β approximation stays valid) and an **iterative tightening
    pass** (up to 32 forward+backward cycles; each iteration measures
    the discrete segment-pair accel ratio and shrinks `β_max` at
    offending samples by 1/ratio²; converges in 2–4 iterations on
    typical paths). Returns per-segment `dt` satisfying
    `|Δq/dt| ≤ v_max·(1−h_v)` and the discrete corner-accel
    `|(Δq[k+1]/dt[k+1] − Δq[k]/dt[k]) / dt_avg| ≤ a_max·(1−h_a)` per
    DOF — both within ~1% of cap on real piecewise-linear paths. Pass 3 calls `topp.resample_to_uniform_grid(dt, 1/hz)`,
    linearly-interpolates arm DOFs onto the dense grid (non-arm DOFs
    are constant along the Phase 2 path so they pass through),
    slerp-interpolates target quats, and re-FKs each dense sample for
    cartesian-deviation + self-collision verification. QPoints carry
    **control-time `t_global = k / hz` directly** — no path-time /
    control-time semantic flip is needed downstream. Vel + accel
    feasible by construction, so Phase 4 collapses to a no-op when
    `resample.joint_vel_limits == joint_vel_limits` and Phase 5's
    `topp_retime` collapses to identity on most segments. IK failures
    in Pass 1 short-circuit with the same `kind="ik"` failure +
    sentinel QPoint as the legacy path; the TOPP pipeline is skipped
    and `metadata.robot_check.densify.failed = true`.
    Required knob: `robot.densify.joint_accel_limits` (length-7 or
    scalar rad/s²). Optional knobs: `vel_headroom` (default 0.05),
    `accel_headroom` (default 0.05), `dt_min` (default 0.005 s).
  - **`slerp` (legacy)** — single-pass: per sub-step emit a QPoint with
    `t_global` accumulated as path-time (`Σ seg_dist / N`). Joint
    velocity tracks the Jacobian, joint acceleration is **unbounded**
    at user-waypoint boundaries (instantaneous velocity flips). Phase 4
    later restamps `t_global = i · dt` to recover control-time
    semantics. Kept around for back-compat / A/B debugging.
- Cartesian deviation checked per sub-step (both methods) against
  `max_cartesian_deviation`. Arm velocities checked **after Phase 3 / 4**
  (so the velocity check runs on the post-resample trajectory) against
  `joint_vel_limits`.
- **Stats** in `CheckResult.densify_stats` →
  `metadata.robot_check.densify` of the JSON (topp_smooth only): `method`,
  `failed`, `total_dist`, `n_fine_samples`, `n_dense_samples`,
  `topp_total_time`, `dt_min_used`, `vel_headroom`, `accel_headroom`,
  `max_vel_ratio`, `max_accel_ratio`, `n_collision_failures`,
  `n_dev_failures`, `joint_vel_limits`, `joint_accel_limits`.

### Phase 3 jerk-region refinement (`cfg.refine_enabled`)
- **Detection**: `_detect_jerk_regions` flags destination indices `i`
  where `max|q_arm[i] − q_arm[i−1]| > refine_jump_threshold` (default
  0.1 rad), coalesces consecutive flags into runs, pads each run by
  `refine_region_buffer` (default 2) on each side, and merges
  overlapping pads. Anchor steps (run.lo / run.hi) keep their Phase 2
  q_arm; interior steps are refinable.
- **Relaxed pool per interior step** (`_build_refine_pool`): cartesian
  product of yaw × pitch × xy offsets perturbs the slerp-canonical
  target pose. Defaults: yaw / pitch ∈ ±5° at 1° spacing (11 each), xy
  in **PlaneFrame (u, v)** ∈ ±1 cm at 0.2 cm spacing (11×11=121),
  `nb_redundant_search=8`. Dedup at 1e-3 rad. Collision deferred to the
  picker.
- **Per-step greedy pick** (`_pick_best_from_refine_pool`): walk the
  region's interior steps left-to-right; at each step, sort the relaxed
  pool by weighted L2 |Δq_arm| from the previous step's q_arm
  (`JOINT_DIST_WEIGHTS` reused) and return the first collision-free
  candidate (sort then early-exit, per the IK-selection convention
  in [feedback_ik_selection.md]). Replace the `q_trajectory` entry
  in place.
- **Per-step fallback**: when the pool is empty or every entry
  self-collides, keep the Phase 2 entry for that step and continue.
  Phase 3 never makes the trajectory worse.
- **Stats** in `CheckResult.refine_stats` →
  `metadata.robot_check.refine` of the JSON:
  `n_jerk_regions`, `n_steps_refined`, `n_steps_kept`,
  `max_jump_inf_before` / `_after`, `total_ik_calls`.

### Phase 4 joint-velocity resampling (`cfg.resample_enabled`)
- **Goal**: enforce per-joint velocity caps at control rate `1/dt = cfg.hz`
  by **slowing the trajectory** through violating intervals — splice
  linearly-interpolated joint-space samples in, don't re-plan. Runs after
  Phase 3 and before the existing velocity check.
- **Two limit knobs**: `resample.joint_vel_limits` (rad/s, scalar or
  length-7) is the resampler target; top-level `joint_vel_limits` is the
  velocity-check limit. When `resample.joint_vel_limits` is null it falls
  back to the top-level value (in which case the velocity check becomes a
  free post-condition assertion). Set the resample limit tighter than the
  velocity-check limit to slow the executed trajectory below what the
  hardware actually allows — e.g. resample at 0.1 rad/s, check at 1.0.
- **Per pair (q_trajectory[i], q_trajectory[i+1])**: compute
  `ratio_k = |Δq_full[arm_indices[k]]| / (dt * joint_vel_limits[k])` for
  every arm DOF k with positive limit; `worst = max(ratios)`. If
  `worst ≤ 1`, pass through. Else `n_sub = ceil(worst)` and emit
  `n_sub - 1` interior QPoints at fractions `t ∈ {1/n_sub, …, (n_sub-1)/n_sub}`.
- **Inserted QPoint fields**: `q_full = (1-t)*q_a + t*q_b` (linear in
  joint space, per the user's "insert in joint space"); `target_position`
  lerped, `target_quat_wxyz` slerped via `Tf.slerp` (Phase 2's path);
  `achieved_position` / `achieved_quat_wxyz` come from **FK on q_full**
  via `_gripper_fk_to_tool_position` / `_gripper_fk_to_tool_quat_wxyz`,
  so `cartesian_deviation = ||achieved - target||` reflects real
  tracking error (not a cosmetic lerp).
- **Optional per-insert collision check** (`resample.check_collision`,
  default false): runs `_endpoint_collides`; on collision logs
  `kind="resample_collision"` to `failures` but keeps the point. Default
  off because at 0.1 rad/s × dt steps are tiny; enable when refining
  trajectories with large unrefined Phase 1 jumps.
- **Endpoint guard**: if either end of a pair has `q_full is None` (a
  Phase 2 IK failure), the pair is passed through unchanged and a
  `kind="resample_skipped"` failure is appended.
- **Safety cap** (`resample.max_expansion`, default 1000): if the
  projected post-resample length exceeds `max_expansion * len_before`,
  abort and emit `kind="resample_overflow"`. Protects against a
  misconfigured `joint_vel_limits` (e.g. 0.001 rad/s) blowing up to
  millions of samples.
- **Re-stamp after splice**: every entry's `t_global = i * dt` and
  `sub_index` is reset sequentially within each `user_wp_index` group.
  Semantic shift: pre-Phase-4 `t_global` was path-time
  (`Σ seg_dist / ee_speed`); post-Phase-4 it is control-rate time —
  which is what the velocity check at `np.diff(positions) / dt` already
  implicitly assumed.
- **Phase 4 never makes the trajectory worse**: under-cap pairs are
  untouched; failed pairs (no q_full / overflow) are passed through.
  The post-resample velocity check at the end of `check_trajectory` is
  the regression alarm.
- **Stats** in `CheckResult.resample_stats` →
  `metadata.robot_check.resample` of the JSON: `n_pairs_processed`,
  `n_pairs_split`, `n_inserted`, `n_pairs_skipped_no_q`,
  `n_collision_inserts`, `max_ratio_before` / `_after`,
  `len_before` / `_after`, `expansion_factor`, `joint_vel_limits`
  (the actual limits used — handy when `resample.joint_vel_limits`
  fell back to the top-level).

### Phase 5 — `optimize/` subpackage (post-process SCO)

> The full strategic notes for the optimizer used to live in
> `sandbox/zyu/trajopt/CLAUDE.md`; key bits are summarized here. The
> module is now an in-tree subpackage of this tool, not a separate sandbox.

- **Constraints policy.** `robot.joint_vel_limits`, `optimize.bounds.max_xy_deviation`,
  `optimize.bounds.max_z_deviation`, `robot.q_lo_arm` / `q_hi_arm`,
  `optimize.collision.d_safe`, and endpoint pinning are **hard** —
  if the solver stalls, fix the algorithm, not the constraints. Loosening
  a hard bound is a spec change. `trust_radius`, `d_warn`, weights, and
  `solver.*` are algorithm knobs and may be tuned freely.
- **Solver default = IPOPT.** `cyipopt`'s restoration phase handles the
  "warmstart-feasible / cost-pulls-infeasible" pathology that stalls
  `trust-constr`. When `cyipopt` isn't installed, the solver silently
  drops back to `trust-constr` (a one-line warning is printed). Switch
  with `optimize.solver.method`.
- **Decision vector** `x ∈ R^(7N + N-1)` packs `q.flatten()` then `dt`.
  Always go through `_pack` / `_unpack` in `optimize.solver`.
- **Costs (smooth):** `time = Σ dt`, `jerk = Σ ‖(q[k+1]-2q[k]+q[k-1])/(dt²)‖²`,
  `track_pos = Σ ‖fk_pos - target_pos‖²`, `track_quat = Σ (1 - ⟨fk_quat,
  target_quat⟩²)`. Collision is **not** a cost — it's a hard
  `NonlinearConstraint dist_pair ≥ d_safe` with analytic Jacobian via
  `ampl_motion`'s sphere-BVH SDF (`SphereBVHScene.compute_collisions` +
  `fill_contact_jacobians`).
- **Cart cap is per-axis, not L2.** `|fk_pos.x - target.x|`,
  `|fk_pos.y - target.y|`, `|fk_pos.z - target.z|` each bounded by
  `max_xy_deviation` / `max_z_deviation`. Each axis auto-relaxes
  independently to `1.05 × warmstart_max` if the warmstart violates the
  configured cap. Recorded as `solve_stats.cart_cap_used.{x,y,z}`.
- **TOPP pre-retime.** Before the SCO sees the warm-start,
  `optimize.retime.topp_retime` stretches each `dt[k]` analytically so
  `|Δq[k,j]| / dt[k] ≤ v_max[j]`. The only legitimate response to
  `joint_vel_limits` violation is slower dt (per the constraints policy),
  never a higher cap.
- **`dt_lo` is `dt_min`, not the retimed warm.** The vel LinearConstraint
  enforces `|Δq[k]| ≤ v_max·dt[k]` directly, so dt cannot drop below the
  per-segment velocity floor regardless of `dt_lo`. Setting `dt_lo` to
  the retimed warmstart (the old behavior) artificially blocked the
  optimizer from reclaiming time on segments where the planner's
  cartesian-level dt was looser than the joint-space vel floor. The
  retimed warmstart is still the *initial iterate* — only the lower
  bound is freed. `solve_stats.topp_floor_total_time` records the
  retimed-warm total; `solve_stats.total_time_after` is the optimizer's
  result. A positive gap = the freed `dt_lo` recovered slack.
  If the retimed `dt_warm.max() > dt_max`, `_build_box_bounds`
  auto-bumps `dt_max` to `1.05 × dt_warm.max()` with a one-line warning
  (rare; only triggers when the warmstart needs >1 s/segment for vel
  feasibility).
- **Sphere-BVH SDF is conservative** — sphere distance ≤ true mesh
  distance. `d_safe` is **hard** and **never auto-relaxed**: when the
  warm-start's min sphere-SDF undercuts the configured `d_safe`, the
  solver prints a one-line warning and runs anyway, asking the
  optimizer to drive `q` *away* from the colliding links rather than
  silently redefining "safe" as "wherever the warm happens to be".
  IPOPT's restoration phase handles warm-start-infeasible starts
  cleanly (the documented motivation for the IPOPT-vs-trust-constr
  default). Per [feedback_trajopt_hard_constraints.md], if the solver
  stalls, tighten upstream planner (Phase 1/2/3 collision avoidance)
  or lower `collision.d_safe` — don't reintroduce auto-relax.
  `solve_stats.d_safe_cfg` == `solve_stats.d_safe_used` always; both
  fields are kept for downstream JSON-schema compatibility. The
  post-solve gate `min_pair_distance_after ≥ d_safe_used` (or `None`,
  meaning every pair stayed past `d_warn`) is the regression alarm.
- **AMPL `fk_pos_and_jac` body-Jacobian is wrong-frame** (tracked
  separately) — the SCO tolerates this because the warm-start is feasible
  and the trust radius pins `q` near the warm, but any *unanchored* use
  (e.g. `densify.project_dense_to_cart`) needs finite-difference
  Jacobians as a workaround.
- **`fill_contact_jacobians` buffer shapes are fussy.** Three buffers
  must match the WBC's allocation exactly (F-order, `(6, 16)` /
  `(50, 16)` / `(50,)`) or nanobind rejects with an unhelpful
  `TypeError: incompatible function arguments`. Pre-allocated once in
  `RobotModel.__init__`; `np.copyto` per call.

### Phase 5b — `local_dp` alternative (`solver.method: local_dp`)

`optimize/local_dp.py` is a drop-in alternative to the SCO selected via
`optimize.solver.method: local_dp`. It runs *after* the same TOPP
pre-retime IPOPT uses (so the warm-start is velocity-feasible) and
*instead of* the SCO. Same `_optimized_<ts>.json` output path; same
visualizers.

- **Pipeline.**
  1. Sparse-sample 1/`sample_stride` of the warm trajectory (default 4).
     Endpoints are always included.
  2. Per sparse layer, sweep an `xy_offsets_m × xy_offsets_m × yaw_offsets_deg`
     grid (default `5 × 5 × 11 = 275` perturbed poses) around the warm
     target. `du`/`dv` translate the target along the surface PlaneFrame
     `u`/`v` axes; `yaw` rotates the quat about tool Z (`= -surface
     normal` in `align_z_to_normal` mode). At each pose run analytical
     IK with `nb_redundant_search` redundancy; collect status-ok arm-DOF
     solutions; dedup at 1e-3 rad.
  3. **Top-K cut**: sort the deduped pool by weighted-L2 distance to the
     warm `q_arm` at that layer (`JOINT_DIST_WEIGHTS = [2,2,2,1,1,1,1]`,
     same as Phase 1/3), keep the first `top_k` (default 100). The warm
     `q_arm` itself is force-included so the DP can recover the warm
     trajectory verbatim when no perturbation pays off.
  4. **Iterative-mask DP** — same iter-mask loop body as
     `_kshortest_dp_phase1` (forward single-best DP → reconstruct →
     `_endpoint_collides` per node, verdict cached across iterations →
     mask failures → repeat up to `max_iterations`), differing only in
     the per-edge cost matrix. The L∞ `jump_threshold` prune is shared.
     Two cost modes (`cost_mode`):
       - **`fastest` (default)** — minimize envelope time per edge:
         `edge = max(t_vel, t_accel)` with
         `t_vel = max_j |Δq[j]| / v_max[j]` (rad/s caps from
         `robot.joint_vel_limits`) and
         `t_accel = sqrt(max_j |Δq[j]| / a_max[j])` (rad/s² caps from
         `optimize.local_dp.densify.joint_accel_limits` — the same a_max
         Phase-2 TOPP will use). Raw caps; the small TOPP headroom is a
         constant scale on dt and doesn't change DP ordering. Surfaces
         IK branches that are kinematically faster for Phase-2 TOPP to
         traverse, not just nearest-in-joint-space. `accel_weight` is
         ignored (t_accel already captures the accel cost per edge).
       - **`shortest` (legacy)** — weighted-L2 joint distance
         (`JOINT_DIST_WEIGHTS = [2,2,2,1,1,1,1]`, Phase-1 style), plus
         the grandparent corner-accel surcharge `accel_weight ·
         weighted_L2(q[L] - 2q[L-1] + q[L-2])`. Kept for A/B comparison.
     Per-iteration progress log prints `cost=…s` in fastest mode and
     `cost=…rad` in shortest. The `solve_stats.local_dp.dp.cost_mode`
     field records which ran.
  5. **TOPP-RA re-densification** (`_toppra_densify`). Builds a
     `toppra.SplineInterpolator` cubic spline through the DP-selected
     sparse `q_arm` skeleton (`ss ∈ [0, 1]`; padded with the
     joint-space midpoint when only 2 sparse waypoints survive — the
     spline interpolator needs ≥ 3 knots). Sets
     `JointVelocityConstraint(v_max)` +
     `JointAccelerationConstraint(a_max)`, runs
     `toppra.algorithm.TOPPRA` with the `ParametrizeConstAccel`
     post-processor, and samples the resulting parametrization
     uniformly at `1/hz` time points. Target `pos` / `quat` at each
     sample are recovered by mapping the toppra grid's `(gridpoints,
     sd_vec)` to a `t→s` lookup and interpolating the sparse skeleton
     at the same `s(t)` (linear for `pos`, slerp for `quat`). FK +
     self-collision gates fire per emitted sample with the same
     `densify_deviation` / `densify_collision` failure shape
     `_phase2_topp_smooth_densify` emits. `compute_trajectory()`
     returning `None` (toppra's infeasibility verdict) surfaces as a
     `RuntimeError` so the caller can adjust caps or DP knobs. The
     result packs into a `SolverResult` so `write_optimizer_json`
     consumes it unchanged. `toppra` is a docker-only dependency —
     pre-installed in `sudodeploy_full_dev` at
     `/usr/local/lib/python3.10/dist-packages/toppra/`.
- **Plane axes** are read from `WarmStart.plane_u_axis` /
  `plane_v_axis`. Threaded from `PlanResult.plane` in end-to-end mode
  (`plan.py:897`) and from `metadata.plane_u_axis` /
  `metadata.plane_v_axis` of the warmstart JSON in optimizer-only mode
  (`load_warmstart`). Legacy warmstart JSONs without these fields fail
  fast with a clear error.
- **Collision verdicts** come from the shared
  `robot_check._collision_checker` singleton (rebuilt at the top of
  `solve_local_dp` with a wall-only `RobotCheckConfig` so the
  optimizer-only path works too — `_setup_collision_checker` reads only
  `cfg.wall`). Same ampl_motion convex-mesh `colcvx` the planner uses —
  Phase 5b agrees with Phase 1–4 on every collision query.
- **IK budget.** Per sparse layer = `len(xy_offsets_m)² ×
  len(yaw_offsets_deg)` perturbed-pose IK calls, each with up to
  `nb_redundant_search` redundant samples. With defaults that's
  `275 × 64 ≈ 17 600` raw IK solutions per layer; after dedup +
  `top_k=100` cut the DP pool size per layer is ≤ 100. For a
  `stride=4`, `N=1000` warm trajectory that's 250 sparse layers
  × 275 IK calls ≈ 69 000 IK calls. Drop `nb_redundant_search` first if
  this dominates.
- **Knobs** (under `optimize.local_dp`):
  - `sample_stride` (int, default 4) — `1/stride` of the warm is sparse-
    sampled.
  - `xy_offsets_m` (list[float], default `[-0.02, -0.01, 0, 0.01, 0.02]`)
    — `du` and `dv` taken from the same list; cartesian product yields
    the xy grid.
  - `yaw_offsets_deg` (list[float], default `range(-5, 6)`) — yaw about
    tool Z, deg.
  - `top_k` (int, default 100) — DP pool size cap per layer.
  - `nb_redundant_search` (int, default 64) — redundancy per perturbed
    pose.
  - `jump_threshold` (float, default 1.5 rad) — L∞ hard prune on
    inter-layer `|Δq_arm|∞`.
  - `cost_mode` (`fastest` | `shortest`, default `fastest`) — DP
    edge-cost objective; see Iterative-mask DP description above.
  - `accel_weight` (float, default 0.3) — grandparent corner-accel
    second-order DP cost weight. **Only consulted in `cost_mode:
    shortest`** (fastest's `t_accel` already accounts for accel per
    edge). Set 0 in shortest mode to fall back to dist-only.
  - `max_iterations` (int, default 20 000) — DP iter cap.
  - `densify.joint_accel_limits` (float | length-7 list, default 2.0
    rad/s²) — `a_max` used by both the DP edge cost (`cost_mode:
    fastest`) and the TOPP-RA `JointAccelerationConstraint`. Scalar
    broadcasts to length 7. `hz` is read from `robot.hz` and drives the
    output sample rate (`1/hz` per emitted QPoint). `ee_speed` is not
    consumed — TOPP-RA operates directly on the sparse joint path.
- **Stats** in `solve_stats.local_dp` of the optimizer JSON:
  `n_sparse_layers`, `sparse_indices`, `raw_pool_{avg,min,max}`,
  `top_k`, `total_ik_calls`, `pool_build_seconds`, `dp` (iter count,
  endpoint-mask total, dp/check timings, `cost_mode`,
  `final_cost_units` ∈ `{s, rad}`), `densify` (the TOPP-RA stats:
  `method: "toppra"`, `n_sparse_in`, `n_dense`, `total_time`,
  `max_vel_ratio`, `max_accel_ratio`, `n_collision_failures`,
  `n_dev_failures`, `joint_vel_limits`, `joint_accel_limits`, `hz`,
  `dt_out`, `toppra_n_gridpoints`), `n_phase2_failures`. The standard
  `solve_stats` fields (`total_time_after`,
  `final_summary.n_collisions`, etc.) are populated for the existing
  console + write path. TOPP-RA's `ParametrizeConstAccel` enforces
  `v_max` and `a_max` **at the toppra gridpoints**; between
  gridpoints the interpolated path can drift modestly above the
  caps (`max_vel_ratio` ~ 1.00 + a few percent; `max_accel_ratio`
  can reach ~1.2–1.4 on high-curvature spline paths with sharp IK
  branch flips). The drift falls with denser gridpoints
  (`toppra_n_gridpoints`); both ratios are reported for visibility
  so a regression in the upstream DP cost (e.g. a new branch flip
  creating non-`C¹` corners) shows up as an accel-ratio spike. The
  DP's `final_cost` in `fastest`
  mode is the lower-bound envelope-time sum; the TOPP-RA densify's
  `total_time` is the actual time the constant-accel parametrization
  needs to traverse the same path — close to the lower bound but
  slightly larger when accel limits gate corners the DP's envelope
  formula ignored.

## Conventions

- **Quaternion: `[w, x, y, z]`** (sapien / SudoDeploy convention).
- **Plane frame**: `origin = polygon[0]`, `u_axis = polygon[1] − polygon[0]`,
  `normal = u_axis × (polygon[2] − polygon[0])`, `v_axis = normal × u_axis`.
  Polygon is assumed coplanar (no SVD fit).
- **`pass_index`** is monotonic, dense, and re-numbered after corner
  rounding so it can keep being used for `n_passes = pass_idx.max() + 1`.
- **`kind` field** distinguishes raster vs arc samples (since corner
  rounding). Defaults to `"row"` for backward compatibility.

## Outputs

For each `plan.py` run with timestamp `TS`:
- `<name>_<TS>.json` — metadata + waypoints (+ `q_trajectory` when
  `robot_check` ran). When Phase 1a produced any IK pool statuses,
  `metadata.robot_check.ik_pools_path` points at a sibling sidecar.
- `<name>_<TS>_ik_pools.json` — sidecar always written when Phase 1a
  ran (even on Phase 1 failure — the IK landscape is exactly what the
  user needs to see when feasibility breaks). Always contains
  `ik_pool_statuses` (flat list[bool] per waypoint),
  `ik_pool_collision_checked` (flat list[int] per waypoint with values
  0=untested / 1=clean / 2=collided — see Phase 1 backends below for the
  sparse source), and, when Phase 1 completed a chain,
  `ik_selected_flat_indices`. The heavy `ik_pools` block (per-waypoint
  `(q_arm, quat_wxyz)` candidates) is included only when
  `save_all_ik_results: true`, since pools can be N × hundreds of
  entries. Regardless of the flag, `CheckResult.ik_pools`
  is always populated in memory so the main 2D PNG can render the
  pool-size colorbar; the flag now only gates this sidecar payload
  (for offline re-inspection / standalone re-renders).
  `visualize.render_png` loads the heavy block transparently when sizing
  waypoints by pool count from disk.
  When any Phase-1 `_endpoint_collides` hit fired, the sidecar also
  carries `collision_pair_stats` — a dict keyed by
  `"entity_a||entity_b"` (canonical alphabetic order, "||" delimiter
  that colcvx entity names never contain), values are the count of
  Phase-1 candidates whose endpoint check fired that specific pair.
  Insertion order is descending by count. Multiple pairs can hit at
  the same `q_full`, so the sum of counts can exceed the candidate
  count; that's the signal that lots of constraints stack up per
  pose. `check_trajectory` also prints the top-20 of this dict to
  stderr as a `=== Phase 1 collision pair stats ===` table — pointed
  at the question "which entity pair rejected most of my IK
  candidates?".
- `<name>_<TS>.png` — 2D (u,v) trajectory plot. Always colored by IK
  pool size — the heavy `ik_pools` block is always retained in memory
  after Phase 1a, so `plan.py` passes per-waypoint pool sizes directly
  into `render_png` and the viridis colorbar (`# IK candidates (per
  waypoint)`) renders on every run. When `render_png` is invoked
  standalone via the CLI, it falls back to reading the heavy block from
  the sidecar — so the colorbar there still requires
  `save_all_ik_results: true`. If `robot_check` did not run at all
  (`robot.enabled: false`), the renderer falls back to row/turn `kind`
  shading (no colorbar).
- `<name>_<TS>_joint_jump.png` — only when `robot_check` ran.
  Densified trajectory in (u,v) frame, each point colored by L∞ |Δq_arm|
  to its predecessor (`plasma` colormap, scaled to data max). Red line on
  the colorbar marks `robot.search.jump_threshold` for reference.
- `<name>_<TS>_ik_grid.png` — always emitted whenever Phase 1a ran.
  Binary IK convergence grid: x = waypoint id (left to right, ordered),
  y = candidate index (bottom to top, in cartesian-product order: roll
  outermost, then pitch, then yaw, then sorted branch key, then
  redundant index), white = IK converged, black = IK failed.
  Pre-collision-filter and pre-dedup, so it shows the *raw* IK landscape
  across the search pool — useful for spotting waypoints that are
  starved of solutions. When Phase 1 completed a chain, a red polyline
  overlays the chain of selected candidate flat-indices (the "Phase 1
  path"), so jumps in y-coordinate map directly to (roll, pitch, yaw,
  branch) switches the search made between waypoints. Reads `ik_pool_statuses` and
  `ik_selected_flat_indices` from the `*_ik_pools.json` sidecar (the
  heavy `ik_pools` block is not required by the grid renderer).
- `<name>_<TS>_collision_free.png` — emitted whenever
  `ik_pool_collision_checked` is populated (i.e. Phase 1a ran). Same
  shape and code path as the main `<name>_<TS>.png` 2D (u,v) trajectory
  plot — re-uses `render_png` / `visualize_matplotlib_2d` with a
  different `pool_sizes` and a custom `colorbar_label`. The viridis
  colorbar encodes the **per-waypoint count of IK candidates Phase 1
  confirmed collision-free** (sum of `1` cells in
  `ik_pool_collision_checked[wp]`). This is a *lower bound*: cells
  Phase 1 never visited (`0` = untested) are excluded, so the value
  is always `≤` the pre-collision pool size in `<name>_<TS>.png`.
  Backend matters: `astar` only `_endpoint_collides`-checks pool[0]
  (seed candidates), so this PNG will be saturated at 0 for every
  waypoint > 0; `kshortest_dp` exercises the cache densely across
  iterations, so its lower bound is close to tight. Diagnostic for
  "where did Phase 1 fail collision" — cold spots in this PNG are the
  collision-starved waypoints that masked the DP. **Waypoints whose
  *entire* dedup'd IK pool was collision-tested and every candidate
  collided are overlaid with a red X marker** (Phase 1 has no fallback
  at that waypoint — IK has options but every reachable arm config
  self-collides). The cross condition uses `check_result.ik_pools` for
  the dedup'd-pool-size denominator, so it's computed in `plan.py`
  alongside the per-waypoint collision-free count.

Phase 5 (when `optimize.enabled: true`) additionally emits:
- `<name>_<TS>_optimized_<ts>.json` — coarse SCO output (q_arm + dt per
  waypoint, `solve_stats` with `topp_floor_total_time` and
  `total_time_after` for the speedup audit).
- `<name>_<TS>_optimized_<ts>_densified.json` — uniform-dt control-rate
  trajectory (only when `output.densify_dt` is set).
- `<name>_<TS>_optimized_<ts>_{iter,qtraj,cart,dist,vel_accel}.png` —
  visualization PNGs. `vel_accel.png` plots |q_dot| (with v_max line)
  and |q_ddot| vs absolute time, warm overlaid with optimized — the
  velocity panel should show the opt curve saturating v_max on
  reclaimed-slack segments (proof the freed `dt_lo` is working).
- **Console block** `=== final constraint status ===` printed at the
  end of `_run_optimizer` — tabular SATISFIED/VIOLATED per hard
  constraint, plus `trust radius` utilization and `time speedup` vs
  the retimed warmstart. Built by `optimize.io.format_constraint_report`.

### Tactile-center pose alongside tool TCP

Every JSON that emits a tool-TCP cartesian pose also emits the matching
**tactile-center** pose, derived as
`T_world_tactile = T_world_tool @ inv(tool_offset_4x4)` via
`geometry.tool_pose_to_tactile_pose`. The tactile-center frame is the
kinematic frame `<arm>_link_tactile_center` that `ampl.KinHB11` solves IK
to; downstream consumers that need to plug the trajectory straight back
into IK / FK should read the `*_tactile_*` fields rather than reconstruct
the inverse offset themselves. When `tool_offset_4x4` is identity (no
offset configured) the tactile fields equal the TCP fields. Sentinel
rows where the achieved pose was never set (`achieved_quat_wxyz == null`)
also emit `achieved_tactile_quat_wxyz: null`.

Per-file additions:
- `<name>_<TS>.json` waypoints — `tactile_position`, `tactile_quat_wxyz`.
- `<name>_<TS>.json q_trajectory[]`,
  `<name>_<TS>_optimized_<ts>.json q_trajectory[]`, and
  `<input>_dense_<hz>hz.json q_trajectory[]` (densify CLI output) —
  `target_tactile_position`, `target_tactile_quat_wxyz`,
  `achieved_tactile_position`, `achieved_tactile_quat_wxyz`.

The existing `position` / `quaternion` / `target_position` /
`target_quat_wxyz` / `achieved_position` / `achieved_quat_wxyz` keys keep
their tool-tip semantics; visualizers and sidecar readers are unchanged.

## Gotchas

- **Docker required for `robot.enabled: true`** — `ampl`, `wbc_py`,
  RoboSkill imports fail outside the SudoDeploy container.
- **Geometry-only flow uses `.venv/` (Python 3.10)**, NOT the SudoDeploy
  main venv (3.14, no open3d wheels).
- **Two CWD assumptions**: `plan.py` resolves config-relative paths from
  the config file's directory; `visualize.py` is path-agnostic.
- **Quaternions can be perturbed by `robot_check`**: when the canonical
  TCP pose is in collision, A* may pick a non-zero yaw/pitch offset from
  the IK pool. The output JSON's `quaternion` reflects the *selected*
  pose, which can differ from the planner's canonical pose. Don't assume
  `quaternion[i]` matches what `boustrophedon.build_tcp_pose` would emit.
- **Polygon `contains_points` excludes some boundary samples**.
  `_generate_uv_grid` may emit one fewer point per line than naive count
  suggests. Don't tighten downstream length assumptions.
- **Corner rounding adds passes**. `n_passes` after rounding ≈
  `2 * raster_passes − 1` (interleaved row+turn). Anywhere consumers
  expected raster-only passes, gate behavior on `kind` instead of
  `pass_index` parity.
- **Adding more user waypoints → more A* layers → slower Phase 1**.
  The corner-densification trade-off is intentional: smoother arcs cost
  IK calls. If feasibility checks blow up, raise `arc_spacing` or trim
  `yaw_search_offsets_deg` / `pitch_search_offsets_deg` /
  `roll_search_offsets_deg`.
- **Phase 3 IK budget scales as
  `n_yaw·n_pitch·n_xy·refine_nb_redundant_search` per interior step**
  — with the canonical defaults that's ~117k IK calls per step. If
  Phase 3 dominates a run, the cheap knobs are `xy_offsets_m` (drop to
  `[[0,0]]`) and `refine_nb_redundant_search`; the 1° yaw/pitch grid
  is the user's explicit ask. Per-region failures fall back to the
  Phase 2 slice automatically — Phase 3 never makes the trajectory
  worse.
- **`MOTION_PLAN_ASSETS_DIR`** env var is read by `robot_check` for
  collision meshes — set when running standalone outside compose.
- **`ampl_motion-0.0.2` ships a `np.bool` type-hint bug** at
  `util_hb.py:229` (etc.). NumPy ≥ 1.20 removed `np.bool`, so `import
  ampl_motion` raises `AttributeError` on a fresh container. `robot_check.py`
  re-adds the alias as a module attribute (`numpy.bool = bool`) before
  importing — assignment bypasses NumPy's `__getattr__` deprecation gate.
- **Workspace check is arm-only.** `ampl_motion.configure_workspace` would
  by default include every link, but the torso sits below `z_min`
  (typically 0.6 m, the surface-cover working plane) and would flag every
  pose. `_setup_collision_checker` restricts the workspace `includes` list
  to link names starting with `left-` or `right-` (the two arm chains
  including hand / tactile frames). Matches HBMP's `set_wall(...)`
  semantics. If a different robot ships with different link-name prefixes,
  this filter needs updating.
- **Wall is configured once, read from a stash.** `_setup_collision_checker`
  calls `ampl_motion.configure_workspace(colcvx, ..., min3, max3)` for the
  actual workspace check, and separately stashes the same bounds in
  `_collision_checker["wall"]` for `_diag_cause`'s failure printout. No
  `agent.set_wall(...)` / `agent.wall()` calls remain (HBMP layer was
  fully removed in the collision swap); `_diag_cause` reads
  `_collision_checker["wall"]` directly. If you re-route the wall config,
  update both the `configure_workspace` call and the stash.
