"""Core boustrophedon trajectory generator.

Pipeline:
    build plane frame from polygon
    raster lines in (u, v)  ->  clip to polygon  ->  unproject to 3D
    build TCP pose per sample (surface = polygon plane)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath

from geometry import (
    PlaneFrame,
    build_mesh_transform,
    build_tcp_pose,
    build_tool_offset_transform,
    plane_from_polygon,
    project_to_plane_uv,
    unproject_from_plane_uv,
)


@dataclass
class Waypoint:
    name: str
    position: np.ndarray                          # (3,)
    quaternion_wxyz: np.ndarray                   # (4,) — robot_check may overwrite with a ±1° perturbation
    surface_point: np.ndarray                     # (3,)
    surface_normal: np.ndarray                    # (3,)  outward, post-flip
    pass_index: int
    point_in_pass: int
    kind: str = "row"                             # "row" for raster sweep points, "turn" for arc samples between passes


@dataclass
class PlanResult:
    waypoints: list[Waypoint]
    plane: PlaneFrame
    polygon_xyz: np.ndarray
    mesh_transform_4x4: np.ndarray
    tool_offset_4x4: np.ndarray  # 4x4; identity when no tcp.tool_offset configured


def _generate_uv_grid(
    polygon_uv: np.ndarray,
    step_over: float,
    point_spacing: float,
    sweep_axis: str,
    serpentine: bool,
    start_corner: str,
) -> list[tuple[int, np.ndarray]]:
    """Return [(pass_index, (M, 2) array of (u, v) samples in pass-order), ...].

    Each pass is a contiguous run of samples along `sweep_axis` clipped to the polygon.
    Samples that fall outside the polygon are dropped. A single (u, v) line that crosses
    the polygon multiple times is split into multiple passes.
    """
    if sweep_axis not in ("u", "v"):
        raise ValueError(f"sweep_axis must be 'u' or 'v', got {sweep_axis!r}")

    u_min, v_min = polygon_uv.min(axis=0)
    u_max, v_max = polygon_uv.max(axis=0)

    if sweep_axis == "u":
        # parallel lines run along u; lines are stacked along v
        line_axis_min, line_axis_max = u_min, u_max
        stack_axis_min, stack_axis_max = v_min, v_max
        line_axis_step = point_spacing
        stack_axis_step = step_over
    else:
        line_axis_min, line_axis_max = v_min, v_max
        stack_axis_min, stack_axis_max = u_min, u_max
        line_axis_step = point_spacing
        stack_axis_step = step_over

    n_lines = max(2, int(np.ceil((stack_axis_max - stack_axis_min) / stack_axis_step)) + 1)
    n_pts = max(2, int(np.ceil((line_axis_max - line_axis_min) / line_axis_step)) + 1)
    stack_coords = np.linspace(stack_axis_min, stack_axis_max, n_lines)
    line_coords = np.linspace(line_axis_min, line_axis_max, n_pts)

    apply_corner_swap = start_corner in ("uv_max", "u_max_v_min")
    apply_corner_flip_stack = start_corner in ("uv_max", "u_min_v_max")
    apply_corner_flip_line = start_corner in ("uv_max", "u_max_v_min")

    if apply_corner_flip_stack:
        stack_coords = stack_coords[::-1]
    if apply_corner_flip_line:
        line_coords = line_coords[::-1]
    del apply_corner_swap

    poly_path = MplPath(polygon_uv)

    passes: list[tuple[int, np.ndarray]] = []
    next_pass_idx = 0
    for line_idx, stack_v in enumerate(stack_coords):
        if serpentine and line_idx % 2 == 1:
            this_line_coords = line_coords[::-1]
        else:
            this_line_coords = line_coords

        if sweep_axis == "u":
            uv = np.stack([this_line_coords, np.full_like(this_line_coords, stack_v)], axis=-1)
        else:
            uv = np.stack([np.full_like(this_line_coords, stack_v), this_line_coords], axis=-1)

        inside = poly_path.contains_points(uv)
        # split into contiguous-inside segments
        seg_start = None
        for i, ok in enumerate(inside):
            if ok and seg_start is None:
                seg_start = i
            elif not ok and seg_start is not None:
                passes.append((next_pass_idx, uv[seg_start:i]))
                next_pass_idx += 1
                seg_start = None
        if seg_start is not None:
            passes.append((next_pass_idx, uv[seg_start:]))
            next_pass_idx += 1

    return passes


# ---------------------------------------------------------------------------
# Corner rounding — convert sharp 180° serpentine flips into G1-continuous arcs.
# All operations are in flat (u, v) plane coordinates; arcs unproject to circles
# in 3D since the polygon plane is flat.
# ---------------------------------------------------------------------------


@dataclass
class _UvSeg:
    pass_index: int
    uv: np.ndarray         # (M, 2)
    kinds: np.ndarray      # (M,) object dtype, "row" or "turn"


def _polyline_length(uv: np.ndarray) -> float:
    if len(uv) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(uv, axis=0), axis=1).sum())


def _unit_tangent(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    d = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return np.array([1.0, 0.0])
    return d / n


def _trim_uv(uv: np.ndarray, *, head: float, tail: float) -> np.ndarray:
    """Drop arc-length `head` from the start and `tail` from the end of `uv`,
    interpolating to land exactly at the trim distances. Returns an empty
    (0, 2) array if the trims overlap."""
    if len(uv) == 0 or (head < 1e-12 and tail < 1e-12):
        return uv.copy()

    seg = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])

    s_lo = float(head)
    s_hi = total - float(tail)
    if s_lo >= s_hi - 1e-9:
        return uv[:0].copy()

    def _interp(s: float) -> np.ndarray:
        idx = int(np.searchsorted(cum, s, side="right") - 1)
        idx = max(0, min(idx, len(uv) - 2))
        seg_len = cum[idx + 1] - cum[idx]
        if seg_len < 1e-12:
            return uv[idx].astype(np.float64).copy()
        t = (s - cum[idx]) / seg_len
        return uv[idx] * (1.0 - t) + uv[idx + 1] * t

    p_lo = _interp(s_lo)
    p_hi = _interp(s_hi)
    interior_mask = (cum > s_lo + 1e-12) & (cum < s_hi - 1e-12)
    interior = uv[interior_mask]
    return np.vstack([p_lo[None, :], interior, p_hi[None, :]])


def _is_turn_pair(prev_uv: np.ndarray, next_uv: np.ndarray, step_over: float) -> bool:
    """A serpentine turn between adjacent passes: anti-parallel end/start
    tangents, displacement magnitude near `step_over`, and that displacement
    perpendicular to the tangent (rejects polygon-induced same-line splits
    whose displacement is *along* the line direction)."""
    if len(prev_uv) < 2 or len(next_uv) < 2:
        return False
    t_prev = _unit_tangent(prev_uv[-2], prev_uv[-1])
    t_next = _unit_tangent(next_uv[0], next_uv[1])
    if float(np.dot(t_prev, t_next)) > -0.9:
        return False
    delta = next_uv[0] - prev_uv[-1]
    delta_norm = float(np.linalg.norm(delta))
    if not (0.75 * step_over <= delta_norm <= 1.25 * step_over):
        return False
    along = abs(float(np.dot(delta, t_prev)))
    if along > 0.25 * step_over:
        return False
    return True


def _half_circle_arc(
    p0: np.ndarray,
    p1: np.ndarray,
    tangent0: np.ndarray,
    radius: float,
    arc_spacing: float,
    num_points: int | None = None,
) -> np.ndarray:
    """Sample the *interior* of a circular arc from p0 to p1 with tangent0 at p0.

    Endpoints p0 / p1 are excluded — they belong to the trimmed row segments.
    When `num_points` is set, exactly that many interior samples are emitted
    at evenly-spaced angles in (0, π); otherwise sample count is derived from
    `arc_spacing`. For a serpentine corner the chord |p1 - p0| = 2*radius
    (half-circle) but the function falls back to a chord densification if the
    geometry doesn't match (defensive only — should never trigger with
    symmetric trim)."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    tangent0 = np.asarray(tangent0, dtype=np.float64)

    chord = p1 - p0
    chord_len = float(np.linalg.norm(chord))
    if chord_len < 1e-9:
        return np.zeros((0, 2), dtype=np.float64)

    # Half-circle requires chord = diameter; degrade gracefully otherwise.
    if abs(chord_len - 2.0 * radius) > 1e-3 * max(radius, 1.0):
        if num_points is not None:
            n_interior = max(1, int(num_points))
        else:
            n_interior = max(1, int(np.round(chord_len / max(arc_spacing, 1e-6))) - 1)
        ts = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
        return p0[None, :] * (1.0 - ts[:, None]) + p1[None, :] * ts[:, None]

    # Center sits on the perpendicular to tangent0 on the side of chord.
    perp = np.array([-tangent0[1], tangent0[0]], dtype=np.float64)
    if float(np.dot(perp, chord)) < 0.0:
        perp = -perp
    center = p0 + radius * perp

    v0 = p0 - center
    a0 = float(np.arctan2(v0[1], v0[0]))
    cross_t = float(v0[0] * tangent0[1] - v0[1] * tangent0[0])
    sign = 1.0 if cross_t > 0.0 else -1.0

    if num_points is not None:
        n_interior = max(1, int(num_points))
    else:
        n_interior = max(1, int(np.round(np.pi * radius / max(arc_spacing, 1e-6))) - 1)
    ts = np.linspace(0.0, np.pi, n_interior + 2)[1:-1]
    angles = a0 + sign * ts
    arc = center[None, :] + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return arc


def _round_uv_corners(
    passes: list[tuple[int, np.ndarray]],
    *,
    step_over: float,
    trim: float,
    arc_spacing: float,
    polygon_uv: np.ndarray | None,
    num_points: int | None = None,
) -> list[_UvSeg]:
    """Replace sharp serpentine turns with sampled half-circle arcs.

    Each row keeps its own pass_index (re-numbered sequentially); each arc
    becomes its own pass with its own pass_index. The arc radius is derived
    per-pair from the actual chord between trimmed-row endpoints
    (`chord_len / 2`) — `step_over` is the user's *target* spacing, but the
    raster's `linspace` lays rows out at `range / (n_lines - 1)` which only
    matches `step_over` when the polygon range divides evenly. Using the
    actual chord lets the half-circle fit exactly between every pair,
    instead of degenerating to a straight chord when the spacing is off by
    a few mm.

    Rounding is *disabled* for a pair when (a) the row would shrink below
    `arc_spacing` after trim, (b) the arc samples fall outside the polygon,
    or (c) the pair isn't a valid serpentine turn (parallel tangents,
    polygon-induced splits, …).
    """
    n = len(passes)
    if n == 0:
        return []

    poly_path = MplPath(polygon_uv) if polygon_uv is not None else None

    is_turn = [False] * max(0, n - 1)
    for i in range(n - 1):
        if _is_turn_pair(passes[i][1], passes[i + 1][1], step_over):
            is_turn[i] = True

    # Per-pair half-circle radius from the raw row endpoints. The chord is
    # invariant under symmetric trim (trim moves endpoints along the row
    # tangent, not perpendicular to it), so the raw-endpoint distance is the
    # same as the trimmed-endpoint distance. Cached up front so the trim /
    # disable loop and the emit loop both see the same value.
    pair_radius: list[float] = []
    for i in range(n - 1):
        if is_turn[i]:
            chord = passes[i + 1][1][0] - passes[i][1][-1]
            pair_radius.append(float(np.linalg.norm(chord)) / 2.0)
        else:
            pair_radius.append(0.0)

    pass_total_len = [_polyline_length(uv) for _, uv in passes]
    min_remaining = max(arc_spacing, 1e-3)

    def _disable(idx: int, reason: str) -> None:
        print(f"[boustrophedon] turn rounding disabled between pass {idx} and {idx + 1}: {reason}")
        is_turn[idx] = False

    while True:
        any_change = False
        for i in range(n - 1):
            if not is_turn[i]:
                continue
            ht_a = trim if (i > 0 and is_turn[i - 1]) else 0.0
            tt_a = trim
            ht_b = trim
            tt_b = trim if (i + 1 < n - 1 and is_turn[i + 1]) else 0.0
            if pass_total_len[i] < ht_a + tt_a + min_remaining:
                _disable(i, f"pass {i} too short for trim ({pass_total_len[i]:.4f}m)")
                any_change = True
                continue
            if pass_total_len[i + 1] < ht_b + tt_b + min_remaining:
                _disable(i, f"pass {i + 1} too short for trim ({pass_total_len[i + 1]:.4f}m)")
                any_change = True
                continue
            tr_a = _trim_uv(passes[i][1], head=ht_a, tail=tt_a)
            tr_b = _trim_uv(passes[i + 1][1], head=ht_b, tail=tt_b)
            if len(tr_a) < 2 or len(tr_b) < 2:
                _disable(i, "trimmed pass < 2 points")
                any_change = True
                continue
            t_a = _unit_tangent(tr_a[-2], tr_a[-1])
            arc_uv = _half_circle_arc(
                p0=tr_a[-1], p1=tr_b[0], tangent0=t_a,
                radius=pair_radius[i], arc_spacing=arc_spacing,
                num_points=num_points,
            )
            if poly_path is not None and len(arc_uv) > 0:
                if not poly_path.contains_points(arc_uv, radius=1e-9).all():
                    _disable(i, "arc bulges outside polygon")
                    any_change = True
                    continue
        if not any_change:
            break

    out: list[_UvSeg] = []
    next_pid = 0
    for i, (_, uv) in enumerate(passes):
        ht = trim if (i > 0 and is_turn[i - 1]) else 0.0
        tt = trim if (i < n - 1 and is_turn[i]) else 0.0
        trimmed_uv = _trim_uv(uv, head=ht, tail=tt) if (ht > 0.0 or tt > 0.0) else uv.copy()
        out.append(_UvSeg(
            pass_index=next_pid,
            uv=trimmed_uv,
            kinds=np.array(["row"] * len(trimmed_uv), dtype=object),
        ))
        next_pid += 1
        if i < n - 1 and is_turn[i]:
            ht_b = trim
            tt_b = trim if (i + 1 < n - 1 and is_turn[i + 1]) else 0.0
            next_trimmed_uv = _trim_uv(passes[i + 1][1], head=ht_b, tail=tt_b)
            t_a = _unit_tangent(trimmed_uv[-2], trimmed_uv[-1])
            arc_uv = _half_circle_arc(
                p0=trimmed_uv[-1], p1=next_trimmed_uv[0], tangent0=t_a,
                radius=pair_radius[i], arc_spacing=arc_spacing,
                num_points=num_points,
            )
            out.append(_UvSeg(
                pass_index=next_pid,
                uv=arc_uv,
                kinds=np.array(["turn"] * len(arc_uv), dtype=object),
            ))
            next_pid += 1

    return out


def plan_boustrophedon(cfg: dict) -> PlanResult:
    mesh_cfg = cfg["mesh"]
    region_cfg = cfg["region"]
    pat_cfg = cfg["pattern"]
    tcp_cfg = cfg["tcp"]

    # Mesh is not loaded here — it's only used by visualize.py. We still build
    # the transform so it's recorded in the JSON metadata for the visualizer.
    transform_4x4 = build_mesh_transform(mesh_cfg.get("transform", {}))
    flip_normals = bool(mesh_cfg.get("flip_normals", False))

    # 1. plane frame from polygon
    polygon_xyz = np.asarray(region_cfg["polygon"], dtype=np.float64)
    plane = plane_from_polygon(polygon_xyz)
    polygon_uv = project_to_plane_uv(polygon_xyz, plane)

    # 2. raster grid + clip
    passes = _generate_uv_grid(
        polygon_uv=polygon_uv,
        step_over=float(pat_cfg["step_over"]),
        point_spacing=float(pat_cfg["point_spacing"]),
        sweep_axis=str(pat_cfg.get("sweep_axis", "u")),
        serpentine=bool(pat_cfg.get("serpentine", True)),
        start_corner=str(pat_cfg.get("start_corner", "uv_min")),
    )

    if not passes:
        raise RuntimeError("no passes generated — polygon may be too small for step_over/point_spacing")

    # 2b. round serpentine corners (sharp 180° flips → tangent-continuous arcs)
    turn_cfg = pat_cfg.get("turn") or {}
    serpentine = bool(pat_cfg.get("serpentine", True))
    if serpentine and bool(turn_cfg.get("enabled", True)):
        shape = str(turn_cfg.get("shape", "half_circle"))
        if shape != "half_circle":
            raise ValueError(
                f"pattern.turn.shape={shape!r} not implemented (only 'half_circle')"
            )
        step_over_val = float(pat_cfg["step_over"])
        # The arc radius is now derived per-pair from the actual chord between
        # trimmed rows inside `_round_uv_corners` (the raster's linspace lays
        # rows out at `range / (n_lines - 1)`, which only matches `step_over`
        # when the polygon range divides evenly). `pattern.turn.radius` is
        # therefore advisory: it's logged if the caller sets it but never
        # overrides the geometric value.
        radius_cfg = turn_cfg.get("radius")
        if radius_cfg is not None:
            print(
                f"[boustrophedon] pattern.turn.radius={float(radius_cfg)} is advisory "
                f"and ignored; the half-circle radius is derived per turn-pair "
                f"from the actual row spacing (≈ step_over/2 = {step_over_val / 2.0})."
            )
        radius = step_over_val / 2.0  # used only to size `trim` defaults / floors below
        trim_cfg = turn_cfg.get("trim")
        trim = float(trim_cfg) if trim_cfg is not None else radius
        if trim < radius - 1e-9:
            raise ValueError(
                f"pattern.turn.trim={trim} < radius={radius}; the arc would bulge "
                f"beyond the polygon edge. Increase trim to >= step_over/2."
            )
        spacing_cfg = turn_cfg.get("point_spacing")
        arc_spacing = (
            float(spacing_cfg)
            if spacing_cfg is not None
            else float(pat_cfg["point_spacing"]) / 2.0
        )
        num_points_cfg = turn_cfg.get("num_points", 8)
        num_points = int(num_points_cfg) if num_points_cfg is not None else None
        segs = _round_uv_corners(
            passes,
            step_over=step_over_val,
            trim=trim,
            arc_spacing=arc_spacing,
            polygon_uv=polygon_uv,
            num_points=num_points,
        )
    else:
        segs = [
            _UvSeg(pid, uv, np.array(["row"] * len(uv), dtype=object))
            for pid, uv in passes
        ]

    # 3. unproject samples back to 3D on the polygon plane
    flat_uv = np.concatenate([s.uv for s in segs], axis=0)
    flat_kinds = np.concatenate([s.kinds for s in segs], axis=0)
    pass_ids_flat = np.concatenate([np.full(len(s.uv), s.pass_index) for s in segs])
    point_in_pass_flat = np.concatenate([np.arange(len(s.uv)) for s in segs])

    flat_xyz = unproject_from_plane_uv(flat_uv, plane)

    # 4. build poses — surface = polygon plane, normal = plane.normal
    surface_normal = -plane.normal if flip_normals else plane.normal

    rotation_mode = str(tcp_cfg.get("rotation_mode", "align_z_to_normal"))
    standoff = float(tcp_cfg.get("standoff", 0.0))
    yaw_deg = float(tcp_cfg.get("yaw_deg", 0.0))
    yaw_ref = np.asarray(tcp_cfg.get("yaw_reference_axis", [1.0, 0.0, 0.0]), dtype=np.float64)
    fixed_quat = np.asarray(tcp_cfg.get("fixed_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)

    tool_offset_4x4 = build_tool_offset_transform(tcp_cfg.get("tool_offset"))

    waypoints: list[Waypoint] = []
    for i in range(flat_xyz.shape[0]):
        position, q_wxyz = build_tcp_pose(
            surface_point=flat_xyz[i],
            surface_normal_outward=surface_normal,
            standoff=standoff,
            rotation_mode=rotation_mode,
            fixed_quat_wxyz=fixed_quat,
            yaw_deg=yaw_deg,
            yaw_reference_axis=yaw_ref,
        )
        waypoints.append(
            Waypoint(
                name=f"wp_{len(waypoints):04d}",
                position=position,
                quaternion_wxyz=q_wxyz,
                surface_point=flat_xyz[i],
                surface_normal=surface_normal,
                pass_index=int(pass_ids_flat[i]),
                point_in_pass=int(point_in_pass_flat[i]),
                kind=str(flat_kinds[i]),
            )
        )

    return PlanResult(
        waypoints=waypoints,
        plane=plane,
        polygon_xyz=polygon_xyz,
        mesh_transform_4x4=transform_4x4,
        tool_offset_4x4=tool_offset_4x4,
    )
