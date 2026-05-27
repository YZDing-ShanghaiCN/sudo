"""Standalone visualizer for surface-cover trajectories.

Reads only the mesh (--mesh) and the trajectory JSON (--trajectory). Independent
of plan.py state — can be re-run on any historical trajectory.

Default backend is matplotlib (saves a PNG, also opens an interactive window
unless --no-show). Pass --interactive to use open3d instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_inputs(mesh_path: str, traj_path: str):
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"loaded asset is not a Trimesh: {type(mesh).__name__}")

    with open(traj_path) as f:
        traj = json.load(f)
    return mesh, traj


def _apply_metadata_transform(mesh, traj_meta) -> None:
    T = np.asarray(traj_meta.get("mesh_transform_matrix_4x4", np.eye(4).tolist()), dtype=np.float64)
    if not np.allclose(T, np.eye(4)):
        mesh.apply_transform(T)


def _extract_arrays(traj: dict):
    wps = traj["waypoints"]
    positions = np.array([w["position"] for w in wps], dtype=np.float64)
    quats = np.array([w["quaternion"] for w in wps], dtype=np.float64)  # wxyz
    surface_pts = np.array([w["surface_point"] for w in wps], dtype=np.float64)
    pass_idx = np.array([w["pass_index"] for w in wps], dtype=np.int32)
    kinds = np.array([w.get("kind", "row") for w in wps], dtype=object)  # back-compat default
    polygon = np.array(traj["metadata"].get("polygon_xyz", []), dtype=np.float64)
    return positions, quats, surface_pts, pass_idx, kinds, polygon


def _project_to_uv(pts: np.ndarray, origin: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray) -> np.ndarray:
    """Project (N, 3) world-frame points into the plane's local (u, v) frame."""
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    rel = pts - origin
    u = rel @ u_axis
    v = rel @ v_axis
    return np.stack([u, v], axis=1)


def _ik_pool_sizes(traj: dict, n_waypoints: int) -> np.ndarray | None:
    """Return per-waypoint IK pool size from `ik_pools`, or None if absent."""
    pools = traj.get("ik_pools")
    if not pools:
        return None
    sizes = np.zeros(n_waypoints, dtype=np.int32)
    for entry in pools:
        idx = int(entry["user_wp_index"])
        if 0 <= idx < n_waypoints:
            sizes[idx] = len(entry.get("candidates", []))
    return sizes


def visualize_matplotlib_2d(
    positions: np.ndarray,
    polygon: np.ndarray,
    plane_origin: np.ndarray,
    plane_u_axis: np.ndarray,
    plane_v_axis: np.ndarray,
    pass_idx: np.ndarray,
    pool_sizes: np.ndarray | None,
    out_png: str | None,
    show: bool,
    kinds: np.ndarray | None = None,
    colorbar_label: str | None = None,
    cross_mask: np.ndarray | None = None,
) -> None:
    """2D plot in the plane's local (u, v) frame.

    Waypoints are colored by IK pool size when available (gray fallback when
    `ik_pools` was not saved in the JSON).
    """
    import matplotlib.pyplot as plt

    pos_uv = _project_to_uv(positions, plane_origin, plane_u_axis, plane_v_axis)
    poly_uv = _project_to_uv(polygon, plane_origin, plane_u_axis, plane_v_axis)

    fig, ax = plt.subplots(figsize=(11, 9))

    if poly_uv.size:
        loop = np.vstack([poly_uv, poly_uv[:1]])
        ax.plot(loop[:, 0], loop[:, 1], color="#1f77b4", linewidth=2.0, label="region polygon")

    if pos_uv.shape[0] > 1:
        ax.plot(pos_uv[:, 0], pos_uv[:, 1], color="#888888", linewidth=0.8, alpha=0.6, label="trajectory")

    if pool_sizes is not None:
        sc = ax.scatter(
            pos_uv[:, 0], pos_uv[:, 1],
            c=pool_sizes, cmap="viridis",
            s=40, edgecolor="black", linewidth=0.4,
            vmin=0, vmax=max(int(pool_sizes.max()), 1),
            zorder=3,
        )
        cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(colorbar_label or "# IK candidates (per waypoint)")
    elif kinds is not None and np.any(kinds == "turn"):
        is_row = (kinds == "row")
        is_turn = (kinds == "turn")
        ax.scatter(
            pos_uv[is_row, 0], pos_uv[is_row, 1],
            color="#d62728", marker="s", s=18, edgecolor="black", linewidth=0.3,
            zorder=3, label=f"row ({int(is_row.sum())})",
        )
        ax.scatter(
            pos_uv[is_turn, 0], pos_uv[is_turn, 1],
            color="#1f77b4", marker="o", s=14, edgecolor="black", linewidth=0.3,
            zorder=3, label=f"turn ({int(is_turn.sum())})",
        )
    else:
        ax.scatter(
            pos_uv[:, 0], pos_uv[:, 1],
            color="#d62728", s=18, edgecolor="black", linewidth=0.3,
            zorder=3, label="waypoints (ik_pools not saved)",
        )

    if cross_mask is not None and bool(np.any(cross_mask)):
        m = np.asarray(cross_mask, dtype=bool)
        if m.shape[0] == pos_uv.shape[0]:
            ax.scatter(
                pos_uv[m, 0], pos_uv[m, 1],
                color="#d62728", marker="x", s=110, linewidths=2.2,
                zorder=6, label=f"all candidates collided ({int(m.sum())})",
            )

    n_passes = int(pass_idx.max() + 1) if pass_idx.size else 0
    ax.set_xlabel("u (plane axis, m)")
    ax.set_ylabel("v (plane axis, m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title(f"surface-cover trajectory — {len(positions)} waypoints, {n_passes} passes")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"saved {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def _quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return R.from_quat(q_xyzw).as_matrix()


def visualize_matplotlib(
    mesh,
    positions: np.ndarray,
    quats: np.ndarray,
    surface_pts: np.ndarray,
    pass_idx: np.ndarray,
    polygon: np.ndarray,
    out_png: str | None,
    show: bool,
    frame_size: float,
    frame_stride: int,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # mesh as triangle collection (downsample if huge)
    faces = mesh.faces
    if faces.shape[0] > 8000:
        # face-stride downsample to keep matplotlib responsive
        keep = np.linspace(0, faces.shape[0] - 1, 8000).astype(int)
        faces = faces[keep]
    tri = mesh.vertices[faces]
    coll = Poly3DCollection(tri, alpha=0.18, facecolor="#bbbbbb", edgecolor="#888888", linewidth=0.2)
    ax.add_collection3d(coll)

    # polygon outline
    if polygon.size:
        loop = np.vstack([polygon, polygon[:1]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="#1f77b4", linewidth=2.0, label="region polygon")

    # trajectory polyline
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="#d62728", linewidth=1.2, label="trajectory")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color="#d62728", s=4)

    # surface points (tiny dots) for reference
    ax.scatter(surface_pts[:, 0], surface_pts[:, 1], surface_pts[:, 2], color="#2ca02c", s=2, alpha=0.6, label="surface hits")

    # frame triads (every Nth)
    for i in range(0, len(positions), max(1, frame_stride)):
        rot = _quat_wxyz_to_rotmat(quats[i])
        p = positions[i]
        for k, color in enumerate(("#cc0000", "#00aa00", "#0044cc")):  # X, Y, Z
            d = rot[:, k] * frame_size
            ax.plot([p[0], p[0] + d[0]], [p[1], p[1] + d[1]], [p[2], p[2] + d[2]], color=color, linewidth=1.4)

    # axis bounds: union of mesh + trajectory
    pts_all = np.vstack([mesh.vertices, positions])
    mins = pts_all.min(axis=0)
    maxs = pts_all.max(axis=0)
    centers = (mins + maxs) / 2
    half = float((maxs - mins).max()) / 2 * 1.05
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"surface-cover trajectory — {len(positions)} waypoints, "
                 f"{int(pass_idx.max() + 1) if pass_idx.size else 0} passes")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"saved {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_open3d(
    mesh,
    positions: np.ndarray,
    quats: np.ndarray,
    surface_pts: np.ndarray,
    polygon: np.ndarray,
    frame_size: float,
    frame_stride: int,
) -> None:
    import open3d as o3d

    geoms = []

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color([0.75, 0.75, 0.75])
    geoms.append(o3d_mesh)

    if polygon.size:
        loop = np.vstack([polygon, polygon[:1]])
        lines = [[i, i + 1] for i in range(len(polygon))]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(loop)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([[0.12, 0.47, 0.71]] * len(lines))
        geoms.append(ls)

    if len(positions) > 1:
        traj_lines = [[i, i + 1] for i in range(len(positions) - 1)]
        traj_ls = o3d.geometry.LineSet()
        traj_ls.points = o3d.utility.Vector3dVector(positions)
        traj_ls.lines = o3d.utility.Vector2iVector(traj_lines)
        traj_ls.colors = o3d.utility.Vector3dVector([[0.84, 0.15, 0.16]] * len(traj_lines))
        geoms.append(traj_ls)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector([[0.84, 0.15, 0.16]] * len(positions))
    geoms.append(pcd)

    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(surface_pts)
    surface_pcd.colors = o3d.utility.Vector3dVector([[0.17, 0.63, 0.17]] * len(surface_pts))
    geoms.append(surface_pcd)

    for i in range(0, len(positions), max(1, frame_stride)):
        rot = _quat_wxyz_to_rotmat(quats[i])
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = positions[i]
        ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        ax.transform(T)
        geoms.append(ax)

    o3d.visualization.draw_geometries(geoms, window_name="surface-cover trajectory")


def render_joint_jump_png(
    trajectory_path: str,
    out_png: str,
    jump_threshold: float | None = None,
) -> None:
    """Plot the densified TCP path in the plane's (u, v) frame, colored by
    the L∞ joint jump |Δq_arm| between consecutive steps.

    Each densified step is a scatter point at its target (u, v); color
    encodes max |Δq| across the active arm's 7 DOFs (rad). Polygon outline
    is overlaid. `jump_threshold`, when given, sets the colorbar's vmax so
    the same color scale is comparable across runs.
    """
    import matplotlib.pyplot as plt

    with open(trajectory_path) as f:
        traj = json.load(f)

    qt = traj.get("q_trajectory") or []
    if len(qt) < 2:
        print("[visualize] q_trajectory missing or too short; skipping joint-jump plot")
        return

    has_left = any(qp.get("q_left_arm") is not None for qp in qt)
    has_right = any(qp.get("q_right_arm") is not None for qp in qt)
    if not (has_left or has_right):
        print("[visualize] no per-step arm joints in q_trajectory; skipping joint-jump plot")
        return
    arm_key = "q_left_arm" if has_left else "q_right_arm"

    pos_xyz = []
    q_arm = []
    for qp in qt:
        v = qp.get(arm_key)
        tp = qp.get("target_position")
        if v is None or tp is None:
            continue
        pos_xyz.append(tp)
        q_arm.append(v)
    if len(q_arm) < 2:
        print("[visualize] < 2 valid arm joint samples; skipping joint-jump plot")
        return
    pos_xyz = np.asarray(pos_xyz, dtype=np.float64)        # (N, 3)
    q_arm = np.asarray(q_arm, dtype=np.float64)            # (N, n_dof)

    # |Δq| L∞ aligned to each step. Step 0 has no predecessor → use NaN
    # (rendered as a hollow grey marker so it doesn't bias the colorbar).
    dq = np.abs(np.diff(q_arm, axis=0))                    # (N-1, n_dof)
    linf = np.concatenate([[np.nan], dq.max(axis=1)])       # (N,)

    meta = traj.get("metadata") or {}
    plane_origin = np.asarray(meta["plane_origin"], dtype=np.float64)
    plane_u = np.asarray(meta["plane_u_axis"], dtype=np.float64)
    plane_v = np.asarray(meta["plane_v_axis"], dtype=np.float64)
    polygon = np.asarray(meta.get("polygon_xyz", []), dtype=np.float64)

    pos_uv = _project_to_uv(pos_xyz, plane_origin, plane_u, plane_v)
    poly_uv = _project_to_uv(polygon, plane_origin, plane_u, plane_v)

    finite = np.isfinite(linf)
    vmax = float(np.nanmax(linf)) if finite.any() else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(11, 9))

    if poly_uv.size:
        loop = np.vstack([poly_uv, poly_uv[:1]])
        ax.plot(loop[:, 0], loop[:, 1], color="#1f77b4", linewidth=2.0, label="region polygon")

    if pos_uv.shape[0] > 1:
        ax.plot(pos_uv[:, 0], pos_uv[:, 1], color="#888888", linewidth=0.6, alpha=0.4, label="trajectory")

    # First step (no predecessor) — drawn separately so the colorbar is undisturbed.
    if (~finite).any():
        ax.scatter(
            pos_uv[~finite, 0], pos_uv[~finite, 1],
            facecolor="white", edgecolor="grey", s=22, linewidth=0.5,
            zorder=3, label="step 0 (no Δq)",
        )

    sc = ax.scatter(
        pos_uv[finite, 0], pos_uv[finite, 1],
        c=linf[finite], cmap="plasma",
        s=18, edgecolor="black", linewidth=0.25,
        vmin=0.0, vmax=vmax,
        zorder=4,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("|Δq|∞ between consecutive steps (rad)")
    if jump_threshold is not None and 0.0 < jump_threshold <= vmax:
        cb.ax.axhline(float(jump_threshold), color="red", linewidth=1.0)
        cb.ax.text(0.5, float(jump_threshold), f" jump_threshold={jump_threshold:.2f}",
                   color="red", fontsize=7, va="center", ha="left",
                   transform=cb.ax.get_yaxis_transform())

    rc_meta = meta.get("robot_check") or {}
    title_bits = [f"joint jump in (u,v) — {arm_key} ({len(q_arm)} steps)"]
    if finite.any():
        title_bits.append(f"max|Δq|∞={float(np.nanmax(linf)):.3f} rad")
    if "worst_joint_speed_ratio" in rc_meta:
        title_bits.append(
            f"worst_speed_ratio={rc_meta['worst_joint_speed_ratio']:.3f} (dof {rc_meta.get('worst_dof', '?')})"
        )
    ax.set_title("  |  ".join(title_bits))
    ax.set_xlabel("u (plane axis, m)")
    ax.set_ylabel("v (plane axis, m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


def render_ik_grid_png(trajectory_path: str, out_png: str) -> None:
    """Render the per-waypoint IK convergence grid as a binary PNG.

    Reads `<stem>_ik_pools.json` (referenced by metadata.robot_check.ik_pools_path
    on the trajectory JSON), extracts `ik_pool_statuses` (one flat list[bool]
    per waypoint, ordered: pitch outermost, then yaw, then sorted branch key,
    then redundant index), builds a 2D image with rows = candidate index,
    columns = waypoint id, and saves white=ok / black=not-ok.

    Shorter rows pad with False (black) so the grid is rectangular. Skips
    silently when the sidecar is missing or has no ik_pool_statuses block
    (e.g. save_all_ik_results was off).
    """
    import matplotlib.pyplot as plt

    with open(trajectory_path) as f:
        traj = json.load(f)

    meta = traj.get("metadata") or {}
    rc_meta = meta.get("robot_check") or {}
    sidecar_name = rc_meta.get("ik_pools_path")
    if not sidecar_name:
        print("[visualize] no ik_pools sidecar reference; skipping IK grid")
        return

    sidecar_path = Path(trajectory_path).with_name(sidecar_name)
    if not sidecar_path.exists():
        print(f"[visualize] ik_pools sidecar {sidecar_path} missing; skipping IK grid")
        return

    with open(sidecar_path) as f:
        ik_data = json.load(f)

    statuses = ik_data.get("ik_pool_statuses")
    if not statuses:
        print("[visualize] sidecar has no ik_pool_statuses; skipping IK grid")
        return

    n_wp = len(statuses)
    n_cand = max((len(s) for s in statuses), default=0)
    if n_wp == 0 or n_cand == 0:
        print("[visualize] empty ik_pool_statuses; skipping IK grid")
        return

    grid = np.zeros((n_cand, n_wp), dtype=np.uint8)
    for j, s in enumerate(statuses):
        for i, ok in enumerate(s):
            if ok:
                grid[i, j] = 1

    # Sized so each waypoint column maps to >= 1 figure pixel even for big N.
    fig_w = max(6.0, min(20.0, n_wp / 30.0))
    fig_h = max(4.0, min(24.0, n_cand / 80.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(
        grid,
        cmap="gray",
        vmin=0, vmax=1,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )

    # Overlay A* selected path: one (x=waypoint, y=flat-row) point per
    # waypoint, joined by a line. Skips silently if absent.
    selected = ik_data.get("ik_selected_flat_indices")
    has_path = (
        isinstance(selected, list)
        and len(selected) > 0
        and all(isinstance(v, (int, float)) for v in selected)
    )
    if has_path:
        sel_arr = np.asarray(selected, dtype=np.float64)
        xs = np.arange(min(len(sel_arr), n_wp), dtype=np.float64)
        ys = sel_arr[: len(xs)]
        ax.plot(xs, ys, color="#ff3366", linewidth=1.0, alpha=0.9, zorder=5,
                label="A* selected path")
        ax.scatter(xs, ys, s=10, facecolor="#ff3366", edgecolor="white",
                   linewidth=0.4, zorder=6)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    n_ok = int(grid.sum())
    n_total = n_wp * n_cand
    title_bits = [
        f"IK convergence grid — {n_wp} waypoints x {n_cand} candidates",
        f"{n_ok}/{n_total} ok ({100.0 * n_ok / max(n_total, 1):.1f}%)",
    ]
    if has_path:
        title_bits.append("A* path overlay (red)")
    ax.set_title("  |  ".join(title_bits))
    ax.set_xlabel("waypoint id")
    ax.set_ylabel("IK candidate index (pitch × yaw × branch × redundant)")
    fig.tight_layout()

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


def render_png(
    mesh_path: str,
    trajectory_path: str,
    out_png: str,
    frame_size: float = 0.03,
    frame_stride: int = 10,
    pool_sizes: np.ndarray | None = None,
    colorbar_label: str | None = None,
    cross_mask: np.ndarray | None = None,
) -> None:
    """Render a 2D trajectory PNG via matplotlib without opening a window.

    Plots the polygon and waypoints in the plane's local (u, v) frame, with
    each waypoint colored by its IK pool size. When invoked end-to-end from
    `plan.py`, `pool_sizes` is supplied directly from the in-memory Phase 1a
    pools (so the colorbar always renders, regardless of
    `save_all_ik_results`). When invoked standalone via the CLI, falls back
    to reading the heavy `ik_pools` block from the trajectory JSON (or the
    sibling `<stem>_ik_pools.json` sidecar written when
    `save_all_ik_results: true`). `mesh_path`, `frame_size`, `frame_stride`
    are kept for signature compatibility but unused in 2D mode.
    """
    del mesh_path, frame_size, frame_stride  # unused in 2D mode
    with open(trajectory_path) as f:
        traj = json.load(f)
    if "ik_pools" not in traj:
        # Look for the sidecar file written by plan.write_trajectory_json.
        sidecar = Path(trajectory_path).with_name(
            Path(trajectory_path).stem + "_ik_pools.json"
        )
        if sidecar.exists():
            with open(sidecar) as f:
                traj["ik_pools"] = json.load(f).get("ik_pools")
    positions, _quats, _surface_pts, pass_idx, kinds, polygon = _extract_arrays(traj)
    meta = traj["metadata"]
    plane_origin = np.asarray(meta["plane_origin"], dtype=np.float64)
    plane_u = np.asarray(meta["plane_u_axis"], dtype=np.float64)
    plane_v = np.asarray(meta["plane_v_axis"], dtype=np.float64)
    if pool_sizes is None:
        pool_sizes = _ik_pool_sizes(traj, len(positions))
    visualize_matplotlib_2d(
        positions=positions,
        polygon=polygon,
        plane_origin=plane_origin,
        plane_u_axis=plane_u,
        plane_v_axis=plane_v,
        pass_idx=pass_idx,
        pool_sizes=pool_sizes,
        out_png=out_png,
        show=False,
        kinds=kinds,
        colorbar_label=colorbar_label,
        cross_mask=cross_mask,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a surface-cover trajectory.")
    parser.add_argument("--mesh", required=True, help="path to mesh file (obj/glb/etc.)")
    parser.add_argument("--trajectory", required=True, help="path to trajectory JSON")
    parser.add_argument("--out", default=None, help="output PNG path (matplotlib backend)")
    parser.add_argument("--interactive", action="store_true", help="use open3d backend instead of matplotlib")
    parser.add_argument("--no-show", action="store_true", help="matplotlib: do not open a window, only save PNG")
    parser.add_argument("--frame-size", type=float, default=0.03, help="length of frame triad axes (m)")
    parser.add_argument("--frame-stride", type=int, default=10, help="draw a triad every Nth waypoint")
    args = parser.parse_args()

    mesh, traj = _load_inputs(args.mesh, args.trajectory)
    _apply_metadata_transform(mesh, traj["metadata"])
    positions, quats, surface_pts, pass_idx, _kinds, polygon = _extract_arrays(traj)

    if args.interactive:
        visualize_open3d(
            mesh=mesh,
            positions=positions,
            quats=quats,
            surface_pts=surface_pts,
            polygon=polygon,
            frame_size=args.frame_size,
            frame_stride=args.frame_stride,
        )
    else:
        out_png = args.out
        if out_png is None and args.no_show:
            out_png = str(Path(args.trajectory).with_suffix(".png"))
        visualize_matplotlib(
            mesh=mesh,
            positions=positions,
            quats=quats,
            surface_pts=surface_pts,
            pass_idx=pass_idx,
            polygon=polygon,
            out_png=out_png,
            show=not args.no_show,
            frame_size=args.frame_size,
            frame_stride=args.frame_stride,
        )


if __name__ == "__main__":
    main()
