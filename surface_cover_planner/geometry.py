"""Geometry helpers for the surface-cover planner.

All vectors are 3D numpy arrays in world frame unless noted. Quaternions are
[w, x, y, z] (sapien convention) to match the rest of SudoDeploy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError(f"cannot normalize near-zero vector {v!r}")
    return v / n


def build_mesh_transform(cfg: dict) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a `mesh.transform` config block.

    Order of composition: scale -> rotation -> translation.
    """
    translation = np.asarray(cfg.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    scale = float(cfg.get("scale", 1.0))
    mode = cfg.get("rotation_mode", "quat")

    if mode == "quat":
        q_wxyz = np.asarray(cfg.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        rot = R.from_quat(np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])).as_matrix()
    elif mode == "euler_xyz_deg":
        rpy = np.asarray(cfg.get("euler_xyz_deg", [0.0, 0.0, 0.0]), dtype=np.float64)
        rot = R.from_euler("XYZ", rpy, degrees=True).as_matrix()
    elif mode == "matrix":
        rot = np.asarray(cfg.get("matrix", np.eye(3).tolist()), dtype=np.float64)
        if rot.shape != (3, 3):
            raise ValueError(f"mesh.transform.matrix must be 3x3, got {rot.shape}")
    else:
        raise ValueError(f"unknown mesh.transform.rotation_mode: {mode!r}")

    T = np.eye(4)
    T[:3, :3] = rot * scale
    T[:3, 3] = translation
    return T


def build_tool_offset_transform(cfg: dict | None) -> np.ndarray:
    """Build a 4x4 rigid transform from a `tcp.tool_offset` config block.

    Same rotation_mode options as build_mesh_transform (quat / euler_xyz_deg /
    matrix). Returns identity when cfg is None or empty.
    """
    if cfg is None or not cfg:
        return np.eye(4, dtype=np.float64)

    translation = np.asarray(cfg.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    mode = cfg.get("rotation_mode", "quat")

    if mode == "quat":
        q_wxyz = np.asarray(cfg.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        rot = R.from_quat(np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])).as_matrix()
    elif mode == "euler_xyz_deg":
        rpy = np.asarray(cfg.get("euler_xyz_deg", [0.0, 0.0, 0.0]), dtype=np.float64)
        rot = R.from_euler("XYZ", rpy, degrees=True).as_matrix()
    elif mode == "matrix":
        rot = np.asarray(cfg.get("matrix", np.eye(3).tolist()), dtype=np.float64)
        if rot.shape != (3, 3):
            raise ValueError(f"tcp.tool_offset.matrix must be 3x3, got {rot.shape}")
    else:
        raise ValueError(f"unknown tcp.tool_offset.rotation_mode: {mode!r}")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = translation
    return T


@dataclass
class PlaneFrame:
    """Plane defined by an origin and an orthonormal (u, v, n) basis."""

    origin: np.ndarray  # (3,)
    u_axis: np.ndarray  # (3,)  in-plane
    v_axis: np.ndarray  # (3,)  in-plane, orthogonal to u
    normal: np.ndarray  # (3,)  perpendicular to (u, v)


def plane_from_polygon(polygon_xyz: np.ndarray) -> PlaneFrame:
    """Build a plane frame from a polygon's first three vertices.

    No SVD, no coplanarity check: the user has guaranteed the points are coplanar.
    """
    if polygon_xyz.shape[0] < 3:
        raise ValueError(f"polygon needs >=3 vertices, got {polygon_xyz.shape[0]}")
    p0, p1, p2 = polygon_xyz[0], polygon_xyz[1], polygon_xyz[2]
    u = _normalize(p1 - p0)
    n = _normalize(np.cross(p1 - p0, p2 - p0))
    v = _normalize(np.cross(n, u))
    return PlaneFrame(origin=p0.copy(), u_axis=u, v_axis=v, normal=n)


def project_to_plane_uv(points_xyz: np.ndarray, plane: PlaneFrame) -> np.ndarray:
    """Project (N, 3) world points to (N, 2) (u, v) coordinates in the plane frame."""
    rel = points_xyz - plane.origin
    u = rel @ plane.u_axis
    v = rel @ plane.v_axis
    return np.stack([u, v], axis=-1)


def unproject_from_plane_uv(points_uv: np.ndarray, plane: PlaneFrame) -> np.ndarray:
    """Lift (N, 2) (u, v) coordinates back to (N, 3) points on the plane."""
    return plane.origin[None, :] + points_uv[:, [0]] * plane.u_axis[None, :] + points_uv[:, [1]] * plane.v_axis[None, :]


def rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a [w, x, y, z] quaternion."""
    q_xyzw = R.from_matrix(rot).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert a [w, x, y, z] quaternion to a 3x3 rotation matrix."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    return R.from_quat(q_xyzw).as_matrix()


def tool_pose_to_tactile_pose(
    position: np.ndarray | list[float],
    quat_wxyz: np.ndarray | list[float] | None,
    tool_offset_4x4: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert a world-frame tool-tip pose to the tactile-center pose.

    Uses the identity `T_world_tool = T_world_tactile @ tool_offset_4x4`,
    so the inverse is `T_world_tactile = T_world_tool @ inv(tool_offset_4x4)`.

    - `tool_offset_4x4 is None` → tactile == tool; returns copies.
    - `quat_wxyz is None` → tactile quat is also None (sentinel rows where
      the achieved pose was never set); the position is returned unchanged
      for parity with how the rest of the schema treats zero-sentinel rows.
    """
    pos = np.asarray(position, dtype=np.float64)
    if quat_wxyz is None:
        return pos.copy(), None
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    if tool_offset_4x4 is None:
        return pos.copy(), quat.copy()

    tool_4x4 = np.eye(4, dtype=np.float64)
    tool_4x4[:3, :3] = quat_wxyz_to_rotmat(quat)
    tool_4x4[:3, 3] = pos
    tactile_4x4 = tool_4x4 @ np.linalg.inv(np.asarray(tool_offset_4x4, dtype=np.float64))
    return tactile_4x4[:3, 3].copy(), rotmat_to_quat_wxyz(tactile_4x4[:3, :3])


def build_align_z_to_normal_rotation(
    surface_normal_outward: np.ndarray,
    yaw_deg: float,
    yaw_reference_axis: np.ndarray,
) -> np.ndarray:
    """Build a 3x3 rotation matrix that aligns tool Z with -surface_normal_outward.

    Tool X is `yaw_reference_axis` projected onto the plane perpendicular to tool Z,
    then rotated by `yaw_deg` around tool Z. Tool Y = tool Z x tool X.

    If `yaw_reference_axis` is parallel to tool Z, falls back to the next world axis
    (X -> Y -> Z) until a non-degenerate reference is found.
    """
    n_out = _normalize(surface_normal_outward)
    tool_z = -n_out

    candidates = [yaw_reference_axis, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    ref_in_plane = None
    for cand in candidates:
        cand = np.asarray(cand, dtype=np.float64)
        proj = cand - np.dot(cand, tool_z) * tool_z
        if np.linalg.norm(proj) > 1e-6:
            ref_in_plane = _normalize(proj)
            break
    if ref_in_plane is None:
        raise RuntimeError("failed to find a non-degenerate yaw reference axis")

    yaw_rad = float(np.deg2rad(yaw_deg))
    cos_y, sin_y = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    tool_x = cos_y * ref_in_plane + sin_y * np.cross(tool_z, ref_in_plane)
    tool_x = _normalize(tool_x)
    tool_y = _normalize(np.cross(tool_z, tool_x))

    return np.stack([tool_x, tool_y, tool_z], axis=1)


def build_tcp_pose(
    surface_point: np.ndarray,
    surface_normal_outward: np.ndarray,
    standoff: float,
    rotation_mode: str,
    fixed_quat_wxyz: np.ndarray,
    yaw_deg: float,
    yaw_reference_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (position, quaternion_wxyz) for a single waypoint.

    `surface_normal_outward` should already be the post-flip outward normal.
    """
    n_out = _normalize(surface_normal_outward)
    position = surface_point + standoff * n_out

    if rotation_mode == "fixed_quat":
        q_wxyz = np.asarray(fixed_quat_wxyz, dtype=np.float64)
        q_wxyz = q_wxyz / float(np.linalg.norm(q_wxyz))
        return position, q_wxyz
    if rotation_mode == "align_z_to_normal":
        rot = build_align_z_to_normal_rotation(n_out, yaw_deg, yaw_reference_axis)
        return position, rotmat_to_quat_wxyz(rot)
    raise ValueError(f"unknown tcp.rotation_mode: {rotation_mode!r}")
