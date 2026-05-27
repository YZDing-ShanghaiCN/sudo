#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-run STL/depth point cloud viewer with a manual 6D pose."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
	import open3d as o3d
except Exception as exc:
	raise RuntimeError("open3d is required: pip install open3d") from exc


BASE_DIR = Path(__file__).resolve().parent
# DEPTH_PATH = BASE_DIR / "result" / "nearpose_left_chest_origin" / "depth_mean.tiff"
DEPTH_PATH = Path("/home/user/Desktop/main/main2/20260508/depth_all_new/nearpose_left_chest_origin/000000.tiff")
STL_PATH = BASE_DIR / "底盘.STL"
CAMERA_YAML = BASE_DIR.parent / "aililight_cameras" / "chest_left_camera.yaml"

DEPTH_STRIDE = 2
DEPTH_SCALE = 1.0
MESH_SCALE = 0.00005
SAMPLE_POINTS = 60000
WINDOW_NAME = "mod_icp"

# Edit these six values manually, then run once.
# Rotation order is X -> Y -> Z.
POSE_RX_DEG = 30
POSE_RY_DEG = 135.0
POSE_RZ_DEG = 180.0
POSE_TX_M = 0
POSE_TY_M = 0
POSE_TZ_M = 0.1


def parse_intrinsics_from_text(text: str) -> np.ndarray:
	lines = [line.strip() for line in text.splitlines()]
	start = None
	for index, line in enumerate(lines):
		if line.startswith("intrinsic:"):
			start = index + 1
			break
	if start is None:
		raise ValueError("Missing intrinsic section in camera YAML.")

	rows: list[list[float]] = []
	for line in lines[start:]:
		if line.startswith("- [") and line.endswith("]"):
			rows.append([float(value.strip()) for value in line[3:-1].split(",")])
			if len(rows) == 3:
				break
	if len(rows) != 3:
		raise ValueError("Failed to parse 3x3 intrinsic matrix.")
	return np.array(rows, dtype=np.float64)


def load_intrinsics(yaml_path: Path) -> np.ndarray:
	if not yaml_path.exists():
		raise FileNotFoundError(f"Camera intrinsics file not found: {yaml_path}")
	text = yaml_path.read_text(encoding="utf-8")
	try:
		import yaml  # type: ignore

		data = yaml.safe_load(text)
		matrix = np.array(data["intrinsic"], dtype=np.float64)
	except Exception:
		matrix = parse_intrinsics_from_text(text)
	if matrix.shape != (3, 3):
		raise ValueError(f"Invalid intrinsic matrix shape: {matrix.shape}")
	return matrix


def center_crop_intrinsics(k: np.ndarray, source_shape: tuple[int, int], crop_shape: tuple[int, int]) -> np.ndarray:
	source_height, source_width = source_shape
	crop_height, crop_width = crop_shape
	top = (source_height - crop_height) // 2
	left = (source_width - crop_width) // 2
	adjusted = np.array(k, dtype=np.float64, copy=True)
	adjusted[0, 2] -= left
	adjusted[1, 2] -= top
	return adjusted


def depth_to_colors(z_values: np.ndarray) -> np.ndarray:
	if z_values.size == 0:
		return np.empty((0, 3), dtype=np.float64)
	lower = float(np.percentile(z_values, 5))
	upper = float(np.percentile(z_values, 95))
	if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
		return np.full((z_values.shape[0], 3), 0.7, dtype=np.float64)
	normalized = np.clip((z_values - lower) / (upper - lower), 0.0, 1.0)
	red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
	green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
	blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
	return np.stack((red, green, blue), axis=1)


def build_xyz_rotation(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
	rx = np.deg2rad(float(rx_deg))
	ry = np.deg2rad(float(ry_deg))
	rz = np.deg2rad(float(rz_deg))

	cx, sx = float(np.cos(rx)), float(np.sin(rx))
	cy, sy = float(np.cos(ry)), float(np.sin(ry))
	cz, sz = float(np.cos(rz)), float(np.sin(rz))

	rot_x = np.array(
		[[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
		dtype=np.float64,
	)
	rot_y = np.array(
		[[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
		dtype=np.float64,
	)
	rot_z = np.array(
		[[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
		dtype=np.float64,
	)
	rot3 = rot_z @ rot_y @ rot_x
	rot = np.eye(4, dtype=np.float64)
	rot[:3, :3] = rot3
	return rot


def pose_to_transform(
	rx_deg: float,
	ry_deg: float,
	rz_deg: float,
	tx: float,
	ty: float,
	tz: float,
) -> np.ndarray:
	transform = build_xyz_rotation(rx_deg, ry_deg, rz_deg)
	transform[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
	return transform


def depth_to_points(depth: np.ndarray, k: np.ndarray, stride: int) -> np.ndarray:
	if depth.ndim == 3 and depth.shape[2] >= 1:
		depth = depth[..., 0]
	if depth.ndim != 2:
		raise ValueError(f"depth map must be a 2D array, got {depth.shape}")
	if stride < 1:
		raise ValueError("stride must be >= 1")

	sampled = np.asarray(depth, dtype=np.float64)[::stride, ::stride]
	valid = np.isfinite(sampled) & (sampled > 0.0)
	if not np.any(valid):
		return np.empty((0, 3), dtype=np.float64)

	ys, xs = np.indices(sampled.shape)
	xs = xs.astype(np.float64) * float(stride)
	ys = ys.astype(np.float64) * float(stride)
	z = sampled[valid]
	fx = float(k[0, 0])
	fy = float(k[1, 1])
	cx = float(k[0, 2])
	cy = float(k[1, 2])
	x = (xs[valid] - cx) * z / fx
	y = (ys[valid] - cy) * z / fy
	return np.stack((x, y, z), axis=1)


def load_depth_map(depth_path: Path) -> np.ndarray:
	suffix = depth_path.suffix.lower()
	if suffix == ".npy":
		depth = np.load(str(depth_path))
	elif suffix in {".tif", ".tiff"}:
		try:
			import tifffile  # type: ignore
		except Exception as exc:
			raise RuntimeError("tifffile is required to read TIFF depth maps: pip install tifffile") from exc
		depth = tifffile.imread(str(depth_path))
	else:
		raise ValueError(f"Unsupported depth file format: {depth_path.suffix}")

	depth = np.asarray(depth)
	if depth.ndim == 3:
		if depth.shape[-1] in (1, 3, 4):
			depth = depth[..., 0]
		else:
			depth = depth[0]
	if depth.ndim != 2:
		raise ValueError(f"Depth map must be 2D after loading, got {depth.shape} from {depth_path}")
	return depth


def load_depth_points() -> np.ndarray:
	depth = load_depth_map(DEPTH_PATH)
	k = center_crop_intrinsics(load_intrinsics(CAMERA_YAML), (800, 1280), depth.shape[:2])
	print(k)
	points = depth_to_points(depth, k, DEPTH_STRIDE)
	points = points * float(DEPTH_SCALE)
	mask = np.isfinite(points).all(axis=1)
	return points[mask]


def load_stl_points() -> np.ndarray:
	mesh = o3d.io.read_triangle_mesh(str(STL_PATH))
	if mesh.is_empty():
		raise RuntimeError(f"Failed to load STL: {STL_PATH}")
	if MESH_SCALE != 1.0:
		mesh.scale(MESH_SCALE, center=(0.0, 0.0, 0.0))
	pcd = mesh.sample_points_uniformly(number_of_points=SAMPLE_POINTS)
	return np.asarray(pcd.points, dtype=np.float64)


def make_point_cloud(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	if colors is not None and len(colors) == len(points):
		pcd.colors = o3d.utility.Vector3dVector(colors)
	return pcd


def point_cloud_center(points: np.ndarray) -> np.ndarray:
	if points.size == 0:
		raise ValueError("Cannot compute the center of an empty point cloud")
	return np.mean(points, axis=0)


def main() -> None:
	global_transform = pose_to_transform(
		POSE_RX_DEG,
		POSE_RY_DEG,
		POSE_RZ_DEG,
		POSE_TX_M,
		POSE_TY_M,
		POSE_TZ_M,
	)

	depth_points = load_depth_points()
	stl_points = load_stl_points()
	depth_center = point_cloud_center(depth_points)
	stl_center = point_cloud_center(stl_points)
	center_delta = depth_center - stl_center

	depth_pcd = make_point_cloud(depth_points, depth_to_colors(depth_points[:, 2]))
	stl_pcd = make_point_cloud(stl_points)
	stl_pcd.paint_uniform_color([0.95, 0.4, 0.1])
	stl_pcd.transform(global_transform)

	depth_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
	stl_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
	stl_frame.transform(global_transform)

	print("GLOBAL_POSE: rx_deg, ry_deg, rz_deg, tx_m, ty_m, tz_m")
	print(np.array([POSE_RX_DEG, POSE_RY_DEG, POSE_RZ_DEG, POSE_TX_M, POSE_TY_M, POSE_TZ_M], dtype=np.float64))
	print("GLOBAL_TRANSFORM:")
	print(np.array2string(global_transform, precision=6, suppress_small=True))
	print("AXIS COLORS: X=red, Y=green, Z=blue")
	print("DEPTH FRAME: at origin for the depth point cloud")
	print("STL FRAME: transformed with the manual pose")
	print("DEPTH CENTER (m):")
	print(np.array2string(depth_center, precision=6, suppress_small=True))
	print("STL CENTER (m):")
	print(np.array2string(stl_center, precision=6, suppress_small=True))
	print("CENTER ALIGN TRANSLATION, raw centroid delta (depth - stl, m):")
	print(np.array2string(center_delta, precision=6, suppress_small=True))
	rotation3 = global_transform[:3, :3]
	rotation_aware_translation = depth_center - rotation3 @ stl_center
	print("CENTER ALIGN TRANSLATION, keeping current rotation (m):")
	print(np.array2string(rotation_aware_translation, precision=6, suppress_small=True))
	print(f"depth points: {len(depth_points)}")
	print(f"stl points: {len(stl_points)}")

	vis = o3d.visualization.Visualizer()
	vis.create_window(window_name=WINDOW_NAME)
	opt = vis.get_render_option()
	opt.background_color = np.asarray([0.06, 0.06, 0.06], dtype=np.float64)
	opt.point_size = 2.0
	vis.add_geometry(depth_frame)
	vis.add_geometry(depth_pcd)
	vis.add_geometry(stl_pcd)
	vis.add_geometry(stl_frame)
	vis.run()
	vis.destroy_window()


if __name__ == "__main__":
	main()