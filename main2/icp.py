#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICP alignment between an observed depth point cloud and an STL model.

Example:
	python icp.py \
		--depth /home/user/Desktop/main/main2/near_pose/output/000000_depth_pred0.exr \
		--stl /home/user/Desktop/main/main2/add/底盘.STL \
		--camera left
"""

from __future__ import annotations

import argparse
import os
import json
from typing import Dict, Optional

import numpy as np
import open3d as o3d
import imageio.v3


DEFAULT_INTRINSICS: Dict[str, str] = {
	"center": "/home/user/Desktop/main/main2/aililight_cameras/left_hand_center_camera_20260423.yaml",
	"left": "/home/user/Desktop/main/main2/aililight_cameras/left_hand_left_camera_20260423.yaml",
	"right": "/home/user/Desktop/main/main2/aililight_cameras/left_hand_right_camera_20260423.yaml",
}

DEFAULT_DEPTH = "/home/user/Desktop/main/main2/near_pose/output/000000_depth_pred0.exr"
DEFAULT_STL = "/home/user/Desktop/main/main2/add/底盘.STL"


def read_depth_exr(path: str) -> np.ndarray:
	"""Read a depth EXR image into a float32 numpy array."""
	try:
		import OpenEXR  # type: ignore
		import Imath  # type: ignore

		exr = OpenEXR.InputFile(path)
		header = exr.header()
		dw = header["dataWindow"]
		width = dw.max.x - dw.min.x + 1
		height = dw.max.y - dw.min.y + 1
		channels = list(header["channels"].keys())
		if "Z" in channels:
			channel = "Z"
		elif "R" in channels:
			channel = "R"
		else:
			channel = channels[0]
		pt = Imath.PixelType(Imath.PixelType.FLOAT)
		raw = exr.channel(channel, pt)
		depth = np.frombuffer(raw, dtype=np.float32).reshape((height, width))
		return depth
	except Exception:
		try:
			import imageio.v3 as iio  # type: ignore

			depth = iio.imread(path)
			if depth.ndim == 3:
				depth = depth[..., 0]
			return depth.astype(np.float32)
		except Exception as exc:
			raise RuntimeError(
				"Failed to read EXR depth. Install OpenEXR or imageio with EXR support."
			) from exc


def _parse_intrinsic_from_text(text: str) -> np.ndarray:
	lines = [line.strip() for line in text.splitlines()]
	start = None
	for i, line in enumerate(lines):
		if line.startswith("intrinsic:"):
			start = i + 1
			break
	if start is None:
		raise ValueError("Missing 'intrinsic' section in YAML.")

	rows = []
	for line in lines[start:]:
		if line.startswith("- [") and line.endswith("]"):
			row_text = line[3:-1]
			row = [float(x.strip()) for x in row_text.split(",")]
			rows.append(row)
			if len(rows) == 3:
				break
	if len(rows) != 3:
		raise ValueError("Failed to parse 3x3 intrinsic matrix.")
	return np.array(rows, dtype=np.float64)


def load_intrinsics_from_yaml(yaml_path: str) -> np.ndarray:
	with open(yaml_path, "r", encoding="utf-8") as f:
		text = f.read()

	try:
		import yaml  # type: ignore

		data = yaml.safe_load(text)
		k = np.array(data["intrinsic"], dtype=np.float64)
	except Exception:
		# Fallback parser for a simple YAML list format.
		k = _parse_intrinsic_from_text(text)

	if k.shape != (3, 3):
		raise ValueError(f"Invalid intrinsic matrix shape: {k.shape}")
	return k


def load_intrinsics_from_json(json_path: str) -> np.ndarray:
	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)
	k = np.array(data["intrinsic"], dtype=np.float64)
	if k.shape != (3, 3):
		raise ValueError(f"Invalid intrinsic matrix shape: {k.shape}")
	return k


def load_intrinsics(path: str) -> np.ndarray:
	ext = os.path.splitext(path)[1].lower()
	if ext == ".json":
		return load_intrinsics_from_json(path)
	return load_intrinsics_from_yaml(path)


def depth_to_points(
	depth: np.ndarray,
	k: np.ndarray,
	stride: int = 1,
	depth_scale: float = 1.0,
	depth_trunc: Optional[float] = None,
) -> np.ndarray:
	if stride < 1:
		raise ValueError("stride must be >= 1")

	depth = depth.astype(np.float32)
	if stride > 1:
		depth = depth[::stride, ::stride]

	h, w = depth.shape
	v, u = np.indices((h, w))
	u = u.astype(np.float32) * float(stride)
	v = v.astype(np.float32) * float(stride)

	z = depth * float(depth_scale)
	if depth_trunc is None:
		mask = (z > 0.0) & np.isfinite(z)
	else:
		mask = (z > 0.0) & np.isfinite(z) & (z < float(depth_trunc))

	z = z[mask]
	u = u[mask]
	v = v[mask]

	fx = k[0, 0]
	fy = k[1, 1]
	cx = k[0, 2]
	cy = k[1, 2]

	x = (u - cx) * z / fx
	y = (v - cy) * z / fy
	points = np.stack((x, y, z), axis=1)
	return points


def make_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	return pcd


def load_and_sample_stl(stl_path: str, num_points: int = 10000, scale: float = 0.001) -> o3d.geometry.PointCloud:
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load STL: {stl_path}")
    mesh.scale(scale, center=(0, 0, 0))
    mesh.compute_vertex_normals()
    return mesh.sample_points_uniformly(number_of_points=num_points)


def resolve_intrinsics_path(depth_path: str, camera: str, override: str) -> str:
	if override:
		return override

	depth_dir = os.path.dirname(depth_path)
	stem = os.path.splitext(os.path.basename(depth_path))[0]
	frame_id = stem.split("_")[0]
	candidates = [
		os.path.join(depth_dir, f"{stem}_intrinsic.json"),
		os.path.join(depth_dir, f"{frame_id}_{camera}_intrinsic.json"),
	]
	for candidate in candidates:
		if os.path.exists(candidate):
			return candidate
	return DEFAULT_INTRINSICS[camera]


def rotation_matrix_from_xyz_deg(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
	rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
	cx, sx = np.cos(rx), np.sin(rx)
	cy, sy = np.cos(ry), np.sin(ry)
	cz, sz = np.cos(rz), np.sin(rz)

	rx_m = np.array(
		[
			[1.0, 0.0, 0.0],
			[0.0, cx, -sx],
			[0.0, sx, cx],
		],
		dtype=np.float64,
	)
	ry_m = np.array(
		[
			[cy, 0.0, sy],
			[0.0, 1.0, 0.0],
			[-sy, 0.0, cy],
		],
		dtype=np.float64,
	)
	rz_m = np.array(
		[
			[cz, -sz, 0.0],
			[sz, cz, 0.0],
			[0.0, 0.0, 1.0],
		],
		dtype=np.float64,
	)
	return rz_m @ ry_m @ rx_m


def parse_init_rotation(text: str) -> np.ndarray:
	parts = [p.strip() for p in text.split(",") if p.strip()]
	if len(parts) != 3:
		raise ValueError("--init-rotation must be 'rx,ry,rz' in degrees")
	rx_deg, ry_deg, rz_deg = (float(p) for p in parts)
	return rotation_matrix_from_xyz_deg(rx_deg, ry_deg, rz_deg)


def remove_outliers(
	pcd: o3d.geometry.PointCloud,
	method: str,
	sor_nb_neighbors: int,
	sor_std_ratio: float,
	ror_nb_points: int,
	ror_radius: float,
) -> o3d.geometry.PointCloud:
	if len(pcd.points) == 0:
		return pcd
	if method == "sor":
		filtered, _ = pcd.remove_statistical_outlier(
			nb_neighbors=int(sor_nb_neighbors),
			std_ratio=float(sor_std_ratio),
		)
		return filtered
	if method == "ror":
		filtered, _ = pcd.remove_radius_outlier(
			nb_points=int(ror_nb_points),
			radius=float(ror_radius),
		)
		return filtered
	return pcd


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int) -> None:
	if len(pcd.points) == 0:
		return
	pcd.estimate_normals(
		o3d.geometry.KDTreeSearchParamHybrid(
			radius=float(radius),
			max_nn=int(max_nn),
		)
	)
	pcd.normalize_normals()


def compute_fpfh(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.pipelines.registration.Feature:
	search_param = o3d.geometry.KDTreeSearchParamHybrid(
		radius=float(voxel_size) * 5.0,
		max_nn=100,
	)
	return o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param)


def run_global_registration(
	source: o3d.geometry.PointCloud,
	target: o3d.geometry.PointCloud,
	voxel_size: float,
) -> o3d.pipelines.registration.RegistrationResult:
	if voxel_size <= 0:
		raise ValueError("voxel_size must be > 0 for global registration")

	source_fpfh = compute_fpfh(source, voxel_size)
	target_fpfh = compute_fpfh(target, voxel_size)
	distance_threshold = float(voxel_size) * 1.5
	result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
		source,
		target,
		source_fpfh,
		target_fpfh,
		True,
		distance_threshold,
		o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
		4,
		[
			o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
			o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
		],
		o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
	)
	return result


def run_icp(
	source: o3d.geometry.PointCloud,
	target: o3d.geometry.PointCloud,
	threshold: float = 0.02,
	max_iter: int = 80,
	init: Optional[np.ndarray] = None,
) -> o3d.pipelines.registration.RegistrationResult:
	if init is None:
		init = np.eye(4, dtype=np.float64)

	criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
		max_iteration=int(max_iter)
	)
	result = o3d.pipelines.registration.registration_icp(
		source,
		target,
		threshold,
		init,
		o3d.pipelines.registration.TransformationEstimationPointToPlane(),
		criteria,
	)
	return result


def main() -> None:
	parser = argparse.ArgumentParser(description="Estimate pose by ICP alignment.")
	parser.add_argument("--depth", default=DEFAULT_DEPTH, help="Path to depth EXR")
	parser.add_argument("--stl", default=DEFAULT_STL, help="Path to STL model")
	parser.add_argument(
		"--camera",
		default="left",
		choices=sorted(DEFAULT_INTRINSICS.keys()),
		help="Which camera intrinsics to use (left is the stereo reference)",
	)
	parser.add_argument(
		"--intrinsics",
		default="",
		help="Override intrinsics path (JSON or YAML, takes precedence over --camera)",
	)
	parser.add_argument("--stride", type=int, default=2, help="Pixel stride")
	parser.add_argument(
		"--depth-scale",
		type=float,
		default=1.0,
		help="Scale factor to convert depth units to meters",
	)
	parser.add_argument(
		"--model-scale",
		type=float,
		default=0.001,
		help="Scale factor to convert STL model units (e.g., mm to m)",
	)
	parser.add_argument(
		"--depth-trunc",
		type=float,
		default=0.0,
		help="Discard depths beyond this range (0 to disable)",
	)
	parser.add_argument(
		"--sample-points",
		type=int,
		default=10000,
		help="Number of points to sample from the STL mesh",
	)
	parser.add_argument(
		"--icp-threshold",
		type=float,
		default=0.01,
		help="ICP distance threshold",
	)
	parser.add_argument(
		"--voxel-size",
		type=float,
		default=0.005,
		help="Voxel size for downsampling (0 to disable)",
	)
	parser.add_argument(
		"--outlier-removal",
		choices=["none", "sor", "ror"],
		default="sor",
		help="Outlier removal for observed point cloud",
	)
	parser.add_argument(
		"--sor-nb-neighbors",
		type=int,
		default=20,
		help="SOR: number of neighbors",
	)
	parser.add_argument(
		"--sor-std-ratio",
		type=float,
		default=2.0,
		help="SOR: standard deviation ratio",
	)
	parser.add_argument(
		"--ror-nb-points",
		type=int,
		default=16,
		help="ROR: minimum number of neighbors",
	)
	parser.add_argument(
		"--ror-radius",
		type=float,
		default=0.02,
		help="ROR: radius",
	)
	parser.add_argument(
		"--normal-radius",
		type=float,
		default=0.02,
		help="Normal estimation radius",
	)
	parser.add_argument(
		"--normal-max-nn",
		type=int,
		default=30,
		help="Normal estimation max neighbors",
	)
	parser.add_argument(
		"--init-rotation",
		type=str,
		default="0,0,0",
		help="Initial rotation in degrees as 'rx,ry,rz' (XYZ order)",
	)
	parser.add_argument(
		"--use-global-registration",
		action="store_true",
		help="Use FPFH + RANSAC global registration before ICP",
	)
	parser.add_argument(
		"--center-pixel",
		type=str,
		default="640,400",
		help="Center pixel of object in middle camera 'u,v', default is 640,400 (center of 1280x800)",
	)
	parser.add_argument(
		"--center-depth",
		type=float,
		default=0.0,
		help="Depth (Z) of that center pixel in middle camera (in meters)",
	)
	parser.add_argument(
		"--max-iter",
		type=int,
		default=100,
		help="Maximum number of ICP iterations",
	)

	args = parser.parse_args()

	intrinsics_path = resolve_intrinsics_path(args.depth, args.camera, args.intrinsics)

	if not os.path.exists(args.depth):
		raise FileNotFoundError(f"Depth file not found: {args.depth}")
	if not os.path.exists(args.stl):
		raise FileNotFoundError(f"STL file not found: {args.stl}")
	if not os.path.exists(intrinsics_path):
		raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")

	k = load_intrinsics(intrinsics_path)
	depth = read_depth_exr(args.depth)

	depth_trunc = None if args.depth_trunc <= 0 else float(args.depth_trunc)
	points = depth_to_points(
		depth, k, stride=args.stride, depth_scale=args.depth_scale, depth_trunc=depth_trunc
	)
	observed = make_point_cloud(points)
	if args.outlier_removal != "none":
		observed = remove_outliers(
			observed,
			method=args.outlier_removal,
			sor_nb_neighbors=args.sor_nb_neighbors,
			sor_std_ratio=args.sor_std_ratio,
			ror_nb_points=args.ror_nb_points,
			ror_radius=args.ror_radius,
		)
	model = load_and_sample_stl(args.stl, num_points=args.sample_points, scale=args.model_scale)

	if args.voxel_size > 0:
		observed = observed.voxel_down_sample(args.voxel_size)
		model = model.voxel_down_sample(args.voxel_size)

	normal_radius = float(args.normal_radius)
	if normal_radius <= 0:
		normal_radius = max(float(args.voxel_size) * 2.5, 0.01)
	estimate_normals(observed, normal_radius, args.normal_max_nn)
	estimate_normals(model, normal_radius, args.normal_max_nn)
		
	# Method for initial guess:
	# 1. User specified estimated translation (e.g. from Center camera projection)
	# currently we approximate with observed point cloud center, but you can plug global center here.
	obs_center = observed.get_center()

	if args.center_pixel and args.center_depth > 0:
		u_str, v_str = args.center_pixel.split(",")
		u, v = float(u_str), float(v_str)
		# Load center camera intrinsics
		k_center = load_intrinsics(DEFAULT_INTRINSICS["center"])
		fx_c = k_center[0, 0]
		fy_c = k_center[1, 1]
		cx_c = k_center[0, 2]
		cy_c = k_center[1, 2]
		z_c = args.center_depth
		x_c = (u - cx_c) * z_c / fx_c
		y_c = (v - cy_c) * z_c / fy_c
		estimated_obj_center_in_camera = np.array([x_c, y_c, z_c])
		print(f"Using estimated center from Center Camera RGB pixel: {estimated_obj_center_in_camera}")
	else:
		estimated_obj_center_in_camera = obs_center

	mod_center = model.get_center()
	init_transform = np.eye(4, dtype=np.float64)

	r_init = parse_init_rotation(args.init_rotation)
	init_transform[:3, :3] = r_init
	init_transform[:3, 3] = estimated_obj_center_in_camera - (r_init @ mod_center)

	if args.use_global_registration:
		global_voxel = args.voxel_size if args.voxel_size > 0 else 0.005
		global_result = run_global_registration(model, observed, global_voxel)
		init_transform = global_result.transformation

	result = run_icp(
		source=model,
		target=observed,
		threshold=float(args.icp_threshold),
		max_iter=int(args.max_iter),
		init=init_transform
	)

	tform = result.transformation
	r = tform[:3, :3]
	t = tform[:3, 3]

	np.set_printoptions(precision=6, suppress=True)
	print("Transformation (4x4):")
	print(tform)
	print("\nR (3x3):")
	print(r)
	print("\nt (3,):")
	print(t)
	print(f"\nICP fitness (overlapping ratio of model points): {result.fitness:.4f}")
	print(f"ICP inlier_rmse (avg distance of overlapping points): {result.inlier_rmse:.4f}")


if __name__ == "__main__":
	main()
