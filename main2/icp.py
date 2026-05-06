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
from typing import Dict, Optional

import numpy as np
import open3d as o3d


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


def load_intrinsics(yaml_path: str) -> np.ndarray:
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
		o3d.pipelines.registration.TransformationEstimationPointToPoint(),
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
		help="Override intrinsics YAML path (takes precedence over --camera)",
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
		default=0.0,
		help="Voxel size for downsampling (0 to disable)",
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

	if args.intrinsics:
		intrinsics_path = args.intrinsics
	else:
		intrinsics_path = DEFAULT_INTRINSICS[args.camera]

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
	model = load_and_sample_stl(args.stl, num_points=args.sample_points, scale=args.model_scale)

	if args.voxel_size > 0:
		observed = observed.voxel_down_sample(args.voxel_size)
		model = model.voxel_down_sample(args.voxel_size)
		
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
	
	# 2. Apply suggested rotation: X +90 deg, Z -90 deg
	rx_90 = np.array([
		[1.0,  0.0,  0.0],
		[0.0,  0.0, -1.0],
		[0.0,  1.0,  0.0]
	])
	rz_n90 = np.array([
		[ 0.0,  1.0,  0.0],
		[-1.0,  0.0,  0.0],
		[ 0.0,  0.0,  1.0]
	])
	# Combine rotations: first X, then Z (R = Rz * Rx)
	r_init = rz_n90 @ rx_90
	
	# Apply initial rotation and translation
	# Using intermediate camera RGB pixel to get the spatial center,
	# and we subtract the rotated model's native center offset. 
	init_transform[:3, :3] = r_init
	init_transform[:3, 3] = estimated_obj_center_in_camera - (r_init @ mod_center)

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
