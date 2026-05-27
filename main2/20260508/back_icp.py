#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize STL -> depth alignment and project the transformed STL to RGB."""

from __future__ import annotations

import argparse
import csv
import copy
import os
import re
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

try:
	import open3d as o3d
except Exception as exc:
	raise RuntimeError("open3d is required: pip install open3d") from exc

try:
	import cv2
except Exception as exc:
	raise RuntimeError("opencv-python is required: pip install opencv-python") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
RGB_ORIGINAL_SHAPE = (800, 1280)
RGB_CROP_SHAPE = (400, 640)
OPEN3D_WINDOW_NAME = "back_icp"

DEFAULT_STL_PATH = SCRIPT_DIR / "底盘.STL"
DEFAULT_DEPTH_PATH = SCRIPT_DIR / "result" / "nearpose_left_chest_origin" / "depth_mean.npy"
DEFAULT_CSV_PATH = SCRIPT_DIR / "0512" / "near_left_chest_origin.csv"
DEFAULT_RGB_PATH = SCRIPT_DIR / "rgb_new" / "nearpose_left_chest_origin" / "000000.png"
DEFAULT_CAMERA_YAML = SCRIPT_DIR.parent / "aililight_cameras" / "chest_left_camera.yaml"
DEFAULT_SAVE_DIR = SCRIPT_DIR / "temp" / "back_icp_view"


def parse_intrinsics_from_text(text: str) -> np.ndarray:
	lines = [line.strip() for line in text.splitlines()]
	start = None
	for index, line in enumerate(lines):
		if line.startswith("intrinsic:"):
			start = index + 1
			break
	if start is None:
		raise ValueError("Missing intrinsic section in camera YAML.")

	rows = []
	for line in lines[start:]:
		if line.startswith("- [") and line.endswith("]"):
			row_text = line[3:-1]
			row = [float(value.strip()) for value in row_text.split(",")]
			rows.append(row)
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


def center_crop_intrinsics(
	k: np.ndarray,
	source_shape: tuple[int, int],
	crop_shape: tuple[int, int],
) -> np.ndarray:
	source_height, source_width = source_shape
	crop_height, crop_width = crop_shape
	top = (source_height - crop_height) // 2
	left = (source_width - crop_width) // 2
	adjusted = np.array(k, dtype=np.float64, copy=True)
	adjusted[0, 2] -= left
	adjusted[1, 2] -= top
	return adjusted


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


def load_transform_from_csv(csv_path: Path, rank: int) -> np.ndarray:
	if not csv_path.exists():
		raise FileNotFoundError(f"Transform CSV not found: {csv_path}")

	# First try: parse the ICP CSV format (angles + translation).
	try:
		with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
			reader = csv.DictReader(f)
			if reader.fieldnames and "收敛结束_rx_度" in reader.fieldnames:
				rows = list(reader)
				if not rows:
					raise ValueError("CSV has no data rows.")
				chosen = None
				for row in rows:
					if str(row.get("排序", "")).strip() == str(rank):
						chosen = row
						break
				if chosen is None:
					chosen = rows[0]

				rx = float(chosen["收敛结束_rx_度"])
				ry = float(chosen["收敛结束_ry_度"])
				rz = float(chosen["收敛结束_rz_度"])
				tx = float(chosen["最终平移_x_米"])
				ty = float(chosen["最终平移_y_米"])
				tz = float(chosen["最终平移_z_米"])

				t = build_xyz_rotation(rx, ry, rz)
				t[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
				return t
	except Exception:
		pass

	# Fallback: parse any 16 numbers as a 4x4 matrix.
	text = csv_path.read_text(encoding="utf-8", errors="ignore")
	floats = []
	for token in text.replace("[", " ").replace("]", " ").replace(",", " ").split():
		try:
			floats.append(float(token))
		except Exception:
			continue
	if len(floats) >= 16:
		mat = np.array(floats[:16], dtype=np.float64).reshape((4, 4))
		return mat
	raise ValueError("Failed to parse transform from CSV.")


def depth_to_colors(z_values: np.ndarray) -> np.ndarray:
	if z_values.size == 0:
		return np.empty((0, 3), dtype=np.float64)
	lower = float(np.percentile(z_values, 5))
	upper = float(np.percentile(z_values, 95))
	if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
		return np.full((z_values.shape[0], 3), 0.8, dtype=np.float64)
	normalized = np.clip((z_values - lower) / (upper - lower), 0.0, 1.0)
	red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
	green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
	blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
	return np.stack((red, green, blue), axis=1)


def depth_to_point_cloud(depth: np.ndarray, k: np.ndarray, stride: int) -> np.ndarray:
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


def load_points(
	np_path: Path,
	k: np.ndarray,
	input_mode: str,
	depth_stride: int,
) -> np.ndarray:
	pts = np.load(str(np_path))
	mode = input_mode
	if mode not in {"auto", "points", "depth"}:
		raise ValueError(f"Unsupported input_mode: {mode}")

	if mode == "auto":
		if pts.ndim == 1:
			mode = "points"
		elif pts.ndim == 2:
			mode = "points" if pts.shape[1] in (3, 4) else "depth"
		elif pts.ndim == 3:
			mode = "points" if pts.shape[2] >= 3 else "depth"
		else:
			raise ValueError(f"Unsupported numpy shape for points: {pts.shape}")

	if mode == "depth":
		if pts.ndim == 3 and pts.shape[2] >= 1:
			pts = pts[..., 0]
		if pts.ndim != 2:
			raise ValueError(f"Depth input must be HxW array. Got {pts.shape}")
		pts = depth_to_point_cloud(pts, k, depth_stride)
	elif mode == "points":
		if pts.ndim == 1:
			if pts.size % 3 == 0:
				pts = pts.reshape((-1, 3))
			else:
				raise ValueError(f"Unsupported numpy shape for points: {pts.shape}")
		elif pts.ndim == 2 and pts.shape[1] >= 3:
			pts = pts[:, :3]
		elif pts.ndim == 3 and pts.shape[2] >= 3:
			pts = pts.reshape((-1, pts.shape[2]))[:, :3]
		else:
			raise ValueError(f"Point array must have shape (N,>=3). Got {pts.shape}")

	mask = np.isfinite(pts).all(axis=1)
	return pts[mask]


def project_points(points: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	pts = np.asarray(points, dtype=np.float64)
	if pts.size == 0:
		return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=bool)
	z = pts[:, 2]
	valid = np.isfinite(z) & (z > 0.0)
	if not np.any(valid):
		return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64), valid
	x = pts[valid, 0]
	y = pts[valid, 1]
	z = z[valid]
	fx = float(k[0, 0])
	fy = float(k[1, 1])
	cx = float(k[0, 2])
	cy = float(k[1, 2])
	u = fx * x / z + cx
	v = fy * y / z + cy
	return u, v, valid


def overlay_points_on_image(
	image: np.ndarray,
	u: np.ndarray,
	v: np.ndarray,
	colors_bgr: np.ndarray,
	alpha: float,
) -> tuple[np.ndarray, int]:
	overlay = image.copy()
	if u.size == 0:
		return overlay, 0
	h, w = overlay.shape[:2]
	u_i = np.rint(u).astype(np.int64)
	v_i = np.rint(v).astype(np.int64)
	in_bounds = (u_i >= 0) & (u_i < w) & (v_i >= 0) & (v_i < h)
	if not np.any(in_bounds):
		return overlay, 0
	u_i = u_i[in_bounds]
	v_i = v_i[in_bounds]
	colors = colors_bgr[in_bounds]
	base = overlay[v_i, u_i].astype(np.float64)
	blended = (1.0 - alpha) * base + alpha * colors
	overlay[v_i, u_i] = blended.astype(np.uint8)
	return overlay, int(u_i.size)


def format_metric_value(value: float | None, digits: int = 5, percent: bool = False) -> str:
	if value is None or not np.isfinite(value):
		return "n/a"
	if percent:
		return f"{value:.{digits}f} ({value * 100.0:.2f}%)"
	return f"{value:.{digits}f}"


def scale_points(points: np.ndarray, scale: float) -> np.ndarray:
	if scale == 1.0:
		return points
	return points * float(scale)


def estimate_extent(points: np.ndarray) -> float:
	if points.size == 0:
		return 0.0
	extent = points.max(axis=0) - points.min(axis=0)
	return float(np.linalg.norm(extent))


def resolve_path(base: Path, path_like: str) -> Path:
	path = Path(path_like)
	return path if path.is_absolute() else (base / path)


def main() -> None:
	parser = argparse.ArgumentParser(description="Apply ICP transform to STL and project onto RGB.")
	parser.add_argument("--base-dir", default=str(SCRIPT_DIR), help="base directory for relative paths")
	parser.add_argument("--stl-file", default="底盘.STL", help="STL mesh (relative to base-dir)")
	parser.add_argument(
		"--depth-file",
		default="result/farpose_left_chest_origin/depth_mean.npy",
		help="depth npy (relative to base-dir)",
	)
	parser.add_argument(
		"--transform-csv",
		default="0512/near_left_chest_origin.csv",
		help="CSV that contains ICP transformation (relative to base-dir)",
	)
	parser.add_argument("--csv-rank", type=int, default=1, help="CSV row rank to use (default: 排序=1)")
	parser.add_argument(
		"--rgb-file",
		default="rgb_new/nearpose_left_chest_origin/000000.png",
		help="RGB image path (relative to base-dir)",
	)
	parser.add_argument(
		"--camera-yaml",
		default=str(SCRIPT_DIR.parent / "aililight_cameras" / "chest_left_camera.yaml"),
		help="camera intrinsics yaml",
	)
	parser.add_argument("--input-mode", choices=["auto", "points", "depth"], default="auto")
	parser.add_argument("--depth-stride", type=int, default=2, help="pixel stride for depth sampling")
	parser.add_argument("--mesh-units", choices=["m", "mm"], default="mm")
	parser.add_argument("--depth-units", choices=["m", "mm"], default="m")
	parser.add_argument("--sample-points", type=int, default=30000, help="points for overlay and metrics")
	parser.add_argument("--overlay-alpha", type=float, default=0.75)
	parser.add_argument(
		"--overlay-out",
		default="0512/overlay_back_icp.png",
		help="output overlay image (relative to base-dir)",
	)
	parser.add_argument("--no-3d", action="store_true", help="skip Open3D visualization")
	parser.add_argument("--no-rgb", action="store_true", help="skip RGB overlay visualization")
	args = parser.parse_args()

	base = Path(args.base_dir)
	stl_path = resolve_path(base, args.stl_file)
	depth_path = resolve_path(base, args.depth_file)
	csv_path = resolve_path(base, args.transform_csv)
	rgb_path = resolve_path(base, args.rgb_file)
	camera_yaml = resolve_path(base, args.camera_yaml)

	if not stl_path.exists():
		raise FileNotFoundError(f"STL not found: {stl_path}")
	if not depth_path.exists():
		raise FileNotFoundError(f"Depth npy not found: {depth_path}")
	if not csv_path.exists():
		raise FileNotFoundError(f"Transform CSV not found: {csv_path}")
	if not rgb_path.exists():
		raise FileNotFoundError(f"RGB image not found: {rgb_path}")
	if not camera_yaml.exists():
		raise FileNotFoundError(f"Camera intrinsics not found: {camera_yaml}")

	transform = load_transform_from_csv(csv_path, args.csv_rank)
	print("[INFO] Transformation (STL -> depth/camera):")
	print(np.array2string(transform, precision=6, suppress_small=True))
	print(
		"[NOTE] Projection assumes OpenCV camera coords: +X right, +Y down, +Z forward. "
		"If overlay is flipped/rotated, check axis conventions and cropping."
	)
	print(
		"[NOTE] Most common misalignment causes: using inverse T, wrong units (mm vs m), "
		"or mismatched depth/RGB task folders."
	)

	k_full = load_intrinsics(camera_yaml)

	depth_arr = np.load(str(depth_path))
	if depth_arr.ndim == 3 and depth_arr.shape[2] >= 1:
		depth_arr = depth_arr[..., 0]
	if depth_arr.ndim == 2:
		depth_k = center_crop_intrinsics(k_full, RGB_ORIGINAL_SHAPE, depth_arr.shape[:2])
	else:
		depth_k = k_full

	points = load_points(depth_path, depth_k, args.input_mode, args.depth_stride)
	if args.depth_units == "mm":
		points = scale_points(points, 0.001)

	if points.size == 0:
		raise RuntimeError("No valid depth points loaded.")

	depth_extent = estimate_extent(points)
	depth_z = points[:, 2]
	print(f"[INFO] Depth points: {points.shape[0]} pts, z-range=({depth_z.min():.4f}, {depth_z.max():.4f})")

	mesh = o3d.io.read_triangle_mesh(str(stl_path))
	if mesh.is_empty():
		raise RuntimeError(f"Failed to load STL mesh: {stl_path}")
	mesh.compute_vertex_normals()
	if args.mesh_units == "mm":
		mesh.scale(0.001, center=(0.0, 0.0, 0.0))

	mesh_points = np.asarray(mesh.vertices)
	mesh_extent = estimate_extent(mesh_points) if mesh_points.size > 0 else 0.0
	print(f"[INFO] STL extent (after unit scaling): {mesh_extent:.4f} (meters)")

	if mesh_extent > 0.0 and depth_extent > 0.0:
		ratio = mesh_extent / max(depth_extent, 1e-9)
		if ratio > 10.0 or ratio < 0.1:
			print(
				"[WARN] Mesh/depth scale looks mismatched. "
				"Check --mesh-units and --depth-units.",
			)

	# Do NOT invert the transformation: ICP already maps STL -> depth/camera.
	mesh.transform(transform)

	depth_colors = depth_to_colors(points[:, 2])
	depth_pcd = o3d.geometry.PointCloud()
	depth_pcd.points = o3d.utility.Vector3dVector(points)
	depth_pcd.colors = o3d.utility.Vector3dVector(depth_colors)

	mesh.paint_uniform_color([0.9, 0.35, 0.1])

	cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
	stl_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07)
	stl_frame.transform(transform)

	if not args.no_3d:
		o3d.visualization.draw_geometries(
			[depth_pcd, mesh, cam_frame, stl_frame],
			window_name="Transformed STL + depth (camera coords)",
		)

	# Project transformed STL to RGB.
	if not args.no_rgb:
		rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
		if rgb is None:
			raise RuntimeError(f"Failed to load RGB image: {rgb_path}")
		k_rgb = center_crop_intrinsics(k_full, RGB_ORIGINAL_SHAPE, rgb.shape[:2])

		mesh_sample = mesh.sample_points_uniformly(number_of_points=args.sample_points)
		mesh_sample_points = np.asarray(mesh_sample.points)
		u, v, valid = project_points(mesh_sample_points, k_rgb)
		z_valid = mesh_sample_points[valid, 2]
		colors = depth_to_colors(z_valid)
		colors_bgr = (colors[:, ::-1] * 255.0).astype(np.float64)

		overlay, count = overlay_points_on_image(rgb, u, v, colors_bgr, args.overlay_alpha)
		out_path = resolve_path(base, args.overlay_out)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		cv2.imwrite(str(out_path), overlay)
		print(f"[INFO] Overlay saved: {out_path} (points drawn: {count})")

		cv2.imshow("STL projected to RGB", overlay)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


def load_rgb_image(rgb_path: Path) -> np.ndarray:
	image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
	if image is None:
		raise RuntimeError(f"Failed to read RGB image: {rgb_path}")
	return image


def _parse_float(value: object) -> float | None:
	if value is None:
		return None
	text = str(value).strip()
	if not text:
		return None
	try:
		return float(text)
	except Exception:
		return None


def _matrix_from_flat_values(values: list[float]) -> np.ndarray | None:
	if len(values) >= 16:
		return np.array(values[:16], dtype=np.float64).reshape((4, 4))
	if len(values) == 12:
		matrix = np.eye(4, dtype=np.float64)
		matrix[:3, :] = np.array(values, dtype=np.float64).reshape((3, 4))
		return matrix
	return None


def load_transform_records(csv_path: Path) -> list[dict[str, object]]:
	if not csv_path.exists():
		raise FileNotFoundError(f"Transform CSV not found: {csv_path}")

	records: list[dict[str, object]] = []

	try:
		with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
			reader = csv.DictReader(f)
			if reader.fieldnames:
				fieldnames = {name.strip() for name in reader.fieldnames if name}
				angle_fields = {
					"收敛结束_rx_度",
					"收敛结束_ry_度",
					"收敛结束_rz_度",
					"最终平移_x_米",
					"最终平移_y_米",
					"最终平移_z_米",
				}
				rows = list(reader)
				if rows and angle_fields.issubset(fieldnames):
					for row in rows:
						rx = float(row["收敛结束_rx_度"])
						ry = float(row["收敛结束_ry_度"])
						rz = float(row["收敛结束_rz_度"])
						tx = float(row["最终平移_x_米"])
						ty = float(row["最终平移_y_米"])
						tz = float(row["最终平移_z_米"])
						transform = build_xyz_rotation(rx, ry, rz)
						transform[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
						records.append(
							{
								"transform": transform,
								"coverage": _parse_float(row.get("覆盖程度")),
								"loss": _parse_float(row.get("loss")),
							}
						)
					return records

				if rows:
					for row in rows:
						values = [value for value in (_parse_float(v) for v in row.values()) if value is not None]
						transform = _matrix_from_flat_values(values)
						if transform is not None:
							records.append(
								{
									"transform": transform,
									"coverage": _parse_float(row.get("覆盖程度")),
									"loss": _parse_float(row.get("loss")),
								}
							)
					if records:
						return records
	except Exception:
		records = []

	number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
	with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			values = [float(match) for match in number_pattern.findall(line)]
			transform = _matrix_from_flat_values(values)
			if transform is not None:
				records.append({"transform": transform, "coverage": None, "loss": None})

	if not records:
		raise ValueError(f"No transforms could be parsed from {csv_path}")
	return records


class PoseBrowser:
	def __init__(
		self,
		records: list[dict[str, object]],
		depth_pcd: o3d.geometry.PointCloud,
		mesh_template: o3d.geometry.TriangleMesh,
		rgb_image: np.ndarray,
		rgb_intrinsics: np.ndarray,
		save_dir: Path,
		overlay_alpha: float,
		overlay_sample_points: int,
		start_index: int = 0,
	) -> None:
		if not records:
			raise ValueError("Transform list cannot be empty.")

		self.records = records
		self.transforms = [record["transform"] for record in records]
		self.depth_pcd = depth_pcd
		self.mesh_template = mesh_template
		self.rgb_image = rgb_image
		self.rgb_intrinsics = rgb_intrinsics
		self.save_dir = save_dir
		self.overlay_alpha = float(overlay_alpha)
		self.overlay_sample_points = int(overlay_sample_points)
		self.index = start_index % len(records)
		self.alignment_visible = False
		self.metrics_window_name = "back_icp overlay/loss"
		self.current_mesh: o3d.geometry.TriangleMesh | None = None
		self.current_frame: o3d.geometry.TriangleMesh | None = None
		self.current_overlay: np.ndarray | None = None
		self.current_overlay_drawn: int = 0
		self.current_coverage: float | None = None
		self.current_loss: float | None = None
		self.metrics_window_enabled = False

		self.vis = o3d.visualization.VisualizerWithKeyCallback()
		self.vis.create_window(window_name=OPEN3D_WINDOW_NAME)
		try:
			cv2.namedWindow(self.metrics_window_name, cv2.WINDOW_NORMAL)
			cv2.resizeWindow(self.metrics_window_name, 420, 320)
			self.metrics_window_enabled = True
		except Exception:
			self.metrics_window_enabled = False

		render_option = self.vis.get_render_option()
		render_option.background_color = np.asarray([0.06, 0.06, 0.06], dtype=np.float64)
		render_option.point_size = 2.0

		self.camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

		self.vis.add_geometry(self.depth_pcd, reset_bounding_box=True)
		self.vis.add_geometry(self.camera_frame, reset_bounding_box=False)

		self.vis.register_key_callback(ord("n"), self._on_next)
		self.vis.register_key_callback(ord("N"), self._on_next)
		self.vis.register_key_callback(ord("p"), self._on_prev)
		self.vis.register_key_callback(ord("P"), self._on_prev)
		self.vis.register_key_callback(ord("s"), self._on_save)
		self.vis.register_key_callback(ord("S"), self._on_save)
		self.vis.register_key_callback(ord("y"), self._on_reveal)
		self.vis.register_key_callback(ord("Y"), self._on_reveal)
		self.vis.register_key_callback(ord("q"), self._on_quit)
		self.vis.register_key_callback(ord("Q"), self._on_quit)

		self._show_index(self.index)

	def _make_overlay(self, transform: np.ndarray) -> tuple[np.ndarray, int]:
		mesh = copy.deepcopy(self.mesh_template)
		mesh.transform(transform)
		sample = mesh.sample_points_uniformly(number_of_points=self.overlay_sample_points)
		points = np.asarray(sample.points)
		u, v, valid = project_points(points, self.rgb_intrinsics)
		colors = depth_to_colors(points[valid, 2])
		colors_bgr = (colors[:, ::-1] * 255.0).astype(np.float64)
		overlay, count = overlay_points_on_image(self.rgb_image, u, v, colors_bgr, self.overlay_alpha)
		return overlay, count

	def _update_metrics_window(self) -> None:
		preview_width = 360
		header_height = 92
		if self.current_overlay is None:
			overlay_preview = np.zeros((200, preview_width, 3), dtype=np.uint8)
		else:
			overlay = self.current_overlay
			height, width = overlay.shape[:2]
			scale = preview_width / max(width, 1)
			preview_height = max(1, int(round(height * scale)))
			overlay_preview = cv2.resize(overlay, (preview_width, preview_height), interpolation=cv2.INTER_AREA)

		header = np.full((header_height, preview_width, 3), 18, dtype=np.uint8)
		texts = [
			f"row: {self.index + 1}/{len(self.records)}",
			f"coverage: {format_metric_value(self.current_coverage, 5, percent=True)}",
			f"loss: {format_metric_value(self.current_loss, 6)}",
			f"drawn: {self.current_overlay_drawn} pts",
		]
		for line_index, line in enumerate(texts):
			y = 22 + line_index * 18
			cv2.putText(
				header,
				line,
				(10, y),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.45,
				(240, 240, 240),
				1,
				cv2.LINE_AA,
			)
		cv2.line(header, (0, header_height - 1), (preview_width - 1, header_height - 1), (64, 64, 64), 1)
		panel = np.vstack([header, overlay_preview])
		if self.metrics_window_enabled:
			cv2.imshow(self.metrics_window_name, panel)
			cv2.waitKey(1)

	def _show_index(self, index: int) -> None:
		new_index = index % len(self.transforms)
		record = self.records[new_index]
		transform = record["transform"]
		if not isinstance(transform, np.ndarray):
			raise TypeError("Record transform must be a numpy array.")
		self.current_coverage = record.get("coverage") if isinstance(record, dict) else None
		self.current_loss = record.get("loss") if isinstance(record, dict) else None

		if self.alignment_visible and self.current_mesh is not None:
			self.vis.remove_geometry(self.current_mesh, reset_bounding_box=False)
		if self.alignment_visible and self.current_frame is not None:
			self.vis.remove_geometry(self.current_frame, reset_bounding_box=False)

		mesh = copy.deepcopy(self.mesh_template)
		mesh.transform(transform)
		mesh.paint_uniform_color([0.92, 0.34, 0.12])

		frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
		frame.transform(transform)

		self.current_mesh = mesh
		self.current_frame = frame
		if self.alignment_visible:
			self.vis.add_geometry(self.current_mesh, reset_bounding_box=False)
			self.vis.add_geometry(self.current_frame, reset_bounding_box=False)

		self.current_overlay, self.current_overlay_drawn = self._make_overlay(transform)
		self._update_metrics_window()
		self.vis.poll_events()
		self.vis.update_renderer()
		self.index = new_index

	def _on_reveal(self, vis: o3d.visualization.Visualizer) -> bool:
		if self.alignment_visible:
			return False
		self.alignment_visible = True
		if self.current_mesh is not None:
			self.vis.add_geometry(self.current_mesh, reset_bounding_box=False)
		if self.current_frame is not None:
			self.vis.add_geometry(self.current_frame, reset_bounding_box=False)
		self.vis.poll_events()
		self.vis.update_renderer()
		return False

	def _on_next(self, vis: o3d.visualization.Visualizer) -> bool:
		self._show_index(self.index + 1)
		return False

	def _on_prev(self, vis: o3d.visualization.Visualizer) -> bool:
		self._show_index(self.index - 1)
		return False

	def _on_save(self, vis: o3d.visualization.Visualizer) -> bool:
		self.save_dir.mkdir(parents=True, exist_ok=True)
		view_path = self.save_dir / f"view_{self.index:03d}.png"
		vis.capture_screen_image(str(view_path), do_render=True)
		if self.current_overlay is not None:
			overlay_path = self.save_dir / f"rgb_{self.index:03d}.png"
			cv2.imwrite(str(overlay_path), self.current_overlay)
		return False

	def _on_quit(self, vis: o3d.visualization.Visualizer) -> bool:
		try:
			cv2.destroyAllWindows()
		except Exception:
			pass
		try:
			vis.destroy_window()
		except Exception:
			pass
		return False

	def run(self) -> None:
		try:
			self.vis.run()
		finally:
			try:
				cv2.destroyAllWindows()
			except Exception:
				pass
			try:
				self.vis.destroy_window()
			except Exception:
				pass


def interactive_main() -> None:
	parser = argparse.ArgumentParser(description="Interactive near_left STL/depth alignment browser.")
	parser.add_argument("--base-dir", default=str(SCRIPT_DIR), help="base directory for relative paths")
	parser.add_argument("--start-index", type=int, default=0, help="initial row index")
	parser.add_argument("--depth-stride", type=int, default=2, help="pixel stride when converting depth map to points")
	parser.add_argument("--overlay-sample-points", type=int, default=20000, help="points sampled from STL for RGB overlay")
	parser.add_argument("--mesh-units", choices=["m", "mm"], default="mm", help="STL units")
	parser.add_argument("--depth-units", choices=["m", "mm"], default="m", help="depth units")
	parser.add_argument("--overlay-alpha", type=float, default=0.75, help="overlay alpha")
	args = parser.parse_args()

	base = Path(args.base_dir)
	stl_path = resolve_path(base, str(DEFAULT_STL_PATH))
	depth_path = resolve_path(base, str(DEFAULT_DEPTH_PATH))
	csv_path = resolve_path(base, str(DEFAULT_CSV_PATH))
	rgb_path = resolve_path(base, str(DEFAULT_RGB_PATH))
	camera_yaml = resolve_path(base, str(DEFAULT_CAMERA_YAML))
	save_dir = resolve_path(base, str(DEFAULT_SAVE_DIR))

	if not stl_path.exists():
		raise FileNotFoundError(f"STL not found: {stl_path}")
	if not depth_path.exists():
		raise FileNotFoundError(f"Depth file not found: {depth_path}")
	if not csv_path.exists():
		raise FileNotFoundError(f"Transform CSV not found: {csv_path}")
	if not rgb_path.exists():
		raise FileNotFoundError(f"RGB image not found: {rgb_path}")
	if not camera_yaml.exists():
		raise FileNotFoundError(f"Camera intrinsics not found: {camera_yaml}")

	records = load_transform_records(csv_path)
	rgb_image = load_rgb_image(rgb_path)
	if rgb_image.shape[:2] != RGB_CROP_SHAPE:
		raise ValueError(
			f"RGB image must be center-cropped to {RGB_CROP_SHAPE[1]}x{RGB_CROP_SHAPE[0]} without resizing; "
			f"got {rgb_image.shape[1]}x{rgb_image.shape[0]}"
		)

	k_full = load_intrinsics(camera_yaml)
	rgb_intrinsics = center_crop_intrinsics(k_full, RGB_ORIGINAL_SHAPE, rgb_image.shape[:2])

	depth_raw = np.load(str(depth_path))
	if depth_raw.ndim == 3 and depth_raw.shape[2] >= 1:
		depth_raw = depth_raw[..., 0]
	if depth_raw.ndim == 2:
		depth_k = center_crop_intrinsics(k_full, RGB_ORIGINAL_SHAPE, depth_raw.shape[:2])
	else:
		depth_k = k_full
	depth_points = load_points(depth_path, depth_k, "auto", args.depth_stride)
	if args.depth_units == "mm":
		depth_points = scale_points(depth_points, 0.001)
	if depth_points.size == 0:
		raise RuntimeError(f"No valid depth points in {depth_path}")

	mesh = o3d.io.read_triangle_mesh(str(stl_path))
	if mesh.is_empty():
		raise RuntimeError(f"Failed to load mesh: {stl_path}")
	mesh.compute_vertex_normals()
	if args.mesh_units == "mm":
		mesh.scale(0.001, center=(0.0, 0.0, 0.0))

	depth_extent = estimate_extent(depth_points)
	mesh_extent = estimate_extent(np.asarray(mesh.vertices))
	if mesh_extent > 0.0 and depth_extent > 0.0:
		ratio = mesh_extent / max(depth_extent, 1e-9)
		if ratio > 10.0 or ratio < 0.1:
			print("[WARN] Mesh/depth scale looks mismatched. Check --mesh-units and --depth-units.")

	depth_colors = depth_to_colors(depth_points[:, 2])
	depth_pcd = o3d.geometry.PointCloud()
	depth_pcd.points = o3d.utility.Vector3dVector(depth_points)
	depth_pcd.colors = o3d.utility.Vector3dVector(depth_colors)

	browser = PoseBrowser(
		records=records,
		depth_pcd=depth_pcd,
		mesh_template=mesh,
		rgb_image=rgb_image,
		rgb_intrinsics=rgb_intrinsics,
		save_dir=save_dir,
		overlay_alpha=args.overlay_alpha,
		overlay_sample_points=args.overlay_sample_points,
		start_index=args.start_index,
	)
	browser.run()


if __name__ == "__main__":
	interactive_main()
