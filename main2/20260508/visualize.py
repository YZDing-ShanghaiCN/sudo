#!/usr/bin/env python3
"""Visualize depth_mean.npy point clouds for the 20260508 tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR / "result"
CAMERA_INTRINSICS = {
	"left": SCRIPT_DIR.parent / "aililight_cameras" / "chest_left_camera.yaml",
	"right": SCRIPT_DIR.parent / "aililight_cameras" / "chest_right_camera.yaml",
}
RGB_ORIGINAL_SHAPE = (800, 1280)
DEFAULT_STRIDE = 2
SCREENSHOT_NAME = "point_cloud.png"


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


def choose_camera(task_name: str) -> str:
	if "left_chest_origin" in task_name:
		return "left"
	if "right_chest_origin" in task_name:
		return "right"
	raise ValueError(f"Cannot infer camera side from task name: {task_name}")


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


def depth_to_point_cloud(depth: np.ndarray, k: np.ndarray, stride: int) -> o3d.geometry.PointCloud:
	if stride < 1:
		raise ValueError("stride must be >= 1")

	sampled = np.asarray(depth, dtype=np.float64)[::stride, ::stride]
	valid = np.isfinite(sampled) & (sampled > 0.0)
	if not np.any(valid):
		return o3d.geometry.PointCloud()

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
	points = np.stack((x, y, z), axis=1)
	colors = depth_to_colors(z)

	cloud = o3d.geometry.PointCloud()
	cloud.points = o3d.utility.Vector3dVector(points)
	cloud.colors = o3d.utility.Vector3dVector(colors)
	return cloud


def load_task_dirs(task_name: str | None = None) -> list[Path]:
	if task_name:
		task_dir = RESULT_ROOT / task_name
		if not task_dir.exists():
			raise FileNotFoundError(f"Task directory not found: {task_dir}")
		return [task_dir]

	return [
		path
		for path in sorted(RESULT_ROOT.iterdir())
		if path.is_dir() and (path / "depth_mean.npy").exists()
	]


def screenshot_path(task_dir: Path) -> Path:
	return task_dir / SCREENSHOT_NAME


def save_current_view(vis: o3d.visualization.Visualizer, task_dir: Path) -> bool:
	save_path = screenshot_path(task_dir)
	vis.capture_screen_image(str(save_path), do_render=True)
	print(f"[OK] saved screenshot: {save_path}", flush=True)
	return False


def show_point_cloud(cloud: o3d.geometry.PointCloud, task_dir: Path, window_name: str) -> None:
	visualizer = o3d.visualization.VisualizerWithKeyCallback()
	visualizer.create_window(window_name=window_name)
	visualizer.add_geometry(cloud)

	render_option = visualizer.get_render_option()
	render_option.background_color = np.asarray([0.08, 0.08, 0.08], dtype=np.float64)
	render_option.point_size = 2.0

	def on_save(vis: o3d.visualization.Visualizer) -> bool:
		return save_current_view(vis, task_dir)

	visualizer.register_key_callback(ord("S"), on_save)
	visualizer.register_key_callback(ord("s"), on_save)
	print(f"[INFO] press S to save screenshot to {screenshot_path(task_dir)}", flush=True)
	visualizer.run()
	visualizer.destroy_window()


def visualize_task(task_dir: Path, stride: int) -> None:
	task_name = task_dir.name
	depth_path = task_dir / "depth_mean.npy"
	if not depth_path.exists():
		raise FileNotFoundError(f"Missing depth_mean.npy: {depth_path}")

	depth = np.load(depth_path)
	camera_key = choose_camera(task_name)
	intrinsics = center_crop_intrinsics(
		load_intrinsics(CAMERA_INTRINSICS[camera_key]),
		RGB_ORIGINAL_SHAPE,
		depth.shape[:2],
	)
	cloud = depth_to_point_cloud(depth, intrinsics, stride)
	if len(cloud.points) == 0:
		raise RuntimeError(f"No valid depth points in {depth_path}")

	print(
		f"[INFO] {task_name}: camera={camera_key}, depth_shape={depth.shape}, "
		f"points={len(cloud.points)}",
		flush=True,
	)
	show_point_cloud(cloud, task_dir, f"{task_name} point cloud")


def main() -> None:
	parser = argparse.ArgumentParser(description="Visualize depth_mean.npy point clouds.")
	parser.add_argument("--task", default="", help="Task directory name to visualize only one task.")
	parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Pixel stride when sampling depth.")
	args = parser.parse_args()

	task_dirs = load_task_dirs(args.task or None)
	if not task_dirs:
		raise RuntimeError(f"No task directories with depth_mean.npy found under {RESULT_ROOT}")

	for task_dir in task_dirs:
		visualize_task(task_dir, args.stride)


if __name__ == "__main__":
	main()
# requirements:
# - 可视化点云 我/home/user/Desktop/main/main2/20260508/result 下有6个任务的深度数据
# - 名字叫做 depth_mean.npy 需要可视化这些点云数据 打开窗口能够按s保存 终端打印保存路径
# - 保存到/home/user/Desktop/main/main2/20260508/result/{任务名称} 下面创建一个图片文件，因为一个任务取一次平均，因此每个任务只会保存一张图片