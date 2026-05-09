from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import cv2
import numpy as np


# =========================
# Global config (edit here)
# =========================
ROOT_DIR = Path(__file__).resolve().parent

# Dataset directory names under ROOT_DIR.
RGB_DIRNAME = "rgb_all"
DEPTH_DIRNAME = "depth_all_new"

# Output directory name under ROOT_DIR.
RESULT_DIRNAME = "result"

# Mask file (Label Studio export). Polygons are used as ROI.
MASK_JSON_PATH = Path("/home/user/Desktop/main/main2/20260508/project-5-at-2026-05-08-09-02-e9719ff4.json")

SUPPORTED_TASKS = [
	"farpose_left_chest_origin",
	"farpose_right_chest_origin",
	"nearpose_left_chest_origin",
	"nearpose_right_chest_origin",
	"waitpose_left_chest_origin",
	"waitpose_right_chest_origin",
]
# Run all tasks in SUPPORTED_TASKS when True.
RUN_ALL_TASKS = True
# You can switch this to run one task each time if RUN_ALL_TASKS=False.
TASK_NAME = SUPPORTED_TASKS[5]


@dataclass
class ImageDepthStats:
	file_name: str
	valid_pixels: int
	min_depth: float
	max_depth: float
	mean_depth: float
	variance_depth: float
	std_depth: float
	median_depth: float
	p25_depth: float
	p75_depth: float


def _validate_config() -> None:
	if TASK_NAME not in SUPPORTED_TASKS:
		raise ValueError(
			f"Invalid TASK_NAME: {TASK_NAME}. Supported: {SUPPORTED_TASKS}"
		)


def _collect_pairs(rgb_task_dir: Path, depth_task_dir: Path) -> List[tuple[Path, Path]]:
	rgb_map = {p.stem: p for p in rgb_task_dir.iterdir() if p.is_file()}
	depth_map = {p.stem: p for p in depth_task_dir.iterdir() if p.is_file()}

	common_stems = sorted(set(rgb_map.keys()) & set(depth_map.keys()))
	return [(rgb_map[s], depth_map[s]) for s in common_stems]


def _read_depth(depth_path: Path) -> np.ndarray:
	depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
	if depth is None:
		raise RuntimeError(f"Failed to read depth image: {depth_path}")

	depth = depth.astype(np.float64)
	valid_mask = np.isfinite(depth) & (depth > 0)
	return depth[valid_mask]


def _calc_stats(values: np.ndarray) -> dict:
	return {
		"min_depth": float(np.min(values)),
		"max_depth": float(np.max(values)),
		"mean_depth": float(np.mean(values)),
		"variance_depth": float(np.var(values)),
		"std_depth": float(np.std(values)),
		"median_depth": float(np.median(values)),
		"p25_depth": float(np.percentile(values, 25)),
		"p75_depth": float(np.percentile(values, 75)),
	}


def _load_task_masks(mask_json_path: Path) -> dict[str, np.ndarray]:
	if not mask_json_path.exists():
		raise FileNotFoundError(f"Mask json not found: {mask_json_path}")
	data = json.loads(mask_json_path.read_text(encoding="utf-8"))
	masks: dict[str, np.ndarray] = {}
	for item in data:
		image_name = None
		if isinstance(item, dict):
			image_name = item.get("data", {}).get("image") or item.get("file_upload")
		if not image_name:
			continue
		image_stem = Path(image_name).stem
		mask = None
		for ann in item.get("annotations", []):
			for res in ann.get("result", []):
				if res.get("type") != "polygonlabels":
					continue
				w = int(res.get("original_width", 0))
				h = int(res.get("original_height", 0))
				points = res.get("value", {}).get("points", [])
				if w <= 0 or h <= 0 or not points:
					continue
				if mask is None:
					mask = np.zeros((h, w), dtype=np.uint8)
				pts = np.array(
					[[[p[0] / 100.0 * w, p[1] / 100.0 * h]] for p in points],
					dtype=np.int32,
				)
				cv2.fillPoly(mask, [pts], 1)
		if mask is not None:
			masks[image_stem] = mask.astype(bool)
	return masks


def _match_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
	if mask.shape == target_shape:
		return mask
	resized = cv2.resize(mask.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
	return resized.astype(bool)


def _compute_pixel_mean_std(depth_paths: List[Path], output_dir: Path, mask: np.ndarray) -> dict:
	# Find first readable depth image to determine shape and sample values
	first_sample = None
	for p in depth_paths:
		d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
		if d is None:
			continue
		if d.ndim == 3:
			d = d[:, :, 0]
		first_sample = d.astype(np.float64)
		mask_use = _match_mask(mask, first_sample.shape[:2])
		finite_vals = first_sample[np.isfinite(first_sample) & (first_sample > 0) & mask_use]
		break

	if first_sample is None:
		raise RuntimeError("No readable depth images found for pixel stats")

	h, w = first_sample.shape[:2]
	mask_use = _match_mask(mask, (h, w))

	# Heuristic to detect units: if typical values are large, assume mm
	units = "m"
	meters_per_unit = 1.0
	if finite_vals.size > 0:
		mean_val = float(np.nanmean(finite_vals))
		max_val = float(np.nanmax(finite_vals))
		if max_val > 1000 or mean_val > 10:
			meters_per_unit = 0.001
			units = "mm"

	# Accumulators for mean/std (online, ignoring invalid pixels)
	sum_arr = np.zeros((h, w), dtype=np.float64)
	sumsq_arr = np.zeros((h, w), dtype=np.float64)
	count_arr = np.zeros((h, w), dtype=np.int32)

	for p in depth_paths:
		d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
		if d is None:
			print(f"[WARN] unreadable depth: {p}")
			continue
		if d.ndim == 3:
			d = d[:, :, 0]
		d = d.astype(np.float64)
		valid_mask = np.isfinite(d) & (d > 0) & mask_use
		if not np.any(valid_mask):
			continue
		# convert to meters
		d_m = d * meters_per_unit
		sum_arr[valid_mask] += d_m[valid_mask]
		sumsq_arr[valid_mask] += (d_m[valid_mask] ** 2)
		count_arr[valid_mask] += 1

	valid_pixel_mask = count_arr > 0
	mean_img = np.full((h, w), np.nan, dtype=np.float32)
	std_img = np.full((h, w), np.nan, dtype=np.float32)

	mean_img[valid_pixel_mask] = (sum_arr[valid_pixel_mask] / count_arr[valid_pixel_mask]).astype(np.float32)
	var = (sumsq_arr[valid_pixel_mask] / count_arr[valid_pixel_mask]) - (mean_img[valid_pixel_mask].astype(np.float64) ** 2)
	var = np.maximum(var, 0.0)
	std_img[valid_pixel_mask] = np.sqrt(var).astype(np.float32)

	# Save numpy arrays (in meters)
	np.save(output_dir / "depth_mean.npy", mean_img)
	np.save(output_dir / "depth_std.npy", std_img)

	# Compute proportions for thresholds (1cm,2cm,5cm,10cm)
	thresholds_cm = [1, 2, 5, 10]
	thresholds_m = [c / 100.0 for c in thresholds_cm]
	total_valid = int(np.count_nonzero(valid_pixel_mask))
	proportions = {}
	counts = {}
	for c, tm in zip(thresholds_cm, thresholds_m):
		mask = valid_pixel_mask & (std_img <= tm)
		cnt = int(np.count_nonzero(mask))
		counts[f"{c}cm"] = cnt
		proportions[f"{c}cm"] = cnt / total_valid if total_valid > 0 else None

	summary = {
		"units_assumed": units,
		"meters_per_unit": meters_per_unit,
		"num_frames": len(depth_paths),
		"total_valid_pixels": total_valid,
		"thresholds_meters": {f"{c}cm": tm for c, tm in zip(thresholds_cm, thresholds_m)},
		"counts": counts,
		"proportions": proportions,
	}

	json_path = output_dir / "depth_pixel_summary.json"
	json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	print(f"[OK] Saved depth_mean.npy and depth_std.npy to: {output_dir}")
	print(f"[OK] Saved pixel summary json to: {json_path}")

	return summary


def run() -> None:
	masks = _load_task_masks(MASK_JSON_PATH)
	if RUN_ALL_TASKS:
		tasks_to_run = SUPPORTED_TASKS
	else:
		_validate_config()
		tasks_to_run = [TASK_NAME]

	for task in tasks_to_run:
		if task not in masks:
			print(f"[WARN] Mask not found for task: {task}")
			continue
		mask = masks[task]

		rgb_task_dir = ROOT_DIR / RGB_DIRNAME / task
		depth_task_dir = ROOT_DIR / DEPTH_DIRNAME / task

		if not rgb_task_dir.exists():
			print(f"[WARN] RGB task dir not found: {rgb_task_dir}")
			continue
		if not depth_task_dir.exists():
			print(f"[WARN] Depth task dir not found: {depth_task_dir}")
			continue

		pairs = _collect_pairs(rgb_task_dir, depth_task_dir)
		if not pairs:
			print("[WARN] No matched rgb/depth pairs found. Check file names under task folders.")
			continue

		output_dir = ROOT_DIR / RESULT_DIRNAME / task
		output_dir.mkdir(parents=True, exist_ok=True)

		per_image_stats: List[ImageDepthStats] = []
		all_depth_values: List[np.ndarray] = []

		for _, depth_path in pairs:
			depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
			if depth_raw is None:
				print(f"[WARN] Skip unreadable depth: {depth_path}")
				continue

			h, w = depth_raw.shape[:2]
			mask_use = _match_mask(mask, (h, w))
			valid_values = depth_raw.astype(np.float64)
			valid_values = valid_values[np.isfinite(valid_values) & (valid_values > 0) & mask_use]
			if valid_values.size == 0:
				print(f"[WARN] Skip empty-valid depth: {depth_path}")
				continue

			stats = _calc_stats(valid_values)
			per_image_stats.append(
				ImageDepthStats(
					file_name=depth_path.name,
					valid_pixels=int(valid_values.size),
					min_depth=stats["min_depth"],
					max_depth=stats["max_depth"],
					mean_depth=stats["mean_depth"],
					variance_depth=stats["variance_depth"],
					std_depth=stats["std_depth"],
					median_depth=stats["median_depth"],
					p25_depth=stats["p25_depth"],
					p75_depth=stats["p75_depth"],
				)
			)
			all_depth_values.append(valid_values)

		if not per_image_stats:
			print("[WARN] No valid depth pixels found in matched pairs.")
			continue

		merged_values = np.concatenate(all_depth_values, axis=0)
		overall = _calc_stats(merged_values)

		summary = {
			"task_name": task,
			"rgb_dir": str(rgb_task_dir),
			"depth_dir": str(depth_task_dir),
			"num_pairs": len(pairs),
			"num_valid_images": len(per_image_stats),
			"total_valid_pixels": int(merged_values.size),
			**overall,
		}

		summary_path = output_dir / "depth_summary.json"
		summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

		csv_path = output_dir / "per_image_depth_stats.csv"
		with csv_path.open("w", newline="", encoding="utf-8") as f:
			writer = csv.DictWriter(
				f,
				fieldnames=list(asdict(per_image_stats[0]).keys()),
			)
			writer.writeheader()
			for item in per_image_stats:
				writer.writerow(asdict(item))

		# Compute per-pixel mean/std across the matched depth frames and save .npy + summary json
		depth_paths = [depth_path for _, depth_path in pairs]
		try:
			_compute_pixel_mean_std(depth_paths, output_dir, mask)
		except Exception as e:
			print(f"[WARN] Failed to compute per-pixel stats: {e}")

		print(f"[OK] Task: {task}")
		print(f"[OK] Summary saved to: {summary_path}")
		print(f"[OK] Per-image CSV saved to: {csv_path}")


if __name__ == "__main__":
	run()
