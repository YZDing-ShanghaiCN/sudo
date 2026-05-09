#!/usr/bin/env python3
"""Copy chest camera images and write undistorted outputs."""

from pathlib import Path
import shutil

import cv2
import numpy as np


POSES = ["near_pose", "far_pose", "wait_pose"]
CAMERAS = {
	"left_chest": {
		"input_dir": "chest_left_camera",
		"calib_file": "chest_left_camera.yaml",
	},
	"right_chest": {
		"input_dir": "chest_right_camera",
		"calib_file": "chest_right_camera.yaml",
	},
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_calibration(yaml_path: Path) -> tuple[np.ndarray, np.ndarray]:
	try:
		import yaml
	except ImportError as exc:
		raise SystemExit("Missing dependency: PyYAML. Please install pyyaml.") from exc

	data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
	if "intrinsic" not in data or "distortion" not in data:
		raise ValueError(f"Invalid calibration file: {yaml_path}")

	camera_matrix = np.array(data["intrinsic"], dtype=np.float32)
	dist_coeffs = np.array(data["distortion"], dtype=np.float32).reshape(-1, 1)
	return camera_matrix, dist_coeffs


def iter_images(folder: Path) -> list[Path]:
	if not folder.exists():
		return []
	return sorted(
		[p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
	)


def undistort_pose(
	pose: str,
	input_root: Path,
	output_root: Path,
	calib_root: Path,
	calibrations: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
	pose_input = input_root / pose / "rgb"
	pose_output = output_root / pose / "rgb"

	for side, cfg in CAMERAS.items():
		input_dir = pose_input / cfg["input_dir"]
		origin_dir = pose_output / side / "origin"
		undist_dir = pose_output / side / "undistorted"
		origin_dir.mkdir(parents=True, exist_ok=True)
		undist_dir.mkdir(parents=True, exist_ok=True)

		if side not in calibrations:
			calib_path = calib_root / cfg["calib_file"]
			calibrations[side] = load_calibration(calib_path)

		camera_matrix, dist_coeffs = calibrations[side]
		images = iter_images(input_dir)
		if not images:
			print(f"[WARN] No images found in {input_dir}")
			continue

		for image_path in images:
			shutil.copy2(image_path, origin_dir / image_path.name)
			image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
			if image is None:
				print(f"[WARN] Failed to read {image_path}")
				continue

			undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
			cv2.imwrite(str(undist_dir / image_path.name), undistorted)


def main() -> None:
	output_root = Path(__file__).resolve().parent
	input_root = output_root.parent
	calib_root = input_root / "aililight_cameras"
	calibrations: dict[str, tuple[np.ndarray, np.ndarray]] = {}

	for pose in POSES:
		undistort_pose(pose, input_root, output_root, calib_root, calibrations)


if __name__ == "__main__":
	main()
