from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import yaml


AUTO_5X5_DICTIONARIES = (
	"DICT_5X5_50",
	"DICT_5X5_100",
	"DICT_5X5_250",
	"DICT_5X5_1000",
)


def load_camera_intrinsics(calib_path: Path, calib_key: str) -> tuple[np.ndarray, np.ndarray]:
	with calib_path.open("r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	node = cfg.get(calib_key)
	if not isinstance(node, dict):
		raise ValueError(f"calibration key not found: {calib_key}")

	camera_matrix = np.array(node.get("camera_matrix"), dtype=np.float64)
	dist_coeffs = np.array(node.get("dist_coeffs"), dtype=np.float64)

	if camera_matrix.shape != (3, 3):
		raise ValueError(f"invalid camera_matrix shape: {camera_matrix.shape}")

	if dist_coeffs.ndim == 2 and dist_coeffs.shape[0] == 1:
		dist_coeffs = dist_coeffs.reshape(-1)

	return camera_matrix, dist_coeffs


def build_detector(dictionary_name: str) -> Callable[[np.ndarray], tuple[list[np.ndarray], Optional[np.ndarray]]]:
	if not hasattr(cv2, "aruco"):
		raise RuntimeError("OpenCV aruco module is unavailable. Install opencv-contrib-python.")

	if not hasattr(cv2.aruco, dictionary_name):
		raise ValueError(f"unsupported marker dictionary: {dictionary_name}")

	dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))

	if hasattr(cv2.aruco, "DetectorParameters"):
		params = cv2.aruco.DetectorParameters()
	else:
		params = cv2.aruco.DetectorParameters_create()

	if hasattr(cv2.aruco, "ArucoDetector"):
		detector = cv2.aruco.ArucoDetector(dictionary, params)

		def _detect(gray: np.ndarray) -> tuple[list[np.ndarray], Optional[np.ndarray]]:
			corners, ids, _ = detector.detectMarkers(gray)
			return list(corners), ids

	else:

		def _detect(gray: np.ndarray) -> tuple[list[np.ndarray], Optional[np.ndarray]]:
			corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
			return list(corners), ids

	return _detect


def resolve_dictionary_name(dictionary_name: str, sample_gray: np.ndarray) -> str:
	if dictionary_name != "AUTO_5X5":
		return dictionary_name

	best_name: Optional[str] = None
	best_count = -1

	for candidate in AUTO_5X5_DICTIONARIES:
		if not hasattr(cv2.aruco, candidate):
			continue
		detect = build_detector(candidate)
		_corners, ids = detect(sample_gray)
		count = 0 if ids is None else int(ids.size)
		if count > best_count:
			best_count = count
			best_name = candidate

	if best_name is None:
		raise RuntimeError("No 5x5 ArUco dictionary is available in this OpenCV build.")

	if best_count <= 0:
		raise RuntimeError("AUTO_5X5 could not detect any marker in the sample image.")

	print(f"[INFO] AUTO_5X5 selected dictionary: {best_name} (markers={best_count})")
	return best_name


def create_charuco_board(
	squares_x: int,
	squares_y: int,
	square_size_mm: float,
	marker_size_mm: float,
	dictionary_name: str,
	legacy_pattern: bool = False,
):
	if not hasattr(cv2.aruco, dictionary_name):
		raise ValueError(f"unsupported marker dictionary: {dictionary_name}")

	aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))

	if hasattr(cv2.aruco, "CharucoBoard"):
		try:
			board = cv2.aruco.CharucoBoard(
				(squares_x, squares_y),
				square_size_mm,
				marker_size_mm,
				aruco_dict,
			)
			if legacy_pattern and hasattr(board, "setLegacyPattern"):
				board.setLegacyPattern(True)
			return board
		except TypeError:
			pass

	if hasattr(cv2.aruco, "CharucoBoard_create"):
		board = cv2.aruco.CharucoBoard_create(
			squares_x,
			squares_y,
			square_size_mm,
			marker_size_mm,
			aruco_dict,
		)
		if legacy_pattern and hasattr(board, "setLegacyPattern"):
			board.setLegacyPattern(True)
		return board

	raise RuntimeError("CharucoBoard API is unavailable in current OpenCV build.")


def select_best_charuco_board(
	sample_gray: np.ndarray,
	detect: Callable[[np.ndarray], tuple[list[np.ndarray], Optional[np.ndarray]]],
	dictionary_name: str,
	squares_x: int,
	squares_y: int,
	square_size_mm: float,
	marker_size_mm: float,
	camera_matrix: np.ndarray,
	dist_coeffs: np.ndarray,
):
	marker_corners, marker_ids = detect(sample_gray)
	if marker_ids is None or int(marker_ids.size) == 0:
		raise RuntimeError("No markers detected in sample image for CharuCo board selection.")

	candidates: list[tuple[int, int]] = [(squares_x, squares_y)]
	if squares_x != squares_y:
		candidates.append((squares_y, squares_x))

	best_board = None
	best_info: Optional[tuple[int, int, bool, int]] = None

	for sx, sy in candidates:
		for legacy in (False, True):
			try:
				board = create_charuco_board(
					squares_x=sx,
					squares_y=sy,
					square_size_mm=square_size_mm,
					marker_size_mm=marker_size_mm,
					dictionary_name=dictionary_name,
					legacy_pattern=legacy,
				)
			except Exception:
				continue

			ok, _charuco_corners, charuco_ids = interpolate_charuco(
				sample_gray,
				marker_corners,
				marker_ids,
				board,
				camera_matrix,
				dist_coeffs,
			)
			count = 0 if charuco_ids is None else int(charuco_ids.size)
			if not ok:
				count = 0

			if best_info is None or count > best_info[3]:
				best_board = board
				best_info = (sx, sy, legacy, count)

	if best_board is None or best_info is None:
		raise RuntimeError("Failed to build a valid CharuCo board candidate.")

	if best_info[3] <= 0:
		raise RuntimeError(
			"No valid CharuCo interpolation for current board parameters. "
			"Check squares-x/squares-y/square-size-mm/marker-size-mm/dictionary."
		)

	print(
		f"[INFO] Selected CharuCo board: squares_x={best_info[0]} squares_y={best_info[1]} "
		f"legacy={best_info[2]} interpolated={best_info[3]}"
	)
	return best_board


def interpolate_charuco(
	gray: np.ndarray,
	marker_corners: list[np.ndarray],
	marker_ids: np.ndarray,
	board,
	camera_matrix: np.ndarray,
	dist_coeffs: np.ndarray,
) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
	try:
		num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
			marker_corners,
			marker_ids,
			gray,
			board,
			camera_matrix,
			dist_coeffs,
		)
	except TypeError:
		num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
			marker_corners,
			marker_ids,
			gray,
			board,
		)

	count = int(num) if num is not None else 0
	ok = count > 0 and charuco_corners is not None and charuco_ids is not None
	if not ok:
		return False, None, None
	return True, charuco_corners, charuco_ids


def estimate_charuco_pose(
	charuco_corners: np.ndarray,
	charuco_ids: np.ndarray,
	board,
	camera_matrix: np.ndarray,
	dist_coeffs: np.ndarray,
) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
	if not hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
		raise RuntimeError("estimatePoseCharucoBoard is unavailable in current OpenCV build.")

	try:
		retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
			charuco_corners,
			charuco_ids,
			board,
			camera_matrix,
			dist_coeffs,
			None,
			None,
		)
	except TypeError:
		retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
			charuco_corners,
			charuco_ids,
			board,
			camera_matrix,
			dist_coeffs,
		)

	ok = bool(retval) if isinstance(retval, (bool, np.bool_)) else float(retval) > 0.0
	if not ok or rvec is None or tvec is None:
		return False, None, None
	return True, rvec, tvec


def process_one_image(
	image_path: Path,
	output_name: str,
	detect: Callable[[np.ndarray], tuple[list[np.ndarray], Optional[np.ndarray]]],
	charuco_board,
	camera_matrix: np.ndarray,
	dist_coeffs: np.ndarray,
	min_markers: int,
	min_charuco_corners: int,
) -> tuple[bool, str]:
	output_path = image_path.parent / output_name

	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image is None:
		payload = {"rvec": None, "tvec_mm": None, "tvec_m": None, "error": "failed to read image"}
		output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
		return False, "read image failed"

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	corners, ids = detect(gray)
	marker_count = 0 if ids is None else int(ids.size)

	if ids is None or marker_count < min_markers:
		payload = {
			"rvec": None,
			"tvec_mm": None,
			"tvec_m": None,
			"error": f"insufficient markers: {marker_count} < {min_markers}",
		}
		output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
		return False, "insufficient markers"

	ok, charuco_corners, charuco_ids = interpolate_charuco(
		gray,
		corners,
		ids,
		charuco_board,
		camera_matrix,
		dist_coeffs,
	)
	charuco_count = 0 if charuco_ids is None else int(charuco_ids.size)
	if (not ok) or charuco_corners is None or charuco_ids is None or charuco_count < min_charuco_corners:
		payload = {
			"rvec": None,
			"tvec_mm": None,
			"tvec_m": None,
			"error": f"insufficient charuco corners: {charuco_count} < {min_charuco_corners}",
		}
		output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
		return False, "insufficient charuco corners"

	ok, rvec, tvec = estimate_charuco_pose(charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs)
	if not ok or rvec is None or tvec is None:
		payload = {"rvec": None, "tvec_mm": None, "tvec_m": None, "error": "estimatePoseCharucoBoard failed"}
		output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
		return False, "pose estimation failed"

	rvec_v = rvec.reshape(3).astype(float)
	tvec_mm = tvec.reshape(3).astype(float)
	tvec_m = tvec_mm / 1000.0

	payload = {
		"rvec": [float(x) for x in rvec_v],
		"tvec_mm": [float(x) for x in tvec_mm],
		"tvec_m": [float(x) for x in tvec_m],
	}
	output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
	return True, "ok"


def parse_args() -> argparse.Namespace:
	project_root = Path(__file__).resolve().parents[1]
	parser = argparse.ArgumentParser(
		description="Batch estimate CharuCo board pose for each capture_raw.png under result directory."
	)
	parser.add_argument(
		"--calib",
		type=Path,
		default=project_root / "calibrate" / "camera_cfg.yaml",
		help="Camera intrinsics YAML path.",
	)
	parser.add_argument("--calib-key", type=str, default="camera_1", help="YAML key for camera intrinsics.")
	parser.add_argument(
		"--result-root",
		type=Path,
		default=project_root / "main" / "result",
		help="Directory containing run sub-folders.",
	)
	parser.add_argument(
		"--image-name",
		type=str,
		default="capture_raw.png",
		help="Image file name inside each run directory.",
	)
	parser.add_argument(
		"--output-name",
		type=str,
		default="capture_raw_pose.json",
		help="Output JSON name written in the same image directory.",
	)
	parser.add_argument(
		"--dictionary",
		type=str,
		default="AUTO_5X5",
		help="Marker dictionary name, or AUTO_5X5.",
	)
	parser.add_argument("--squares-x", type=int, default=9, help="CharuCo chessboard square count in X direction.")
	parser.add_argument("--squares-y", type=int, default=14, help="CharuCo chessboard square count in Y direction.")
	parser.add_argument("--square-size-mm", type=float, default=20.0, help="CharuCo square size in mm.")
	parser.add_argument("--marker-size-mm", type=float, default=15.0, help="Marker side length in mm.")
	parser.add_argument("--min-markers", type=int, default=6, help="Minimum detected markers required.")
	parser.add_argument("--min-charuco-corners", type=int, default=8, help="Minimum interpolated CharuCo corners required.")
	return parser.parse_args()


def main() -> int:
	args = parse_args()

	if args.squares_x <= 1 or args.squares_y <= 1:
		print("[ERROR] --squares-x and --squares-y must both be > 1")
		return 1
	if args.square_size_mm <= 0:
		print("[ERROR] --square-size-mm must be > 0")
		return 1
	if args.marker_size_mm <= 0:
		print("[ERROR] --marker-size-mm must be > 0")
		return 1
	if args.marker_size_mm >= args.square_size_mm:
		print("[ERROR] --marker-size-mm must be smaller than --square-size-mm")
		return 1
	if args.min_charuco_corners < 4:
		print("[ERROR] --min-charuco-corners must be >= 4")
		return 1

	image_paths = sorted(args.result_root.glob(f"*/{args.image_name}"))
	if not image_paths:
		print(f"[ERROR] No image matched: {args.result_root}/*/{args.image_name}")
		return 1

	sample_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
	if sample_image is None:
		print(f"[ERROR] Failed to read sample image: {image_paths[0]}")
		return 1

	resolved_dict_name = resolve_dictionary_name(str(args.dictionary), sample_image)
	camera_matrix, dist_coeffs = load_camera_intrinsics(args.calib, args.calib_key)
	detect = build_detector(resolved_dict_name)
	charuco_board = select_best_charuco_board(
		sample_gray=sample_image,
		detect=detect,
		dictionary_name=resolved_dict_name,
		squares_x=int(args.squares_x),
		squares_y=int(args.squares_y),
		square_size_mm=float(args.square_size_mm),
		marker_size_mm=float(args.marker_size_mm),
		camera_matrix=camera_matrix,
		dist_coeffs=dist_coeffs,
	)

	ok_count = 0
	fail_count = 0

	for image_path in image_paths:
		ok, reason = process_one_image(
			image_path=image_path,
			output_name=args.output_name,
			detect=detect,
			charuco_board=charuco_board,
			camera_matrix=camera_matrix,
			dist_coeffs=dist_coeffs,
			min_markers=int(args.min_markers),
			min_charuco_corners=int(args.min_charuco_corners),
		)
		if ok:
			ok_count += 1
			print(f"[OK] {image_path.parent.name}/{args.output_name}")
		else:
			fail_count += 1
			print(f"[WARN] {image_path.parent.name}/{args.output_name}: {reason}")

	print(f"[DONE] total={len(image_paths)} ok={ok_count} fail={fail_count}")
	return 0 if fail_count == 0 else 2


if __name__ == "__main__":
	raise SystemExit(main())
