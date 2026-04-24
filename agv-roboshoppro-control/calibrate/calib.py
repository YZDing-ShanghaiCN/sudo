"""Interactive ArUco board camera calibration script.

Workflow:
1) Open camera index 1 by default (index 0 is usually laptop webcam).
2) Live preview first, press 'c' to confirm and continue.
3) A timestamped output directory is created for this run.
4) In capture mode:
   - Press Enter to capture a frame (only when enough markers are detected).
   - Press 'd' to delete last captured frame.
   - Press 'q' / Esc to finish and run calibration.
5) Save calibration outputs into the same timestamped directory.

Default board setup (as requested):
- 4 rows x 6 cols markers
- marker size: 30 mm
- marker gap: 5 mm

Examples:
	python calibrate/calib.py
	python calibrate/calib.py --camera-index 1 --rows 4 --cols 6 --marker-size 30 --marker-gap 5 --first-marker-id 1
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


PREVIEW_WINDOW = "Calibration - Preview"
CAPTURE_WINDOW = "Calibration - ArUco Capture"
UNDISTORT_WINDOW = "Calibration - Undistort Preview"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Interactive camera calibration with ArUco GridBoard.")
	parser.add_argument(
		"--camera-index",
		type=int,
		default=1,
		help="Camera device index to open. Default: 1.",
	)
	parser.add_argument(
		"--rows",
		type=int,
		default=4,
		help="ArUco board marker rows. Default: 4.",
	)
	parser.add_argument(
		"--cols",
		type=int,
		default=6,
		help="ArUco board marker columns. Default: 6.",
	)
	parser.add_argument(
		"--marker-size",
		type=float,
		default=30.0,
		help="Single marker size in mm. Default: 30.",
	)
	parser.add_argument(
		"--marker-gap",
		type=float,
		default=5.0,
		help="Gap between adjacent markers in mm. Default: 5.",
	)
	parser.add_argument(
		"--dict",
		type=str,
		default="DICT_4X4_50",
		help="ArUco dictionary name. Default: DICT_4X4_50.",
	)
	parser.add_argument(
		"--first-marker-id",
		type=int,
		default=1,
		help="First marker ID on the printed board. Default: 1.",
	)
	parser.add_argument(
		"--min-markers",
		type=int,
		default=8,
		help="Minimum detected markers required to accept a capture. Default: 8.",
	)
	parser.add_argument(
		"--min-samples",
		type=int,
		default=8,
		help="Minimum accepted images required for calibration. Default: 8.",
	)
	parser.add_argument(
		"--width",
		type=int,
		default=1280,
		help="Requested camera frame width. <=0 means keep driver default.",
	)
	parser.add_argument(
		"--height",
		type=int,
		default=720,
		help="Requested camera frame height. <=0 means keep driver default.",
	)
	parser.add_argument(
		"--output-root",
		type=str,
		default=str(Path(__file__).resolve().parent / "runs"),
		help="Base output directory for timestamped run folders.",
	)
	parser.add_argument(
		"--show-undistorted",
		action="store_true",
		help="Show live undistorted preview after calibration.",
	)
	return parser.parse_args()


def _camera_backend() -> Optional[int]:
	if platform.system().lower().startswith("win") and hasattr(cv2, "CAP_DSHOW"):
		return cv2.CAP_DSHOW
	return None


def open_camera(index: int, width: int, height: int) -> Optional[cv2.VideoCapture]:
	backend = _camera_backend()
	cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)

	if not cap.isOpened():
		cap.release()
		return None

	if width > 0:
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	if height > 0:
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

	for _ in range(12):
		ok, frame = cap.read()
		if ok and frame is not None:
			return cap

	cap.release()
	return None


def create_run_dir(output_root: Path) -> Path:
	output_root.mkdir(parents=True, exist_ok=True)
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = output_root / stamp
	run_dir.mkdir(parents=True, exist_ok=False)
	return run_dir


def _draw_status_lines(frame: np.ndarray, lines: List[str], color: Tuple[int, int, int]) -> np.ndarray:
	view = frame.copy()
	y = 28
	for line in lines:
		cv2.putText(view, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
		y += 28
	return view


def wait_for_confirm(cap: cv2.VideoCapture, camera_index: int) -> bool:
	while True:
		ok, frame = cap.read()
		if not ok or frame is None:
			frame = np.zeros((480, 640, 3), dtype=np.uint8)
			cv2.putText(
				frame,
				"No frame from camera",
				(30, 250),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.9,
				(0, 0, 255),
				2,
				cv2.LINE_AA,
			)

		frame = _draw_status_lines(
			frame,
			[
				f"Camera index: {camera_index}",
				"Press c to confirm this camera and start capture",
				"Press q or Esc to quit",
			],
			(0, 255, 0),
		)
		cv2.imshow(PREVIEW_WINDOW, frame)

		key = cv2.waitKey(1) & 0xFF
		if key == ord("c"):
			cv2.destroyWindow(PREVIEW_WINDOW)
			return True
		if key in (ord("q"), 27):
			cv2.destroyWindow(PREVIEW_WINDOW)
			return False


def _create_dictionary(name: str):
	if not hasattr(cv2, "aruco"):
		raise RuntimeError("OpenCV ArUco module is unavailable. Install opencv-contrib-python.")
	if not hasattr(cv2.aruco, name):
		raise ValueError(f"Unknown ArUco dictionary name: {name}")
	return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _create_detector_params():
	if hasattr(cv2.aruco, "DetectorParameters"):
		return cv2.aruco.DetectorParameters()
	if hasattr(cv2.aruco, "DetectorParameters_create"):
		return cv2.aruco.DetectorParameters_create()
	raise RuntimeError("ArUco DetectorParameters API is unavailable.")


def _create_grid_board(
	cols: int,
	rows: int,
	marker_size_mm: float,
	marker_gap_mm: float,
	aruco_dict,
	first_marker_id: int,
):
	ids = np.arange(first_marker_id, first_marker_id + rows * cols, dtype=np.int32)

	if hasattr(cv2.aruco, "GridBoard"):
		try:
			return cv2.aruco.GridBoard((cols, rows), marker_size_mm, marker_gap_mm, aruco_dict, ids)
		except TypeError:
			try:
				if first_marker_id != 0:
					raise RuntimeError(
						"Current OpenCV GridBoard constructor does not accept IDs. "
						"Use a board with first marker id 0 or upgrade OpenCV."
					)
				return cv2.aruco.GridBoard((cols, rows), marker_size_mm, marker_gap_mm, aruco_dict)
			except TypeError:
				pass
	if hasattr(cv2.aruco, "GridBoard_create"):
		try:
			return cv2.aruco.GridBoard_create(
				cols,
				rows,
				marker_size_mm,
				marker_gap_mm,
				aruco_dict,
				first_marker_id,
			)
		except TypeError:
			if first_marker_id != 0:
				raise RuntimeError(
					"Current OpenCV GridBoard_create does not accept firstMarker argument. "
					"Use a board with first marker id 0 or upgrade OpenCV."
				)
			return cv2.aruco.GridBoard_create(cols, rows, marker_size_mm, marker_gap_mm, aruco_dict)
	raise RuntimeError("ArUco GridBoard API is unavailable.")


def detect_markers(gray: np.ndarray, aruco_dict, detector_params):
	if hasattr(cv2.aruco, "ArucoDetector"):
		detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
		return detector.detectMarkers(gray)
	return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)


def capture_samples(
	cap: cv2.VideoCapture,
	run_dir: Path,
	aruco_dict,
	detector_params,
	min_markers: int,
) -> Tuple[List[List[np.ndarray]], List[np.ndarray], Optional[Tuple[int, int]], List[Path]]:
	detected_corners_per_frame: List[List[np.ndarray]] = []
	detected_ids_per_frame: List[np.ndarray] = []
	image_size: Optional[Tuple[int, int]] = None
	saved_paths: List[Path] = []

	while True:
		ok, frame = cap.read()
		if not ok or frame is None:
			frame = np.zeros((480, 640, 3), dtype=np.uint8)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		corners, ids, _ = detect_markers(gray, aruco_dict, detector_params)
		detected_count = int(len(ids)) if ids is not None else 0

		view = frame.copy()
		if ids is not None and detected_count > 0:
			cv2.aruco.drawDetectedMarkers(view, corners, ids)

		view = _draw_status_lines(
			view,
			[
				f"Captured images: {len(saved_paths)}",
				f"Detected markers now: {detected_count}",
				f"Enter: capture frame (requires >= {min_markers} markers)",
				"d: delete last capture   q/Esc: finish and calibrate",
			],
			(255, 255, 255),
		)

		cv2.imshow(CAPTURE_WINDOW, view)
		key = cv2.waitKey(1) & 0xFF

		if key in (ord("q"), 27):
			break

		if key == ord("d"):
			if saved_paths:
				last_path = saved_paths.pop()
				if last_path.exists():
					last_path.unlink()
				detected_corners_per_frame.pop()
				detected_ids_per_frame.pop()
				print(f"[INFO] Deleted last capture: {last_path.name}")
			else:
				print("[INFO] No captured frame to delete.")
			continue

		if key in (10, 13):
			if ids is None or detected_count < min_markers:
				print(f"[WARN] Need >= {min_markers} markers, currently {detected_count}.")
				continue

			image_size = (gray.shape[1], gray.shape[0])
			idx = len(saved_paths) + 1
			image_path = run_dir / f"image_{idx:03d}.png"
			cv2.imwrite(str(image_path), frame)

			corners_snapshot = [np.array(c, dtype=np.float32).copy() for c in corners]
			ids_snapshot = np.array(ids, dtype=np.int32).copy()
			detected_corners_per_frame.append(corners_snapshot)
			detected_ids_per_frame.append(ids_snapshot)
			saved_paths.append(image_path)
			print(f"[INFO] Captured {image_path.name} with {detected_count} markers.")

	cv2.destroyWindow(CAPTURE_WINDOW)
	return detected_corners_per_frame, detected_ids_per_frame, image_size, saved_paths


def calibrate_aruco_and_save(
	run_dir: Path,
	detected_corners_per_frame: List[List[np.ndarray]],
	detected_ids_per_frame: List[np.ndarray],
	image_size: Tuple[int, int],
	board,
	camera_index: int,
	rows: int,
	cols: int,
	marker_size: float,
	marker_gap: float,
	dict_name: str,
	first_marker_id: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
	if not hasattr(cv2.aruco, "calibrateCameraAruco"):
		raise RuntimeError("cv2.aruco.calibrateCameraAruco is unavailable in current OpenCV build.")

	all_corners: List[np.ndarray] = []
	all_ids: List[np.ndarray] = []
	counter: List[int] = []

	for corners_i, ids_i in zip(detected_corners_per_frame, detected_ids_per_frame):
		if ids_i is None or len(ids_i) == 0:
			continue
		all_corners.extend(corners_i)
		all_ids.extend(list(ids_i.reshape(-1, 1)))
		counter.append(int(len(ids_i)))

	if not all_corners or not all_ids:
		raise RuntimeError("No valid ArUco detections collected for calibration.")

	ids_array = np.array(all_ids, dtype=np.int32).reshape(-1, 1)
	counter_array = np.array(counter, dtype=np.int32)

	rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
		all_corners,
		ids_array,
		counter_array,
		board,
		image_size,
		None,
		None,
	)

	result_json = {
		"camera_index": camera_index,
		"image_size": [int(image_size[0]), int(image_size[1])],
		"board_rows": int(rows),
		"board_cols": int(cols),
		"marker_size_mm": float(marker_size),
		"marker_gap_mm": float(marker_gap),
		"dictionary": dict_name,
		"first_marker_id": int(first_marker_id),
		"image_count": len(detected_ids_per_frame),
		"rms": float(rms),
		"camera_matrix": camera_matrix.tolist(),
		"dist_coeffs": dist_coeffs.tolist(),
		"per_image_marker_count": [int(len(x)) for x in detected_ids_per_frame],
		"total_detected_markers": int(sum(len(x) for x in detected_ids_per_frame)),
	}

	json_path = run_dir / "calibration_result.json"
	json_path.write_text(json.dumps(result_json, indent=2), encoding="utf-8")

	npz_path = run_dir / "calibration_result.npz"
	np.savez(
		str(npz_path),
		camera_matrix=camera_matrix,
		dist_coeffs=dist_coeffs,
		image_size=np.array(image_size, dtype=np.int32),
		board_rows=np.array([rows], dtype=np.int32),
		board_cols=np.array([cols], dtype=np.int32),
		marker_size_mm=np.array([marker_size], dtype=np.float32),
		marker_gap_mm=np.array([marker_gap], dtype=np.float32),
		rms=np.array([rms], dtype=np.float64),
	)

	print("\n[RESULT] Calibration finished.")
	print(f"[RESULT] RMS: {rms:.6f}")
	if rms > 3.0:
		print("[WARN] RMS is high. Check board config (rows/cols/marker size/gap/first marker id) and capture diversity.")
	print(f"[RESULT] Saved JSON: {json_path}")
	print(f"[RESULT] Saved NPZ : {npz_path}")

	# Keep variables used to avoid lint false-positives in some editors.
	_ = (rvecs, tvecs)
	return camera_matrix, dist_coeffs, float(rms)


def preview_undistortion(
	cap: cv2.VideoCapture,
	camera_matrix: np.ndarray,
	dist_coeffs: np.ndarray,
) -> None:
	ok, frame = cap.read()
	if not ok or frame is None:
		print("[WARN] Cannot open undistorted preview because no frame is available.")
		return

	h, w = frame.shape[:2]
	new_mtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

	while True:
		ok, frame = cap.read()
		if not ok or frame is None:
			continue

		undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_mtx)
		merged = np.hstack([frame, undistorted])
		merged = _draw_status_lines(
			merged,
			["Left: original   Right: undistorted", "Press q or Esc to close"],
			(255, 255, 255),
		)

		cv2.imshow(UNDISTORT_WINDOW, merged)
		key = cv2.waitKey(1) & 0xFF
		if key in (ord("q"), 27):
			break

	cv2.destroyWindow(UNDISTORT_WINDOW)


def main() -> int:
	args = parse_args()

	if args.rows <= 0 or args.cols <= 0:
		print("[ERROR] --rows and --cols must both be > 0.")
		return 1
	if args.marker_size <= 0:
		print("[ERROR] --marker-size must be > 0.")
		return 1
	if args.marker_gap < 0:
		print("[ERROR] --marker-gap must be >= 0.")
		return 1
	if args.min_markers <= 0:
		print("[ERROR] --min-markers must be > 0.")
		return 1
	if args.first_marker_id < 0:
		print("[ERROR] --first-marker-id must be >= 0.")
		return 1

	try:
		aruco_dict = _create_dictionary(args.dict)
		detector_params = _create_detector_params()
		board = _create_grid_board(
			args.cols,
			args.rows,
			args.marker_size,
			args.marker_gap,
			aruco_dict,
			args.first_marker_id,
		)
	except Exception as exc:
		print(f"[ERROR] Failed to initialize ArUco components: {exc}")
		return 1

	cap = open_camera(args.camera_index, args.width, args.height)
	if cap is None:
		print(f"[ERROR] Failed to open camera index {args.camera_index}.")
		print("[TIP] Run: python calibrate/detect.py  to check available indexes.")
		return 1

	try:
		print(f"[INFO] Camera {args.camera_index} opened. Showing preview...")
		if not wait_for_confirm(cap, args.camera_index):
			print("[INFO] Cancelled by user before capture.")
			return 0

		run_dir = create_run_dir(Path(args.output_root))
		print(f"[INFO] Run directory created: {run_dir}")

		session_info = {
			"camera_index": args.camera_index,
			"board_rows": args.rows,
			"board_cols": args.cols,
			"marker_size_mm": args.marker_size,
			"marker_gap_mm": args.marker_gap,
			"dictionary": args.dict,
			"first_marker_id": args.first_marker_id,
			"min_markers": args.min_markers,
			"min_samples": args.min_samples,
			"requested_size": [args.width, args.height],
		}
		(run_dir / "session_info.json").write_text(json.dumps(session_info, indent=2), encoding="utf-8")

		detected_corners_per_frame, detected_ids_per_frame, image_size, saved_paths = capture_samples(
			cap,
			run_dir,
			aruco_dict,
			detector_params,
			args.min_markers,
		)

		if len(saved_paths) < args.min_samples:
			print(
				f"[ERROR] Only {len(saved_paths)} images captured, "
				f"but at least {args.min_samples} are required."
			)
			print(f"[INFO] Captured images remain in: {run_dir}")
			return 1

		if image_size is None:
			print("[ERROR] Image size is unknown; no valid capture was recorded.")
			return 1

		try:
			camera_matrix, dist_coeffs, _ = calibrate_aruco_and_save(
				run_dir,
				detected_corners_per_frame,
				detected_ids_per_frame,
				image_size,
				board,
				args.camera_index,
				args.rows,
				args.cols,
				args.marker_size,
				args.marker_gap,
				args.dict,
				args.first_marker_id,
			)
		except Exception as exc:
			print(f"[ERROR] Calibration failed: {exc}")
			print("[TIP] Try collecting more diverse angles/distances and ensure markers are clearly visible.")
			return 1

		if args.show_undistorted:
			print("[INFO] Showing undistorted preview...")
			preview_undistortion(cap, camera_matrix, dist_coeffs)

		print(f"[DONE] All outputs saved under: {run_dir}")
		return 0
	finally:
		cap.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	raise SystemExit(main())
