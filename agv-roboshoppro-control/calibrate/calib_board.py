"""Interactive ChArUco monocular camera calibration.

Default board parameters match the requested board:
- ChArUco 9x14
- checker(square) size: 20 mm
- marker size: 15 mm
- ArUco dictionary: 5x5 (AUTO_5X5 picks the best 5x5 dictionary)

Workflow:
1) Open camera preview and press "c" to start.
2) Capture target images (default 20) with Space/Enter.
3) Per-frame quality gates reject blurry/overexposed/partial-board images.
4) Script calibrates and saves camera intrinsics with per-image reprojection errors.

Example:
    python calibrate/calib_board.py --camera-index 1 --target-count 20
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


PREVIEW_WINDOW = "Charuco Calibration - Preview"
CAPTURE_WINDOW = "Charuco Calibration - Capture"

AUTO_5X5_DICTIONARIES = (
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
)


@dataclass
class BoardCandidate:
    name: str
    board: object


@dataclass
class CapturedSample:
    image_path: Path
    marker_corners: list[np.ndarray]
    marker_ids: np.ndarray


@dataclass
class QualityReport:
    ok: bool
    reasons: list[str]
    blur_var: float
    over_ratio: float
    under_ratio: float
    reflection_ratio: float
    illumination_ratio: float
    marker_ratio: float
    edge_margin_px: int
    board_area_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive ChArUco monocular calibration.")
    parser.add_argument("--camera-index", type=int, default=1, help="Camera index. Default: 1")
    parser.add_argument("--target-count", type=int, default=20, help="Target capture count. Default: 20")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum images for calibration. Default: 10")
    parser.add_argument("--min-markers", type=int, default=8, help="Minimum markers for capture. Default: 8")
    parser.add_argument(
        "--min-charuco",
        type=int,
        default=12,
        help="Minimum interpolated ChArUco corners per frame. Default: 12",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="AUTO_5X5",
        help="ArUco dictionary name or AUTO_5X5. Default: AUTO_5X5",
    )
    parser.add_argument("--squares-x", type=int, default=9, help="ChArUco squares count along X. Default: 9")
    parser.add_argument("--squares-y", type=int, default=14, help="ChArUco squares count along Y. Default: 14")
    parser.add_argument("--square-size-mm", type=float, default=20.0, help="Square size in mm. Default: 20")
    parser.add_argument("--marker-size-mm", type=float, default=15.0, help="Marker size in mm. Default: 15")
    parser.add_argument("--width", type=int, default=1280, help="Requested width. <=0 keeps camera default")
    parser.add_argument("--height", type=int, default=720, help="Requested height. <=0 keeps camera default")
    parser.add_argument(
        "--keep-autofocus",
        action="store_true",
        help="Keep autofocus enabled. By default the script disables autofocus for calibration stability.",
    )
    parser.add_argument(
        "--manual-focus",
        type=float,
        default=-1.0,
        help="Manual focus value (camera-dependent). <0 means reuse current focus value after autofocus is disabled.",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=120.0,
        help="Minimum Laplacian variance for sharp image acceptance. Default: 120",
    )
    parser.add_argument(
        "--max-overexposed-ratio",
        type=float,
        default=0.015,
        help="Maximum ratio of pixels >= overexposed-threshold. Default: 0.015",
    )
    parser.add_argument(
        "--max-underexposed-ratio",
        type=float,
        default=0.020,
        help="Maximum ratio of pixels <= underexposed-threshold. Default: 0.020",
    )
    parser.add_argument(
        "--overexposed-threshold",
        type=int,
        default=245,
        help="Gray threshold considered overexposed. Default: 245",
    )
    parser.add_argument(
        "--underexposed-threshold",
        type=int,
        default=10,
        help="Gray threshold considered underexposed. Default: 10",
    )
    parser.add_argument(
        "--max-reflection-ratio",
        type=float,
        default=0.010,
        help="Maximum high-light ratio in board ROI to reject reflections. Default: 0.010",
    )
    parser.add_argument(
        "--min-illumination-ratio",
        type=float,
        default=0.55,
        help="Minimum block-mean ratio in board ROI to reduce strong shadow. Default: 0.55",
    )
    parser.add_argument(
        "--min-marker-ratio",
        type=float,
        default=0.55,
        help="Minimum detected-marker ratio against full board markers. Default: 0.55",
    )
    parser.add_argument(
        "--min-edge-margin-px",
        type=int,
        default=15,
        help="Minimum board-to-image edge margin in pixels. Default: 15",
    )
    parser.add_argument(
        "--min-board-area-ratio",
        type=float,
        default=0.08,
        help="Minimum board bounding-box area ratio in image. Default: 0.08",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs_charuco"),
        help="Output root directory.",
    )
    return parser.parse_args()


def ensure_aruco_available() -> None:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is unavailable. Install opencv-contrib-python.")


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

    for _ in range(10):
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap

    cap.release()
    return None


def configure_focus_lock(cap: cv2.VideoCapture, args: argparse.Namespace) -> None:
    if args.keep_autofocus:
        print("[INFO] keep-autofocus is enabled; autofocus settings were not changed.")
        return

    if not hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
        print("[WARN] CAP_PROP_AUTOFOCUS is unavailable on this OpenCV build.")
    else:
        autofocus_prop = getattr(cv2, "CAP_PROP_AUTOFOCUS")
        before = cap.get(autofocus_prop)
        ok = cap.set(autofocus_prop, 0)
        after = cap.get(autofocus_prop)
        print(f"[INFO] Autofocus lock request: ok={ok} before={before:.3f} after={after:.3f}")

    if not hasattr(cv2, "CAP_PROP_FOCUS"):
        print("[WARN] CAP_PROP_FOCUS is unavailable on this camera/backend.")
        return

    focus_prop = getattr(cv2, "CAP_PROP_FOCUS")
    current_focus = cap.get(focus_prop)
    target_focus = args.manual_focus if args.manual_focus >= 0 else current_focus
    ok = cap.set(focus_prop, float(target_focus))
    after = cap.get(focus_prop)
    print(f"[INFO] Manual focus lock request: ok={ok} target={target_focus:.3f} after={after:.3f}")


def _marker_bbox(marker_corners: list[np.ndarray], width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    if not marker_corners:
        return None

    pts = np.concatenate(marker_corners, axis=0).reshape(-1, 2)
    x0 = int(np.clip(np.floor(np.min(pts[:, 0])), 0, width - 1))
    y0 = int(np.clip(np.floor(np.min(pts[:, 1])), 0, height - 1))
    x1 = int(np.clip(np.ceil(np.max(pts[:, 0])), 0, width - 1))
    y1 = int(np.clip(np.ceil(np.max(pts[:, 1])), 0, height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _illumination_ratio(gray_roi: np.ndarray) -> float:
    if gray_roi.size == 0:
        return 0.0

    block_means: list[float] = []
    rows = np.array_split(gray_roi, 4, axis=0)
    for row_block in rows:
        cols = np.array_split(row_block, 4, axis=1)
        for cell in cols:
            if cell.size > 0:
                block_means.append(float(np.mean(cell)))

    if not block_means:
        return 0.0

    min_v = float(np.min(block_means))
    max_v = float(np.max(block_means))
    if max_v <= 1e-6:
        return 0.0
    return float(min_v / max_v)


def evaluate_frame_quality(
    gray: np.ndarray,
    marker_corners: list[np.ndarray],
    marker_count: int,
    expected_markers: int,
    args: argparse.Namespace,
) -> QualityReport:
    height, width = gray.shape[:2]
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    over_ratio = float(np.mean(gray >= int(args.overexposed_threshold)))
    under_ratio = float(np.mean(gray <= int(args.underexposed_threshold)))

    bbox = _marker_bbox(marker_corners, width, height)
    if bbox is None:
        roi = gray
        edge_margin_px = -1
        board_area_ratio = 0.0
    else:
        x0, y0, x1, y1 = bbox
        roi = gray[y0 : y1 + 1, x0 : x1 + 1]
        edge_margin_px = int(min(x0, y0, width - 1 - x1, height - 1 - y1))
        board_area_ratio = float(((x1 - x0 + 1) * (y1 - y0 + 1)) / float(width * height))

    reflection_ratio = float(np.mean(roi >= int(args.overexposed_threshold))) if roi.size > 0 else 1.0
    illumination_ratio = _illumination_ratio(roi)

    safe_expected = max(1, int(expected_markers))
    marker_ratio = float(marker_count / safe_expected)

    reasons: list[str] = []
    if blur_var < float(args.blur_threshold):
        reasons.append(f"blur({blur_var:.1f}<{args.blur_threshold:.1f})")
    if over_ratio > float(args.max_overexposed_ratio):
        reasons.append(f"overexp({over_ratio:.3f}>{args.max_overexposed_ratio:.3f})")
    if under_ratio > float(args.max_underexposed_ratio):
        reasons.append(f"underexp({under_ratio:.3f}>{args.max_underexposed_ratio:.3f})")
    if reflection_ratio > float(args.max_reflection_ratio):
        reasons.append(f"reflection({reflection_ratio:.3f}>{args.max_reflection_ratio:.3f})")
    if illumination_ratio < float(args.min_illumination_ratio):
        reasons.append(f"shadow({illumination_ratio:.2f}<{args.min_illumination_ratio:.2f})")
    if marker_ratio < float(args.min_marker_ratio):
        reasons.append(f"board_partial({marker_ratio:.2f}<{args.min_marker_ratio:.2f})")
    if edge_margin_px < int(args.min_edge_margin_px):
        reasons.append(f"board_cut(edge_margin={edge_margin_px})")
    if board_area_ratio < float(args.min_board_area_ratio):
        reasons.append(f"board_small({board_area_ratio:.3f}<{args.min_board_area_ratio:.3f})")

    return QualityReport(
        ok=(len(reasons) == 0),
        reasons=reasons,
        blur_var=blur_var,
        over_ratio=over_ratio,
        under_ratio=under_ratio,
        reflection_ratio=reflection_ratio,
        illumination_ratio=illumination_ratio,
        marker_ratio=marker_ratio,
        edge_margin_px=edge_margin_px,
        board_area_ratio=board_area_ratio,
    )


def create_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"session_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    return run_dir


def draw_status(frame: np.ndarray, lines: list[str], color: tuple[int, int, int]) -> np.ndarray:
    view = frame.copy()
    y = 28
    for line in lines:
        cv2.putText(view, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += 28
    return view


def wait_for_confirm(cap: cv2.VideoCapture, camera_index: int) -> bool:
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        view = draw_status(
            frame,
            [
                f"Camera index: {camera_index}",
                "Press c to start capture",
                "Press q or Esc to quit",
            ],
            (0, 255, 0),
        )
        cv2.imshow(PREVIEW_WINDOW, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            cv2.destroyWindow(PREVIEW_WINDOW)
            return True
        if key in (ord("q"), 27):
            cv2.destroyWindow(PREVIEW_WINDOW)
            return False


def create_detector_params():
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    raise RuntimeError("ArUco DetectorParameters API is unavailable.")


def make_detector(dictionary_name: str) -> tuple[object, Callable[[np.ndarray], tuple[list[np.ndarray], Optional[np.ndarray]]]]:
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary name: {dictionary_name}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    params = create_detector_params()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        def _detect(gray: np.ndarray) -> tuple[list[np.ndarray], Optional[np.ndarray]]:
            corners, ids, _ = detector.detectMarkers(gray)
            return list(corners), ids

        return aruco_dict, _detect

    def _detect(gray: np.ndarray) -> tuple[list[np.ndarray], Optional[np.ndarray]]:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        return list(corners), ids

    return aruco_dict, _detect


def resolve_dictionary_name(input_name: str, sample_gray: np.ndarray) -> str:
    if input_name != "AUTO_5X5":
        if not hasattr(cv2.aruco, input_name):
            raise ValueError(f"Unknown ArUco dictionary name: {input_name}")
        return input_name

    best_name: Optional[str] = None
    best_count = -1

    for candidate in AUTO_5X5_DICTIONARIES:
        if not hasattr(cv2.aruco, candidate):
            continue
        _, detect = make_detector(candidate)
        _corners, ids = detect(sample_gray)
        count = 0 if ids is None else int(ids.size)
        if count > best_count:
            best_count = count
            best_name = candidate

    if best_name is None:
        raise RuntimeError("No 5x5 ArUco dictionary is available in current OpenCV build.")
    if best_count <= 0:
        raise RuntimeError("AUTO_5X5 could not detect any marker in preview frame.")

    print(f"[INFO] AUTO_5X5 selected dictionary: {best_name} (markers={best_count})")
    return best_name


def create_charuco_board(
    squares_x: int,
    squares_y: int,
    square_size_mm: float,
    marker_size_mm: float,
    aruco_dict,
    legacy_pattern: bool,
):
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


def build_board_candidates(
    squares_x: int,
    squares_y: int,
    square_size_mm: float,
    marker_size_mm: float,
    aruco_dict,
) -> list[BoardCandidate]:
    candidates: list[BoardCandidate] = []
    orientations = [(squares_x, squares_y)]
    if squares_x != squares_y:
        orientations.append((squares_y, squares_x))

    for sx, sy in orientations:
        for legacy in (False, True):
            name = f"sx{sx}_sy{sy}_legacy{int(legacy)}"
            board = create_charuco_board(sx, sy, square_size_mm, marker_size_mm, aruco_dict, legacy)
            candidates.append(BoardCandidate(name=name, board=board))

    return candidates


def interpolate_charuco(
    gray: np.ndarray,
    marker_corners: list[np.ndarray],
    marker_ids: np.ndarray,
    board,
) -> tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
        )
    except TypeError:
        num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=marker_corners,
            markerIds=marker_ids,
            image=gray,
            board=board,
        )

    count = int(num) if num is not None else 0
    if count <= 0 or charuco_corners is None or charuco_ids is None:
        return 0, None, None
    return count, charuco_corners, charuco_ids


def frame_best_charuco_count(
    gray: np.ndarray,
    marker_corners: list[np.ndarray],
    marker_ids: np.ndarray,
    candidates: list[BoardCandidate],
) -> int:
    best = 0
    for candidate in candidates:
        count, _corners, _ids = interpolate_charuco(gray, marker_corners, marker_ids, candidate.board)
        if count > best:
            best = count
    return best


def capture_samples(
    cap: cv2.VideoCapture,
    run_dir: Path,
    target_count: int,
    min_markers: int,
    min_charuco: int,
    detect: Callable[[np.ndarray], tuple[list[np.ndarray], Optional[np.ndarray]]],
    board_candidates: list[BoardCandidate],
) -> tuple[list[CapturedSample], tuple[int, int]]:
    samples: list[CapturedSample] = []
    image_size = (0, 0)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = (int(gray.shape[1]), int(gray.shape[0]))

        marker_corners, marker_ids = detect(gray)
        marker_count = 0 if marker_ids is None else int(marker_ids.size)
        charuco_count = 0

        view = frame.copy()
        if marker_ids is not None and marker_count > 0:
            cv2.aruco.drawDetectedMarkers(view, marker_corners, marker_ids)
            charuco_count = frame_best_charuco_count(gray, marker_corners, marker_ids, board_candidates)

        view = draw_status(
            view,
            [
                f"Captured: {len(samples)}/{target_count}",
                f"Detected markers: {marker_count} (need >= {min_markers})",
                f"Best ChArUco corners: {charuco_count} (need >= {min_charuco})",
                "Space/Enter: capture  d: delete last  q/Esc: finish",
            ],
            (255, 255, 255),
        )
        cv2.imshow(CAPTURE_WINDOW, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        if key == ord("d"):
            if samples:
                last = samples.pop()
                if last.image_path.exists():
                    last.image_path.unlink()
                print(f"[INFO] Deleted: {last.image_path.name}")
            continue

        if key in (ord(" "), 10, 13):
            if marker_ids is None or marker_count < min_markers:
                print(f"[WARN] marker count {marker_count} < {min_markers}")
                continue
            if charuco_count < min_charuco:
                print(f"[WARN] charuco count {charuco_count} < {min_charuco}")
                continue

            idx = len(samples) + 1
            image_path = run_dir / "images" / f"image_{idx:03d}.png"
            cv2.imwrite(str(image_path), frame)

            samples.append(
                CapturedSample(
                    image_path=image_path,
                    marker_corners=[np.array(c, dtype=np.float32).copy() for c in marker_corners],
                    marker_ids=np.array(marker_ids, dtype=np.int32).copy(),
                )
            )
            print(f"[INFO] Captured: {image_path.name} markers={marker_count} charuco={charuco_count}")

            if len(samples) >= target_count:
                break

    cv2.destroyWindow(CAPTURE_WINDOW)
    return samples, image_size


def calibrate_charuco(
    charuco_corners: list[np.ndarray],
    charuco_ids: list[np.ndarray],
    board,
    image_size: tuple[int, int],
):
    if hasattr(cv2.aruco, "calibrateCameraCharuco"):
        try:
            return cv2.aruco.calibrateCameraCharuco(
                charuco_corners,
                charuco_ids,
                board,
                image_size,
                None,
                None,
            )
        except TypeError:
            return cv2.aruco.calibrateCameraCharuco(
                charuco_corners,
                charuco_ids,
                board,
                image_size,
            )

    if hasattr(cv2.aruco, "calibrateCameraCharucoExtended"):
        return cv2.aruco.calibrateCameraCharucoExtended(
            charuco_corners,
            charuco_ids,
            board,
            image_size,
            None,
            None,
        )

    raise RuntimeError("calibrateCameraCharuco is unavailable in current OpenCV build.")


def get_board_chessboard_corners(board) -> Optional[np.ndarray]:
    if hasattr(board, "getChessboardCorners"):
        return np.array(board.getChessboardCorners(), dtype=np.float32)
    if hasattr(board, "chessboardCorners"):
        return np.array(board.chessboardCorners, dtype=np.float32)
    return None


def compute_mean_reprojection_error(
    board,
    charuco_corners: list[np.ndarray],
    charuco_ids: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Optional[float]:
    board_corners = get_board_chessboard_corners(board)
    if board_corners is None:
        return None

    errors = []
    for corners_i, ids_i, rvec, tvec in zip(charuco_corners, charuco_ids, rvecs, tvecs):
        if ids_i is None or len(ids_i) == 0:
            continue
        indices = ids_i.reshape(-1)
        if np.max(indices) >= len(board_corners):
            continue

        obj_points = board_corners[indices]
        proj, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
        err = cv2.norm(corners_i, proj, cv2.NORM_L2) / max(len(proj), 1)
        errors.append(float(err))

    if not errors:
        return None
    return float(np.mean(errors))


def choose_best_calibration(
    samples: list[CapturedSample],
    board_candidates: list[BoardCandidate],
    image_size: tuple[int, int],
    min_charuco: int,
    min_samples: int,
) -> tuple[dict, list[dict]]:
    results: list[dict] = []

    for candidate in board_candidates:
        charuco_corners_all: list[np.ndarray] = []
        charuco_ids_all: list[np.ndarray] = []

        for sample in samples:
            image = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            count, charuco_corners, charuco_ids = interpolate_charuco(
                image,
                sample.marker_corners,
                sample.marker_ids,
                candidate.board,
            )
            if count < min_charuco or charuco_corners is None or charuco_ids is None:
                continue

            charuco_corners_all.append(charuco_corners)
            charuco_ids_all.append(charuco_ids)

        used_count = len(charuco_corners_all)
        if used_count < min_samples:
            results.append(
                {
                    "candidate": candidate.name,
                    "used_images": used_count,
                    "ok": False,
                    "reason": f"used_images {used_count} < min_samples {min_samples}",
                }
            )
            continue

        calib_out = calibrate_charuco(charuco_corners_all, charuco_ids_all, candidate.board, image_size)
        rms = float(calib_out[0])
        camera_matrix = np.array(calib_out[1], dtype=np.float64)
        dist_coeffs = np.array(calib_out[2], dtype=np.float64)
        rvecs = list(calib_out[3])
        tvecs = list(calib_out[4])

        mean_error = compute_mean_reprojection_error(
            candidate.board,
            charuco_corners_all,
            charuco_ids_all,
            rvecs,
            tvecs,
            camera_matrix,
            dist_coeffs,
        )

        results.append(
            {
                "candidate": candidate.name,
                "used_images": used_count,
                "ok": True,
                "rms": rms,
                "mean_reprojection_error": mean_error,
                "camera_matrix": camera_matrix,
                "dist_coeffs": dist_coeffs,
            }
        )

    ok_results = [r for r in results if r.get("ok")]
    if not ok_results:
        raise RuntimeError("No board candidate produced a valid calibration result.")

    ok_results.sort(key=lambda r: (-int(r["used_images"]), float(r["rms"])))
    best = ok_results[0]

    summary: list[dict] = []
    for row in results:
        entry = {
            "candidate": row["candidate"],
            "used_images": row["used_images"],
            "ok": row["ok"],
        }
        if row["ok"]:
            entry["rms"] = float(row["rms"])
            entry["mean_reprojection_error"] = row["mean_reprojection_error"]
        else:
            entry["reason"] = row.get("reason")
        summary.append(entry)

    return best, summary


def save_results(
    run_dir: Path,
    args: argparse.Namespace,
    resolved_dict_name: str,
    image_size: tuple[int, int],
    samples: list[CapturedSample],
    best: dict,
    candidate_summary: list[dict],
) -> None:
    camera_matrix = best["camera_matrix"]
    dist_coeffs = best["dist_coeffs"]

    result_json = {
        "camera_index": int(args.camera_index),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "target_count": int(args.target_count),
        "captured_count": len(samples),
        "dictionary": resolved_dict_name,
        "squares_x": int(args.squares_x),
        "squares_y": int(args.squares_y),
        "square_size_mm": float(args.square_size_mm),
        "marker_size_mm": float(args.marker_size_mm),
        "best_candidate": best["candidate"],
        "used_images": int(best["used_images"]),
        "rms": float(best["rms"]),
        "mean_reprojection_error": best["mean_reprojection_error"],
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(1, -1).tolist(),
        "candidate_summary": candidate_summary,
    }

    json_path = run_dir / "intrinsics.json"
    json_path.write_text(json.dumps(result_json, ensure_ascii=False, indent=2), encoding="utf-8")

    np.savez(
        run_dir / "intrinsics.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms=np.array([best["rms"]], dtype=np.float64),
    )

    if yaml is not None:
        yaml_payload = {
            "camera_1": {
                "camera_index": int(args.camera_index),
                "image_size": [int(image_size[0]), int(image_size[1])],
                "dictionary": resolved_dict_name,
                "squares_x": int(args.squares_x),
                "squares_y": int(args.squares_y),
                "square_size_mm": float(args.square_size_mm),
                "marker_size_mm": float(args.marker_size_mm),
                "image_count": int(best["used_images"]),
                "rms": float(best["rms"]),
                "mean_reprojection_error": best["mean_reprojection_error"],
                "camera_matrix": camera_matrix.tolist(),
                "dist_coeffs": dist_coeffs.reshape(1, -1).tolist(),
            }
        }
        with (run_dir / "camera_cfg.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_payload, f, sort_keys=False, allow_unicode=True)


def main() -> int:
    args = parse_args()
    ensure_aruco_available()

    if args.marker_size_mm >= args.square_size_mm:
        print("[ERROR] marker-size-mm must be smaller than square-size-mm.")
        return 2

    cap = open_camera(args.camera_index, args.width, args.height)
    if cap is None:
        print(f"[ERROR] Failed to open camera index {args.camera_index}")
        return 2

    try:
        if not wait_for_confirm(cap, args.camera_index):
            print("[INFO] Exit before capture.")
            return 0

        ok, sample_frame = cap.read()
        if not ok or sample_frame is None:
            raise RuntimeError("Failed to read sample frame from camera.")
        sample_gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)

        resolved_dict_name = resolve_dictionary_name(args.dictionary, sample_gray)
        aruco_dict, detect = make_detector(resolved_dict_name)
        board_candidates = build_board_candidates(
            args.squares_x,
            args.squares_y,
            args.square_size_mm,
            args.marker_size_mm,
            aruco_dict,
        )

        run_dir = create_run_dir(Path(args.output_root))
        print(f"[INFO] Output directory: {run_dir}")
        print(f"[INFO] Using dictionary: {resolved_dict_name}")
        print(f"[INFO] Board candidates: {[c.name for c in board_candidates]}")

        samples, image_size = capture_samples(
            cap=cap,
            run_dir=run_dir,
            target_count=args.target_count,
            min_markers=args.min_markers,
            min_charuco=args.min_charuco,
            detect=detect,
            board_candidates=board_candidates,
        )

        if len(samples) < args.min_samples:
            raise RuntimeError(
                f"Captured {len(samples)} images, which is below min_samples={args.min_samples}."
            )

        best, summary = choose_best_calibration(
            samples=samples,
            board_candidates=board_candidates,
            image_size=image_size,
            min_charuco=args.min_charuco,
            min_samples=args.min_samples,
        )

        save_results(
            run_dir=run_dir,
            args=args,
            resolved_dict_name=resolved_dict_name,
            image_size=image_size,
            samples=samples,
            best=best,
            candidate_summary=summary,
        )

        print("[INFO] Calibration completed.")
        print(f"[INFO] best_candidate={best['candidate']} used_images={best['used_images']}")
        print(f"[INFO] RMS={best['rms']:.6f}")
        print("[INFO] Saved: intrinsics.json, intrinsics.npz, camera_cfg.yaml(if PyYAML is installed)")
        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())