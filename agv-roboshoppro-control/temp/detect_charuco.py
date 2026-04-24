"""ChArUco board detection and visualization.

Auto-enumerates common ArUco dictionaries to locate markers, then attempts
ChArUco corner interpolation for several plausible board layouts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


DICT_CANDIDATES: dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

# Common ChArUco layouts (cols, rows) — cells of squares, not interior corners.
BOARD_LAYOUT_CANDIDATES: list[tuple[int, int]] = [
    (12, 9),
    (11, 8),
    (10, 7),
    (9, 6),
    (8, 6),
    (8, 5),
    (7, 5),
    (6, 4),
    (5, 7),
    (6, 9),
    (13, 9),
    (14, 10),
]


def build_detector_params() -> cv2.aruco.DetectorParameters:
    """Build ArUco detector parameters tuned for partially motion-blurred IR frames."""
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 35
    params.adaptiveThreshWinSizeStep = 6
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.04
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params


def enumerate_dictionaries(
    gray: np.ndarray,
    params: cv2.aruco.DetectorParameters,
) -> list[tuple[str, int, int, np.ndarray, list[np.ndarray]]]:
    """Run detection with every candidate dictionary.

    Returns:
        List of (name, dict_enum, n_markers, ids, corners) sorted by marker count.
    """
    results: list[tuple[str, int, int, np.ndarray, list[np.ndarray]]] = []
    for name, dict_id in DICT_CANDIDATES.items():
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _rejected = detector.detectMarkers(gray)
        n = 0 if ids is None else int(ids.size)
        if n > 0:
            results.append((name, dict_id, n, ids, list(corners)))
    results.sort(key=lambda row: row[2], reverse=True)
    return results


def try_charuco_interpolate(
    gray: np.ndarray,
    image_corners: list[np.ndarray],
    image_ids: np.ndarray,
    dict_id: int,
    board_size: tuple[int, int],
    square_length: float = 0.04,
    marker_length: float = 0.03,
) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    """Attempt ChArUco corner interpolation for a hypothesized board layout.

    Returns:
        (num_interpolated_corners, charuco_corners, charuco_ids)
    """
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        board_size,
        square_length,
        marker_length,
        dictionary,
    )
    num, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=image_corners,
        markerIds=image_ids,
        image=gray,
        board=board,
    )
    return int(num), ch_corners, ch_ids


def search_best_board_layout(
    gray: np.ndarray,
    image_corners: list[np.ndarray],
    image_ids: np.ndarray,
    dict_id: int,
) -> tuple[tuple[int, int], int, np.ndarray | None, np.ndarray | None]:
    """Search common layouts to find the one that interpolates the most corners."""
    best: tuple[tuple[int, int], int, np.ndarray | None, np.ndarray | None] = (
        (0, 0),
        0,
        None,
        None,
    )
    max_marker_id = int(image_ids.max())
    for cols, rows in BOARD_LAYOUT_CANDIDATES:
        markers_available = (cols * rows) // 2
        # Skip layouts whose marker count cannot accommodate observed IDs.
        if max_marker_id >= markers_available:
            continue
        try:
            num, ch_corners, ch_ids = try_charuco_interpolate(
                gray, image_corners, image_ids, dict_id, (cols, rows)
            )
        except cv2.error:
            continue
        if num > best[1]:
            best = ((cols, rows), num, ch_corners, ch_ids)
    return best


def render_visualization(
    image_bgr: np.ndarray,
    marker_corners: list[np.ndarray],
    marker_ids: np.ndarray,
    charuco_corners: np.ndarray | None,
    charuco_ids: np.ndarray | None,
) -> np.ndarray:
    """Render markers and ChArUco corners onto a copy of the image."""
    vis = image_bgr.copy()
    cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids, borderColor=(0, 255, 0))
    if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(
            vis,
            charuco_corners,
            charuco_ids,
            cornerColor=(0, 0, 255),
        )
    return vis


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(
            "/home/u24/ws_lq/sudo/aaa_useful_scripts/V4L2/multi_hard_sync/V4L2/"
            "obs_data/session_20260417_163314/TestIR/motion_check/002828.jpg"
        ),
        help="Input image path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/u24/ws_lq/tmp/charuco_detected.png"),
        help="Annotated output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also show the result in a GUI window (requires display).",
    )
    return parser.parse_args()


def main() -> int:
    """Script entrypoint."""
    args = parse_args()

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"[ERROR] Failed to read image: {args.image}", file=sys.stderr)
        return 2
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    print(f"[INFO] Image loaded: {image_bgr.shape[1]}x{image_bgr.shape[0]}")

    params = build_detector_params()
    results = enumerate_dictionaries(gray, params)
    if not results:
        print("[ERROR] No ArUco markers detected under any candidate dictionary.")
        return 3

    print("[INFO] Dictionary enumeration — top candidates:")
    for name, _dict_id, n, ids, _corners in results[:5]:
        id_list = sorted(int(v) for v in ids.ravel().tolist())
        print(f"  {name:25s} markers={n:3d}  ids={id_list}")

    best_name, best_dict_id, _n_markers, best_ids, best_corners = results[0]
    print(f"[INFO] Best dictionary: {best_name}")

    (cols, rows), num_charuco, ch_corners, ch_ids = search_best_board_layout(
        gray, best_corners, best_ids, best_dict_id
    )
    if num_charuco > 0:
        print(
            f"[INFO] ChArUco layout match: squares={cols}x{rows}, "
            f"interpolated_corners={num_charuco}"
        )
    else:
        print("[WARN] No ChArUco layout in the candidate set produced interpolated corners.")

    vis = render_visualization(image_bgr, best_corners, best_ids, ch_corners, ch_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), vis)
    print(f"[INFO] Annotated image saved: {args.output}")

    if args.show:
        cv2.imshow("charuco detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
