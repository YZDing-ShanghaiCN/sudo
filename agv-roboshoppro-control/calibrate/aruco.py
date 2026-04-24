"""Generate a printable A4 ArUco board for calibration.

Key points to avoid gray/antialias artifacts:
- Render at high DPI (default 300) so marker edges are sharp when printed.
- Force strict binary output (only 0 and 255 pixel values).
- Preview with nearest-neighbor scaling for faithful on-screen inspection.

Run:
    python calibrate/aruco.py
    python calibrate/aruco.py --dpi 600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _draw_marker(aruco_dict, marker_id: int, marker_size_px: int) -> np.ndarray:
    """Draw one ArUco marker across OpenCV API versions."""
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = np.zeros((marker_size_px, marker_size_px), dtype=np.uint8)
        cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px, marker, 1)
        return marker

    if hasattr(cv2.aruco, "drawMarker"):
        return cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size_px)

    raise RuntimeError("OpenCV ArUco marker drawing API not available.")


def _mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def _ensure_binary(img: np.ndarray) -> np.ndarray:
    # Keep only black(0) and white(255) pixels.
    return np.where(img > 127, 255, 0).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a printable ArUco board image.")
    parser.add_argument("--rows", type=int, default=4, help="Marker rows. Default: 4")
    parser.add_argument("--cols", type=int, default=6, help="Marker cols. Default: 6")
    parser.add_argument("--marker-size-mm", type=float, default=30.0, help="Marker size in mm. Default: 30")
    parser.add_argument("--gap-mm", type=float, default=5.0, help="Gap between markers in mm. Default: 5")
    parser.add_argument("--paper-width-mm", type=float, default=210.0, help="Paper width in mm. Default: 210")
    parser.add_argument("--paper-height-mm", type=float, default=297.0, help="Paper height in mm. Default: 297")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI. Default: 300")
    parser.add_argument("--dict", type=str, default="DICT_4X4_50", help="ArUco dictionary name. Default: DICT_4X4_50")
    parser.add_argument("--first-marker-id", type=int, default=1, help="First marker ID. Default: 1")
    return parser.parse_args()


def build_aruco_board(args: argparse.Namespace) -> np.ndarray:
    if args.first_marker_id < 0:
        raise ValueError("--first-marker-id must be >= 0")

    marker_size_px = _mm_to_px(args.marker_size_mm, args.dpi)
    gap_px = _mm_to_px(args.gap_mm, args.dpi)
    img_width_px = _mm_to_px(args.paper_width_mm, args.dpi)
    img_height_px = _mm_to_px(args.paper_height_mm, args.dpi)

    img = np.full((img_height_px, img_width_px), 255, dtype=np.uint8)

    board_width_px = args.cols * marker_size_px + (args.cols - 1) * gap_px
    board_height_px = args.rows * marker_size_px + (args.rows - 1) * gap_px
    if board_width_px > img_width_px or board_height_px > img_height_px:
        raise ValueError("Markers do not fit on A4 with current size and gap.")

    start_x = (img_width_px - board_width_px) // 2
    start_y = (img_height_px - board_height_px) // 2

    if not hasattr(cv2.aruco, args.dict):
        raise ValueError(f"Unknown ArUco dictionary: {args.dict}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))

    marker_id = int(args.first_marker_id)
    for row in range(args.rows):
        for col in range(args.cols):
            x_pos = start_x + col * (marker_size_px + gap_px)
            y_pos = start_y + row * (marker_size_px + gap_px)
            marker_img = _draw_marker(aruco_dict, marker_id, marker_size_px)
            marker_img = _ensure_binary(marker_img)
            img[y_pos : y_pos + marker_size_px, x_pos : x_pos + marker_size_px] = marker_img
            marker_id += 1

    return _ensure_binary(img)


def main() -> None:
    args = parse_args()
    img = build_aruco_board(args)

    output_path = Path(__file__).resolve().parent / "aruco_board.png"
    cv2.imwrite(str(output_path), img)

    meta = {
        "rows": args.rows,
        "cols": args.cols,
        "first_marker_id": args.first_marker_id,
        "marker_size_mm": args.marker_size_mm,
        "gap_mm": args.gap_mm,
        "paper_width_mm": args.paper_width_mm,
        "paper_height_mm": args.paper_height_mm,
        "dpi": args.dpi,
        "dictionary": args.dict,
        "image_size_px": [int(img.shape[1]), int(img.shape[0])],
        "pixel_values": [0, 255],
        "print_hint": "Print at 100% actual size, disable fit-to-page/smart scaling.",
    }
    meta_path = Path(__file__).resolve().parent / "aruco_board_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Preview with nearest-neighbor to avoid display interpolation blur.
    preview_scale = 0.4
    preview = cv2.resize(
        img,
        (int(img.shape[1] * preview_scale), int(img.shape[0] * preview_scale)),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imshow("Calibration Markers", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Saved board: {output_path}")
    print(f"Saved meta : {meta_path}")


if __name__ == "__main__":
    main()
