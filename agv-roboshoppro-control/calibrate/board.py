"""Generate a printable chessboard for camera calibration.

This script outputs:
1) A binary PNG chessboard image.
2) A JSON metadata file used by calibrate/calib2.py.

Example:
    python calibrate/board.py
    python calibrate/board.py --squares-cols 9 --squares-rows 6 --square-size-mm 30 --dpi 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def ensure_binary(img: np.ndarray) -> np.ndarray:
    return np.where(img > 127, 255, 0).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a printable chessboard image.")
    parser.add_argument("--squares-cols", type=int, default=10, help="Chessboard square columns. Default: 10")
    parser.add_argument("--squares-rows", type=int, default=7, help="Chessboard square rows. Default: 7")
    parser.add_argument("--square-size-mm", type=float, default=30.0, help="Single square size in mm. Default: 30")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI. Default: 300")
    parser.add_argument("--paper-width-mm", type=float, default=210.0, help="Paper width in mm. Default: 210")
    parser.add_argument("--paper-height-mm", type=float, default=297.0, help="Paper height in mm. Default: 297")
    parser.add_argument("--margin-mm", type=float, default=8.0, help="Page margin in mm. Default: 8")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "chessboard.png"),
        help="Output chessboard PNG path.",
    )
    parser.add_argument(
        "--meta-output",
        type=str,
        default=str(Path(__file__).resolve().parent / "chessboard_meta.json"),
        help="Output metadata JSON path.",
    )
    parser.add_argument("--show", action="store_true", help="Preview generated board.")
    return parser.parse_args()


def build_chessboard(args: argparse.Namespace) -> np.ndarray:
    if args.squares_cols < 2 or args.squares_rows < 2:
        raise ValueError("squares-cols and squares-rows must both be >= 2")
    if args.square_size_mm <= 0:
        raise ValueError("square-size-mm must be > 0")
    if args.dpi <= 0:
        raise ValueError("dpi must be > 0")

    square_px = mm_to_px(args.square_size_mm, args.dpi)
    paper_w_px = mm_to_px(args.paper_width_mm, args.dpi)
    paper_h_px = mm_to_px(args.paper_height_mm, args.dpi)
    margin_px = mm_to_px(args.margin_mm, args.dpi)

    board_w_px = args.squares_cols * square_px
    board_h_px = args.squares_rows * square_px

    if board_w_px + 2 * margin_px > paper_w_px or board_h_px + 2 * margin_px > paper_h_px:
        raise ValueError("Chessboard does not fit paper with current size/margin settings")

    img = np.full((paper_h_px, paper_w_px), 255, dtype=np.uint8)
    start_x = (paper_w_px - board_w_px) // 2
    start_y = (paper_h_px - board_h_px) // 2

    for row in range(args.squares_rows):
        for col in range(args.squares_cols):
            if (row + col) % 2 == 0:
                x0 = start_x + col * square_px
                y0 = start_y + row * square_px
                img[y0 : y0 + square_px, x0 : x0 + square_px] = 0

    return ensure_binary(img)


def main() -> int:
    args = parse_args()

    try:
        img = build_chessboard(args)
    except Exception as exc:
        print(f"[ERROR] Failed to build board: {exc}")
        return 1

    output_path = Path(args.output)
    meta_path = Path(args.meta_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output_path), img)
    if not ok:
        print(f"[ERROR] Failed to write image: {output_path}")
        return 1

    meta = {
        "board_type": "chessboard",
        "squares_cols": int(args.squares_cols),
        "squares_rows": int(args.squares_rows),
        "inner_corners_cols": int(args.squares_cols - 1),
        "inner_corners_rows": int(args.squares_rows - 1),
        "square_size_mm": float(args.square_size_mm),
        "dpi": int(args.dpi),
        "paper_width_mm": float(args.paper_width_mm),
        "paper_height_mm": float(args.paper_height_mm),
        "margin_mm": float(args.margin_mm),
        "image_size_px": [int(img.shape[1]), int(img.shape[0])],
        "pixel_values": [0, 255],
        "print_hint": "Print at 100% actual size; disable fit-to-page and image smoothing.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] Saved chessboard image: {output_path}")
    print(f"[OK] Saved metadata      : {meta_path}")
    print(
        f"[INFO] Calibration pattern (inner corners): "
        f"{meta['inner_corners_cols']} x {meta['inner_corners_rows']}"
    )

    if args.show:
        preview = cv2.resize(
            img,
            (int(img.shape[1] * 0.4), int(img.shape[0] * 0.4)),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("Chessboard", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
