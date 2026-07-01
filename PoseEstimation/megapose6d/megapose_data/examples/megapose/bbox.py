#!/usr/bin/env python3
"""Manual bbox selection from a single RGB image.

Outputs MegaPose-compatible bbox arrays without saving any files.
"""

import argparse
import json
import os
import sys

import cv2


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Select a bbox on an RGB image and print MegaPose formats."
	)
	parser.add_argument(
		"--image",
		required=True,
		help="Path to the RGB image file.",
	)
	parser.add_argument(
		"--label",
		required=True,
		help="Object label used in MegaPose JSON output.",
	)
	return parser.parse_args()


def _fail(message: str, exit_code: int = 1) -> None:
	print(f"Error: {message}", file=sys.stderr)
	sys.exit(exit_code)


def _clamp(val: int, lower: int, upper: int) -> int:
	return max(lower, min(val, upper))


def main() -> None:
	args = parse_args()

	if not os.path.isfile(args.image):
		_fail(f"Image path does not exist: {args.image}")

	image = cv2.imread(args.image, cv2.IMREAD_COLOR)
	if image is None:
		_fail(f"Failed to read image: {args.image}")

	select_win = "Select ROI"
	cv2.namedWindow(select_win, cv2.WINDOW_NORMAL)
	roi = cv2.selectROI(select_win, image, showCrosshair=True, fromCenter=False)
	cv2.destroyWindow(select_win)

	x, y, w, h = roi
	if w <= 0 or h <= 0:
		_fail("No valid bbox selected (selection canceled or zero area).")

	xmin = int(x)
	ymin = int(y)
	xmax = int(x + w)
	ymax = int(y + h)

	bbox = [xmin, ymin, xmax, ymax]

	print(bbox)
	mp_json = [
		{
			"label": args.label,
			"bbox_modal": bbox,
		}
	]
	print(json.dumps(mp_json, indent=2, ensure_ascii=False))

	overlay = image.copy()
	cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

	text_color = (0, 0, 255)
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.6
	thickness = 2

	label_text = f"label: {args.label}"
	bbox_text = f"bbox: [{xmin}, {ymin}, {xmax}, {ymax}]"

	h_img, w_img = overlay.shape[:2]
	text_x = _clamp(xmin, 5, max(5, w_img - 10))
	text_y = _clamp(ymin - 10, 20, max(20, h_img - 10))
	cv2.putText(overlay, label_text, (text_x, text_y), font, font_scale, text_color, thickness)
	cv2.putText(
		overlay,
		bbox_text,
		(text_x, _clamp(text_y + 22, 20, max(20, h_img - 10))),
		font,
		font_scale,
		text_color,
		thickness,
	)

	check_win = "BBox Check"
	cv2.namedWindow(check_win, cv2.WINDOW_NORMAL)
	cv2.imshow(check_win, overlay)
	cv2.waitKey(0)
	cv2.destroyWindow(check_win)


if __name__ == "__main__":
	main()
