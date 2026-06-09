"""
Generate one PDF file with 6 ArUco 7x7 markers per page.

Outputs:
- aruco_5cm.pdf (marker side length 5cm)
"""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def make_aruco_image(marker_id: int, marker_size_cm: float, aruco_dict) -> Image.Image:
	pixels_per_cm = 100
	marker_px = max(1, int(round(marker_size_cm * pixels_per_cm)))

	if hasattr(cv2.aruco, "generateImageMarker"):
		marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
	else:
		marker = np.zeros((marker_px, marker_px), dtype=np.uint8)
		cv2.aruco.drawMarker(aruco_dict, marker_id, marker_px, marker, 1)

	return Image.fromarray(marker).convert("RGB")


def build_pdf(filename: str, marker_ids: list[int], marker_size_cm: float) -> None:
	page_w, page_h = A4
	cols, rows = 2, 3
	cell_size = 7 * cm
	grid_w = cols * cell_size
	grid_h = rows * cell_size
	margin_x = (page_w - grid_w) / 2
	margin_y = (page_h - grid_h) / 2

	marker_size = marker_size_cm * cm
	aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)

	c = canvas.Canvas(filename, pagesize=A4)

	for idx, marker_id in enumerate(marker_ids):
		r = idx // cols
		if r >= rows:
			break
		col = idx % cols

		cell_left = margin_x + col * cell_size
		cell_bottom = margin_y + (rows - 1 - r) * cell_size

		marker_x = cell_left + (cell_size - marker_size) / 2
		marker_y = cell_bottom + (cell_size - marker_size) / 2

		img = make_aruco_image(marker_id, marker_size_cm, aruco_dict)
		img_io = BytesIO()
		img.save(img_io, format="PNG")
		img_io.seek(0)

		c.drawImage(
			ImageReader(img_io),
			marker_x,
			marker_y,
			width=marker_size,
			height=marker_size,
			preserveAspectRatio=True,
		)

	c.showPage()
	c.save()


def main() -> None:
	marker_ids = [100, 101, 102, 103, 104, 106]
	build_pdf("aruco_5cm.pdf", marker_ids, marker_size_cm=5.0)


if __name__ == "__main__":
	main()
