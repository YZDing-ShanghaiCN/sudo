#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple photo capture tool (no detection stage).

Reuses camera config style from configS.json:
- cameras[].cam_id
- cameras[].device
- cameras[].enable

Camera note:
- This script does not write any camera parameters (resolution/fps/exposure/etc.).
- It only opens device, reads frames, previews, and saves JPG.

Controls:
- s: save one frame from all opened cameras
- c: toggle continuous save
- q: quit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import numpy as np


DEFAULT_CONFIG: dict[str, Any] = {
	"cameras": [
		{
			"cam_id": "cam0",
			"device": "/dev/video0",
			"width": 1280,
			"height": 720,
			"fps": 30,
			"is_auto_exposure": True,
			"manual_exposure_value": 100,
			"backlight_compensation": 0,
			"enable": True,
		}
	],
	"display_height": 720,
}


@dataclass
class CameraHandle:
	"""Runtime handle for one opened camera."""

	cam_id: str
	device: str | int
	width: int
	height: int
	cap: cv2.VideoCapture


def load_config(config_path: str) -> dict[str, Any]:
	"""Load JSON config, fallback to defaults when unavailable."""
	resolved_path = resolve_config_path(config_path)
	try:
		with open(resolved_path, "r", encoding="utf-8") as f:
			cfg: dict[str, Any] = json.load(f)
		print(f"[Config] Loaded: {resolved_path}")
	except FileNotFoundError:
		print(f"[Config] Not found: {config_path}, using defaults")
		cfg = DEFAULT_CONFIG.copy()
	except json.JSONDecodeError as exc:
		print(f"[Config] JSON parse error: {exc}, using defaults")
		cfg = DEFAULT_CONFIG.copy()

	cfg.setdefault("cameras", DEFAULT_CONFIG["cameras"])
	cfg.setdefault("display_height", DEFAULT_CONFIG["display_height"])
	return cfg


def resolve_config_path(config_path: str) -> str:
	"""Resolve config path with fallback to the script directory."""
	if os.path.isabs(config_path):
		return config_path

	script_dir = os.path.dirname(os.path.abspath(__file__))
	candidates = [
		os.path.abspath(config_path),
		os.path.join(script_dir, config_path),
	]

	for path in candidates:
		if os.path.exists(path):
			return path
	return config_path


def parse_device(device_value: Any) -> str | int:
	"""Parse device from config.

	Supports:
	- integer index (e.g. 0)
	- numeric string (e.g. "0")
	- V4L2 path (e.g. "/dev/video2")
	"""
	if isinstance(device_value, int):
		return device_value
	if isinstance(device_value, str):
		stripped = device_value.strip()
		dev_match = re.match(r"^/dev/video(\d+)$", stripped, re.IGNORECASE)
		if dev_match and os.name == "nt":
			return int(dev_match.group(1))
		if stripped.isdigit():
			return int(stripped)
		return stripped
	raise ValueError(f"Unsupported device value: {device_value!r}")


def open_camera(cam_cfg: dict[str, Any]) -> CameraHandle | None:
	"""Open one camera with best-effort property setup."""
	cam_id = str(cam_cfg.get("cam_id", "cam"))
	device = parse_device(cam_cfg.get("device", 0))
	cap = open_capture(device)

	if not cap.isOpened():
		print(f"[ERROR] {cam_id}: failed to open device {device}")
		return None

	actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	actual_fps = cap.get(cv2.CAP_PROP_FPS)
	print(
		f"[Open] {cam_id}: device={device}, "
		f"size={actual_w}x{actual_h}, fps={actual_fps:.1f}"
	)

	# Warm up camera buffer and infer resolution if backend reports zeros.
	last_frame_shape: tuple[int, int] | None = None
	dark_hits = 0
	for _ in range(12):
		ok, frame = cap.read()
		if not ok or frame is None:
			time.sleep(0.02)
			continue
		last_frame_shape = (frame.shape[1], frame.shape[0])
		if frame.mean() < 2.0 and frame.std() < 2.0:
			dark_hits += 1

	if (actual_w <= 0 or actual_h <= 0) and last_frame_shape is not None:
		actual_w, actual_h = last_frame_shape
	if actual_w <= 0:
		actual_w = 640
	if actual_h <= 0:
		actual_h = 480
	if dark_hits >= 8:
		print(
			f"[WARN] {cam_id}: stream is almost black. "
			"Try another device index or set is_auto_exposure=true"
		)

	return CameraHandle(
		cam_id=cam_id,
		device=device,
		width=actual_w,
		height=actual_h,
		cap=cap,
	)


def open_capture(device: str | int) -> cv2.VideoCapture:
	"""Open capture with platform-specific backend fallback."""
	backend_candidates: list[int | None]
	if os.name == "nt":
		backend_candidates = [
			getattr(cv2, "CAP_DSHOW", None),
			getattr(cv2, "CAP_MSMF", None),
			None,
		]
	else:
		backend_candidates = [getattr(cv2, "CAP_V4L2", None), None]

	for backend in backend_candidates:
		if backend is None:
			cap = cv2.VideoCapture(device)
		else:
			cap = cv2.VideoCapture(device, backend)
		if cap.isOpened():
			return cap
		cap.release()

	return cv2.VideoCapture(device)


def create_session_dir(base_dir: str) -> str:
	"""Create timestamped output directory."""
	session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
	session_dir = os.path.join(base_dir, session_name)
	os.makedirs(session_dir, exist_ok=True)
	return session_dir


def resize_with_letterbox(image: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
	"""Resize image to fixed tile while preserving aspect ratio."""
	canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
	h, w = image.shape[:2]
	if h <= 0 or w <= 0:
		return canvas

	scale = min(tile_w / w, tile_h / h)
	new_w = max(1, int(w * scale))
	new_h = max(1, int(h * scale))
	resized = cv2.resize(image, (new_w, new_h))

	x = (tile_w - new_w) // 2
	y = (tile_h - new_h) // 2
	canvas[y : y + new_h, x : x + new_w] = resized
	return canvas


def build_preview_grid(frames: list[np.ndarray], tile_h: int) -> np.ndarray:
	"""Build preview layout for N cameras."""
	if not frames:
		return np.zeros((tile_h, 640, 3), dtype=np.uint8)

	max_aspect = max((img.shape[1] / img.shape[0]) for img in frames if img.shape[0] > 0)
	tile_w = max(320, int(tile_h * max_aspect))

	tiles = [resize_with_letterbox(img, tile_h, tile_w) for img in frames]
	n = len(tiles)

	if n <= 3:
		return np.hstack(tiles)

	cols = 2
	rows = (n + cols - 1) // cols
	while len(tiles) < rows * cols:
		tiles.append(np.zeros_like(tiles[0]))

	row_images: list[np.ndarray] = []
	for r in range(rows):
		row_images.append(np.hstack(tiles[r * cols : (r + 1) * cols]))
	return np.vstack(row_images)


def make_placeholder(width: int, height: int, line1: str, line2: str = "") -> np.ndarray:
	"""Create a readable placeholder tile for missing camera frames."""
	img = np.zeros((max(240, height), max(320, width), 3), dtype=np.uint8)
	color = (0, 200, 255)
	cv2.putText(
		img,
		line1,
		(20, 60),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.8,
		color,
		2,
		cv2.LINE_AA,
	)
	if line2:
		cv2.putText(
			img,
			line2,
			(20, 100),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			color,
			2,
			cv2.LINE_AA,
		)
	return img


def save_frames(session_dir: str, frame_id: int, camera_frames: dict[str, np.ndarray]) -> None:
	"""Save one image per camera under session_dir/cam_id/000001.jpg."""
	for cam_id, frame in camera_frames.items():
		cam_dir = os.path.join(session_dir, cam_id)
		os.makedirs(cam_dir, exist_ok=True)
		path = os.path.join(cam_dir, f"{frame_id:06d}.jpg")
		cv2.imwrite(path, frame)


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments."""
	parser = argparse.ArgumentParser(description="Simple multi-camera photo capture")
	parser.add_argument(
		"--config",
		default=os.path.join(os.path.dirname(__file__), "configS.json"),
		help="Path to camera JSON config",
	)
	parser.add_argument(
		"--output-dir",
		default=os.path.join(os.path.dirname(__file__), "obs_data"),
		help="Base output directory",
	)
	parser.add_argument(
		"--interval",
		type=float,
		default=1.0,
		help="Continuous save interval (seconds)",
	)
	parser.add_argument(
		"--probe",
		action="store_true",
		help="Probe camera indices and exit",
	)
	parser.add_argument(
		"--probe-max-index",
		type=int,
		default=10,
		help="Maximum camera index for --probe",
	)
	return parser.parse_args()


def probe_cameras(max_index: int) -> int:
	"""Probe camera indices and report open/read status."""
	print(f"[Probe] scanning camera index 0..{max_index}")
	found = 0
	for idx in range(max_index + 1):
		cap = open_capture(idx)
		if not cap.isOpened():
			print(f"  [{idx}] open=FAIL")
			continue

		ok, frame = cap.read()
		if ok and frame is not None:
			h, w = frame.shape[:2]
			print(f"  [{idx}] open=OK read=OK size={w}x{h}")
			found += 1
		else:
			print(f"  [{idx}] open=OK read=FAIL")
		cap.release()

	if found == 0:
		print("[Probe] no readable camera found")
	else:
		print(f"[Probe] readable cameras: {found}")
	return 0


def main() -> int:
	"""Entry point."""
	args = parse_args()
	if args.probe:
		return probe_cameras(max(0, args.probe_max_index))

	config = load_config(args.config)

	camera_cfgs = [c for c in config.get("cameras", []) if c.get("enable", True)]
	if not camera_cfgs:
		print("[ERROR] No enabled cameras in config")
		return 2

	cameras: list[CameraHandle] = []
	for cam_cfg in camera_cfgs:
		handle = open_camera(cam_cfg)
		if handle is not None:
			cameras.append(handle)

	if not cameras:
		print("[ERROR] No camera opened successfully")
		return 3

	os.makedirs(args.output_dir, exist_ok=True)
	session_dir = create_session_dir(args.output_dir)
	print(f"[Session] {session_dir}")
	print("[Controls] s:save once, c:toggle continuous, q:quit")

	display_height = int(config.get("display_height", 720))
	continuous = False
	next_save_time = time.monotonic()
	frame_id = 0
	read_fail_streak: dict[str, int] = {cam.cam_id: 0 for cam in cameras}

	try:
		while True:
			latest: dict[str, np.ndarray] = {}
			preview_frames: list[np.ndarray] = []

			for cam in cameras:
				ok, frame = cam.cap.read()
				if not ok or frame is None:
					read_fail_streak[cam.cam_id] += 1
					preview_frames.append(
						make_placeholder(
							cam.width,
							cam.height,
							f"{cam.cam_id} ({cam.device})",
							"No frame from camera",
						)
					)
					if read_fail_streak[cam.cam_id] in (30, 120):
						print(
							f"[WARN] {cam.cam_id}: no frame for "
							f"{read_fail_streak[cam.cam_id]} reads"
						)
					continue

				read_fail_streak[cam.cam_id] = 0

				label = f"{cam.cam_id} ({cam.device})"
				cv2.putText(
					frame,
					label,
					(10, 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.8,
					(255, 255, 255),
					2,
					cv2.LINE_AA,
				)
				if frame.mean() < 2.0 and frame.std() < 2.0:
					cv2.putText(
						frame,
						"Very dark frame",
						(10, 65),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.8,
						(0, 0, 255),
						2,
						cv2.LINE_AA,
					)
				latest[cam.cam_id] = frame
				preview_frames.append(frame)

			if preview_frames:
				view = build_preview_grid(preview_frames, display_height)
				status = "CONTINUOUS ON" if continuous else "CONTINUOUS OFF"
				cv2.putText(
					view,
					status,
					(10, 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.8,
					(0, 255, 0),
					2,
					cv2.LINE_AA,
				)
				cv2.imshow("Photo Capture", view)

			now = time.monotonic()
			if continuous and latest and now >= next_save_time:
				frame_id += 1
				save_frames(session_dir, frame_id, latest)
				print(f"[Saved] #{frame_id:06d}")
				next_save_time = now + max(0.05, args.interval)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
			if key == ord("s") and latest:
				frame_id += 1
				save_frames(session_dir, frame_id, latest)
				print(f"[Saved] #{frame_id:06d}")
			if key == ord("c"):
				continuous = not continuous
				next_save_time = time.monotonic()
				print(f"[Mode] continuous={'ON' if continuous else 'OFF'}")

	except KeyboardInterrupt:
		pass
	finally:
		for cam in cameras:
			cam.cap.release()
		cv2.destroyAllWindows()

	print(f"[Done] total saved groups: {frame_id}")
	print(f"[Done] output: {session_dir}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
