#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyright: reportMissingImports=false

"""Single-camera YUYV capture using the same config schema as V4L2_capture.py.

This script reads temp/configS.json (or a provided config path), opens one enabled
camera in YUYV format, previews frames, and saves JPG files on demand.

Controls:
    c - toggle continuous save
    s - save one frame
    q - quit
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from linuxpy.video.device import BufferType, Device, PixelFormat


def _timeout_handler(signum: int, frame: Any) -> None:
    del signum, frame
    raise TimeoutError("Timed out while waiting for first frame")


def load_config(config_path: str) -> dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg: dict[str, Any] = json.load(f)
        print(f"[Config] Loaded: {config_path}")
        return cfg
    except FileNotFoundError:
        print(f"[Config] Not found: {config_path}")
        return {}
    except json.JSONDecodeError as exc:
        print(f"[Config] JSON error: {exc}")
        return {}


def select_camera(config: dict[str, Any], cam_id: str | None) -> dict[str, Any]:
    cameras = config.get("cameras", [])
    if not isinstance(cameras, list):
        return {}

    enabled = [c for c in cameras if isinstance(c, dict) and c.get("enable", True)]
    if not enabled:
        return {}

    if cam_id is not None:
        for cam in enabled:
            if str(cam.get("cam_id", "")) == cam_id:
                return cam
        print(f"[Config] cam_id={cam_id} not found among enabled cameras, using first")

    if len(enabled) > 1:
        cam_names = [str(c.get("cam_id", "unknown")) for c in enabled]
        print(f"[Config] Multiple enabled cameras: {', '.join(cam_names)}")
        print("[Config] This YUYV script captures only one camera (the first enabled)")

    return enabled[0]


def apply_controls(device: Device, cam_cfg: dict[str, Any]) -> None:
    if cam_cfg.get("is_auto_exposure", True):
        try:
            device.controls.auto_exposure.value = 3
        except Exception as exc:
            print(f"[Capture] Auto exposure set failed: {exc}")
    else:
        try:
            device.controls.auto_exposure.value = 1
            exposure_value = cam_cfg.get("manual_exposure_value", 150)
            exposure_set = False
            for attr in ("exposure_absolute", "exposure", "exposure_time_absolute"):
                if hasattr(device.controls, attr):
                    getattr(device.controls, attr).value = exposure_value
                    exposure_set = True
                    break
            if exposure_set:
                print(f"[Capture] Manual exposure set: {exposure_value}")
            else:
                print("[Capture] No matching manual exposure control")
        except Exception as exc:
            print(f"[Capture] Manual exposure set failed: {exc}")

    try:
        backlight = cam_cfg.get("backlight_compensation", 0)
        device.controls.backlight_compensation.value = backlight
    except Exception as exc:
        print(f"[Capture] Backlight compensation set failed: {exc}")


def decode_yuyv_to_bgr(raw: bytes, width: int, height: int) -> np.ndarray:
    expected = width * height * 2
    if len(raw) < expected:
        raise ValueError(f"YUYV buffer too small: got={len(raw)} expected={expected}")
    arr = np.frombuffer(raw[:expected], dtype=np.uint8).reshape((height, width, 2))
    return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUY2)


def build_session_dir(base_dir: str) -> str:
    obs_dir = os.path.join(base_dir, "obs_data")
    os.makedirs(obs_dir, exist_ok=True)
    name = datetime.now().strftime("session_yuyv_%Y%m%d_%H%M%S")
    path = os.path.join(obs_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YUYV single-camera capture")
    parser.add_argument(
        "config_path",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "configS.json"),
        help="Path to JSON config (default: temp/configS.json)",
    )
    parser.add_argument("--cam-id", default=None, help="Capture this cam_id from config")
    parser.add_argument(
        "--first-frame-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for first frame before aborting (0=wait forever)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPG quality when saving (1-100)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config_path)
    cam_cfg = select_camera(config, args.cam_id)
    if not cam_cfg:
        print("[Config] No enabled camera found")
        return 1

    cam_id = str(cam_cfg.get("cam_id", "cam0"))
    device_path = str(cam_cfg.get("device", "/dev/video0"))
    width = int(cam_cfg.get("width", 1280))
    height = int(cam_cfg.get("height", 720))
    fps = int(cam_cfg.get("fps", 30))
    display_height = int(config.get("display_height", config.get("preview_height", 400)))
    if display_height <= 0:
        display_height = 400

    session_dir = build_session_dir(os.path.dirname(os.path.abspath(__file__)))
    print(f"[Session] {session_dir}")
    print(f"[Camera] {cam_id} -> {device_path}")

    csv_path = os.path.join(session_dir, "ts.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_id", "timestamp", "sequence", "file"])

    save_continuous = False
    save_single = False
    frame_count = 0
    written = 0
    last_status_t = time.monotonic()
    last_frame_count = 0

    device = Device(device_path)
    old_handler = None
    timeout_armed = False
    got_first_frame = False

    try:
        device.open()

        # Force YUYV path for capture.
        try:
            device.set_format(
                BufferType.VIDEO_CAPTURE,
                width,
                height,
                pixel_format=PixelFormat.YUYV,
            )
        except Exception as exc:
            print(f"[Capture] YUYV format set failed: {exc}")
            print("[Capture] Trying driver default format")
            device.set_format(BufferType.VIDEO_CAPTURE, width, height)

        try:
            device.set_fps(BufferType.VIDEO_CAPTURE, fps)
        except Exception as exc:
            print(f"[Capture] FPS set failed: {exc}")

        apply_controls(device, cam_cfg)

        fmt = device.get_format(BufferType.VIDEO_CAPTURE)
        actual_w = int(fmt.width)
        actual_h = int(fmt.height)
        print(f"[Capture] Opened {device_path} {actual_w}x{actual_h} pixel_format={fmt.pixel_format}")

        try:
            actual_fps = device.get_fps(BufferType.VIDEO_CAPTURE)
            print(f"[Capture] FPS: {actual_fps}")
        except Exception as exc:
            print(f"[Capture] FPS query failed: {exc}")

        print("Controls: c=continuous save, s=single save, q=quit")

        first_timeout = max(0.0, float(args.first_frame_timeout))
        if first_timeout > 0.0 and hasattr(signal, "setitimer"):
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, first_timeout)
            timeout_armed = True
            print(f"[Capture] Waiting for first frame (timeout={first_timeout:.1f}s)")

        for frame_data in device:
            if timeout_armed and not got_first_frame:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                timeout_armed = False
                got_first_frame = True
                print("[Capture] First frame received")

            raw = bytes(frame_data)
            frame_count += 1

            try:
                bgr = decode_yuyv_to_bgr(raw, actual_w, actual_h)
            except Exception as exc:
                print(f"[Capture] Decode failed at frame {frame_count}: {exc}")
                continue

            # Preview resize keeps aspect ratio while targeting configured display height.
            disp_h = display_height
            disp_w = int(actual_w * (disp_h / actual_h))
            preview = cv2.resize(bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

            save_text = "ON" if save_continuous else "OFF"
            line = f"frame={frame_count} save={save_text} written={written}"
            cv2.putText(
                preview,
                line,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("YUYV Capture (q quit, c save, s single)", preview)

            if save_continuous or save_single:
                out_name = f"{frame_count:06d}.jpg"
                out_path = os.path.join(session_dir, out_name)
                ok = cv2.imwrite(
                    out_path,
                    bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, int(args.jpeg_quality)))],
                )
                if ok:
                    written += 1
                    csv_writer.writerow(
                        [frame_count, f"{frame_data.timestamp:.6f}", frame_data.frame_nb, out_name]
                    )
                    if written % 20 == 0:
                        csv_file.flush()
                else:
                    print(f"[Capture] Failed to save {out_path}")

                if save_single:
                    save_single = False
                    print(f"[Capture] Saved single frame #{frame_count}")

            now = time.monotonic()
            if now - last_status_t >= 1.0:
                fps_now = frame_count - last_frame_count
                last_frame_count = frame_count
                last_status_t = now
                sys.stdout.write(
                    f"\rFrames: {frame_count:6d} | FPS: {fps_now:3d} | Written: {written:6d}    "
                )
                sys.stdout.flush()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[Capture] Quit requested")
                break
            if key == ord("c"):
                save_continuous = not save_continuous
                state = "ON" if save_continuous else "OFF"
                print(f"\n[Capture] Continuous save: {state}")
            if key == ord("s"):
                save_single = True

    except TimeoutError as exc:
        print(f"[Capture] Failed: {exc}")
        print("[Capture] No frame arrived. Check trigger signal or reduce fps/resolution")
        return 1
    except KeyboardInterrupt:
        print("\n[Capture] Interrupted by user")
    except Exception as exc:
        print(f"[Capture] Error: {exc}")
        return 1
    finally:
        if timeout_armed and hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)
        try:
            device.close()
        except Exception:
            pass
        csv_file.flush()
        csv_file.close()
        cv2.destroyAllWindows()

    print("\n========== Summary ==========")
    print(f"  Captured: {frame_count}")
    print(f"  Written:  {written}")
    print(f"  Session:  {session_dir}")
    print("=============================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
