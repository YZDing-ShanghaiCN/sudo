#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyright: reportMissingImports=false

"""Capture one photo from a V4L2 device using settings compatible with V4L2_capture.py.

Default target device is /dev/video1 as requested, but this script will report
clearly if the selected node is metadata-only and cannot provide VIDEO_CAPTURE.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from typing import Any

from linuxpy.video.device import BufferType, Device, PixelFormat


def _alarm_timeout_handler(signum: int, frame: Any) -> None:
    del signum, frame
    raise TimeoutError("Timed out while waiting for first frame")


def probe_video_capture_capability(device_path: str) -> tuple[bool | None, str]:
    """Probe whether a V4L2 node exposes VIDEO_CAPTURE in Device Caps.

    Returns:
        (True, msg): node supports VIDEO_CAPTURE
        (False, msg): node does not support VIDEO_CAPTURE
        (None, msg): probe could not determine capability
    """
    try:
        proc = subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--all"],
            capture_output=True,
            text=True,
            check=False,
            timeout=1.5,
        )
    except FileNotFoundError:
        return None, "v4l2-ctl not found; skip capability precheck"
    except subprocess.TimeoutExpired:
        return None, "v4l2-ctl probe timeout; skip capability precheck"
    except KeyboardInterrupt:
        return None, "capability precheck interrupted by user"
    except Exception as exc:
        return None, f"capability precheck failed: {exc}"

    if proc.returncode != 0:
        err_text = (proc.stderr or proc.stdout or "").strip()
        return None, f"v4l2-ctl failed: {err_text}"

    lines = proc.stdout.splitlines()
    in_device_caps = False
    device_caps: list[str] = []

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if "Device Caps" in line:
            in_device_caps = True
            continue

        if not in_device_caps:
            continue

        if stripped == "":
            break

        # Capability item lines are plain labels, while next fields usually have ':'.
        if ":" in stripped:
            break

        device_caps.append(stripped)

    has_video_capture = any(item.startswith("Video Capture") for item in device_caps)
    if has_video_capture:
        return True, "VIDEO_CAPTURE capability detected"

    if device_caps:
        joined = ", ".join(device_caps)
        return False, f"no VIDEO_CAPTURE in Device Caps: {joined}"

    return None, "could not parse Device Caps from v4l2-ctl output"


def load_config(config_path: str) -> dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        print(f"[Config] JSON parse failed: {exc}")
        return {}


def pick_camera_config(config: dict[str, Any], cam_id: str | None) -> dict[str, Any]:
    cameras = config.get("cameras", [])
    if not isinstance(cameras, list):
        return {}

    enabled: list[dict[str, Any]] = []
    for cam in cameras:
        if isinstance(cam, dict) and cam.get("enable", True):
            enabled.append(cam)

    if not enabled:
        return {}

    if cam_id:
        for cam in enabled:
            if str(cam.get("cam_id", "")) == cam_id:
                return cam

    for cam in enabled:
        if str(cam.get("device", "")) == "/dev/video1":
            return cam

    return enabled[0]


def ensure_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def apply_controls(device: Device, cfg: dict[str, Any]) -> None:
    if cfg.get("is_auto_exposure", True):
        try:
            device.controls.auto_exposure.value = 3
        except Exception as exc:
            print(f"[Capture] Auto exposure set failed: {exc}")
    else:
        try:
            device.controls.auto_exposure.value = 1
            exposure_value = cfg.get("manual_exposure_value", 150)
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
        backlight = cfg.get("backlight_compensation", 0)
        device.controls.backlight_compensation.value = backlight
    except Exception as exc:
        print(f"[Capture] Backlight compensation set failed: {exc}")

    # Optional image controls. Apply only when present and supported.
    for key in (
        "gain",
        "brightness",
        "contrast",
        "saturation",
        "sharpness",
        "gamma",
    ):
        if key not in cfg:
            continue
        try:
            if hasattr(device.controls, key):
                getattr(device.controls, key).value = cfg[key]
                print(f"[Capture] {key} set: {cfg[key]}")
            else:
                print(f"[Capture] {key} not supported by this device")
        except Exception as exc:
            print(f"[Capture] {key} set failed: {exc}")


def capture_one_frame(
    device_path: str,
    width: int,
    height: int,
    fps: int,
    cfg: dict[str, Any],
    frame_timeout_s: float,
    warmup_frames: int,
) -> bytes:
    device = Device(device_path)
    try:
        device.open()

        try:
            device.set_format(
                BufferType.VIDEO_CAPTURE,
                width,
                height,
                pixel_format=PixelFormat.MJPEG,
            )
        except Exception as exc:
            print(f"[Capture] MJPEG format set failed: {exc}")
            device.set_format(BufferType.VIDEO_CAPTURE, width, height)

        if cfg.get("backlight_compensation", 0) != 2:
            try:
                device.set_fps(BufferType.VIDEO_CAPTURE, fps)
            except Exception as exc:
                print(f"[Capture] FPS set failed: {exc}")
        else:
            print("[Capture] Skipping FPS setting (backlight_compensation=2)")

        apply_controls(device, cfg)

        try:
            fmt = device.get_format(BufferType.VIDEO_CAPTURE)
            print(
                f"[Capture] Opened {device_path} {fmt.width}x{fmt.height} pixel_format={fmt.pixel_format}"
            )
        except Exception as exc:
            print(f"[Capture] Format query failed: {exc}")

        try:
            actual_fps = device.get_fps(BufferType.VIDEO_CAPTURE)
            print(f"[Capture] FPS: {actual_fps}")
        except Exception as exc:
            print(f"[Capture] FPS query failed: {exc}")

        print(f"[Capture] Waiting for first frame (timeout={frame_timeout_s:.1f}s)...")

        warmup_left = max(0, warmup_frames)

        # linuxpy iterator blocks until a frame arrives. For external-trigger cameras,
        # this can block forever if no trigger is present, so add an alarm timeout.
        if frame_timeout_s > 0 and hasattr(signal, "setitimer"):
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _alarm_timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, frame_timeout_s)
            try:
                for frame_data in device:
                    if warmup_left > 0:
                        warmup_left -= 1
                        if warmup_left == 0:
                            print("[Capture] Warmup finished, capturing next frame")
                        continue
                    return bytes(frame_data)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            for frame_data in device:
                if warmup_left > 0:
                    warmup_left -= 1
                    if warmup_left == 0:
                        print("[Capture] Warmup finished, capturing next frame")
                    continue
                return bytes(frame_data)

        raise RuntimeError("No frame received from device")
    finally:
        try:
            device.close()
        except Exception:
            pass


def build_default_output() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    shot_dir = os.path.join(base_dir, "obs_data", "single_shot")
    os.makedirs(shot_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(shot_dir, f"shot_{ts}.jpg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one photo from a V4L2 device")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "configS.json"),
        help="Path to JSON config file (default: temp/configS.json)",
    )
    parser.add_argument("--cam-id", default=None, help="Use camera profile by cam_id")
    parser.add_argument(
        "--device",
        default="/dev/video1",
        help="V4L2 device path (default: /dev/video1)",
    )
    parser.add_argument("--width", type=int, default=None, help="Override width")
    parser.add_argument("--height", type=int, default=None, help="Override height")
    parser.add_argument("--fps", type=int, default=None, help="Override FPS")
    parser.add_argument(
        "--frame-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for first frame (0 means wait forever)",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="Drop N initial frames before saving one frame (helps first-frame blur).",
    )
    parser.add_argument("--gain", type=float, default=None, help="Override gain")
    parser.add_argument("--brightness", type=float, default=None, help="Override brightness")
    parser.add_argument("--contrast", type=float, default=None, help="Override contrast")
    parser.add_argument("--saturation", type=float, default=None, help="Override saturation")
    parser.add_argument("--sharpness", type=float, default=None, help="Override sharpness")
    parser.add_argument("--gamma", type=float, default=None, help="Override gamma")
    parser.add_argument(
        "--output",
        default=None,
        help="Output jpg path (default: temp/obs_data/single_shot/shot_*.jpg)",
    )
    parser.add_argument(
        "--capability-precheck",
        action="store_true",
        help="Run v4l2-ctl capability precheck before opening device (disabled by default).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    base_cam_cfg = pick_camera_config(config, args.cam_id)

    width = int(args.width if args.width is not None else base_cam_cfg.get("width", 1920))
    height = int(args.height if args.height is not None else base_cam_cfg.get("height", 1080))
    fps = int(args.fps if args.fps is not None else base_cam_cfg.get("fps", 60))

    cam_cfg = dict(base_cam_cfg)
    cam_cfg["device"] = args.device

    # CLI control overrides
    for key in (
        "gain",
        "brightness",
        "contrast",
        "saturation",
        "sharpness",
        "gamma",
    ):
        value = getattr(args, key)
        if value is not None:
            cam_cfg[key] = value

    output_path = args.output if args.output else build_default_output()
    ensure_dir(output_path)

    print(
        f"[Capture] device={args.device}, width={width}, height={height}, fps={fps}, "
        f"auto_exposure={cam_cfg.get('is_auto_exposure', True)}, "
        f"manual_exposure_value={cam_cfg.get('manual_exposure_value', 150)}, "
        f"backlight_compensation={cam_cfg.get('backlight_compensation', 0)}"
    )

    if args.capability_precheck:
        has_cap, cap_msg = probe_video_capture_capability(args.device)
        if has_cap is True:
            print(f"[Capture] Capability precheck: {cap_msg}")
        elif has_cap is False:
            print(f"[Capture] Capability precheck: {cap_msg}")
            print(
                "[Capture] This node is metadata-only (cannot output image frames). "
                "Try a video capture node such as /dev/video0."
            )
            return 2
        else:
            print(f"[Capture] Capability precheck: {cap_msg}")
    else:
        print("[Capture] Capability precheck: skipped")

    try:
        frame_bytes = capture_one_frame(
            args.device,
            width,
            height,
            fps,
            cam_cfg,
            max(0.0, float(args.frame_timeout)),
            max(0, int(args.warmup_frames)),
        )
    except Exception as exc:
        print(f"[Capture] Failed: {exc}")
        if "VIDEO_CAPTURE capability" in str(exc):
            print(
                "[Capture] This node is metadata-only (no VIDEO_CAPTURE). "
                "Please use a video capture node such as /dev/video0."
            )
        elif isinstance(exc, TimeoutError):
            print(
                "[Capture] No frame arrived before timeout. If this is an external-trigger "
                "camera, please confirm trigger signal is present; otherwise try another node "
                "or set --frame-timeout 0 to wait indefinitely."
            )
        return 1
    except KeyboardInterrupt:
        print("\n[Capture] Interrupted by user")
        return 130

    with open(output_path, "wb") as f:
        f.write(frame_bytes)

    print(f"[Capture] Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
