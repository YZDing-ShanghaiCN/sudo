"""Single-camera high-quality preview.

Goals:
- Preview only one camera (default index 1).
- Request high-quality capture format (MJPG) and apply camera controls from config profile.
- Keep preview in 1:1 pixels (no resize) so display never introduces blur.

Usage:
    python calibrate/detect.py
    python calibrate/detect.py --index 1 --center-crop --preview-width 800 --preview-height 600
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


WINDOW_NAME = "High-Quality Camera Preview"
DEFAULT_INDEX = 1
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30
DEFAULT_PREVIEW_WIDTH = 800
DEFAULT_PREVIEW_HEIGHT = 450
DEFAULT_FOURCC = "MJPG"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configS.json"
DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parent / "runs"


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _camera_backend(use_dshow: bool) -> Optional[int]:
    if use_dshow and platform.system().lower().startswith("win") and hasattr(cv2, "CAP_DSHOW"):
        return cv2.CAP_DSHOW
    return None


def _open_capture(index: int, backend: Optional[int]):
    cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def _device_index_from_path(device_value: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*$", str(device_value))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _select_camera_profile(config_data: dict, camera_index: int) -> Optional[dict]:
    cameras = config_data.get("cameras", [])
    if not isinstance(cameras, list):
        return None

    enabled = [cam for cam in cameras if isinstance(cam, dict) and _safe_bool(cam.get("enable", True), True)]
    if not enabled:
        return None

    for cam in enabled:
        if "index" in cam and _safe_int(cam.get("index"), -1) == camera_index:
            return cam

    for cam in enabled:
        device = str(cam.get("device", ""))
        if _device_index_from_path(device) == camera_index:
            return cam

    return enabled[0]


def _load_profile(config_path: Path, camera_index: int) -> Optional[dict]:
    if not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
        return _select_camera_profile(config_data, camera_index)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Failed to load config profile from {config_path}: {exc}")
        return None


def _fourcc_text(value: int) -> str:
    raw = [(value >> (8 * i)) & 0xFF for i in range(4)]
    printable = [chr(b) if 32 <= b <= 126 else "" for b in raw]
    text = "".join(printable).strip()
    if len(text) == 4:
        return text
    return f"0x{int(value) & 0xFFFFFFFF:08X}"


def _configure_capture(cap: cv2.VideoCapture, width: int, height: int, fps: int, fourcc: str) -> None:
    if len(fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))


def _warmup_capture(cap: cv2.VideoCapture, frames: int = 12) -> None:
    for _ in range(max(1, frames)):
        cap.read()


def _apply_stream_by_order(
    cap: cv2.VideoCapture,
    width: int,
    height: int,
    fps: int,
    fourcc: str,
    order_name: str,
) -> None:
    if order_name == "fourcc-size-fps":
        _configure_capture(cap, 0, 0, 0, fourcc)
        _configure_capture(cap, width, height, fps, "")
        return
    if order_name == "size-fourcc-fps":
        _configure_capture(cap, width, height, 0, "")
        _configure_capture(cap, 0, 0, 0, fourcc)
        _configure_capture(cap, 0, 0, fps, "")
        return
    # default: size-fps-fourcc
    _configure_capture(cap, width, height, fps, "")
    _configure_capture(cap, 0, 0, 0, fourcc)


def _stream_score(req_fourcc: str, req_w: int, req_h: int, req_fps: int, act_w: int, act_h: int, act_fps: float, act_fourcc: str) -> float:
    score = 0.0
    if act_fourcc.upper() == req_fourcc.upper():
        score += 1_000_000_000.0
    score += float(max(0, act_w) * max(0, act_h))
    score += float(min(max(act_fps, 0.0), max(float(req_fps), 1.0)) * 1000.0)
    score -= abs(float(act_w - req_w)) * 50.0
    score -= abs(float(act_h - req_h)) * 50.0
    return score


def _candidate_resolutions(req_w: int, req_h: int) -> list[Tuple[int, int]]:
    candidates = [
        (req_w, req_h),
        (1920, 1080),
        (1600, 1200),
        (1280, 720),
        (1280, 960),
    ]
    unique: list[Tuple[int, int]] = []
    for w, h in candidates:
        if w > 0 and h > 0 and (w, h) not in unique:
            unique.append((w, h))
    return unique


def _negotiate_stream(cap: cv2.VideoCapture, req_w: int, req_h: int, req_fps: int, req_fourcc: str) -> Tuple[int, int, float, str]:
    orders = ["fourcc-size-fps", "size-fourcc-fps", "size-fps-fourcc"]
    best_tuple: Optional[Tuple[float, int, int, float, str]] = None

    for cand_w, cand_h in _candidate_resolutions(req_w, req_h):
        for order in orders:
            _apply_stream_by_order(cap, cand_w, cand_h, req_fps, req_fourcc, order)
            _warmup_capture(cap, frames=10)
            act_w, act_h, act_fps, act_fourcc = _actual_stream_info(cap)
            score = _stream_score(req_fourcc, req_w, req_h, req_fps, act_w, act_h, act_fps, act_fourcc)
            print(
                "[INFO] stream try: "
                f"order={order}, req={cand_w}x{cand_h}@{req_fps} {req_fourcc}, "
                f"actual={act_w}x{act_h}@{act_fps:.2f} {act_fourcc}"
            )
            if best_tuple is None or score > best_tuple[0]:
                best_tuple = (score, act_w, act_h, act_fps, act_fourcc)
            if act_fourcc.upper() == req_fourcc.upper() and act_w == cand_w and act_h == cand_h:
                return act_w, act_h, act_fps, act_fourcc

    if best_tuple is None:
        return _actual_stream_info(cap)
    return best_tuple[1], best_tuple[2], best_tuple[3], best_tuple[4]

def _set_prop_if_supported(cap: cv2.VideoCapture, prop_name: str, value: float, label: str) -> None:
    if not hasattr(cv2, prop_name):
        return
    prop_id = getattr(cv2, prop_name)
    cap.set(prop_id, value)
    current = cap.get(prop_id)
    print(f"[INFO] {label}: set={value} readback={current:.4f}")


def _read_prop_if_supported(cap: cv2.VideoCapture, prop_name: str) -> Optional[float]:
    if not hasattr(cv2, prop_name):
        return None
    return float(cap.get(getattr(cv2, prop_name)))


def _bump_prop_if_supported(cap: cv2.VideoCapture, prop_name: str, delta: float, label: str) -> None:
    current = _read_prop_if_supported(cap, prop_name)
    if current is None:
        return
    new_value = current + delta
    _set_prop_if_supported(cap, prop_name, new_value, label)


def _apply_profile_controls(cap: cv2.VideoCapture, profile: Optional[dict]) -> None:
    if not profile:
        return

    is_auto_exposure = _safe_bool(profile.get("is_auto_exposure", True), True)
    if is_auto_exposure:
        _set_prop_if_supported(cap, "CAP_PROP_AUTO_EXPOSURE", 0.75, "auto_exposure")
    else:
        _set_prop_if_supported(cap, "CAP_PROP_AUTO_EXPOSURE", 0.25, "auto_exposure")
        manual_exposure = float(_safe_int(profile.get("manual_exposure_value", 120), 120))
        _set_prop_if_supported(cap, "CAP_PROP_EXPOSURE", manual_exposure, "exposure")

        # Some Windows drivers expect negative log-scale exposure values.
        if platform.system().lower().startswith("win") and manual_exposure > 0:
            approx_win_exposure = -float(max(1, min(13, int(round(np.log2(manual_exposure))))))
            _set_prop_if_supported(cap, "CAP_PROP_EXPOSURE", approx_win_exposure, "exposure_windows_alt")

    if "backlight_compensation" in profile:
        backlight = float(_safe_int(profile.get("backlight_compensation", 0), 0))
        _set_prop_if_supported(cap, "CAP_PROP_BACKLIGHT", backlight, "backlight_compensation")

    if "is_auto_focus" in profile:
        is_auto_focus = _safe_bool(profile.get("is_auto_focus", True), True)
        _set_prop_if_supported(cap, "CAP_PROP_AUTOFOCUS", 1.0 if is_auto_focus else 0.0, "auto_focus")
    if "manual_focus_value" in profile:
        _set_prop_if_supported(cap, "CAP_PROP_FOCUS", float(_safe_int(profile.get("manual_focus_value", 0), 0)), "focus")

    # Additional optional controls if provided in config.
    control_map = {
        "gain": "CAP_PROP_GAIN",
        "brightness": "CAP_PROP_BRIGHTNESS",
        "contrast": "CAP_PROP_CONTRAST",
        "saturation": "CAP_PROP_SATURATION",
        "sharpness": "CAP_PROP_SHARPNESS",
        "gamma": "CAP_PROP_GAMMA",
        "focus": "CAP_PROP_FOCUS",
    }
    for key, prop_name in control_map.items():
        if key in profile:
            _set_prop_if_supported(cap, prop_name, float(profile[key]), key)


def _actual_stream_info(cap: cv2.VideoCapture) -> Tuple[int, int, float, str]:
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc = _fourcc_text(int(cap.get(cv2.CAP_PROP_FOURCC)))
    return width, height, fps, fourcc


def _center_crop_no_scale(frame: np.ndarray, preview_w: int, preview_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0 or preview_w <= 0 or preview_h <= 0:
        return frame

    crop_w = min(w, preview_w)
    crop_h = min(h, preview_h)
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    return frame[y0 : y0 + crop_h, x0 : x0 + crop_w].copy()


def _estimate_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _build_snapshot_path(snapshot_dir: Path, prefix: str, camera_index: int, width: int, height: int, file_ext: str) -> Path:
    safe_prefix = prefix.strip() if isinstance(prefix, str) and prefix.strip() else "cam"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"{safe_prefix}_idx{camera_index}_{width}x{height}_{ts}.{file_ext}"
    return snapshot_dir / name


def _save_snapshot(frame: np.ndarray, output_path: Path, file_ext: str, jpeg_quality: int) -> bool:
    params = []
    if file_ext == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, int(jpeg_quality)))]
    ok = cv2.imwrite(str(output_path), frame, params)
    return bool(ok)


def _draw_overlay(
    view: np.ndarray,
    index: int,
    req_w: int,
    req_h: int,
    req_fps: int,
    req_fourcc: str,
    act_w: int,
    act_h: int,
    act_fps: float,
    act_fourcc: str,
    sharpness: float,
) -> np.ndarray:
    out = view.copy()
    h, _ = out.shape[:2]

    cv2.putText(out, f"Camera index: {index}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"Requested: {req_w}x{req_h} @ {req_fps}fps {req_fourcc}",
        (14, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"Actual: {act_w}x{act_h} @ {act_fps:.2f}fps {act_fourcc}",
        (14, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"Sharpness (Laplacian var): {sharpness:.1f}",
        (14, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (120, 255, 180),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "q/esc: quit  s:snapshot  a:toggle AF  j/k:focus-/+  n/m:exposure-/+",
        (14, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-camera high-quality preview (MJPG 1920x1080 by default).")
    parser.add_argument("--index", type=int, default=DEFAULT_INDEX, help=f"Camera index to preview. Default: {DEFAULT_INDEX}.")
    parser.add_argument("--width", type=int, default=0, help="Requested capture width. 0 means use config/default.")
    parser.add_argument("--height", type=int, default=0, help="Requested capture height. 0 means use config/default.")
    parser.add_argument("--fps", type=int, default=0, help="Requested capture FPS. 0 means use config/default.")
    parser.add_argument(
        "--fourcc",
        type=str,
        default="",
        help="Requested pixel format FourCC. Empty means use config/default.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Config profile json path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument("--ignore-config", action="store_true", help="Do not load camera profile from config json.")
    parser.add_argument(
        "--auto-exposure",
        type=str,
        choices=("on", "off"),
        default=None,
        help="Override auto exposure mode (on/off).",
    )
    parser.add_argument(
        "--manual-exposure",
        type=int,
        default=None,
        help="Override manual exposure value.",
    )
    parser.add_argument(
        "--backlight",
        type=int,
        default=None,
        help="Override backlight compensation value.",
    )
    parser.add_argument(
        "--auto-focus",
        type=str,
        choices=("on", "off"),
        default=None,
        help="Override autofocus mode (on/off).",
    )
    parser.add_argument(
        "--focus",
        type=int,
        default=None,
        help="Override manual focus value (driver dependent range).",
    )
    parser.add_argument(
        "--preview-width",
        type=int,
        default=DEFAULT_PREVIEW_WIDTH,
        help=f"Preview window width (display only). Default: {DEFAULT_PREVIEW_WIDTH}.",
    )
    parser.add_argument(
        "--preview-height",
        type=int,
        default=DEFAULT_PREVIEW_HEIGHT,
        help=f"Preview crop height (display only, no scaling). Default: {DEFAULT_PREVIEW_HEIGHT}.",
    )
    parser.add_argument(
        "--center-crop",
        action="store_true",
        help="Use 1:1 center-crop preview at --preview-width/--preview-height. Default is full frame 1:1.",
    )
    parser.add_argument(
        "--snapshot-on-start",
        action="store_true",
        help="Capture one frame and save to disk immediately, then exit.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default=str(DEFAULT_SNAPSHOT_DIR),
        help=f"Directory to save snapshots. Default: {DEFAULT_SNAPSHOT_DIR}.",
    )
    parser.add_argument(
        "--snapshot-prefix",
        type=str,
        default="cam",
        help="Filename prefix for saved snapshots.",
    )
    parser.add_argument(
        "--snapshot-format",
        type=str,
        choices=("jpg", "png", "bmp"),
        default="jpg",
        help="Saved snapshot file format.",
    )
    parser.add_argument(
        "--snapshot-jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality used when --snapshot-format=jpg (1-100).",
    )
    parser.add_argument("--no-dshow", action="store_true", help="Disable DirectShow backend on Windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    profile = None
    config_path = Path(args.config)
    if not args.ignore_config:
        profile = _load_profile(config_path, args.index)
        if profile:
            profile_name = str(profile.get("cam_id", "unknown"))
            print(
                "[INFO] Loaded profile: "
                f"{profile_name} from {config_path} "
                f"(width={profile.get('width')}, height={profile.get('height')}, fps={profile.get('fps')}, "
                f"auto_exp={profile.get('is_auto_exposure')}, exp={profile.get('manual_exposure_value')}, "
                f"backlight={profile.get('backlight_compensation')})"
            )
        else:
            print(f"[INFO] No usable profile found at: {config_path}")

    effective_profile = dict(profile) if isinstance(profile, dict) else {}
    if args.auto_exposure is not None:
        effective_profile["is_auto_exposure"] = args.auto_exposure == "on"
    if args.manual_exposure is not None:
        effective_profile["manual_exposure_value"] = args.manual_exposure
        # Manual exposure should disable auto exposure unless explicitly forced on.
        if args.auto_exposure is None:
            effective_profile["is_auto_exposure"] = False
    if args.backlight is not None:
        effective_profile["backlight_compensation"] = args.backlight
    if args.auto_focus is not None:
        effective_profile["is_auto_focus"] = args.auto_focus == "on"
    if args.focus is not None:
        effective_profile["manual_focus_value"] = args.focus
        if args.auto_focus is None:
            effective_profile["is_auto_focus"] = False

    req_w = args.width if args.width > 0 else _safe_int(effective_profile.get("width"), DEFAULT_WIDTH)
    req_h = args.height if args.height > 0 else _safe_int(effective_profile.get("height"), DEFAULT_HEIGHT)
    req_fps = args.fps if args.fps > 0 else _safe_int(effective_profile.get("fps"), DEFAULT_FPS)

    configured_fourcc = str(effective_profile.get("fourcc", DEFAULT_FOURCC)).strip().upper()
    fourcc = args.fourcc.strip().upper() if isinstance(args.fourcc, str) and args.fourcc.strip() else configured_fourcc

    backend = _camera_backend(use_dshow=not args.no_dshow)
    cap = _open_capture(args.index, backend)
    if cap is None:
        print(f"[ERROR] Cannot open camera index {args.index}")
        return

    try:
        if len(fourcc) != 4:
            print(f"[WARN] Invalid fourcc={fourcc!r}, fallback to {DEFAULT_FOURCC}")
            fourcc = DEFAULT_FOURCC

        _configure_capture(cap, req_w, req_h, req_fps, fourcc)
        _set_prop_if_supported(cap, "CAP_PROP_CONVERT_RGB", 1.0, "convert_rgb")
        _apply_profile_controls(cap, effective_profile if effective_profile else None)

        act_w, act_h, act_fps, act_fourcc = _negotiate_stream(cap, req_w, req_h, req_fps, fourcc)
        _apply_profile_controls(cap, effective_profile if effective_profile else None)
        _warmup_capture(cap, frames=20)
        act_w, act_h, act_fps, act_fourcc = _actual_stream_info(cap)

        print(
            "[INFO] Stream configured: "
            f"requested={req_w}x{req_h}@{req_fps} {fourcc}, "
            f"actual={act_w}x{act_h}@{act_fps:.2f} {act_fourcc}"
        )
        print(
            "[INFO] Preview mode: "
            + (
                f"1:1 center-crop, crop={max(200, args.preview_width)}x{max(150, args.preview_height)}"
                if args.center_crop
                else "1:1 full-frame (no resize, no crop)"
            )
        )

        if args.snapshot_on_start:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[ERROR] Failed to capture frame for snapshot-on-start")
                return
            shot_path = _build_snapshot_path(
                snapshot_dir=snapshot_dir,
                prefix=args.snapshot_prefix,
                camera_index=args.index,
                width=frame.shape[1],
                height=frame.shape[0],
                file_ext=args.snapshot_format,
            )
            saved = _save_snapshot(frame, shot_path, args.snapshot_format, args.snapshot_jpeg_quality)
            if saved:
                print(f"[INFO] Snapshot saved: {shot_path}")
            else:
                print(f"[ERROR] Failed to save snapshot: {shot_path}")
            return

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        latest_raw_frame: Optional[np.ndarray] = None

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                blank_h = max(150, args.preview_height)
                blank_w = max(200, args.preview_width)
                blank = np.zeros((blank_h, blank_w, 3), dtype=np.uint8)
                cv2.putText(blank, "No frame from camera", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow(WINDOW_NAME, blank)
            else:
                latest_raw_frame = frame.copy()
                # Preview is strict 1:1 pixels. Default is full frame; optional center-crop mode.
                if args.center_crop:
                    view = _center_crop_no_scale(frame, max(200, args.preview_width), max(150, args.preview_height))
                else:
                    view = frame.copy()
                view = _draw_overlay(
                    view,
                    index=args.index,
                    req_w=req_w,
                    req_h=req_h,
                    req_fps=req_fps,
                    req_fourcc=fourcc,
                    act_w=frame.shape[1],
                    act_h=frame.shape[0],
                    act_fps=act_fps,
                    act_fourcc=act_fourcc,
                    sharpness=_estimate_sharpness(view),
                )
                cv2.imshow(WINDOW_NAME, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                if latest_raw_frame is None:
                    print("[WARN] Snapshot skipped: no valid frame yet")
                else:
                    shot_path = _build_snapshot_path(
                        snapshot_dir=snapshot_dir,
                        prefix=args.snapshot_prefix,
                        camera_index=args.index,
                        width=latest_raw_frame.shape[1],
                        height=latest_raw_frame.shape[0],
                        file_ext=args.snapshot_format,
                    )
                    saved = _save_snapshot(latest_raw_frame, shot_path, args.snapshot_format, args.snapshot_jpeg_quality)
                    if saved:
                        print(f"[INFO] Snapshot saved: {shot_path}")
                    else:
                        print(f"[ERROR] Failed to save snapshot: {shot_path}")
            if key == ord("a"):
                af = _read_prop_if_supported(cap, "CAP_PROP_AUTOFOCUS")
                if af is not None:
                    next_af = 0.0 if af >= 0.5 else 1.0
                    _set_prop_if_supported(cap, "CAP_PROP_AUTOFOCUS", next_af, "auto_focus")
            if key == ord("j"):
                _bump_prop_if_supported(cap, "CAP_PROP_FOCUS", -1.0, "focus")
            if key == ord("k"):
                _bump_prop_if_supported(cap, "CAP_PROP_FOCUS", 1.0, "focus")
            if key == ord("n"):
                _bump_prop_if_supported(cap, "CAP_PROP_EXPOSURE", -1.0, "exposure")
            if key == ord("m"):
                _bump_prop_if_supported(cap, "CAP_PROP_EXPOSURE", 1.0, "exposure")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
