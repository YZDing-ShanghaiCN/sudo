#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = "outputs/rgbd"
DEFAULT_PREFIX = "orbbec_rgbd"
DEFAULT_TIMEOUT_MS = 1000
DEFAULT_WARMUP_FRAMES = 5
DEFAULT_FRAME_COUNT = 1
DEFAULT_MAX_MISSED_FRAMES = 30
DEFAULT_MIN_VALID_RATIO = 0.001
DEFAULT_DEPTH_PREVIEW_MAX_M = 5.0
DEFAULT_DEPTH_PNG_SCALE_M = 0.001
DEFAULT_DEPTH_WIDTH = 1024
DEFAULT_DEPTH_HEIGHT = 1024
DEFAULT_DEPTH_FPS = 15

DEPTH_OUTPUT_CHOICES = ("both", "npy", "png")
COLOR_OUTPUT_CHOICES = ("jpg", "png")
DEPTH_FORMAT_CHOICES = ("y16",)
COLOR_FORMAT_CHOICES = ("default", "mjpg", "rgb", "bgr", "yuyv", "uyvy")
ALIGN_MODE_CHOICES = ("off", "sw", "hw")


@dataclass(frozen=True)
class CapturePaths:
    stem: str
    depth_m_npy: Path
    depth_m_png: Path
    depth_preview_image: Path
    color_image: Path | None
    metadata_json: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture Orbbec depth/RGBD frames with pyorbbecsdk and save depth files."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"RGBD output directory, default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"output file prefix, default: {DEFAULT_PREFIX}",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAME_COUNT,
        help=(
            "number of RGBD frames to save, default: "
            f"{DEFAULT_FRAME_COUNT}; use 0 to capture until Ctrl+C"
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="seconds to wait between saved frames, default: 0",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=DEFAULT_WARMUP_FRAMES,
        help=f"frames to discard after starting the pipeline, default: {DEFAULT_WARMUP_FRAMES}",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help=f"wait timeout for each frame set, default: {DEFAULT_TIMEOUT_MS}",
    )
    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=DEFAULT_MAX_MISSED_FRAMES,
        help=(
            "consecutive timeouts/missing depth frames before failing, default: "
            f"{DEFAULT_MAX_MISSED_FRAMES}"
        ),
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=DEFAULT_MIN_VALID_RATIO,
        help=(
            "minimum non-zero depth pixel ratio for --check-only, default: "
            f"{DEFAULT_MIN_VALID_RATIO:g}"
        ),
    )
    parser.add_argument(
        "--depth-output",
        default="both",
        choices=DEPTH_OUTPUT_CHOICES,
        help="depth output type: float32 meter .npy, scaled 16-bit meter .png, or both",
    )
    parser.add_argument(
        "--color-output",
        default="jpg",
        choices=COLOR_OUTPUT_CHOICES,
        help="color image format when color capture is enabled, default: jpg",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="capture depth only; do not enable or save the RGB stream",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="start the depth stream, read one valid frame, print stats, and save nothing",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="list Orbbec SDK devices and exit",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="list depth/color stream profiles for the selected SDK device and exit",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="optional Orbbec SDK device index; omitted uses the SDK default device",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open RGB/depth preview windows; press s to save RGBD, q/Esc to quit",
    )
    parser.add_argument(
        "--depth-preview-max-m",
        type=float,
        default=DEFAULT_DEPTH_PREVIEW_MAX_M,
        help=(
            "maximum depth in meters for viewer colorization, default: "
            f"{DEFAULT_DEPTH_PREVIEW_MAX_M:g}"
        ),
    )
    parser.add_argument(
        "--save-depth-preview",
        action="store_true",
        help="save a pseudo-color depth preview JPG; default: do not save",
    )
    parser.add_argument(
        "--depth-png-scale-m",
        type=float,
        default=DEFAULT_DEPTH_PNG_SCALE_M,
        help=(
            "meters represented by one depth PNG integer step, default: "
            f"{DEFAULT_DEPTH_PNG_SCALE_M:g}"
        ),
    )
    parser.add_argument(
        "--align-mode",
        default="off",
        choices=ALIGN_MODE_CHOICES,
        help=(
            "optional SDK depth/color alignment mode: off, sw, or hw; "
            "default: off"
        ),
    )
    parser.add_argument(
        "--depth-width",
        type=int,
        default=None,
        help=(
            f"requested depth width, default: {DEFAULT_DEPTH_WIDTH}; "
            "set depth width/height/fps all to 0 to use the SDK default profile"
        ),
    )
    parser.add_argument(
        "--depth-height",
        type=int,
        default=None,
        help=(
            f"requested depth height, default: {DEFAULT_DEPTH_HEIGHT}; "
            "set depth width/height/fps all to 0 to use the SDK default profile"
        ),
    )
    parser.add_argument(
        "--depth-fps",
        type=int,
        default=None,
        help=(
            f"requested depth FPS, default: {DEFAULT_DEPTH_FPS}; "
            "set depth width/height/fps all to 0 to use the SDK default profile"
        ),
    )
    parser.add_argument(
        "--depth-format",
        default="y16",
        choices=DEPTH_FORMAT_CHOICES,
        help="requested depth pixel format when width/height/fps are set, default: y16",
    )
    parser.add_argument(
        "--color-width",
        type=int,
        default=0,
        help="requested color width; 0 uses the SDK default profile",
    )
    parser.add_argument(
        "--color-height",
        type=int,
        default=0,
        help="requested color height; 0 uses the SDK default profile",
    )
    parser.add_argument(
        "--color-fps",
        type=int,
        default=0,
        help="requested color FPS; 0 uses the SDK default profile",
    )
    parser.add_argument(
        "--color-format",
        default="default",
        choices=COLOR_FORMAT_CHOICES,
        help=(
            "requested color pixel format when width/height/fps are set, default: "
            "SDK default profile"
        ),
    )
    parser.add_argument(
        "--depth-preview-max-mm",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    depth_profile_values = (args.depth_width, args.depth_height, args.depth_fps)
    depth_profile_requested = any(value is not None for value in depth_profile_values)
    args.depth_profile_incomplete = depth_profile_requested and not all(
        value is not None for value in depth_profile_values
    )
    if not depth_profile_requested:
        args.depth_width = DEFAULT_DEPTH_WIDTH
        args.depth_height = DEFAULT_DEPTH_HEIGHT
        args.depth_fps = DEFAULT_DEPTH_FPS

    if args.depth_preview_max_mm is not None:
        args.depth_preview_max_m = args.depth_preview_max_mm / 1000.0
    return args


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def safe_label(text: str) -> str:
    cleaned = []
    for char in text.strip():
        if char.isalnum() or char in ("-", "_", "."):
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or DEFAULT_PREFIX


def build_capture_paths(
    session_dir: Path,
    prefix: str,
    frame_index: int,
    color_ext: str,
    *,
    include_color: bool = True,
) -> CapturePaths:
    stem = f"{safe_label(prefix)}_{frame_index:04d}"
    color_path = session_dir / "rgb" / f"{stem}_color.{color_ext}" if include_color else None
    return CapturePaths(
        stem=stem,
        depth_m_npy=session_dir / "depth" / f"{stem}_depth_m.npy",
        depth_m_png=session_dir / "depth" / f"{stem}_depth_m.png",
        depth_preview_image=session_dir / "depth" / f"{stem}_depth_preview.jpg",
        color_image=color_path,
        metadata_json=session_dir / f"{stem}.json",
    )


def profile_is_specific(width: int, height: int, fps: int) -> bool:
    return any(value > 0 for value in (width, height, fps))


def validate_profile_triplet(
    label: str,
    width: int,
    height: int,
    fps: int,
) -> bool:
    values = (width, height, fps)
    if any(value < 0 for value in values):
        print(f"错误：{label} 的 width/height/fps 不能小于 0。", file=sys.stderr)
        return False
    if profile_is_specific(width, height, fps) and not all(value > 0 for value in values):
        print(
            f"错误：指定 {label} profile 时需要同时设置 width、height 和 fps。",
            file=sys.stderr,
        )
        return False
    return True


def validate_args(args: argparse.Namespace) -> bool:
    if args.frames < 0:
        print("错误：--frames 不能小于 0。", file=sys.stderr)
        return False
    if args.interval < 0:
        print("错误：--interval 不能小于 0。", file=sys.stderr)
        return False
    if args.warmup_frames < 0:
        print("错误：--warmup-frames 不能小于 0。", file=sys.stderr)
        return False
    if args.timeout_ms <= 0:
        print("错误：--timeout-ms 必须大于 0。", file=sys.stderr)
        return False
    if args.max_missed_frames < 1:
        print("错误：--max-missed-frames 必须大于等于 1。", file=sys.stderr)
        return False
    if not 0 <= args.min_valid_ratio <= 1:
        print("错误：--min-valid-ratio 必须在 0 到 1 之间。", file=sys.stderr)
        return False
    if args.depth_preview_max_m <= 0:
        print("错误：--depth-preview-max-m 必须大于 0。", file=sys.stderr)
        return False
    if args.depth_png_scale_m <= 0:
        print("错误：--depth-png-scale-m 必须大于 0。", file=sys.stderr)
        return False
    if args.viewer and args.check_only:
        print("错误：--viewer 和 --check-only 不能同时使用。", file=sys.stderr)
        return False
    if args.device_index is not None and args.device_index < 0:
        print("错误：--device-index 不能小于 0。", file=sys.stderr)
        return False
    if getattr(args, "depth_profile_incomplete", False):
        print(
            "错误：指定 depth profile 时需要同时设置 width、height 和 fps。",
            file=sys.stderr,
        )
        return False
    if not validate_profile_triplet(
        "depth", args.depth_width, args.depth_height, args.depth_fps
    ):
        return False
    if not validate_profile_triplet(
        "color", args.color_width, args.color_height, args.color_fps
    ):
        return False
    return True


def load_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        print("错误：无法导入 NumPy，深度文件保存需要 NumPy。", file=sys.stderr)
        print("请先安装 Python 依赖：", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1) from exc
    return np


def load_cv2():
    try:
        import cv2
    except ImportError as exc:
        print("错误：无法导入 OpenCV，保存 PNG/JPG 需要 OpenCV。", file=sys.stderr)
        print("请先安装 Python 依赖：", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1) from exc
    return cv2


def load_orbbec_sdk():
    try:
        import pyorbbecsdk as sdk
    except ImportError as exc:
        print("错误：无法导入 pyorbbecsdk。", file=sys.stderr)
        print("深度/RGBD 采集需要 Orbbec SDK 的 Python 绑定。", file=sys.stderr)
        print("Orbbec SDK v2 官方包名是 pyorbbecsdk2，模块名仍然是 pyorbbecsdk。", file=sys.stderr)
        print("请在当前 Python 环境中安装：", file=sys.stderr)
        print("  pip install --upgrade pyorbbecsdk2", file=sys.stderr)
        print("如果之前装过旧包 pyorbbecsdk，建议先卸载：", file=sys.stderr)
        print("  pip uninstall pyorbbecsdk", file=sys.stderr)
        raise SystemExit(1) from exc
    return sdk


def enum_member(sdk: Any, enum_type_name: str, member_name: str) -> Any:
    enum_type = getattr(sdk, enum_type_name)
    return getattr(enum_type, member_name)


def enum_name(value: Any) -> str:
    name = getattr(value, "name", "")
    if name:
        return str(name).upper()
    return str(value).split(".")[-1].upper()


def sdk_format_from_arg(sdk: Any, value: str, fallback: str | None = None) -> Any | None:
    if value == "default":
        value = fallback or "default"
    if value == "default":
        return None

    mapping = {
        "y16": "Y16",
        "mjpg": "MJPG",
        "rgb": "RGB",
        "bgr": "BGR",
        "yuyv": "YUYV",
        "uyvy": "UYVY",
    }
    return enum_member(sdk, "OBFormat", mapping[value])


def select_stream_profile(
    sdk: Any,
    pipeline: Any,
    sensor_member: str,
    width: int,
    height: int,
    fps: int,
    format_arg: str,
    *,
    default_specific_format: str | None = None,
) -> Any:
    sensor_type = enum_member(sdk, "OBSensorType", sensor_member)
    profile_list = pipeline.get_stream_profile_list(sensor_type)

    if profile_is_specific(width, height, fps):
        stream_format = sdk_format_from_arg(
            sdk, format_arg, fallback=default_specific_format
        )
        return profile_list.get_video_stream_profile(width, height, stream_format, fps)

    return profile_list.get_default_video_stream_profile()


def describe_profile(profile: Any) -> str:
    values: dict[str, Any] = {}
    for key, method_name in (
        ("width", "get_width"),
        ("height", "get_height"),
        ("fps", "get_fps"),
        ("format", "get_format"),
    ):
        method = getattr(profile, method_name, None)
        if method is None:
            continue
        try:
            values[key] = method()
        except Exception:
            continue

    size = (
        f"{values['width']}x{values['height']}"
        if "width" in values and "height" in values
        else "unknown-size"
    )
    fps = f"@{values['fps']}fps" if "fps" in values else ""
    fmt = f" {enum_name(values['format'])}" if "format" in values else ""
    return f"{size}{fps}{fmt}"


def safe_call(obj: Any, method_name: str, default: Any = "") -> Any:
    method = getattr(obj, method_name, None)
    if method is None:
        return default
    try:
        return method()
    except Exception:
        return default


def get_device_list(sdk: Any) -> Any:
    context = sdk.Context()
    return context.query_devices()


def get_device_count(device_list: Any) -> int:
    get_count = getattr(device_list, "get_count", None)
    if get_count is not None:
        return int(get_count())
    return len(device_list)


def get_device_by_index(device_list: Any, index: int) -> Any:
    getter = getattr(device_list, "get_device_by_index", None)
    if getter is not None:
        return getter(index)
    return device_list[index]


def get_profile_count(profile_list: Any) -> int:
    get_count = getattr(profile_list, "get_count", None)
    if get_count is not None:
        return int(get_count())
    return len(profile_list)


def get_profile_by_index(profile_list: Any, index: int) -> Any:
    for method_name in ("get_stream_profile_by_index", "get_profile"):
        getter = getattr(profile_list, method_name, None)
        if getter is not None:
            return getter(index)
    return profile_list[index]


def require_device_by_index(sdk: Any, index: int) -> Any:
    device_list = get_device_list(sdk)
    device_count = get_device_count(device_list)
    if device_count == 0:
        raise RuntimeError("没有发现 Orbbec SDK 设备。")
    if index >= device_count:
        raise RuntimeError(f"--device-index {index} 超出范围，当前设备数量为 {device_count}。")
    return get_device_by_index(device_list, index)


def create_pipeline_for_device(sdk: Any, device: Any) -> Any:
    try:
        return sdk.Pipeline(device)
    except TypeError:
        pipeline = sdk.Pipeline()
        setter = getattr(pipeline, "set_device", None)
        if setter is None:
            raise
        setter(device)
        return pipeline


def device_info_row(index: int, device: Any) -> str:
    info = safe_call(device, "get_device_info", None)
    if info is None:
        return f"[{index}] <unknown device>"

    name = safe_call(info, "get_name", "unknown")
    serial = safe_call(info, "get_serial_number", "")
    uid = safe_call(info, "get_uid", "")
    connection = safe_call(info, "get_connection_type", "")
    vid = safe_call(info, "get_vid", "")
    pid = safe_call(info, "get_pid", "")
    firmware = safe_call(info, "get_firmware_version", "")

    details = [
        f"name={name}",
        f"serial={serial or '-'}",
        f"uid={uid or '-'}",
        f"connection={connection or '-'}",
        f"vid={vid}",
        f"pid={pid}",
        f"firmware={firmware or '-'}",
    ]
    return f"[{index}] " + ", ".join(details)


def list_devices(sdk: Any) -> int:
    device_list = get_device_list(sdk)
    device_count = get_device_count(device_list)
    print(f"Orbbec SDK devices: {device_count}")
    for index in range(device_count):
        print("  " + device_info_row(index, get_device_by_index(device_list, index)))
    return 0


def list_stream_profiles(sdk: Any, device_index: int | None) -> int:
    if device_index is None:
        pipeline = sdk.Pipeline()
        print("Orbbec SDK device: default")
    else:
        device = require_device_by_index(sdk, device_index)
        pipeline = create_pipeline_for_device(sdk, device)
        print(f"Orbbec SDK device index: {device_index}")
        print("  " + device_info_row(device_index, device))

    for label, sensor_member in (
        ("Depth profiles", "DEPTH_SENSOR"),
        ("Color profiles", "COLOR_SENSOR"),
    ):
        print(f"{label}:")
        sensor_type = enum_member(sdk, "OBSensorType", sensor_member)
        profile_list = pipeline.get_stream_profile_list(sensor_type)
        profile_count = get_profile_count(profile_list)
        for index in range(profile_count):
            profile = get_profile_by_index(profile_list, index)
            video_profile = safe_call(profile, "as_video_stream_profile", None)
            if video_profile is not None:
                profile = video_profile
            print(f"  [{index}] {describe_profile(profile)}")
    return 0


def configure_align_mode(sdk: Any, config: Any, align_mode: str) -> None:
    if align_mode == "off":
        return

    align_enum = getattr(sdk, "OBAlignMode", None)
    setter = getattr(config, "set_align_mode", None)
    if align_enum is None or setter is None:
        raise RuntimeError("当前 pyorbbecsdk 不支持设置 align mode。")

    member_name = {
        "sw": "SW_MODE",
        "hw": "HW_MODE",
    }[align_mode]
    setter(getattr(align_enum, member_name))
    print(f"Align mode: {member_name}")


def start_pipeline(args: argparse.Namespace, sdk: Any) -> tuple[Any, Any, Any | None]:
    if args.device_index is None:
        print("Device: SDK default")
        pipeline = sdk.Pipeline()
    else:
        device = require_device_by_index(sdk, args.device_index)
        print("Device: " + device_info_row(args.device_index, device))
        pipeline = create_pipeline_for_device(sdk, device)
    config = sdk.Config()

    depth_profile = select_stream_profile(
        sdk,
        pipeline,
        "DEPTH_SENSOR",
        args.depth_width,
        args.depth_height,
        args.depth_fps,
        args.depth_format,
        default_specific_format="y16",
    )
    config.enable_stream(depth_profile)

    color_profile = None
    if not args.no_color:
        color_profile = select_stream_profile(
            sdk,
            pipeline,
            "COLOR_SENSOR",
            args.color_width,
            args.color_height,
            args.color_fps,
            args.color_format,
            default_specific_format="mjpg",
        )
        config.enable_stream(color_profile)

    configure_align_mode(sdk, config, args.align_mode)

    print(f"Depth profile: {describe_profile(depth_profile)}")
    if color_profile is not None:
        print(f"Color profile: {describe_profile(color_profile)}")

    pipeline.start(config)

    if color_profile is not None and hasattr(pipeline, "enable_frame_sync"):
        try:
            pipeline.enable_frame_sync()
        except Exception as exc:
            print(f"提示：开启 frame sync 失败，将继续采集：{exc}", file=sys.stderr)

    return pipeline, depth_profile, color_profile


def stop_pipeline(pipeline: Any) -> None:
    stop = getattr(pipeline, "stop", None)
    if stop is None:
        return
    try:
        stop()
    except Exception as exc:
        print(f"提示：停止 Orbbec pipeline 时出现异常：{exc}", file=sys.stderr)


def depth_frame_to_array(depth_frame: Any):
    np = load_numpy()
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    depth_scale = float(depth_frame.get_depth_scale())
    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    expected_size = width * height

    if raw.size < expected_size:
        raise RuntimeError(
            f"深度帧数据不足：期望 {expected_size} 像素，实际 {raw.size}。"
        )

    return raw[:expected_size].reshape((height, width)).copy(), depth_scale


def depth_raw_to_meters(raw_depth: Any, depth_scale: float):
    return raw_depth.astype("float32") * float(depth_scale) / 1000.0


def depth_stats(depth_m: Any) -> dict[str, float | int]:
    np = load_numpy()
    valid = depth_m > 0
    valid_pixels = int(np.count_nonzero(valid))
    total_pixels = int(depth_m.size)
    valid_ratio = valid_pixels / total_pixels if total_pixels else 0.0

    stats: dict[str, float | int] = {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "valid_ratio": valid_ratio,
    }
    if valid_pixels:
        valid_depth = depth_m[valid]
        stats.update(
            {
                "min_depth_m": float(valid_depth.min()),
                "max_depth_m": float(valid_depth.max()),
                "mean_depth_m": float(valid_depth.mean()),
            }
        )
    else:
        stats.update(
            {
                "min_depth_m": 0.0,
                "max_depth_m": 0.0,
                "mean_depth_m": 0.0,
            }
        )
    return stats


def depth_m_to_uint16_png(depth_m: Any, scale_m: float):
    np = load_numpy()
    scaled = np.rint(np.clip(depth_m / scale_m, 0, 65535))
    return scaled.astype("uint16")


def color_frame_to_bgr(color_frame: Any):
    np = load_numpy()
    cv2 = load_cv2()

    width = int(color_frame.get_width())
    height = int(color_frame.get_height())
    frame_format = enum_name(color_frame.get_format())
    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

    if "MJPG" in frame_format or "MJPEG" in frame_format:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("MJPG 彩色帧解码失败。")
        return image

    if "RGB" in frame_format and data.size >= width * height * 3:
        image = data[: width * height * 3].reshape((height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if "BGR" in frame_format and data.size >= width * height * 3:
        return data[: width * height * 3].reshape((height, width, 3)).copy()

    if "YUYV" in frame_format or "YUY2" in frame_format:
        image = data[: width * height * 2].reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)

    if "UYVY" in frame_format:
        image = data[: width * height * 2].reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)

    raise RuntimeError(f"暂不支持的彩色帧格式：{frame_format}")


def save_color_image(color_frame: Any, output_path: Path) -> None:
    image = color_frame_to_bgr(color_frame)
    save_color_bgr(image, output_path)


def save_color_bgr(image: Any, output_path: Path) -> None:
    cv2 = load_cv2()
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"彩色图保存失败：{output_path}")


def save_depth_files(
    depth_m: Any,
    paths: CapturePaths,
    depth_output: str,
    depth_png_scale_m: float,
) -> dict[str, str | None]:
    saved = {
        "depth_m_npy": None,
        "depth_m_png": None,
    }

    if depth_output in ("both", "npy"):
        np = load_numpy()
        np.save(paths.depth_m_npy, depth_m.astype("float32"))
        saved["depth_m_npy"] = str(paths.depth_m_npy)

    if depth_output in ("both", "png"):
        cv2 = load_cv2()
        depth_png = depth_m_to_uint16_png(depth_m, depth_png_scale_m)
        if not cv2.imwrite(str(paths.depth_m_png), depth_png):
            raise RuntimeError(f"深度 PNG 保存失败：{paths.depth_m_png}")
        saved["depth_m_png"] = str(paths.depth_m_png)

    return saved


def write_metadata(
    session_dir: Path,
    paths: CapturePaths,
    metadata: dict[str, Any],
) -> None:
    text = json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
    paths.metadata_json.write_text(text + "\n", encoding="utf-8")

    manifest_path = session_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as manifest:
        manifest.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")


def print_depth_stats(stats: dict[str, float | int]) -> None:
    print(
        "Depth stats: "
        f"valid={stats['valid_pixels']}/{stats['total_pixels']} "
        f"({stats['valid_ratio']:.2%}), "
        f"min={stats['min_depth_m']:.3f}m, "
        f"max={stats['max_depth_m']:.3f}m, "
        f"mean={stats['mean_depth_m']:.3f}m"
    )


def depth_preview_image(depth_m: Any, max_depth_m: float):
    np = load_numpy()
    cv2 = load_cv2()
    clipped = np.clip(depth_m, 0, max_depth_m)
    valid = clipped > 0
    preview = np.zeros(depth_m.shape, dtype=np.uint8)
    preview[valid] = np.rint(clipped[valid] * 255.0 / max_depth_m).astype(np.uint8)
    colorized = cv2.applyColorMap(preview, cv2.COLORMAP_JET)
    colorized[~valid] = (0, 0, 0)
    return colorized


def save_depth_preview_image(depth_m: Any, output_path: Path, max_depth_m: float) -> None:
    cv2 = load_cv2()
    preview = depth_preview_image(depth_m, max_depth_m)
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"深度伪彩色图保存失败：{output_path}")


def wait_for_frames(pipeline: Any, timeout_ms: int) -> Any | None:
    frames = pipeline.wait_for_frames(timeout_ms)
    if frames is None:
        return None
    return frames


def get_depth_frame(frames: Any) -> Any | None:
    getter = getattr(frames, "get_depth_frame", None)
    if getter is None:
        return None
    return getter()


def get_color_frame(frames: Any) -> Any | None:
    getter = getattr(frames, "get_color_frame", None)
    if getter is None:
        return None
    return getter()


def create_session_dirs(
    output_dir: Path,
    *,
    include_color: bool,
) -> tuple[Path, Path, Path]:
    session_dir = output_dir / timestamp_label()
    rgb_dir = session_dir / "rgb"
    depth_dir = session_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    if include_color:
        rgb_dir.mkdir(parents=True, exist_ok=True)
    print(f"RGBD output directory: {session_dir}")
    return session_dir, rgb_dir, depth_dir


def save_rgbd_frame(
    args: argparse.Namespace,
    session_dir: Path,
    rgb_dir: Path,
    depth_dir: Path,
    frame_number: int,
    raw_depth: Any,
    depth_m: Any,
    depth_scale: float,
    stats: dict[str, float | int],
    *,
    color_frame: Any | None = None,
    color_image: Any | None = None,
) -> CapturePaths:
    paths = build_capture_paths(
        session_dir,
        args.prefix,
        frame_number,
        args.color_output,
        include_color=not args.no_color,
    )

    saved_depth_paths = save_depth_files(
        depth_m,
        paths,
        args.depth_output,
        args.depth_png_scale_m,
    )
    depth_preview_path = None
    if args.save_depth_preview:
        save_depth_preview_image(
            depth_m,
            paths.depth_preview_image,
            args.depth_preview_max_m,
        )
        depth_preview_path = str(paths.depth_preview_image)

    color_path = None
    if not args.no_color:
        if color_image is not None and paths.color_image is not None:
            save_color_bgr(color_image, paths.color_image)
            color_path = str(paths.color_image)
        elif color_frame is not None and paths.color_image is not None:
            save_color_image(color_frame, paths.color_image)
            color_path = str(paths.color_image)
        else:
            print("警告：当前帧集中没有彩色帧，仅保存深度。", file=sys.stderr)

    metadata = {
        "stem": paths.stem,
        "captured_at": datetime.now().isoformat(timespec="milliseconds"),
        "session_dir": str(session_dir),
        "rgb_dir": str(rgb_dir) if not args.no_color else None,
        "depth_dir": str(depth_dir),
        "device_index": args.device_index,
        "align_mode": args.align_mode,
        "requested_depth_profile": {
            "width": args.depth_width or None,
            "height": args.depth_height or None,
            "fps": args.depth_fps or None,
            "format": args.depth_format,
        },
        "requested_color_profile": None
        if args.no_color
        else {
            "width": args.color_width or None,
            "height": args.color_height or None,
            "fps": args.color_fps or None,
            "format": args.color_format,
        },
        "depth_unit": "m",
        "raw_depth_scale_mm": depth_scale,
        "depth_unit_note": "depth_m = raw_depth * raw_depth_scale_mm / 1000",
        "depth_png_scale_m": args.depth_png_scale_m,
        "depth_png_note": "depth_m = png_value * depth_png_scale_m",
        "depth_width": int(raw_depth.shape[1]),
        "depth_height": int(raw_depth.shape[0]),
        "depth_m_npy": saved_depth_paths["depth_m_npy"],
        "depth_m_png": saved_depth_paths["depth_m_png"],
        "depth_preview_image": depth_preview_path,
        "depth_preview_max_m": args.depth_preview_max_m if depth_preview_path else None,
        "color_image": color_path,
        "stats": stats,
    }
    metadata["metadata_json"] = str(paths.metadata_json)
    write_metadata(session_dir, paths, metadata)
    return paths


def capture_loop(args: argparse.Namespace, pipeline: Any, output_dir: Path) -> int:
    session_dir, rgb_dir, depth_dir = create_session_dirs(
        output_dir,
        include_color=not args.no_color,
    )

    for _ in range(args.warmup_frames):
        wait_for_frames(pipeline, args.timeout_ms)

    saved_count = 0
    missed_count = 0
    target_count = None if args.frames == 0 else args.frames

    while target_count is None or saved_count < target_count:
        frames = wait_for_frames(pipeline, args.timeout_ms)
        if frames is None:
            missed_count += 1
            print(
                f"警告：等待 RGBD 帧超时 ({missed_count}/{args.max_missed_frames})。",
                file=sys.stderr,
            )
            if missed_count >= args.max_missed_frames:
                print("错误：连续等待 RGBD 帧失败。", file=sys.stderr)
                return 1
            continue

        depth_frame = get_depth_frame(frames)
        if depth_frame is None:
            missed_count += 1
            print(
                f"警告：当前帧集中没有深度帧 ({missed_count}/{args.max_missed_frames})。",
                file=sys.stderr,
            )
            if missed_count >= args.max_missed_frames:
                print("错误：连续没有收到深度帧。", file=sys.stderr)
                return 1
            continue

        missed_count = 0
        raw_depth, depth_scale = depth_frame_to_array(depth_frame)
        depth_m = depth_raw_to_meters(raw_depth, depth_scale)
        stats = depth_stats(depth_m)
        print_depth_stats(stats)

        if args.check_only:
            if stats["valid_ratio"] < args.min_valid_ratio:
                print(
                    "错误：已收到深度帧，但有效深度像素比例低于阈值。"
                    f" threshold={args.min_valid_ratio:g}",
                    file=sys.stderr,
                )
                return 1
            print("深度检测通过：已收到有效深度帧。")
            return 0

        frame_number = saved_count + 1
        color_frame = None
        if not args.no_color:
            color_frame = get_color_frame(frames)

        paths = save_rgbd_frame(
            args,
            session_dir,
            rgb_dir,
            depth_dir,
            frame_number,
            raw_depth,
            depth_m,
            depth_scale,
            stats,
            color_frame=color_frame,
        )

        saved_count += 1
        print(f"已保存 RGBD 帧 {saved_count}: {session_dir / paths.stem}")

        if args.interval > 0 and (target_count is None or saved_count < target_count):
            time.sleep(args.interval)

    return 0


def viewer_loop(args: argparse.Namespace, pipeline: Any, output_dir: Path) -> int:
    cv2 = load_cv2()
    session_dir, rgb_dir, depth_dir = create_session_dirs(
        output_dir,
        include_color=not args.no_color,
    )

    for _ in range(args.warmup_frames):
        wait_for_frames(pipeline, args.timeout_ms)

    color_window = "Orbbec RGB"
    depth_window = f"Orbbec Depth 0-{args.depth_preview_max_m:g}m"
    if not args.no_color:
        cv2.namedWindow(color_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(depth_window, cv2.WINDOW_NORMAL)

    print("RGBD 预览已打开：按 s 同时保存 RGB 和深度；按 i 打印深度统计；按 q 或 Esc 退出。")

    saved_count = 0
    missed_count = 0
    latest: dict[str, Any] | None = None

    try:
        while True:
            frames = wait_for_frames(pipeline, args.timeout_ms)
            if frames is None:
                missed_count += 1
                print(
                    f"警告：等待 RGBD 帧超时 ({missed_count}/{args.max_missed_frames})。",
                    file=sys.stderr,
                )
                if missed_count >= args.max_missed_frames:
                    print("错误：连续等待 RGBD 帧失败。", file=sys.stderr)
                    return 1
                key = cv2.waitKey(10) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            depth_frame = get_depth_frame(frames)
            if depth_frame is None:
                missed_count += 1
                print(
                    f"警告：当前帧集中没有深度帧 ({missed_count}/{args.max_missed_frames})。",
                    file=sys.stderr,
                )
                if missed_count >= args.max_missed_frames:
                    print("错误：连续没有收到深度帧。", file=sys.stderr)
                    return 1
                key = cv2.waitKey(10) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            missed_count = 0
            raw_depth, depth_scale = depth_frame_to_array(depth_frame)
            depth_m = depth_raw_to_meters(raw_depth, depth_scale)
            stats = depth_stats(depth_m)
            color_image = None

            if not args.no_color:
                color_frame = get_color_frame(frames)
                if color_frame is not None:
                    try:
                        color_image = color_frame_to_bgr(color_frame)
                        cv2.imshow(color_window, color_image)
                    except RuntimeError as exc:
                        print(f"警告：彩色帧解码失败：{exc}", file=sys.stderr)
                else:
                    print("警告：当前帧集中没有彩色帧。", file=sys.stderr)

            cv2.imshow(depth_window, depth_preview_image(depth_m, args.depth_preview_max_m))

            latest = {
                "raw_depth": raw_depth,
                "depth_m": depth_m,
                "depth_scale": depth_scale,
                "stats": stats,
                "color_image": color_image,
            }

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("i"):
                print_depth_stats(stats)
            if key == ord("s"):
                if latest is None:
                    print("警告：当前还没有可保存的 RGBD 帧。", file=sys.stderr)
                    continue
                if not args.no_color and latest["color_image"] is None:
                    print("警告：当前没有可保存的彩色帧，本次仅保存深度。", file=sys.stderr)

                frame_number = saved_count + 1
                paths = save_rgbd_frame(
                    args,
                    session_dir,
                    rgb_dir,
                    depth_dir,
                    frame_number,
                    latest["raw_depth"],
                    latest["depth_m"],
                    latest["depth_scale"],
                    latest["stats"],
                    color_image=latest["color_image"],
                )
                saved_count += 1
                print(f"已保存 RGBD 帧 {saved_count}: {session_dir / paths.stem}")
    finally:
        cv2.destroyAllWindows()

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not validate_args(args):
        return 1

    sdk = load_orbbec_sdk()
    if args.list_devices:
        try:
            return list_devices(sdk)
        except Exception as exc:
            print(f"错误：枚举 Orbbec 设备失败：{exc}", file=sys.stderr)
            print(
                "当前 SDK 版本可能不支持 Context 设备枚举；单相机时可以不传 "
                "--device-index，直接使用 SDK 默认设备。",
                file=sys.stderr,
            )
            return 1
    if args.list_profiles:
        try:
            return list_stream_profiles(sdk, args.device_index)
        except Exception as exc:
            print(f"错误：枚举 Orbbec profile 失败：{exc}", file=sys.stderr)
            return 1

    load_numpy()
    if (
        args.viewer
        or args.depth_output in ("both", "png")
        or args.save_depth_preview
        or not args.no_color
    ):
        load_cv2()

    output_dir = resolve_project_path(args.output_dir)

    try:
        pipeline, _, _ = start_pipeline(args, sdk)
    except Exception as exc:
        print(f"错误：启动 Orbbec RGBD pipeline 失败：{exc}", file=sys.stderr)
        print("请确认设备已连接，当前用户有 USB 设备访问权限，并已安装 Orbbec SDK。", file=sys.stderr)
        return 1

    try:
        if args.viewer:
            return viewer_loop(args, pipeline, output_dir)
        return capture_loop(args, pipeline, output_dir)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在退出。")
        return 0
    except Exception as exc:
        print(f"错误：RGBD 采集失败：{exc}", file=sys.stderr)
        return 1
    finally:
        stop_pipeline(pipeline)


if __name__ == "__main__":
    raise SystemExit(main())
