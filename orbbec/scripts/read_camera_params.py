#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import capture_rgbd_orbbec_sdk as rgbd


DEFAULT_OUTPUT = "outputs/camera_params/orbbec_camera_params.json"
SENSOR_CHOICES = ("all", "depth", "color")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Orbbec intrinsics and distortion coefficients for each "
            "supported depth/color resolution."
        )
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"JSON output path, default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="optional Orbbec SDK device index; omitted uses the SDK default device",
    )
    parser.add_argument(
        "--sensor",
        default="all",
        choices=SENSOR_CHOICES,
        help="which sensor profiles to export, default: all",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> bool:
    if args.device_index is not None and args.device_index < 0:
        print("错误：--device-index 不能小于 0。", file=sys.stderr)
        return False
    return True


def resolve_project_path(path_text: str) -> Path:
    return rgbd.resolve_project_path(path_text)


def sdk_version(sdk: Any) -> str:
    get_version = getattr(sdk, "get_version", None)
    if get_version is not None:
        try:
            return str(get_version())
        except Exception:
            pass
    return str(getattr(sdk, "__version__", "unknown"))


def simple_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def device_info_dict(device: Any | None) -> dict[str, Any] | None:
    if device is None:
        return None

    info = rgbd.safe_call(device, "get_device_info", None)
    if info is None:
        return None

    fields = (
        ("name", "get_name"),
        ("serial_number", "get_serial_number"),
        ("firmware_version", "get_firmware_version"),
    )
    result: dict[str, Any] = {}
    for key, method_name in fields:
        value = rgbd.safe_call(info, method_name, None)
        if value is not None:
            result[key] = simple_value(value)
    return result


def intrinsic_to_dict(intrinsic: Any) -> dict[str, Any]:
    fx = float(intrinsic.fx)
    fy = float(intrinsic.fy)
    cx = float(intrinsic.cx)
    cy = float(intrinsic.cy)
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
    }


def distortion_to_dict(distortion: Any) -> dict[str, Any]:
    return {
        key: float(getattr(distortion, key))
        for key in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
        if hasattr(distortion, key)
    }


def opencv_distortion(distortion: dict[str, float]) -> list[float]:
    return [
        distortion.get("k1", 0.0),
        distortion.get("k2", 0.0),
        distortion.get("p1", 0.0),
        distortion.get("p2", 0.0),
        distortion.get("k3", 0.0),
    ]


def create_pipeline(args: argparse.Namespace, sdk: Any) -> tuple[Any, Any | None]:
    if args.device_index is None:
        print("Device: SDK default")
        pipeline = sdk.Pipeline()
        return pipeline, rgbd.safe_call(pipeline, "get_device", None)

    device = rgbd.require_device_by_index(sdk, args.device_index)
    print("Device: " + rgbd.device_info_row(args.device_index, device))
    return rgbd.create_pipeline_for_device(sdk, device), device


def iter_video_profiles(sdk: Any, pipeline: Any, sensor_member: str):
    sensor_type = rgbd.enum_member(sdk, "OBSensorType", sensor_member)
    profile_list = pipeline.get_stream_profile_list(sensor_type)
    for index in range(rgbd.get_profile_count(profile_list)):
        profile = rgbd.get_profile_by_index(profile_list, index)
        video_profile = rgbd.safe_call(profile, "as_video_stream_profile", None)
        if video_profile is not None:
            profile = video_profile
        yield profile


def calibration_for_profile(profile: Any) -> dict[str, Any]:
    width = int(profile.get_width())
    height = int(profile.get_height())
    distortion = distortion_to_dict(profile.get_distortion())
    return {
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "intrinsic": intrinsic_to_dict(profile.get_intrinsic()),
        "distortion": distortion,
        "opencv_distortion_k1_k2_p1_p2_k3": opencv_distortion(distortion),
    }


def same_calibration(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["intrinsic"] == right["intrinsic"]
        and left["distortion"] == right["distortion"]
        and left["opencv_distortion_k1_k2_p1_p2_k3"]
        == right["opencv_distortion_k1_k2_p1_p2_k3"]
    )


def append_resolution(
    resolutions: dict[str, Any],
    profile_calibration: dict[str, Any],
) -> None:
    key = profile_calibration["resolution"]
    existing = resolutions.get(key)
    if existing is None:
        resolutions[key] = profile_calibration
        return

    if same_calibration(existing, profile_calibration):
        return

    variants = existing.setdefault("variants", [])
    if not variants:
        variants.append(
            {
                "width": existing["width"],
                "height": existing["height"],
                "intrinsic": existing["intrinsic"],
                "distortion": existing["distortion"],
                "opencv_distortion_k1_k2_p1_p2_k3": existing[
                    "opencv_distortion_k1_k2_p1_p2_k3"
                ],
            }
        )
    variants.append(
        {
            "width": profile_calibration["width"],
            "height": profile_calibration["height"],
            "intrinsic": profile_calibration["intrinsic"],
            "distortion": profile_calibration["distortion"],
            "opencv_distortion_k1_k2_p1_p2_k3": profile_calibration[
                "opencv_distortion_k1_k2_p1_p2_k3"
            ],
        }
    )


def sort_resolution_map(resolutions: dict[str, Any]) -> dict[str, Any]:
    sorted_items = sorted(
        resolutions.items(),
        key=lambda item: (int(item[1]["width"]), int(item[1]["height"]), item[0]),
    )
    result = {}
    for key, value in sorted_items:
        result[key] = value
    return result


def read_sensor_calibrations(
    sdk: Any,
    pipeline: Any,
    sensor_member: str,
) -> dict[str, Any]:
    resolutions: dict[str, Any] = {}
    errors = []
    for profile in iter_video_profiles(sdk, pipeline, sensor_member):
        try:
            append_resolution(resolutions, calibration_for_profile(profile))
        except Exception as exc:
            errors.append(
                {
                    "profile": rgbd.describe_profile(profile),
                    "error": str(exc),
                }
            )

    data: dict[str, Any] = sort_resolution_map(resolutions)
    if errors:
        data["_errors"] = errors
    return data


def build_payload(args: argparse.Namespace, sdk: Any, pipeline: Any, device: Any | None):
    payload: dict[str, Any] = {}

    if args.sensor in ("all", "depth"):
        payload["depth"] = read_sensor_calibrations(sdk, pipeline, "DEPTH_SENSOR")
    if args.sensor in ("all", "color"):
        payload["color"] = read_sensor_calibrations(sdk, pipeline, "COLOR_SENSOR")

    payload["device"] = device_info_dict(device)
    payload["sdk_version"] = sdk_version(sdk)
    payload["captured_at"] = datetime.now().isoformat(timespec="seconds")
    payload["note"] = (
        "Intrinsics and distortion are exported per sensor resolution/profile "
        "from Orbbec SDK."
    )
    return payload


def print_sensor_summary(label: str, data: dict[str, Any]) -> None:
    resolutions = {key: value for key, value in data.items() if not key.startswith("_")}
    print(f"{label}: {len(resolutions)} resolutions")
    for resolution, item in resolutions.items():
        intrinsic = item["intrinsic"]
        distortion = item["opencv_distortion_k1_k2_p1_p2_k3"]
        print(
            f"  {resolution}: "
            f"fx={intrinsic['fx']:.3f}, fy={intrinsic['fy']:.3f}, "
            f"cx={intrinsic['cx']:.3f}, cy={intrinsic['cy']:.3f}, "
            f"dist=[{', '.join(f'{value:.6g}' for value in distortion)}]"
        )
        if "variants" in item:
            print("    提示：该分辨率存在多组不同内参，JSON 中已放入 variants。")


def print_summary(payload: dict[str, Any]) -> None:
    print()
    print("=== Camera Intrinsics By Resolution ===")
    if "depth" in payload:
        print_sensor_summary("Depth", payload["depth"])
    if "color" in payload:
        print_sensor_summary("Color", payload["color"])


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not validate_args(args):
        return 1

    sdk = rgbd.load_orbbec_sdk()
    try:
        pipeline, device = create_pipeline(args, sdk)
        payload = build_payload(args, sdk, pipeline, device)
    except Exception as exc:
        print(f"错误：读取 Orbbec 相机参数失败：{exc}", file=sys.stderr)
        print("请确认设备已连接，当前用户有 USB 设备访问权限，并已安装 Orbbec SDK。", file=sys.stderr)
        return 1

    output_path = resolve_project_path(args.output)
    try:
        write_payload(output_path, payload)
    except Exception as exc:
        print(f"错误：保存 JSON 失败：{exc}", file=sys.stderr)
        return 1

    print_summary(payload)
    print()
    print(f"已保存 JSON：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
