#!/usr/bin/env python3
"""Interactively save ordered RGB-D test captures into one timestamped session."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_MODULE_PATH = PROJECT_ROOT / "scripts" / "capture_rgbd_orbbec_sdk.py"
RESULTS_ROOT = PROJECT_ROOT / "results_test"
PROJECT_NAME = "DemoProject"


def load_capture_module():
    spec = importlib.util.spec_from_file_location(
        "capture_rgbd_orbbec_sdk_loop_test",
        CAPTURE_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载采集模块：{CAPTURE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def session_name(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%y%m%d_%H%M%S")


def create_project_dirs(session_dir: Path) -> tuple[Path, Path, Path]:
    try:
        session_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(f"测试目录已存在，拒绝覆盖：{session_dir}") from exc

    data_dir = session_dir / "data"
    model_dir = session_dir / "model"
    recon_dir = session_dir / "recon"
    model_dir.mkdir()
    recon_dir.mkdir()
    return data_dir, model_dir, recon_dir


def profile_configuration(color_profile: Any) -> tuple[list[int], list[list[float]]]:
    if color_profile is None:
        raise RuntimeError("没有启用彩色相机 profile，无法生成 configuration.json。")

    intrinsic = color_profile.get_intrinsic()
    resolution = [int(color_profile.get_width()), int(color_profile.get_height())]
    intrinsic_matrix = [
        [float(intrinsic.fx), 0.0, float(intrinsic.cx)],
        [0.0, float(intrinsic.fy), float(intrinsic.cy)],
        [0.0, 0.0, 1.0],
    ]
    return resolution, intrinsic_matrix


def build_configuration(
    data_dir: Path,
    model_dir: Path,
    recon_dir: Path,
    color_profile: Any,
) -> dict[str, Any]:
    resolution, intrinsic_matrix = profile_configuration(color_profile)
    return {
        "projectname": PROJECT_NAME,
        "environment": {
            "modelsrc": str(model_dir.resolve()),
            "reconstructionsrc": str(recon_dir.resolve()),
            "datasrc": str(data_dir.resolve()),
        },
        "camera": {
            "resolution": resolution,
            "intrinsic": intrinsic_matrix,
            "inverse_pose": False,
            "lens": 30.0,
        },
        "reconstruction": {
            "scale": 1.0,
            "cameradisplayscale": 0.01,
            "recon_trans": "1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,1;",
        },
        "data": {
            "sample_rate": 0.1,
            "depth_scale": 0.001,
            "depth_ignore": 8.0,
        },
    }


def write_configuration(session_dir: Path, configuration: dict[str, Any]) -> Path:
    output_path = session_dir / "configuration.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(configuration, file, ensure_ascii=False, indent=4)
        file.write("\n")
    return output_path


def capture_args(capture_rgbd: Any, session_dir: Path):
    return capture_rgbd.parse_args(
        [
            "--viewer",
            "--output-dir",
            str(session_dir),
            "--session-name",
            "data",
            "--no-metadata",
        ]
    )


def main() -> int:
    capture_rgbd = load_capture_module()
    session_dir = RESULTS_ROOT / session_name()
    args = capture_args(capture_rgbd, session_dir)
    if not capture_rgbd.validate_args(args):
        return 1

    sdk = capture_rgbd.load_orbbec_sdk()
    capture_rgbd.load_numpy()
    capture_rgbd.load_cv2()

    try:
        pipeline, _, color_profile = capture_rgbd.start_pipeline(args, sdk)
    except Exception as exc:
        print(f"错误：启动 Orbbec RGBD pipeline 失败：{exc}", file=sys.stderr)
        print("请确认设备已连接、当前用户有 USB 访问权限并已安装 Orbbec SDK。", file=sys.stderr)
        return 1

    try:
        data_dir, model_dir, recon_dir = create_project_dirs(session_dir)
        configuration = build_configuration(
            data_dir,
            model_dir,
            recon_dir,
            color_profile,
        )
        config_path = write_configuration(session_dir, configuration)
        print(f"Project configuration: {config_path}")
        return capture_rgbd.viewer_loop(args, pipeline, session_dir)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在退出。")
        return 0
    except Exception as exc:
        print(f"错误：RGBD 测试采集失败：{exc}", file=sys.stderr)
        return 1
    finally:
        capture_rgbd.stop_pipeline(pipeline)


if __name__ == "__main__":
    raise SystemExit(main())
