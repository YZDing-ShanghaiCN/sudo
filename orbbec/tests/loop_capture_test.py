#!/usr/bin/env python3
"""Interactively save ordered RGB-D test captures into one timestamped session."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_MODULE_PATH = PROJECT_ROOT / "scripts" / "capture_rgbd_orbbec_sdk.py"


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


def main() -> int:
    capture_rgbd = load_capture_module()
    return capture_rgbd.main(
        [
            "--viewer",
            "--output-dir",
            "results_test",
            "--session-name",
            session_name(),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
