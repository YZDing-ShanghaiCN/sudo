#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


DEFAULT_DEVICE = "/dev/video2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List supported V4L2 formats for a camera device."
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"camera device node, default: {DEFAULT_DEVICE}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tool = shutil.which("v4l2-ctl")

    if tool is None:
        print("错误：系统中没有找到 v4l2-ctl。", file=sys.stderr)
        print("请先安装 v4l-utils：", file=sys.stderr)
        print("  sudo apt update", file=sys.stderr)
        print("  sudo apt install v4l-utils", file=sys.stderr)
        return 1

    command = [tool, "-d", args.device, "--list-formats-ext"]
    result = subprocess.run(command, check=False)

    if result.returncode != 0:
        print(f"错误：v4l2-ctl 执行失败，设备节点：{args.device}", file=sys.stderr)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
