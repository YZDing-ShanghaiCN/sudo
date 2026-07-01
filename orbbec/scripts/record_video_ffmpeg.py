#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_DEVICE = "/dev/video2"
DEFAULT_VIDEO_DIR = "outputs/videos"
DEFAULT_WIDTH = 3840
DEFAULT_HEIGHT = 2160
DEFAULT_FPS = 30
DEFAULT_INPUT_FORMAT = "mjpeg"
INPUT_FORMAT_CHOICES = ("mjpeg", "yuyv422")
DEFAULT_OUTPUT = ""
DEFAULT_CONTAINER = "mp4"
CONTAINER_CHOICES = ("mp4",)
DEFAULT_CODEC = "copy"
CODEC_CHOICES = ("copy", "libx264")
DEFAULT_PRESET = "veryfast"
DEFAULT_CRF = 23
DEFAULT_PIXEL_FORMAT = "yuv420p"
DEFAULT_DURATION = 0.0
DEFAULT_OVERWRITE = False
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record Orbbec RGB/UVC video through ffmpeg."
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"camera device node, default: {DEFAULT_DEVICE}",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"target video width, default: {DEFAULT_WIDTH}",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"target video height, default: {DEFAULT_HEIGHT}",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"target frames per second, default: {DEFAULT_FPS}",
    )
    parser.add_argument(
        "--input-format",
        default=DEFAULT_INPUT_FORMAT,
        choices=INPUT_FORMAT_CHOICES,
        help=f"V4L2 input pixel format, default: {DEFAULT_INPUT_FORMAT}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "output video path, default: "
            f"{DEFAULT_VIDEO_DIR}/orbbec_WIDTHxHEIGHT_FPSfps_CODEC_TIMESTAMP.mp4"
        ),
    )
    parser.add_argument(
        "--container",
        default=DEFAULT_CONTAINER,
        choices=CONTAINER_CHOICES,
        help=f"output container, fixed: {DEFAULT_CONTAINER}",
    )
    parser.add_argument(
        "--codec",
        default=DEFAULT_CODEC,
        choices=CODEC_CHOICES,
        help=(
            "output video codec, default: copy. "
            "Use copy for stable 4K capture; use libx264 for smaller MP4 files."
        ),
    )
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        help=f"libx264 preset, default: {DEFAULT_PRESET}",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=DEFAULT_CRF,
        help=f"libx264 quality CRF, default: {DEFAULT_CRF}",
    )
    parser.add_argument(
        "--pix-fmt",
        default=DEFAULT_PIXEL_FORMAT,
        help=f"libx264 pixel format, default: {DEFAULT_PIXEL_FORMAT}",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help=(
            "optional recording duration in seconds, default: "
            f"{DEFAULT_DURATION:g} means record until q/Ctrl+C"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERWRITE,
        help="overwrite output file if it already exists",
    )
    return parser.parse_args()


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_output_path(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (
        f"orbbec_{args.width}x{args.height}_{args.fps}fps_"
        f"{args.codec}_{timestamp}.{args.container}"
    )
    return PROJECT_ROOT / DEFAULT_VIDEO_DIR / name


def ffmpeg_output_format(container: str) -> str:
    formats = {
        "mp4": "mp4",
    }
    return formats[container]


def build_ffmpeg_command(args: argparse.Namespace, output_path: Path) -> list[str]:
    overwrite_flag = "-y" if args.overwrite else "-n"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        overwrite_flag,
        "-fflags",
        "+genpts",
        "-f",
        "v4l2",
        "-thread_queue_size",
        "512",
        "-rtbufsize",
        "512M",
        "-use_wallclock_as_timestamps",
        "1",
        "-input_format",
        args.input_format,
        "-video_size",
        f"{args.width}x{args.height}",
        "-framerate",
        str(args.fps),
        "-i",
        args.device,
        "-avoid_negative_ts",
        "make_zero",
    ]

    if args.codec == "copy":
        command.extend(["-c:v", "copy"])
    else:
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                args.preset,
                "-crf",
                str(args.crf),
                "-pix_fmt",
                args.pix_fmt,
            ]
        )
        if args.container == "mp4":
            command.extend(["-movflags", "+faststart"])

    command.extend(["-f", ffmpeg_output_format(args.container)])

    if args.duration > 0:
        command.extend(["-t", str(args.duration)])

    command.append(str(output_path))
    return command


def print_command(command: list[str]) -> None:
    print("将执行 ffmpeg 命令:")
    print("  " + " ".join(command))
    print("录制中建议按 q 正常停止；按 Ctrl+C 时脚本会等待 ffmpeg 写完文件索引。")
    sys.stdout.flush()


def validate_args(args: argparse.Namespace) -> bool:
    if args.duration < 0:
        print("错误：--duration 不能小于 0。", file=sys.stderr)
        return False
    if args.crf < 0 or args.crf > 51:
        print("错误：--crf 必须在 0 到 51 之间。", file=sys.stderr)
        return False
    return True


def validate_output_path(output_path: Path) -> bool:
    if output_path.suffix.lower() != ".mp4":
        print("错误：录像输出统一使用 MP4，请把 --output 后缀改为 .mp4。", file=sys.stderr)
        return False
    return True


def run_ffmpeg(command: list[str]) -> int:
    process = subprocess.Popen(command)

    try:
        return process.wait()
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在等待 ffmpeg 收尾写入文件索引，请稍等。")
        try:
            return process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            print("ffmpeg 未及时退出，发送 SIGINT。", file=sys.stderr)
            process.send_signal(signal.SIGINT)

        try:
            return process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            print("ffmpeg 仍未退出，终止进程。当前文件可能无法播放。", file=sys.stderr)
            process.terminate()

        try:
            return process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()


def main() -> int:
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        print("错误：系统中没有找到 ffmpeg。", file=sys.stderr)
        print("请先安装：", file=sys.stderr)
        print("  sudo apt update", file=sys.stderr)
        print("  sudo apt install ffmpeg", file=sys.stderr)
        return 1

    if not validate_args(args):
        return 1

    output_path = (
        resolve_project_path(args.output)
        if args.output
        else default_output_path(args)
    )
    if not validate_output_path(output_path):
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_ffmpeg_command(args, output_path)
    print_command(command)
    return_code = run_ffmpeg(command)

    if return_code == 0:
        print(f"录像已保存：{output_path}")
    else:
        print(f"ffmpeg 退出码：{return_code}", file=sys.stderr)
        print("请检查设备节点、分辨率、FPS 和 --input-format 是否匹配。", file=sys.stderr)
        if args.container == "mp4":
            print(
                "如果 MP4 不能播放，请优先用 q 停止或使用 --duration 自动结束。",
                file=sys.stderr,
            )

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
