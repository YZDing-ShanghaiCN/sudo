#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_DEVICE = "/dev/video2"
DEFAULT_SAVE_DIR = "outputs/screenshots"
DEFAULT_MAX_READ_ERRORS = 30
FALLBACK_DEVICES = ("/dev/video3", "/dev/video4", "/dev/video5")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BACKEND_CHOICES = ("any", "gstreamer", "v4l2")
cv2 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View an Orbbec Femto Bolt RGB/UVC camera stream with OpenCV."
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"camera device node, default: {DEFAULT_DEVICE}",
    )
    parser.add_argument("--width", type=int, default=3840, help="target frame width")
    parser.add_argument("--height", type=int, default=2160, help="target frame height")
    parser.add_argument("--fps", type=int, default=30, help="target frames per second")
    parser.add_argument(
        "--fourcc",
        default="MJPG",
        help="optional pixel format, for example MJPG or YUYV",
    )
    parser.add_argument(
        "--backend",
        default="v4l2",
        choices=BACKEND_CHOICES,
        help="OpenCV video backend, default: v4l2",
    )
    parser.add_argument(
        "--save-dir",
        default=DEFAULT_SAVE_DIR,
        help=f"screenshot directory, default: {DEFAULT_SAVE_DIR}",
    )
    parser.add_argument(
        "--max-read-errors",
        type=int,
        default=DEFAULT_MAX_READ_ERRORS,
        help=f"consecutive failed frame reads before exiting, default: {DEFAULT_MAX_READ_ERRORS}",
    )
    return parser.parse_args()


def load_cv2():
    global cv2

    if cv2 is not None:
        return cv2

    try:
        import cv2 as cv2_module
    except ImportError as exc:
        print("错误：无法导入 OpenCV 或 NumPy。", file=sys.stderr)
        print("请先安装 Python 依赖：", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1) from exc

    cv2 = cv2_module
    return cv2


def backend_id(backend_name: str) -> int:
    backends = {
        "any": cv2.CAP_ANY,
        "v4l2": cv2.CAP_V4L2,
        "gstreamer": cv2.CAP_GSTREAMER,
    }
    return backends[backend_name]


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def fourcc_to_text(value: float) -> str:
    code = int(value)
    if code <= 0:
        return "unknown"

    chars = [chr((code >> (8 * index)) & 0xFF) for index in range(4)]
    text = "".join(char if char.isprintable() else "?" for char in chars).strip()
    return text or f"0x{code:08x}"


def get_actual_params(cap: cv2.VideoCapture) -> dict[str, object]:
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "fourcc": fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC)),
    }


def print_requested_params(args: argparse.Namespace, save_dir: Path) -> None:
    print("请求设置的参数:")
    print(f"  device : {args.device}")
    print(f"  size   : {args.width}x{args.height}")
    print(f"  fps    : {args.fps}")
    print(f"  fourcc : {args.fourcc or 'auto'}")
    print(f"  backend: {args.backend}")
    print(f"  save   : {save_dir}")


def print_actual_params(cap: cv2.VideoCapture) -> None:
    params = get_actual_params(cap)
    print("实际生效的参数:")
    print(f"  width : {params['width']}")
    print(f"  height: {params['height']}")
    print(f"  fps   : {params['fps']:.2f}")
    print(f"  fourcc: {params['fourcc']}")


def print_open_error(device: str) -> None:
    print(f"错误：无法打开摄像头节点 {device}", file=sys.stderr)
    print("请确认设备已连接，并且当前用户有权限访问该节点。", file=sys.stderr)
    print("如果 /dev/video2 无法打开或没有画面，可以将 --device 改为：", file=sys.stderr)
    for fallback_device in FALLBACK_DEVICES:
        print(f"  {fallback_device}", file=sys.stderr)


def current_script_command() -> str:
    script = sys.argv[0] or "camera_viewer.py"
    return f"python {script}"


def print_read_error_hint(args: argparse.Namespace) -> None:
    command = current_script_command()
    print("建议尝试下面几种方式：", file=sys.stderr)
    print(
        "  1. 先确认 1080p MJPG 是否稳定："
        f"{command} --device {args.device} --width 1920 --height 1080 --fps 30 --fourcc MJPG",
        file=sys.stderr,
    )
    print(
        "  2. 如果 4K 仍然需要测试，可以先降帧率："
        f"{command} --device {args.device} --width 3840 --height 2160 --fps 15 --fourcc MJPG",
        file=sys.stderr,
    )
    print("  3. 确认相机接在 USB 3.x 口，并尝试重新插拔相机。", file=sys.stderr)
    print("  4. 如果当前节点一直失败，尝试 /dev/video3、/dev/video4、/dev/video5。", file=sys.stderr)


def short_cv2_error(exc: cv2.error) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return lines[-1] if lines else str(exc)


def read_frame(cap: cv2.VideoCapture):
    try:
        ok, frame = cap.read()
    except cv2.error as exc:
        return False, None, short_cv2_error(exc)

    if not ok or frame is None:
        return False, None, "cap.read() returned no frame"
    if getattr(frame, "size", 0) == 0:
        return False, None, "cap.read() returned an empty frame"

    return True, frame, ""


def save_frame(frame, save_dir: Path, device: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device_label = device.strip("/").replace("/", "_")
    output_path = save_dir / f"{device_label}_{timestamp}.png"

    if cv2.imwrite(str(output_path), frame):
        print(f"已保存截图：{output_path}")
    else:
        print(f"错误：截图保存失败：{output_path}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    save_dir = resolve_project_path(args.save_dir)

    if args.fourcc and len(args.fourcc) != 4:
        print("错误：--fourcc 必须是 4 个字符，例如 MJPG 或 YUYV。", file=sys.stderr)
        return 1
    if args.max_read_errors < 1:
        print("错误：--max-read-errors 必须大于等于 1。", file=sys.stderr)
        return 1

    load_cv2()
    backend = backend_id(args.backend)

    print_requested_params(args, save_dir)

    device_path = Path(args.device)
    if not device_path.exists():
        print(f"提示：设备节点 {args.device} 不存在，仍会尝试打开。", file=sys.stderr)

    cap = cv2.VideoCapture(args.device, backend)
    if not cap.isOpened():
        print_open_error(args.device)
        return 1

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if args.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

        print_actual_params(cap)

        window_title = (
            f"{args.device} | target {args.width}x{args.height}@{args.fps}fps"
        )
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

        consecutive_read_errors = 0
        while True:
            ok, frame, read_error = read_frame(cap)
            if not ok:
                consecutive_read_errors += 1
                print(
                    f"警告：读取或解码画面失败 "
                    f"({consecutive_read_errors}/{args.max_read_errors})：{read_error}",
                    file=sys.stderr,
                )

                if consecutive_read_errors >= args.max_read_errors:
                    print("错误：连续读取相机画面失败。", file=sys.stderr)
                    print_read_error_hint(args)
                    return 1

                key = cv2.waitKey(10) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            consecutive_read_errors = 0

            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save_frame(frame, save_dir, args.device)
            if key == ord("i"):
                print_actual_params(cap)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在退出。")
    except cv2.error as exc:
        print(f"OpenCV 错误：{exc}", file=sys.stderr)
        return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
