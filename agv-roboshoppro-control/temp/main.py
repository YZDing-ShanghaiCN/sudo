#!/usr/bin/env python3
# pyright: reportMissingImports=false

"""WSL-oriented AGV navigation runner with post-arrival V4L2 capture.

What this script adds on top of the original main/main.py flow:
1) Accept Windows map paths in map.yaml and convert them to WSL paths.
2) Keep local bind IP optional (default disabled for WSL2 NAT).
3) After navigation arrives, capture one JPEG per enabled V4L2 camera
   using temp/configS.json-style camera settings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import struct
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

PACK_FMT_MSGTYPE = "!BBHLH6s"
HEADER_LEN = 16

DEFAULT_NAV_MSG_TYPE = 3066
DEFAULT_NAV_PORT = 19206
DEFAULT_STATUS_PORT = 19301

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DEFAULT_MAP_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "map.yaml")
DEFAULT_CAPTURE_CONFIG_PATH = os.path.join(HERE, "configS.json")
DEFAULT_CAPTURE_OUTPUT_ROOT = os.path.join(HERE, "06")


class FrameTimeoutError(TimeoutError):
    """Raised when waiting too long for a V4L2 frame."""


def _alarm_timeout_handler(signum: int, frame: Any) -> None:
    del signum, frame
    raise FrameTimeoutError("timed out while waiting for first frame")


def is_windows_style_path(path: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", path))


def windows_path_to_wsl(path: str) -> str:
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if not match:
        return path
    drive = match.group(1).lower()
    rest = match.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def normalize_path_for_runtime(path: str) -> str:
    if os.name == "posix" and is_windows_style_path(path):
        return windows_path_to_wsl(path)
    return path


def parse_json_object(text: str, arg_name: str) -> Dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{arg_name} must be a JSON object")
    return obj


def load_map_file_path_from_config(map_config_path: str) -> str:
    normalized_cfg = normalize_path_for_runtime(map_config_path)
    with open(normalized_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid map config format: {normalized_cfg}")

    map_file_path = cfg.get("map_file_path")
    if not isinstance(map_file_path, str) or not map_file_path.strip():
        raise ValueError(f"map_file_path missing in map config: {normalized_cfg}")

    candidate = normalize_path_for_runtime(map_file_path)
    if os.path.exists(candidate):
        return candidate

    # Fallback: if map path in yaml is relative, resolve against config dir.
    rel_candidate = os.path.normpath(os.path.join(os.path.dirname(normalized_cfg), map_file_path))
    rel_candidate = normalize_path_for_runtime(rel_candidate)
    if os.path.exists(rel_candidate):
        return rel_candidate

    return candidate


def load_smap_topology(map_file_path: str) -> Dict[str, Any]:
    normalized = normalize_path_for_runtime(map_file_path)
    with open(normalized, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"invalid smap root object: {normalized}")

    nodes: Set[str] = set()
    edges: Set[Tuple[str, str]] = set()

    for point in obj.get("advancedPointList", []):
        if isinstance(point, dict):
            name = point.get("instanceName")
            if isinstance(name, str) and name:
                nodes.add(name)

    for curve in obj.get("advancedCurveList", []):
        if not isinstance(curve, dict):
            continue
        start = curve.get("startPos")
        end = curve.get("endPos")
        if isinstance(start, dict) and isinstance(end, dict):
            s_name = start.get("instanceName")
            e_name = end.get("instanceName")
            if isinstance(s_name, str) and s_name and isinstance(e_name, str) and e_name:
                nodes.add(s_name)
                nodes.add(e_name)
                edges.add((s_name, e_name))

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "map_file": normalized,
    }


def validate_path_nodes(nodes: List[str], topo: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    topo_nodes = topo.get("nodes")
    topo_edges = topo.get("edges")
    if not isinstance(topo_nodes, set) or not isinstance(topo_edges, set):
        return ["invalid topology object"]

    for idx in range(len(nodes) - 1):
        source_id = nodes[idx]
        target_id = nodes[idx + 1]

        if source_id not in topo_nodes:
            errors.append(f"step {idx + 1}: source node {source_id!r} not found")
        if target_id not in topo_nodes:
            errors.append(f"step {idx + 1}: target node {target_id!r} not found")
        if source_id in topo_nodes and target_id in topo_nodes and (source_id, target_id) not in topo_edges:
            errors.append(f"step {idx + 1}: no direct line from {source_id!r} to {target_id!r}")

    return errors


def build_move_task_list(nodes: List[str], task_id_prefix: str, linear_speed: float) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for idx in range(len(nodes) - 1):
        tasks.append(
            {
                "source_id": nodes[idx],
                "id": nodes[idx + 1],
                "task_id": f"{task_id_prefix}_{idx + 1}",
                "max_speed": linear_speed,
            }
        )
    return tasks


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def build_msgtype_frame(payload_bytes: bytes, req_id: int, msg_type: int) -> bytes:
    header = struct.pack(
        PACK_FMT_MSGTYPE,
        0x5A,
        0x01,
        req_id & 0xFFFF,
        len(payload_bytes),
        msg_type & 0xFFFF,
        b"\x00\x00\x00\x00\x00\x00",
    )
    return header + payload_bytes


def parse_response_frame(header: bytes, payload: bytes) -> Dict[str, Any]:
    payload_text = payload.decode("utf-8", errors="replace")
    payload_json = None
    try:
        payload_json = json.loads(payload_text)
    except json.JSONDecodeError:
        payload_json = None

    h = struct.unpack(PACK_FMT_MSGTYPE, header)
    frame = {
        "magic": h[0],
        "version": h[1],
        "req_id": h[2],
        "payload_len": h[3],
        "msg_type": h[4],
        "reserved_hex": h[5].hex(" "),
    }

    return {
        "frame": frame,
        "payload_text": payload_text,
        "payload_json": payload_json,
    }


def connect_robot(ip: str, port: int, local_ip: Optional[str], timeout: float) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    if local_ip:
        sock.bind((local_ip, 0))
    sock.connect((ip, port))
    return sock


def send_one(
    ip: str,
    port: int,
    local_ip: Optional[str],
    timeout: float,
    req_id: int,
    msg_type: int,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    payload_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    packet = build_msgtype_frame(payload_bytes, req_id=req_id, msg_type=msg_type)

    t0 = time.time()
    with connect_robot(ip, port, local_ip=local_ip, timeout=timeout) as sock:
        sock.sendall(packet)
        raw_header = recv_exact(sock, HEADER_LEN)
        if len(raw_header) < HEADER_LEN:
            return {
                "request": payload,
                "bytes_sent": len(packet),
                "roundtrip_ms": round((time.time() - t0) * 1000, 2),
                "response": {
                    "error": f"short header: {len(raw_header)} bytes",
                    "header_hex": raw_header.hex(" "),
                },
            }

        expected_len = struct.unpack(PACK_FMT_MSGTYPE, raw_header)[3]
        raw_payload = recv_exact(sock, expected_len)

    parsed = parse_response_frame(raw_header, raw_payload)
    return {
        "request": payload,
        "bytes_sent": len(packet),
        "roundtrip_ms": round((time.time() - t0) * 1000, 2),
        "response": parsed,
        "response_header_hex": raw_header.hex(" "),
    }


def read_one_status_sample(
    ip: str,
    port: int,
    local_ip: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    with connect_robot(ip, port, local_ip=local_ip, timeout=timeout) as sock:
        sock.settimeout(timeout)
        header = recv_exact(sock, HEADER_LEN)
        if len(header) < HEADER_LEN:
            return {
                "frame": None,
                "payload_json": None,
                "payload_text": f"short header: {len(header)}",
            }

        payload_len = struct.unpack(PACK_FMT_MSGTYPE, header)[3]
        payload = recv_exact(sock, payload_len)
        return parse_response_frame(header, payload)


def evaluate_navigation_ready(status_payload: Dict[str, Any]) -> Dict[str, Any]:
    blockers: List[str] = []
    warnings: List[str] = []

    if status_payload.get("blocked") is True:
        blockers.append("blocked=True")

    status_ret = status_payload.get("ret_code")
    if isinstance(status_ret, int) and status_ret != 0:
        blockers.append(f"status ret_code={status_ret}")

    if status_payload.get("is_stop") is True:
        warnings.append("is_stop=True")

    ready = len(blockers) == 0
    return {
        "ready": ready,
        "blockers": blockers,
        "warnings": warnings,
        "reasons": blockers + warnings,
        "observed": {
            "current_station": status_payload.get("current_station"),
            "is_stop": status_payload.get("is_stop"),
            "blocked": status_payload.get("blocked"),
            "running_status": status_payload.get("running_status"),
            "task_status": status_payload.get("task_status"),
            "task_type": status_payload.get("task_type"),
            "ret_code": status_payload.get("ret_code"),
        },
    }


def wait_until_arrival(
    ip: str,
    status_port: int,
    local_ip: Optional[str],
    timeout: float,
    target_station: str,
    wait_seconds: float,
    poll_interval: float,
    require_stop: bool,
    require_depart_before_arrival: bool,
) -> Dict[str, Any]:
    start = time.monotonic()
    deadline = start + max(0.1, wait_seconds)
    last_sample: Optional[Dict[str, Any]] = None
    first_on_target: Optional[bool] = None
    departure_required_runtime = require_depart_before_arrival
    departed_from_target = False

    while time.monotonic() < deadline:
        try:
            sample = read_one_status_sample(
                ip=ip,
                port=status_port,
                local_ip=local_ip,
                timeout=timeout,
            )
        except OSError as exc:
            return {
                "arrived": False,
                "error": str(exc),
                "elapsed_s": round(time.monotonic() - start, 2),
                "last_sample": last_sample,
            }

        last_sample = sample
        payload = sample.get("payload_json")
        if isinstance(payload, dict):
            current_station = payload.get("current_station")
            is_stop_now = payload.get("is_stop") is True
            on_station = isinstance(current_station, str) and current_station == target_station

            if first_on_target is None:
                first_on_target = on_station
                departure_required_runtime = require_depart_before_arrival and on_station

            if departure_required_runtime and not departed_from_target:
                left_target = False
                if isinstance(current_station, str):
                    left_target = current_station != target_station
                elif current_station is None:
                    left_target = True
                if left_target or (not is_stop_now):
                    departed_from_target = True

            if (
                on_station
                and (not departure_required_runtime or departed_from_target)
                and ((not require_stop) or is_stop_now)
            ):
                return {
                    "arrived": True,
                    "arrived_by": "current_station",
                    "required_departure": departure_required_runtime,
                    "departure_observed": departed_from_target,
                    "elapsed_s": round(time.monotonic() - start, 2),
                    "last_sample": sample,
                }

        time.sleep(max(0.1, poll_interval))

    return {
        "arrived": False,
        "error": f"timeout waiting for arrival at {target_station}",
        "required_departure": departure_required_runtime,
        "departure_observed": departed_from_target,
        "elapsed_s": round(time.monotonic() - start, 2),
        "last_sample": last_sample,
    }


def run_interactive_v4l2_capture(
    capture_config_path: str,
    output_root: str,
    frame_timeout_s: float,
    warmup_frames: int,
) -> Dict[str, Any]:
    cfg = _load_capture_config(capture_config_path)
    cameras = cfg.get("cameras")
    if not isinstance(cameras, list):
        raise ValueError("capture config must contain cameras as list")

    enabled: List[Dict[str, Any]] = []
    for cam in cameras:
        if isinstance(cam, dict) and cam.get("enable", True):
            enabled.append(cam)

    if not enabled:
        raise ValueError("no enabled cameras in capture config")

    cam_cfg = enabled[0]
    cam_id = str(cam_cfg.get("cam_id", "cam")).strip() or "cam"
    if len(enabled) > 1:
        print(f"[capture] 检测到 {len(enabled)} 个启用相机，仅使用第一个: {cam_id}")

    normalized_root = normalize_path_for_runtime(output_root)
    os.makedirs(normalized_root, exist_ok=True)

    print("[capture] 已到终点。输入 s 后回车拍照，输入 q 后回车取消。")
    while True:
        user_input = input("[capture] > ").strip().lower()
        if user_input == "q":
            return {
                "ok": False,
                "canceled": True,
                "reason": "user canceled capture",
            }
        if user_input != "s":
            print("[capture] 无效输入，请输入 s 或 q")
            continue

        frame_bytes = _capture_one_frame_bytes(
            cam_cfg=cam_cfg,
            frame_timeout_s=max(0.0, frame_timeout_s),
            warmup_frames=max(0, warmup_frames),
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = os.path.join(normalized_root, f"{ts}.jpg")
        with open(output_path, "wb") as f:
            f.write(frame_bytes)

        return {
            "ok": True,
            "mode": "manual_s",
            "cam_id": cam_id,
            "device": cam_cfg.get("device"),
            "image_path": output_path,
            "bytes": len(frame_bytes),
        }


def _load_capture_config(path: str) -> Dict[str, Any]:
    normalized = normalize_path_for_runtime(path)
    with open(normalized, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid capture config format: {normalized}")
    return cfg


def _apply_camera_controls(device: Any, cfg: Dict[str, Any]) -> None:
    if cfg.get("is_auto_exposure", True):
        try:
            device.controls.auto_exposure.value = 3
        except Exception:
            pass
    else:
        try:
            device.controls.auto_exposure.value = 1
        except Exception:
            pass

        manual_exposure = cfg.get("manual_exposure_value", 150)
        for attr in ("exposure_absolute", "exposure", "exposure_time_absolute"):
            try:
                if hasattr(device.controls, attr):
                    getattr(device.controls, attr).value = manual_exposure
                    break
            except Exception:
                continue

    try:
        backlight = cfg.get("backlight_compensation", 0)
        device.controls.backlight_compensation.value = backlight
    except Exception:
        pass


def _capture_one_frame_bytes(
    cam_cfg: Dict[str, Any],
    frame_timeout_s: float,
    warmup_frames: int,
) -> bytes:
    try:
        from linuxpy.video.device import BufferType, Device, PixelFormat
    except Exception as exc:
        raise RuntimeError(
            "linuxpy is required for V4L2 capture. Install in WSL env: pip install linuxpy"
        ) from exc

    device_path = str(cam_cfg.get("device", "")).strip()
    if not device_path:
        raise ValueError(f"camera {cam_cfg.get('cam_id', '<unknown>')} has empty device path")

    width = int(cam_cfg.get("width", 1920))
    height = int(cam_cfg.get("height", 1080))
    fps = int(cam_cfg.get("fps", 30))

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
        except Exception:
            device.set_format(BufferType.VIDEO_CAPTURE, width, height)

        if cam_cfg.get("backlight_compensation", 0) != 2:
            try:
                device.set_fps(BufferType.VIDEO_CAPTURE, fps)
            except Exception:
                pass

        _apply_camera_controls(device, cam_cfg)

        warmup_left = max(0, warmup_frames)

        if frame_timeout_s > 0 and hasattr(signal, "setitimer"):
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _alarm_timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, frame_timeout_s)
            try:
                for frame_data in device:
                    if warmup_left > 0:
                        warmup_left -= 1
                        continue
                    return bytes(frame_data)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            for frame_data in device:
                if warmup_left > 0:
                    warmup_left -= 1
                    continue
                return bytes(frame_data)

        raise RuntimeError("no frame received from device")
    finally:
        try:
            device.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WSL AGV navigation with post-arrival V4L2 capture")

    p.add_argument("--robot-ip", default="172.16.52.85", help="Robot IP")
    p.add_argument(
        "--local-ip",
        default="",
        help="Optional local source IP to bind; keep empty in WSL2 unless needed",
    )
    p.add_argument("--port", type=int, default=DEFAULT_NAV_PORT, help="Navigation service port")
    p.add_argument("--status-port", type=int, default=DEFAULT_STATUS_PORT, help="Status push port")
    p.add_argument("--timeout", type=float, default=3.0, help="Socket timeout in seconds")

    p.add_argument("--req-id", type=int, default=1, help="Request id")
    p.add_argument("--msg-type", type=int, default=DEFAULT_NAV_MSG_TYPE, help="Navigation msg type")

    p.add_argument("--path", required=True, help="Comma-separated nodes, e.g. LM1,LM2,LM3")
    p.add_argument("--task-id-prefix", default=None, help="Task id prefix")
    p.add_argument(
        "--linear-speed",
        type=float,
        required=True,
        help="Required linear speed for each navigation step (no default)",
    )

    p.add_argument("--map-config", default=DEFAULT_MAP_CONFIG_PATH, help="Path to map yaml")
    p.add_argument("--map-file", default=None, help="Direct smap path override")
    p.add_argument("--skip-map-check", action="store_true", help="Skip map topology validation")

    p.add_argument("--precheck-only", action="store_true", help="Only read one status sample and exit")
    p.add_argument("--require-ready", action="store_true", help="Block sending when precheck is not ready")

    p.add_argument("--wait-arrival-seconds", type=float, default=180.0, help="Wait timeout for target station")
    p.add_argument("--arrival-poll-interval", type=float, default=0.5, help="Status polling interval")
    p.add_argument(
        "--arrival-require-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require is_stop=True at target station",
    )

    p.add_argument(
        "--capture-after-arrival",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After arrival, wait for 's' input then save one timestamped jpg",
    )
    p.add_argument("--capture-config", default=DEFAULT_CAPTURE_CONFIG_PATH, help="Capture config JSON path")
    p.add_argument("--capture-output-root", default=DEFAULT_CAPTURE_OUTPUT_ROOT, help="Capture output root dir")
    p.add_argument("--capture-frame-timeout", type=float, default=5.0, help="Frame wait timeout seconds")
    p.add_argument("--capture-warmup-frames", type=int, default=0, help="Drop N initial frames")

    p.add_argument("--dry-run", action="store_true", help="Only print plan")
    p.add_argument("--json", action="store_true", help="Reserved for compatibility")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    nodes = [part.strip() for part in args.path.split(",") if part.strip()]
    if len(nodes) < 1:
        print("ERROR: --path must include at least 1 node, e.g. LM1 or LM1,LM2")
        return 2
    if args.linear_speed <= 0:
        print("ERROR: --linear-speed must be > 0")
        return 2

    stay_put_mode = len(nodes) == 1 or (len(nodes) == 2 and nodes[0] == nodes[1])

    task_id_prefix = args.task_id_prefix or f"wsl_nav_{int(time.time() * 1000)}"
    move_task_list: List[Dict[str, Any]] = []
    if not stay_put_mode:
        move_task_list = build_move_task_list(nodes, task_id_prefix, args.linear_speed)

    map_check: Dict[str, Any] = {"enabled": not args.skip_map_check}
    resolved_map_file_path: Optional[str] = None

    if not args.skip_map_check:
        try:
            map_file_path = args.map_file if args.map_file else load_map_file_path_from_config(args.map_config)
            resolved_map_file_path = normalize_path_for_runtime(map_file_path)
            topo = load_smap_topology(resolved_map_file_path)
            errors: List[str] = []
            topo_nodes = topo.get("nodes")
            if stay_put_mode:
                target_station = nodes[-1]
                if not isinstance(topo_nodes, set) or target_station not in topo_nodes:
                    errors.append(f"target node {target_station!r} not found in map topology")
            else:
                errors = validate_path_nodes(nodes, topo)
            map_check.update(
                {
                    "ok": len(errors) == 0,
                    "errors": errors,
                    "map_file": topo.get("map_file"),
                    "node_count": topo.get("node_count"),
                    "edge_count": topo.get("edge_count"),
                }
            )
            if errors:
                print("ERROR: map topology validation failed:")
                for err in errors:
                    print(f"  - {err}")
                return 2
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"ERROR: map topology load failed: {exc}")
            print("HINT: in WSL, pass --map-file /mnt/c/... or fix map.yaml path")
            print("HINT: temporary bypass is --skip-map-check")
            return 2

    local_ip = args.local_ip.strip() if isinstance(args.local_ip, str) else ""
    local_ip_or_none = local_ip if local_ip else None

    request_payload = {"move_task_list": move_task_list} if move_task_list else {}
    plan = {
        "target": {
            "robot_ip": args.robot_ip,
            "port": args.port,
            "local_ip": local_ip_or_none,
            "status_port": args.status_port,
        },
        "msg": {
            "req_id": args.req_id,
            "msg_type": args.msg_type,
            "request_payload": request_payload,
        },
        "path_nodes": nodes,
        "mode": "stay_put" if stay_put_mode else "navigate",
        "linear_speed": args.linear_speed,
        "map_check": map_check,
        "resolved_map_file": resolved_map_file_path,
    }

    precheck: Dict[str, Any] = {}
    if args.precheck_only or args.require_ready:
        try:
            sample = read_one_status_sample(
                ip=args.robot_ip,
                port=args.status_port,
                local_ip=local_ip_or_none,
                timeout=args.timeout,
            )
            payload = sample.get("payload_json")
            if isinstance(payload, dict):
                precheck = evaluate_navigation_ready(payload)
                precheck["status_sample"] = sample
            else:
                precheck = {
                    "ready": False,
                    "reasons": ["status payload_json unavailable"],
                    "status_sample": sample,
                }
        except OSError as exc:
            precheck = {
                "ready": False,
                "reasons": ["status connection failed"],
                "error": str(exc),
            }

    if args.precheck_only:
        print(json.dumps({"plan": plan, "precheck": precheck}, ensure_ascii=False, indent=2))
        return 0 if precheck.get("ready") else 5

    if args.require_ready and not precheck.get("ready"):
        print(json.dumps({"plan": plan, "precheck": precheck}, ensure_ascii=False, indent=2))
        print("HINT: --require-ready enabled and precheck is not ready")
        return 5

    if args.dry_run:
        print(json.dumps({"plan": plan, "precheck": precheck}, ensure_ascii=False, indent=2))
        return 0

    send_result: Dict[str, Any]
    if stay_put_mode:
        send_result = {
            "skipped": True,
            "reason": "stay_put mode (single node or start==target)",
        }
    else:
        try:
            send_result = send_one(
                ip=args.robot_ip,
                port=args.port,
                local_ip=local_ip_or_none,
                timeout=args.timeout,
                req_id=args.req_id,
                msg_type=args.msg_type,
                payload=request_payload,
            )
        except OSError as exc:
            print(f"ERROR: socket failed: {exc}")
            return 1

    output: Dict[str, Any] = {
        "plan": plan,
        "send_result": send_result,
    }
    if precheck:
        output["precheck"] = precheck

    ret_code: Optional[int] = 0 if stay_put_mode else None
    err_msg: Optional[str] = None
    if not stay_put_mode:
        payload_json = send_result.get("response", {}).get("payload_json")
        if isinstance(payload_json, dict):
            rc = payload_json.get("ret_code")
            err = payload_json.get("err_msg")
            if isinstance(rc, int):
                ret_code = rc
            if isinstance(err, str):
                err_msg = err

    arrival_ok = True
    if ret_code == 0:
        arrival = wait_until_arrival(
            ip=args.robot_ip,
            status_port=args.status_port,
            local_ip=local_ip_or_none,
            timeout=args.timeout,
            target_station=nodes[-1],
            wait_seconds=max(1.0, args.wait_arrival_seconds),
            poll_interval=max(0.1, args.arrival_poll_interval),
            require_stop=bool(args.arrival_require_stop),
            require_depart_before_arrival=not stay_put_mode,
        )
        output["arrival_wait"] = arrival
        arrival_ok = bool(arrival.get("arrived"))

        if arrival.get("arrived") and args.capture_after_arrival:
            try:
                capture_result = run_interactive_v4l2_capture(
                    capture_config_path=args.capture_config,
                    output_root=args.capture_output_root,
                    frame_timeout_s=args.capture_frame_timeout,
                    warmup_frames=args.capture_warmup_frames,
                )
            except (OSError, ValueError, RuntimeError, FrameTimeoutError) as exc:
                capture_result = {
                    "ok": False,
                    "error": str(exc),
                }
            output["capture"] = capture_result
        elif args.capture_after_arrival:
            output["capture"] = {
                "ok": False,
                "error": "skip capture because arrival was not confirmed",
            }

    print(json.dumps(output, ensure_ascii=False, indent=2))

    if err_msg:
        print(f"INFO: robot err_msg={err_msg}")

    if stay_put_mode:
        return 0 if arrival_ok else 6

    if ret_code is None:
        return 4
    if ret_code != 0:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
