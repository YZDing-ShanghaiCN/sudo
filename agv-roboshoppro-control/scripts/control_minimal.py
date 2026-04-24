import argparse
import json
import socket
import struct
import sys
import time
from typing import Any, Dict, Optional

PACK_FMT_MSGTYPE = "!BBHLH6s"
HEADER_LEN = 16
DEFAULT_MOVE_MSG_TYPE = 2010  # 0x07DA
DEFAULT_STOP_MSG_TYPE = 2000  # 0x07D0
DEFAULT_TURN_TASK_MSG_TYPE = 3056  # 0x0BF0
DEFAULT_VELOCITY_PORT = 19205
DEFAULT_TURN_TASK_PORT = 19206


def parse_json_object(text: str, arg_name: str) -> Dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{arg_name} must be a JSON object")
    return obj


def make_motion_payload(vx: float, vy: float, w: float, extra: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "vx": vx,
        "vy": vy,
        "w": w,
    }
    payload.update(extra)
    return payload


def make_turn_task_payload(angle: float, vw: float, mode: int, extra: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "angle": angle,
        "vw": vw,
        "mode": mode,
    }
    payload.update(extra)
    return payload


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


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


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


def send_one(
    sock: socket.socket,
    payload: Dict[str, Any],
    timeout: float,
    req_id: int,
    msg_type: int,
) -> Dict[str, Any]:
    payload_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    packet = build_msgtype_frame(payload_bytes, req_id=req_id, msg_type=msg_type)

    t0 = time.time()
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

    sock.settimeout(timeout)
    raw_payload = recv_exact(sock, expected_len)
    parsed = parse_response_frame(raw_header, raw_payload)

    return {
        "request": payload,
        "bytes_sent": len(packet),
        "roundtrip_ms": round((time.time() - t0) * 1000, 2),
        "response": parsed,
        "response_header_hex": raw_header.hex(" "),
    }


def connect_robot(ip: str, port: int, local_ip: Optional[str], timeout: float) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    if local_ip:
        sock.bind((local_ip, 0))
    sock.connect((ip, port))
    return sock


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal AGV motion test (forward then stop)")
    p.add_argument("--robot-ip", default="172.16.52.85", help="Robot IP")
    p.add_argument("--local-ip", default="172.16.52.47", help="Local source IP to bind; use empty string to disable")
    p.add_argument("--port", type=int, default=None, help="Legacy: use one port for both move and stop")
    p.add_argument("--move-port", type=int, default=None, help="TCP port for move command service")
    p.add_argument("--stop-port", type=int, default=None, help="TCP port for stop command service")
    p.add_argument("--timeout", type=float, default=3.0, help="Socket timeout in seconds")

    p.add_argument("--req-id", type=int, default=1, help="Initial request id")
    p.add_argument("--move-msg-type", type=int, default=None, help="Request msg type; default is 2010 for velocity mode, 3056 for turn-task mode")
    p.add_argument("--stop-msg-type", type=int, default=DEFAULT_STOP_MSG_TYPE, help="Open-loop stop request msg type")
    p.add_argument(
        "--payload-mode",
        choices=["velocity", "turn-task"],
        default="velocity",
        help="Payload format: velocity(vx/vy/w) or turn-task(angle/vw/mode)",
    )

    p.add_argument("--vx", type=float, default=0.05, help="Forward speed")
    p.add_argument("--vy", type=float, default=0.0, help="Lateral speed")
    p.add_argument("--w", type=float, default=0.0, help="Angular speed")
    p.add_argument("--angle", type=float, default=0.35, help="Turn-task angle in rad for msg_type 3056")
    p.add_argument("--vw", type=float, default=0.35, help="Turn-task angular speed in rad/s for msg_type 3056")
    p.add_argument("--nav-mode", type=int, default=0, help="Turn-task mode field (usually 0 odometry, 1 localization)")
    p.add_argument("--move-seconds", type=float, default=1.0, help="Forward motion duration before stop")
    p.add_argument("--stream-interval", type=float, default=0.1, help="Interval seconds for repeated move commands")
    p.add_argument("--single-shot", action="store_true", help="Send move command once instead of streaming")
    p.add_argument("--extra-json", default="{}", help="Extra JSON merged into motion body")

    p.add_argument("--no-stop", action="store_true", help="Do not send stop command")
    p.add_argument("--dry-run", action="store_true", help="Only print plan and payloads")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        extra = parse_json_object(args.extra_json, "--extra-json")
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    local_ip = args.local_ip.strip() if args.local_ip is not None else ""
    if local_ip == "":
        local_ip = None

    default_move_port = DEFAULT_VELOCITY_PORT
    default_stop_port = DEFAULT_VELOCITY_PORT

    if args.payload_mode == "turn-task":
        move_payload = make_turn_task_payload(angle=args.angle, vw=args.vw, mode=args.nav_mode, extra=extra)
        move_msg_type = args.move_msg_type if args.move_msg_type is not None else DEFAULT_TURN_TASK_MSG_TYPE
        default_move_port = DEFAULT_TURN_TASK_PORT
        default_stop_port = DEFAULT_VELOCITY_PORT
    else:
        move_payload = make_motion_payload(vx=args.vx, vy=args.vy, w=args.w, extra=extra)
        move_msg_type = args.move_msg_type if args.move_msg_type is not None else DEFAULT_MOVE_MSG_TYPE

    if args.port is not None:
        move_port = args.port
        stop_port = args.port
    else:
        move_port = args.move_port if args.move_port is not None else default_move_port
        stop_port = args.stop_port if args.stop_port is not None else default_stop_port

    stop_payload: Dict[str, Any] = {}

    plan = {
        "target": {
            "robot_ip": args.robot_ip,
            "move_port": move_port,
            "stop_port": stop_port,
            "local_ip": local_ip,
        },
        "motion_msg_type": move_msg_type,
        "stop_msg_type": args.stop_msg_type,
        "payload_mode": args.payload_mode,
        "move_seconds": max(0.0, args.move_seconds),
        "stream_interval": max(0.01, args.stream_interval),
        "single_shot": args.single_shot,
        "no_stop": args.no_stop,
        "move_payload": move_payload,
        "stop_payload": stop_payload,
    }

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    results = []
    req_id = args.req_id
    move_seconds = max(0.0, args.move_seconds)
    stream_interval = max(0.01, args.stream_interval)

    try:
        with connect_robot(args.robot_ip, move_port, local_ip=local_ip, timeout=args.timeout) as sock:
            move_send_count = 0
            move_nonzero_ret_count = 0
            first_move_result = None
            last_move_result = None

            def send_move_once() -> None:
                nonlocal req_id, move_send_count, move_nonzero_ret_count, first_move_result, last_move_result
                move_result = send_one(
                    sock,
                    payload=move_payload,
                    timeout=args.timeout,
                    req_id=req_id,
                    msg_type=move_msg_type,
                )
                req_id += 1
                move_send_count += 1

                payload_json = move_result.get("response", {}).get("payload_json")
                if isinstance(payload_json, dict):
                    ret_code = payload_json.get("ret_code")
                    if isinstance(ret_code, int) and ret_code != 0:
                        move_nonzero_ret_count += 1

                if first_move_result is None:
                    first_move_result = move_result
                last_move_result = move_result

            if args.single_shot or move_seconds <= 0.0:
                send_move_once()
            else:
                end_at = time.monotonic() + move_seconds
                next_at = time.monotonic()
                while True:
                    now = time.monotonic()
                    if now >= end_at and move_send_count > 0:
                        break
                    if now < next_at:
                        time.sleep(min(0.01, next_at - now))
                        continue
                    send_move_once()
                    next_at += stream_interval
                    if next_at < time.monotonic() - stream_interval:
                        next_at = time.monotonic()

            if first_move_result is not None:
                results.append({"step": "move_first", **first_move_result})
            if last_move_result is not None and last_move_result is not first_move_result:
                results.append({"step": "move_last", **last_move_result})
            results.append(
                {
                    "step": "move_stream_stats",
                    "move_send_count": move_send_count,
                    "move_nonzero_ret_count": move_nonzero_ret_count,
                    "stream_interval": stream_interval,
                }
            )

        if not args.no_stop:
            if args.single_shot and move_seconds > 0.0:
                time.sleep(move_seconds)
            with connect_robot(args.robot_ip, stop_port, local_ip=local_ip, timeout=args.timeout) as stop_sock:
                stop_result = send_one(
                    stop_sock,
                    payload=stop_payload,
                    timeout=args.timeout,
                    req_id=req_id,
                    msg_type=args.stop_msg_type,
                )
                results.append({"step": "stop", **stop_result})
    except OSError as exc:
        print(f"ERROR: socket failed: {exc}")
        return 1

    output = {
        "plan": plan,
        "results": results,
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))

    nonzero_ret = False
    saw_api_type_error = False
    for item in results:
        payload_json = item.get("response", {}).get("payload_json")
        if isinstance(payload_json, dict):
            ret_code = payload_json.get("ret_code")
            if isinstance(ret_code, int) and ret_code != 0:
                nonzero_ret = True
                err_msg = str(payload_json.get("err_msg", ""))
                if "api type" in err_msg.lower():
                    saw_api_type_error = True

    if saw_api_type_error:
        if args.payload_mode == "turn-task" and move_port == DEFAULT_VELOCITY_PORT:
            print("HINT: turn-task(3056) 建议使用 --move-port 19206")
        if args.payload_mode == "velocity" and move_port == DEFAULT_TURN_TASK_PORT:
            print("HINT: velocity(2010) 建议使用 --move-port 19205")
        if move_port == 19204:
            print("HINT: 当前 move_port=19204 可能不是控制口，请改用 19205/19206")

    return 3 if nonzero_ret else 0


if __name__ == "__main__":
    sys.exit(main())
