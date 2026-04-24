import argparse
import json
import os
import socket
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

PACK_FMT_MSGTYPE = "!BBHLH6s"
HEADER_LEN = 16

DEFAULT_NAV_MSG_TYPE = 3066  # 0x0BFA robot_task_gotargetlist_req
DEFAULT_NAV_PORT = 19206
DEFAULT_STATUS_PORT = 19301
DEFAULT_MAP_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "map.yaml")


def parse_json_object(text: str, arg_name: str) -> Dict[str, Any]:
	try:
		obj = json.loads(text)
	except json.JSONDecodeError as exc:
		raise ValueError(f"{arg_name} is not valid JSON: {exc}") from exc
	if not isinstance(obj, dict):
		raise ValueError(f"{arg_name} must be a JSON object")
	return obj


def parse_json_array(text: str, arg_name: str) -> List[Any]:
	try:
		arr = json.loads(text)
	except json.JSONDecodeError as exc:
		raise ValueError(f"{arg_name} is not valid JSON: {exc}") from exc
	if not isinstance(arr, list):
		raise ValueError(f"{arg_name} must be a JSON array")
	return arr


def load_json_object_from_file(path: str, arg_name: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		text = f.read()
	return parse_json_object(text, arg_name)


def load_map_file_path_from_config(map_config_path: str) -> str:
	with open(map_config_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)
	if not isinstance(cfg, dict):
		raise ValueError(f"invalid map config format: {map_config_path}")
	map_file_path = cfg.get("map_file_path")
	if not isinstance(map_file_path, str) or not map_file_path.strip():
		raise ValueError(f"map_file_path missing in map config: {map_config_path}")
	return map_file_path


def load_smap_topology(map_file_path: str) -> Dict[str, Any]:
	with open(map_file_path, "r", encoding="utf-8") as f:
		obj = json.load(f)

	if not isinstance(obj, dict):
		raise ValueError(f"invalid smap root object: {map_file_path}")

	nodes: Set[str] = set()
	edges: Set[Tuple[str, str]] = set()

	for p in obj.get("advancedPointList", []):
		if isinstance(p, dict):
			name = p.get("instanceName")
			if isinstance(name, str) and name:
				nodes.add(name)

	for c in obj.get("advancedCurveList", []):
		if not isinstance(c, dict):
			continue
		start = c.get("startPos")
		end = c.get("endPos")
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
	}


def validate_move_task_list_against_topology(tasks: List[Dict[str, Any]], topo: Dict[str, Any]) -> List[str]:
	errors: List[str] = []
	nodes = topo.get("nodes")
	edges = topo.get("edges")
	if not isinstance(nodes, set) or not isinstance(edges, set):
		return ["invalid topology object"]

	for idx, item in enumerate(tasks, start=1):
		source_id = item.get("source_id")
		target_id = item.get("id")
		if not isinstance(source_id, str) or not isinstance(target_id, str):
			continue

		if source_id == "SELF_POSITION" and target_id == "SELF_POSITION":
			continue
		if source_id == "SELF_POSITION" or target_id == "SELF_POSITION":
			continue

		if source_id not in nodes:
			errors.append(f"step {idx}: source_id {source_id!r} not found in map topology")
		if target_id not in nodes:
			errors.append(f"step {idx}: id {target_id!r} not found in map topology")
		if source_id in nodes and target_id in nodes and (source_id, target_id) not in edges:
			errors.append(f"step {idx}: no direct line from {source_id!r} to {target_id!r} in map topology")

	for idx in range(len(tasks) - 1):
		cur_target = tasks[idx].get("id")
		next_source = tasks[idx + 1].get("source_id")
		if isinstance(cur_target, str) and isinstance(next_source, str):
			if cur_target != next_source:
				errors.append(
					f"step {idx + 2}: source_id {next_source!r} must equal previous step id {cur_target!r}"
				)

	return errors


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


def build_move_task_list_from_path(
	nodes: List[str],
	task_id_prefix: str,
	operation_steps: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
	tasks: List[Dict[str, Any]] = []
	for idx in range(len(nodes) - 1):
		source_id = nodes[idx]
		target_id = nodes[idx + 1]
		task: Dict[str, Any] = {
			"source_id": source_id,
			"id": target_id,
			"task_id": f"{task_id_prefix}_{idx + 1}",
		}
		if (idx + 1) in operation_steps:
			task.update(operation_steps[idx + 1])
		tasks.append(task)
	return tasks


def load_move_task_list_from_file(path: str) -> List[Dict[str, Any]]:
	with open(path, "r", encoding="utf-8") as f:
		text = f.read()
	obj = parse_json_object(text, "--request-file")
	move_task_list = obj.get("move_task_list")
	if not isinstance(move_task_list, list):
		raise ValueError("--request-file must include move_task_list as JSON array")
	for item in move_task_list:
		if not isinstance(item, dict):
			raise ValueError("all items in move_task_list must be JSON objects")
	return move_task_list


def validate_move_task_list(tasks: List[Dict[str, Any]]) -> List[str]:
	errors: List[str] = []
	if not tasks:
		errors.append("move_task_list must not be empty")
		return errors

	task_ids: List[str] = []
	for idx, item in enumerate(tasks, start=1):
		source_id = item.get("source_id")
		target_id = item.get("id")
		task_id = item.get("task_id")

		if not isinstance(source_id, str) or not source_id:
			errors.append(f"step {idx}: source_id is required and must be non-empty string")
		if not isinstance(target_id, str) or not target_id:
			errors.append(f"step {idx}: id is required and must be non-empty string")
		if not isinstance(task_id, str) or not task_id:
			errors.append(f"step {idx}: task_id is required and must be non-empty string")

		if isinstance(source_id, str) and isinstance(target_id, str):
			if target_id != "SELF_POSITION" and source_id == "SELF_POSITION":
				errors.append(
					f"step {idx}: invalid SELF_POSITION usage (id != SELF_POSITION while source_id == SELF_POSITION)"
				)

		if isinstance(task_id, str) and task_id:
			task_ids.append(task_id)

	if len(task_ids) != len(set(task_ids)):
		errors.append("task_id must be unique within move_task_list")

	return errors


def listen_status_push(
	ip: str,
	port: int,
	local_ip: Optional[str],
	timeout: float,
	seconds: float,
	max_samples: int,
	max_payload_chars: int,
	compact: bool,
) -> List[Dict[str, Any]]:
	samples: List[Dict[str, Any]] = []
	end_at = time.monotonic() + max(0.0, seconds)

	with connect_robot(ip, port, local_ip=local_ip, timeout=timeout) as sock:
		sock.settimeout(0.5)
		while time.monotonic() < end_at:
			try:
				header = recv_exact(sock, HEADER_LEN)
			except socket.timeout:
				continue
			if len(header) < HEADER_LEN:
				break

			payload_len = struct.unpack(PACK_FMT_MSGTYPE, header)[3]
			payload = recv_exact(sock, payload_len)
			parsed = parse_response_frame(header, payload)

			if compact:
				item: Dict[str, Any] = {"frame": parsed.get("frame")}
				payload_json = parsed.get("payload_json")
				if isinstance(payload_json, dict):
					keys_of_interest = [
						"running_status",
						"dispatch_mode",
						"is_stop",
						"blocked",
						"slowed",
						"reloc_status",
						"task_id",
						"task_status",
						"task_type",
						"current_station",
						"id",
						"source_id",
						"target_id",
						"target_label",
						"ret_code",
						"err_msg",
						"x",
						"y",
						"angle",
					]
					picked = {k: payload_json[k] for k in keys_of_interest if k in payload_json}
					item["payload_json"] = picked if picked else {"keys": list(payload_json.keys())[:20]}
				else:
					text_preview = parsed.get("payload_text", "")
					if not isinstance(text_preview, str):
						text_preview = str(text_preview)
					item["payload_text_preview"] = text_preview[:max_payload_chars]
				samples.append(item)
			else:
				samples.append(parsed)

			if len(samples) >= max_samples:
				break

	return samples


def read_one_status_sample(
	ip: str,
	port: int,
	local_ip: Optional[str],
	timeout: float,
	max_payload_chars: int,
	compact: bool,
) -> Dict[str, Any]:
	with connect_robot(ip, port, local_ip=local_ip, timeout=timeout) as sock:
		sock.settimeout(timeout)
		header = recv_exact(sock, HEADER_LEN)
		if len(header) < HEADER_LEN:
			return {
				"frame": None,
				"payload_json": None,
				"payload_text_preview": f"short header: {len(header)}",
			}

		payload_len = struct.unpack(PACK_FMT_MSGTYPE, header)[3]
		payload = recv_exact(sock, payload_len)
		parsed = parse_response_frame(header, payload)

	if not compact:
		return parsed

	item: Dict[str, Any] = {"frame": parsed.get("frame")}
	payload_json = parsed.get("payload_json")
	if isinstance(payload_json, dict):
		keys_of_interest = [
			"running_status",
			"dispatch_mode",
			"is_stop",
			"blocked",
			"slowed",
			"reloc_status",
			"task_id",
			"task_status",
			"task_type",
			"current_station",
			"id",
			"source_id",
			"target_id",
			"target_label",
			"ret_code",
			"err_msg",
			"x",
			"y",
			"angle",
		]
		picked = {k: payload_json[k] for k in keys_of_interest if k in payload_json}
		item["payload_json"] = picked if picked else {"keys": list(payload_json.keys())[:20]}
	else:
		text_preview = parsed.get("payload_text", "")
		if not isinstance(text_preview, str):
			text_preview = str(text_preview)
		item["payload_text_preview"] = text_preview[:max_payload_chars]

	return item


def evaluate_navigation_ready(status_payload: Dict[str, Any]) -> Dict[str, Any]:
	# blocked/status error are hard blockers; is_stop is warning-only because task creation can still succeed.
	blockers: List[str] = []
	warnings: List[str] = []

	if status_payload.get("blocked") is True:
		blockers.append("blocked=True")

	status_ret = status_payload.get("ret_code")
	if isinstance(status_ret, int) and status_ret != 0:
		blockers.append(f"status ret_code={status_ret}")

	if status_payload.get("is_stop") is True:
		warnings.append("is_stop=True")

	if status_payload.get("slowed") is True:
		warnings.append("slowed=True")

	ready = len(blockers) == 0
	return {
		"ready": ready,
		"reasons": blockers + warnings,
		"blockers": blockers,
		"warnings": warnings,
		"observed": {
			"is_stop": status_payload.get("is_stop"),
			"blocked": status_payload.get("blocked"),
			"slowed": status_payload.get("slowed"),
			"running_status": status_payload.get("running_status"),
			"task_status": status_payload.get("task_status"),
			"task_type": status_payload.get("task_type"),
			"dispatch_mode": status_payload.get("dispatch_mode"),
			"ret_code": status_payload.get("ret_code"),
		},
	}


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="AGV fixed-route navigation sender (msg_type 3066)")
	p.add_argument("--robot-ip", default="172.16.52.85", help="Robot IP")
	p.add_argument("--local-ip", default="172.16.52.47", help="Local source IP to bind; use empty string to disable")
	p.add_argument("--port", type=int, default=DEFAULT_NAV_PORT, help="Navigation service port (default 19206)")
	p.add_argument("--status-port", type=int, default=DEFAULT_STATUS_PORT, help="Status push port for optional listen")
	p.add_argument("--timeout", type=float, default=3.0, help="Socket timeout in seconds")

	p.add_argument("--req-id", type=int, default=1, help="Request id")
	p.add_argument("--msg-type", type=int, default=DEFAULT_NAV_MSG_TYPE, help="Navigation request msg type (default 3066)")

	p.add_argument("--path", default=None, help="Comma-separated nodes, e.g. LM1,LM2,AP1")
	p.add_argument("--request-file", default=None, help="JSON file with move_task_list")
	p.add_argument("--task-id-prefix", default=None, help="Task id prefix when building from --path")
	p.add_argument("--list-map-nodes", action="store_true", help="List available station/node names from map and exit")
	p.add_argument("--skip-map-check", action="store_true", help="Skip loading map topology and connectivity validation")
	p.add_argument("--map-config", default=DEFAULT_MAP_CONFIG_PATH, help="Path to map config yaml containing map_file_path")
	p.add_argument("--map-file", default=None, help="Direct smap path override (JSON .smap)")

	p.add_argument(
		"--operation-steps-json",
		default="{}",
		help=(
			"JSON object mapping 1-based step index to extra fields. "
			"Example: {\"2\":{\"operation\":\"JackHeight\",\"jack_height\":0.2}}"
		),
	)
	p.add_argument("--operation-steps-file", default=None, help="JSON file for operation steps mapping, avoids shell escaping issues")

	p.add_argument("--listen-status-seconds", type=float, default=0.0, help="After send, listen status push for N seconds")
	p.add_argument("--precheck-only", action="store_true", help="Only check robot ready state from status port; do not send 3066")
	p.add_argument("--require-ready", action="store_true", help="Abort sending 3066 when precheck says not ready")
	p.add_argument("--wait-ready-seconds", type=float, default=0.0, help="Wait up to N seconds for ready state before sending")
	p.add_argument("--wait-ready-interval", type=float, default=0.5, help="Polling interval for wait-ready check")
	p.add_argument("--status-max-samples", type=int, default=20, help="Maximum status samples to keep")
	p.add_argument("--status-max-payload-chars", type=int, default=240, help="Preview length when compacting status payload")
	p.add_argument("--full-status-samples", action="store_true", help="Keep full raw status payloads (can be very large)")
	p.add_argument("--dry-run", action="store_true", help="Only print request plan")
	p.add_argument("--json", action="store_true", help="Print machine-readable JSON")
	return p.parse_args()


def main() -> int:
	args = parse_args()

	if args.list_map_nodes:
		try:
			map_file_path = args.map_file if args.map_file else load_map_file_path_from_config(args.map_config)
			topo = load_smap_topology(map_file_path)
		except (OSError, ValueError, json.JSONDecodeError) as exc:
			print(f"ERROR: map topology load failed: {exc}")
			return 2

		nodes = sorted(topo.get("nodes", []))
		edges = sorted(topo.get("edges", []))
		print(json.dumps({
			"map_file": map_file_path,
			"node_count": len(nodes),
			"edge_count": len(edges),
			"nodes": nodes,
			"edges_preview": [list(e) for e in edges[:50]],
		}, ensure_ascii=False, indent=2))
		return 0

	if bool(args.path) == bool(args.request_file):
		print("ERROR: exactly one of --path or --request-file must be provided")
		return 2

	op_steps_obj: Dict[str, Any] = {}

	if args.operation_steps_file:
		try:
			file_obj = load_json_object_from_file(args.operation_steps_file, "--operation-steps-file")
		except (OSError, ValueError) as exc:
			print(f"ERROR: {exc}")
			return 2
		op_steps_obj.update(file_obj)

	json_text = args.operation_steps_json.strip() if isinstance(args.operation_steps_json, str) else ""
	if json_text and json_text != "{}":
		try:
			json_obj = parse_json_object(json_text, "--operation-steps-json")
		except ValueError as exc:
			print(f"ERROR: {exc}")
			print("HINT: Windows PowerShell 示例:")
			print("  --operation-steps-json '{\"2\":{\"operation\":\"JackHeight\",\"jack_height\":0.2}}'")
			print("HINT: 更稳的方式是用 --operation-steps-file <json_file>")
			return 2
		op_steps_obj.update(json_obj)

	operation_steps: Dict[int, Dict[str, Any]] = {}
	for k, v in op_steps_obj.items():
		try:
			step_idx = int(k)
		except (TypeError, ValueError):
			print(f"ERROR: operation step key {k!r} is not an integer index")
			return 2
		if step_idx <= 0:
			print(f"ERROR: operation step index must be >= 1, got {step_idx}")
			return 2
		if not isinstance(v, dict):
			print(f"ERROR: operation step {step_idx} must map to JSON object")
			return 2
		operation_steps[step_idx] = v

	local_ip = args.local_ip.strip() if args.local_ip is not None else ""
	if local_ip == "":
		local_ip = None

	if args.path:
		nodes = [part.strip() for part in args.path.split(",") if part.strip()]
		if len(nodes) < 2:
			print("ERROR: --path must include at least 2 nodes, e.g. LM1,LM2")
			return 2
		task_id_prefix = args.task_id_prefix or f"nav_{int(time.time() * 1000)}"
		move_task_list = build_move_task_list_from_path(nodes, task_id_prefix, operation_steps)
	else:
		try:
			move_task_list = load_move_task_list_from_file(args.request_file)
		except (OSError, ValueError) as exc:
			print(f"ERROR: {exc}")
			return 2

	validation_errors = validate_move_task_list(move_task_list)
	if validation_errors:
		print("ERROR: move_task_list validation failed:")
		for err in validation_errors:
			print(f"  - {err}")
		return 2

	map_check: Dict[str, Any] = {
		"enabled": not args.skip_map_check,
	}

	if not args.skip_map_check:
		try:
			map_file_path = args.map_file if args.map_file else load_map_file_path_from_config(args.map_config)
			topo = load_smap_topology(map_file_path)
			map_errors = validate_move_task_list_against_topology(move_task_list, topo)
			map_check.update(
				{
					"map_config": args.map_config,
					"map_file": map_file_path,
					"node_count": topo.get("node_count"),
					"edge_count": topo.get("edge_count"),
					"ok": len(map_errors) == 0,
					"errors": map_errors,
				}
			)
			if map_errors:
				print("ERROR: map topology validation failed:")
				for err in map_errors:
					print(f"  - {err}")
				print("HINT: 如需临时跳过地图校验，可添加 --skip-map-check")
				return 2
		except (OSError, ValueError, json.JSONDecodeError) as exc:
			print(f"ERROR: map topology load failed: {exc}")
			print("HINT: 检查 --map-config/--map-file 指向的 .smap 是否存在且为 JSON")
			print("HINT: 如需临时跳过地图校验，可添加 --skip-map-check")
			return 2

	request_payload = {"move_task_list": move_task_list}
	plan = {
		"target": {
			"robot_ip": args.robot_ip,
			"port": args.port,
			"local_ip": local_ip,
			"status_port": args.status_port,
		},
		"req_id": args.req_id,
		"msg_type": args.msg_type,
		"request_payload": request_payload,
		"notes": [
			"source_id and id direct-connection is validated against map topology when map check is enabled",
			"task_id must be globally unique in robot task queue",
		],
		"map_check": map_check,
	}

	if args.dry_run:
		print(json.dumps(plan, ensure_ascii=False, indent=2))
		return 0

	precheck: Dict[str, Any] = {}

	if args.precheck_only or args.require_ready or args.wait_ready_seconds > 0.0:
		wait_deadline = time.monotonic() + max(0.0, args.wait_ready_seconds)
		wait_interval = max(0.1, args.wait_ready_interval)
		last_sample: Optional[Dict[str, Any]] = None
		last_eval: Optional[Dict[str, Any]] = None

		while True:
			try:
				status_sample = read_one_status_sample(
					ip=args.robot_ip,
					port=args.status_port,
					local_ip=local_ip,
					timeout=args.timeout,
					max_payload_chars=max(80, args.status_max_payload_chars),
					compact=not args.full_status_samples,
				)
			except OSError as exc:
				precheck = {
					"ready": False,
					"error": str(exc),
					"reasons": ["status connection failed"],
				}
				break

			last_sample = status_sample
			payload_json = status_sample.get("payload_json")
			if isinstance(payload_json, dict):
				last_eval = evaluate_navigation_ready(payload_json)
			else:
				last_eval = {
					"ready": False,
					"reasons": ["status payload_json unavailable"],
					"blockers": ["status payload_json unavailable"],
					"warnings": [],
					"observed": {},
				}

			if last_eval.get("ready"):
				break

			if args.wait_ready_seconds <= 0.0 or time.monotonic() >= wait_deadline:
				break

			time.sleep(wait_interval)

		if not precheck:
			precheck = {
				"ready": bool(last_eval and last_eval.get("ready")),
				"status_sample": last_sample,
				"reasons": last_eval.get("reasons") if isinstance(last_eval, dict) else [],
				"blockers": last_eval.get("blockers") if isinstance(last_eval, dict) else [],
				"warnings": last_eval.get("warnings") if isinstance(last_eval, dict) else [],
				"observed": last_eval.get("observed") if isinstance(last_eval, dict) else {},
			}

		if args.precheck_only:
			print(json.dumps({"plan": plan, "precheck": precheck}, ensure_ascii=False, indent=2))
			if precheck.get("ready"):
				warnings = precheck.get("warnings")
				if isinstance(warnings, list) and warnings:
					print("HINT: precheck 通过，但存在告警（例如 is_stop=True）。任务可下发，机器人可能需先解除停止态才会立刻移动。")
				return 0
			print("HINT: precheck 未通过（硬阻塞），先在机器人侧解除占用/异常状态后再发导航任务")
			return 5

		if args.require_ready and not precheck.get("ready"):
			print(json.dumps({"plan": plan, "precheck": precheck}, ensure_ascii=False, indent=2))
			print("HINT: --require-ready 已启用，当前未就绪，已阻止发送 3066")
			return 5

	try:
		send_result = send_one(
			ip=args.robot_ip,
			port=args.port,
			local_ip=local_ip,
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

	if args.listen_status_seconds > 0.0:
		try:
			samples = listen_status_push(
				ip=args.robot_ip,
				port=args.status_port,
				local_ip=local_ip,
				timeout=args.timeout,
				seconds=args.listen_status_seconds,
				max_samples=max(1, args.status_max_samples),
				max_payload_chars=max(80, args.status_max_payload_chars),
				compact=not args.full_status_samples,
			)
			output["status_samples"] = samples
			output["status_sample_count"] = len(samples)
		except OSError as exc:
			output["status_error"] = str(exc)

	print(json.dumps(output, ensure_ascii=False, indent=2))

	ret_code = None
	err_msg = None
	payload_json = send_result.get("response", {}).get("payload_json")
	if isinstance(payload_json, dict):
		rc = payload_json.get("ret_code")
		err_msg = payload_json.get("err_msg")
		if isinstance(rc, int):
			ret_code = rc

	if ret_code == 40020:
		print("HINT: ret_code=40020 表示机器人当前控制权被占用（preempted）。")
		print("HINT: 需先结束当前被占用的控制任务/模式，再发 3066。")

		status_samples = output.get("status_samples")
		latest_status_payload = None
		if isinstance(status_samples, list) and status_samples:
			last = status_samples[-1]
			if isinstance(last, dict):
				candidate = last.get("payload_json")
				if isinstance(candidate, dict):
					latest_status_payload = candidate

		if isinstance(latest_status_payload, dict):
			if latest_status_payload.get("is_stop") is True:
				print("HINT: 状态中 is_stop=True，先在机器人侧解除停止/暂停状态再试。")
			task_status = latest_status_payload.get("task_status")
			task_type = latest_status_payload.get("task_type")
			if task_status is not None or task_type is not None:
				print(f"INFO: 当前 task_status={task_status}, task_type={task_type}")
			running_status = latest_status_payload.get("running_status")
			if running_status is not None:
				print(f"INFO: 当前 running_status={running_status}")
		else:
			print("TIP: 可加 --listen-status-seconds 2 查看实时状态字段，辅助定位占用原因。")
	elif ret_code == 60000:
		print("HINT: ret_code=60000 error api type，通常是端口/服务不匹配。3066 建议走 19206。")
	if isinstance(err_msg, str) and err_msg:
		print(f"INFO: robot err_msg={err_msg}")

	if ret_code is None:
		return 4
	return 0 if ret_code == 0 else 3


if __name__ == "__main__":
	sys.exit(main())
