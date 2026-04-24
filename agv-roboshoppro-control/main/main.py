import argparse
import csv
import json
import os
import re
import socket
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml

PACK_FMT_MSGTYPE = "!BBHLH6s"
HEADER_LEN = 16

DEFAULT_NAV_MSG_TYPE = 3066  # 0x0BFA robot_task_gotargetlist_req
DEFAULT_NAV_PORT = 19206
DEFAULT_STATUS_PORT = 19301
DEFAULT_MAP_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "map.yaml")
DEFAULT_CAMERA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "camera.yaml")
DEFAULT_CAMERA_CALIB_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "camera_cfg.yaml")
DEFAULT_RESULT_ROOT = os.path.join(os.path.dirname(__file__), "result")


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


def load_yaml_object(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		obj = yaml.safe_load(f)
	if not isinstance(obj, dict):
		raise ValueError(f"invalid yaml object: {path}")
	return obj


def load_station_positions(map_file_path: str) -> Dict[str, Tuple[float, float]]:
	with open(map_file_path, "r", encoding="utf-8") as f:
		obj = json.load(f)

	if not isinstance(obj, dict):
		return {}

	positions: Dict[str, Tuple[float, float]] = {}
	for p in obj.get("advancedPointList", []):
		if not isinstance(p, dict):
			continue
		name = p.get("instanceName")
		pos = p.get("pos")
		if not isinstance(name, str) or not isinstance(pos, dict):
			continue
		x = pos.get("x")
		y = pos.get("y")
		if isinstance(x, (int, float)) and isinstance(y, (int, float)):
			positions[name] = (float(x), float(y))
	return positions


def wait_until_arrival(
	ip: str,
	status_port: int,
	local_ip: Optional[str],
	timeout: float,
	target_station: str,
	target_pos_xy: Optional[Tuple[float, float]],
	arrival_tolerance_m: float,
	wait_seconds: float,
	poll_interval: float,
	require_stop: bool,
	min_stop_samples: int,
) -> Dict[str, Any]:
	start = time.monotonic()
	deadline = start + max(0.1, wait_seconds)
	last_sample: Optional[Dict[str, Any]] = None
	last_log = 0.0
	arrival_tol = max(0.01, arrival_tolerance_m)
	require_stop_samples = max(1, int(min_stop_samples))
	first_on_target: Optional[bool] = None
	require_depart_before_arrival = False
	departed_from_target = False
	stop_consecutive = 0
	departure_distance_threshold = max(arrival_tol * 1.5, arrival_tol + 0.2)

	while time.monotonic() < deadline:
		try:
			sample = read_one_status_sample(
				ip=ip,
				port=status_port,
				local_ip=local_ip,
				timeout=timeout,
				max_payload_chars=400,
				compact=True,
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
			x_val = payload.get("x")
			y_val = payload.get("y")
			dist_to_target = None
			if (
				target_pos_xy is not None
				and isinstance(x_val, (int, float))
				and isinstance(y_val, (int, float))
			):
				dx = float(x_val) - float(target_pos_xy[0])
				dy = float(y_val) - float(target_pos_xy[1])
				dist_to_target = float((dx * dx + dy * dy) ** 0.5)

			on_station = isinstance(current_station, str) and current_station == target_station
			within_distance = dist_to_target is not None and dist_to_target <= arrival_tol
			on_target_now = bool(on_station or within_distance)
			if on_target_now and is_stop_now:
				stop_consecutive += 1
			else:
				stop_consecutive = 0

			if first_on_target is None:
				first_on_target = on_target_now
				require_depart_before_arrival = on_target_now

			if require_depart_before_arrival and not departed_from_target:
				left_station = isinstance(current_station, str) and current_station != "" and current_station != target_station
				left_by_distance = dist_to_target is not None and dist_to_target > departure_distance_threshold
				if left_station or left_by_distance:
					departed_from_target = True
					print(
						f"[arrival] departure observed: current_station={current_station} "
						f"dist_to_target={dist_to_target}"
					)

			now = time.monotonic()
			if now - last_log >= 2.0:
				print(
					f"[arrival] current_station={current_station} target={target_station} "
					f"is_stop={payload.get('is_stop')} x={x_val} y={y_val} dist_to_target={dist_to_target} "
					f"require_depart={require_depart_before_arrival} departed={departed_from_target} "
					f"stop_samples={stop_consecutive}/{require_stop_samples}"
				)
				last_log = now

			stop_ready = (not require_stop) or (stop_consecutive >= require_stop_samples)

			if on_station and (not require_depart_before_arrival or departed_from_target) and stop_ready:
				return {
					"arrived": True,
					"arrived_by": "current_station",
					"required_stop": require_stop,
					"stop_samples": stop_consecutive,
					"required_departure": require_depart_before_arrival,
					"departure_observed": departed_from_target,
					"elapsed_s": round(now - start, 2),
					"last_sample": sample,
				}

			if within_distance and (not require_depart_before_arrival or departed_from_target) and stop_ready:
				return {
					"arrived": True,
					"arrived_by": "xy_distance",
					"distance_to_target_m": dist_to_target,
					"required_stop": require_stop,
					"stop_samples": stop_consecutive,
					"required_departure": require_depart_before_arrival,
					"departure_observed": departed_from_target,
					"elapsed_s": round(now - start, 2),
					"last_sample": sample,
				}

		time.sleep(max(0.1, poll_interval))

	return {
		"arrived": False,
		"error": f"timeout waiting for arrival at {target_station}",
		"required_stop": require_stop,
		"stop_samples": stop_consecutive,
		"required_departure": require_depart_before_arrival,
		"departure_observed": departed_from_target,
		"elapsed_s": round(time.monotonic() - start, 2),
		"last_sample": last_sample,
	}


def open_camera_by_index(index: int):
	cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
	if cap.isOpened():
		return cap

	cap.release()
	cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
	if cap.isOpened():
		return cap

	cap.release()
	cap = cv2.VideoCapture(index)
	if cap.isOpened():
		return cap

	cap.release()
	return None


def parse_camera_index_from_device_path(device: Any) -> Optional[int]:
	if not isinstance(device, str):
		return None
	match = re.search(r"video(\d+)$", device.strip())
	if not match:
		return None
	return int(match.group(1))


def normalize_fourcc(value: Any, default_code: str = "MJPG") -> str:
	if not isinstance(value, str):
		return default_code
	code = value.strip().upper()
	if len(code) != 4:
		return default_code
	return code


def fourcc_int_to_str(value: float) -> str:
	iv = int(round(value)) & 0xFFFFFFFF
	raw = iv.to_bytes(4, byteorder="little", signed=False)
	if all(32 <= b <= 126 for b in raw):
		return raw.decode("ascii", errors="replace")
	return f"0x{iv:08X}"


def _build_resolution_candidates(width: int, height: int) -> List[Tuple[int, int]]:
	# Keep the requested profile first, then try common high-resolution modes.
	presets: List[Tuple[int, int]] = [
		(width, height),
		(3840, 2160),
		(2560, 1440),
		(2048, 1536),
		(1920, 1680),
		(1920, 1200),
		(1920, 1080),
		(1600, 1300),
		(1600, 1200),
		(1440, 1080),
		(1280, 1024),
		(1280, 960),
		(1280, 720),
	]

	seen: Set[Tuple[int, int]] = set()
	unique: List[Tuple[int, int]] = []
	for item in presets:
		if item not in seen:
			seen.add(item)
			unique.append(item)
	return unique


def _apply_camera_controls(cap: cv2.VideoCapture, cam_cfg: Dict[str, Any]) -> None:
	if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)

	auto_exp = cam_cfg.get("is_auto_exposure")
	if isinstance(auto_exp, bool):
		if auto_exp:
			cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
			cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)
		else:
			cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
			cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
			manual_exposure = cam_cfg.get("manual_exposure_value")
			if isinstance(manual_exposure, (int, float)):
				cap.set(cv2.CAP_PROP_EXPOSURE, float(manual_exposure))

	backlight = cam_cfg.get("backlight_compensation")
	if isinstance(backlight, (int, float)) and hasattr(cv2, "CAP_PROP_BACKLIGHT"):
		cap.set(cv2.CAP_PROP_BACKLIGHT, float(backlight))

	auto_focus = cam_cfg.get("is_auto_focus")
	if isinstance(auto_focus, bool) and hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
		cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if auto_focus else 0.0)

	focus_value = cam_cfg.get("manual_focus_value")
	if isinstance(focus_value, (int, float)) and hasattr(cv2, "CAP_PROP_FOCUS"):
		if auto_focus is not True and hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
			cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
		cap.set(cv2.CAP_PROP_FOCUS, float(focus_value))


def _read_stream_state(cap: cv2.VideoCapture) -> Dict[str, Any]:
	actual_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
	actual_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
	raw_fourcc = cap.get(cv2.CAP_PROP_FOURCC)
	actual_fourcc = fourcc_int_to_str(raw_fourcc)

	return {
		"width": actual_width,
		"height": actual_height,
		"fps": round(actual_fps, 2),
		"fourcc": actual_fourcc,
		"fourcc_raw": f"0x{(int(round(raw_fourcc)) & 0xFFFFFFFF):08X}",
		"auto_exposure": float(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)) if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE") else None,
		"exposure": float(cap.get(cv2.CAP_PROP_EXPOSURE)) if hasattr(cv2, "CAP_PROP_EXPOSURE") else None,
		"autofocus": float(cap.get(cv2.CAP_PROP_AUTOFOCUS)) if hasattr(cv2, "CAP_PROP_AUTOFOCUS") else None,
		"focus": float(cap.get(cv2.CAP_PROP_FOCUS)) if hasattr(cv2, "CAP_PROP_FOCUS") else None,
	}


def frame_sharpness_score(frame: np.ndarray) -> float:
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	lap = cv2.Laplacian(gray, cv2.CV_64F)
	return float(lap.var())


def _score_stream_profile(profile: Dict[str, Any]) -> Tuple[int, float]:
	# Prefer higher pixel count; use fps as secondary tie-breaker.
	width = int(profile.get("width", 0))
	height = int(profile.get("height", 0))
	fps = float(profile.get("fps", 0.0))
	return width * height, fps


def apply_camera_capture_settings(cap: cv2.VideoCapture, cam_cfg: Dict[str, Any]) -> Dict[str, Any]:
	requested_width = int(cam_cfg.get("width", 1280))
	requested_height = int(cam_cfg.get("height", 720))
	requested_fps = int(cam_cfg.get("fps", 30))
	fourcc = normalize_fourcc(cam_cfg.get("fourcc", "MJPG"))
	request_list = _build_resolution_candidates(requested_width, requested_height)

	best_actual: Optional[Dict[str, Any]] = None
	best_request: Optional[Dict[str, Any]] = None
	negotiation_log: List[Dict[str, Any]] = []

	for req_w, req_h in request_list:
		cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
		cap.set(cv2.CAP_PROP_FPS, requested_fps)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
		_apply_camera_controls(cap, cam_cfg)

		# Read a couple frames to let backend finish mode switch.
		for _ in range(2):
			cap.read()

		actual_profile = _read_stream_state(cap)
		attempt = {
			"request": {"width": req_w, "height": req_h, "fps": requested_fps, "fourcc": fourcc},
			"actual": actual_profile,
		}
		negotiation_log.append(attempt)

		if best_actual is None or _score_stream_profile(actual_profile) > _score_stream_profile(best_actual):
			best_actual = actual_profile
			best_request = attempt["request"]

		if actual_profile["width"] == requested_width and actual_profile["height"] == requested_height:
			break

	if best_actual is None or best_request is None:
		best_request = {
			"width": requested_width,
			"height": requested_height,
			"fps": requested_fps,
			"fourcc": fourcc,
		}
		best_actual = _read_stream_state(cap)
	else:
		# Re-apply the best negotiated request so downstream reads stay stable.
		cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
		cap.set(cv2.CAP_PROP_FPS, requested_fps)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(best_request["width"]))
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(best_request["height"]))
		_apply_camera_controls(cap, cam_cfg)
		for _ in range(2):
			cap.read()
		best_actual = _read_stream_state(cap)

	backend_name = "UNKNOWN"
	if hasattr(cap, "getBackendName"):
		try:
			backend_name = str(cap.getBackendName())
		except Exception:
			backend_name = "UNKNOWN"

	return {
		"backend": backend_name,
		"requested": {
			"width": requested_width,
			"height": requested_height,
			"fps": requested_fps,
			"fourcc": fourcc,
		},
		"selected_request": best_request,
		"actual": best_actual,
		"negotiation": negotiation_log,
	}


def build_aruco_detector(dictionary_name: str):
	if not hasattr(cv2, "aruco"):
		raise RuntimeError("OpenCV aruco module is unavailable; install opencv-contrib-python")

	if not hasattr(cv2.aruco, dictionary_name):
		raise ValueError(f"unsupported marker dictionary: {dictionary_name}")

	dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
	if hasattr(cv2.aruco, "DetectorParameters"):
		params = cv2.aruco.DetectorParameters()
	else:
		params = cv2.aruco.DetectorParameters_create()

	if hasattr(cv2.aruco, "ArucoDetector"):
		detector = cv2.aruco.ArucoDetector(dictionary, params)

		def _detect(gray: np.ndarray):
			corners, ids, _ = detector.detectMarkers(gray)
			return corners, ids

	else:

		def _detect(gray: np.ndarray):
			corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
			return corners, ids

	return _detect


def load_camera_runtime_config(path: str) -> Dict[str, Any]:
	cfg = load_yaml_object(path)
	camera = cfg.get("camera")
	marker = cfg.get("marker")
	if isinstance(camera, dict):
		marker_cfg = marker if isinstance(marker, dict) else {}
		return {
			"camera_index": int(camera.get("index", 1)),
			"width": int(camera.get("width", 1280)),
			"height": int(camera.get("height", 720)),
			"fps": int(camera.get("fps", 30)),
			"fourcc": normalize_fourcc(camera.get("fourcc", "MJPG")),
			"is_auto_exposure": bool(camera.get("is_auto_exposure", True)),
			"manual_exposure_value": camera.get("manual_exposure_value"),
			"backlight_compensation": camera.get("backlight_compensation"),
			"is_auto_focus": camera.get("is_auto_focus"),
			"manual_focus_value": camera.get("manual_focus_value"),
			"dictionary": str(marker_cfg.get("dictionary", "DICT_APRILTAG_36h11")),
			"marker_id": int(marker_cfg.get("id", 1)),
			"marker_size_m": float(marker_cfg.get("size_m", 0.08)),
		}

	cameras = cfg.get("cameras")
	if isinstance(cameras, list) and len(cameras) > 0:
		camera_node: Optional[Dict[str, Any]] = None
		for item in cameras:
			if isinstance(item, dict) and bool(item.get("enable", True)):
				camera_node = item
				break
		if camera_node is None:
			for item in cameras:
				if isinstance(item, dict):
					camera_node = item
					break
		if camera_node is None:
			raise ValueError(f"invalid cameras list in config: {path}")

		device_index = parse_camera_index_from_device_path(camera_node.get("device"))
		camera_index = camera_node.get("index")
		if not isinstance(camera_index, int):
			camera_index = camera_node.get("camera_index")
		if not isinstance(camera_index, int):
			camera_index = device_index if device_index is not None else 1

		return {
			"camera_index": int(camera_index),
			"width": int(camera_node.get("width", 1280)),
			"height": int(camera_node.get("height", 720)),
			"fps": int(camera_node.get("fps", 30)),
			"fourcc": normalize_fourcc(camera_node.get("fourcc", "MJPG")),
			"is_auto_exposure": bool(camera_node.get("is_auto_exposure", True)),
			"manual_exposure_value": camera_node.get("manual_exposure_value"),
			"backlight_compensation": camera_node.get("backlight_compensation"),
			"is_auto_focus": camera_node.get("is_auto_focus"),
			"manual_focus_value": camera_node.get("manual_focus_value"),
			"dictionary": "DICT_APRILTAG_36h11",
			"marker_id": 1,
			"marker_size_m": 0.08,
		}

	raise ValueError(f"unsupported camera config format: {path}")


def load_camera_intrinsics(path: str, key: str) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
	cfg = load_yaml_object(path)
	node = cfg.get(key)
	if not isinstance(node, dict):
		raise ValueError(f"camera calib key not found: {key}")

	cm = np.array(node.get("camera_matrix"), dtype=np.float64)
	dc = np.array(node.get("dist_coeffs"), dtype=np.float64)
	if dc.ndim == 2 and dc.shape[0] == 1:
		dc = dc.reshape(-1)
	if cm.shape != (3, 3):
		raise ValueError(f"invalid camera_matrix shape for {key}: {cm.shape}")

	calib_cam_idx = node.get("camera_index")
	return cm, dc, int(calib_cam_idx) if isinstance(calib_cam_idx, int) else None


def marker_center_and_area(pts: np.ndarray) -> Tuple[float, float, float]:
	cx = float((pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4.0)
	cy = float((pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4.0)
	area = float(abs(cv2.contourArea(pts.astype("float32"))))
	return cx, cy, area


def append_row_csv(csv_path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
	os.makedirs(os.path.dirname(csv_path), exist_ok=True)
	file_exists = os.path.exists(csv_path)
	with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		if not file_exists:
			writer.writeheader()
		writer.writerow(row)


def capture_apriltag_pose_after_arrival(args: argparse.Namespace) -> Dict[str, Any]:
	cam_cfg = load_camera_runtime_config(args.camera_config)
	if args.camera_fourcc:
		cam_cfg["fourcc"] = normalize_fourcc(args.camera_fourcc)
	camera_index = int(args.camera_index) if args.camera_index is not None else int(cam_cfg["camera_index"])

	cap = open_camera_by_index(camera_index)
	if cap is None:
		raise RuntimeError(f"cannot open camera index {camera_index}")

	stream_profile = apply_camera_capture_settings(cap, cam_cfg)
	print(f"[camera] backend   {stream_profile.get('backend', 'UNKNOWN')}")
	print(
		"[camera] requested "
		f"{stream_profile['requested']['width']}x{stream_profile['requested']['height']} "
		f"@{stream_profile['requested']['fps']} fourcc={stream_profile['requested']['fourcc']}"
	)
	selected = stream_profile.get("selected_request")
	if isinstance(selected, dict):
		print(
			"[camera] selected  "
			f"{selected.get('width')}x{selected.get('height')} "
			f"@{selected.get('fps')} fourcc={selected.get('fourcc')}"
		)
	print(
		"[camera] actual    "
		f"{stream_profile['actual']['width']}x{stream_profile['actual']['height']} "
		f"@{stream_profile['actual']['fps']} fourcc={stream_profile['actual']['fourcc']} "
		f"raw={stream_profile['actual'].get('fourcc_raw')}"
	)
	for _ in range(max(0, args.camera_warmup_frames)):
		cap.read()

	run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = os.path.join(args.result_root, run_stamp)
	os.makedirs(run_dir, exist_ok=False)

	start = time.monotonic()
	captured_frame: Optional[np.ndarray] = None
	best_sharpness = -1.0
	frames_seen = 0
	window_name = "capture_after_arrival"
	if args.camera_show:
		cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

	try:
		burst_min_seconds = max(0.0, float(args.camera_burst_min_seconds))
		burst_frames = max(1, int(args.camera_burst_frames))
		while time.monotonic() - start < max(1.0, args.camera_max_seconds):
			ok, frame = cap.read()
			if not ok or frame is None:
				continue
			frames_seen += 1
			sharpness = frame_sharpness_score(frame)
			if captured_frame is None or sharpness > best_sharpness:
				captured_frame = frame.copy()
				best_sharpness = sharpness

			if args.camera_show:
				view = frame.copy()
				cv2.putText(
					view,
					(
						f"elapsed={time.monotonic()-start:.1f}s cam={camera_index} "
						f"sharp={sharpness:.1f} best={best_sharpness:.1f}"
					),
					(18, 40),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7,
					(255, 255, 255),
					2,
				)
				cv2.imshow(window_name, view)
				key = cv2.waitKey(1) & 0xFF
				if key in (ord("q"), 27):
					break

			elapsed = time.monotonic() - start
			if frames_seen >= burst_frames and elapsed >= burst_min_seconds:
				break
	finally:
		cap.release()
		if args.camera_show:
			cv2.destroyWindow(window_name)

	raw_path = os.path.join(run_dir, "capture_raw.png")
	json_path = os.path.join(run_dir, "summary.json")
	summary: Dict[str, Any] = {
		"ok": captured_frame is not None,
		"run_dir": run_dir,
		"camera_index": camera_index,
		"camera_stream": stream_profile,
		"capture_metrics": {
			"frames_seen": frames_seen,
			"selected_sharpness": round(best_sharpness, 3) if best_sharpness >= 0.0 else None,
			"burst_frames": int(args.camera_burst_frames),
			"burst_min_seconds": float(args.camera_burst_min_seconds),
		},
		"raw_image": None,
	}
	if captured_frame is not None:
		cv2.imwrite(raw_path, captured_frame)
		summary["raw_image"] = raw_path
	else:
		summary["error"] = "no valid frame captured after arrival"

	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)

	summary["summary_json"] = json_path
	return summary


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
	p.add_argument("--capture-only", action="store_true", help="Only open camera and capture a photo without sending navigation")
	p.add_argument("--wait-arrival-seconds", type=float, default=180.0, help="Wait up to N seconds for target station arrival")
	p.add_argument("--arrival-poll-interval", type=float, default=0.5, help="Arrival polling interval seconds")
	p.add_argument(
		"--arrival-tolerance-m",
		type=float,
		default=0.25,
		help="Use x/y fallback arrival distance threshold (meters) when current_station is empty",
	)
	p.add_argument(
		"--arrival-require-stop",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Require is_stop=True before considering arrival confirmed",
	)
	p.add_argument(
		"--arrival-stop-samples",
		type=int,
		default=3,
		help="Consecutive arrival samples with is_stop=True required before capture",
	)
	p.add_argument(
		"--detect-after-arrival",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="After ret_code=0 and arrival confirmed, open camera and capture one photo",
	)
	p.add_argument("--camera-config", default=DEFAULT_CAMERA_CONFIG_PATH, help="Camera runtime config yaml")
	p.add_argument("--camera-calib", default=DEFAULT_CAMERA_CALIB_PATH, help="Camera intrinsics yaml")
	p.add_argument("--camera-calib-key", default="camera_1", help="Camera intrinsics key in --camera-calib")
	p.add_argument("--camera-index", type=int, default=None, help="Override camera index")
	p.add_argument("--camera-fourcc", default=None, help="Override camera FourCC, e.g. MJPG or YUY2")
	p.add_argument("--marker-id", type=int, default=None, help="Override marker id")
	p.add_argument("--expected-tag-count", type=int, default=2, help="Expected number of tags with the same marker id (left-to-right slots)")
	p.add_argument("--marker-dictionary", default=None, help="Override marker dictionary name")
	p.add_argument("--marker-size-m", type=float, default=None, help="Override marker side length in meters")
	p.add_argument("--camera-show", action=argparse.BooleanOptionalAction, default=True, help="Show camera preview window")
	p.add_argument("--camera-warmup-frames", type=int, default=20, help="Camera warmup frame count")
	p.add_argument("--camera-burst-frames", type=int, default=12, help="Capture N frames and pick the sharpest frame")
	p.add_argument("--camera-burst-min-seconds", type=float, default=0.8, help="Minimum burst capture duration before selecting sharpest frame")
	p.add_argument("--camera-max-seconds", type=float, default=15.0, help="Max seconds to collect marker samples")
	p.add_argument("--camera-min-hits", type=int, default=5, help="Min valid marker hits per tag slot before summarizing")
	p.add_argument("--camera-min-marker-area-px", type=float, default=120.0, help="Min marker area in pixels")
	p.add_argument("--result-root", default=DEFAULT_RESULT_ROOT, help="Result root dir, data saved under timestamp folder")
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

	if args.capture_only:
		try:
			result = capture_apriltag_pose_after_arrival(args)
		except (OSError, ValueError, RuntimeError) as exc:
			print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
			return 7
		print(json.dumps(result, ensure_ascii=False, indent=2))
		return 0 if result.get("ok") else 7

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
	resolved_map_file_path: Optional[str] = None

	if not args.skip_map_check:
		try:
			map_file_path = args.map_file if args.map_file else load_map_file_path_from_config(args.map_config)
			resolved_map_file_path = map_file_path
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

	ret_code = None
	err_msg = None
	payload_json = send_result.get("response", {}).get("payload_json")
	if isinstance(payload_json, dict):
		rc = payload_json.get("ret_code")
		err_msg = payload_json.get("err_msg")
		if isinstance(rc, int):
			ret_code = rc

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

	if ret_code == 0 and args.detect_after_arrival:
		target_station = None
		if move_task_list:
			candidate = move_task_list[-1].get("id")
			if isinstance(candidate, str) and candidate:
				target_station = candidate

		if target_station is None:
			output["arrival_wait"] = {
				"arrived": False,
				"error": "cannot resolve target station from move_task_list",
			}
			output["arrival_capture"] = {
				"ok": False,
				"error": "skip camera capture because target station is unknown",
			}
		else:
			station_positions: Dict[str, Tuple[float, float]] = {}
			if resolved_map_file_path is None:
				try:
					resolved_map_file_path = args.map_file if args.map_file else load_map_file_path_from_config(args.map_config)
				except (OSError, ValueError) as exc:
					output["arrival_wait_warning"] = f"failed to load map file path: {exc}"

			if resolved_map_file_path:
				try:
					station_positions = load_station_positions(resolved_map_file_path)
				except (OSError, ValueError, json.JSONDecodeError) as exc:
					output["arrival_wait_warning"] = f"failed to load station positions: {exc}"

			wait_result = wait_until_arrival(
				ip=args.robot_ip,
				status_port=args.status_port,
				local_ip=local_ip,
				timeout=args.timeout,
				target_station=target_station,
				target_pos_xy=station_positions.get(target_station),
				arrival_tolerance_m=max(0.01, args.arrival_tolerance_m),
				wait_seconds=max(1.0, args.wait_arrival_seconds),
				poll_interval=max(0.1, args.arrival_poll_interval),
				require_stop=bool(args.arrival_require_stop),
				min_stop_samples=max(1, int(args.arrival_stop_samples)),
			)
			output["arrival_wait"] = wait_result

			if wait_result.get("arrived"):
				try:
					capture_result = capture_apriltag_pose_after_arrival(args)
				except (OSError, ValueError, RuntimeError) as exc:
					capture_result = {
						"ok": False,
						"error": str(exc),
					}
				output["arrival_capture"] = capture_result
			else:
				output["arrival_capture"] = {
					"ok": False,
					"error": "skip camera capture because arrival confirmation failed",
				}

	print(json.dumps(output, ensure_ascii=False, indent=2))

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
	if ret_code != 0:
		return 3

	if args.detect_after_arrival:
		detect_result = output.get("arrival_capture")
		if isinstance(detect_result, dict) and detect_result.get("ok") is False:
			return 7

	return 0


if __name__ == "__main__":
	sys.exit(main())
