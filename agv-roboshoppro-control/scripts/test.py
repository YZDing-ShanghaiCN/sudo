import argparse
import json
import socket
import struct
import sys
import time
from typing import List


def tcp_probe(host: str, port: int, timeout: float) -> dict:
	start = time.time()
	try:
		with socket.create_connection((host, port), timeout=timeout):
			return {
				"ok": True,
				"latency_ms": round((time.time() - start) * 1000, 2),
				"error": None,
			}
	except OSError as exc:
		return {
			"ok": False,
			"latency_ms": round((time.time() - start) * 1000, 2),
			"error": str(exc),
		}


def tcp_exchange(
	host: str,
	port: int,
	timeout: float,
	payload: bytes,
	recv_bytes: int,
	recv_timeout: float,
) -> dict:
	start = time.time()
	result = {
		"ok": False,
		"error": None,
		"bytes_sent": 0,
		"response_received": False,
		"recv_timed_out": False,
		"peer_closed": False,
		"response_len": 0,
		"response_hex": None,
		"response_text": None,
		"response_frame": None,
		"response_json": None,
		"roundtrip_ms": None,
	}

	try:
		with socket.create_connection((host, port), timeout=timeout) as sock:
			sock.settimeout(recv_timeout)
			sock.sendall(payload)
			result["bytes_sent"] = len(payload)

			if recv_bytes > 0:
				try:
					resp = sock.recv(recv_bytes)
				except socket.timeout:
					result["recv_timed_out"] = True
					resp = None

				if resp is None:
					pass
				elif resp == b"":
					result["peer_closed"] = True
				elif resp:
					result["response_received"] = True
					result["response_len"] = len(resp)
					result["response_hex"] = resp.hex(" ")
					result["response_text"] = resp.decode("utf-8", errors="replace")
					parsed = parse_seer_frame(resp)
					if parsed is not None:
						result["response_frame"] = parsed["frame"]
						result["response_json"] = parsed["payload_json"]

		result["ok"] = True
		result["roundtrip_ms"] = round((time.time() - start) * 1000, 2)
		return result
	except OSError as exc:
		result["error"] = str(exc)
		result["roundtrip_ms"] = round((time.time() - start) * 1000, 2)
		return result


def tcp_receive_only(host: str, port: int, timeout: float, recv_bytes: int, recv_timeout: float) -> dict:
	start = time.time()
	result = {
		"ok": False,
		"error": None,
		"response_received": False,
		"recv_timed_out": False,
		"peer_closed": False,
		"response_len": 0,
		"response_hex": None,
		"response_text": None,
		"response_frame": None,
		"response_json": None,
		"roundtrip_ms": None,
	}

	try:
		with socket.create_connection((host, port), timeout=timeout) as sock:
			sock.settimeout(recv_timeout)
			try:
				resp = sock.recv(recv_bytes)
			except socket.timeout:
				result["recv_timed_out"] = True
				resp = None

			if resp is None:
				pass
			elif resp == b"":
				result["peer_closed"] = True
			else:
				result["response_received"] = True
				result["response_len"] = len(resp)
				result["response_hex"] = resp.hex(" ")
				result["response_text"] = resp.decode("utf-8", errors="replace")
				parsed = parse_seer_frame(resp)
				if parsed is not None:
					result["response_frame"] = parsed["frame"]
					result["response_json"] = parsed["payload_json"]

		result["ok"] = True
		result["roundtrip_ms"] = round((time.time() - start) * 1000, 2)
		return result
	except OSError as exc:
		result["error"] = str(exc)
		result["roundtrip_ms"] = round((time.time() - start) * 1000, 2)
		return result


def normalize_path(path: str) -> str:
	return path if path.startswith("/") else f"/{path}"


def build_base_url(scheme: str, ip: str, port: int) -> str:
	default_port = 443 if scheme == "https" else 80
	if port == default_port:
		return f"{scheme}://{ip}"
	return f"{scheme}://{ip}:{port}"


def http_probe(base_url: str, paths: List[str], timeout: float, verify_tls: bool) -> List[dict]:
	import requests

	results = []
	for path in paths:
		full_path = normalize_path(path)
		url = f"{base_url}{full_path}"
		start = time.time()
		try:
			resp = requests.get(url, timeout=timeout, verify=verify_tls)
			results.append(
				{
					"url": url,
					"ok": 200 <= resp.status_code < 400,
					"status_code": resp.status_code,
					"latency_ms": round((time.time() - start) * 1000, 2),
				}
			)
		except requests.RequestException as exc:
			results.append(
				{
					"url": url,
					"ok": False,
					"status_code": None,
					"latency_ms": round((time.time() - start) * 1000, 2),
					"error": str(exc),
				}
			)
	return results


def parse_seer_frame(resp: bytes):
	if len(resp) < 16:
		return None
	if not (resp[0] == 0x5A and resp[1] == 0x01):
		return None

	payload_len = int.from_bytes(resp[4:8], byteorder="big", signed=False)
	payload_bytes = resp[16 : 16 + min(payload_len, max(0, len(resp) - 16))]
	payload_text = payload_bytes.decode("utf-8", errors="replace")
	payload_json = None
	try:
		payload_json = json.loads(payload_text)
	except json.JSONDecodeError:
		payload_json = None

	frame = {
		"magic": f"{resp[0]:02x}",
		"version": resp[1],
		"flag_a": resp[2],
		"flag_b": resp[3],
		"payload_len": payload_len,
		"field_8_11_hex": resp[8:12].hex(" "),
		"field_12_15_hex": resp[12:16].hex(" "),
	}

	return {
		"frame": frame,
		"payload_json": payload_json,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="AGV connection test")
	parser.add_argument("--ip", required=True, help="AGV IP address, e.g. 192.168.1.20")
	parser.add_argument("--port", type=int, default=19204, help="AGV TCP port")
	parser.add_argument("--timeout", type=float, default=2.5, help="Timeout in seconds")
	parser.add_argument("--json", action="store_true", help="Print JSON result")
	parser.add_argument("--send-text", default=None, help="Send text payload over TCP, e.g. PING")
	parser.add_argument("--send-hex", default=None, help="Send hex payload over TCP, e.g. AA55FF00")
	parser.add_argument(
		"--frame",
		choices=["none", "seer16-be", "seer16-be-status", "seer16-le"],
		default="none",
		help="Optional payload framing mode",
	)
	parser.add_argument("--seq", type=int, default=1, help="Sequence id used by framed payload")
	parser.add_argument("--append-crlf", action="store_true", help="Append CRLF when using --send-text")
	parser.add_argument("--recv-bytes", type=int, default=256, help="Bytes to read after sending payload")
	parser.add_argument("--recv-timeout", type=float, default=1.0, help="Receive timeout after sending")
	parser.add_argument("--recv-only", action="store_true", help="Connect and only receive data without sending payload")
	parser.add_argument("--with-http", action="store_true", help="Also probe HTTP endpoints after TCP is connected")
	parser.add_argument("--scheme", choices=["http", "https"], default="http")
	parser.add_argument(
		"--probe-path",
		action="append",
		default=None,
		help="HTTP path to probe; can be repeated, e.g. --probe-path /api/status",
	)
	parser.add_argument(
		"--insecure",
		action="store_true",
		help="Disable TLS certificate verification when using HTTPS",
	)
	return parser.parse_args()


def apply_frame(frame: str, payload: bytes, seq: int) -> bytes:
	if frame == "none":
		return payload

	if frame == "seer16-be":
		header = bytearray([0x5A, 0x01, 0x00, 0x01])
		header.extend(struct.pack(">I", len(payload)))
		header.extend(struct.pack(">I", seq & 0xFFFFFFFF))
		header.extend(b"\x00\x00\x00\x00")
		return bytes(header) + payload

	if frame == "seer16-be-status":
		header = bytearray([0x5A, 0x01, 0x00, 0x00])
		header.extend(struct.pack(">I", len(payload)))
		header.extend(struct.pack(">I", seq & 0xFFFFFFFF))
		header.extend(b"\x00\x00\x00\x00")
		return bytes(header) + payload

	if frame == "seer16-le":
		header = bytearray([0x5A, 0x01, 0x00, 0x01])
		header.extend(struct.pack("<I", len(payload)))
		header.extend(struct.pack("<I", seq & 0xFFFFFFFF))
		header.extend(b"\x00\x00\x00\x00")
		return bytes(header) + payload

	raise ValueError(f"unsupported frame mode: {frame}")


def build_payload(args: argparse.Namespace) -> bytes:
	if args.send_text is not None and args.send_hex is not None:
		raise ValueError("--send-text and --send-hex cannot be used together")

	if args.send_hex is not None:
		hex_text = args.send_hex.replace(" ", "")
		if len(hex_text) % 2 != 0:
			raise ValueError("--send-hex must have even number of hex characters")
		return apply_frame(args.frame, bytes.fromhex(hex_text), args.seq)

	if args.send_text is not None:
		text = args.send_text + ("\r\n" if args.append_crlf else "")
		return apply_frame(args.frame, text.encode("utf-8"), args.seq)

	return b""


def main() -> int:
	args = parse_args()
	tcp_result = tcp_probe(args.ip, args.port, args.timeout)
	payload = b""
	send_test = None
	try:
		payload = build_payload(args)
	except ValueError as exc:
		print(f"INVALID_ARGS: {exc}")
		return 2

	if tcp_result["ok"] and payload:
		send_test = tcp_exchange(
			host=args.ip,
			port=args.port,
			timeout=args.timeout,
			payload=payload,
			recv_bytes=max(0, args.recv_bytes),
			recv_timeout=max(0.0, args.recv_timeout),
		)
	elif payload:
		send_test = {
			"ok": False,
			"error": "tcp_not_connected",
			"bytes_sent": 0,
			"response_received": False,
			"recv_timed_out": False,
			"peer_closed": False,
			"response_len": 0,
			"response_hex": None,
			"response_text": None,
			"response_frame": None,
			"response_json": None,
			"roundtrip_ms": None,
		}
	elif tcp_result["ok"] and args.recv_only:
		send_test = tcp_receive_only(
			host=args.ip,
			port=args.port,
			timeout=args.timeout,
			recv_bytes=max(1, args.recv_bytes),
			recv_timeout=max(0.0, args.recv_timeout),
		)

	http_results = []
	base_url = None
	if args.with_http:
		base_url = build_base_url(args.scheme, args.ip, args.port)
		if args.probe_path:
			probe_paths = args.probe_path
		else:
			probe_paths = ["/health", "/status", "/api/health", "/api/status", "/"]
		if tcp_result["ok"]:
			http_results = http_probe(base_url, probe_paths, args.timeout, verify_tls=not args.insecure)

	http_ok = any(item.get("ok") for item in http_results)
	summary = {
		"agv_ip": args.ip,
		"port": args.port,
		"base_url": base_url,
		"tcp_connected": tcp_result["ok"],
		"send_test_enabled": bool(payload) or bool(args.recv_only),
		"send_test": send_test,
		"http_endpoint_found": http_ok,
		"tcp": tcp_result,
		"http_probes": http_results,
	}

	if args.json:
		print(json.dumps(summary, ensure_ascii=False, indent=2))
	else:
		if tcp_result["ok"]:
			print(f"CONNECTED: {args.ip}:{args.port} (latency={tcp_result['latency_ms']}ms)")
		else:
			print(f"DISCONNECTED: {args.ip}:{args.port} (error={tcp_result['error']})")

		if payload:
			if send_test and send_test.get("ok"):
				print(f"SEND_OK: bytes_sent={send_test['bytes_sent']}, response_received={send_test['response_received']}")
				if send_test.get("response_received"):
					if send_test.get("response_json") is not None:
						print(f"RESPONSE_JSON: {json.dumps(send_test['response_json'], ensure_ascii=False)}")
					else:
						print(f"RESPONSE_HEX: {send_test['response_hex']}")
				elif send_test.get("peer_closed"):
					print("RECV_STATE: peer_closed")
				elif send_test.get("recv_timed_out"):
					print("RECV_STATE: timeout_no_response")
				else:
					print("RECV_STATE: no_data")
		elif args.recv_only:
			if send_test and send_test.get("response_received"):
				if send_test.get("response_json") is not None:
					print(f"RECV_JSON: {json.dumps(send_test['response_json'], ensure_ascii=False)}")
				else:
					print("RECV_OK: binary_or_partial_data")
			elif send_test and send_test.get("recv_timed_out"):
				print("RECV_STATE: timeout_no_response")
			else:
				err = send_test.get("error") if send_test else "unknown"
				print(f"SEND_FAIL: error={err}")

	if not tcp_result["ok"]:
		return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
