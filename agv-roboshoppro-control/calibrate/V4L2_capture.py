#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Camera External Trigger High-Performance Capture System.

Supports 1 to N cameras with hardware synchronization.

Architecture (N+2 threads):
    CaptureThread 0 --queue_0--+
    CaptureThread 1 --queue_1--+
    ...                         +-- GrouperThread --grouped_queue--+-- MainThread (preview)
    CaptureThread N --queue_N--+                                   +-- WriterThread (.jpg)

Key optimization: MJPEG raw bytes written directly to disk — zero encode/decode on the
capture and storage path. Only the preview path decodes for display.

Usage:
    python3 V4L2_dual_capture.py config.json
    python3 V4L2_dual_capture.py  (uses default config)

Controls:
    'c' - toggle continuous save
    's' - save single group
    'q' - quit
"""

import argparse
import csv
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import IO, Any

import cv2
import numpy as np
from linuxpy.video.device import BufferType, Device, PixelFormat


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RawFrame:
    """A single raw MJPEG frame captured from V4L2."""

    cam_id: str
    raw_bytes: bytes
    timestamp: float
    sequence: int
    recv_time: float


@dataclass(slots=True)
class FrameGroup:
    """A matched group of frames from all cameras."""

    group_id: int
    frames: dict[str, RawFrame]
    sync_window_us: float
    group_time: float


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "cameras": [
        {
            "cam_id": "camA",
            "device": "/dev/video0",
            "width": 1920,
            "height": 1080,
            "fps": 60,
            "is_auto_exposure": False,
            "manual_exposure_value": 1,
            "backlight_compensation": 2,
            "enable": True,
        },
        {
            "cam_id": "camB",
            "device": "/dev/video4",
            "width": 1920,
            "height": 1080,
            "fps": 60,
            "is_auto_exposure": False,
            "manual_exposure_value": 1,
            "backlight_compensation": 2,
            "enable": True,
        },
    ],
    "trigger_freq_hz": 60,
    "sync_threshold_ms": 5.0,
    "preview_fps": 30,
    "preview_height": 400,
    "display_height": 400,
}


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from a JSON file, falling back to defaults.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        Merged configuration dictionary.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config: dict[str, Any] = json.load(f)
        print(f"[Config] Loaded: {config_path}")
    except FileNotFoundError:
        print(f"[Config] File not found: {config_path}, using defaults")
        config = DEFAULT_CONFIG.copy()
    except json.JSONDecodeError as e:
        print(f"[Config] JSON error: {e}, using defaults")
        config = DEFAULT_CONFIG.copy()

    # Ensure fields have defaults
    config.setdefault("trigger_freq_hz", 60)
    config.setdefault("preview_fps", 30)
    # display_height is an alias for preview_height
    config.setdefault("preview_height", config.get("display_height", 400))
    config.setdefault("display_height", 400)

    # Default sync threshold: half period of trigger frequency
    if "sync_threshold_ms" not in config:
        freq: float = config.get("trigger_freq_hz", 60)
        config["sync_threshold_ms"] = (1000.0 / freq) / 2.0

    return config


def fourcc_to_string(fourcc: int) -> str:
    """Convert a FourCC integer to a human-readable 4-char string.

    Args:
        fourcc: FourCC encoded integer.

    Returns:
        4-character string representation.
    """
    chars: list[str] = []
    for i in range(4):
        chars.append(chr((fourcc >> 8 * i) & 0xFF))
    return "".join(chars)


# ---------------------------------------------------------------------------
# CaptureThread — one per camera
# ---------------------------------------------------------------------------


class CaptureThread(threading.Thread):
    """Captures raw MJPEG frames from a single V4L2 device."""

    def __init__(
        self,
        cam_config: dict[str, Any],
        out_queue: queue.Queue[RawFrame],
        stop_event: threading.Event,
    ) -> None:
        """Initialize the capture thread.

        Args:
            cam_config: Camera configuration dict (cam_id, device, width, …).
            out_queue: Queue to push captured RawFrame objects.
            stop_event: Event signalling graceful shutdown.
        """
        super().__init__(daemon=True)
        self.cam_config = cam_config
        self.out_queue = out_queue
        self.stop_event = stop_event

        self.cam_id: str = cam_config["cam_id"]
        self.device_path: str = cam_config["device"]
        self.width: int = cam_config["width"]
        self.height: int = cam_config["height"]
        self.fps: int = cam_config.get("fps", 60)

        self.device: Device | None = None
        self.frame_count: int = 0

    # ---- device setup ----

    def _setup_device(self) -> bool:
        """Open the V4L2 device and configure format / exposure / backlight.

        Returns:
            True on success, False on failure.
        """
        try:
            self.device = Device(self.device_path)
            self.device.open()

            # Set MJPEG format
            try:
                self.device.set_format(
                    BufferType.VIDEO_CAPTURE,
                    self.width,
                    self.height,
                    pixel_format=PixelFormat.MJPEG,  # type: ignore[arg-type]
                )
            except Exception as exc:
                print(f"[{self.cam_id}] MJPEG format failed: {exc}")
                try:
                    self.device.set_format(
                        BufferType.VIDEO_CAPTURE,
                        self.width,
                        self.height,
                    )
                except Exception as exc2:
                    print(f"[{self.cam_id}] Default format also failed: {exc2}")

            # Set FPS (skip when backlight_compensation == 2)
            if self.cam_config.get("backlight_compensation", 0) != 2:
                try:
                    self.device.set_fps(BufferType.VIDEO_CAPTURE, self.fps)
                except Exception as exc:
                    print(f"[{self.cam_id}] Set FPS failed: {exc}")
            else:
                print(
                    f"[{self.cam_id}] Skipping FPS setting (backlight_compensation=2)"
                )

            # Exposure
            self._setup_exposure()

            # Backlight compensation
            assert self.device is not None
            try:
                backlight = self.cam_config.get("backlight_compensation", 0)
                ctrl = self.device.controls.backlight_compensation
                ctrl.value = backlight
            except Exception as exc:
                print(f"[{self.cam_id}] Backlight compensation failed: {exc}")

            # Print actual format
            self._print_device_info()
            return True

        except Exception as exc:
            print(f"[{self.cam_id}] Device setup failed: {exc}")
            import traceback

            traceback.print_exc()
            return False

    def _setup_exposure(self) -> None:
        """Configure auto or manual exposure."""
        assert self.device is not None
        device = self.device
        if self.cam_config.get("is_auto_exposure", True):
            try:
                device.controls.auto_exposure.value = 3  # type: ignore[union-attr]
            except Exception as exc:
                print(f"[{self.cam_id}] Auto exposure failed: {exc}")
        else:
            try:
                device.controls.auto_exposure.value = 1  # type: ignore[union-attr]
                exposure_value = self.cam_config.get("manual_exposure_value", 150)
                exposure_set = False

                for attr in ("exposure_absolute", "exposure", "exposure_time_absolute"):
                    if hasattr(device.controls, attr):
                        getattr(device.controls, attr).value = exposure_value
                        exposure_set = True
                        break

                if exposure_set:
                    print(f"[{self.cam_id}] Manual exposure set: {exposure_value}")
                else:
                    available: list[str] = []
                    for n in dir(device.controls):
                        if not n.startswith("_"):
                            available.append(n)
                    print(f"[{self.cam_id}] WARNING: No exposure control found")
                    print(f"[{self.cam_id}] Available controls: {', '.join(available)}")

            except Exception as exc:
                print(f"[{self.cam_id}] Manual exposure failed: {exc}")

    def _print_device_info(self) -> None:
        """Print the actual device format and FPS."""
        assert self.device is not None
        device = self.device
        try:
            fmt = device.get_format(BufferType.VIDEO_CAPTURE)
            px_str = fourcc_to_string(fmt.pixel_format)
            print(
                f"[{self.cam_id}] Opened {self.device_path}  "
                f"{fmt.width}x{fmt.height}  {px_str}"
            )
        except Exception as exc:
            print(
                f"[{self.cam_id}] Opened {self.device_path} (format query failed: {exc})"
            )
        try:
            fps = device.get_fps(BufferType.VIDEO_CAPTURE)
            print(f"[{self.cam_id}] FPS: {fps}")
        except Exception as exc:
            print(f"[{self.cam_id}] FPS query failed: {exc}")

    # ---- main loop ----

    def run(self) -> None:
        """Capture loop: iterate V4L2 frames, push RawFrame to queue."""
        try:
            if not self._setup_device():
                return

            assert self.device is not None
            device = self.device
            for frame_data in device:
                if self.stop_event.is_set():
                    break

                self.frame_count += 1
                raw = RawFrame(
                    cam_id=self.cam_id,
                    raw_bytes=bytes(frame_data),
                    timestamp=frame_data.timestamp,
                    sequence=frame_data.frame_nb,
                    recv_time=time.monotonic(),
                )
                try:
                    self.out_queue.put(raw, timeout=0.1)
                except queue.Full:
                    pass  # Drop frame if queue is full

        except Exception as exc:
            print(f"[{self.cam_id}] Exception: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            if self.device is not None:
                try:
                    self.device.close()
                except Exception:
                    pass
            print(f"[{self.cam_id}] Thread exited (captured {self.frame_count} frames)")


# ---------------------------------------------------------------------------
# GrouperThread — match frames from N cameras by timestamp
# ---------------------------------------------------------------------------


class GrouperThread(threading.Thread):
    """Groups frames from N cameras based on V4L2 buffer timestamps.

    Algorithm (hold-and-compare for N cameras):
        1. Hold one frame per camera slot.
        2. When all slots are filled, compute timestamp spread (max - min).
        3. If spread <= threshold, emit a FrameGroup and clear all slots.
        4. Otherwise, drop the frame with the oldest timestamp.

    For N=1, every frame passes through immediately (spread is always 0).
    For N=2, this is equivalent to the original PairerThread algorithm.
    """

    def __init__(
        self,
        queues: dict[str, queue.Queue[RawFrame]],
        grouped_queue: queue.Queue[FrameGroup],
        sync_threshold_ms: float,
        stop_event: threading.Event,
    ) -> None:
        """Initialize the grouper thread.

        Args:
            queues: Mapping of cam_id to incoming frame queue.
            grouped_queue: Output queue for matched FrameGroup objects.
            sync_threshold_ms: Maximum allowed timestamp spread in milliseconds.
            stop_event: Event signalling graceful shutdown.
        """
        super().__init__(daemon=True)
        self.queues = queues
        self.cam_ids: list[str] = list(queues.keys())
        self.grouped_queue = grouped_queue
        self.stop_event = stop_event

        self.threshold: float = sync_threshold_ms / 1000.0
        self.group_count: int = 0
        self.drops: dict[str, int] = {}
        for cam_id in self.cam_ids:
            self.drops[cam_id] = 0

    def run(self) -> None:
        """Grouping loop: hold-and-compare algorithm for N cameras."""
        held: dict[str, RawFrame | None] = {}
        for cam_id in self.cam_ids:
            held[cam_id] = None

        while not self.stop_event.is_set():
            # Fill empty slots with non-blocking get, then one blocking wait
            # to avoid busy-loop when no frames are arriving.
            got_any = False
            first_empty: str | None = None
            for cam_id in self.cam_ids:
                if held[cam_id] is not None:
                    continue
                if first_empty is None:
                    first_empty = cam_id
                try:
                    held[cam_id] = self.queues[cam_id].get_nowait()
                    got_any = True
                except queue.Empty:
                    pass

            # If nothing was fetched, do a short blocking wait on the first
            # empty slot to yield CPU instead of spinning.
            if not got_any and first_empty is not None:
                try:
                    held[first_empty] = self.queues[first_empty].get(timeout=0.05)
                except queue.Empty:
                    pass
                continue

            # Check if all slots are filled
            all_filled = True
            for cam_id in self.cam_ids:
                if held[cam_id] is None:
                    all_filled = False
                    break

            if not all_filled:
                continue

            # Compute timestamp spread across all held frames
            ts_min = float("inf")
            ts_max = float("-inf")
            for cam_id in self.cam_ids:
                frame = held[cam_id]
                assert frame is not None
                ts = frame.timestamp
                if ts < ts_min:
                    ts_min = ts
                if ts > ts_max:
                    ts_max = ts

            diff = ts_max - ts_min

            if diff <= self.threshold:
                # All cameras synced — emit group
                self.group_count += 1
                frames: dict[str, RawFrame] = {}
                for cam_id in self.cam_ids:
                    frame = held[cam_id]
                    assert frame is not None
                    frames[cam_id] = frame

                group = FrameGroup(
                    group_id=self.group_count,
                    frames=frames,
                    sync_window_us=diff * 1e6,
                    group_time=time.monotonic(),
                )
                try:
                    self.grouped_queue.put(group, timeout=0.1)
                except queue.Full:
                    pass

                # Clear all held frames
                for cam_id in self.cam_ids:
                    held[cam_id] = None
            else:
                # Drop the frame with the oldest timestamp
                oldest_cam = self.cam_ids[0]
                oldest_ts = held[oldest_cam].timestamp  # type: ignore[union-attr]
                for cam_id in self.cam_ids[1:]:
                    frame = held[cam_id]
                    assert frame is not None
                    if frame.timestamp < oldest_ts:
                        oldest_ts = frame.timestamp
                        oldest_cam = cam_id
                self.drops[oldest_cam] += 1
                held[oldest_cam] = None

        # Exit summary
        drop_parts: list[str] = []
        for cam_id in self.cam_ids:
            drop_parts.append(f"{cam_id}: {self.drops[cam_id]}")
        print(
            f"[Grouper] Exited — groups: {self.group_count}, "
            f"dropped {{ {', '.join(drop_parts)} }}"
        )


# ---------------------------------------------------------------------------
# WriterThread — raw MJPEG bytes to disk (zero decode)
# ---------------------------------------------------------------------------


class WriterThread(threading.Thread):
    """Writes raw MJPEG bytes to disk and maintains a CSV log."""

    def __init__(
        self,
        write_queue: queue.Queue[FrameGroup],
        session_dir: str,
        cam_ids: list[str],
        stop_event: threading.Event,
    ) -> None:
        """Initialize the writer thread.

        Args:
            write_queue: Queue of FrameGroup objects to persist.
            session_dir: Base session directory for output files.
            cam_ids: Ordered list of camera identifiers.
            stop_event: Event signalling graceful shutdown.
        """
        super().__init__(daemon=True)
        self.write_queue = write_queue
        self.session_dir = session_dir
        self.cam_ids = cam_ids
        self.stop_event = stop_event

        self.written_count: int = 0

        # Subdirectories for each camera
        self.cam_dirs: dict[str, str] = {}
        for cam_id in cam_ids:
            cam_dir = os.path.join(session_dir, cam_id)
            os.makedirs(cam_dir, exist_ok=True)
            self.cam_dirs[cam_id] = cam_dir

        # CSV log with dynamic columns
        csv_path = os.path.join(session_dir, "ts_group.csv")
        self._csv_file: IO[str] = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)

        header: list[str] = ["group_id"]
        for cam_id in cam_ids:
            header.append(f"ts_{cam_id}")
        for cam_id in cam_ids:
            header.append(f"seq_{cam_id}")
        header.append("sync_window_us")
        self._csv_writer.writerow(header)

    def run(self) -> None:
        """Writer loop: dequeue groups, write raw bytes + CSV rows."""
        try:
            while not self.stop_event.is_set():
                try:
                    group = self.write_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Write raw MJPEG bytes directly — no encode/decode
                for cam_id in self.cam_ids:
                    frame = group.frames[cam_id]
                    path = os.path.join(
                        self.cam_dirs[cam_id], f"{group.group_id:06d}.jpg"
                    )
                    with open(path, "wb") as f:
                        f.write(frame.raw_bytes)

                # CSV row with dynamic columns
                row: list[str | int] = [group.group_id]
                for cam_id in self.cam_ids:
                    row.append(f"{group.frames[cam_id].timestamp:.6f}")
                for cam_id in self.cam_ids:
                    row.append(group.frames[cam_id].sequence)
                row.append(f"{group.sync_window_us:.1f}")
                self._csv_writer.writerow(row)

                self.written_count += 1
                if self.written_count % 30 == 0:
                    self._csv_file.flush()

        except Exception as exc:
            print(f"[Writer] Exception: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            self._csv_file.flush()
            self._csv_file.close()
            print(f"[Writer] Exited — wrote {self.written_count} groups")


# ---------------------------------------------------------------------------
# MultiCameraCapture — main controller
# ---------------------------------------------------------------------------


class MultiCameraCapture:
    """Main controller orchestrating capture, grouping, writing and preview."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the multi-camera capture system.

        Args:
            config: Full configuration dictionary.
        """
        self.config = config

        # Collect all enabled cameras
        enabled: list[dict[str, Any]] = []
        for cam in config.get("cameras", []):
            if cam.get("enable", True):
                enabled.append(cam)
        if len(enabled) < 1:
            print("ERROR: Need at least 1 enabled camera in config!")
            sys.exit(1)

        self.cam_configs = enabled
        self.cam_ids: list[str] = []
        self._cam_config_map: dict[str, dict[str, Any]] = {}
        for cam_cfg in enabled:
            self.cam_ids.append(cam_cfg["cam_id"])
            self._cam_config_map[cam_cfg["cam_id"]] = cam_cfg

        self.sync_threshold_ms: float = config.get("sync_threshold_ms", 5.0)
        self.preview_fps: int = config.get("preview_fps", 30)
        self.preview_height: int = config.get("preview_height", 400)

        # Fixed tile size for stable preview layout — use widest aspect ratio
        max_aspect: float = 0.0
        for cam_cfg in enabled:
            aspect = cam_cfg["width"] / cam_cfg["height"]
            if aspect > max_aspect:
                max_aspect = aspect
        if max_aspect <= 0.0:
            max_aspect = 16.0 / 9.0
        self._preview_tile_width: int = int(self.preview_height * max_aspect)

        # Thread communication — one queue per camera
        self.capture_queues: dict[str, queue.Queue[RawFrame]] = {}
        for cam_id in self.cam_ids:
            self.capture_queues[cam_id] = queue.Queue(maxsize=10)
        self.grouped_queue: queue.Queue[FrameGroup] = queue.Queue(maxsize=10)
        self.write_queue: queue.Queue[FrameGroup] = queue.Queue(maxsize=60)

        self.stop_event = threading.Event()

        # Threads (created in _start_threads)
        self.capture_threads: list[CaptureThread] = []
        self.grouper: GrouperThread | None = None
        self.writer: WriterThread | None = None

        # Session directory
        self.session_dir: str = ""

        # State
        self.saving: bool = False
        self.save_single: bool = False
        self.displayed_groups: int = 0
        self.preview_interval: float = 1.0 / self.preview_fps

    # ---- setup ----

    def _create_session_dir(self) -> str:
        """Create a timestamped session directory under obs_data/.

        Returns:
            Absolute path to the session directory.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        obs_dir = os.path.join(base_dir, "obs_data")
        os.makedirs(obs_dir, exist_ok=True)
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        session_dir = os.path.join(obs_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def _start_threads(self) -> None:
        """Create and start all worker threads in dependency order."""
        # Create capture threads
        for cam_cfg in self.cam_configs:
            cam_id = cam_cfg["cam_id"]
            ct = CaptureThread(
                cam_cfg,
                self.capture_queues[cam_id],
                self.stop_event,
            )
            self.capture_threads.append(ct)

        # Create grouper (replaces the old PairerThread)
        self.grouper = GrouperThread(
            self.capture_queues,
            self.grouped_queue,
            self.sync_threshold_ms,
            self.stop_event,
        )

        # Create writer
        self.writer = WriterThread(
            self.write_queue,
            self.session_dir,
            self.cam_ids,
            self.stop_event,
        )

        # Start in dependency order
        for ct in self.capture_threads:
            ct.start()
        self.grouper.start()
        self.writer.start()

        # Brief pause for device init
        time.sleep(1.0)

    # ---- preview helpers ----

    def _build_preview_grid(self, images: list[np.ndarray]) -> np.ndarray:
        """Arrange decoded images into a grid for preview display.

        Layout rules:
            - 1-3 cameras: single row (horizontal stack).
            - 4+ cameras: 2-column grid, padded with black if odd count.

        All input images must have the same height (preview_height).

        Args:
            images: List of decoded BGR images, all resized to preview_height.

        Returns:
            Concatenated grid image ready for cv2.imshow.
        """
        n = len(images)
        if n == 0:
            return np.zeros((self.preview_height, 320, 3), dtype=np.uint8)

        if n <= 3:
            return np.hstack(images)

        # 2-column grid for 4+ cameras
        cols = 2
        rows_needed = (n + cols - 1) // cols

        # Pad with black images to fill the grid
        while len(images) < rows_needed * cols:
            images.append(np.zeros_like(images[0]))

        row_imgs: list[np.ndarray] = []
        for r in range(rows_needed):
            start = r * cols
            end = start + cols
            row_imgs.append(np.hstack(images[start:end]))
        return np.vstack(row_imgs)

    def _decode_and_preview(self, group: FrameGroup) -> None:
        """Decode MJPEG, resize, arrange grid, overlay status, and display.

        Args:
            group: The FrameGroup to display.
        """
        images: list[np.ndarray] = []
        for cam_id in self.cam_ids:
            frame = group.frames[cam_id]
            arr = np.frombuffer(frame.raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            # Fixed tile size keeps window layout stable across frames
            tile_w = self._preview_tile_width
            tile_h = self.preview_height
            canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

            if img is not None:
                # Letterbox: scale to fit tile while preserving aspect ratio
                h, w = img.shape[:2]
                scale = min(tile_w / w, tile_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv2.resize(img, (new_w, new_h))

                # Center on black canvas
                x_off = (tile_w - new_w) // 2
                y_off = (tile_h - new_h) // 2
                canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

            img = canvas

            # Overlay device identifier on each camera tile
            device_path: str = self._cam_config_map[cam_id]["device"]
            label = f"{cam_id} ({device_path})"
            cv2.putText(
                img,
                label,
                (10, img.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            images.append(img)

        if len(images) == 0:
            return

        concat = self._build_preview_grid(images)

        # Overlay status text
        if self.saving:
            save_str = "SAVING"
        else:
            save_str = "OFF"

        group_count = 0
        if self.grouper is not None:
            group_count = self.grouper.group_count

        written = 0
        if self.writer is not None:
            written = self.writer.written_count

        line1 = (
            f"Group: {group_count}  Sync: {group.sync_window_us:.0f}us  "
            f"Save: {save_str}  Written: {written}"
        )

        drop_parts: list[str] = []
        if self.grouper is not None:
            for cam_id in self.cam_ids:
                drop_parts.append(
                    f"{cam_id}: {self.grouper.drops.get(cam_id, 0)}"
                )
        line2 = f"Drops: {', '.join(drop_parts)}"

        cv2.putText(
            concat,
            line1,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            concat,
            line2,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Multi Capture (q:quit  c:save  s:single)", concat)

    # ---- main loop ----

    def run(self) -> None:
        """Run the full capture pipeline with preview in main thread."""
        self.session_dir = self._create_session_dir()
        print(f"[Session] {self.session_dir}")
        print(
            f"[Config] sync_threshold={self.sync_threshold_ms}ms  "
            f"preview={self.preview_fps}Hz"
        )
        print(f"[Config] cameras ({len(self.cam_ids)}): {', '.join(self.cam_ids)}")
        for cfg in self.cam_configs:
            print(f"  {cfg['cam_id']}: {cfg['device']}")
        print()

        self._start_threads()

        print("\nAll threads started. Controls:")
        print("  'c' - toggle continuous save")
        print("  's' - save single group")
        print("  'q' - quit\n")

        last_preview_time = 0.0
        last_status_time = time.monotonic()
        last_group_count = 0

        try:
            while not self.stop_event.is_set():
                # Drain grouped_queue — take the latest group for preview
                group: FrameGroup | None = None
                try:
                    while True:
                        group = self.grouped_queue.get_nowait()

                        # Route to writer if saving
                        if self.saving or self.save_single:
                            try:
                                self.write_queue.put_nowait(group)
                            except queue.Full:
                                pass
                            if self.save_single:
                                self.save_single = False
                                print(f"\n  [Saved single group #{group.group_id}]")

                except queue.Empty:
                    pass

                # Preview at target FPS
                now = time.monotonic()
                if (
                    group is not None
                    and (now - last_preview_time) >= self.preview_interval
                ):
                    self._decode_and_preview(group)
                    last_preview_time = now
                    self.displayed_groups += 1

                # Terminal status line (once per second)
                if now - last_status_time >= 1.0:
                    current_groups = 0
                    if self.grouper is not None:
                        current_groups = self.grouper.group_count
                    fps = current_groups - last_group_count
                    last_group_count = current_groups

                    drop_parts: list[str] = []
                    if self.grouper is not None:
                        for cam_id in self.cam_ids:
                            drop_parts.append(
                                f"{cam_id}:{self.grouper.drops.get(cam_id, 0)}"
                            )

                    written = 0
                    if self.writer is not None:
                        written = self.writer.written_count

                    sys.stdout.write(
                        f"\rGroups: {current_groups:6d} | "
                        f"FPS: {fps:3d} | "
                        f"Drops: {'/'.join(drop_parts)} | "
                        f"Written: {written:6d}   "
                    )
                    sys.stdout.flush()
                    last_status_time = now

                # Keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n\nQuitting...")
                    break
                elif key == ord("c"):
                    self.saving = not self.saving
                    if self.saving:
                        state = "ON"
                    else:
                        state = "OFF"
                    print(f"\n  [Continuous save: {state}]")
                elif key == ord("s"):
                    self.save_single = True

        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt...")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Signal all threads to stop, join them, and print summary."""
        print("\nCleaning up...")
        self.stop_event.set()

        all_threads: list[threading.Thread] = list(self.capture_threads)
        if self.grouper is not None:
            all_threads.append(self.grouper)
        if self.writer is not None:
            all_threads.append(self.writer)

        for t in all_threads:
            t.join(timeout=3.0)

        cv2.destroyAllWindows()

        # Summary
        group_count = 0
        if self.grouper is not None:
            group_count = self.grouper.group_count

        cap_parts: list[str] = []
        for ct in self.capture_threads:
            cap_parts.append(f"{ct.cam_id}: {ct.frame_count}")

        drop_parts: list[str] = []
        if self.grouper is not None:
            for cam_id in self.cam_ids:
                drop_parts.append(f"{cam_id}: {self.grouper.drops.get(cam_id, 0)}")

        written = 0
        if self.writer is not None:
            written = self.writer.written_count

        print("\n========== Summary ==========")
        print(f"  Captured:   {', '.join(cap_parts)}")
        print(f"  Grouped:    {group_count}")
        print(f"  Dropped:    {', '.join(drop_parts)}")
        print(f"  Written:    {written}")
        print(f"  Previewed:  {self.displayed_groups}")
        print(f"  Session:    {self.session_dir}")
        print("=============================\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace with config_path.
    """
    parser = argparse.ArgumentParser(
        description="Multi-camera external trigger capture system",
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=None,
        help="Path to JSON config file (default: built-in config)",
    )
    return parser.parse_args()


def main() -> None:
    """Application entry point."""
    args = parse_arguments()

    if args.config_path is not None:
        config = load_config(args.config_path)
    else:
        print("[Config] No config file specified, using defaults")
        config = DEFAULT_CONFIG.copy()

    capture = MultiCameraCapture(config)
    capture.run()


if __name__ == "__main__":
    main()
