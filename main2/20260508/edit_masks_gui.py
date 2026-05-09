#!/usr/bin/env python3
"""Interactive mask editor for Label Studio polygon annotations.

Load the project JSON, overlay each task mask on a representative image, let the
user refine the mask edge with the mouse, and save the updated polygon points
back to JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


SUPPORTED_TASKS = [
    "farpose_left_chest_origin",
    "farpose_right_chest_origin",
    "nearpose_left_chest_origin",
    "nearpose_right_chest_origin",
    "waitpose_left_chest_origin",
    "waitpose_right_chest_origin",
]


# Edit these values directly before running the script.
JSON_PATH = "/home/user/Desktop/main/main2/20260508/project-5-at-2026-05-08-09-02-e9719ff4.json"
OUTPUT_JSON_PATH = None
TASK_NAME = "waitpose_right_chest_origin"
IMAGE_DIR = "/home/user/Desktop/main/main2/20260508/rgb_new/waitpose_right_chest_origin"
MAX_FRAMES = 20


def polygon_percent_to_pixels(points: Iterable[Iterable[float]], width: int, height: int) -> np.ndarray:
    pixels = []
    for x_pct, y_pct in points:
        pixels.append([
            int(round(float(x_pct) / 100.0 * width)),
            int(round(float(y_pct) / 100.0 * height)),
        ])
    return np.asarray(pixels, dtype=np.int32)


def polygon_pixels_to_percent(points: np.ndarray, width: int, height: int) -> list[list[float]]:
    percent_points: list[list[float]] = []
    for x, y in points:
        x_pct = float(x) / float(width) * 100.0
        y_pct = float(y) / float(height) * 100.0
        percent_points.append([x_pct, y_pct])
    return percent_points


def mask_from_points(points: Iterable[Iterable[float]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = polygon_percent_to_pixels(points, width, height)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_to_polygon(mask: np.ndarray, min_area: float = 25.0) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.int32)

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return np.empty((0, 2), dtype=np.int32)

    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.5, 0.004 * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approx = approx.reshape(-1, 2)
    if approx.shape[0] < 3:
        approx = contour.reshape(-1, 2)
    return approx.astype(np.int32)


def load_image_path(image_path: Path) -> Path:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image is not a file: {image_path}")
    return image_path


def load_image_paths(image_dir: Path, limit: int) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image path is not a directory: {image_dir}")

    preferred_exts = [".png", ".jpg", ".jpeg", ".bmp"]
    candidates: list[Path] = []
    for ext in preferred_exts:
        candidates.extend(sorted(image_dir.glob(f"*{ext}")))
    if not candidates:
        candidates = sorted([p for p in image_dir.iterdir() if p.is_file()])
    if not candidates:
        raise FileNotFoundError(f"No image files found in {image_dir}")
    return candidates[:limit]


def extract_task_name(entry: dict) -> str | None:
    fname = entry.get("file_upload") or entry.get("data", {}).get("image")
    if not fname:
        return None
    return Path(fname).stem


def get_first_polygon_result(entry: dict) -> dict | None:
    annotations = entry.get("annotations", [])
    for ann in annotations:
        for result in ann.get("result", []):
            if result.get("type") == "polygonlabels":
                return result
    return None


def update_entry_from_mask(entry: dict, mask: np.ndarray, width: int, height: int) -> bool:
    polygon = mask_to_polygon(mask)
    if polygon.shape[0] < 3:
        return False

    result = get_first_polygon_result(entry)
    if result is None:
        return False

    result.setdefault("value", {})["points"] = polygon_pixels_to_percent(polygon, width, height)
    result["value"]["closed"] = True
    result["original_width"] = width
    result["original_height"] = height
    return True


def make_overlay(image: np.ndarray, file_mask: np.ndarray, new_mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()

    file_tint = np.zeros_like(overlay)
    file_tint[..., 1] = 255
    file_alpha = 0.20
    overlay = np.where(file_mask[..., None] > 0, (overlay * (1.0 - file_alpha) + file_tint * file_alpha), overlay).astype(np.uint8)

    new_tint = np.zeros_like(overlay)
    new_tint[..., 2] = 255
    new_alpha = 0.45
    overlay = np.where(new_mask[..., None] > 0, (overlay * (1.0 - new_alpha) + new_tint * new_alpha), overlay).astype(np.uint8)

    file_contours, _ = cv2.findContours((file_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_contours, _ = cv2.findContours((new_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, file_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(overlay, new_contours, -1, (0, 0, 255), 2)
    return overlay


def draw_text_block(image: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = image.copy()
    y = 24
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        y += 24
    return canvas


class MaskEditor:
    def __init__(self, image: np.ndarray, initial_mask: np.ndarray, title: str) -> None:
        self.image = image
        self.base_mask = initial_mask.copy()
        self.mask = initial_mask.copy()
        self.title = title
        self.brush_radius = 10
        self.mode: str | None = None
        self.undo_stack: list[np.ndarray] = []
        self.dirty = False
        self.last_message = ""

    def set_frame(self, image: np.ndarray, title: str) -> None:
        self.image = image
        self.title = title

    def push_undo(self) -> None:
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)

    def undo(self) -> None:
        if self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.dirty = True
            self.last_message = "undo"

    def reset(self) -> None:
        self.mask = self.base_mask.copy()
        self.dirty = True
        self.last_message = "reset"

    def apply_brush(self, x: int, y: int, value: int) -> None:
        cv2.circle(self.mask, (x, y), self.brush_radius, int(value), -1)
        self.dirty = True

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:  # noqa: ANN001
        if event == cv2.EVENT_LBUTTONDOWN:
            self.push_undo()
            self.mode = "add"
            self.apply_brush(x, y, 1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.push_undo()
            self.mode = "erase"
            self.apply_brush(x, y, 0)
        elif event == cv2.EVENT_MOUSEMOVE and self.mode == "add" and (flags & cv2.EVENT_FLAG_LBUTTON):
            self.apply_brush(x, y, 1)
        elif event == cv2.EVENT_MOUSEMOVE and self.mode == "erase" and (flags & cv2.EVENT_FLAG_RBUTTON):
            self.apply_brush(x, y, 0)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.mode = None

    def render(self) -> np.ndarray:
        overlay = make_overlay(self.image, self.base_mask, self.mask)
        delta_pixels = int(np.count_nonzero(self.mask != self.base_mask))
        info = [
            self.title,
            f"Brush: {self.brush_radius}  |  file mask: {int(np.count_nonzero(self.base_mask))}  |  new mask: {int(np.count_nonzero(self.mask))}  |  delta: {delta_pixels}",
            "Left-drag: add  Right-drag: erase  z: undo  r: reset  [ ]: brush  s: save  n/p: next/prev  q/ESC: quit",
        ]
        if self.last_message:
            info.append(f"Last: {self.last_message}")
        return draw_text_block(overlay, info)


def choose_task_entries(data: list[dict], selected_tasks: list[str] | None) -> list[tuple[int, dict, str]]:
    task_map: dict[str, tuple[int, dict, str]] = {}
    for index, entry in enumerate(data):
        task_name = extract_task_name(entry)
        if task_name in SUPPORTED_TASKS:
            task_map[task_name] = (index, entry, task_name)

    ordered_tasks = selected_tasks or SUPPORTED_TASKS
    result: list[tuple[int, dict, str]] = []
    for task_name in ordered_tasks:
        if task_name in task_map:
            result.append(task_map[task_name])
    return result


def save_json(path: Path, data: list[dict]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def edit_task(index: int, entry: dict, task_name: str, image_paths: list[Path], output_json: Path, data: list[dict]) -> str:
    first_image_path = load_image_path(image_paths[0])
    first_image = cv2.imread(str(first_image_path), cv2.IMREAD_COLOR)
    if first_image is None:
        raise RuntimeError(f"Failed to read image: {first_image_path}")

    height, width = first_image.shape[:2]
    result = get_first_polygon_result(entry)
    if result is None:
        raise RuntimeError(f"No polygon annotation found for task: {task_name}")

    points = result.get("value", {}).get("points", [])
    initial_mask = mask_from_points(points, width, height)

    editor = MaskEditor(image=first_image, initial_mask=initial_mask, title=f"[{index + 1}] {task_name}  ({first_image_path.name})")
    window_name = "mask-editor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1400, width), min(1000, height))
    cv2.setMouseCallback(window_name, editor.on_mouse)

    action = "next"
    frame_index = 0
    dirty_to_save = False
    while True:
        current_image_path = load_image_path(image_paths[frame_index])
        current_image = cv2.imread(str(current_image_path), cv2.IMREAD_COLOR)
        if current_image is None:
            raise RuntimeError(f"Failed to read image: {current_image_path}")

        if current_image.shape[:2] != (height, width):
            current_image = cv2.resize(current_image, (width, height), interpolation=cv2.INTER_LINEAR)

        editor.set_frame(current_image, f"[{index + 1}] {task_name}  ({frame_index + 1}/{len(image_paths)}: {current_image_path.name})")
        cv2.imshow(window_name, editor.render())
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            action = "quit"
            break
        if key == ord("s"):
            if update_entry_from_mask(entry, editor.mask, width, height):
                editor.dirty = False
                editor.last_message = f"saved to polygon with {int(np.count_nonzero(editor.mask))} pixels"
                save_json(output_json, data)
            else:
                editor.last_message = "save failed: no valid contour"
        elif key == ord("n"):
            if frame_index < len(image_paths) - 1:
                frame_index += 1
            else:
                action = "next"
                break
        elif key == ord("p"):
            if frame_index > 0:
                frame_index -= 1
            else:
                action = "prev"
                break
        elif key == ord("r"):
            editor.reset()
            dirty_to_save = True
        elif key == ord("z"):
            editor.undo()
            dirty_to_save = True
        elif key == ord("["):
            editor.brush_radius = max(1, editor.brush_radius - 1)
            editor.last_message = f"brush={editor.brush_radius}"
        elif key == ord("]"):
            editor.brush_radius += 1
            editor.last_message = f"brush={editor.brush_radius}"

        if editor.dirty:
            dirty_to_save = True
        if dirty_to_save:
            if update_entry_from_mask(entry, editor.mask, width, height):
                save_json(output_json, data)
                editor.dirty = False
                dirty_to_save = False
                editor.last_message = "auto-saved"
            else:
                editor.last_message = "auto-save failed: no valid contour"

    cv2.destroyWindow(window_name)

    if editor.dirty:
        if update_entry_from_mask(entry, editor.mask, width, height):
            editor.dirty = False
            save_json(output_json, data)
        else:
            raise RuntimeError(f"Cannot convert mask to polygon for task: {task_name}")

    if action in {"next", "prev"}:
        editor.last_message = "auto-saved on navigation"
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Label Studio mask editor")
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Label Studio JSON path; defaults to JSON_PATH in the script",
    )
    parser.add_argument(
        "--image",
        dest="image_path",
        default=None,
        help="Image file to display and edit; defaults to IMAGE_PATH in the script",
    )
    parser.add_argument(
        "--image-dir",
        dest="image_dir",
        default=None,
        help="Image directory to browse; defaults to IMAGE_DIR in the script",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Where to save the updated JSON; defaults to OUTPUT_JSON_PATH or JSON_PATH",
    )
    parser.add_argument(
        "--task",
        default=None,
        choices=SUPPORTED_TASKS,
        help="Task name to edit; defaults to TASK_NAME in the script",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path or JSON_PATH).expanduser().resolve()
    output_json_value = args.output_json if args.output_json is not None else OUTPUT_JSON_PATH
    output_json = Path(output_json_value).expanduser().resolve() if output_json_value else json_path
    image_dir_value = args.image_dir or IMAGE_DIR
    image_path_value = args.image_path
    task_name = args.task or TASK_NAME

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of Label Studio tasks")

    task_entries = choose_task_entries(data, [task_name])
    if not task_entries:
        raise RuntimeError(f"Task not found in JSON: {task_name}")

    item_index, entry, task_name = task_entries[0]
    if image_path_value:
        image_paths = [Path(image_path_value).expanduser().resolve()]
    else:
        image_dir = Path(image_dir_value).expanduser().resolve()
        image_paths = load_image_paths(image_dir, MAX_FRAMES)

    edit_task(item_index, entry, task_name, image_paths, output_json, data)

    save_json(output_json, data)
    print(f"Saved updated JSON to {output_json}")


if __name__ == "__main__":
    main()