#!/usr/bin/env python3
"""
Scan subdirectories under this result folder, find depth_std.npy in each task,
and compute per-pixel std distribution in millimeters using 4 fixed groups.

Also render mask-only std heatmaps for 6 tasks (matplotlib).
Output: depth_stats.json with per-task group counts and ratios.
"""
import os
import sys
import json
import csv
from glob import glob

try:
    import numpy as np
except Exception:
    print('numpy required', file=sys.stderr)
    raise

try:
    import cv2
except Exception:
    print('opencv-python required', file=sys.stderr)
    raise

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    print('matplotlib required', file=sys.stderr)
    raise


ROOT = os.path.dirname(os.path.abspath(__file__))
GROUPS_MM = [
    (0.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
    (20.0, 50.0),
    (50.0, 100.0),
    (100.0, None),
]

MASK_JSON_PATH = "/home/user/Desktop/main/main2/20260508/project-5-at-2026-05-08-09-02-e9719ff4.json"
HEATMAP_OUTPUT_DIR = "std_heatmaps"
HEATMAP_CMAP = "turbo"
HEATMAP_VMIN_MM = 0.0
HEATMAP_VMAX_MM = None
HEATMAP_MAX_PERCENTILE = 99.0


def load_depth_std(path):
    try:
        return np.load(path)
    except Exception:
        return None


def find_depth_std_file(task_dir):
    candidates = glob(os.path.join(task_dir, '**', 'depth_std*.npy'), recursive=True)
    if not candidates:
        return None
    exact = [p for p in candidates if os.path.basename(p) == 'depth_std.npy']
    return sorted(exact)[0] if exact else sorted(candidates)[0]


def find_depth_mean_file(task_dir):
    candidates = glob(os.path.join(task_dir, '**', 'depth_mean*.npy'), recursive=True)
    if not candidates:
        return None
    exact = [p for p in candidates if os.path.basename(p) == 'depth_mean.npy']
    return sorted(exact)[0] if exact else sorted(candidates)[0]


def compute_mm_groups(values_mm):
    vals = np.asarray(values_mm, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals >= 0]
    if vals.size == 0:
        return []
    total = int(vals.size)
    groups = []
    for start_mm, end_mm in GROUPS_MM:
        if end_mm is None:
            mask = vals >= start_mm
            range_label = f">={int(start_mm)}mm"
        else:
            mask = (vals >= start_mm) & (vals < end_mm)
            range_label = f"{int(start_mm)}-{int(end_mm)}mm"
        count = int(np.count_nonzero(mask))
        ratio = count / total if total > 0 else 0.0
        groups.append({
            "range_mm": range_label,
            "count": count,
            "ratio": ratio,
        })
    return groups


def _read_mean_frame_std(task_dir):
    csv_path = os.path.join(task_dir, "per_image_depth_stats.csv")
    if not os.path.exists(csv_path):
        return None, 0
    vals = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            v = row.get("std_depth")
            if v is None or v == "":
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
    if not vals:
        return None, 0
    return float(np.mean(vals)), int(len(vals))


def _load_task_masks(mask_json_path):
    if not os.path.exists(mask_json_path):
        raise FileNotFoundError(f"Mask json not found: {mask_json_path}")
    data = json.loads(open(mask_json_path, "r", encoding="utf-8").read())
    masks = {}
    for item in data:
        image_name = None
        if isinstance(item, dict):
            image_name = item.get("data", {}).get("image") or item.get("file_upload")
        if not image_name:
            continue
        image_stem = os.path.splitext(os.path.basename(image_name))[0]
        mask = None
        for ann in item.get("annotations", []):
            for res in ann.get("result", []):
                if res.get("type") != "polygonlabels":
                    continue
                w = int(res.get("original_width", 0))
                h = int(res.get("original_height", 0))
                points = res.get("value", {}).get("points", [])
                if w <= 0 or h <= 0 or not points:
                    continue
                if mask is None:
                    mask = np.zeros((h, w), dtype=np.uint8)
                pts = np.array(
                    [[[p[0] / 100.0 * w, p[1] / 100.0 * h]] for p in points],
                    dtype=np.int32,
                )
                cv2.fillPoly(mask, [pts], 1)
        if mask is not None:
            masks[image_stem] = mask.astype(bool)
    return masks


def _match_mask(mask, target_shape):
    if mask.shape == target_shape:
        return mask
    resized = cv2.resize(mask.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _save_heatmap(std_mm, mask, out_path, title=None):
    h, w = std_mm.shape[:2]
    mask_use = _match_mask(mask, (h, w))
    vals = np.where(mask_use & np.isfinite(std_mm) & (std_mm >= 0), std_mm, np.nan)
    data = np.ma.array(vals, mask=~mask_use | ~np.isfinite(vals))
    vmin = HEATMAP_VMIN_MM
    vmax = HEATMAP_VMAX_MM
    if vmax is None:
        valid = data.compressed()
        if valid.size > 0:
            vmax = float(np.nanpercentile(valid, HEATMAP_MAX_PERCENTILE))
        else:
            vmax = 1.0

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    im = ax.imshow(data, cmap=HEATMAP_CMAP, vmin=vmin, vmax=vmax)
    ax.axis("off")
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("std (mm)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def _make_grid(images, cols):
    if not images:
        return None
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    padded = []
    for img in images:
        h, w = img.shape[:2]
        top = 0
        bottom = max_h - h
        left = 0
        right = max_w - w
        padded.append(cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    rows = []
    for i in range(0, len(padded), cols):
        row_imgs = padded[i:i + cols]
        if len(row_imgs) < cols:
            blank = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            row_imgs += [blank] * (cols - len(row_imgs))
        rows.append(cv2.hconcat(row_imgs))
    return cv2.vconcat(rows)


def main():
    masks = _load_task_masks(MASK_JSON_PATH)
    tasks = [d for d in sorted(os.listdir(ROOT)) if os.path.isdir(os.path.join(ROOT, d))]
    result = {}
    heatmap_dir = os.path.join(ROOT, HEATMAP_OUTPUT_DIR)
    os.makedirs(heatmap_dir, exist_ok=True)
    for task in tasks:
        task_dir = os.path.join(ROOT, task)
        depth_std_path = find_depth_std_file(task_dir)
        if depth_std_path is None:
            continue
        depth_mean_path = find_depth_mean_file(task_dir)
        arr = load_depth_std(depth_std_path)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 2:
            arr = arr[..., 0]
        if task in masks:
            std_mm_img = arr * 1000.0
            out_img_path = os.path.join(heatmap_dir, f"{task}_std.png")
            _save_heatmap(std_mm_img, masks[task], out_img_path, title=task)
            if depth_mean_path is not None:
                mean_arr = load_depth_std(depth_mean_path)
                if mean_arr is not None:
                    mean_arr = np.asarray(mean_arr, dtype=np.float32)
                    if mean_arr.ndim > 2:
                        mean_arr = mean_arr[..., 0]
                    mean_mm_img = mean_arr * 1000.0
                    mean_out = os.path.join(heatmap_dir, f"{task}_mean.png")
                    _save_heatmap(mean_mm_img, masks[task], mean_out, title=f"{task} mean")
        vals = arr.ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        vals_mm = vals * 1000.0
        groups = compute_mm_groups(vals_mm)
        mean_std, std_count = _read_mean_frame_std(task_dir)
        result[task] = {
            'source': os.path.relpath(depth_std_path, ROOT),
            'count': int(vals_mm.size),
            'groups': groups,
            'mean_frame_std': mean_std,
            'mean_frame_std_count': std_count,
        }


    out_path = os.path.join(ROOT, 'depth_stats.json')
    with open(out_path, 'w') as fout:
        json.dump(result, fout, indent=2)
    print(out_path)


if __name__ == '__main__':
    main()
