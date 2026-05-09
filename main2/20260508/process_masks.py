#!/usr/bin/env python3
"""Process masks (Label Studio JSON), apply erosion, mask RGB/depth, and compute consistency.

Saves outputs into a `res` folder next to the dataset.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import cv2

try:
    import tifffile
    _read_tiff = lambda p: tifffile.imread(p)
except Exception:
    import imageio
    _read_tiff = lambda p: imageio.imread(p)

# =========================
# Global config (edit here)
# =========================
# Set to None to process all tasks, or specify a task name to process only that one
# Available tasks: farpose_left_chest_origin, farpose_right_chest_origin, 
#                  nearpose_left_chest_origin, nearpose_right_chest_origin,
#                  waitpose_left_chest_origin, waitpose_right_chest_origin
TASK_NAME = "waitpose_right_chest_origin"

def polygon_percent_to_pixels(points, width, height):
    pts = []
    for x_pct, y_pct in points:
        x = float(x_pct) / 100.0 * width
        y = float(y_pct) / 100.0 * height
        pts.append([int(round(x)), int(round(y))])
    return np.array(pts, dtype=np.int32)

def mask_from_annotation(entry, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    annotations = entry.get('annotations', [])
    for ann in annotations:
        for res in ann.get('result', []):
            val = res.get('value', {})
            pts = val.get('points') or []
            if not pts:
                continue
            poly = polygon_percent_to_pixels(pts, width, height)
            cv2.fillPoly(mask, [poly], 1)
    return mask

def erode_mask(mask, ksize=5, iters=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=iters)

def apply_mask_to_rgb(rgb, mask):
    out = rgb.copy()
    if out.ndim == 2:
        out[mask == 0] = 0
    else:
        out[mask == 0] = 0
    return out

def apply_mask_to_depth(depth, mask):
    # Keep original dtype, set invalid pixels to 0
    d = depth.copy()
    d[mask == 0] = 0
    return d

def save_overlay(rgb, mask, out_path):
    overlay = rgb.copy()
    # color mask red with alpha blend
    colored = np.zeros_like(overlay)
    colored[..., 2] = (mask * 255).astype(np.uint8)
    alpha = 0.5
    cv2.addWeighted(colored, alpha, overlay, 1 - alpha, 0, overlay)
    cv2.imwrite(str(out_path), overlay)

def compute_frame_stats(depth_masked):
    valid = depth_masked != 0
    vals = depth_masked[valid]
    if vals.size == 0:
        return {'count': 0, 'mean': None, 'median': None, 'std': None}
    vals = vals.astype(np.float64)
    return {'count': int(vals.size), 'mean': float(np.mean(vals)), 'median': float(np.median(vals)), 'std': float(np.std(vals))}

def main(base_dir: Path):
    base_dir = Path(base_dir)
    res_dir = base_dir / 'res'
    res_dir.mkdir(exist_ok=True)

    # find json
    json_files = list(base_dir.glob('*.json'))
    if not json_files:
        print('No JSON mask files found in', base_dir)
        return
    json_path = json_files[0]
    print('Using', json_path)
    data = json.loads(json_path.read_text())

    rgb_dir = base_dir / 'rgb_new'
    depth_dir = base_dir / 'depth_all_new'

    per_frame_stats = []
    masks_rgb_list = []
    masks_depth_list = []
    depth_masked_list = []

    for entry in data:
        fname = entry.get('file_upload') or entry.get('data', {}).get('image')
        if not fname:
            continue
        # fname is like "farpose_left_chest_origin.png"; extract task name
        task_name = Path(fname).stem  # e.g., "farpose_left_chest_origin"
        
        # Skip if TASK_NAME is specified and doesn't match
        if TASK_NAME is not None and task_name != TASK_NAME:
            continue
        
        # Find all frame files for this task
        task_rgb_dir = rgb_dir / task_name
        task_depth_dir = depth_dir / task_name
        
        if not task_rgb_dir.exists():
            print(f'Task RGB dir not found: {task_rgb_dir}')
            continue
        if not task_depth_dir.exists():
            print(f'Task depth dir not found: {task_depth_dir}')
            continue

        # Create output directories for this task
        task_res_dir = res_dir / task_name
        rgb_res_dir = task_res_dir / 'rgb_masked'
        overlay_res_dir = task_res_dir / 'overlay'
        depth_masked_npy_dir = task_res_dir / 'depth_masked_npy'
        depth_masked_tiff_dir = task_res_dir / 'depth_masked_tiff'
        stats_res_dir = task_res_dir / 'stats'
        for out_dir in (rgb_res_dir, overlay_res_dir, depth_masked_npy_dir, depth_masked_tiff_dir, stats_res_dir):
            out_dir.mkdir(parents=True, exist_ok=True)

        # Collect all frame files for this task
        rgb_frames = sorted([p for p in task_rgb_dir.iterdir() if p.is_file()])
        if not rgb_frames:
            print(f'No RGB frames found in {task_rgb_dir}')
            continue

        # Use first frame to build mask (assume mask is same for all frames)
        rgb_path = rgb_frames[0]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            print('Failed to read rgb', rgb_path)
            continue
        h, w = rgb.shape[:2]

        # Build mask from annotation (Label Studio percent points)
        # Use original_width/height if present
        ann_result = None
        anns = entry.get('annotations', [])
        if anns:
            res = anns[0].get('result', [])
            if res:
                ann_result = res[0]
        orig_w = ann_result.get('original_width') if ann_result else w
        orig_h = ann_result.get('original_height') if ann_result else h

        mask = mask_from_annotation(entry, orig_w, orig_h)
        # If JSON used different size, rescale mask to actual image size
        if (orig_w, orig_h) != (w, h):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        mask_eroded = erode_mask(mask, ksize=5, iters=1)

        # Process all frames for this task
        depth_frames = sorted([p for p in task_depth_dir.iterdir() if p.is_file()])
        if not depth_frames:
            print(f'No depth frames found in {task_depth_dir}')
            continue

        task_masks_rgb_list = []
        task_masks_depth_list = []
        task_depth_masked_list = []

        for frame_idx, (rgb_path, depth_path) in enumerate(zip(rgb_frames, depth_frames)):
            # Process RGB frame
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb is None:
                print(f'Failed to read rgb: {rgb_path}')
                continue

            masked_rgb = apply_mask_to_rgb(rgb, mask_eroded)
            frame_name = rgb_path.stem  # e.g., "000000"
            out_rgb_path = rgb_res_dir / (f'{task_name}_{frame_name}_masked_rgb.jpg')
            cv2.imwrite(str(out_rgb_path), masked_rgb)

            overlay_path = overlay_res_dir / (f'{task_name}_{frame_name}_overlay.jpg')
            save_overlay(rgb, mask_eroded, overlay_path)

            # Process depth frame
            depth = _read_tiff(str(depth_path))
            if depth is None:
                print(f'Failed to read depth: {depth_path}')
                continue
            if depth.ndim == 3:
                depth_arr = depth[..., 0]
            else:
                depth_arr = depth

            dh, dw = depth_arr.shape[:2]
            # resize mask to depth resolution if needed
            if (mask_eroded.shape[0], mask_eroded.shape[1]) != (dh, dw):
                mask_for_depth = cv2.resize(mask_eroded.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)
            else:
                mask_for_depth = mask_eroded

            depth_masked = apply_mask_to_depth(depth_arr, mask_for_depth)
            # save masked depth as npy and tiff separately
            np.save(depth_masked_npy_dir / (f'{task_name}_{frame_name}_masked_depth.npy'), depth_masked)
            try:
                tifffile.imwrite(str(depth_masked_tiff_dir / (f'{task_name}_{frame_name}_masked_depth.tiff')), depth_masked)
            except Exception:
                pass

            stats = compute_frame_stats(depth_masked)
            stats.update({'file': f'{task_name}/{frame_name}'})
            per_frame_stats.append(stats)

            task_masks_rgb_list.append(mask_eroded)
            task_masks_depth_list.append(mask_for_depth)
            task_depth_masked_list.append(depth_masked)

            # check if there's any non-masked content left in the RGB
            outside = rgb.copy()
            outside[mask_eroded == 1] = 0
            nonzero_outside = int(np.count_nonzero(outside))
            if nonzero_outside == 0:
                note = 'no other visible content outside mask'
            else:
                note = f'{nonzero_outside} nonzero pixels outside mask'
            print(f'{task_name}/{frame_name}: {note}')

        # Add this task's data to overall lists
        masks_rgb_list.extend(task_masks_rgb_list)
        masks_depth_list.extend(task_masks_depth_list)
        depth_masked_list.extend(task_depth_masked_list)

    # Multi-frame consistency
    summary = {}
    if depth_masked_list:
        # compute stats of per-frame means
        means = [s['mean'] for s in per_frame_stats if s['mean'] is not None]
        summary['per_frame_means'] = means
        if means:
            summary['mean_of_means'] = float(np.mean(means))
            summary['std_of_means'] = float(np.std(means))

        # pixel-wise intersection mask (use masks at depth resolution)
        common_mask = np.logical_and.reduce([m.astype(bool) for m in masks_depth_list])
        if np.count_nonzero(common_mask) > 0:
            # stack depths where common_mask True; replace zero with nan for proper stats
            stack = np.stack([d.astype(np.float64) for d in depth_masked_list], axis=0)
            stack[stack == 0] = np.nan
            common_pixels = stack[:, common_mask]
            pixel_std = np.nanstd(common_pixels, axis=0)
            summary['common_mask_count'] = int(np.count_nonzero(common_mask))
            summary['pixel_std_mean'] = float(np.nanmean(pixel_std))
            # fraction of pixels with small std (tolerance set in absolute units of depth values)
            tol = 5.0
            consistent_frac = float(np.sum(pixel_std < tol) / pixel_std.size)
            summary['consistent_fraction_tol_5'] = consistent_frac
        else:
            summary['common_mask_count'] = 0

    # save stats
    import csv
    global_stats_dir = res_dir / 'stats'
    global_stats_dir.mkdir(parents=True, exist_ok=True)
    csv_path = global_stats_dir / 'per_frame_depth_stats.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'count', 'mean', 'median', 'std'])
        writer.writeheader()
        for s in per_frame_stats:
            writer.writerow({k: s.get(k) for k in ['file', 'count', 'mean', 'median', 'std']})

    json_path = global_stats_dir / 'multi_frame_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('Done. Results in', res_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).parent
    parser.add_argument('base_dir', nargs='?', default=str(script_dir), help='dataset directory (contains rgb/, depth/, and json)')
    args = parser.parse_args()
    main(Path(args.base_dir))
