#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICP 对齐脚本：将 depth_mean.npy 点云与 STL 模型对齐。

每次运行前在下方修改全局 task_name，仅处理该任务。
仅生成 <本脚本目录>/0512/<task_name>.csv：每行一组多起点 ICP。列为「搜索起始」三轴角（度，与网格初值一致）、「收敛结束」三轴角（度，由 fine ICP 最终 4×4 变换的旋转部分按与 build_xyz_rotation 相同的 Rx→Ry→Rz 约定分解，一般非 90° 倍数）、最终平移 x/y/z（米）、覆盖程度、loss；按覆盖程度降序排列。额外保存覆盖率最高的 4 张 RGB 叠加图到 <本脚本目录>/0512。终端仅输出 CSV 路径，异常走 stderr。

可选：将 ICP 最佳变换作用到 STL 点云并投影到 RGB 图像上，结果保存到 base-dir。
"""

from pathlib import Path
import argparse
import csv
import itertools
import sys
import numpy as np

# 每次运行前修改为当前要处理的任务名称（仅此任务参与 ICP）
task_name_list = ["near_left_chest_origin",
                  "near_right_chest_origin",
                  "far_left_chest_origin",
                  "far_right_chest_origin"]
task_name = task_name_list[0]

# 逻辑任务名 -> result/ 下实际文件夹名（与 visualize 生成的目录一致）
RESULT_TASK_ALIASES = {
    "near_left_chest_origin": "nearpose_left_chest_origin",
    "near_right_chest_origin": "nearpose_right_chest_origin",
    "far_left_chest_origin": "farpose_left_chest_origin",
    "far_right_chest_origin": "farpose_right_chest_origin",
    "wait_left_chest_origin": "waitpose_left_chest_origin",
    "wait_right_chest_origin": "waitpose_right_chest_origin",
}

try:
    import open3d as o3d
except Exception as e:
    print("Error: open3d is required for this script. Install with `pip install open3d`.")
    raise

try:
    import cv2
except Exception:
    cv2 = None

RGB_ORIGINAL_SHAPE = (800, 1280)
CAMERA_INTRINSICS = {
    "left": Path(__file__).resolve().parent.parent / "aililight_cameras" / "chest_left_camera.yaml",
    "right": Path(__file__).resolve().parent.parent / "aililight_cameras" / "chest_right_camera.yaml",
}

def parse_intrinsics_from_text(text):
    lines = [line.strip() for line in text.splitlines()]
    start = None
    for index, line in enumerate(lines):
        if line.startswith("intrinsic:"):
            start = index + 1
            break
    if start is None:
        raise ValueError("Missing intrinsic section in camera YAML.")

    rows = []
    for line in lines[start:]:
        if line.startswith("- [") and line.endswith("]"):
            row_text = line[3:-1]
            row = [float(value.strip()) for value in row_text.split(",")]
            rows.append(row)
            if len(rows) == 3:
                break
    if len(rows) != 3:
        raise ValueError("Failed to parse 3x3 intrinsic matrix.")
    return np.array(rows, dtype=np.float64)


def load_intrinsics(yaml_path):
    if not yaml_path.exists():
        raise FileNotFoundError(f"Camera intrinsics file not found: {yaml_path}")
    text = yaml_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        matrix = np.array(data["intrinsic"], dtype=np.float64)
    except Exception:
        matrix = parse_intrinsics_from_text(text)
    if matrix.shape != (3, 3):
        raise ValueError(f"Invalid intrinsic matrix shape: {matrix.shape}")
    return matrix


def center_crop_intrinsics(k, source_shape, crop_shape):
    source_height, source_width = source_shape
    crop_height, crop_width = crop_shape
    top = (source_height - crop_height) // 2
    left = (source_width - crop_width) // 2
    adjusted = np.array(k, dtype=np.float64, copy=True)
    adjusted[0, 2] -= left
    adjusted[1, 2] -= top
    return adjusted


def load_rgb_image(rgb_path: Path) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for RGB overlay.")
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read RGB image: {rgb_path}")
    return image


def depth_to_colors(z_values: np.ndarray) -> np.ndarray:
    if z_values.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    lower = float(np.percentile(z_values, 5))
    upper = float(np.percentile(z_values, 95))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return np.full((z_values.shape[0], 3), 0.8, dtype=np.float64)
    normalized = np.clip((z_values - lower) / (upper - lower), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    return np.stack((red, green, blue), axis=1)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape((0, 3))
    homo = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    out = (np.asarray(transform, dtype=np.float64) @ homo.T).T
    return out[:, :3]


def project_points(points: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=bool)
    z = pts[:, 2]
    valid = np.isfinite(z) & (z > 0.0)
    if not np.any(valid):
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64), valid
    x = pts[valid, 0]
    y = pts[valid, 1]
    z = z[valid]
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    u = fx * x / z + cx
    v = fy * y / z + cy
    return u, v, valid


def overlay_points_on_image(image: np.ndarray, u: np.ndarray, v: np.ndarray, colors: np.ndarray, alpha: float) -> tuple[np.ndarray, int]:
    overlay = image.copy()
    if u.size == 0:
        return overlay, 0
    h, w = overlay.shape[:2]
    u_i = np.rint(u).astype(np.int64)
    v_i = np.rint(v).astype(np.int64)
    in_bounds = (u_i >= 0) & (u_i < w) & (v_i >= 0) & (v_i < h)
    if not np.any(in_bounds):
        return overlay, 0
    u_i = u_i[in_bounds]
    v_i = v_i[in_bounds]
    colors = colors[in_bounds]
    base = overlay[v_i, u_i].astype(np.float64)
    blended = (1.0 - alpha) * base + alpha * colors
    overlay[v_i, u_i] = blended.astype(np.uint8)
    return overlay, int(u_i.size)


def resolve_rgb_path(base: Path, pattern: str, task: str, task_fs: str) -> Path:
    candidates = []
    if "{task}" in pattern:
        candidates.append(pattern.format(task=task, task_fs=task_fs))
        if task_fs != task:
            candidates.append(pattern.format(task=task_fs, task_fs=task_fs))
    else:
        candidates.append(pattern.format(task=task, task_fs=task_fs))
    for entry in candidates:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = base / candidate
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"RGB image not found for pattern: {pattern}")


def resolve_overlay_path(base: Path, pattern: str, task: str, task_fs: str, rank: int) -> Path:
    rendered = pattern.format(task=task, task_fs=task_fs, rank=rank)
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate


def save_top_overlay_images(
    base: Path,
    task: str,
    task_fs: str,
    source_points: np.ndarray,
    sorted_candidates: list[dict],
    rgb_pattern: str,
    overlay_pattern: str,
    overlay_alpha: float,
) -> list[Path]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for RGB overlay.")

    rgb_path = resolve_rgb_path(base, rgb_pattern, task, task_fs)
    rgb = load_rgb_image(rgb_path)
    camera_key = choose_camera(task)
    k_rgb = center_crop_intrinsics(
        load_intrinsics(CAMERA_INTRINSICS[camera_key]),
        RGB_ORIGINAL_SHAPE,
        rgb.shape[:2],
    )

    saved_paths: list[Path] = []
    for rank, candidate in enumerate(sorted_candidates[:4], start=1):
        transformed = transform_points(source_points, candidate["T"])
        u, v, valid = project_points(transformed, k_rgb)
        colors = depth_to_colors(transformed[valid, 2])
        colors_bgr = (colors[:, ::-1] * 255.0).astype(np.float64)
        overlay, count = overlay_points_on_image(rgb, u, v, colors_bgr, overlay_alpha)
        out_path = resolve_overlay_path(base, overlay_pattern, task, task_fs, rank)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), overlay):
            raise RuntimeError(f"Failed to write overlay image: {out_path}")
        print(
            f"[{task}] overlay saved: {out_path} "
            f"(rank={rank}, fitness={candidate['fitness']:.6f}, loss={candidate['loss']:.6f}, points={count})",
            file=sys.stderr,
        )
        saved_paths.append(out_path)

    return saved_paths


def choose_camera(task_name):
    if "left_chest_origin" in task_name:
        return "left"
    if "right_chest_origin" in task_name:
        return "right"
    raise ValueError(f"Cannot infer camera side from task name: {task_name}")


def depth_map_to_points(depth, task_name, stride=1):
    if stride < 1:
        raise ValueError("stride must be >= 1")
    camera_key = choose_camera(task_name)
    k = center_crop_intrinsics(
        load_intrinsics(CAMERA_INTRINSICS[camera_key]),
        RGB_ORIGINAL_SHAPE,
        depth.shape[:2],
    )
    sampled = np.asarray(depth, dtype=np.float64)[::stride, ::stride]
    valid = np.isfinite(sampled) & (sampled > 0.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)

    ys, xs = np.indices(sampled.shape)
    xs = xs.astype(np.float64) * float(stride)
    ys = ys.astype(np.float64) * float(stride)
    z = sampled[valid]
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    x = (xs[valid] - cx) * z / fx
    y = (ys[valid] - cy) * z / fy
    return np.stack((x, y, z), axis=1)


def load_points(np_path, task_name, input_mode="auto", depth_stride=1):
    pts = np.load(str(np_path))
    mode = input_mode
    if mode not in {"auto", "points", "depth"}:
        raise ValueError(f"Unsupported input_mode: {mode}")

    if mode == "auto":
        if pts.ndim == 1:
            mode = "points"
        elif pts.ndim == 2:
            # Nx3/Nx4 is treated as point list, image-like arrays are treated as depth maps.
            mode = "points" if pts.shape[1] in (3, 4) else "depth"
        elif pts.ndim == 3:
            mode = "points" if pts.shape[2] >= 3 else "depth"
        else:
            raise ValueError("Unsupported numpy shape for points: {}".format(pts.shape))

    if mode == "depth":
        if pts.ndim == 3 and pts.shape[2] >= 1:
            pts = pts[..., 0]
        if pts.ndim != 2:
            raise ValueError("Depth input must be HxW array. Got {}".format(pts.shape))
        pts = depth_map_to_points(pts, task_name, depth_stride)
    elif mode == "points":
        if pts.ndim == 1:
            if pts.size % 3 == 0:
                pts = pts.reshape((-1, 3))
            else:
                raise ValueError("Unsupported numpy shape for points: {}".format(pts.shape))
        elif pts.ndim == 2 and pts.shape[1] >= 3:
            pts = pts[:, :3]
        elif pts.ndim == 3 and pts.shape[2] >= 3:
            pts = pts.reshape((-1, pts.shape[2]))[:, :3]
        else:
            raise ValueError("Point array must have shape (N,>=3). Got {}".format(pts.shape))

    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    return pts


def prepare_pcd(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if len(pcd.points) == 0:
        return pcd
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    if len(pcd.points) == 0:
        return pcd
    radius = max(voxel_size * 2.0, 0.01)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return pcd


def load_and_sample_mesh(stl_path, sample_points, voxel_size):
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh: {stl_path}")
    mesh.compute_vertex_normals()
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
    if voxel_size and voxel_size > 0:
        mesh_pcd = mesh_pcd.voxel_down_sample(voxel_size)
    radius = max(voxel_size * 2.0, 0.01)
    mesh_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return mesh_pcd


def point_cloud_extent_diag(pcd):
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return 0.0
    extent = pts.max(axis=0) - pts.min(axis=0)
    return float(np.linalg.norm(extent))


def resolve_mesh_scale(mode, source_pcd, target_pcd):
    if mode == "m":
        return 1.0, "m"
    if mode == "mm":
        return 0.001, "mm"

    src_diag = point_cloud_extent_diag(source_pcd)
    tgt_diag = point_cloud_extent_diag(target_pcd)
    # Heuristic: STL often in mm while depth is in m.
    # If source is far larger than target, apply mm->m scale.
    if src_diag > 10.0 and tgt_diag > 0 and (src_diag / max(tgt_diag, 1e-9)) > 50.0:
        return 0.001, "auto(mm)"
    return 1.0, "auto(m)"


def build_xyz_rotation(rx_deg, ry_deg, rz_deg):
    rx = np.deg2rad(float(rx_deg))
    ry = np.deg2rad(float(ry_deg))
    rz = np.deg2rad(float(rz_deg))

    cx, sx = float(np.cos(rx)), float(np.sin(rx))
    cy, sy = float(np.cos(ry)), float(np.sin(ry))
    cz, sz = float(np.cos(rz)), float(np.sin(rz))

    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float64,
    )
    rot_y = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float64,
    )
    rot_z = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # Apply X->Y->Z initial rotations.
    rot3 = rot_z @ rot_y @ rot_x
    rot = np.eye(4, dtype=np.float64)
    rot[:3, :3] = rot3
    return rot


def orthogonalize_rotation_3x3(r: np.ndarray) -> np.ndarray:
    """Project 3x3 matrix onto SO(3) for stable angle extraction."""
    r = np.asarray(r, dtype=np.float64)[:3, :3]
    u, _, vt = np.linalg.svd(r)
    r2 = u @ vt
    if np.linalg.det(r2) < 0.0:
        u[:, 2] *= -1.0
        r2 = u @ vt
    return r2


def rotation_zyx_to_xyz_euler_deg(r3: np.ndarray) -> tuple[float, float, float]:
    """
    Inverse of rot_z @ rot_y @ rot_x used in build_xyz_rotation (same axis order).
    Returns (rx_deg, ry_deg, rz_deg).
    """
    r = orthogonalize_rotation_3x3(r3)
    r20 = float(r[2, 0])
    r21 = float(r[2, 1])
    r22 = float(r[2, 2])
    r00 = float(r[0, 0])
    r10 = float(r[1, 0])
    hyp = float(np.hypot(r00, r10))
    if hyp < 1e-12:
        rz = 0.0
        ry = float(np.arctan2(-r20, r22))
        rx = float(np.arctan2(r[0, 1], r[0, 2]))
    else:
        ry = float(np.arctan2(-r20, hyp))
        rx = float(np.arctan2(r21, r22))
        rz = float(np.arctan2(r10, r00))
    return float(np.rad2deg(rx)), float(np.rad2deg(ry)), float(np.rad2deg(rz))


def parse_angle_list(text):
    vals = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not vals:
        raise ValueError("Rotation angle list cannot be empty.")
    return vals


def load_depth_for_init(np_path):
    arr = np.load(str(np_path))
    if arr.ndim == 2:
        return np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[2] >= 1:
        return np.asarray(arr[..., 0], dtype=np.float64)
    return None


def center_depth_translation(task_name, depth_arr):
    camera_key = choose_camera(task_name)
    k = center_crop_intrinsics(
        load_intrinsics(CAMERA_INTRINSICS[camera_key]),
        RGB_ORIGINAL_SHAPE,
        depth_arr.shape[:2],
    )
    valid = np.isfinite(depth_arr) & (depth_arr > 0.0)
    if not np.any(valid):
        raise ValueError("Depth map has no valid positive values for center-depth translation.")
    z = float(np.mean(depth_arr[valid]))
    h, w = depth_arr.shape[:2]
    u = (float(w) - 1.0) / 2.0
    v = (float(h) - 1.0) / 2.0
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64), camera_key, z


def write_icp_convergence_csv(csv_path, rows):
    """rows: 排序 + 搜索起始三角度 + 收敛结束三角度 + 最终平移 + 覆盖程度 + loss（角度单位：度，平移单位：米）。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "排序",
        "搜索起始_rx_度",
        "搜索起始_ry_度",
        "搜索起始_rz_度",
        "收敛结束_rx_度",
        "收敛结束_ry_度",
        "收敛结束_rz_度",
        "最终平移_x_米",
        "最终平移_y_米",
        "最终平移_z_米",
        "覆盖程度",
        "loss",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_result_subdir(logical: str, result_root: Path) -> str:
    """Map logical task name to existing result/<name> folder (alias or same name)."""
    if (result_root / logical).is_dir():
        return logical
    alt = RESULT_TASK_ALIASES.get(logical)
    if alt and (result_root / alt).is_dir():
        return alt
    return logical


def resolve_depth_npy(base: Path, out_root: str, task: str, depth_file: str) -> Path:
    """Locate depth npy: same layout as visualize.py (result/<task>/), with rglob fallback under task dirs."""
    if "{task}" in depth_file:
        p = Path(depth_file.format(task=task))
        return p if p.is_absolute() else (base / p)

    stems = (base / out_root / task, base / task)
    for root in stems:
        direct = root / depth_file
        if direct.is_file():
            return direct
    for root in stems:
        if root.is_dir():
            name = Path(depth_file).name
            matches = [p for p in root.rglob(depth_file) if p.is_file() and p.name == name]
            if matches:
                return min(matches, key=lambda p: len(p.parts))

    root_only = base / depth_file
    if root_only.is_file():
        return root_only
    return base / out_root / task / depth_file


def main():
    parser = argparse.ArgumentParser(description="ICP align depth_mean.npy point cloud with STL model")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parent),
        help="default: 20260508/ (this script's folder). Depth: <out-root>/<task>/depth_mean.npy (same as visualize.py); override if data lives elsewhere.",
    )
    parser.add_argument("--depth-file", default="depth_mean.npy", help="depth point cloud numpy file (relative to base-dir if not absolute)")
    parser.add_argument("--stl-file", default="底盘.STL", help="STL model file (relative to base-dir if not absolute)")
    parser.add_argument("--out-root", default="result", help="root folder for results (relative to base-dir)")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="voxel size for downsampling (meters)")
    parser.add_argument("--sample-points", type=int, default=20000, help="number of points to sample from mesh")
    parser.add_argument("--max-corr-coarse", type=float, default=0.05, help="max correspondence distance for coarse ICP")
    parser.add_argument("--max-corr-fine", type=float, default=0.01, help="max correspondence distance for fine ICP")
    parser.add_argument(
        "--input-mode",
        choices=["auto", "points", "depth"],
        default="auto",
        help="how to parse npy: auto detect, explicit point list, or depth map",
    )
    parser.add_argument(
        "--depth-stride",
        type=int,
        default=1,
        help="pixel stride when converting depth map to points (same as visualize.py)",
    )
    parser.add_argument(
        "--mesh-units",
        choices=["auto", "m", "mm"],
        default="mm",
        help="units of STL mesh coordinates; default mm scales mesh by 0.001 to meters",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        default=True,
        help="project the top 4 ICP STL poses onto RGB images and save overlays",
    )
    parser.add_argument(
        "--rgb-pattern",
        default="rgb_new/{task}/000000.png",
        help="RGB image path pattern; {task} or {task_fs} will be replaced (relative to base-dir)",
    )
    parser.add_argument(
        "--overlay-out",
        default="0512/icp_overlay_{task}_rank{rank}.png",
        help="output overlay file path (relative to base-dir), supports {task}, {task_fs}, and {rank}",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.75,
        help="overlay alpha for projected points",
    )
    parser.add_argument("--init-rot-x", type=str, default="0,45,90,135,180,225,270,315", help="comma-separated initial X rotations in degrees")
    parser.add_argument("--init-rot-y", type=str, default="0,45,90,135,180,225,270,315", help="comma-separated initial Y rotations in degrees")
    parser.add_argument("--init-rot-z", type=str, default="0,45,90,135,180,225,270,315", help="comma-separated initial Z rotations in degrees")
    args = parser.parse_args()
    init_rot_x = parse_angle_list(args.init_rot_x)
    init_rot_y = parse_angle_list(args.init_rot_y)
    init_rot_z = parse_angle_list(args.init_rot_z)
    init_rotation_combos = list(itertools.product(init_rot_x, init_rot_y, init_rot_z))

    base = Path(args.base_dir)
    csv_out_dir = base / "0512"

    # 仅处理文件顶部全局变量 task_name 指定的任务
    task_list = [task_name]

    for task in task_list:
        try:
            result_root = base / args.out_root
            task_fs = resolve_result_subdir(task, result_root)

            # resolve depth file (disk layout uses task_fs)
            depth_path = resolve_depth_npy(base, args.out_root, task_fs, args.depth_file)

            # resolve stl for this task
            if "{task}" in args.stl_file:
                stl_path = Path(args.stl_file.format(task=task_fs))
                if not stl_path.is_absolute():
                    stl_path = base / stl_path
            else:
                stl_path = Path(args.stl_file)
                if not stl_path.is_absolute():
                    stl_path = base / stl_path

            if not depth_path.is_file():
                hint = ""
                result_root = base / args.out_root
                if result_root.is_dir():
                    with_depth = sorted(
                        p.name
                        for p in result_root.iterdir()
                        if p.is_dir()
                        and resolve_depth_npy(base, args.out_root, p.name, args.depth_file).is_file()
                    )
                    if with_depth:
                        hint = f" Under {result_root}, tasks that contain {args.depth_file}: {', '.join(with_depth)}."
                print(f"[{task}] depth file not found: {depth_path}.{hint}", file=sys.stderr)
                continue
            if not stl_path.exists():
                print(f"[{task}] stl file not found: {stl_path}", file=sys.stderr)
                continue

            points = load_points(depth_path, task, args.input_mode, args.depth_stride)
            if points.shape[0] == 0:
                print(f"[{task}] no valid points in depth file", file=sys.stderr)
                continue

            target_pcd = prepare_pcd(points, args.voxel_size)
            source_pcd = load_and_sample_mesh(stl_path, args.sample_points, args.voxel_size)

            mesh_scale, _ = resolve_mesh_scale(args.mesh_units, source_pcd, target_pcd)
            if mesh_scale != 1.0:
                source_pcd.scale(mesh_scale, center=(0.0, 0.0, 0.0))

            src_np = np.asarray(source_pcd.points)
            tgt_np = np.asarray(target_pcd.points)
            if src_np.size == 0 or tgt_np.size == 0:
                print(f"[{task}] empty point cloud after loading/downsampling", file=sys.stderr)
                continue

            src_centroid = src_np.mean(axis=0)
            depth_init = load_depth_for_init(depth_path) if args.input_mode in ("auto", "depth") else None
            if depth_init is not None:
                center_pt, _, _ = center_depth_translation(task, depth_init)
                init_t = center_pt - src_centroid
            else:
                tgt_centroid = tgt_np.mean(axis=0)
                init_t = tgt_centroid - src_centroid

            base_init_trans = np.eye(4)
            base_init_trans[:3, 3] = init_t

            candidates = []
            for rx_deg, ry_deg, rz_deg in init_rotation_combos:
                init_trans = np.array(base_init_trans, copy=True)
                init_trans = init_trans @ build_xyz_rotation(rx_deg, ry_deg, rz_deg)
                icp_coarse = o3d.pipelines.registration.registration_icp(
                    source_pcd, target_pcd, args.max_corr_coarse, init_trans,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )
                icp_fine = o3d.pipelines.registration.registration_icp(
                    source_pcd, target_pcd, args.max_corr_fine, icp_coarse.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
                T = np.asarray(icp_fine.transformation, dtype=np.float64)
                fitness = float(icp_fine.fitness)
                loss = float(icp_fine.inlier_rmse)
                r_final = T[:3, :3]
                end_rx, end_ry, end_rz = rotation_zyx_to_xyz_euler_deg(r_final)
                candidates.append(
                    {
                        "T": T,
                        "angles_deg": (float(rx_deg), float(ry_deg), float(rz_deg)),
                        "end_angles_deg": (float(end_rx), float(end_ry), float(end_rz)),
                        "translation": tuple(float(v) for v in T[:3, 3]),
                        "fitness": fitness,
                        "loss": loss,
                    }
                )

            # 覆盖程度降序；相同时 loss 越小越靠前
            sorted_candidates = sorted(candidates, key=lambda item: (-item["fitness"], item["loss"]))
            ranked_rows = []
            for i, candidate in enumerate(sorted_candidates, start=1):
                start_rx, start_ry, start_rz = candidate["angles_deg"]
                end_rx, end_ry, end_rz = candidate["end_angles_deg"]
                trans_x, trans_y, trans_z = candidate["translation"]
                ranked_rows.append(
                    {
                        "排序": i,
                        "搜索起始_rx_度": start_rx,
                        "搜索起始_ry_度": start_ry,
                        "搜索起始_rz_度": start_rz,
                        "收敛结束_rx_度": end_rx,
                        "收敛结束_ry_度": end_ry,
                        "收敛结束_rz_度": end_rz,
                        "最终平移_x_米": trans_x,
                        "最终平移_y_米": trans_y,
                        "最终平移_z_米": trans_z,
                        "覆盖程度": candidate["fitness"],
                        "loss": candidate["loss"],
                    }
                )

            csv_path = csv_out_dir / f"{task}2.csv"
            write_icp_convergence_csv(csv_path, ranked_rows)
            print(csv_path)

            if args.save_overlay and sorted_candidates:
                try:
                    save_top_overlay_images(
                        base=base,
                        task=task,
                        task_fs=task_fs,
                        source_points=np.asarray(source_pcd.points),
                        sorted_candidates=sorted_candidates,
                        rgb_pattern=args.rgb_pattern,
                        overlay_pattern=args.overlay_out,
                        overlay_alpha=args.overlay_alpha,
                    )
                except Exception as e:
                    print(f"[{task}] overlay failed: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[{task}] {e}", file=sys.stderr)
            continue


if __name__ == '__main__':
    main()

