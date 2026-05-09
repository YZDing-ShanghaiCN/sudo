#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICP 对齐脚本：将 depth_mean.npy 点云与 STL 模型对齐并输出变换矩阵与损失。

输出路径：<base_dir>/result/<task_name>/result.txt
格式示例：
transform metrix:
[[r11, r12, r13, t1],
 [r21, r22, r23, t2],
 [r31, r32, r33, t3],
 [0,   0,   0,   1]]
loss: 0.12345
"""

from pathlib import Path
import argparse
import sys
import itertools
import numpy as np

try:
    import open3d as o3d
except Exception as e:
    print("Error: open3d is required for this script. Install with `pip install open3d`.")
    raise

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


def choose_best_icp_result(candidates):
    # Higher fitness is better; with equal fitness, lower rmse is better.
    return max(candidates, key=lambda c: (float(c["fine"].fitness), -float(c["fine"].inlier_rmse)))


def write_result(transform, loss, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mat = np.array(transform)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("transform metrix:\n")
        f.write("[")
        for i in range(4):
            row = mat[i]
            f.write("[")
            f.write(", ".join([f"{float(x):.6f}" for x in row]))
            f.write("]")
            if i < 3:
                f.write(",\n ")
            else:
                f.write("\n")
        f.write("]\n")
        f.write(f"loss: {float(loss):.5f}\n")


def main():
    parser = argparse.ArgumentParser(description="ICP align depth_mean.npy point cloud with STL model")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent), help="base directory where depth_mean.npy and STL are located")
    parser.add_argument("--depth-file", default="depth_mean.npy", help="depth point cloud numpy file (relative to base-dir if not absolute)")
    parser.add_argument("--stl-file", default="底盘.STL", help="STL model file (relative to base-dir if not absolute)")
    parser.add_argument("--task-name", default="task", help="task name used for result subfolder")
    parser.add_argument("--task-names", nargs='*', default=None, help="space-separated list of task names to process (overrides --task-name if provided)")
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
    parser.add_argument("--init-rot-x", type=str, default="0,90,180,270", help="comma-separated initial X rotations in degrees")
    parser.add_argument("--init-rot-y", type=str, default="0,90,180,270", help="comma-separated initial Y rotations in degrees")
    parser.add_argument("--init-rot-z", type=str, default="0,90,180,270", help="comma-separated initial Z rotations in degrees")
    args = parser.parse_args()
    init_rot_x = parse_angle_list(args.init_rot_x)
    init_rot_y = parse_angle_list(args.init_rot_y)
    init_rot_z = parse_angle_list(args.init_rot_z)
    init_rotation_combos = list(itertools.product(init_rot_x, init_rot_y, init_rot_z))

    base = Path(args.base_dir)

    # determine tasks to run
    if args.task_names and len(args.task_names) > 0:
        # allow comma-separated single argument
        if len(args.task_names) == 1 and isinstance(args.task_names[0], str) and "," in args.task_names[0]:
            task_list = [t.strip() for t in args.task_names[0].split(",") if t.strip()]
        else:
            task_list = args.task_names
    else:
        # auto-discover task names by listing subdirectories in base/<out_root>
        result_dir = base / args.out_root
        if result_dir.exists() and result_dir.is_dir():
            task_list = sorted(
                [
                    p.name
                    for p in result_dir.iterdir()
                    if p.is_dir() and (p / args.depth_file).exists()
                ]
            )
            if len(task_list) == 0:
                task_list = [args.task_name]
            else:
                print(f"Discovered tasks: {task_list}")
        else:
            task_list = [args.task_name]

    for task in task_list:
        try:
            # resolve depth file for this task
            if "{task}" in args.depth_file:
                depth_path = Path(args.depth_file.format(task=task))
                if not depth_path.is_absolute():
                    depth_path = base / depth_path
            else:
                # prefer base/<out_root>/<task>/<depth-file>, then base/<task>/<depth-file>, then base/<depth-file>
                cand0 = base / args.out_root / task / args.depth_file
                cand1 = base / task / args.depth_file
                cand2 = base / args.depth_file
                if cand0.exists():
                    depth_path = cand0
                elif cand1.exists():
                    depth_path = cand1
                elif cand2.exists():
                    depth_path = cand2
                else:
                    depth_path = Path(args.depth_file)
                    if not depth_path.is_absolute():
                        depth_path = base / depth_path

            # resolve stl for this task
            if "{task}" in args.stl_file:
                stl_path = Path(args.stl_file.format(task=task))
                if not stl_path.is_absolute():
                    stl_path = base / stl_path
            else:
                stl_path = Path(args.stl_file)
                if not stl_path.is_absolute():
                    stl_path = base / stl_path

            if not depth_path.exists():
                print(f"[{task}] Warning: depth file not found: {depth_path} - skipping")
                continue
            if not stl_path.exists():
                print(f"[{task}] Warning: stl file not found: {stl_path} - skipping")
                continue

            print(f"[{task}] Loading point cloud from {depth_path}")
            points = load_points(depth_path, task, args.input_mode, args.depth_stride)
            print(f"[{task}] Points loaded: {points.shape[0]}")
            if args.input_mode in ("auto", "depth"):
                print(f"[{task}] Depth->point config: stride={args.depth_stride}, camera={choose_camera(task)}")
            if points.shape[0] == 0:
                print(f"[{task}] No valid points in depth file - skipping")
                continue

            print(f"[{task}] Preparing point cloud (voxel_size={args.voxel_size})")
            target_pcd = prepare_pcd(points, args.voxel_size)

            print(f"[{task}] Loading and sampling mesh from {stl_path} (sample {args.sample_points} points)")
            source_pcd = load_and_sample_mesh(stl_path, args.sample_points, args.voxel_size)

            mesh_scale, mesh_mode = resolve_mesh_scale(args.mesh_units, source_pcd, target_pcd)
            if mesh_scale != 1.0:
                source_pcd.scale(mesh_scale, center=(0.0, 0.0, 0.0))
            print(f"[{task}] Mesh units={mesh_mode}, applied scale={mesh_scale}")

            src_np = np.asarray(source_pcd.points)
            tgt_np = np.asarray(target_pcd.points)
            if src_np.size == 0 or tgt_np.size == 0:
                print(f"[{task}] Empty point cloud after loading/downsampling - skipping")
                continue

            src_centroid = src_np.mean(axis=0)
            depth_init = load_depth_for_init(depth_path) if args.input_mode in ("auto", "depth") else None
            if depth_init is not None:
                center_pt, camera_key, mean_z = center_depth_translation(task, depth_init)
                init_t = center_pt - src_centroid
                print(
                    f"[{task}] Init translation from center-depth: "
                    f"camera={camera_key}, mean_depth={mean_z:.6f}m, center_cam_xyz={center_pt}"
                )
            else:
                tgt_centroid = tgt_np.mean(axis=0)
                init_t = tgt_centroid - src_centroid
                print(f"[{task}] Init translation fallback to centroid delta: {init_t}")

            base_init_trans = np.eye(4)
            base_init_trans[:3, 3] = init_t
            print(f"[{task}] Multi-start ICP rotation combos: {len(init_rotation_combos)}")

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
                candidates.append(
                    {
                        "angles_deg": (rx_deg, ry_deg, rz_deg),
                        "coarse": icp_coarse,
                        "fine": icp_fine,
                    }
                )

            best = choose_best_icp_result(candidates)
            best_angles = tuple(float(v) for v in best["angles_deg"])
            best_fine = best["fine"]
            print(
                f"[{task}] Selected best init (rx,ry,rz)=({best_angles[0]:.1f},{best_angles[1]:.1f},{best_angles[2]:.1f}), "
                f"fitness={best_fine.fitness:.6f}, rmse={best_fine.inlier_rmse:.6f}"
            )
            if best_fine.fitness < 1e-6:
                print(
                    f"[{task}] Warning: near-zero fitness. Check mesh units/scale and max correspondence distance."
                )

            transform = best_fine.transformation
            loss = best_fine.inlier_rmse

            out_dir = base / args.out_root / task
            out_file = out_dir / "result.txt"
            print(f"[{task}] Writing result to {out_file}")
            write_result(transform, loss, out_file)

            # also save npy for convenience
            np.save(out_dir / "transform.npy", transform)

            print(f"[{task}] Done. Transform:")
            np.set_printoptions(precision=6, suppress=True)
            print(transform)
            print(
                f"[{task}] best_init_rxyz_deg: "
                f"({best_angles[0]:.1f}, {best_angles[1]:.1f}, {best_angles[2]:.1f})"
            )
            print(f"[{task}] loss: {loss:.5f}")
        except Exception as e:
            print(f"[{task}] Error during processing: {e}")
            continue


if __name__ == '__main__':
    main()

