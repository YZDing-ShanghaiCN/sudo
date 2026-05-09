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
import numpy as np

try:
    import open3d as o3d
except Exception as e:
    print("Error: open3d is required for this script. Install with `pip install open3d`.")
    raise


def load_points(np_path):
    pts = np.load(str(np_path))
    if pts.ndim == 1:
        if pts.size % 3 == 0:
            pts = pts.reshape((-1, 3))
        else:
            raise ValueError("Unsupported numpy shape for points: {}".format(pts.shape))
    if pts.ndim == 2 and pts.shape[1] >= 3:
        pts = pts[:, :3]
    else:
        raise ValueError("Point array must have shape (N,>=3). Got {}".format(pts.shape))
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    return pts


def prepare_pcd(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
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
    parser.add_argument("--out-root", default="result", help="root folder for results (relative to base-dir)")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="voxel size for downsampling (meters)")
    parser.add_argument("--sample-points", type=int, default=20000, help="number of points to sample from mesh")
    parser.add_argument("--max-corr-coarse", type=float, default=0.05, help="max correspondence distance for coarse ICP")
    parser.add_argument("--max-corr-fine", type=float, default=0.01, help="max correspondence distance for fine ICP")
    args = parser.parse_args()

    base = Path(args.base_dir)
    depth_path = Path(args.depth_file)
    if not depth_path.is_absolute():
        depth_path = base / depth_path
    stl_path = Path(args.stl_file)
    if not stl_path.is_absolute():
        stl_path = base / stl_path

    if not depth_path.exists():
        print(f"Error: depth file not found: {depth_path}")
        sys.exit(2)
    if not stl_path.exists():
        print(f"Error: stl file not found: {stl_path}")
        sys.exit(2)

    print(f"Loading point cloud from {depth_path}")
    points = load_points(depth_path)
    print(f"Points loaded: {points.shape[0]}")

    print(f"Preparing point cloud (voxel_size={args.voxel_size})")
    target_pcd = prepare_pcd(points, args.voxel_size)

    print(f"Loading and sampling mesh from {stl_path} (sample {args.sample_points} points)")
    source_pcd = load_and_sample_mesh(stl_path, args.sample_points, args.voxel_size)

    # initial translation to align centroids
    src_np = np.asarray(source_pcd.points)
    tgt_np = np.asarray(target_pcd.points)
    if src_np.size == 0 or tgt_np.size == 0:
        raise RuntimeError("Empty point cloud after loading/downsampling")
    src_centroid = src_np.mean(axis=0)
    tgt_centroid = tgt_np.mean(axis=0)
    init_trans = np.eye(4)
    init_trans[:3, 3] = tgt_centroid - src_centroid

    print("Running coarse ICP...")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, args.max_corr_coarse, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    print("Running fine ICP (point-to-plane)...")
    icp_fine = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, args.max_corr_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    transform = icp_fine.transformation
    loss = icp_fine.inlier_rmse

    out_dir = base / args.out_root / args.task_name
    out_file = out_dir / "result.txt"
    print(f"Writing result to {out_file}")
    write_result(transform, loss, out_file)

    # also save npy for convenience
    np.save(out_dir / "transform.npy", transform)

    print("Done. Transform:")
    np.set_printoptions(precision=6, suppress=True)
    print(transform)
    print(f"loss: {loss:.5f}")


if __name__ == '__main__':
    main()

