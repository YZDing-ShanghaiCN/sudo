#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch ICP alignment for the 20260508 depth_mean tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR / "result"
MODEL_PATH = SCRIPT_DIR / "底盘.STL"
CAMERA_INTRINSICS = {
    "left": SCRIPT_DIR.parent / "aililight_cameras" / "chest_left_camera.yaml",
    "right": SCRIPT_DIR.parent / "aililight_cameras" / "chest_right_camera.yaml",
}

DEPTH_STRIDE = 2
MODEL_SAMPLE_POINTS = 10000
POINT_VOXEL_SIZE = 0.005
NORMAL_RADIUS = 0.02
NORMAL_MAX_NN = 30
SEARCH_THRESHOLD = 0.03
SEARCH_MAX_ITER = 35
FINAL_THRESHOLD = 0.02
FINAL_MAX_ITER = 80
INTRINSIC_SCALE = 0.5

SEARCH_ROTATIONS = (
    (90.0, 0.0, -90.0),
    (90.0, 0.0, 90.0),
    (-90.0, 0.0, -90.0),
    (-90.0, 0.0, 90.0),
    (0.0, 0.0, 0.0),
    (180.0, 0.0, 0.0),
    (0.0, 90.0, 0.0),
    (0.0, -90.0, 0.0),
    (0.0, 0.0, 90.0),
    (0.0, 0.0, -90.0),
    (180.0, 90.0, 0.0),
    (180.0, -90.0, 0.0),
)


def parse_intrinsics_from_text(text: str) -> np.ndarray:
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


def load_intrinsics(yaml_path: Path) -> np.ndarray:
    if not yaml_path.exists():
        raise FileNotFoundError(f"Camera intrinsics file not found: {yaml_path}")

    text = yaml_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        k = np.array(data["intrinsic"], dtype=np.float64)
    except Exception:
        k = parse_intrinsics_from_text(text)

    if k.shape != (3, 3):
        raise ValueError(f"Invalid intrinsic matrix shape: {k.shape}")
    return k


def scale_intrinsics(k: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.array(k, dtype=np.float64, copy=True)
    scaled[0, 0] *= scale
    scaled[1, 1] *= scale
    scaled[0, 2] *= scale
    scaled[1, 2] *= scale
    return scaled


def rotation_matrix_from_xyz_deg(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rx_m = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64
    )
    ry_m = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64
    )
    rz_m = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return rz_m @ ry_m @ rx_m


def depth_to_points(depth: np.ndarray, k: np.ndarray, stride: int) -> np.ndarray:
    if stride < 1:
        raise ValueError("stride must be >= 1")

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


def points_to_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return cloud


def prepare_cloud(points: np.ndarray, voxel_size: float) -> o3d.geometry.PointCloud:
    cloud = points_to_cloud(points)
    if len(cloud.points) == 0:
        return cloud

    if voxel_size > 0:
        downsampled = cloud.voxel_down_sample(voxel_size)
        if len(downsampled.points) > 0:
            cloud = downsampled

    if len(cloud.points) > 0:
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=float(NORMAL_RADIUS),
                max_nn=int(NORMAL_MAX_NN),
            )
        )
        cloud.normalize_normals()

    return cloud


def load_model_cloud() -> o3d.geometry.PointCloud:
    mesh = o3d.io.read_triangle_mesh(str(MODEL_PATH))
    if mesh.is_empty():
        raise ValueError(f"Failed to load STL mesh: {MODEL_PATH}")

    mesh.scale(0.001, center=(0.0, 0.0, 0.0))
    mesh.compute_vertex_normals()
    model = mesh.sample_points_uniformly(number_of_points=int(MODEL_SAMPLE_POINTS))
    if len(model.points) == 0:
        raise ValueError(f"Failed to sample points from STL: {MODEL_PATH}")
    return prepare_cloud(np.asarray(model.points), POINT_VOXEL_SIZE)


def choose_camera(task_name: str) -> str:
    if "left_chest_origin" in task_name:
        return "left"
    if "right_chest_origin" in task_name:
        return "right"
    raise ValueError(f"Cannot infer camera side from task name: {task_name}")


def score_result(result: o3d.pipelines.registration.RegistrationResult) -> tuple[float, float]:
    fitness = float(result.fitness)
    rmse = float(result.inlier_rmse)
    if not np.isfinite(fitness):
        fitness = -np.inf
    if not np.isfinite(rmse):
        rmse = np.inf
    return fitness, -rmse


def search_initial_transform(
    model: o3d.geometry.PointCloud,
    observed: o3d.geometry.PointCloud,
) -> tuple[np.ndarray, o3d.pipelines.registration.RegistrationResult, tuple[float, float, float]]:
    model_center = np.asarray(model.get_center(), dtype=np.float64)
    observed_center = np.asarray(observed.get_center(), dtype=np.float64)

    best_result = None
    best_transform = np.eye(4, dtype=np.float64)
    best_rotation = (0.0, 0.0, 0.0)
    best_score = (-np.inf, -np.inf)

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=int(SEARCH_MAX_ITER)
    )
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    for rotation_spec in SEARCH_ROTATIONS:
        rotation = rotation_matrix_from_xyz_deg(*rotation_spec)
        init = np.eye(4, dtype=np.float64)
        init[:3, :3] = rotation
        init[:3, 3] = observed_center - rotation @ model_center

        result = o3d.pipelines.registration.registration_icp(
            model,
            observed,
            float(SEARCH_THRESHOLD),
            init,
            estimator,
            criteria,
        )

        score = score_result(result)
        if score > best_score:
            best_score = score
            best_result = result
            best_transform = result.transformation
            best_rotation = rotation_spec

    if best_result is None:
        raise RuntimeError("ICP search failed to produce a valid result.")

    return best_transform, best_result, best_rotation


def refine_transform(
    model: o3d.geometry.PointCloud,
    observed: o3d.geometry.PointCloud,
    init: np.ndarray,
) -> o3d.pipelines.registration.RegistrationResult:
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=int(FINAL_MAX_ITER)
    )
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    return o3d.pipelines.registration.registration_icp(
        model,
        observed,
        float(FINAL_THRESHOLD),
        np.asarray(init, dtype=np.float64),
        estimator,
        criteria,
    )


def format_float(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if text in {"-0", ""}:
        text = "0"
    if "." not in text:
        text += ".0"
    return text


def format_matrix(matrix: np.ndarray) -> str:
    matrix = np.asarray(matrix, dtype=np.float64)
    rows = []
    for index, row in enumerate(matrix):
        row_text = ", ".join(format_float(value) for value in row)
        if index == 0:
            rows.append(f"[[{row_text}],")
        elif index == len(matrix) - 1:
            rows.append(f"    [{row_text}]]")
        else:
            rows.append(f"    [{row_text}],")
    return "\n".join(rows)


def write_result_file(task_dir: Path, loss: float, transform: np.ndarray) -> Path:
    content = f"loss: {loss:.6f}\nT: {format_matrix(transform)}\n"
    result_path = task_dir / "result.txt"
    result_path.write_text(content, encoding="utf-8")
    return result_path


def load_task_dirs(task_name: str | None = None) -> list[Path]:
    if task_name:
        task_dir = RESULT_ROOT / task_name
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")
        return [task_dir]

    task_dirs = [
        path
        for path in sorted(RESULT_ROOT.iterdir())
        if path.is_dir() and (path / "depth_mean.npy").exists()
    ]
    return task_dirs


def align_task(task_dir: Path, model: o3d.geometry.PointCloud) -> tuple[float, np.ndarray, o3d.pipelines.registration.RegistrationResult]:
    task_name = task_dir.name
    camera_key = choose_camera(task_name)
    intrinsics = scale_intrinsics(load_intrinsics(CAMERA_INTRINSICS[camera_key]), INTRINSIC_SCALE)

    depth_path = task_dir / "depth_mean.npy"
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth_mean.npy: {depth_path}")

    depth = np.load(depth_path)
    points = depth_to_points(depth, intrinsics, DEPTH_STRIDE)
    if points.size == 0:
        raise RuntimeError(f"No valid depth points in {depth_path}")

    observed = prepare_cloud(points, POINT_VOXEL_SIZE)
    if len(observed.points) == 0:
        raise RuntimeError(f"Failed to build observed cloud for {task_name}")

    init_transform, coarse_result, rotation_spec = search_initial_transform(model, observed)

    try:
        final_result = refine_transform(model, observed, init_transform)
    except Exception:
        final_result = coarse_result

    loss = float(final_result.inlier_rmse)
    result_path = write_result_file(task_dir, loss, final_result.transformation)

    print(
        f"[OK] {task_name}: camera={camera_key}, rotation={rotation_spec}, "
        f"loss={loss:.6f}, fitness={final_result.fitness:.4f}, saved={result_path}"
    )

    return loss, final_result.transformation, final_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ICP alignment for depth_mean.npy tasks.")
    parser.add_argument("--task", default="", help="Process only one task directory name.")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"STL model not found: {MODEL_PATH}")

    model = load_model_cloud()
    task_dirs = load_task_dirs(args.task or None)
    if not task_dirs:
        raise RuntimeError(f"No task directories with depth_mean.npy found under {RESULT_ROOT}")

    failures = []
    for task_dir in task_dirs:
        try:
            align_task(task_dir, model)
        except Exception as exc:
            message = f"{task_dir.name}: {exc}"
            print(f"[ERR] {message}")
            failures.append(message)

    if failures:
        raise RuntimeError("Some tasks failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()

