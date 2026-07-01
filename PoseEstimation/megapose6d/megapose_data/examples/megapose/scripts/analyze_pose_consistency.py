from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pose consistency for a single MegaPose task directory."
    )
    parser.add_argument("--task-dir", required=True, help="Path to task directory")
    parser.add_argument("--label", required=True, help="Object label to keep")
    parser.add_argument(
        "--unit",
        choices=["m", "mm"],
        default="m",
        help="Translation unit in object_data.json",
    )
    parser.add_argument(
        "--eps-mm", type=float, default=80.0, help="DBSCAN eps in millimeters"
    )
    parser.add_argument(
        "--min-samples", type=int, default=2, help="DBSCAN min_samples"
    )
    parser.add_argument(
        "--rot-weight",
        type=float,
        default=10.0,
        help="Rotation weight for combined distance",
    )
    return parser.parse_args()


def load_object_data(input_path: Path, label: str) -> Tuple[List[Dict[str, Any]], int]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("object_data.json must be a list")

    valid: List[Dict[str, Any]] = []
    skipped = 0

    for item in data:
        if not isinstance(item, dict):
            skipped += 1
            continue
        if not item.get("success", False):
            skipped += 1
            continue
        if item.get("label") != label:
            skipped += 1
            continue
        image_name = item.get("image_name")
        two = item.get("TWO")
        if not isinstance(image_name, str):
            skipped += 1
            continue
        if not isinstance(two, list) or len(two) != 2:
            skipped += 1
            continue
        q_raw, t_raw = two
        if not isinstance(q_raw, list) or not isinstance(t_raw, list):
            skipped += 1
            continue
        if len(q_raw) != 4 or len(t_raw) != 3:
            skipped += 1
            continue
        try:
            q = np.array(q_raw, dtype=float)
            t = np.array(t_raw, dtype=float)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if not (np.isfinite(q).all() and np.isfinite(t).all()):
            skipped += 1
            continue
        if np.linalg.norm(q) < EPS:
            skipped += 1
            continue
        valid.append({"image_name": image_name, "q": q, "t": t})

    return valid, skipped


def normalize_quaternions(qs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(qs, axis=1, keepdims=True)
    return qs / norms


def align_quaternions(qs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    aligned = qs.copy()
    dots = np.sum(aligned * ref[None, :], axis=1)
    aligned[dots < 0] *= -1.0
    return aligned


def to_mm(ts: np.ndarray, unit: str) -> np.ndarray:
    if unit == "m":
        return ts * 1000.0
    return ts.copy()


def rotation_distance_deg(qs: np.ndarray) -> np.ndarray:
    dots = np.clip(np.abs(qs @ qs.T), 0.0, 1.0)
    theta = 2.0 * np.arccos(dots)
    return np.degrees(theta)


def combined_distance_matrix(
    qs: np.ndarray, ts_mm: np.ndarray, rot_weight: float
) -> np.ndarray:
    diff = ts_mm[:, None, :] - ts_mm[None, :, :]
    dt = np.linalg.norm(diff, axis=2)
    rot_deg = rotation_distance_deg(qs)
    return dt + rot_weight * rot_deg


def mean_quaternion(qs: np.ndarray) -> np.ndarray:
    q_mean = np.mean(qs, axis=0)
    norm = np.linalg.norm(q_mean)
    if norm < EPS:
        return qs[0].copy()
    return q_mean / norm


def rotation_errors_deg(qs: np.ndarray, q_mean: np.ndarray) -> np.ndarray:
    dots = np.clip(np.abs(qs @ q_mean), 0.0, 1.0)
    theta = 2.0 * np.arccos(dots)
    return np.degrees(theta)


def translation_stats(ts_mm: np.ndarray) -> Dict[str, Any]:
    mean = ts_mm.mean(axis=0)
    std = ts_mm.std(axis=0, ddof=0)
    min_v = ts_mm.min(axis=0)
    max_v = ts_mm.max(axis=0)
    return {
        "mean_mm": mean.tolist(),
        "std_mm": std.tolist(),
        "min_mm": min_v.tolist(),
        "max_mm": max_v.tolist(),
        "range_mm": (max_v - min_v).tolist(),
    }


def z_stats(ts_mm: np.ndarray) -> Dict[str, Any]:
    z = ts_mm[:, 2]
    return {
        "mean_mm": float(np.mean(z)),
        "std_mm": float(np.std(z, ddof=0)),
        "range_mm": float(np.max(z) - np.min(z)),
    }


def error_summary(values: np.ndarray) -> Dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "max": float(np.max(values)),
    }


def compute_cluster_info(
    cluster_id: int,
    indices: np.ndarray,
    image_names: List[str],
    qs: np.ndarray,
    ts_mm: np.ndarray,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qs_c = qs[indices]
    ts_c = ts_mm[indices]
    q_mean = mean_quaternion(qs_c)
    r_err = rotation_errors_deg(qs_c, q_mean)
    t_mean = ts_c.mean(axis=0)
    t_err = np.linalg.norm(ts_c - t_mean[None, :], axis=1)

    info = {
        "cluster_id": int(cluster_id),
        "count": int(len(indices)),
        "image_names": [image_names[i] for i in indices.tolist()],
        "translation": translation_stats(ts_c),
        "z": z_stats(ts_c),
        "rotation": {
            "mean_quaternion": q_mean.tolist(),
            "error_deg": error_summary(r_err),
        },
        "translation_error_to_mean_mm": error_summary(t_err),
    }
    return info, q_mean, t_mean, t_err, r_err


def compute_overall_summary(
    qs: np.ndarray, ts_mm: np.ndarray
) -> Dict[str, Any]:
    q_mean = mean_quaternion(qs)
    r_err = rotation_errors_deg(qs, q_mean)
    t_mean = ts_mm.mean(axis=0)
    t_err = np.linalg.norm(ts_mm - t_mean[None, :], axis=1)

    return {
        "count": int(len(qs)),
        "translation": translation_stats(ts_mm),
        "z": z_stats(ts_mm),
        "rotation": {
            "mean_quaternion": q_mean.tolist(),
            "error_deg": error_summary(r_err),
        },
        "translation_error_to_mean_mm": error_summary(t_err),
    }


def format_optional(value: float) -> str:
    if value is None or np.isnan(value):
        return ""
    return f"{float(value):.6f}"


def write_csv(
    output_path: Path,
    image_names: List[str],
    labels: np.ndarray,
    label: str,
    qs: np.ndarray,
    ts: np.ndarray,
    t_errs: np.ndarray,
    r_errs: np.ndarray,
    is_main_cluster: np.ndarray,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_name",
                "cluster_id",
                "label",
                "qw",
                "qx",
                "qy",
                "qz",
                "x",
                "y",
                "z",
                "translation_error_to_cluster_mean_mm",
                "rotation_error_to_cluster_mean_deg",
                "is_main_cluster",
            ]
        )
        for i, name in enumerate(image_names):
            writer.writerow(
                [
                    name,
                    int(labels[i]),
                    label,
                    f"{qs[i, 0]:.8f}",
                    f"{qs[i, 1]:.8f}",
                    f"{qs[i, 2]:.8f}",
                    f"{qs[i, 3]:.8f}",
                    f"{ts[i, 0]:.6f}",
                    f"{ts[i, 1]:.6f}",
                    f"{ts[i, 2]:.6f}",
                    format_optional(t_errs[i]),
                    format_optional(r_errs[i]),
                    "true" if is_main_cluster[i] else "false",
                ]
            )


def print_summary(
    task_name: str,
    valid_count: int,
    cluster_counts: List[Tuple[int, int]],
    noise_count: int,
    main_cluster: Optional[Dict[str, Any]],
    clustering_performed: bool,
) -> None:
    print(f"Task: {task_name}")
    print(f"Valid poses: {valid_count}")

    if not clustering_performed:
        if valid_count < 2:
            print("Clustering skipped: need at least 2 valid poses")
        else:
            print("Clustering skipped: scikit-learn not available")
            print("Install with: pip install scikit-learn")
        return

    print("\nClusters:")
    for cluster_id, count in cluster_counts:
        print(f"  cluster {cluster_id}: {count} poses")
    print(f"  noise: {noise_count} poses")

    if main_cluster:
        ratio_pct = main_cluster["main_cluster_ratio"] * 100.0
        print(f"\nMain cluster ratio: {ratio_pct:.1f}%")
        print("\nMain cluster translation error:")
        print(
            f"  mean={main_cluster['main_cluster_translation_error_mean_mm']:.3f} mm"
        )
        print(
            f"  max={main_cluster['main_cluster_translation_error_max_mm']:.3f} mm"
        )
        print("\nMain cluster rotation error:")
        print(
            f"  mean={main_cluster['main_cluster_rotation_error_mean_deg']:.3f} deg"
        )
        print(
            f"  max={main_cluster['main_cluster_rotation_error_max_deg']:.3f} deg"
        )


def analyze_pose_consistency(
    task_dir: Path,
    label: str,
    unit: str,
    eps_mm: float,
    min_samples: int,
    rot_weight: float,
) -> None:
    input_path = task_dir / "outputs" / "object_data.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}")

    records, skipped = load_object_data(input_path, label)
    outputs_dir = task_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    json_path = outputs_dir / "pose_consistency_analysis.json"
    csv_path = outputs_dir / "pose_consistency_table.csv"

    if not records:
        analysis = {
            "task_name": task_dir.name,
            "input_path": str(input_path),
            "label": label,
            "valid_count": 0,
            "failed_or_skipped_count": skipped,
            "parameters": {
                "unit": unit,
                "eps_mm": eps_mm,
                "min_samples": min_samples,
                "rot_weight": rot_weight,
            },
            "clustering_performed": False,
            "main_cluster": None,
            "clusters": [],
            "noise": [],
            "all_valid_summary": None,
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        write_csv(csv_path, [], np.array([], dtype=int), label, np.empty((0, 4)), np.empty((0, 3)), np.array([]), np.array([]), np.array([]))
        print_summary(task_dir.name, 0, [], 0, None, False)
        return

    image_names = [r["image_name"] for r in records]
    qs = np.stack([r["q"] for r in records], axis=0)
    ts = np.stack([r["t"] for r in records], axis=0)

    qs = normalize_quaternions(qs)
    qs = align_quaternions(qs, qs[0])
    ts_mm = to_mm(ts, unit)

    n = len(records)
    labels = np.full(n, -1, dtype=int)
    clustering_performed = False

    if n >= 2:
        try:
            from sklearn.cluster import DBSCAN
        except Exception:
            print("scikit-learn is required for clustering.")
            print("Install with: pip install scikit-learn")
            clustering_performed = False
        else:
            dist = combined_distance_matrix(qs, ts_mm, rot_weight)
            labels = DBSCAN(
                eps=eps_mm, min_samples=min_samples, metric="precomputed"
            ).fit_predict(dist)
            clustering_performed = True

    cluster_ids = sorted({int(x) for x in labels.tolist() if x != -1})
    clusters: List[Dict[str, Any]] = []
    t_errs = np.full(n, np.nan)
    r_errs = np.full(n, np.nan)

    for cluster_id in cluster_ids:
        indices = np.where(labels == cluster_id)[0]
        info, _, _, t_err, r_err = compute_cluster_info(
            cluster_id, indices, image_names, qs, ts_mm
        )
        clusters.append(info)
        t_errs[indices] = t_err
        r_errs[indices] = r_err

    main_cluster: Optional[Dict[str, Any]] = None
    is_main_cluster = np.zeros(n, dtype=bool)
    if cluster_ids:
        counts = {cid: int(np.sum(labels == cid)) for cid in cluster_ids}
        main_cluster_id = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        is_main_cluster = labels == main_cluster_id
        main_cluster_errors_t = t_errs[is_main_cluster]
        main_cluster_errors_r = r_errs[is_main_cluster]
        main_cluster = {
            "main_cluster_id": int(main_cluster_id),
            "main_cluster_count": int(counts[main_cluster_id]),
            "main_cluster_ratio": float(counts[main_cluster_id] / n),
            "main_cluster_translation_error_mean_mm": float(
                np.mean(main_cluster_errors_t)
            ),
            "main_cluster_translation_error_max_mm": float(
                np.max(main_cluster_errors_t)
            ),
            "main_cluster_rotation_error_mean_deg": float(
                np.mean(main_cluster_errors_r)
            ),
            "main_cluster_rotation_error_max_deg": float(
                np.max(main_cluster_errors_r)
            ),
        }

    if clustering_performed:
        noise_names = [image_names[i] for i in range(n) if labels[i] == -1]
    else:
        noise_names = []
    all_valid_summary = compute_overall_summary(qs, ts_mm)

    analysis = {
        "task_name": task_dir.name,
        "input_path": str(input_path),
        "label": label,
        "valid_count": n,
        "failed_or_skipped_count": skipped,
        "parameters": {
            "unit": unit,
            "eps_mm": eps_mm,
            "min_samples": min_samples,
            "rot_weight": rot_weight,
            "distance": "dt_mm + rot_weight * theta_deg",
        },
        "clustering_performed": clustering_performed,
        "main_cluster": main_cluster,
        "clusters": clusters,
        "noise": noise_names,
        "all_valid_summary": all_valid_summary,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    write_csv(
        csv_path,
        image_names,
        labels,
        label,
        qs,
        ts,
        t_errs,
        r_errs,
        is_main_cluster,
    )

    cluster_counts = [(c["cluster_id"], c["count"]) for c in clusters]
    print_summary(
        task_dir.name,
        n,
        cluster_counts,
        len(noise_names),
        main_cluster,
        clustering_performed,
    )


def main() -> None:
    args = parse_args()
    analyze_pose_consistency(
        task_dir=Path(args.task_dir),
        label=args.label,
        unit=args.unit,
        eps_mm=args.eps_mm,
        min_samples=args.min_samples,
        rot_weight=args.rot_weight,
    )


if __name__ == "__main__":
    main()
