"""
Batch-process images of a fixed camera observing a ChArUco board.

Input:
- Camera intrinsics/distortion from config/camera_cfg.yaml
- A folder of images (e.g. temp/08) containing a ChArUco board

Outputs:
1) Per-image CSV:
   - 4 outer board corner pixel coordinates (top-left, top-right, bottom-right, bottom-left)
   - Board pose w.r.t. camera (rvec/tvec)
2) Summary JSON:
   - Mean pose (tvec + averaged rotation) and mean corner pixels across successful frames

This file intentionally avoids non-standard dependencies. If PyYAML is available it will
be used; otherwise a small YAML subset parser is used for the project's camera_cfg.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class CameraConfig:
    image_size: Tuple[int, int]  # (w, h)
    dictionary_name: str
    squares_x: int
    squares_y: int
    square_size_m: float
    marker_size_m: float
    camera_matrix: np.ndarray  # (3,3)
    dist_coeffs: np.ndarray  # (N,) usually 5


def _parse_scalar(s: str) -> Any:
    s = s.strip()
    if s == "":
        return ""
    # Strip quotes if present
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    # Try int, float, else string
    try:
        if s.lower().startswith("0x"):
            return int(s, 16)
        if any(c in s for c in (".", "e", "E")):
            return float(s)
        return int(s)
    except Exception:
        return s


def _load_yaml_via_pyyaml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _load_camera_cfg_fallback(path: Path, camera_key: str) -> Dict[str, Any]:
    """
    Very small YAML subset parser tailored to this repo's camera_cfg.yaml structure.
    Supports:
    - top-level key (camera_1)
    - scalar values
    - lists and nested lists (camera_matrix, dist_coeffs)
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    # Extract the camera section.
    # Section starts at "<camera_key>:" at indent 0, ends at next indent-0 key or EOF.
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"{camera_key}:" and (len(line) - len(line.lstrip(" "))) == 0:
            start = i
            break
    if start is None:
        raise ValueError(f"camera key not found in yaml: {camera_key}")

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip().endswith(":") and (len(lines[i]) - len(lines[i].lstrip(" "))) == 0:
            end = i
            break

    section = lines[start + 1 : end]

    data: Dict[str, Any] = {}
    i = 0
    while i < len(section):
        line = section[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if not line.startswith("  "):
            # Unexpected indent, ignore
            i += 1
            continue

        stripped = line.strip()
        if ":" in stripped:
            key, rest = stripped.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            if rest != "":
                data[key] = _parse_scalar(rest)
                i += 1
                continue

            # key: (block)
            # Determine if it's a list block by looking ahead.
            j = i + 1
            while j < len(section) and (not section[j].strip() or section[j].lstrip().startswith("#")):
                j += 1
            if j >= len(section) or not section[j].startswith("  -"):
                data[key] = {}
                i += 1
                continue

            # Parse list block (possibly nested list-of-lists)
            lst: List[Any] = []
            i += 1
            while i < len(section):
                l2 = section[i]
                if not l2.startswith("  -"):
                    break
                s2 = l2.strip()
                if s2.startswith("- - "):
                    # A nested list starts here: "- - <scalar>" followed by "    - <scalar>" lines
                    row: List[Any] = [_parse_scalar(s2[4:])]
                    i += 1
                    while i < len(section) and section[i].startswith("    -"):
                        row.append(_parse_scalar(section[i].strip()[2:]))
                        i += 1
                    lst.append(row)
                    continue
                # Regular list item "- <scalar>"
                lst.append(_parse_scalar(s2[2:]))
                i += 1
            data[key] = lst
            continue

        i += 1

    return {camera_key: data}


def load_camera_config(camera_yaml: Path, camera_key: str) -> CameraConfig:
    cfg = _load_yaml_via_pyyaml(camera_yaml)
    if cfg is None:
        cfg = _load_camera_cfg_fallback(camera_yaml, camera_key)

    if camera_key not in cfg:
        raise ValueError(f"camera key '{camera_key}' missing in {camera_yaml}")
    c = cfg[camera_key]

    image_size = tuple(int(x) for x in c["image_size"])
    dictionary_name = str(c["dictionary"])
    squares_x = int(c["squares_x"])
    squares_y = int(c["squares_y"])
    square_size_m = float(c["square_size_mm"]) / 1000.0
    marker_size_m = float(c["marker_size_mm"]) / 1000.0

    camera_matrix = np.array(c["camera_matrix"], dtype=np.float64)
    dist_coeffs_raw = c["dist_coeffs"]
    # dist_coeffs in this repo is [[k1,k2,p1,p2,k3]] but handle flat too
    if isinstance(dist_coeffs_raw, list) and len(dist_coeffs_raw) == 1 and isinstance(dist_coeffs_raw[0], list):
        dist_coeffs_raw = dist_coeffs_raw[0]
    dist_coeffs = np.array(dist_coeffs_raw, dtype=np.float64).reshape(-1, 1)

    return CameraConfig(
        image_size=(int(image_size[0]), int(image_size[1])),
        dictionary_name=dictionary_name,
        squares_x=squares_x,
        squares_y=squares_y,
        square_size_m=square_size_m,
        marker_size_m=marker_size_m,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )


def aruco_dictionary_from_name(name: str) -> cv2.aruco.Dictionary:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV was built without aruco (need opencv-contrib-python).")

    key = name.strip()
    if not key.startswith("DICT_"):
        raise ValueError(f"unexpected dictionary name: {name}")
    if not hasattr(cv2.aruco, key):
        # A common mismatch: user says "5*5" and expects bigger dict.
        # Give a helpful error rather than silently choosing a different dictionary.
        raise ValueError(f"cv2.aruco has no predefined dictionary named: {key}")

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, key))


def _create_charuco_board(
    dictionary: cv2.aruco.Dictionary, squares_x: int, squares_y: int, square_len: float, marker_len: float
):
    # OpenCV Python API differs by version
    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, dictionary)
    return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_len, marker_len, dictionary)


def _rotation_matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [w, x, y, z].
    """
    # Robust conversion
    tr = float(np.trace(R))
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    # Normalize
    n = np.linalg.norm(q)
    if n > 0:
        q /= n
    return q


def _quat_wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def average_quaternions_wxyz(quats: Sequence[np.ndarray]) -> np.ndarray:
    """
    Average quaternions using the Markley method (eigenvector of accumulator).
    Assumes quaternions are [w, x, y, z].
    """
    if not quats:
        raise ValueError("no quaternions to average")
    A = np.zeros((4, 4), dtype=np.float64)
    for q in quats:
        q = np.asarray(q, dtype=np.float64).reshape(4)
        # Handle double-cover: keep in same hemisphere as first quaternion
        if quats and np.dot(q, quats[0].reshape(4)) < 0:
            q = -q
        A += np.outer(q, q)
    A /= float(len(quats))
    w, v = np.linalg.eigh(A)
    q_avg = v[:, int(np.argmax(w))]
    if q_avg[0] < 0:
        q_avg = -q_avg
    q_avg /= np.linalg.norm(q_avg)
    return q_avg


def rvec_tvec_to_pose_dict(rvec: np.ndarray, tvec: np.ndarray) -> Dict[str, Any]:
    return {
        "rvec": [float(x) for x in rvec.reshape(-1)],
        "tvec": [float(x) for x in tvec.reshape(-1)],
    }


def _compute_reproj_error_px(
    board, charuco_corners: np.ndarray, charuco_ids: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K, dist
) -> float:
    if charuco_corners is None or charuco_ids is None or len(charuco_corners) == 0:
        return float("nan")

    # 3D points corresponding to detected charuco corners
    chess_corners = board.getChessboardCorners() if hasattr(board, "getChessboardCorners") else board.chessboardCorners
    ids = charuco_ids.reshape(-1).astype(int)
    objp = chess_corners[ids].reshape(-1, 3).astype(np.float64)
    imgp_obs = charuco_corners.reshape(-1, 2).astype(np.float64)
    imgp_proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    imgp_proj = imgp_proj.reshape(-1, 2)
    err = np.linalg.norm(imgp_obs - imgp_proj, axis=1)
    return float(np.mean(err)) if len(err) else float("nan")


def _put_overlay_lines(
    image: np.ndarray,
    lines: Sequence[str],
    origin_xy: Tuple[int, int] = (18, 28),
    line_height: int = 24,
    text_scale: float = 0.65,
    text_thickness: int = 1,
) -> None:
    """
    Draw multiple text lines with a dark background block for readability.
    """
    if image is None or len(lines) == 0:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    x0, y0 = origin_xy
    margin = 8
    widths: List[int] = []
    heights: List[int] = []
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, font, text_scale, text_thickness)
        widths.append(int(w))
        heights.append(int(h))
    box_w = (max(widths) if widths else 0) + margin * 2
    box_h = (len(lines) * line_height) + margin * 2
    x1 = max(0, x0 - margin)
    y1 = max(0, y0 - heights[0] - margin if heights else y0 - margin)
    x2 = min(image.shape[1] - 1, x1 + box_w)
    y2 = min(image.shape[0] - 1, y1 + box_h)

    # Dark semi-transparent background.
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.58, image, 0.42, 0, image)

    y = y0
    for idx, line in enumerate(lines):
        color = (240, 240, 240)
        if idx == 1 and "OK" in line:
            color = (80, 220, 80)
        if idx == 1 and "FAIL" in line:
            color = (80, 80, 230)
        cv2.putText(image, line, (x0, y), font, text_scale, color, text_thickness, cv2.LINE_AA)
        y += line_height


def process_images(
    images_dir: Path,
    camera_cfg: CameraConfig,
    output_csv: Path,
    output_json: Path,
    save_viz_dir: Optional[Path] = None,
    min_charuco_corners: int = 8,
) -> None:
    images_dir = images_dir.resolve()

    if not images_dir.exists():
        raise FileNotFoundError(str(images_dir))

    dictionary = aruco_dictionary_from_name(camera_cfg.dictionary_name)

    # OpenCV has had layout/legacy-pattern differences; in practice people often
    # also swap squares_x/squares_y by accident. We build a small candidate set
    # and choose the one that interpolates the most ChArUco corners per image.
    class _BoardCandidate:
        def __init__(self, squares_x: int, squares_y: int, legacy: bool):
            self.squares_x = int(squares_x)
            self.squares_y = int(squares_y)
            self.legacy = bool(legacy)
            self.board = _create_charuco_board(
                dictionary,
                self.squares_x,
                self.squares_y,
                camera_cfg.square_size_m,
                camera_cfg.marker_size_m,
            )
            if self.legacy and hasattr(self.board, "setLegacyPattern"):
                try:
                    self.board.setLegacyPattern(True)
                except Exception:
                    pass

    # Candidate order matters for tie-breaking (prefer config orientation, non-legacy).
    candidates: List[_BoardCandidate] = []
    for sx, sy in [(camera_cfg.squares_x, camera_cfg.squares_y), (camera_cfg.squares_y, camera_cfg.squares_x)]:
        for legacy in (False, True):
            candidates.append(_BoardCandidate(sx, sy, legacy))

    params = cv2.aruco.DetectorParameters()
    # Conservative defaults; can be tuned if needed.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = None
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not image_paths:
        raise ValueError(f"no images found in {images_dir}")

    if save_viz_dir is not None:
        save_viz_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    success_quats: List[np.ndarray] = []
    success_tvecs: List[np.ndarray] = []
    success_corners_px: List[np.ndarray] = []
    used_boards: List[Tuple[int, int, bool]] = []
    failures: List[str] = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            failures.append(img_path.name)
            rows.append(
                {
                    "filename": img_path.name,
                    "ok": 0,
                    "reason": "imread_failed",
                }
            )
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if detector is None:
            marker_corners, marker_ids, _rej = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        else:
            marker_corners, marker_ids, _rej = detector.detectMarkers(gray)

        n_markers = int(0 if marker_ids is None else len(marker_ids))

        charuco_corners = None
        charuco_ids = None
        n_charuco = 0
        best_candidate: Optional[_BoardCandidate] = None

        if marker_ids is not None and len(marker_ids) > 0:
            # Try each board candidate, pick the one that yields the most interpolated corners.
            best_n = -1
            best_interp = None
            for cand in candidates:
                try:
                    interp = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, cand.board)
                except Exception:
                    continue
                if not (isinstance(interp, tuple) and len(interp) >= 3):
                    continue
                _retval, ch_c, ch_i = interp[:3]
                n = int(0 if ch_c is None else len(ch_c))
                if n > best_n:
                    best_n = n
                    best_candidate = cand
                    best_interp = (ch_c, ch_i)

            if best_candidate is not None and best_interp is not None:
                charuco_corners, charuco_ids = best_interp
                if charuco_corners is not None:
                    n_charuco = int(len(charuco_corners))

        ok = 0
        reason = ""
        rvec = None
        tvec = None
        corners_px = None
        reproj_err_px = float("nan")

        if best_candidate is None:
            reason = "no_board_candidate_worked"
        elif charuco_corners is None or charuco_ids is None or n_charuco < min_charuco_corners:
            reason = f"not_enough_charuco_corners({n_charuco}<{min_charuco_corners})"
        else:
            # estimatePoseCharucoBoard returns (valid, rvec, tvec)
            valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                best_candidate.board,
                camera_cfg.camera_matrix,
                camera_cfg.dist_coeffs,
                None,
                None,
            )
            if not bool(valid):
                reason = "pose_estimation_failed"
            else:
                ok = 1
                # Board outer corners in board coordinate system (meters) for the chosen layout
                bw = best_candidate.squares_x * camera_cfg.square_size_m
                bh = best_candidate.squares_y * camera_cfg.square_size_m
                board_outer = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], dtype=np.float64)
                corners_img, _ = cv2.projectPoints(
                    board_outer, rvec, tvec, camera_cfg.camera_matrix, camera_cfg.dist_coeffs
                )
                corners_px = corners_img.reshape(-1, 2)  # (4,2)
                reproj_err_px = _compute_reproj_error_px(
                    best_candidate.board,
                    charuco_corners,
                    charuco_ids,
                    rvec,
                    tvec,
                    camera_cfg.camera_matrix,
                    camera_cfg.dist_coeffs,
                )

        row: Dict[str, Any] = {
            "filename": img_path.name,
            "ok": ok,
            "reason": reason,
            "n_markers": n_markers,
            "n_charuco": n_charuco,
            "reproj_err_px": reproj_err_px,
            "board_squares_x": "" if best_candidate is None else int(best_candidate.squares_x),
            "board_squares_y": "" if best_candidate is None else int(best_candidate.squares_y),
            "board_legacy": "" if best_candidate is None else int(bool(best_candidate.legacy)),
        }

        # 4 outer corners in pixels (top-left, top-right, bottom-right, bottom-left)
        if corners_px is not None:
            for idx, name in enumerate(("tl", "tr", "br", "bl")):
                row[f"{name}_u"] = float(corners_px[idx, 0])
                row[f"{name}_v"] = float(corners_px[idx, 1])
        else:
            for name in ("tl", "tr", "br", "bl"):
                row[f"{name}_u"] = ""
                row[f"{name}_v"] = ""

        if ok and rvec is not None and tvec is not None:
            rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            rv = rvec.reshape(-1)
            tv = tvec.reshape(-1)
            row["rvec_x"], row["rvec_y"], row["rvec_z"] = (float(rv[0]), float(rv[1]), float(rv[2]))
            row["tvec_x"], row["tvec_y"], row["tvec_z"] = (float(tv[0]), float(tv[1]), float(tv[2]))

            R, _ = cv2.Rodrigues(rvec)
            q = _rotation_matrix_to_quat_wxyz(R)
            row["quat_w"], row["quat_x"], row["quat_y"], row["quat_z"] = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

            success_quats.append(q)
            success_tvecs.append(tvec.reshape(3))
            success_corners_px.append(corners_px.copy())
            used_boards.append((int(best_candidate.squares_x), int(best_candidate.squares_y), bool(best_candidate.legacy)))
        else:
            row["rvec_x"] = row["rvec_y"] = row["rvec_z"] = ""
            row["tvec_x"] = row["tvec_y"] = row["tvec_z"] = ""
            row["quat_w"] = row["quat_x"] = row["quat_y"] = row["quat_z"] = ""
            if ok == 0:
                failures.append(img_path.name)

        if save_viz_dir is not None:
            viz = img.copy()

            # Draw detections when available.
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(viz, marker_corners, marker_ids)
            if charuco_corners is not None and charuco_ids is not None:
                try:
                    cv2.aruco.drawDetectedCornersCharuco(viz, charuco_corners, charuco_ids)
                except Exception:
                    pass

            # Draw pose axis and projected board corners for successful pose.
            if ok and rvec is not None and tvec is not None:
                cv2.drawFrameAxes(viz, camera_cfg.camera_matrix, camera_cfg.dist_coeffs, rvec, tvec, 0.05)
                if corners_px is not None:
                    ordered_names = ("TL", "TR", "BR", "BL")
                    for idx, p in enumerate(corners_px):
                        u, v = int(round(float(p[0]))), int(round(float(p[1])))
                        cv2.circle(viz, (u, v), 6, (0, 255, 255), 2)
                        cv2.putText(
                            viz,
                            ordered_names[idx],
                            (u + 6, v - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    poly = corners_px.reshape(-1, 1, 2).astype(np.int32)
                    cv2.polylines(viz, [poly], isClosed=True, color=(0, 255, 255), thickness=2)

            # Build overlay text.
            overlay_lines = [
                f"file: {img_path.name}",
                f"status: {'OK' if ok else 'FAIL'}",
                f"markers: {n_markers}   charuco: {n_charuco}   reproj(px): {reproj_err_px:.3f}" if np.isfinite(reproj_err_px)
                else f"markers: {n_markers}   charuco: {n_charuco}   reproj(px): n/a",
            ]
            if best_candidate is not None:
                overlay_lines.append(
                    f"board: {best_candidate.squares_x}x{best_candidate.squares_y} legacy={int(best_candidate.legacy)}"
                )
            else:
                overlay_lines.append("board: n/a")

            if ok and rvec is not None and tvec is not None:
                rv = np.asarray(rvec).reshape(-1)
                tv = np.asarray(tvec).reshape(-1)
                overlay_lines.append(f"rvec: [{rv[0]:.5f}, {rv[1]:.5f}, {rv[2]:.5f}]")
                overlay_lines.append(f"tvec(m): [{tv[0]:.5f}, {tv[1]:.5f}, {tv[2]:.5f}]")
                if corners_px is not None:
                    tl, tr, br, bl = corners_px
                    overlay_lines.append(f"TL({tl[0]:.1f},{tl[1]:.1f}) TR({tr[0]:.1f},{tr[1]:.1f})")
                    overlay_lines.append(f"BR({br[0]:.1f},{br[1]:.1f}) BL({bl[0]:.1f},{bl[1]:.1f})")
            else:
                overlay_lines.append(f"reason: {reason}")

            _put_overlay_lines(viz, overlay_lines, origin_xy=(18, 30))
            cv2.imwrite(str(save_viz_dir / img_path.name), viz)

        rows.append(row)

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "ok",
        "reason",
        "n_markers",
        "n_charuco",
        "reproj_err_px",
        "board_squares_x",
        "board_squares_y",
        "board_legacy",
        "tl_u",
        "tl_v",
        "tr_u",
        "tr_v",
        "br_u",
        "br_v",
        "bl_u",
        "bl_v",
        "rvec_x",
        "rvec_y",
        "rvec_z",
        "tvec_x",
        "tvec_y",
        "tvec_z",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=fieldnames)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow(r)

    # Summary JSON
    summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "images_dir": str(images_dir),
        "num_images": int(len(image_paths)),
        "num_success": int(len(success_tvecs)),
        "num_failed": int(len(image_paths) - len(success_tvecs)),
        "failed_filenames": failures,
        "camera": {
            "dictionary": camera_cfg.dictionary_name,
            "image_size": [int(camera_cfg.image_size[0]), int(camera_cfg.image_size[1])],
            "squares_x_cfg": camera_cfg.squares_x,
            "squares_y_cfg": camera_cfg.squares_y,
            "square_size_m": camera_cfg.square_size_m,
            "marker_size_m": camera_cfg.marker_size_m,
            "camera_matrix": camera_cfg.camera_matrix.tolist(),
            "dist_coeffs": camera_cfg.dist_coeffs.reshape(-1).tolist(),
        },
    }

    if used_boards:
        # Most-common chosen board layout
        from collections import Counter

        cnt = Counter(used_boards)
        (sx, sy, legacy), _n = cnt.most_common(1)[0]
        summary["board_selection"] = {
            "strategy": "max_interpolated_charuco_corners_per_image",
            "most_common": {
                "squares_x": int(sx),
                "squares_y": int(sy),
                "legacy": bool(legacy),
            },
        }

    if success_tvecs:
        t = np.stack(success_tvecs, axis=0)  # (N,3)
        t_mean = t.mean(axis=0)
        t_std = t.std(axis=0)

        q_avg = average_quaternions_wxyz(success_quats)
        R_avg = _quat_wxyz_to_rotation_matrix(q_avg)
        rvec_avg, _ = cv2.Rodrigues(R_avg)

        corners = np.stack(success_corners_px, axis=0)  # (N,4,2)
        corners_mean = corners.mean(axis=0)
        corners_std = corners.std(axis=0)

        summary["mean"] = {
            "tvec": [float(x) for x in t_mean.reshape(-1)],
            "tvec_std": [float(x) for x in t_std.reshape(-1)],
            "quat_wxyz": [float(x) for x in q_avg.reshape(-1)],
            "rvec": [float(x) for x in rvec_avg.reshape(-1)],
            "board_outer_corners_px_order": ["tl", "tr", "br", "bl"],
            "board_outer_corners_px": corners_mean.tolist(),
            "board_outer_corners_px_std": corners_std.tolist(),
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch process ChArUco board images and export pose/corners.")
    parser.add_argument(
        "--camera_yaml",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "camera_cfg.yaml",
        help="Path to camera_cfg.yaml",
    )
    parser.add_argument("--camera_key", type=str, default="camera_1", help="Key inside camera_cfg.yaml")
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "temp" / "08",
        help="Folder with images",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Per-image output CSV",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Summary output JSON",
    )
    parser.add_argument(
        "--viz_dir",
        type=Path,
        default=None,
        help="Directory to save visualization images (detections + pose + text overlay).",
    )
    parser.add_argument(
        "--min_charuco_corners",
        type=int,
        default=8,
        help="Minimum interpolated charuco corners required to attempt pose estimation.",
    )
    args = parser.parse_args()

    camera_cfg = load_camera_config(args.camera_yaml, args.camera_key)

    # Some datasets folders (like temp/08) may be read-only; default to repo output/.
    repo_root = Path(__file__).resolve().parents[1]
    # Keep folder naming straightforward: temp/06 -> output/06
    default_out_dir = repo_root / "output" / f"{args.images_dir.name}"
    if args.output_csv is None:
        args.output_csv = default_out_dir / "charuco_pose.csv"
    if args.output_json is None:
        args.output_json = default_out_dir / "charuco_pose_summary.json"
    if args.viz_dir is None:
        args.viz_dir = default_out_dir / "images_annotated"

    process_images(
        images_dir=args.images_dir,
        camera_cfg=camera_cfg,
        output_csv=args.output_csv,
        output_json=args.output_json,
        save_viz_dir=args.viz_dir,
        min_charuco_corners=args.min_charuco_corners,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
