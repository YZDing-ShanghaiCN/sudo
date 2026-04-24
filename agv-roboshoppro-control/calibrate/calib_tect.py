"""计算相机内参和RMS的脚本 - 基于已拍好的照片

功能：
1. 从照片目录读取所有图片
2. 检测 ArUco 标记和 ChArUco 角点
3. 进行标定计算
4. 显示内参(Camera Matrix)、畸变系数(Dist Coeffs)和RMS
5. 不保存结果

使用示例:
    python calibrate/calib_tect.py --image-dir ./runs_charuco/session_20240101_120000/images
    python calibrate/calib_tect.py --image-dir ./images --squares-x 9 --squares-y 14 --square-size 20 --marker-size 15
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算相机内参和RMS - 基于已拍好的照片")
    parser.add_argument("--image-dir", type=str, required=True, help="照片目录路径")
    parser.add_argument("--squares-x", type=int, default=9, help="ChArUco 棋盘X方向格子数。默认: 9")
    parser.add_argument("--squares-y", type=int, default=14, help="ChArUco 棋盘Y方向格子数。默认: 14")
    parser.add_argument("--square-size", type=float, default=20.0, help="格子大小(mm)。默认: 20")
    parser.add_argument("--marker-size", type=float, default=15.0, help="标记大小(mm)。默认: 15")
    parser.add_argument(
        "--dictionary",
        type=str,
        default="DICT_5X5_100",
        help="ArUco 字典名称。默认: DICT_5X5_100"
    )
    parser.add_argument(
        "--min-charuco",
        type=int,
        default=12,
        help="每张图中要求的最少ChArUco角点数。默认: 12"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="使用legacy标定板pattern"
    )
    return parser.parse_args()


def create_charuco_board(
    squares_x: int,
    squares_y: int,
    square_size_mm: float,
    marker_size_mm: float,
    aruco_dict,
    legacy_pattern: bool = False,
):
    """创建ChArUco标定板"""
    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            board = cv2.aruco.CharucoBoard(
                (squares_x, squares_y),
                square_size_mm,
                marker_size_mm,
                aruco_dict,
            )
            if legacy_pattern and hasattr(board, "setLegacyPattern"):
                board.setLegacyPattern(True)
            return board
        except TypeError:
            pass

    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            square_size_mm,
            marker_size_mm,
            aruco_dict,
        )
        if legacy_pattern and hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(True)
        return board

    raise RuntimeError("CharucoBoard API is unavailable in current OpenCV build.")


def make_detector(dictionary_name: str) -> Tuple[object, Callable]:
    """创建ArUco检测器"""
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary name: {dictionary_name}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))

    # 尝试使用新的API
    if hasattr(cv2.aruco, "DetectorParameters"):
        try:
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)

            def _detect(gray: np.ndarray) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
                corners, ids, _ = detector.detectMarkers(gray)
                return list(corners), ids

            return aruco_dict, _detect
        except TypeError:
            pass

    # 使用旧的API
    def _detect(gray: np.ndarray) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        return list(corners), ids

    return aruco_dict, _detect


def interpolate_charuco(
    gray: np.ndarray,
    marker_corners: List[np.ndarray],
    marker_ids: np.ndarray,
    board,
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """插值ChArUco角点"""
    try:
        num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
        )
    except TypeError:
        num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=marker_corners,
            markerIds=marker_ids,
            image=gray,
            board=board,
        )

    count = int(num) if num is not None else 0
    if count <= 0 or charuco_corners is None or charuco_ids is None:
        return 0, None, None
    return count, charuco_corners, charuco_ids


def calibrate_charuco(
    charuco_corners: List[np.ndarray],
    charuco_ids: List[np.ndarray],
    board,
    image_size: Tuple[int, int],
):
    """使用ChArUco角点进行标定"""
    if hasattr(cv2.aruco, "calibrateCameraCharuco"):
        try:
            return cv2.aruco.calibrateCameraCharuco(
                charuco_corners,
                charuco_ids,
                board,
                image_size,
                None,
                None,
            )
        except TypeError:
            return cv2.aruco.calibrateCameraCharuco(
                charuco_corners,
                charuco_ids,
                board,
                image_size,
            )

    if hasattr(cv2.aruco, "calibrateCameraCharucoExtended"):
        return cv2.aruco.calibrateCameraCharucoExtended(
            charuco_corners,
            charuco_ids,
            board,
            image_size,
            None,
            None,
        )

    raise RuntimeError("calibrateCameraCharuco is unavailable in current OpenCV build.")


def get_board_chessboard_corners(board) -> Optional[np.ndarray]:
    """获取标定板的棋盘角点"""
    if hasattr(board, "getChessboardCorners"):
        return np.array(board.getChessboardCorners(), dtype=np.float32)
    if hasattr(board, "chessboardCorners"):
        return np.array(board.chessboardCorners, dtype=np.float32)
    return None


def compute_mean_reprojection_error(
    board,
    charuco_corners: List[np.ndarray],
    charuco_ids: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Optional[float]:
    """计算平均重投影误差"""
    board_corners = get_board_chessboard_corners(board)
    if board_corners is None:
        return None

    errors = []
    for corners_i, ids_i, rvec, tvec in zip(charuco_corners, charuco_ids, rvecs, tvecs):
        if ids_i is None or len(ids_i) == 0:
            continue
        indices = ids_i.reshape(-1)
        if np.max(indices) >= len(board_corners):
            continue

        obj_points = board_corners[indices]
        proj, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
        err = cv2.norm(corners_i, proj, cv2.NORM_L2) / max(len(proj), 1)
        errors.append(float(err))

    if not errors:
        return None
    return float(np.mean(errors))


def compute_per_image_errors(
    image_dir: Path,
    board,
    detect: Callable,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    charuco_ids_all: List[np.ndarray],
) -> List[tuple]:
    """计算每张图片的重投影误差"""
    board_corners = get_board_chessboard_corners(board)
    if board_corners is None:
        return []

    images_path = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    errors_per_image = []
    
    used_idx = 0
    for img_path in images_path:
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        marker_corners, marker_ids = detect(image)
        if marker_ids is None or len(marker_ids) == 0:
            continue

        count, charuco_corners, charuco_ids = interpolate_charuco(
            image,
            marker_corners,
            marker_ids,
            board,
        )

        if count <= 0 or charuco_corners is None or charuco_ids is None:
            continue

        # 检查这个图片是否被用于标定
        if used_idx >= len(rvecs):
            break

        # 获取对应的 rvec 和 tvec
        rvec = rvecs[used_idx]
        tvec = tvecs[used_idx]
        c_ids = charuco_ids_all[used_idx]

        # 计算误差
        indices = c_ids.reshape(-1)
        if np.max(indices) >= len(board_corners):
            used_idx += 1
            continue

        obj_points = board_corners[indices]
        proj, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
        
        # 计算每个点的误差
        diff = charuco_corners.reshape(-1, 2) - proj.reshape(-1, 2)
        per_point_errors = np.sqrt(np.sum(diff**2, axis=1))
        
        # 计算这张图的平均误差
        img_mean_error = float(np.mean(per_point_errors))
        img_max_error = float(np.max(per_point_errors))
        img_min_error = float(np.min(per_point_errors))
        
        errors_per_image.append({
            'image': img_path.name,
            'mean': img_mean_error,
            'max': img_max_error,
            'min': img_min_error,
            'points': len(per_point_errors),
        })
        
        used_idx += 1

    return errors_per_image


def process_images(
    image_dir: Path,
    aruco_dict,
    detect: Callable,
    board,
    min_charuco: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int], int]:
    """处理照片目录中的所有图片"""
    images_path = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    
    if not images_path:
        raise RuntimeError(f"No images found in {image_dir}")

    charuco_corners_all: List[np.ndarray] = []
    charuco_ids_all: List[np.ndarray] = []
    image_size = (0, 0)
    processed_count = 0

    print(f"[INFO] 开始处理 {len(images_path)} 张图片...")
    print(f"\n{'图片名':<20} | {'检测到标记':<6} | {'ChArUco角点':<8} | {'状态':<15}")
    print("-" * 60)

    for img_path in images_path:
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"{img_path.name:<20} | {'N/A':<6} | {'N/A':<8} | {'读取失败':<15}")
            continue

        image_size = (image.shape[1], image.shape[0])
        
        marker_corners, marker_ids = detect(image)
        marker_count = 0 if marker_ids is None else int(marker_ids.size)

        if marker_ids is None or marker_count == 0:
            print(f"{img_path.name:<20} | {marker_count:<6} | {'0':<8} | {'无标记':<15}")
            continue

        count, charuco_corners, charuco_ids = interpolate_charuco(
            image,
            marker_corners,
            marker_ids,
            board,
        )

        if count < min_charuco or charuco_corners is None or charuco_ids is None:
            print(f"{img_path.name:<20} | {marker_count:<6} | {count:<8} | {'不足最小值':<15}")
            continue

        charuco_corners_all.append(charuco_corners)
        charuco_ids_all.append(charuco_ids)
        processed_count += 1
        print(f"{img_path.name:<20} | {marker_count:<6} | {count:<8} | {'✓ 已使用':<15}")

    print("-" * 60)
    print(f"[INFO] 共处理了 {processed_count} 张有效图片")

    if processed_count < 3:
        raise RuntimeError(f"有效图片数量 {processed_count} 不足,至少需要3张")

    return charuco_corners_all, charuco_ids_all, image_size, processed_count


def main() -> int:
    args = parse_args()

    try:
        # 1. 参数验证
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            print(f"[ERROR] 图片目录不存在: {image_dir}")
            return 2

        if args.marker_size >= args.square_size:
            print("[ERROR] marker-size 必须小于 square-size")
            return 2

        # 2. 创建ArUco字典和检测器
        aruco_dict, detect = make_detector(args.dictionary)
        print(f"[INFO] 使用 ArUco 字典: {args.dictionary}")

        # 3. 创建标定板
        board = create_charuco_board(
            args.squares_x,
            args.squares_y,
            args.square_size,
            args.marker_size,
            aruco_dict,
            args.legacy,
        )
        print(f"[INFO] 标定板配置: {args.squares_x}x{args.squares_y}, 格子大小={args.square_size}mm, 标记大小={args.marker_size}mm")

        # 4. 处理所有图片
        charuco_corners_all, charuco_ids_all, image_size, processed_count = process_images(
            image_dir,
            aruco_dict,
            detect,
            board,
            args.min_charuco,
        )

        # 5. 执行标定
        print(f"\n[INFO] 开始标定...")
        calib_out = calibrate_charuco(charuco_corners_all, charuco_ids_all, board, image_size)
        
        rms = float(calib_out[0])
        camera_matrix = np.array(calib_out[1], dtype=np.float64)
        dist_coeffs = np.array(calib_out[2], dtype=np.float64)
        rvecs = list(calib_out[3])
        tvecs = list(calib_out[4])

        # 6. 计算平均重投影误差
        mean_error = compute_mean_reprojection_error(
            board,
            charuco_corners_all,
            charuco_ids_all,
            rvecs,
            tvecs,
            camera_matrix,
            dist_coeffs,
        )

        # 6.5 计算每张图片的误差
        errors_per_image = compute_per_image_errors(
            image_dir,
            board,
            detect,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
            charuco_ids_all,
        )

        # 7. 显示结果
        print("\n" + "="*70)
        print("【标定结果】")
        print("="*70)
        print(f"图像尺寸: {image_size[0]} x {image_size[1]}")
        print(f"已使用图片数: {processed_count}")
        print(f"\n📷 相机内参矩阵 (Camera Matrix):")
        print(f"┌{camera_matrix[0, :]}")
        print(f"├{camera_matrix[1, :]}")
        print(f"└{camera_matrix[2, :]}")
        
        print(f"\n🔧 畸变系数 (Dist Coeffs): {dist_coeffs.flatten()}")
        
        print(f"\n📊 标定质量:")
        print(f"   RMS (OpenCV calibration error): {rms:.6f}")
        if mean_error is not None:
            print(f"   Mean Reprojection Error: {mean_error:.6f} px")
        print("="*70)

        # 7.5 显示每张图片的误差
        if errors_per_image:
            print(f"\n📸 每张图片的重投影误差:")
            print(f"{'图片名':<20} | {'平均误差 (px)':<15} | {'最大误差':<15} | {'最小误差':<15} | {'角点数':<8}")
            print("-" * 80)
            
            for err_info in errors_per_image:
                print(f"{err_info['image']:<20} | {err_info['mean']:<15.6f} | {err_info['max']:<15.6f} | {err_info['min']:<15.6f} | {err_info['points']:<8}")
            
            print("-" * 80)
            all_means = [e['mean'] for e in errors_per_image]
            print(f"{'总体统计':<20} | {np.mean(all_means):<15.6f} | {np.max(all_means):<15.6f} | {np.min(all_means):<15.6f} | {'ALL':<8}")
            print("-" * 80)

        # 8. 显示为JSON格式(便于复制使用)
        print("\n📋 JSON 格式 (可直接复制到配置文件):")
        print(json.dumps({
            "image_size": list(image_size),
            "used_images": processed_count,
            "rms": float(rms),
            "mean_reprojection_error": float(mean_error) if mean_error else None,
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist() if len(dist_coeffs.shape) == 1 else dist_coeffs.tolist()[0],
        }, indent=2, ensure_ascii=False))

        print("\n[INFO] 标定完成!")
        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())