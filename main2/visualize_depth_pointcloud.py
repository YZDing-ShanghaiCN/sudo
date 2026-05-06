import argparse
import os
import numpy as np
import OpenEXR
import Imath


def read_exr_depth(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EXR file not found: {file_path}")

    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = list(header["channels"].keys())
    channel = channels[0]
    data = exr_file.channel(channel, float_type)
    exr_file.close()

    depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
    return depth


def load_intrinsics(file_path):
    K = np.loadtxt(file_path, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics, got {K.shape} from {file_path}")
    return K


def depth_to_point_cloud(depth, K, min_depth, max_depth, stride):
    h, w = depth.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    grid_x, grid_y = np.meshgrid(xs, ys)

    z = depth[grid_y, grid_x]
    valid = np.isfinite(z)
    if min_depth is not None:
        valid &= z >= min_depth
    if max_depth is not None:
        valid &= z <= max_depth

    grid_x = grid_x[valid].astype(np.float32)
    grid_y = grid_y[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    x = (grid_x - cx) * z / fx
    y = (grid_y - cy) * z / fy

    points = np.stack([x, y, z], axis=1)
    return points


def save_ply(file_path, points):
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    with open(file_path, "w", encoding="ascii") as f:
        f.write("\n".join(header))
        f.write("\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def interactive_measure(points):
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError("Open3D is required for interactive measurement.") from exc

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick points to measure distance")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()
    if len(picked) < 2:
        print("Need at least two picked points to measure distance.")
        return

    p0 = points[picked[-2]]
    p1 = points[picked[-1]]
    dist = np.linalg.norm(p0 - p1)
    print(f"Distance between last two picks: {dist:.6f} meters")


def visualize_point_cloud(points):
    try:
        import open3d as o3d
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("Open3D and matplotlib are required for visualization.") from exc

    pcd = o3d.geometry.PointCloud()
    
    # Open3D 的默认视角是 +Y 朝上，+Z 朝向屏幕外，而 OpenCV 是 +Y 朝下，+Z 朝向屏幕内
    # 这里我们做一个坐标系翻转，以免看起来是倒着的
    flipped_points = points.copy()
    flipped_points[:, 1] *= -1  # 翻转 Y
    flipped_points[:, 2] *= -1  # 翻转 Z
    
    pcd.points = o3d.utility.Vector3dVector(flipped_points)

    # 我们通过深度值Z上个色，让它看起来更直观！
    z_values = points[:, 2]
    z_min, z_max = np.percentile(z_values, 1), np.percentile(z_values, 99)
    z_norm = np.clip((z_values - z_min) / (z_max - z_min + 1e-6), 0, 1)
    
    cmap = plt.get_cmap("turbo") # 或者用 viridis, jet
    colors = cmap(z_norm)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 为了让视图一开始就正对着物体，计算物体的包围盒中心
    center = pcd.get_center()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Depth Point Cloud Visualization (Open3D)")
    # 为了更好看，调一下背景颜色和点大小
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def main():
    depth_path = "./near_pose/output/000003_depth_pred0.exr"
    intrinsics_path = "./near_pose/intrinsics/left_hand_left_camera_intrinsics.txt"
    parser = argparse.ArgumentParser(description="Visualize depth as point cloud and measure real-world distance.")
    parser.add_argument("--depth", default=depth_path, help="Path to depth EXR file (meters).")
    parser.add_argument("--intrinsics", default=intrinsics_path, help="Path to 3x3 intrinsics txt file.")
    parser.add_argument("--output", default="depth_points.ply", help="Output PLY path.")
    parser.add_argument("--min-depth", type=float, default=None, help="Minimum depth in meters.")
    parser.add_argument("--max-depth", type=float, default=None, help="Maximum depth in meters.")
    parser.add_argument("--stride", type=int, default=2, help="Stride for sampling pixels.")
    parser.add_argument("--measure", action="store_true", help="Open interactive viewer to measure distance.")
    parser.add_argument("--save", action="store_true", help="Save point cloud to PLY.")
    parser.add_argument("--no-show", action="store_true", help="Disable visualization window.")
    args = parser.parse_args()

    depth = read_exr_depth(args.depth)
    K = load_intrinsics(args.intrinsics)
    points = depth_to_point_cloud(depth, K, args.min_depth, args.max_depth, args.stride)

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        save_ply(args.output, points)
        print(f"Saved point cloud: {args.output}")
    print(f"Points: {points.shape[0]}")

    if not args.no_show:
        if args.measure:
            interactive_measure(points)
        else:
            visualize_point_cloud(points)


if __name__ == "__main__":
    main()
