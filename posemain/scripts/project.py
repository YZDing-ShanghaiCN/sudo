#!/usr/bin/env python3
import os
import sys
import gc
import json
import argparse
import logging
import contextlib
import types
import importlib
from typing import Tuple, Union, Optional

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
import nvdiffrast.torch as dr

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from segment_anything import sam_model_registry, SamPredictor
from omegaconf import OmegaConf


code_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(code_dir)
sys.path.insert(0, project_root)

import Utils as U
from core.utils.utils import InputPadder
from core.foundation_stereo import FoundationStereo


torch.serialization.add_safe_globals([DetectionModel])


@contextlib.contextmanager
def torch_load_weights_only_false():
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def disable_broken_xformers_if_needed(force_disable=False):
    if not force_disable:
        try:
            import xformers.ops  # noqa: F401
            return
        except Exception as e:
            logging.warning(f"xformers import failed, fallback to no-xformers mode: {e}")
    else:
        logging.info("stereo_device=cpu, force disabling xformers")

    for name in list(sys.modules.keys()):
        if name == "xformers" or name.startswith("xformers."):
            sys.modules.pop(name, None)

    stub_xformers = types.ModuleType("xformers")
    sys.modules["xformers"] = stub_xformers


def rectify_stereo_images(
    img_left: Union[str, np.ndarray],
    img_right: Union[str, np.ndarray],
    K1: np.ndarray,
    D1: Optional[np.ndarray],
    K2: np.ndarray,
    D2: Optional[np.ndarray],
    R: np.ndarray,
    T: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(img_left, str):
        img_left_arr = cv2.imread(img_left)
    else:
        img_left_arr = img_left
    if isinstance(img_right, str):
        img_right_arr = cv2.imread(img_right)
    else:
        img_right_arr = img_right

    height, width = img_left_arr.shape[:2]

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        (width, height),
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=1,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)

    rect_left = cv2.remap(img_left_arr, map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right_arr, map2x, map2y, cv2.INTER_LINEAR)

    return rect_left, rect_right, R1, R2, P1, P2


def get_bboxes(img_bgr, yolo_model, conf=0.7):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model(img_rgb, verbose=False)[0]

    if results.boxes is None:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    keep = scores > conf
    return boxes[keep]


def generate_3d_point_cloud(img_bgr, depth, K, z_far, mask=None):
    xyz_map = U.depth2xyzmap(depth, K)
    points = xyz_map.reshape(-1, 3)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    colors = img_rgb.reshape(-1, 3)

    if mask is not None:
        mask_flat = mask.reshape(-1).astype(bool)
        points = points[mask_flat]
        colors = colors[mask_flat]

    pcd = U.toOpen3dCloud(points, colors)
    pts = np.asarray(pcd.points)
    keep_mask = (pts[:, 2] > 0) & (pts[:, 2] <= z_far)
    keep_ids = np.where(keep_mask)[0]
    pcd = pcd.select_by_index(keep_ids)
    return pcd


def build_mesh_from_depth_mask(rgb, depth, K, ob_mask):
    pcd = o3d.geometry.PointCloud()
    xyz_map = U.depth2xyzmap(depth, K)
    valid = (depth > 0.001) & ob_mask
    pcd.points = o3d.utility.Vector3dVector(xyz_map[valid])
    pcd.colors = o3d.utility.Vector3dVector(rgb[valid] / 255.0)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=8,
        width=0,
        scale=1.0,
        linear_fit=False,
    )[0]
    poisson_mesh.remove_degenerate_triangles()
    poisson_mesh.remove_duplicated_triangles()
    poisson_mesh.remove_duplicated_vertices()
    poisson_mesh.remove_non_manifold_edges()
    poisson_mesh = poisson_mesh.simplify_quadric_decimation(target_number_of_triangles=15000)

    vertices = np.asarray(poisson_mesh.vertices)
    center = vertices.mean(axis=0)
    vertices -= center

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(poisson_mesh.triangles),
        vertex_normals=np.asarray(poisson_mesh.vertex_normals),
        process=True,
    )
    mesh.vertices += center
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    return mesh


def run_depth_distance_window(depth, K, z_far, out_dir):
    H, W = depth.shape[:2]
    save_index = 0
    last_canvas = {"img": None}

    depth_vis = U.vis_disparity(depth, max_val=z_far)
    if depth_vis.ndim == 2:
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
    depth_vis_base = depth_vis.copy()
    depth_window_name = "Depth Distance Measurement"
    clicked_points = []

    def pixel_to_camera_point(u, v):
        z = depth[v, u]
        if z <= 0 or not np.isfinite(z):
            return None
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float64)

    def redraw_depth_window():
        canvas = depth_vis_base.copy()

        for pt in clicked_points:
            cv2.circle(canvas, pt, 4, (0, 255, 0), -1)

        if len(clicked_points) >= 2:
            p1 = clicked_points[-2]
            p2 = clicked_points[-1]
            p1_3d = pixel_to_camera_point(*p1)
            p2_3d = pixel_to_camera_point(*p2)

            if p1_3d is not None and p2_3d is not None:
                dist = np.linalg.norm(p1_3d - p2_3d)
                cv2.line(canvas, p1, p2, (0, 255, 255), 2)
                text_pos = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.putText(canvas, f"{dist:.4f} m", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "Invalid depth for selected point", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        last_canvas["img"] = canvas
        cv2.imshow(depth_window_name, canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= x < W and 0 <= y < H:
                clicked_points.append((x, y))
                redraw_depth_window()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(clicked_points) > 0:
                clicked_points.pop()
                redraw_depth_window()

    os.makedirs(os.path.join(out_dir, "distance"), exist_ok=True)
    cv2.namedWindow(depth_window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(depth_window_name, on_mouse)
    redraw_depth_window()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and last_canvas["img"] is not None:
            save_path = f"{out_dir}/distance/detectsave{save_index}.png"
            cv2.imwrite(save_path, last_canvas["img"])
            print(f"[INFO] saved {save_path}")
            save_index += 1

    cv2.destroyWindow(depth_window_name)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--scale', default=1, type=float)
    parser.add_argument('--hiera', default=0, type=int)
    parser.add_argument('--z_far', default=10, type=float)
    parser.add_argument('--valid_iters', type=int, default=32)
    parser.add_argument('--get_pc', type=int, default=1)
    parser.add_argument('--remove_invisible', default=1, type=int)
    parser.add_argument('--denoise_cloud', type=int, default=1)
    parser.add_argument('--denoise_nb_points', type=int, default=30)
    parser.add_argument('--denoise_radius', type=float, default=0.03)

    parser.add_argument('--mesh', type=int, default=0, help='0 reconstruct, 1 load mesh')
    parser.add_argument('--mesh_path', type=str, default='./pre_result/crate/crate_0_visual.ply')
    parser.add_argument('--pose_iter', type=int, default=5)

    parser.add_argument('--show_rectified', type=int, default=1)
    parser.add_argument('--show_bbox', type=int, default=1)
    parser.add_argument('--show_distance', type=int, default=1)
    parser.add_argument('--show_masks', type=int, default=0)
    parser.add_argument('--show_pose', type=int, default=1)
    parser.add_argument('--show_pose_3d', type=int, default=1)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--yolo_conf', type=float, default=0.7)

    parser.add_argument('--in_dir', type=str, default=f"{project_root}/assets/hand_camera_2")
    parser.add_argument('--json_file', type=str, default='')
    parser.add_argument('--left_file', type=str, default='')
    parser.add_argument('--right_file', type=str, default='')

    parser.add_argument('--out_dir', type=str, default=f"{project_root}/test_outputs/test2")
    parser.add_argument('--save_dir', type=str, default='/home/user/Desktop/main/posemain/detection_result')
    parser.add_argument('--debug_dir', type=str, default='/home/user/Desktop/main/posemain/debug/foundationstereo')

    parser.add_argument('--foundation_ckpt', type=str, default='/home/user/Desktop/checkpoints/foundationstereo/23-51-11')
    parser.add_argument('--yolo_ckpt', type=str, default='/home/user/Desktop/checkpoints/yolo/yolov8l.pt')
    parser.add_argument('--sam_ckpt', type=str, default='/home/user/Desktop/checkpoints/sam/sam_vit_b_01ec64.pth')

    parser.add_argument('--det_device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--sam_device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--stereo_device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--pose_device', type=str, default='cuda', choices=['cuda'])

    parser.add_argument('--pose_source_utils', type=str, default=f'{project_root}/utilcopy.py')
    args = parser.parse_args()

    U.set_logging_format()
    U.set_seed(args.seed)
    torch.autograd.set_grad_enabled(False)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(project_root, 'pre_result'), exist_ok=True)

    json_file = args.json_file if args.json_file else f"{args.in_dir}/hand_camera_data.json"
    left_file = args.left_file if args.left_file else f"{args.in_dir}/left_hand_left_camera.png"
    right_file = args.right_file if args.right_file else f"{args.in_dir}/left_hand_right_camera.png"

    with torch_load_weights_only_false():
        yolo_model = YOLO(args.yolo_ckpt)
    yolo_model.to(args.det_device)

    sam = sam_model_registry['vit_b'](checkpoint=args.sam_ckpt)
    if args.sam_device == 'cuda':
        sam.cuda()
    else:
        sam.cpu()
    predictor = SamPredictor(sam)

    cfg = OmegaConf.load(os.path.join(args.foundation_ckpt, 'cfg.yaml'))
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    fs_args = OmegaConf.create(cfg)

    disable_broken_xformers_if_needed(force_disable=(args.stereo_device == 'cpu'))

    model = FoundationStereo(fs_args)
    ckpt = torch.load(os.path.join(args.foundation_ckpt, 'model_best_bp2.pth'), weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(args.stereo_device)
    model.eval()

    img0 = cv2.imread(left_file)
    img1 = cv2.imread(right_file)
    img0 = cv2.resize(img0, fx=args.scale, fy=args.scale, dsize=None)
    img1 = cv2.resize(img1, fx=args.scale, fy=args.scale, dsize=None)
    H, W = img0.shape[:2]

    with open(json_file, 'r') as f:
        json_data = json.load(f)

    K = np.array(json_data['camera_data']['left_hand_left_camera']['intrinsics'], dtype=np.float64)
    K_right = np.array(json_data['camera_data']['left_hand_right_camera']['intrinsics'], dtype=np.float64)
    Tc2w_left = np.array(json_data['camera_data']['left_hand_left_camera']['extrinsics'], dtype=np.float64)
    Tc2w_right = np.array(json_data['camera_data']['left_hand_right_camera']['extrinsics'], dtype=np.float64)
    K[:2] *= args.scale
    K_right[:2] *= args.scale
    matrix = np.linalg.inv(Tc2w_right) @ Tc2w_left

    img0, img1, _, _, P1, P2 = rectify_stereo_images(img0, img1, K, None, K_right, None, matrix[:3, :3], matrix[:3, 3])

    if args.show_rectified:
        img_concat = np.concatenate((img0, img1), axis=1)
        cv2.imshow('Rectified Stereo Pair', img_concat)
        cv2.waitKey(0)

    bboxes = get_bboxes(img0, yolo_model, conf=args.yolo_conf)
    print(f'[INFO] detected {len(bboxes)} objects')

    if args.show_bbox:
        img0_vis = img0.copy()
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img0_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Detected Bounding Boxes', img0_vis)
        cv2.waitKey(0)

    predictor.set_image(img0)

    img0_t = torch.as_tensor(img0).to(args.stereo_device).float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1).to(args.stereo_device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_t.shape, divis_by=32)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(args.stereo_device == 'cuda')):
            disp = model.forward(img0_t, img1_t, iters=args.valid_iters, test_mode=True)

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)

    baseline = abs(P2[0, 3] / P2[0, 0])
    print(f'[DEBUG] Computed baseline: {baseline:.4f}')

    K_rect = P1[:3, :3]
    valid = disp > 1e-6
    depth = np.zeros_like(disp)
    depth[valid] = K_rect[0, 0] * baseline / disp[valid]

    cv2.imwrite(f"{args.out_dir}/disparity0.png", U.vis_disparity(disp, args.z_far))
    cv2.imwrite(f"{args.out_dir}/depth0.png", U.vis_disparity(depth, max_val=args.z_far))

    if args.show_distance:
        run_depth_distance_window(depth, K_rect, args.z_far, args.out_dir)

    if args.stereo_device == 'cuda':
        model.to('cpu')
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    U.use_pose_implementations(overwrite=True, pose_utils_path=args.pose_source_utils)
    sys.modules['Utils'] = U

    if 'estimater' in sys.modules:
        estimater_mod = importlib.reload(sys.modules['estimater'])
    else:
        estimater_mod = importlib.import_module('estimater')

    FoundationPose = estimater_mod.FoundationPose
    ScorePredictor = estimater_mod.ScorePredictor
    PoseRefinePredictor = estimater_mod.PoseRefinePredictor

    draw_xyz_axis_pose = getattr(U, 'draw_xyz_axis_pose', None)
    draw_posed_3d_box_pose = getattr(U, 'draw_posed_3d_box_pose', None)
    if draw_xyz_axis_pose is None:
        draw_xyz_axis_pose = U.draw_xyz_axis
    if draw_posed_3d_box_pose is None:
        draw_posed_3d_box_pose = U.draw_posed_3d_box

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    mask_list = []

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        input_box = np.array([x1, y1, x2, y2])
        masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
        ob_mask = masks[0].astype(bool)
        mask_list.append(ob_mask.astype(np.uint8))

        if args.show_masks:
            cv2.imshow(f'Mask {i}', ob_mask.astype(np.uint8) * 255)
            cv2.waitKey(0)

        if args.mesh == 0:
            mesh = build_mesh_from_depth_mask(rgb, depth, K_rect, ob_mask)
            print('\nMesh reconstructed successfully!\n')
        else:
            mesh = trimesh.load(args.mesh_path, process=True)
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.update_faces(mesh.unique_faces())
            mesh.merge_vertices()
            print('\nMesh loaded successfully!\n')

        est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=False,
            debug_dir=args.debug_dir,
        )

        vis = rgb.copy()
        pose = est.register(K=K_rect, rgb=vis, depth=depth, ob_mask=ob_mask, iteration=args.pose_iter)
        print(f'\nPose estimation completed! Estimated pose:\n{pose}\n')

        bbox_3d = np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
        vis = draw_posed_3d_box_pose(K_rect, img=vis, ob_in_cam=pose, bbox=bbox_3d)
        vis = draw_xyz_axis_pose(vis, ob_in_cam=pose, scale=0.25, K=K_rect, thickness=6, transparency=0, is_input_rgb=True)

        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.save_dir, f'result_{i:02d}.png'), vis_bgr)

        if args.show_pose:
            cv2.imshow('Estimated Pose', vis_bgr[..., ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if args.show_pose_3d:
            pcd = o3d.geometry.PointCloud()
            xyz_map = U.depth2xyzmap(depth, K_rect)
            valid_mask = (depth > 0.001) & ob_mask
            pcd.points = o3d.utility.Vector3dVector(xyz_map[valid_mask])
            pcd.colors = o3d.utility.Vector3dVector(rgb[valid_mask] / 255.0)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
            pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

            axis_length = 0.25
            axis_points = np.array([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
            c = np.mean(np.asarray(pcd.points), axis=0)
            axis_points_transformed = c + (pose[:3, :3] @ axis_points.T).T
            axis_lines = [[0, 1], [0, 2], [0, 3]]
            axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            axis_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(axis_points_transformed),
                lines=o3d.utility.Vector2iVector(axis_lines),
            )
            axis_line_set.colors = o3d.utility.Vector3dVector(axis_colors)
            o3d.visualization.draw_geometries([pcd, axis_line_set], window_name='Final Pose Visualization', width=800, height=600)

if __name__ == '__main__':
    main()
