#!/usr/bin/env python3
import cv2
import torch
import numpy as np
import trimesh
import argparse
import sys
import open3d as o3d
import os
import imageio
import logging
from datetime import datetime
from tqdm import tqdm
from Utils import draw_xyz_axis, draw_posed_3d_box, depth2xyzmap, toOpen3dCloud
sys.path.append("MiDaS")
from midas.model_loader import load_model
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
import torchvision.transforms as T
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
import nvdiffrast.torch as dr

_midas_model = None
_midas_transform = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_midas():
    global _midas_model, _midas_transform
    if _midas_model is not None:
        return
    model_path = "/home/user/.cache/midas/dpt_large_384.pt"
    _midas_model = DPTDepthModel(
        path=model_path,
        backbone="vitl16_384",
        non_negative=True,
    )
    _midas_model.eval()
    _midas_model.to(_device)
    _midas_transform = T.Compose([
        Resize(
            384,
            384,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
    ])
    logging.info("\nMiDaS model loaded successfully\n")

def estimate_depth(rgb: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    """
        使用 MiDaS 估计深度，返回米制深度 (H, W)
    """
    init_midas()
    input_image = _midas_transform({"image": rgb})["image"]
    input_image = torch.from_numpy(input_image).unsqueeze(0).to(_device)
    
    with torch.no_grad():
        prediction = _midas_model(input_image)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=rgb.shape[:2], mode="bilinear", align_corners=False
        ).squeeze()
        depth = prediction.cpu().numpy()
    
    depth = 1.0 / (depth + 1e-8)
    depth = depth * scale_factor
    return depth.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="input directory name")
    parser.add_argument("--number", type=int, required=True, help="0 for kinect driller, 1 for mustard")
    args = parser.parse_args()

    set_logging_format = lambda: logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    set_logging_format()
    set_seed = lambda x: np.random.seed(x)
    set_seed(0)

    K_orig = np.array([[319.58200073, 0., 320.21498477], 
                       [0., 417.11868286, 244.34866809], 
                       [0., 0., 1.]
                      ], 
                      dtype=np.float32
                     )

    if args.number == 0:
        ## kinect
        base_dir = "/home/user/Desktop/FoundationPose-main/demo_data/kinect_driller_seq"
        rgb_dir = os.path.join(base_dir, "rgb")
        depth0_path = os.path.join(base_dir, "depth", "0000001.png")
        mask_path = os.path.join(base_dir, "masks", "0000001.png")
        mesh_path = os.path.join(base_dir, "mesh", "textured_mesh.obj")
        debug_dir = "/home/user/Desktop/FoundationPose-main/debug/kinect_driller_seq"
    
    elif args.number == 1:
        ## mustard
        base_dir = "/home/user/Desktop/FoundationPose-main/demo_data/mustard0"
        rgb_dir = os.path.join(base_dir, "rgb")
        depth0_path = os.path.join(base_dir, "depth", "1581120424100262102.png")
        mask_path = os.path.join(base_dir, "masks", "1581120424100262102.png")
        mesh_path = os.path.join(base_dir, "mesh", "textured_simple.obj")
        debug_dir = "/home/user/Desktop/FoundationPose-main/debug/mustard0"

    else:
        logging.error("Invalid number! Use --number 0 for kinect or --number 1 for mustard.")
        return

    mesh = trimesh.load(mesh_path)
    ob_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png'))])

    depth0 = cv2.imread(depth0_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if len(depth0.shape) == 3:
        depth0 = depth0[..., 0]
    depth0 /= 1000.0 if depth0.max() > 100 else 1.0
    print("\nmesh mask & RGBD frames loaded successfully!")
    print(f"RGB frames found: {len(rgb_files)}\n")

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    debug = False
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug=debug,
        debug_dir=debug_dir
    )
    print("\nFoundationPose initialized successfully!\n")

    # 第一帧 MiDaS 深度, 用于计算全局 scale
    rgb0 = cv2.imread(os.path.join(rgb_dir, rgb_files[0]))
    rgb0 = cv2.cvtColor(rgb0, cv2.COLOR_BGR2RGB)
    h, w = rgb0.shape[:2]
    resized = False
    if h > 1000 or w > 1000:
        rgb0 = cv2.resize(rgb0, (640, 480), interpolation=cv2.INTER_LINEAR)
        depth0 = cv2.resize(depth0, (640, 480), interpolation=cv2.INTER_NEAREST)
        resized = True

        xc = 640 / w
        yc = 480 / h
        compress_matrix = np.array([[xc, 0. , 0.],
                                    [0., yc , 0.],
                                    [0., 0. , 1.]], 
                                    dtype=np.float32)
        K = compress_matrix @ K_origdef
    else:
        K = K_orig
    global _midas_model, _midas_transform
    if _midas_model is not None:
        return
    model_path = "/home/user/.cache/midas/dpt_large_384.pt"
    _midas_model = DPTDepthModel(
        path=model_path,
        backbone="vitl16_384",
        non_negative=True,
    )
    _midas_model.eval()
    _midas_model.to(_device)
    _midas_transform = T.Compose([
        Resize(
            384,
            384,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
    ])
    logging.info("\nMiDaS model loaded successfully\n")
    
    depth_midas0 = estimate_depth(rgb0)
    
    valid = (depth0 > 0.001)
    if resized:
        ob_mask = cv2.resize(ob_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
    mask0 = ob_mask > 128
    valid = valid & mask0
    if valid.sum() == 0:
        scale = 1.0
        logging.warning("无法计算 scale，使用默认值 1.0")
    else:
        scale = np.median(depth0[valid]) / np.median(depth_midas0[valid])
    print(f"\nEstimated global scale factor: {scale:.4f}\n")

    ## main loop
    poses = []
    vis_list = []

    for i, rgb_name in tqdm(enumerate(rgb_files), total=len(rgb_files), desc="Processing frames"):
        rgb = cv2.imread(os.path.join(rgb_dir, rgb_name))
        rgb_copy = rgb.copy()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR)

        if i == 0:
            depth = depth0
            pose = est.register(
                K=K,
                rgb=rgb,
                depth=depth,
                ob_mask=mask0,
                iteration=5
            )

        else:
            depth = estimate_depth(rgb, scale_factor=scale)
            pose = est.track_one(
                rgb=rgb,
                depth=depth,
                K=K,
                iteration=2
            )

            z_pose = pose[2, 3]
            x_pose = pose[0, 3]
            y_pose = pose[1, 3]
            
            u = int((x_pose / z_pose) * K[0, 0] + K[0, 2])
            v = int((y_pose / z_pose) * K[1, 1] + K[1, 2])
            raw_midas_depth = depth / scale

            h_img, w_img = raw_midas_depth.shape
            window = 40
            u_min, u_max = max(0, u - window), min(w_img, u + window)
            v_min, v_max = max(0, v - window), min(h_img, v + window)
            
            if u_max > u_min and v_max > v_min:
                sampled_midas = raw_midas_depth[v_min:v_max, u_min:u_max]
                valid_midas = sampled_midas[sampled_midas > 0.001]
                
                if len(valid_midas) > 0:
                    midas_z = np.median(valid_midas)
                    new_scale = z_pose / (midas_z + 1e-8)
                    alpha = 0.2
                    scale = alpha * new_scale + (1.0 - alpha) * scale

        poses.append(pose.copy())

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        center_pose = pose @ np.linalg.inv(to_origin)

        vis = draw_posed_3d_box(K_orig, rgb_copy, center_pose, bbox)
        vis = draw_xyz_axis(vis, center_pose, scale=0.1, K=K_orig, thickness=3, is_input_rgb=False)
        vis_list.append(vis)

    ## generate mp4 output video
    vedio_path = "/home/user/Desktop/FoundationPose-main/vedio/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"

    if len(vis_list) > 0:
        h, w = vis_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vedio_path, fourcc, 30, (w, h))
        for frame in vis_list:
            # writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.write(frame)
        writer.release()

    print(f"\nOutput video saved to: {vedio_path}\n")


if __name__ == "__main__":
    main()