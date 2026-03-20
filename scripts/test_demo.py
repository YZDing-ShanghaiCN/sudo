import os,sys
import argparse
import imageio
import torch
import logging
import cv2
import numpy as np
import open3d as o3d
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
from core.foundation_stereo import FoundationStereo
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

def decode_disparity(encoded_disp):
    # 将 BGR 转换为 float
    b = encoded_disp[:, :, 0].astype(np.float32)
    g = encoded_disp[:, :, 1].astype(np.float32)
    r = encoded_disp[:, :, 2].astype(np.float32)
    
    decoded = r * 65536 + g * 256 + b
    decoded /= 256.0
    return decoded

def compute_disp_similarity(disp1, disp2):
    if disp1.shape != disp2.shape:
        raise ValueError("Disparity images must have the same shape")
    
    disp1 = disp1.astype(np.float32)
    disp2 = disp2.astype(np.float32)
    
    mse = np.mean((disp1 - disp2) ** 2)
    mae = np.mean(np.abs(disp1 - disp2))
    max_val = max(np.max(disp1), np.max(disp2))
    
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
    
    return mse, mae, psnr

yolo_model = YOLO("../checkpoints/yolo/yolov8m.pt")
# yolo_model.to("cuda")
yolo_model.to("cpu")

def get_bboxes(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = yolo_model(img_rgb, verbose=False)[0]

    if results.boxes is None:
        return np.zeros((0, 4))

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()

    keep = scores > 0.8
    return boxes[keep]

def generate_3d_point_cloud(img_bgr, depth, K, z_far, mask=None):
    # disp = disp.copy()
    # disp[disp < 1e-6] = np.inf
    # depth = K[0,0] * baseline / disp

    xyz_map = depth2xyzmap(depth, K)
    points = xyz_map.reshape(-1, 3)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    colors = img_rgb.reshape(-1, 3)

    if mask is not None:
        mask_flat = mask.reshape(-1).astype(bool)
        points = points[mask_flat]
        colors = colors[mask_flat]

    pcd = toOpen3dCloud(points, colors)
    pts = np.asarray(pcd.points)
    keep_mask = (pts[:,2] > 0) & (pts[:,2] <= z_far)
    keep_ids = np.where(keep_mask)[0]
    pcd = pcd.select_by_index(keep_ids)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Point Cloud")
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 1.0
    vis.run()
    vis.destroy_window()

    return pcd

if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str)
    parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str)
    parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
    parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=256, help='number of flow-field updates during forward pass')
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
    parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_dir = "../checkpoints/foundationstereo/23-51-11"
    intrinsic_file = "./assets/K.txt"

    cfg = OmegaConf.load(ckpt_dir + "/cfg.yaml")
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)

    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir + "/model_best_bp2.pth", weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()

    img0 = cv2.imread(args.left_file)
    img1 = cv2.imread(args.right_file)
    scale = args.scale
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H, W = img0.shape[:2]

    bboxes = get_bboxes(img0)
    print(f"[INFO] detected {len(bboxes)} objects")
    img0_vis = img0.copy()
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img0_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(f"{args.out_dir}/detected_bboxes.png", img0_vis)
    cv2.imshow("Detected Bounding Boxes", img0_vis)
    cv2.waitKey(0)

    img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0_t.shape, divis_by=32)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    with torch.no_grad():
        with torch.cuda.amp.autocast(True):
            disp = model.forward(img0_t, img1_t, iters=args.valid_iters, test_mode=True)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)

    if True:
        disp_path = "./assets/testdata/disparity.png"
        disp_data = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        disp_data = decode_disparity(disp_data)
        disp_data = cv2.resize(disp_data, fx=scale, fy=scale, dsize=None)
        print(disp.shape, disp_data.shape)
        print("GT max:", disp_data.max(), "GT min:", disp_data.min())
        print("GT dtype:", disp_data.dtype)
        # 同时显示两张图像
        disp_vis = vis_disparity(disp, args.z_far)
        disp_data_vis = vis_disparity(disp_data, args.z_far)
        mse, mae, psnr = compute_disp_similarity(disp, disp_data)
        print(f"Disparity Similarity - MSE: {mse:.4f}, MAE: {mae:.4f}, PSNR: {psnr:.2f} dB")

        combined_vis = np.hstack((disp_vis, disp_data_vis))
        cv2.imshow("Predicted Disparity (Left) vs Ground Truth Disparity (Right)", combined_vis)
        cv2.waitKey(0)
        sys.exit(0)

    with open(intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].split()))).reshape(3,3).astype(np.float32)
        baseline = float(lines[1])
    K[:2] *= scale

    # sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam = sam_model_registry["vit_b"](checkpoint="../checkpoints/sam/sam_vit_b_01ec64.pth")
    sam.cuda()
    predictor = SamPredictor(sam)
    predictor.set_image(img0)

    img_bgr = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    mask_list = []

    valid = disp > 1e-6
    depth = np.zeros_like(disp)
    depth[valid] = K[0,0] * baseline / disp[valid]

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        
        input_box = np.array([x1, y1, x2, y2])
        masks, scores, _ = predictor.predict(
            box=input_box,
            multimask_output=False
        )
        mask = masks[0]

        # pcd_res = generate_3d_point_cloud(img_bgr, disp, K, baseline, args.z_far, mask)
        pcd_res = generate_3d_point_cloud(img0, depth, K, args.z_far, mask)
        mask_list.append(mask.astype(np.uint8))

    cv2.destroyAllWindows()

    np.save("../FoundationPose/pre_result/depth.npy", depth)
    np.save("../FoundationPose/pre_result/masks.npy", np.array(mask_list))
    np.save("../FoundationPose/pre_result/bboxes.npy", bboxes)
    np.save("../FoundationPose/pre_result/intrinsics.npy", K)
    np.save("../FoundationPose/pre_result/rgb.npy", img0)
    print("[INFO] saved depth, masks, bboxes, intrinsics, and rgb to ../FoundationPose/pre_result/ for pose estimation downstream")
    print(img0.shape, depth.shape, K.shape, bboxes.shape, np.array(mask_list).shape)
    cv2.imwrite(f"{args.out_dir}/depth.png", vis_disparity(depth, args.z_far))