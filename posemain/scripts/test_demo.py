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
from ultralytics.nn.tasks import DetectionModel
from segment_anything import sam_model_registry, SamPredictor
import json
from typing import Tuple, Union, Optional

torch.serialization.add_safe_globals([DetectionModel])

yolo_model = YOLO("../checkpoints/yolo/yolov8l.pt")
yolo_model.to("cpu")  # yolo_model.to("cuda")

def rectify_stereo_images(
    img_left: Union[str, np.ndarray], 
    img_right: Union[str, np.ndarray],
    K1: np.ndarray, D1: Optional[np.ndarray], 
    K2: np.ndarray, D2: Optional[np.ndarray],
    R: np.ndarray, T: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Rectify stereo images given their intrinsics and extrinsics.

    """
    if isinstance(img_left, str):
        img_left_arr = cv2.imread(img_left)
    else:
        img_left_arr = img_left
    if isinstance(img_right, str):
        img_right_arr = cv2.imread(img_right)
    else:
        img_right_arr = img_right
    height, width = img_left_arr.shape[:2]

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2,
        (width, height), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=1
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)

    rect_left  = cv2.remap(img_left_arr,  map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right_arr, map2x, map2y, cv2.INTER_LINEAR)

    return rect_left, rect_right, R1, R2, P1, P2

def get_bboxes(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = yolo_model(img_rgb, verbose=False)[0]

    if results.boxes is None:
        return np.zeros((0, 4))

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()

    keep = scores > 0.7
    return boxes[keep]

def generate_3d_point_cloud(img_bgr, depth, K, z_far, mask=None):
    '''
        Generate a 3D point cloud from a depth map and corresponding RGB image, optionally applying a mask to filter points.
    '''
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
    code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
    parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    #################################################
    #         0. edit input and output path         #
    #################################################
    in_dir = f"{code_dir}/assets/hand_camera_2"
    json_file = f"{in_dir}/hand_camera_data.json"
    left_file = f"{in_dir}/left_hand_left_camera.png"
    right_file = f"{in_dir}/left_hand_right_camera.png"

    out_dir = f"{code_dir}/test_outputs/test2"
    os.makedirs(out_dir, exist_ok=True)

    ckpt_dir = "../checkpoints/foundationstereo/23-51-11"
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

    #################################################
    #      1. load images and camera parameters     #
    #################################################
    img0 = cv2.imread(left_file)
    img1 = cv2.imread(right_file)
    scale = args.scale
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H, W = img0.shape[:2]

    with open(json_file, 'r') as f:
        json_data = json.load(f)
    K = np.array(json_data['camera_data']['left_hand_left_camera']['intrinsics'], dtype=np.float64)
    K_right = np.array(json_data['camera_data']['left_hand_right_camera']['intrinsics'], dtype=np.float64)    
    Tc2w_left = np.array(json_data['camera_data']['left_hand_left_camera']['extrinsics'], dtype=np.float64)
    Tc2w_right = np.array(json_data['camera_data']['left_hand_right_camera']['extrinsics'], dtype=np.float64)
    arm_pose = np.array(json_data['arm_pose'], dtype=np.float64)
    K[:2] *= scale
    K_right[:2] *= scale
    matrix = np.linalg.inv(Tc2w_right) @ Tc2w_left

    #################################################
    #           2. rectify stereo images            #
    #################################################
    img0, img1, R1, R2, P1, P2 = rectify_stereo_images(
        img0, img1,
        K, None,
        K_right, None,
        R=matrix[:3, :3],
        T=matrix[:3, 3]
    )

    img_concat = np.concatenate((img0, img1), axis=1)
    cv2.imshow("Rectified Stereo Pair", img_concat)
    # cv2.imwrite(f"{out_dir}/rectified_pair.png", img_concat)
    # cv2.imwrite(f"{out_dir}/rectified_left.png", img0)
    # cv2.imwrite(f"{out_dir}/rectified_right.png", img1)
    cv2.waitKey(0)

    #################################################
    # 3. detect objects and generate masks with SAM #
    #################################################
    bboxes = get_bboxes(img0)
    print(f"[INFO] detected {len(bboxes)} objects")
    img0_vis = img0.copy()
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img0_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imwrite(f"{out_dir}/detected_bboxes.png", img0_vis)
    cv2.imshow("Detected Bounding Boxes", img0_vis)
    cv2.waitKey(0)

    sam = sam_model_registry["vit_b"](checkpoint="../checkpoints/sam/sam_vit_b_01ec64.pth")
    sam.cuda()
    predictor = SamPredictor(sam)
    predictor.set_image(img0)

    ################################################
    #           4. run foundation stereo           #
    ################################################
    img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0_t.shape, divis_by=32)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            disp = model.forward(img0_t, img1_t, iters=args.valid_iters, test_mode=True)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)

    baseline = abs(P2[0, 3] / P2[0, 0]) 
    print(f"[DEBUG] Computed baseline: {baseline:.4f} (If this is > 1.0, your json extrinsics are likely in millimeters, not meters!)")
    K = P1[:3, :3]
    img_bgr = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    mask_list = []
    valid = disp > 1e-6
    depth = np.zeros_like(disp)
    depth[valid] = K[0,0] * baseline / disp[valid]

    cv2.imwrite(f"{out_dir}/disparity0.png", vis_disparity(disp, args.z_far))
    cv2.imwrite(f"{out_dir}/depth0.png", vis_disparity(depth, max_val=args.z_far))

    #################################################
    #  4.5 new update: calculate distance in depth  #
    #################################################
    '''
        function:
            - show depth image window
            - use mouse to click 2 point in the depth image
            - calculate 3D distance in real world by coordinates transformation
            - shou calculated distance in the window image
            - right click to eliminate the last clicked point
            - press q to quit
    '''
    

    save_index = 0
    last_canvas = {"img": None}

    depth_vis = vis_disparity(depth, max_val=args.z_far)
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
                cv2.putText(
                    canvas,
                    f"{dist:.4f} m",
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    canvas,
                    "Invalid depth for selected point",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        # cv2.putText(
        #     canvas,
        #     "Left click: select point | Right click: undo | s: save | q: quit",
        #     (20, H - 20),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.6,
        #     (255, 255, 255),
        #     2,
        #     cv2.LINE_AA,
        # )
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

    cv2.namedWindow(depth_window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(depth_window_name, on_mouse)
    redraw_depth_window()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            if last_canvas["img"] is not None:
                save_path = f"{out_dir}/distance/detectsave{save_index}.png"
                # cv2.imwrite(save_path, last_canvas["img"])
                print(f"[INFO] saved {save_path}")
                save_index += 1

    cv2.destroyWindow(depth_window_name)
    sys.exit(0)

    #################################################
    #    5. generate point cloud for each object    #
    #################################################
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
        # cv2.imwrite(f"{out_dir}/mask_{i}.png", mask.astype(np.uint8)*255)

        pcd_res = generate_3d_point_cloud(img0, depth, K, args.z_far, mask)
        mask_list.append(mask.astype(np.uint8))

        # cv2.imwrite(f"{out_dir}/mask_{i}.png", mask.astype(np.uint8)*255)
        cv2.imshow(f"Mask {i}", mask.astype(np.uint8)*255)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    # np.save("../FoundationPose/pre_result/depth.npy", depth)
    # np.save("../FoundationPose/pre_result/masks.npy", np.array(mask_list))
    # np.save("../FoundationPose/pre_result/bboxes.npy", bboxes)
    # np.save("../FoundationPose/pre_result/intrinsics.npy", K)
    # np.save("../FoundationPose/pre_result/rgb.npy", img0)
    print("[INFO] saved depth, masks, bboxes, intrinsics, and rgb to ../FoundationPose/pre_result/ for pose estimation downstream")
    print(img0.shape, depth.shape, K.shape, bboxes.shape, np.array(mask_list).shape)