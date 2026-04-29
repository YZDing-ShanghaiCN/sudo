import torch
from typing import Literal, Union, List, Tuple
import numpy as np
import cv2
from PIL import Image
import calibur
import os
import OpenEXR
import Imath
import yaml


def read_exr(file_path):
    """
    读取EXR文件
    
    Args:
        file_path (str): EXR文件路径
        
    Returns:
        numpy.ndarray: 图像数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EXR file not found: {file_path}")
    
    # 打开EXR文件
    exr_file = OpenEXR.InputFile(file_path)
    
    # 获取头部信息
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    # 读取数据
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = header['channels'].keys()
    
    # 读取第一个通道（通常是R通道）
    channel = list(channels)[0]
    data = exr_file.channel(channel, FLOAT)
    
    # 转换为numpy数组
    data_array = np.frombuffer(data, dtype=np.float32)
    data_array = data_array.reshape(size[1], size[0])  # 注意：EXR是行优先，需要转置
    
    # 翻转Y轴以匹配原始坐标系
    #data_array = np.flipud(data_array)
    
    exr_file.close()
    return data_array

def save_exr(save_path, depth_map):
    height, width = depth_map.shape
    depth_data = depth_map.astype(np.float32)  # 翻转Y轴
    
    # 创建EXR头部信息
    header = OpenEXR.Header(width, height)
    header['channels'] = {'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    
    # 写入EXR文件
    exr_file = OpenEXR.OutputFile(save_path, header)
    exr_file.writePixels({'Z': depth_data.tobytes()})
    exr_file.close()

def compute_plucker_ray(c2w, K, h, w, ray_scale):
    """
    Computes Plucker rays from camera parameters and camera-to-world transformation.
    c2w: Camera-to-world transformation matrix (4x4) np.ndarray.
    K: Camera intrinsic matrix (3x3) np.ndarray.
    h: Height of the image.
    w: Width of the image.
    ray_scale: Scale factor for the rays. float.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    r_o, r_d = calibur.get_cam_rays_cv(c2w, fx, fy, cx, cy, h, w)
    ray = np.concatenate([np.cross(r_o, r_d), r_d, r_o], axis=-1).reshape(h, w, 9) * ray_scale
    ray = ray.reshape(h, w, 9)
    return ray

def center_crop_and_update_intrinsics(image: Union[Image.Image, np.ndarray], fxfycxcy: np.ndarray, crop_size: int | tuple[int]):
    # Get the original image dimensions
    if isinstance(image, Image.Image):
        width, height = image.size
    else:
        height, width = image.shape[:2]

    # Unpack the crop size
    if isinstance(crop_size, int):
        crop_width = crop_size
        crop_height = crop_size
    else:
        crop_width, crop_height = crop_size

    # Calculate the left, top, right, and bottom coordinates for cropping
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    # Perform center cropping
    if isinstance(image, Image.Image):
        cropped_image = image.crop((left, top, right, bottom))
    else:
        cropped_image = image[top:bottom, left:right]

    # Update the intrinsic values
    fx, fy, cx, cy = fxfycxcy
    new_cx = cx - left
    new_cy = cy - top

    # Create the updated intrinsic values array
    updated_fxfycxcy = np.array([fx, fy, new_cx, new_cy])

    return cropped_image, updated_fxfycxcy

def rectangular_crop_image(images,depths=None,Ks=None,crop_size=None):
    V, H, W, C = images.shape
    new_Ks = []
    new_images = []
    new_depths = []
    for v in range(V):
        image = images[v]
        if depths is not None:
            depth = depths[v]
        K = Ks[v]
        fxfycxcy = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
        if H > W * 0.625:
            crop_W = W
            crop_H = int(W * 0.625)
        else:
            crop_W = int(H * 1.6)
            crop_H = H
        image, fxfycxcy = center_crop_and_update_intrinsics(
            image, fxfycxcy, crop_size=(crop_W, crop_H)
        )
        if depths is not None:
            depth, _= center_crop_and_update_intrinsics(
                depth, fxfycxcy, crop_size=(crop_W, crop_H)
            )
        h, w, _ = image.shape
        if w != crop_size:
            image = cv2.resize(image, (crop_size, int(crop_size * 0.625)), interpolation=cv2.INTER_LANCZOS4)
            if depths is not None:
                depth = cv2.resize(depth, (crop_size, int(crop_size * 0.625)), interpolation=cv2.INTER_NEAREST)
            fxfycxcy *= crop_size / w
        newK = np.array([fxfycxcy[0], 0, fxfycxcy[2],
                         0, fxfycxcy[1], fxfycxcy[3],
                         0, 0, 1], dtype=np.float32).reshape(3, 3)
        new_Ks.append(newK)
        new_images.append(image)
        if depths is not None:
            new_depths.append(depth)
    new_images = np.stack(new_images, axis=0)
    if depths is not None:
        new_depths = np.stack(new_depths, axis=0)
    new_Ks = np.stack(new_Ks, axis=0)
    return new_images, new_Ks, new_depths

def preprocess(images,depths=None,Ks=None, c2ws=None,crop_size=None):
    # images[0] left image, images[1] right image
    images, Ks, depths = rectangular_crop_image(images,depths,Ks,crop_size)
    V, H, W, C = images.shape
    print(Ks)
    #t0_c2ws = recenter_poses(c2ws)
    obs_rays = []
    c2ws = np.linalg.inv(c2ws[0])[None, ...] @ c2ws
    for v in range(V):
        c2w = c2ws[v]
        K = Ks[v]
        obs_rays.append(compute_plucker_ray(c2w, K, H, W, 0.5))
    obs_rays = np.stack(obs_rays, axis=0)
    images = images[np.newaxis, ...]
    if len(depths) > 0:
        depths = depths[np.newaxis, ...]
    else:
        depths = np.zeros((1, V, H, W), dtype=np.float32)
    obs_rays = obs_rays[np.newaxis, ...]
    Ks = Ks[np.newaxis, ...]
    group_ids = np.array([0, 0]) 
    group_ids = np.repeat(group_ids[None], 1, axis=0)
    
    return (
        torch.from_numpy(np.array(images, dtype=np.float32)/255.0).cuda().bfloat16()[None],
        torch.from_numpy(np.array(depths, dtype=np.float32)).cuda().bfloat16()[None],
        torch.from_numpy(np.zeros((1, 2,H, W), dtype=np.float32)).cuda().bfloat16()[None],
        torch.from_numpy(np.zeros((1, 2), dtype=np.float32)).cuda().bfloat16()[None],
        torch.tensor([False]),
        torch.tensor([True]),
        torch.from_numpy(np.array(obs_rays, dtype=np.float32)).cuda().bfloat16()[None],
        torch.tensor([True]),
        torch.from_numpy(np.zeros((1, 2,H, W), dtype=np.float32)).cuda().bfloat16()[None],
        torch.tensor([False]),
        torch.from_numpy(np.zeros((1, 2, 4), dtype=np.float32)).cuda().bfloat16()[None],
        torch.tensor([False]),
        torch.from_numpy(np.zeros((1, 2), dtype=np.float32)).cuda().bfloat16()[None],
        torch.from_numpy(np.zeros((1, 2,H, W), dtype=np.float32)).cuda().bfloat16()[None],
        torch.from_numpy(np.zeros((1, 2), dtype=np.uint8)).cuda().bool()[None],
        torch.from_numpy(np.zeros((1, 2,H, W), dtype=np.float32)).cuda().bfloat16()[None],
        torch.from_numpy(np.zeros((1, 2,H, W, 3), dtype=np.float32)).cuda().bfloat16()[None],
        torch.from_numpy(np.array(Ks, dtype=np.float32)).cuda().bfloat16()[None],
        torch.from_numpy(np.array(c2ws, dtype=np.float32)).cuda().bfloat16()[None],
        torch.from_numpy(np.zeros((1, 2, 4), dtype=np.float32)).cuda().bfloat16()[None],
        torch.from_numpy(np.zeros((1, 2, 3), dtype=np.float32)).cuda().bfloat16()[None],
        torch.from_numpy(np.zeros((1, 2,H, W, 3), dtype=np.float32)).cuda().bfloat16()[None],
        torch.tensor([False]),
        torch.from_numpy(np.array(np.zeros((1, 2,H, W), dtype=np.float32))).cuda().bfloat16()[None],
        torch.tensor([False]),
        torch.from_numpy(np.array(np.zeros((1, 2,H, W), dtype=np.float32))).cuda().bfloat16()[None],
        torch.from_numpy(np.array(np.zeros((1, 2,3, 3), dtype=np.float32))).cuda().bfloat16()[None],
        torch.from_numpy(np.array(np.zeros((1, 2,4, 4), dtype=np.float32))).cuda().bfloat16()[None],
        torch.from_numpy(np.array(np.zeros((1, 2,H, W, 9), dtype=np.float32))).cuda().bfloat16()[None], 
        torch.from_numpy(np.array(np.zeros((1, 2, 4), dtype=np.float32))).cuda().bfloat16()[None],
        torch.from_numpy(np.array(np.zeros((1, 2, 3), dtype=np.float32))).cuda().bfloat16()[None],
        torch.from_numpy(np.array(np.zeros((1, 2,H, W, 3), dtype=np.float32))).cuda().bfloat16()[None],
        torch.from_numpy(np.array(np.zeros((1, 2,H, W, 3), dtype=np.float32))).cuda().bfloat16()[None],   
        torch.tensor([1.0]).cuda().bfloat16(),                            
        torch.from_numpy(np.array(group_ids, dtype=np.int32)).cuda()[None],
        torch.from_numpy(np.array(np.zeros((1, 4, 4), dtype=np.float32))).cuda().bfloat16()[None],
    )

def postprocess(output):
    metric_depth = output.float()
    metric_depth = torch.exp(np.log(10.0 / 0.01) * metric_depth) * 0.01
    metric_depth = metric_depth[0, 0, :, :,:]
    return metric_depth

model = torch.jit.load("/home/user/Desktop/checkpoints/joint-scene-l-640-ps16-fdv210ufdv3base-3G-2view-depth-err-rec-1e-1-noscale-8n-200k-dpt-ft.ts")
# root_path = "/main-cpfs/yachi/data/middleburry/2014"
root_path = "./wait_pose"
# index_list = os.path.join(root_path, "diffx_list.txt")
# output_path = "/main-cpfs/yachi/test/middleburry/2014_dpt_ft_ts_bf16"
output_path = os.path.join(root_path, "output")
os.makedirs(output_path, exist_ok=True)
# with open(index_list, "r") as f:
#     index_list = f.readlines()
#     index_list = [line.strip() for line in index_list]
for i in range(20):
    # left_image = Image.open(os.path.join(root_path, index, "im0.png"))
    # right_image = Image.open(os.path.join(root_path, index, "im1.png"))
    # left_depth = read_exr(os.path.join(root_path, index, "depth0.exr"))
    # right_depth = read_exr(os.path.join(root_path, index, "depth1.exr"))
    # calib_file = os.path.join(root_path, index, "calib.txt")
    left_image = Image.open(os.path.join(root_path, "rgb", "left_hand_left_camera", f"{i:06d}.jpg"))
    right_image = Image.open(os.path.join(root_path, "rgb", "left_hand_right_camera", f"{i:06d}.jpg"))
    # left_depth = read_exr(os.path.join(root_path, "depth0.exr"))
    # right_depth = read_exr(os.path.join(root_path, "depth1.exr"))
    left_depth = np.zeros((left_image.size[1], left_image.size[0]), dtype=np.float32)
    right_depth = np.zeros((right_image.size[1], right_image.size[0]), dtype=np.float32)
    left_calib_file = os.path.join(root_path, "intrinsics", "left_hand_left_camera_intrinsics.txt")
    K1 = np.loadtxt(left_calib_file, dtype=np.float32)
    right_calib_file = os.path.join(root_path, "intrinsics", "left_hand_right_camera_intrinsics.txt")
    K2 = np.loadtxt(right_calib_file, dtype=np.float32)
    # calib_file = os.path.join(root_path, "calib.txt")
    # with open(calib_file, "r") as f:
    #     lines = f.readlines()
    #     K1 = lines[0].split("=")[1].replace('[', '').replace(']', '').replace(';', '').split()
    #     K1 = [float(x) for x in K1]
    #     K1 = np.array(K1).reshape(3, 3)
    #     K2 = lines[1].split("=")[1].replace('[', '').replace(']', '').replace(';', '').split()
    #     K2 = [float(x) for x in K2]
    #     K2 = np.array(K2).reshape(3, 3)
        # width = int(lines[4].split("=")[1])
        # height = int(lines[5].split("=")[1])
        # doffs = float(lines[2].split("=")[1])
        # baseline = float(lines[3].split("=")[1])
    left_extrinsics_file = "./aililight_cameras/left_hand_left_camera_20260423.yaml"
    right_extrinsics_file = "./aililight_cameras/left_hand_right_camera_20260423.yaml"
    with open(left_extrinsics_file, "r") as f:
        left_p = np.array(yaml.safe_load(f)["extrinsic_pose"]["p"])
    with open(right_extrinsics_file, "r") as f:
        right_p = np.array(yaml.safe_load(f)["extrinsic_pose"]["p"])
    baseline = np.linalg.norm(left_p - right_p) * 1000.0
    view_image_0 = {}
    view_image_1 = {}
    extrinsics_0 = np.eye(4)
    extrinsics_1 = np.eye(4)
    extrinsics_1[0, 3] = -1*baseline/1000
    Ks = np.array([K1, K2])
    c2ws = np.array([np.linalg.inv(extrinsics_0), np.linalg.inv(extrinsics_1)])
    images = np.array([left_image, right_image])
    depths = np.array([left_depth, right_depth])
    images = np.stack(images, axis=0)
    depths = np.stack(depths, axis=0)
    Ks = np.stack(Ks, axis=0)
    c2ws = np.stack(c2ws, axis=0)
    crop_size = 640
    inputs = preprocess(images,depths,Ks,c2ws,crop_size)
    outputs = model(*inputs)
    metric_depth = postprocess(outputs[0])
    metric_depth = metric_depth.float().cpu().numpy()
    os.makedirs(os.path.join(output_path), exist_ok=True)
    save_exr(os.path.join(output_path, f"{i:06d}_depth_pred0.exr"), metric_depth[0])
    save_exr(os.path.join(output_path, f"{i:06d}_depth_pred1.exr"), metric_depth[1])
    gt_depth = inputs[1][0, 0, :, :, :]
    gt_depth = gt_depth.float().cpu().numpy()
    save_exr(os.path.join(output_path, f"{i:06d}_depth0.exr"), gt_depth[0])
    save_exr(os.path.join(output_path, f"{i:06d}_depth1.exr"), gt_depth[1])
    images = inputs[0][0, 0, :, :, :]
    images = images.float().cpu().numpy()
    images = images * 255.0
    images = images.astype(np.uint8)
    left_image = Image.fromarray(images[0])
    left_image.save(os.path.join(output_path, f"{i:06d}_im0.png"))
    right_image = Image.fromarray(images[1])
    right_image.save(os.path.join(output_path, f"{i:06d}_im1.png"))
    print(np.median(metric_depth[0]))
    print(np.median(metric_depth[1]))
    print(np.median(gt_depth[0]))
    print(np.median(gt_depth[1]))
    