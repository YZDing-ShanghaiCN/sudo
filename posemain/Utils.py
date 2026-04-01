import os
import sys
import importlib
import importlib.util
import logging
import cv2
import torch
import numpy as np
import open3d as o3d

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)


def set_logging_format(level=logging.INFO):
  importlib.reload(logging)
  fmt = '%(message)s'
  logging.basicConfig(level=level, format=fmt, datefmt='%m-%d|%H:%M:%S')


set_logging_format()


def set_seed(random_seed):
  import random
  np.random.seed(random_seed)
  random.seed(random_seed)
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def toOpen3dCloud(points, colors=None, normals=None):
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
  if colors is not None:
    if colors.max() > 1:
      colors = colors / 255.0
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
  if normals is not None:
    cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
  return cloud


def depth2xyzmap(depth: np.ndarray, K, uvs: np.ndarray = None, zmin=0.1):
  invalid_mask = depth < zmin
  H, W = depth.shape[:2]
  if uvs is None:
    vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
  else:
    uvs = uvs.round().astype(int)
    us = uvs[:, 0]
    vs = uvs[:, 1]

  zs = depth[vs, us]
  xs = (us - K[0, 2]) * zs / K[0, 0]
  ys = (vs - K[1, 2]) * zs / K[1, 1]
  pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)

  xyz_map = np.zeros((H, W, 3), dtype=np.float32)
  xyz_map[vs, us] = pts
  if invalid_mask.any():
    xyz_map[invalid_mask] = 0
  return xyz_map


def freeze_model(model):
  model = model.eval()
  for p in model.parameters():
    p.requires_grad = False
  for p in model.buffers():
    p.requires_grad = False
  return model


def get_resize_keep_aspect_ratio(H, W, divider=16, max_H=1232, max_W=1232):
  assert max_H % divider == 0
  assert max_W % divider == 0

  def round_by_divider(x):
    return int(np.ceil(x / divider) * divider)

  H_resize = round_by_divider(H)
  W_resize = round_by_divider(W)
  if H_resize > max_H or W_resize > max_W:
    if H_resize > W_resize:
      W_resize = round_by_divider(W_resize * max_H / H_resize)
      H_resize = max_H
    else:
      H_resize = round_by_divider(H_resize * max_W / W_resize)
      W_resize = max_W
  return int(H_resize), int(W_resize)


def vis_disparity(disp, min_val=None, max_val=None, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO, cmap=None, other_output={}):
  disp = disp.copy()
  H, W = disp.shape[:2]
  invalid_mask = disp >= invalid_thres
  if (invalid_mask == 0).sum() == 0:
    other_output['min_val'] = None
    other_output['max_val'] = None
    return np.zeros((H, W, 3), dtype=np.uint8)

  if min_val is None:
    min_val = disp[invalid_mask == 0].min()
  if max_val is None:
    max_val = disp[invalid_mask == 0].max()

  other_output['min_val'] = min_val
  other_output['max_val'] = max_val

  vis = ((disp - min_val) / (max_val - min_val)).clip(0, 1) * 255
  if cmap is None:
    vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[..., ::-1]
  else:
    vis = cmap(vis.astype(np.uint8))[..., :3] * 255

  if invalid_mask.any():
    vis[invalid_mask] = 0
  return vis.astype(np.uint8)


def depth_uint8_decoding(depth_uint8, scale=1000):
  depth_uint8 = depth_uint8.astype(float)
  out = depth_uint8[..., 0] * 255 * 255 + depth_uint8[..., 1] * 255 + depth_uint8[..., 2]
  return out / float(scale)


# ----- fallback depth filters for environments without warp -----
def erode_depth(depth, radius=2, depth_diff_thres=0.001, ratio_thres=0.8, zfar=100, device='cuda'):
  is_numpy = isinstance(depth, np.ndarray)
  if is_numpy:
    depth_np = depth.astype(np.float32, copy=True)
  else:
    depth_np = torch.as_tensor(depth, dtype=torch.float32).detach().cpu().numpy()

  out = depth_np.copy()
  invalid = (depth_np < 0.001) | (depth_np >= zfar)
  out[invalid] = 0

  if is_numpy:
    return out
  return torch.as_tensor(out, dtype=depth.dtype, device=depth.device)


def bilateral_filter_depth(depth, radius=2, zfar=100, sigmaD=2, sigmaR=100000, device='cuda'):
  is_numpy = isinstance(depth, np.ndarray)
  if is_numpy:
    depth_np = depth.astype(np.float32, copy=True)
  else:
    depth_np = torch.as_tensor(depth, dtype=torch.float32).detach().cpu().numpy()

  ksize = int(radius) * 2 + 1
  out = cv2.bilateralFilter(depth_np, d=ksize, sigmaColor=float(sigmaR), sigmaSpace=float(sigmaD))
  invalid = (depth_np < 0.001) | (depth_np >= zfar)
  out[invalid] = 0

  if is_numpy:
    return out
  return torch.as_tensor(out, dtype=depth.dtype, device=depth.device)


# ----- pose utils dynamic bridge (no direct code copy) -----
_POSE_UTILS_MODULE = None
_POSE_UTILS_LOADED = False


def _load_pose_utils_module(pose_utils_path=None):
  global _POSE_UTILS_MODULE, _POSE_UTILS_LOADED
  if _POSE_UTILS_LOADED:
    return _POSE_UTILS_MODULE

  if pose_utils_path is None:
    pose_utils_path = os.environ.get('POSE_UTILS_PATH', '/home/user/Desktop/FoundationPose/Utils.py')

  if not os.path.exists(pose_utils_path):
    logging.warning(f'[Utils] pose utils path not found: {pose_utils_path}')
    _POSE_UTILS_LOADED = True
    _POSE_UTILS_MODULE = None
    return None

  spec = importlib.util.spec_from_file_location('pose_utils_external', pose_utils_path)
  if spec is None or spec.loader is None:
    logging.warning(f'[Utils] cannot import pose utils from: {pose_utils_path}')
    _POSE_UTILS_LOADED = True
    _POSE_UTILS_MODULE = None
    return None

  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  _POSE_UTILS_MODULE = module
  _POSE_UTILS_LOADED = True
  logging.info(f'[Utils] loaded pose utils from: {pose_utils_path}')
  return module


def register_pose_utils(pose_utils_path=None, suffix='_pose'):
  module = _load_pose_utils_module(pose_utils_path=pose_utils_path)
  if module is None:
    return

  g = globals()
  for name in dir(module):
    if name.startswith('__'):
      continue
    value = getattr(module, name)
    alias_name = f'{name}{suffix}'
    if alias_name not in g:
      g[alias_name] = value


def use_pose_implementations(overwrite=True, pose_utils_path=None):
  module = _load_pose_utils_module(pose_utils_path=pose_utils_path)
  if module is None:
    return

  g = globals()
  for name in dir(module):
    if name.startswith('__'):
      continue
    value = getattr(module, name)
    if overwrite or name not in g:
      g[name] = value


register_pose_utils()
