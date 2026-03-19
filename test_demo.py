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
import torchvision.transforms as T
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
import nvdiffrast.torch as dr

def main():
    # parser = argparse.ArgumentParser(description="input directory name")
    # parser.add_argument("--number", type=int, required=True, help="0 for kinect driller, 1 for mustard")
    # args = parser.parse_args()

    set_logging_format = lambda: logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    set_logging_format()
    set_seed = lambda x: np.random.seed(x)
    set_seed(0)

    K_orig = np.load("./pre_result/intrinsics.npy").astype(np.float32)
    rgb = np.load("./pre_result/rgb.npy").astype(np.uint8)
    depth = np.load("./pre_result/depth.npy").astype(np.float32)
    ob_masks = np.load("./pre_result/masks.npy").astype(bool)
    bboxes = np.load("./pre_result/bboxes.npy").astype(np.float32)
    print("\nRGBD frames and masks loaded successfully!\n")
    debug_dir = "/home/user/Desktop/FoundationPose/debug/foundationstereo"
    save_dir = "/home/user/Desktop/FoundationPose/detection_result"

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    n = bboxes.shape[0]
    
    for i in range(n):
        ob_mask = ob_masks[i]
        bbox = bboxes[i]
        vis = rgb.copy()

        # 泊松重建得到物体 mesh
        pcd = o3d.geometry.PointCloud()
        xyz_map = depth2xyzmap(depth, K_orig)
        valid = (depth > 0.001) & ob_mask
        pcd.points = o3d.utility.Vector3dVector(xyz_map[valid])
        pcd.colors = o3d.utility.Vector3dVector(rgb[valid] / 255.0)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7, width=0, scale=1.1, linear_fit=False)[0]
        poisson_mesh.remove_degenerate_triangles()
        poisson_mesh.remove_duplicated_triangles()
        poisson_mesh.remove_duplicated_vertices()
        poisson_mesh.remove_non_manifold_edges()    
        
        vertices = np.asarray(poisson_mesh.vertices)
        center = vertices.mean(axis=0)
        vertices -= center
        mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(poisson_mesh.triangles), vertex_normals=np.asarray(poisson_mesh.vertex_normals), process=True)
        print(f"\nMesh reconstructed successfully! Offset from camera: {center}\n")

        # 加载 foundationpose 模型
        est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=False,
            debug_dir=debug_dir
        )
        print("\nFoundationPose initialized successfully!\n")
        
        # pose estimation
        pose = est.register(K=K_orig, rgb=vis, depth=depth, ob_mask=ob_mask, iteration=5)
        print(f"\nPose estimation completed! Estimated pose:\n{pose}\n")

        bbox_3d = np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
        vis = draw_posed_3d_box(K_orig, img=vis, ob_in_cam=pose, bbox=bbox_3d)
        vis = draw_xyz_axis(vis, ob_in_cam=pose, scale=0.1, K=K_orig, thickness=3, transparency=0, is_input_rgb=True)
        del est
        torch.cuda.empty_cache()

        cv2.imshow('Estimated Pose', vis[...,::-1])
        cv2.imwrite(os.path.join(save_dir, f"result_{i:02d}.png"), vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()