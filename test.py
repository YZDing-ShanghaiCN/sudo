#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import trimesh
import argparse
import sys
import open3d as o3d
import os

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

def MiDaS_depth(img):
    """
        input: a RGB iamage
        output: a depth map with the same resolution as input
    """
    global _midas_model, _midas_transform
    if _midas_model is None:
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
            NormalizeImage(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            PrepareForNet(),
        ])

    img_input = _midas_transform({"image": img})["image"]
    img_input = torch.from_numpy(img_input).unsqueeze(0).to(_device)

    with torch.no_grad():
        prediction = _midas_model(img_input)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth = prediction.cpu().numpy()

    depth -= depth.min()
    depth /= depth.max() + 1e-8

    return depth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mode", required=True, type=int)
    parser.add_argument("--xc", default=1.0, type=float, help="compress ratio along x axis for point cloud")
    parser.add_argument("--yc", default=1.0, type=float, help="compress ratio along y axis for point cloud")
    args = parser.parse_args()

    rgb = cv2.imread(args.image)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    rgb = cv2.resize(rgb, (int(w*args.xc), int(h*args.yc)))
    
    K_origin = np.array([[319.58200073, 0., 320.21498477], 
                   [0., 417.11868286, 244.34866809], 
                   [0., 0., 1.]
                 ], 
                 dtype=np.float32
                )
    compress_matrix = np.array([[args.xc, 0., 0.],
                                [0., args.yc, 0.],
                                [0., 0., 1.]], dtype=np.float32)
    K = compress_matrix @ K_origin


    if args.mode == 0:
        dep_path = os.path.join(os.path.dirname(os.path.dirname(args.image)), "depth", os.path.basename(args.image))
        mask_path = os.path.join(os.path.dirname(os.path.dirname(args.image)), "masks", "0000001.png")
        mesh_path = os.path.join(os.path.dirname(os.path.dirname(args.image)), "mesh", "textured_mesh.obj")

        depth = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        depth = cv2.resize(depth, (int(w * args.xc), int(h * args.yc)), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (int(w * args.xc), int(h * args.yc)), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 128).astype(np.uint8) 
        mesh = trimesh.load(mesh_path)

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()

        ## new
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

        print("Initializing FoundationPose...")
        estimator = FoundationPose(
            mesh=mesh,
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals, 
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,                      
            debug=0,
            debug_dir="./debug"
        )

        pose = estimator.register(
            K=K,
            rgb=rgb,
            depth=depth,
            ob_mask=mask.astype(bool),
            iteration=5
        )
        
        pose = estimator.track_one(
            rgb=rgb,
            depth=depth,
            K=K,
            iteration=2
        )
        
        print("Estimated Pose:\n", pose)
        
        if pose is not None:
            center_pose = pose@np.linalg.inv(to_origin)
            vis_img = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
            vis_img = draw_xyz_axis(
                vis_img, 
                ob_in_cam=center_pose,     
                scale=0.1,              
                K=K,
                thickness=3,
                transparency=0,              
                is_input_rgb=True         
            )
            
            vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

            save_path = f"test_result/kinect/{os.path.basename(args.image)}"
            print(f"Saving visualization to {save_path}...")
            cv2.imwrite(save_path, vis_img_bgr)

            cv2.imshow("FoundationPose Result", vis_img_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif args.mode == 1:
        dep_path = os.path.join(os.path.dirname(os.path.dirname(args.image)), "depth", os.path.basename(args.image))
        mask_path = os.path.join(os.path.dirname(os.path.dirname(args.image)), "masks", "1581120424100262102.png")
        mesh_path = os.path.join(os.path.dirname(os.path.dirname(args.image)), "mesh", "textured_simple.obj")

        depth = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        depth = cv2.resize(depth, (int(w * args.xc), int(h * args.yc)), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (int(w * args.xc), int(h * args.yc)), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 128).astype(np.uint8) 
        mesh = trimesh.load(mesh_path)

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()

                ## new
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

        print("Initializing FoundationPose...")
        estimator = FoundationPose(
            mesh=mesh,
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals, 
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,                      
            debug=0,
            debug_dir="./debug"
        )

        pose = estimator.register(
            K=K,
            rgb=rgb,
            depth=depth,
            ob_mask=mask.astype(bool),
            iteration=5
        )
        
        pose = estimator.track_one(
            rgb=rgb,
            depth=depth,
            K=K,
            iteration=2
        )
        
        print("Estimated Pose:\n", pose)
        
        if pose is not None:
            center_pose = pose@np.linalg.inv(to_origin)
            vis_img = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
            vis_img = draw_xyz_axis(
                vis_img, 
                ob_in_cam=center_pose,     
                scale=0.1,              
                K=K,
                thickness=3,
                transparency=0,              
                is_input_rgb=True         
            )
            
            vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

            save_path = f"test_result/mustard/{os.path.basename(args.image)}"
            print(f"Saving visualization to {save_path}...")
            cv2.imwrite(save_path, vis_img_bgr)

            cv2.imshow("FoundationPose Result", vis_img_bgr)
            cv2.waitKey(0)

    else:
        # get depth from MiDaS
        depth = MiDaS_depth(rgb)
        depth = cv2.resize(depth, (int(w*args.xc), int(h*args.yc)))
        print("Depth shape:", depth.shape)

        # generate point cloud
        xyz_map = depth2xyzmap(depth, K)
        valid = depth > 0.001
        pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])

        # point cloud to mesh
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        pcd.orient_normals_consistent_tangent_plane(100)
        poisson_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)
        poisson_mesh.remove_unreferenced_vertices()
        poisson_mesh.remove_degenerate_triangles()
        poisson_mesh.remove_duplicated_triangles()
        poisson_mesh.remove_duplicated_vertices()
        poisson_mesh.remove_non_manifold_edges()
        print("mesh vertices:", len(poisson_mesh.vertices))
    
        # foundation pose estimation
        est = FoundationPose(
            model_pts=np.asarray(poisson_mesh.vertices),
            model_normals=np.asarray(poisson_mesh.vertex_normals),
            mesh=trimesh.Trimesh(vertices=np.asarray(poisson_mesh.vertices), faces=np.asarray(poisson_mesh.triangles)),
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            debug=0,
            debug_dir="./debug"
        )

        print("Running FoundationPose register...")  
        pose = est.register(
            K=K,
            rgb=rgb,
            depth=depth,
            ob_mask=valid,
            iteration=5
        )

        to_origin, extents = trimesh.bounds.oriented_bounds(np.asarray(poisson_mesh.vertices))
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

        center_pose = pose @ np.linalg.inv(to_origin)

        vis = draw_posed_3d_box(K, rgb.copy(), center_pose, bbox)
        vis = draw_xyz_axis(vis, center_pose, 0.1, K)

        save_path = f"test_result/{os.path.basename(args.image)}"
        print(f"Saving visualization to {save_path}...")

        cv2.imwrite(save_path, vis)
        cv2.imshow("pose", vis)
        print("Press any key on the image window to exit...")
        cv2.waitKey(0)


if __name__ == "__main__":
    main()