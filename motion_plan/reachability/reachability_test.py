import numpy as np
import yaml
import ampl
import pyampl
import os
import argparse
from typing import Any, Dict, List, Tuple, Union

def load_config(path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file and recursively resolves mathematical expressions 
    containing 'pi'.

    Args:
        path (str): The absolute or relative path to the .yaml config file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters with resolved values.
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    def resolve_pi(val: Any) -> Any:
        """Recursively replaces 'pi' strings with np.pi and evaluates the expression."""
        if isinstance(val, str):
            if 'pi' in val:
                # Replace pi with np.pi and evaluate.
                safe_dict = {"np": np, "pi": np.pi}
                try:
                    return eval(val.replace('pi', 'np.pi'), {"__builtins__": None}, safe_dict)
                except Exception:
                    return val
        if isinstance(val, list):
            return [resolve_pi(v) for v in val]
        if isinstance(val, dict):
            return {k: resolve_pi(v) for k, v in val.items()}
        return val

    return resolve_pi(cfg)



def genenerate_tcp_pose(
    x_range: List[float], 
    y_range: List[float], 
    z_range: List[float], 
    rx_rot: List[float],
    ry_rot: List[float], 
    rz_rot: List[float]
) -> Tuple[List[List[float]], List[np.ndarray], Tuple[int, int, int, int, int, int]]:
    """
    Generates a list of candidate TCP poses based on a 3D grid and set of rotations.

    Returns:
        Tuple[keys, poses, (nx, ny, nz, nrx, nry, nrz)]
    """
    def get_lin(r):
        num = int((r[1]-r[0])/r[2])
        if num <= 0: return np.array([r[0]])
        if num == 1: return np.array([r[0]])
        return np.linspace(r[0], r[1], num)

    tx = get_lin(x_range)
    ty = get_lin(y_range)
    tz = get_lin(z_range)
    
    keys=[]
    poses=[]
    for x in tx:
        for y in ty:
            for z in tz:
                for rx in rx_rot:
                    for ry in ry_rot:
                        for rz in rz_rot:
                            # Rotation order: Rz(rz) @ Ry(ry) @ Rx(rx)
                            # This keeps consistency with the previous script's Rz @ Ry logic.
                            R_in_base = ampl.so3_upexp(np.array([0,0,rz])) @ \
                                       ampl.so3_upexp(np.array([0,ry,0])) @ \
                                       ampl.so3_upexp(np.array([rx,0,0]))
                            
                            big_key = [x,y,z,rx,ry,rz]
                            tf = np.eye(4,dtype=np.float64)
                            tf[:3,:3] = R_in_base
                            tf[:3,3] = [x,y,z]
                            keys.append(big_key)
                            poses.append(tf)

    return keys, poses, (len(tx), len(ty), len(tz), len(rx_rot), len(ry_rot), len(rz_rot))

def main_v2(config_path: str, DIR_ASSETS: str = "./assets") -> None:    
    """
    Runs the reachability test by checking collision-free IK solutions for a grid of poses.

    This function loads the configuration, initializes the robot agent and collision scene,
    generates a set of target TCP poses relative to a tote, and iterates through them
    to find valid robot configurations.

    Args:
        config_path (str): Path to the .yaml configuration file.
        DIR_ASSETS (str): Path to the directory containing mesh files. Defaults to "./assets".
    """
    # Load config from the provided path
    cfg = load_config(config_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    env_cfg = cfg['environment']
    robot_cfg = env_cfg['robot']
    gripper_cfg = env_cfg['gripper']
    tote_cfg = env_cfg['tote']
    coll_cfg = cfg['collision_check']
    ik_cfg = cfg['ik']

    ############################################################################
    arm_config = pyampl.create_default_arm_config(robot_cfg['arm_name'])
    agent = pyampl.AgentArm(arm_config.name, arm_config.dim, arm_config)
    agent.fk_rwt = agent.state_ref
    agent.wall = np.array(robot_cfg['wall']).astype(pyampl.DTypeFloat)
    state = np.array(agent.state_ref.tolist(), dtype=np.float64)
    
    ############################################################################
    # Helper to resolve paths relative to workspace root if needed
    def get_path(p: str) -> str:
        """Resolves a file path relative to the workspace root or fallback assets dir."""
        if os.path.isabs(p): return p
        workspace_root = os.path.abspath(os.path.join(script_dir, "../../"))
        root_path = os.path.join(workspace_root, p)
        if os.path.exists(root_path): return root_path
        # Fallback to DIR_ASSETS if path starts with motion_plan/assets/
        if p.startswith("motion_plan/assets/"):
            return os.path.join(DIR_ASSETS, p.replace("motion_plan/assets/", ""))
        return p

    cvh_gripper = pyampl.CollisionObjectConvex(ampl.read_trimesh(get_path(gripper_cfg['mesh'])))
    mesh_crate = ampl.read_trimesh(get_path(tote_cfg['mesh']))
    
    collsion_scene = pyampl.CollisionScene()
    collsion_scene.insert_convex("crate", mesh_crate, create_pcd=coll_cfg['create_pcd'], pcd_dx=coll_cfg['pcd_dx'])
    
    ############################################################################
    def update_collision_scene(collsion_scene: pyampl.CollisionScene, dict_pose: Dict[str, np.ndarray]) -> None:        
        """Updates the poses of objects in the collision scene and refreshes the distance field."""
        for name,rwt in dict_pose.items():          
            collsion_scene.update_pose(name, rwt)
            collsion_scene.update_poses_pcd_from_convex()
            collsion_scene.enable_collision(name)
        pcd_tmp = collsion_scene.get_pointcloud(list(dict_pose.keys()))        
        collsion_scene.update_df_from_pcd(pcd_tmp)    
    
    ############################################################################
    R_base_tcp_standard = np.array(gripper_cfg['standard_tcp_rotation'], dtype=np.float64)
    t_base_tcp = np.array(gripper_cfg['standard_tcp_transition'])

    tf_tcp_standard = np.eye(4, dtype=np.float64)
    tf_tcp_standard[:3,:3] = R_base_tcp_standard
    tf_tcp_standard[:3,3] = t_base_tcp.flatten()
    
    tf_crate_standard = np.array(tote_cfg['base_position'], dtype=np.float32)

    def range_to_list(r: Any) -> Union[List[float], Any]:
        """Converts a [start, end, step] list into a list of values using linspace."""
        if isinstance(r, list) and len(r) == 3:
            num = int((r[1]-r[0])/r[2])
            if num <= 0: return [r[0]]
            if num == 1: return [r[0]]
            return np.linspace(r[0], r[1], num).tolist()
        return r

    # Generate tote poses based on self_range if provided
    if 'self_range' in tote_cfg:
        self_range = tote_cfg['self_range']
        tx_crate = range_to_list(self_range['x'])
        ty_crate = range_to_list(self_range['y'])
        tz_crate = range_to_list(self_range['z'])
        
        tfs_crate = []
        for x in tx_crate:
            for y in ty_crate:
                for z in tz_crate:
                    tf = tf_crate_standard.copy()
                    tf[:3, 3] = [x, y, z]
                    tfs_crate.append(tf)
    else:
        # suppose we have a list of crate base poses
        tfs_crate = [tf_crate_standard]
    
    dict_pose = {"crate": ampl.tf44_to_qt7(tf_crate_standard)}
    
    tcp_range = tote_cfg['tcp_range']
    eps = tcp_range['eps']
    x_range_in_crate = [tcp_range['x'][0]+eps, tcp_range['x'][1]-eps, tcp_range['x'][2]]
    y_range_in_crate = [tcp_range['y'][0]+eps, tcp_range['y'][1]-eps, tcp_range['y'][2]]
    z_range_in_crate = tcp_range['z']
    
    list_rx_in_base = range_to_list(tcp_range['rx'])
    list_ry_in_base = range_to_list(tcp_range['ry'])
    list_rz_in_tcp = range_to_list(tcp_range['rz'])

    keys_crate, poses_crate, pose_dims = genenerate_tcp_pose(
        x_range_in_crate, y_range_in_crate, z_range_in_crate,
        rx_rot=list_rx_in_base,
        ry_rot=list_ry_in_base,
        rz_rot=list_rz_in_tcp
    )
    print(f"Total poses to check per crate pose: {len(keys_crate)}")

    n_joints = agent.dim
    # Matrix shape: [NX, NY, NZ, nx, ny, nz, nrx, nry, nrz, n_joints]
    matrix_shape = (len(tx_crate), len(ty_crate), len(tz_crate)) + pose_dims + (n_joints,)
    reachability_matrix = np.full(matrix_shape, np.nan)
    
    # Pose matrix shape: [NX, NY, NZ, nx, ny, nz, nrx, nry, nrz, 7] (7 for qt7: [x,y,z, qw,qx,qy,qz])
    pose_matrix_shape = (len(tx_crate), len(ty_crate), len(tz_crate)) + pose_dims + (7,)
    reachable_pose_matrix = np.full(pose_matrix_shape, np.nan)

    for i_tf, tf_crate in enumerate(tfs_crate):
        # Index for crate pose (X, Y, Z)
        IX = i_tf // (len(ty_crate) * len(tz_crate))
        rem = i_tf % (len(ty_crate) * len(tz_crate))
        IY = rem // len(tz_crate)
        IZ = rem % len(tz_crate)

        dict_pose["crate"] = ampl.tf44_to_qt7(tf_crate)    
        update_collision_scene(collsion_scene, dict_pose)
    
        rwts_tcp = []        
        for itf, tf_in_crate in enumerate(poses_crate):
            tf_tcp = tf_crate @ tf_in_crate.copy()            
            tf_tcp[:3,:3] = tf_tcp[:3,:3] @ R_base_tcp_standard
            rwts_tcp.append(ampl.tf44_to_qt7(tf_tcp))

        mask_tool = collsion_scene.collision_free_external(cvh_gripper, rwts_tcp)
        rwts_tcp = np.array(rwts_tcp)
    
        for i_pose, rwt_tcp_valid in enumerate(rwts_tcp):
            if not mask_tool[i_pose]:
                continue
            tf_tcp_valid = ampl.qt7_to_tf44(rwt_tcp_valid)
            qs_ik = agent.ik_redundant_wall_torso_df(
                tf_tcp_valid, state_ref=state, nb_redundant_search=ik_cfg['nb_redundant_search'], env=collsion_scene
            )
            if len(qs_ik) > 0: # means we found collision free ik
                state = qs_ik[0].copy()
                
                # Index for TCP pose (x, y, z, rx, ry, rz)
                nx, ny, nz, nrx, nry, nrz = pose_dims
                ix = i_pose // (ny * nz * nrx * nry * nrz)
                rem = i_pose % (ny * nz * nrx * nry * nrz)
                iy = rem // (nz * nrx * nry * nrz)
                rem = rem % (nz * nrx * nry * nrz)
                iz = rem // (nrx * nry * nrz)
                rem = rem % (nrx * nry * nrz)
                irx = rem // (nry * nrz)
                rem = rem % (nry * nrz)
                iry = rem // nrz
                irz = rem % nrz

                reachability_matrix[IX, IY, IZ, ix, iy, iz, irx, iry, irz, :] = state
                reachable_pose_matrix[IX, IY, IZ, ix, iy, iz, irx, iry, irz, :] = rwt_tcp_valid
        
    np.save("reachability_results.npy", reachability_matrix)
    np.save("reachable_pose.npy", reachable_pose_matrix)
    
    # Save the grid values separately for reference
    np.savez("reachability_metadata.npz", 
             tx_crate=tx_crate, ty_crate=ty_crate, tz_crate=tz_crate,
             tx=range_to_list(x_range_in_crate), 
             ty=range_to_list(y_range_in_crate), 
             tz=range_to_list(z_range_in_crate),
             rx=list_rx_in_base, ry=list_ry_in_base, rz=list_rz_in_tcp)

    print(f"Saved results to reachability_results.npy and reachable_pose.npy")
    print(f"Matrix shape: {reachability_matrix.shape}")
    
if __name__ == "__main__":
    '''
    python3.11 reachability/reachability_test.py --config reachability/config.yaml
    '''
    parser = argparse.ArgumentParser(description="Reachability test using YAML config.")
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        help="Path to the config.yaml file"
    )
    args = parser.parse_args()

    # If run in subfolder, set DIR_ASSETS to point to motion_plan/assets
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.abspath(os.path.join(script_dir, "../assets"))
    main_v2(config_path=args.config, DIR_ASSETS=assets_dir)
