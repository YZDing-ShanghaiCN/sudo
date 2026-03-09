import numpy as np
import ampl
import pyampl






def create_grid_transforms(
    num_instances: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_size = int(np.ceil(np.sqrt(num_instances)))
    # Create grid positions.

    x = np.arange(grid_size) - (grid_size - 1) / 2
    y = np.arange(grid_size) - (grid_size - 1) / 2
    xx, yy = np.meshgrid(x, y)
    positions = np.zeros((grid_size * grid_size, 3), dtype=np.float32)
    positions[:, 0] = xx.flatten()
    positions[:, 1] = yy.flatten()
    positions[:, 2] = 0.0
    positions = positions[:num_instances]
    rotations = np.zeros((num_instances, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # w component = 1
    # Initial scales.

    scales = np.linalg.norm(positions, axis=-1)
    scales = np.sin(scales * 1.5) * 0.5 + 1.0
    return positions



def batch_fk(b_rwt: np.ndarray, rwt: np.ndarray, b_p: np.ndarray):
    for i_j in range(len(b_rwt)):
        b_rwt[i_j][:] = rwt[i_j]
        b_rwt[i_j][:, -3:] += b_p
    return b_p


def main(DIR_ASSETS: str = "./assets"):
    DIR_WP = "/home/czhou/Playground"
    ############################################################################
    arm_config = pyampl.create_default_arm_config("hillbot_right")
    agent = pyampl.AgentArm(arm_config.name, arm_config.dim, arm_config)
    agent.fk_rwt = agent.state_ref
    agent.wall = np.array([-0.05, -1.6, 0.6, 100, 0.3, 2.0]).astype(pyampl.DTypeFloat)
    state = np.array(agent.state_ref.tolist(), dtype=np.float64)
    ############################################################################
    F_GRIPPER = f"{DIR_ASSETS}/mesh/scene_00/tool/gripper.ply"
    F_CRATE = f"{DIR_ASSETS}/mesh/scene_00/obstacle/convex/crate_0.ply"
    cvh_gripper = pyampl.CollisionObjectConvex(ampl.read_trimesh(F_GRIPPER))
    mesh_crate = ampl.read_trimesh(F_CRATE)
    collsion_scene = pyampl.CollisionScene()
    collsion_scene.insert_convex("crate", mesh_crate, create_pcd=True, pcd_dx=0.0025)
    ############################################################################

    def update_collision_scene(collsion_scene:pyampl.CollisionScene,dict_pose:dict[str,np.ndarray]):        
        for name,rwt in dict_pose.items():          
            collsion_scene.update_pose(name, rwt)
            collsion_scene.update_poses_pcd_from_convex()
            collsion_scene.enable_collision(name)
        pcd_tmp = collsion_scene.get_pointcloud(list(dict_pose.keys()))        
        collsion_scene.update_df_from_pcd(pcd_tmp)    
    
    ############################################################################
    """    
    gripper x -> robot base y
    gripper y -> robot base z
    gripper z -> robot base x
    R_base_griper_standard  =
    [
    [0,0,1],
    [1,0,0],
    [0,1,0]
    ]
    """
    # Rotation of standard tcp frame in base
    R_base_tcp_standard=np.array([    [0,0,1],    [1,0,0],    [0,1,0]    ],dtype=np.float64)    
    # Translation of standard tcp frame in base (can be anything)
    t_base_tcp=np.array([0.5,-0.1,1.3])

    tf_tcp_standard =np.eye(4,dtype=np.float64)
    tf_tcp_standard[:3,:3]=R_base_tcp_standard
    tf_tcp_standard[:3,3]=t_base_tcp.flatten()

    
    tf_crate_standard =np.eye(4,dtype=np.float32)
    tf_crate_standard[:3,3]=np.array([0.75,-0.1,0.7])   

    # suppose we have a list of create poses
    tfs_crate=[tf_crate_standard]
    dict_pose={"crate": ampl.tf44_to_qt7(tf_crate_standard)}

    for tf_crate in tfs_crate:
        dict_pose["crate"] = ampl.tf44_to_qt7(tf_crate)    
        update_collision_scene(collsion_scene,dict_pose)
    
    # a list of pose tcp in base  in rwt format [[rx ry rz w tx ty tz], ...]
        rwts_tcp=[]
        x_range=[tf_crate[0,3]-0.3,tf_crate[0,3]+0.3]
        dx = 0.01
        y_range=[tf_crate[1,3]-0.2,tf_crate[1,3]+0.2]
        dy = 0.01
        z_range=[tf_crate[2,3]+0.05,tf_crate[2,3]+0.1]
        dz = 0.05    
        R_rotx_in_tcp = ampl.so3_upexp(np.array([np.pi/4,0,0]))    
        for z_base in np.linspace(z_range[0],z_range[1], int((z_range[1]-z_range[0])/dz)):
            for x_base in np.linspace(x_range[0],x_range[1], int((x_range[1]-x_range[0])/dx)):
                for y_base in np.linspace(y_range[0],y_range[1],int((y_range[1]-y_range[0])/dy)):
                    # first create a 4x4 tf_tcp in base
                    tf_tcp=tf_tcp_standard.copy()
                    tf_tcp[:3,3]=[x_base,y_base,z_base]
                    tf_tcp[:3,:3] = tf_tcp[:3,:3]@ R_rotx_in_tcp
                    # convert it to rwt format 
                    rwts_tcp.append(ampl.tf44_to_qt7(tf_tcp))

    # now we have a list of pose tcp in base 
    # list of collision status (bool), true = collision free
        mask_tool = collsion_scene.collision_free_external(cvh_gripper,rwts_tcp)
        rwts_tcp = np.array(rwts_tcp)
    
        states=[]
        for i, rwt_tcp_valid in enumerate(rwts_tcp):
            if not mask_tool[i]:
                continue
            tf_tcp_valid = ampl.qt7_to_tf44(rwt_tcp_valid)
            qs_ik = agent.ik_redundant_wall_torso_df(
                tf_tcp_valid, state_ref=state, nb_redundant_search=512, env=collsion_scene
            )
            if len(qs_ik) > 0: # means we found collision free ik
                state=qs_ik[0].copy()
                states.append(state.copy())
        
    
    print(len(states))

    exit()

if __name__ == "__main__":
    main("./assets/")
    


