import numpy as np
import ampl
import pyampl


def genenerate_tcp_pose(x_range:list,y_range:list,z_range:list,y_rot_in_base:list=[0,np.pi/2],z_rot_in_tcp:list=[-np.pi/4,0,np.pi/4]):
    tx = np.linspace(x_range[0],x_range[1], int((x_range[1]-x_range[0])/x_range[2]))
    ty = np.linspace(y_range[0],y_range[1], int((y_range[1]-y_range[0])/y_range[2]))
    tz = np.linspace(z_range[0],z_range[1], int((z_range[1]-z_range[0])/z_range[2]))
    
    key_R=[]
    for ry in y_rot_in_base:
        for rz in z_rot_in_tcp:            
            key_R.append([0,ry,rz])
    mat_R=[]
    for r in key_R:
        R_in_base = ampl.so3_upexp(np.array([0,0,r[2]])) @ ampl.so3_upexp(np.array([0,r[1],0]))    
        mat_R.append(R_in_base)

    keys=[]
    poses=[]
    for z in tz:
        for y in ty:
            for x in tx:
                for iR, R_in_base in enumerate(mat_R):
                    big_key = [x,y,z,key_R[iR][0],key_R[iR][1],key_R[iR][2]]
                    tf =np.eye(4,dtype=np.float64)
                    tf[:3,:3]=R_in_base
                    tf[:3,3]=[x,y,z]
                    keys.append(big_key)
                    poses.append(tf)

    return keys,poses



def main_v2(DIR_ASSETS: str = "./assets"):    
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
    tf_crate_standard[:3,3]=np.array([0.75,-0.2,0.7])

    # suppose we have a list of crate base poses
    tfs_crate=[tf_crate_standard]
    dict_pose={"crate": ampl.tf44_to_qt7(tf_crate_standard)}
    eps=0.025
    x_range_in_crate=[-0.3+eps,0.3-eps,0.025]
    y_range_in_crate=[-0.2+eps,0.2-eps,0.025]
    z_range_in_crate=[0.015,0.25,0.05]
    list_ry_in_base=[np.pi/4,np.pi/2]
    keys_crate, poses_crate = genenerate_tcp_pose(x_range_in_crate,y_range_in_crate, z_range_in_crate,y_rot_in_base=list_ry_in_base)
    print(len(keys_crate))

    for tf_crate in tfs_crate:
        dict_pose["crate"] = ampl.tf44_to_qt7(tf_crate)    
        update_collision_scene(collsion_scene,dict_pose)
    
    # a list of pose tcp in base  in rwt format [[rx ry rz w tx ty tz], ...]
        rwts_tcp=[]        
        for itf, tf_in_crate in enumerate(poses_crate):
            tf_tcp=tf_crate@tf_in_crate.copy()            
            tf_tcp[:3,:3]=tf_tcp[:3,:3]@R_base_tcp_standard
            rwts_tcp.append(ampl.tf44_to_qt7(tf_tcp))


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
        
    
    print("state length", len(states))
    
if __name__ == "__main__":
    main_v2()
