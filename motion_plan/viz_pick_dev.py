import sys
import os

#sys.path.append(["/home/czhou/Projects/pyampl"])
from datetime import datetime
import numpy as np
import sys,json, os, time
import ampl
import pyampl
import viser
from viz_viser import ViserObject
from yourdfpy import URDF
from enum import Enum
#from util_trimesh import load_multiple_mesh

def load_multiple_mesh(directory_path: str):
    import os, trimesh

    files = sorted(
        [
            (os.path.splitext(f)[0], os.path.join(directory_path, f))
            for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f))
        ]
    )
    dict_convex_data = {}
    for f in files:
        m = trimesh.load(f[1], process=False)
        dict_convex_data[f[0]] = {"vf": (m.vertices, m.faces), "trimesh": m}
    return dict_convex_data

class RobotStateOwner(Enum):
    IK = 0
    JPLANNER = 1
    CPLANNER = 2


def main(DIR_ASSETS: str = "./assets"):
    ######################################################
    DOF = 7
    arm_config = pyampl.create_default_arm_config("hillbot_left")
    agent = pyampl.AgentArm(arm_config.name, arm_config.dim, arm_config)
    #agent.collision_free_trajectory_no_attach()

    #agent.pose_base = ampl.tf44_to_qt7(tf_arm_base_home)
    #agent.obb3_torso = OBB3_HILLBOT_TORSO_HOME
    agent.fk_rwt = agent.state_ref
    #agent.wall = np.array([-2, -2, 0.6, 100, 2, 2.0]).astype(pyampl.DTypeFloat)
    agent.wall[2]=0.6
    #agent.collision_free_trajectory_no_attach()
    #agent.collision_free_trajectory = MethodType(agent.collision_free_trajectory_no_attach, agent)
    #pyampl.AgentArm.collision_free_trajectory = pyampl.AgentArm.collision_free_trajectory_no_attach
    #pyampl.AgentArm.collision_free_trajectory = pyampl.AgentArm.collision_free_trajectory_attach
    
    #agent.which_iks=[5,6]
    state = np.array(agent.state_ref.tolist(), dtype=np.float64)
    state_from = np.array(agent.state_ref.tolist(), dtype=np.float64)
    state_to = np.array(agent.state_ref.tolist(), dtype=np.float64)
    state_traj = np.array([state_from,state_to], dtype=np.float64)
    id_state_traj=0
    workbench_center = np.array([0.7, 0, 0.7])
    crate_center = workbench_center + np.array([0, 0, 0.01])
    rwt_gripper_entity_raw = np.loadtxt(
        f"{DIR_ASSETS}/convex/milk_bag/grasp_rwt.txt", dtype=np.float64
    )[::3]
    rwt_gripper_entity_raw[:, :4] /= np.linalg.norm(
        rwt_gripper_entity_raw[:, :4], axis=1, keepdims=True
    )
    rwt_flipy = ampl.tf44_to_qt7(  np.array([[-1.0000000e+00, -1.2246468e-16,  0.0000000e+00,  0.0000000e+00],
       [ 1.2246468e-16, -1.0000000e+00,  0.0000000e+00,  0.0000000e+00],
       [ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00,  0.0000000e+00],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],dtype=np.float64))
    rwt_gripper_entity_flip=np.zeros_like(rwt_gripper_entity_raw)
    for ir in range(0,len(rwt_gripper_entity_flip)):
        rwt_gripper_entity_flip[ir]=ampl.rwtmul(rwt_gripper_entity_raw[ir],rwt_flipy)
        
    rwt_gripper_entity = np.zeros((rwt_gripper_entity_raw.shape[0]*2,7))
    rwt_gripper_entity[0::2] = rwt_gripper_entity_raw
    rwt_gripper_entity[1::2] = rwt_gripper_entity_flip
    ##########################################################################
    # cvh_gripper_vf = 

    cvh_gripper = pyampl.CollisionObjectConvex(ampl.read_trimesh(f"{DIR_ASSETS}/mesh/scene_00/tool/gripper.ply"))
    cvh_pick = pyampl.CollisionObjectConvex(ampl.read_trimesh(f"{DIR_ASSETS}/mesh/scene_00/tool/milk_bag.ply"))
    collsion_scene = pyampl.CollisionScene()

    dict_convex = load_multiple_mesh(f"{DIR_ASSETS}/mesh/scene_00/obstacle/convex")

    for name, data in dict_convex.items():
        tuple_vf = data["vf"]
        collsion_scene.insert_convex(name, tuple_vf, create_pcd=True, pcd_dx=0.0025)
        collsion_scene.enable_collision(name)

    agent.cvh_attach=cvh_pick
    agent.validate()
    ##########################################################################
    server = viser.ViserServer()
    ##########################################################################
    grid_handler = server.scene.add_grid(
        "/ground",
        width=0.8,
        height=0.8,
        cell_size=0.1,
        section_color=[0, 0, 0],
        cell_color=[0, 0, 0],
        cell_thickness=2,
        section_thickness=2,
    )
    scene_manager: dict[str, ViserObject] = {}
    name_obstacle = "desk"
    scene_manager[name_obstacle] = ViserObject(
        f"{DIR_ASSETS}/mesh/desk.ply",
        name=name_obstacle,
        server=server,
        control_scale=0.15,
        color=[128, 255, 128, 255],
    )
    name_obstacle = "crate_0"
    scene_manager[name_obstacle] = ViserObject(
        f"{DIR_ASSETS}/mesh/scene_00/obstacle/visual/crate_0.ply",
        name=name_obstacle,
        server=server,
        control_scale=0.15,
        color=[128, 128, 255, 255],
    )
    scene_manager[name_obstacle].handler.position=np.array([0,0,-0.01])
    name_obstacle = "crate_1"
    scene_manager[name_obstacle] = ViserObject(
        f"{DIR_ASSETS}/mesh/scene_00/obstacle/visual/crate_1.ply",
        name=name_obstacle,
        server=server,
        control_scale=0.15,
        color=[128, 128, 255, 255],
    )
    scene_manager[name_obstacle].handler.position=np.array([0,0,-0.01])
    # name_obstacle = "shelf"
    # scene_manager[name_obstacle] = ViserObject(
    #     f"{DIR_ASSETS}/mesh/scene_00/obstacle/visual/shelf.ply",
    #     name=name_obstacle,
    #     server=server,
    #     control_scale=0.15,
    #     color=[170, 170, 170, 255],
    # )

    scene_manager["torso"] = ViserObject(
        f"{DIR_ASSETS}/mesh/torso_vla_fine.ply",
        name="torso",
        server=server,
    )

    scene_manager["milk_bag"] = ViserObject(
        f"{DIR_ASSETS}/convex/milk_bag/meshes/visual/object.ply",
        name="milk_bag",
        server=server,
        color=[255, 200, 28, 255],
        control_scale=0.1
    )
    scene_manager["milk_bag_collide"] = ViserObject(
        f"{DIR_ASSETS}/convex/milk_bag/meshes/visual/object.ply",
        name="milk_bag",
        server=server,
        color=[255, 0, 0, 255],
        affix="collide",no_control=True
    )
    scene_manager["arm"] = ViserObject(
        URDF.load(f"{DIR_ASSETS}/urdf/{agent.name}/urdf.urdf"),
        name="arm",
        server=server,
    )

    ##########################################################################

    grid_handler.position = (
        agent.pose_base[-3],
        agent.pose_base[-2],
        workbench_center[2],
    )

    for name in ["crate_0", "crate_1", "desk", "milk_bag"]:
        if name == "crate_0":
            scene_manager[name].set_control(
                np.array(
                    [
                        0,
                        0,
                        0,
                        1,
                        crate_center[0],
                        crate_center[1] - 0.2,
                        crate_center[2],
                    ]
                )
            )
        elif name == "crate_1":
            scene_manager[name].set_control(
                np.array(
                    [
                        0,
                        0,
                        0,
                        1,
                        crate_center[0],
                        crate_center[1] + 0.2,
                        crate_center[2],
                    ]
                )
            )
        elif name == "milk_bag":
            x_rot = 0.75
            scene_manager[name].set_control(
                np.array(
                    [
                        x_rot,
                        0,
                        0,
                        np.sqrt(1 - x_rot**2),
                        crate_center[0],
                        crate_center[1] - 0.2,
                        crate_center[2] + 0.6,
                    ]
                )
            )        
        else:
            scene_manager[name].set_control(
                np.array(
                    [
                        0,
                        0,
                        0,
                        1,
                        workbench_center[0],
                        workbench_center[1],
                        workbench_center[2],
                    ]
                )
            )
    # scene_manager["desk"].set_control(np.array([0, 0, 0, 1, workbench_center[0],workbench_center[1], workbench_center[2]]))

    scene_manager["arm"].set_control(agent.pose_base)
    scene_manager["arm"].disable_control()
    scene_manager["arm"].urdf.update_cfg(agent.state_ref)
    #scene_manager["desk"].disable_control()
    scene_manager["torso"].disable_control()
    scene_manager["desk"].control.disable_rotations=True
    scene_manager["milk_bag_collide"].handler.visible=False

    def update_collision_scene(collsion_scene:pyampl.CollisionScene):
        names = ["crate_0", "crate_1"]
        for name in names:          
            collsion_scene.update_pose(name, scene_manager[name].pose_rwt())
        collsion_scene.update_poses_pcd_from_convex()
        pcd_tmp = collsion_scene.get_pointcloud(names)
        #ampl.write_pointcloud("/home/czhou/Playground/s.ply", pcd_tmp)
        collsion_scene.update_df_from_pcd(pcd_tmp)
    update_collision_scene(collsion_scene)
    ##########################################################################
    server.gui.configure_theme(
        control_width="large", show_logo=False, show_share_button=False
    )

    @server.on_client_connect
    async def _(client: viser.ClientHandle) -> None:
        client.camera.position = (20, 20, 20)
        client.camera.look_at = (
            workbench_center[0],
            workbench_center[1],
            workbench_center[0] + 0.5,
        )
        client.camera.fov = 0.05

    state_render_owner = RobotStateOwner.IK
    bn_objpose = server.gui.add_button("Update Scene Poses")
    slider_gripper = server.gui.add_slider(
        "pose id", initial_value=857, min=0, max=len(rwt_gripper_entity) - 1, step=1
    )
    bn_grasp = server.gui.add_button("Collision-Free Grasp")    
    hl_path = server.gui.add_text("Path Scene JSON", str(datetime.now().strftime("%Y%m%d%I%M%S"))+ ".json")
    
    
    bn_save = server.gui.add_button("Save",icon=viser.Icon.DOWNLOAD)    

    gui_upload_button = server.gui.add_upload_button(
                "Upload", icon=viser.Icon.UPLOAD
            )
    bn_state = server.gui.add_button_group("State",options=["Begin","End"])    
    bn_plan = server.gui.add_button_group("Plan",options=["With Attach","Without Attach"])    
    bn_play = server.gui.add_button("Play Trajectory")    
    
    @bn_play.on_hold(callback_hz=10.0)
    def _(_: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
        nonlocal id_state_traj,state_traj,state,scene_manager        
        #print(id_state_traj,"/",len(state_traj))
        
        id_state_traj= (id_state_traj) % len(state_traj)
        np.copyto(state,state_traj[id_state_traj])
        
        scene_manager["arm"].urdf.update_cfg(state)
        agent.fk_rwt=state

        rwt_base_tcp = agent.fk_rwt[-1].copy()
        rwt_tcp_obj = ampl.tf44_to_qt7( np.linalg.inv(ampl.qt7_to_tf44( rwt_gripper_entity[slider_gripper.value])).astype(np.float64))
        rwt_base_obj = ampl.rwtmul(rwt_base_tcp,rwt_tcp_obj)       
        scene_manager["milk_bag"].set_control(rwt_base_obj)
        id_state_traj= (id_state_traj+1) % len(state_traj)        
    
    @ (bn_state).on_click
    async def _(event: viser.GuiEvent) -> None:
        nonlocal state_from,state_to,state        
        tag = bn_state.value        
        if tag=="Begin":
            np.copyto(state_from,state)            
        if tag=="End":
            np.copyto(state_to,state)



    @ (bn_plan).on_click
    async def _(event: viser.GuiEvent) -> None:
        nonlocal state_from,state_to,state
        nonlocal agent, collsion_scene, state, state_traj
        tag = bn_plan.value        
        
        if "Without A" in tag:        
            print(tag)
            pyampl.AgentArm.collision_free_trajectory = pyampl.AgentArm.collision_free_trajectory_no_attach
        elif "With A" in tag:
            print(tag)
            pyampl.AgentArm.collision_free_trajectory = pyampl.AgentArm.collision_free_trajectory_attach
        
            
        mp = pyampl.RRTConnect(
        agent=agent,
        env=collsion_scene,
        q_init=state_from,
        q_goal=state_to,
        max_edge_length=0.5,
        max_samples=2000,
        edge_discrete_resolution=16)
        path = mp.rrt_connect()
        if len(path) == 0:
            bn_plan.label="Plan [Infeasible]"
            return
        bn_plan.label="Plan [Success]!"
        path = np.array(path)
        path_shortcut = mp.shortcut(path, nb_subdivision_internal=3)
        path_refine = pyampl.refine_trajectory_trivial(path_shortcut, nb_refine=32)
        len_feasible = pyampl.get_traj_length_from_waypoints(path)
        len_shortcut = pyampl.get_traj_length_from_waypoints(path_shortcut)
        print(f"{len_feasible}->{len_shortcut}>={np.linalg.norm(state_from-state_to)}")
        state_traj = path_refine.copy()
        #print(path)
            

    # @ (bn_play).
    # async def _(event: viser.GuiEvent) -> None:
    #     nonlocal state_from,state_to,state
    #     nonlocal agent, collsion_scene, state, state_traj

    @ gui_upload_button.on_upload
    def _(_) -> None:
        file = gui_upload_button.value
        dict_scene = json.loads(file.content)
        for name, obj in dict_scene.items():
            if name not in scene_manager:
                continue
            if hasattr(scene_manager[name],"control"):                
                scene_manager[name].set_control(np.array(obj["rwt"]))
            if name=="arm":                        
                np.copyto(state,np.array(dict_scene["arm"]["state"]))
                scene_manager["arm"].urdf.update_cfg(state)
                
                #scene_manager["milk_bag"].set_control(agent.)
        update_collision_scene(collsion_scene)    

    @ (bn_objpose).on_click
    async def _(event: viser.GuiEvent) -> None:
        nonlocal collsion_scene
        update_collision_scene(collsion_scene)    
                
    @ (bn_save).on_click
    async def _(event: viser.GuiEvent) -> None:
        nonlocal scene_manager
        bn_save.label="Save to "+hl_path.value
        if 1:
            dict_scene={}
            for key, value in scene_manager.items():                        
                if hasattr(value,"control"):                
                    wxzy = value.control.wxyz
                    pos = value.control.position
                    dict_scene[key]={"rwt":[wxzy[1],wxzy[2],wxzy[3],wxzy[0],pos[0],pos[1],pos[2]]}
                if key=="arm":

                    dict_scene[key]["state"]=state.tolist()
            with open(hl_path.value, 'w') as f:
                json.dump(dict_scene, f, indent=4) # indent=4 makes it human-readable
            json_string = json.dumps(dict_scene)

# Encode the JSON string to bytes
            bytes_object = json_string.encode('utf-8')
            server.send_file_download(hl_path.value,bytes_object)

    
    @ (bn_grasp).on_click
    def _(event: viser.GuiEvent) -> None:
        nonlocal rwt_gripper_entity,state
        TF_Z = np.array([[-1.0000000e+00, -1.2246468e-16,  0.0000000e+00,  0.0000000e+00],
       [ 1.2246468e-16, -1.0000000e+00,  0.0000000e+00,  0.0000000e+00],
       [ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00,  0.0000000e+00],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],dtype=np.float64)
        rwt_base_obj = scene_manager["milk_bag"].pose_rwt()
        rwts_tool0=np.zeros_like(rwt_gripper_entity)
        for irwt, rwt_gripper in enumerate(rwt_gripper_entity):
            rwts_tool0[irwt]=ampl.rwtmul(rwt_base_obj, rwt_gripper).copy() 
        is_obj_colfree = collsion_scene.collision_free_external(cvh_gripper,rwts_tool0)
        for ig , rwt_obj_tcp in enumerate(rwt_gripper_entity):            
            if (not is_obj_colfree[ig]):
                continue
            rwt_tool0 = ampl.rwtmul(rwt_base_obj, rwt_obj_tcp)       
            tf_tool0 = ampl.qt7_to_tf44(rwt_tool0)
            #tf_tool0=tf_tool0 @ TF_Z            
            if (tf_tool0[0,2]<0.3 ):
                continue
            if (tf_tool0[2,1]<0.2 ):
                continue
            qs_ik = agent.ik_redundant_wall_torso_df(
            tf_tool0, state_ref=state, nb_redundant_search=512, env=collsion_scene
        )
        #print(qs_ik)
            if len(qs_ik) > 0:
                np.copyto(state, qs_ik[0])
                slider_gripper.value = ig
                agent.pose_attach = ampl.rwtinv(rwt_gripper_entity[ig])
                break
           
        scene_manager["arm"].urdf.update_cfg(state)

        #print(np.sum(is_obj_colfree))

    @ (scene_manager["milk_bag"].control).on_update
    def _(event: viser.GuiEvent) -> None:
        nonlocal state_render_owner, scene_manager, collsion_scene, slider_gripper, rwt_gripper_entity , state
        #print(scene_manager["milk_bag"].control.position)
        rwt_obj_base = scene_manager["milk_bag"].pose_rwt()
        is_obj_colfree = collsion_scene.collision_free_external(cvh_pick,[rwt_obj_base])
        #print(is_obj_colfree)
        scene_manager["milk_bag"].handler.visible=is_obj_colfree[0]
        scene_manager["milk_bag_collide"].handler.visible=not is_obj_colfree[0]
        rwt_tcp_obj = rwt_gripper_entity[slider_gripper.value]
        rwt_tool0 = ampl.rwtmul(rwt_obj_base, rwt_tcp_obj)        
        tf_tool0 = ampl.qt7_to_tf44(rwt_tool0)
        qs_ik = agent.ik_redundant_wall_torso_df(
            tf_tool0, state_ref=state, nb_redundant_search=512, env=collsion_scene
        )
        #print(qs_ik)
        if len(qs_ik) > 0:
            np.copyto(state, qs_ik[0])
        scene_manager["arm"].urdf.update_cfg(state)

    ##########################################################################
    #agent.validate()


    while True:
        
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main("./assets/")
