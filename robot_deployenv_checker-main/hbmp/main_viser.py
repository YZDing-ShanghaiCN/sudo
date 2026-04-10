import os
import time
import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy
import json
import ampl
import pywbc
from hbmp import tictoc, FrameEnum, ColGroup, Robot_T2DA2, MopRRTC, TrajLinear
import json
from scene_manager import ConfigSceneManager
#from util_trimesh import create_obb_trimesh
from enum import Enum, auto


q_ref = np.array(
    [
        0.15,
        0.3,  # noqa
        0.8,
        0.64,
        1.5,
        -1.65,
        -0.8,
        -0.8,
        0.6,  # NOAQ
        0.8,
        0.64,
        1.5,
        -1.65,
        -0.8,
        -0.8,
        0.6,
    ]  # noqa
)
 


class RobotMode(Enum):
    IDLE = auto()
    MOVE_S = auto()
    PLAN_Q = auto()
    TRACK = auto()
    RAND_Q = auto()


# 2. Create a simple State Manager object
class ModeManager:
    def __init__(self):
        self.current_mode = RobotMode.IDLE

    def set_mode(self, new_mode: RobotMode):
        self.current_mode = new_mode


def to_viz(q16: np.ndarray, gripper_left: float, gripper_right: float) -> np.ndarray:
    q_viz = np.zeros(18, dtype=np.float64)
    np.copyto(q_viz[:9], q16[:9])
    np.copyto(q_viz[9 + 1 : 16 + 1], q16[9:16])
    q_viz[2 + 7] = gripper_left
    q_viz[-1] = gripper_right
    return q_viz

def create_obb_trimesh(obb3: ampl.OBB3, offset_radius: float = 0):
    import trimesh

    obb_trimesh = trimesh.creation.box(
        extents=[e - offset_radius / 2 for e in obb3.half_extents]
    )
    obb_trimesh.apply_scale(2.0)
    tf = np.eye(4)
    tf[:3, :3] = np.array([obb3.u, obb3.v, obb3.w]).T
    tf[:3, 3] = np.array(obb3.center)
    obb_trimesh.apply_transform(tf)
    if offset_radius == 0:
        return obb_trimesh
    sphere_template = trimesh.creation.uv_sphere(
        radius=offset_radius / 2, count=[16, 16]
    )
    return trimesh.convex.convex_hull(
        np.vstack(
            [sphere_template.vertices + corner for corner in obb_trimesh.vertices]
        )
    )
def main():

    server = viser.ViserServer()
    manager = ConfigSceneManager(server)
    CONFIG_FILE = "scene_config_hb.json"
    manager.load_scene(CONFIG_FILE)
    mode_manager = ModeManager()    

    CONFIG_FILE_WBC = "wbc_config_hb.yaml"
    agent = Robot_T2DA2("hb11_left", "hb11_right", "hb11_torso", CONFIG_FILE_WBC, 16)
    # q_ref[2:] *= 0

    np.copyto(agent.q, q_ref)
    agent.update_kin(agent.q)
    agent.set_wall(x_wall=[0, 1.25], z_wall=[0.9, 2.0])
    traj = []
    i_traj = 0
    ####################################################
    manager.toggle_gizmos("t2da2", False)

    # @(manager.active_handles["left_tool0"].on_update)
    # async def _(_):
    #     tf_target = pywbc.Tf(manager.get_object_pose("left_tool0"))
    #     agent.q = agent.track_tcp(FrameEnum.FRAME_TACTILE_L, tf_target)
    #     manager.set_object_state("t2da2", to_viz(agent.q, 0.05, 0.05))

    # @(manager.active_handles["right_tool0"].on_update)
    # async def _(_):
    #     tf_target = pywbc.Tf(manager.get_object_pose("right_tool0"))
    #     agent.q = agent.track_tcp(FrameEnum.FRAME_TACTILE_R, tf_target)
    #     manager.set_object_state("t2da2", to_viz(agent.q, 0.05, 0.05))
    viz_grid = server.scene.add_grid(
        "z elbo", 2, 2, cell_size=0.1, section_color=(0, 0, 0), cell_color=(0, 0, 0)
    )

    ##########################################
    # wall = agent.wall()
    # agent.set_wall(
    #             y_wall=wall[1],
    #             z_wall=[sld_z.value[0], sld_z.value[1]],
    #             x_wall=wall[0],
    #         )

    viz_wall = manager.add_bounding_box(
        "/wall", [item for sublist in agent.wall() for item in sublist]
    )

    vf = create_obb_trimesh(agent._obb_T.entity, 0.0)
    if vf is not None:
        viz_obb = server.scene.add_mesh_simple(
            "obb",
            vf.vertices,
            vf.faces,
            color=(128, 0, 0),
            flat_shading=True,
            opacity=0.5,
        )
    viz_col = server.scene.add_point_cloud(
        "col",
        np.array(
            [
                [
                    0,
                    0,
                    0,
                ]
            ]
        ),
        colors=(255, 0, 0),
        point_size=0.005,
    )
    ##############################################

    with server.gui.add_folder("Agent Controls"):

        btn_track = server.gui.add_button("Track")

        @btn_track.on_click
        def _(_):
            mode_manager.set_mode(RobotMode.TRACK)

        btn_plan = server.gui.add_button("Plan Q")

        @btn_plan.on_click
        def _(_):
            nonlocal traj, i_traj
            mode_manager.set_mode(RobotMode.PLAN_Q)
            lims = agent.get_limits()
            full_bounds = [(jlo, jhi) for jlo, jhi in zip(lims[0], lims[1])]
            q_freeze = {0: agent.q[0], 1: agent.q[1]}

            def collision_wrapper(state: np.ndarray) -> bool:
                agent.update_kin(state)
                agent.update_col_self()
                return agent.check_self_collision()

            planner = MopRRTC(
                bounds=full_bounds,
                collision_fn=collision_wrapper,
                step_size=0.1,
                max_extend_dist=1.0,
                frozen_dofs=q_freeze,
            )
            waypoints_feasible = planner.plan(agent.q, q_ref, max_iterations=2000)
            # print(len(waypoints_feasible))
            if waypoints_feasible is None:
                traj = None
                i_traj = 0
            else:
                waypoints_shortcut = planner.shortcut(waypoints_feasible)

                simple_traj = TrajLinear(waypoints_shortcut)
                traj = simple_traj.evaluate(
                    np.linspace(
                        simple_traj.arc_lengths[0], simple_traj.arc_lengths[-1], 128
                    )
                )                
                # traj = np.vstack([traj, traj[::-1]])
                i_traj = 0

        btn_move_s = server.gui.add_button("Move Slerp")

        @btn_move_s.on_click
        async def _(_):
            mode_manager.set_mode(RobotMode.MOVE_S)
            # global q_ref
            nonlocal traj, i_traj,agent            
            agent.update_kin(q_ref)
            tf_to_L = agent.get_fk(FrameEnum.FRAME_TACTILE_L)
            tf_to_R = agent.get_fk(FrameEnum.FRAME_TACTILE_R)
            agent.update_kin(agent.q)
            tf_from_L = agent.get_fk(FrameEnum.FRAME_TACTILE_L)
            tf_from_R = agent.get_fk(FrameEnum.FRAME_TACTILE_R)
            
            
            agent.update_kin(agent.q)            
            traj = []
            i_traj = 0
            for t in np.linspace(0, 1, 20, endpoint=True):
                tf_s = tf_from_L.slerp(t, tf_to_L)            
                agent.q = agent.track_tcp(FrameEnum.FRAME_TACTILE_L, tf_s)
                tf_s = tf_from_R.slerp(t, tf_to_R)            
                agent.q = agent.track_tcp(FrameEnum.FRAME_TACTILE_R, tf_s)
                traj.append(agent.q)


            
            # traj = np.vstack([traj, traj[::-1]])
            # print(len(traj))
            # print(traj[0], traj[-1])
            

        btn_rand_q = server.gui.add_button("Random Q")

        @btn_rand_q.on_click
        def _(_):
            mode_manager.set_mode(RobotMode.RAND_Q)

            lims = agent.get_limits()

            find_q = False
            while not find_q:
                q_rand = np.random.uniform(low=lims[0], high=lims[1])
                agent.update_kin(q_rand)
                agent.update_col_self()
                if agent.check_self_collision() == ColGroup.NONE:
                    np.copyto(agent.q, q_rand)
                    break

        # sld_z = server.gui.add_slider("wall_z_min", 0.6, 1.5, 0.01)
        sld_y = server.gui.add_slider(
            "wall_y_max",
            0.1,
            1.5,
            0.01,
            1,
        )
        sld_z = server.gui.add_multi_slider(
            "wall_z_minmax",
            min=0.0,
            max=2,
            step=0.01,
            initial_value=(agent.wall()[2][0], agent.wall()[2][1]),
        )

        @sld_z.on_update
        def _(_):
            nonlocal viz_wall
            wall = agent.wall()
            agent.set_wall(
                y_wall=wall[1],
                z_wall=[sld_z.value[0], sld_z.value[1]],
                x_wall=wall[0],
            )

            viz_wall = manager.add_bounding_box(
                "/wall", [item for sublist in agent.wall() for item in sublist]
            )

        @sld_y.on_update
        def _(_):
            nonlocal viz_wall
            wall = agent.wall()
            agent.set_wall(
                y_wall=[-sld_y.value, sld_y.value],
                x_wall=wall[0],
                z_wall=wall[2],
            )

            viz_wall = manager.add_bounding_box(
                "/wall", [item for sublist in agent.wall() for item in sublist]
            )
            # tmp = viz_wall["ymax"].position
            # tmp[1] = sld_y.value
            # print(tmp)
            # aaaa = np.array([0, float(sld_y.value), 0])

            # print(agent.wall())

    while True:
        current = mode_manager.current_mode
        # print(current)
        if current == RobotMode.TRACK:
            tf_target = ampl.Tf(manager.get_object_pose("left_tool0"))
            agent.q = agent.track_tcp(FrameEnum.FRAME_TACTILE_L, tf_target)
            manager.set_object_state("t2da2", to_viz(agent.q, 0.05, 0.05))
            tf_target = ampl.Tf(manager.get_object_pose("right_tool0"))
            agent.q = agent.track_tcp(FrameEnum.FRAME_TACTILE_R, tf_target)
            manager.set_object_state("t2da2", to_viz(agent.q, 0.05, 0.05))

            agent.update_kin(agent.q)
            agent.update_col_self()
            tf_torso = agent.get_fk(FrameEnum.FRAME_TORSO_2)
            viz_obb.position, viz_obb.wxyz = (tf_torso.position, tf_torso.wxyz)
            viz_obb.visible = (agent.check_self_collision()) != ColGroup.NONE
        elif (current == RobotMode.MOVE_S) or (current == RobotMode.PLAN_Q) :                       
        
            
            if traj is None:
                continue
            if len(traj) == 0:
                continue
            if (len(traj) != 20) and (len(traj) != 128):
                continue
            if i_traj == len(traj):
                mode_manager.set_mode(RobotMode.IDLE)
                i_traj = 0
                traj = []
                continue

            
            manager.set_object_state("t2da2", to_viz(traj[i_traj], 0.05, 0.05))            
            np.copyto(agent.q, traj[i_traj])
            i_traj = i_traj + 1

        elif current == RobotMode.IDLE or RobotMode.RAND_Q:
            manager.set_object_state("t2da2", to_viz(agent.q, 0.05, 0.05))
            agent.update_kin(agent.q)
            agent.update_col_self()
            tf_torso = agent.get_fk(FrameEnum.FRAME_TORSO_2)
            viz_obb.position, viz_obb.wxyz = (tf_torso.position, tf_torso.wxyz)
            viz_obb.visible = (agent.check_self_collision()) != ColGroup.NONE
        # elif current==RobotMode.RAND_Q:

        viz_grid.position = np.array(
            [
                0,
                0,
                min(
                    agent.get_arm_bound(FrameEnum.FRAME_TACTILE_L, "z_min"),
                    agent.get_arm_bound(FrameEnum.FRAME_TACTILE_R, "z_min"),
                ),
            ]
        )
        time.sleep(1e-3)


if __name__ == "__main__":
    main()
