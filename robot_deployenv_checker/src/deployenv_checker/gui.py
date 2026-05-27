"""GUI panel builder: per-robot Reachability / Torso&Rail / Cameras + global Display + Objects."""

from typing import Dict

import numpy as np
import viser
from hbmp import ColGroup

from .camera import CameraManager
from .config import CAMERA_GROUPS


class GuiBuilder:
    """Builds the Viser GUI for the deploy environment checker.

    Layout:
        Display (global view-mode)
        Robot: <name>          (one folder per robot)
            Reachability
            Torso & Rail
            Cameras
        Objects (global)
    """

    def __init__(
        self,
        server: viser.ViserServer,
        robots: Dict[str, "RobotController"],  # noqa: F821
        camera_mgrs: Dict[str, CameraManager],
        scene_manager,
        app=None,
    ):
        self.server = server
        self.robots = robots
        self.camera_mgrs = camera_mgrs
        self.scene = scene_manager
        self.app = app

        self.collision_labels: Dict[str, viser.GuiMarkdownHandle] = {}
        self.image_handles: Dict[str, dict] = {}  # robot_name -> {cam_name: image_handle}
        self.traj_status: viser.GuiMarkdownHandle = None

        self._build_panels()

    def _build_panels(self):
        self._build_display_panel()
        self._build_trajectory_panel()
        for robot_name, robot in self.robots.items():
            with self.server.gui.add_folder(f"Robot: {robot_name}"):
                self._build_reachability_panel(robot_name, robot)
                self._build_torso_rail_panel(robot_name, robot)
                self._build_joint_print_panel(robot_name, robot)
                if robot_name in self.camera_mgrs:
                    self._build_camera_panel(robot_name, self.camera_mgrs[robot_name])
        self._build_objects_panel()

    def _build_trajectory_panel(self):
        if self.app is None:
            return
        with self.server.gui.add_folder("Trajectory"):
            upload = self.server.gui.add_upload_button(
                "Load JSON", mime_type="application/json"
            )
            btn_play = self.server.gui.add_button("Play")
            btn_pause = self.server.gui.add_button("Pause")
            btn_stop = self.server.gui.add_button("Stop")
            sld_speed = self.server.gui.add_slider(
                "Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0,
            )
            self.traj_status = self.server.gui.add_markdown("No trajectory loaded.")

            @sld_speed.on_update
            def _(_):
                self.app.set_trajectory_speed(sld_speed.value)

            @upload.on_upload
            def _(_):
                ok, msg = self.app.load_trajectory_json(upload.value.content)
                prefix = "" if ok else "**Error:** "
                self.traj_status.content = f"{prefix}{msg}"

            @btn_play.on_click
            def _(_):
                self.app.play_trajectory()

            @btn_pause.on_click
            def _(_):
                self.app.pause_trajectory()

            @btn_stop.on_click
            def _(_):
                self.app.stop_trajectory()
                if self.app._traj_t is not None:
                    self.update_trajectory_status(0, len(self.app._traj_t),
                                                  float(self.app._traj_t[0]))

    def update_trajectory_status(self, idx: int, n: int, t: float):
        if self.traj_status is not None:
            self.traj_status.content = f"Frame {idx + 1}/{n}  t={t:.2f}s"

    def _build_reachability_panel(self, robot_name, robot):
        with self.server.gui.add_folder("Reachability"):
            btn_track = self.server.gui.add_button("Go to Target")

            @btn_track.on_click
            def _(_):
                if self.app is not None:
                    self.app.move_to_eef_targets(robot_name)

            self.collision_labels[robot_name] = self.server.gui.add_markdown(
                "Collision: **NONE**"
            )

            wall = robot.wall()
            sld_y = self.server.gui.add_slider(
                "wall_y_max", 0.1, 1.5, 0.01, initial_value=wall[1][1],
            )
            sld_z = self.server.gui.add_multi_slider(
                "wall_z_minmax", min=0.0, max=2.0, step=0.01,
                initial_value=(wall[2][0], wall[2][1]),
            )

            @sld_z.on_update
            def _(_):
                w = robot.wall()
                robot.set_wall(
                    y_wall=w[1],
                    z_wall=[sld_z.value[0], sld_z.value[1]],
                    x_wall=w[0],
                )
                bounds = [item for sub in robot.wall() for item in sub]
                self.scene.update_wall(bounds)

            @sld_y.on_update
            def _(_):
                w = robot.wall()
                robot.set_wall(
                    y_wall=[-sld_y.value, sld_y.value],
                    x_wall=w[0],
                    z_wall=w[2],
                )
                bounds = [item for sub in robot.wall() for item in sub]
                self.scene.update_wall(bounds)

    def _build_torso_rail_panel(self, robot_name, robot):
        q_min, q_max = robot.get_limits()
        q = robot.q

        with self.server.gui.add_folder("Torso & Rail"):
            sld_rail = self.server.gui.add_slider(
                "Rail", float(q_min[0]), float(q_max[0]), 0.005,
                initial_value=float(q[0]),
            )
            sld_torso = self.server.gui.add_slider(
                "Torso", float(q_min[1]), float(q_max[1]), 0.01,
                initial_value=float(q[1]),
            )

            @sld_rail.on_update
            def _(_):
                robot.q[0] = sld_rail.value

            @sld_torso.on_update
            def _(_):
                robot.q[1] = sld_torso.value

    def _build_joint_print_panel(self, robot_name, robot):
        with self.server.gui.add_folder("Joints"):
            display = self.server.gui.add_markdown("_(click to read joints)_")
            btn = self.server.gui.add_button("Print Joints")

            @btn.on_click
            def _(_):
                q = np.asarray(robot.q, dtype=float)
                rail, torso = q[0], q[1]
                left = q[2:9]
                right = q[9:16]
                flat = ", ".join(f"{v:.4f}" for v in q)
                md = (
                    f"**q (16):** `[{flat}]`\n\n"
                    f"- rail: `{rail:.4f}`\n"
                    f"- torso: `{torso:.4f}`\n"
                    f"- left_arm: `[{', '.join(f'{v:.4f}' for v in left)}]`\n"
                    f"- right_arm: `[{', '.join(f'{v:.4f}' for v in right)}]`"
                )
                display.content = md
                print(f"[{robot_name}] q = [{flat}]")

    def _build_camera_panel(self, robot_name, camera_mgr: CameraManager):
        per_robot_handles: dict = {}
        placeholder = np.zeros((1, 1, 3), dtype=np.uint8)

        with self.server.gui.add_folder("Cameras"):
            frustum_toggle = self.server.gui.add_checkbox(
                "Show Frustums", initial_value=camera_mgr.cfg.show_frustums,
            )

            @frustum_toggle.on_update
            def _(_):
                camera_mgr.set_frustums_visible(frustum_toggle.value)

            for group_name, cam_names in CAMERA_GROUPS.items():
                with self.server.gui.add_folder(group_name):
                    for cam_name in cam_names:
                        if cam_name not in camera_mgr.camera_map:
                            continue
                        btn = self.server.gui.add_button(f"Render: {cam_name}")
                        img_handle = self.server.gui.add_image(
                            placeholder, label=cam_name, visible=False,
                        )
                        per_robot_handles[cam_name] = img_handle

                        def make_render_cb(name, handle):
                            @btn.on_click
                            def _(_):
                                for client in self.server.get_clients().values():
                                    img = camera_mgr.render_camera(client, name)
                                    if img is not None:
                                        handle.image = img
                                        handle.visible = True
                                    break

                        make_render_cb(cam_name, img_handle)

            btn_all = self.server.gui.add_button("Render All Cameras")

            @btn_all.on_click
            def _(_):
                for client in self.server.get_clients().values():
                    for cam in camera_mgr.cameras:
                        img = camera_mgr.render_camera(client, cam.name)
                        if img is not None:
                            handle = per_robot_handles.get(cam.name)
                            if handle is not None:
                                handle.image = img
                                handle.visible = True
                    break

        self.image_handles[robot_name] = per_robot_handles

    def _build_display_panel(self):
        with self.server.gui.add_folder("Display"):
            display_mode = self.server.gui.add_button_group(
                "View", ("visual", "collision", "both"),
            )

            @display_mode.on_click
            def _(event):
                mode = display_mode.value
                for name in self.robots.keys():
                    rv = self.scene.robot_visuals.get(name)
                    rc = self.scene.robot_collisions.get(name)
                    if rv is None or rc is None:
                        continue
                    if mode == "visual":
                        rv.show_visual = True
                        rc.show_visual = False
                    elif mode == "collision":
                        rv.show_visual = False
                        rc.show_visual = True
                    elif mode == "both":
                        rv.show_visual = True
                        rc.show_visual = True

    def _build_objects_panel(self):
        objects = self.scene.config.objects
        if not objects:
            return
        with self.server.gui.add_folder("Objects"):
            for obj in objects:
                with self.server.gui.add_folder(obj.name):
                    inp_x = self.server.gui.add_number(
                        "x", initial_value=obj.position[0], step=0.01,
                    )
                    inp_y = self.server.gui.add_number(
                        "y", initial_value=obj.position[1], step=0.01,
                    )
                    inp_z = self.server.gui.add_number(
                        "z", initial_value=obj.position[2], step=0.01,
                    )

                    def make_obj_cb(name, ix, iy, iz):
                        handle = self.scene.object_handles[name]

                        @ix.on_update
                        def _(_):
                            handle.position = np.array([ix.value, iy.value, iz.value])

                        @iy.on_update
                        def _(_):
                            handle.position = np.array([ix.value, iy.value, iz.value])

                        @iz.on_update
                        def _(_):
                            handle.position = np.array([ix.value, iy.value, iz.value])

                        def gizmo_to_gui(pos, wxyz):
                            ix.value = float(pos[0])
                            iy.value = float(pos[1])
                            iz.value = float(pos[2])

                        self.scene.register_object_update_cb(name, gizmo_to_gui)

                    make_obj_cb(obj.name, inp_x, inp_y, inp_z)

    def update_collision_status(self, robot_name: str, col: ColGroup):
        label = self.collision_labels.get(robot_name)
        if label is None:
            return
        if col == ColGroup.NONE:
            label.content = f"Collision: **NONE** ✓"
        else:
            label.content = f"Collision: **{col.name}** ✗"
