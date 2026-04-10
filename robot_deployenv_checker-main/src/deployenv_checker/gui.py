"""GUI panel builder: camera views, collision status, wall controls."""

import numpy as np
import viser
from typing import Dict
from hbmp import ColGroup, FrameEnum

from .camera import CameraManager, CAMERA_GROUPS


class GuiBuilder:
    """Builds the Viser GUI panels for the deploy environment checker."""

    def __init__(
        self,
        server: viser.ViserServer,
        robot_controller,
        camera_manager: CameraManager,
        scene_manager,
        app=None,
    ):
        self.server = server
        self.robot = robot_controller
        self.camera_mgr = camera_manager
        self.scene = scene_manager
        self.app = app
        self._build_panels()

    def _build_panels(self):
        self._build_reachability_panel()
        self._build_torso_rail_panel()
        self._build_camera_panel()
        self._build_display_panel()
        self._build_objects_panel()

    def _build_reachability_panel(self):
        with self.server.gui.add_folder("Reachability"):
            btn_track = self.server.gui.add_button("Go to Target")

            @btn_track.on_click
            def _(_):
                if self.app is not None:
                    self.app.move_to_eef_targets()

            self.collision_label = self.server.gui.add_markdown("Collision: **NONE**")

            wall = self.robot.wall()
            self.sld_y = self.server.gui.add_slider(
                "wall_y_max", 0.1, 1.5, 0.01, initial_value=wall[1][1],
            )
            self.sld_z = self.server.gui.add_multi_slider(
                "wall_z_minmax", min=0.0, max=2.0, step=0.01,
                initial_value=(wall[2][0], wall[2][1]),
            )

            @self.sld_z.on_update
            def _(_):
                w = self.robot.wall()
                self.robot.set_wall(
                    y_wall=w[1],
                    z_wall=[self.sld_z.value[0], self.sld_z.value[1]],
                    x_wall=w[0],
                )
                bounds = [item for sub in self.robot.wall() for item in sub]
                self.scene.update_wall(bounds)

            @self.sld_y.on_update
            def _(_):
                w = self.robot.wall()
                self.robot.set_wall(
                    y_wall=[-self.sld_y.value, self.sld_y.value],
                    x_wall=w[0],
                    z_wall=w[2],
                )
                bounds = [item for sub in self.robot.wall() for item in sub]
                self.scene.update_wall(bounds)

    def _build_torso_rail_panel(self):
        q_min, q_max = self.robot.get_limits()
        q = self.robot.q

        with self.server.gui.add_folder("Torso & Rail"):
            self.sld_rail = self.server.gui.add_slider(
                "Rail", float(q_min[0]), float(q_max[0]), 0.005,
                initial_value=float(q[0]),
            )
            self.sld_torso = self.server.gui.add_slider(
                "Torso", float(q_min[1]), float(q_max[1]), 0.01,
                initial_value=float(q[1]),
            )

            @self.sld_rail.on_update
            def _(_):
                self.robot.q[0] = self.sld_rail.value

            @self.sld_torso.on_update
            def _(_):
                self.robot.q[1] = self.sld_torso.value

    def _build_camera_panel(self):
        # Pre-create one image handle per camera (placeholder 1x1 black image)
        self.image_handles: dict = {}
        placeholder = np.zeros((1, 1, 3), dtype=np.uint8)

        with self.server.gui.add_folder("Cameras"):
            self.frustum_toggle = self.server.gui.add_checkbox(
                "Show Frustums", initial_value=self.camera_mgr.cfg.show_frustums,
            )

            @self.frustum_toggle.on_update
            def _(_):
                self.camera_mgr.set_frustums_visible(self.frustum_toggle.value)

            for group_name, cam_names in CAMERA_GROUPS.items():
                with self.server.gui.add_folder(group_name):
                    for cam_name in cam_names:
                        if cam_name not in self.camera_mgr.camera_map:
                            continue
                        btn = self.server.gui.add_button(f"Render: {cam_name}")
                        img_handle = self.server.gui.add_image(
                            placeholder, label=cam_name, visible=False,
                        )
                        self.image_handles[cam_name] = img_handle

                        def make_render_cb(name, handle):
                            @btn.on_click
                            def _(_):
                                for client in self.server.get_clients().values():
                                    img = self.camera_mgr.render_camera(client, name)
                                    if img is not None:
                                        handle.image = img
                                        handle.visible = True
                                    break

                        make_render_cb(cam_name, img_handle)

            btn_all = self.server.gui.add_button("Render All Cameras")

            @btn_all.on_click
            def _(_):
                for client in self.server.get_clients().values():
                    for cam in self.camera_mgr.cameras:
                        img = self.camera_mgr.render_camera(client, cam.name)
                        if img is not None:
                            handle = self.image_handles.get(cam.name)
                            if handle is not None:
                                handle.image = img
                                handle.visible = True
                    break

    def _build_display_panel(self):
        with self.server.gui.add_folder("Display"):
            self.display_mode = self.server.gui.add_button_group(
                "View", ("visual", "collision", "both"),
            )

            @self.display_mode.on_click
            def _(event):
                mode = self.display_mode.value
                if self.scene.robot_visual and self.scene.robot_collision:
                    if mode == "visual":
                        self.scene.robot_visual.show_visual = True
                        self.scene.robot_collision.show_visual = False
                    elif mode == "collision":
                        self.scene.robot_visual.show_visual = False
                        self.scene.robot_collision.show_visual = True
                    elif mode == "both":
                        self.scene.robot_visual.show_visual = True
                        self.scene.robot_collision.show_visual = True

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

                        # Sync GUI when gizmo is dragged in the viewport
                        def gizmo_to_gui(pos, wxyz):
                            ix.value = float(pos[0])
                            iy.value = float(pos[1])
                            iz.value = float(pos[2])

                        self.scene.register_object_update_cb(name, gizmo_to_gui)

                    make_obj_cb(obj.name, inp_x, inp_y, inp_z)

    def update_collision_status(self, col: ColGroup):
        if col == ColGroup.NONE:
            self.collision_label.content = "Collision: **NONE** ✓"
        else:
            self.collision_label.content = f"Collision: **{col.name}** ✗"
