import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf
import trimesh
from typing import Union, Tuple
import ampl
from .common_util import colorize


class ViserObject:
    def __init__(
        self,
        mesh: Union[
            str,
            trimesh.Trimesh,
            yourdfpy.URDF,
            np.ndarray,
            list[np.ndarray],
        ],
        name: str,
        server: viser.ViserServer,
        **kwargs,
    ):
        self._kwargs = kwargs

        if ("no_control" not in kwargs):
        
            if "control_scale" in kwargs:
                self.control = server.scene.add_transform_controls(
                    f"/{name}", kwargs["control_scale"]
                )
            else:
                self.control = server.scene.add_transform_controls(f"/{name}", 0.2)


        if isinstance(mesh, str):
            self.mesh = trimesh.load_mesh(mesh, process=False)
            if "color" in kwargs:
                color = kwargs["color"]
                self.mesh.visual.face_colors = list(color)
                self.mesh.visual = self.mesh.visual.to_texture()
                self.mesh.visual.material.alphaMode = "BLEND"
            mn = f"/{name}/mesh"
            if "affix" in kwargs:
                mn+="_"+kwargs["affix"]
            #print(mn)
            self.handler = server.scene.add_mesh_trimesh(mn, self.mesh)


        elif isinstance(mesh, trimesh.Trimesh):
            self.mesh = mesh
            if "color" in kwargs:
                color = kwargs["color"]
                self.mesh.visual.face_colors = list(color)
                self.mesh.visual = self.mesh.visual.to_texture()
                self.mesh.visual.material.alphaMode = "BLEND"
            else:
                color = [128, 128, 128]

            if len(color) == 4:
                opacity = color[3] / 255.0
            else:
                opacity = 1.0

            #    self.handler = server.scene.add_mesh_trimesh("/" + name, self.mesh)
            # print(color)
            self.handler = server.scene.add_mesh_simple(
                "/" + name,
                self.mesh.vertices,
                self.mesh.faces,
                color=(color[0], color[1], color[2]),
                flat_shading=True,
                opacity=opacity,
                wireframe=kwargs["wireframe"] if "wireframe" in kwargs else False,
            )

            # dd = server.scene.add_mesh_simple()
            # dd.

        elif isinstance(mesh, yourdfpy.URDF):
            self.handler = server.scene.add_frame(f"/{name}", show_axes=False)
            self.urdf = ViserUrdf(server, mesh, root_node_name=f"/{name}/arm")
        elif isinstance(mesh, np.ndarray):
            self.pcd = np.array(mesh).astype(np.float16)
            self.pcd_color = colorize(mesh)
            self.handler = server.scene.add_point_cloud(
                f"/{name}",
                points=self.pcd,
                colors=self.pcd_color,
                point_size=1e-2,
            )
            if "point_size" in kwargs:
                self.handler.point_size = kwargs["point_size"]
            # self.handler.visible = True
        elif isinstance(mesh, list) and all(
            isinstance(value, np.ndarray) for value in mesh
        ):
            self.pcd = np.array(mesh[0]).astype(np.float16)
            self.pcd_color = mesh[1].copy()
            self.handler = server.scene.add_point_cloud(
                f"/{name}", self.pcd, self.pcd_color, point_size=1e-2
            )
            if "point_size" in kwargs:
                self.handler.point_size = kwargs["point_size"]
        else:
            self.handler = self.control

    # def __init__(self, trimesh: trimesh.Trimesh, name: str, scene: viser.SceneApi):
    #     self.mesh = trimesh
    #     self.handler = scene.add_mesh_trimesh(name, self.mesh)
    #     return
    def disable_control(self):
        self.control.disable_rotations = True
        self.control.disable_sliders = True
        self.control.disable_axes = True

    def enable_control(self):
        self.control.disable_rotations = False
        self.control.disable_sliders = False
        self.control.disable_axes = False

    def update_pose(self):
        self.handler.wxyz = self.control.wxyz
        self.handler.position = self.control.position

    def set_control(self, rwt: Union[np.ndarray, viser.TransformControlsHandle]):
        if isinstance(rwt, np.ndarray):
            if len(rwt.shape) == 1:
                self.control.wxyz = rwt[[3, 0, 1, 2]]
                self.control.position = rwt[-3:]

            else:
                rwt_tmp = ampl.tf44_to_qt7(rwt)
                self.control.wxyz = rwt_tmp[[3, 0, 1, 2]]
                self.control.position = rwt_tmp[-3:]
        if isinstance(rwt, viser.TransformControlsHandle):
            self.control.wxyz = rwt.wxyz
            self.control.position = rwt.position
        #self.update_pose()

    def pose_tuple(self):
        return (self.control.wxyz, self.control.position)

    def pose_rwt(self):
        return np.hstack([self.control.wxyz[[1, 2, 3, 0]], self.control.position])


    