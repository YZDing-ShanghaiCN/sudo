import json
import viser
import numpy as np
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, asdict
from dacite import from_dict, Config

import trimesh
from viser.extras import ViserUrdf
from yourdfpy import URDF

# class ViserURDFInterface(ViserUrdf):


@dataclass
class ObjectHandles:
    root: viser.TransformControlsHandle  # The Gizmo/Parent
    visual: viser.SceneNodeHandle  # The pretty mesh
    collision: viser.SceneNodeHandle  # The convex hull/collision mesh


@dataclass
class CameraState:
    # Type hints say np.ndarray, but __init__ accepts lists (from JSON)
    position: np.ndarray
    wxyz: np.ndarray
    fov: float
    look_at: Optional[np.ndarray]

    def __post_init__(self):
        """
        Automatically converts lists to numpy arrays after initialization.
        """
        self.position = np.array(self.position, dtype=np.float64)
        self.wxyz = np.array(self.wxyz, dtype=np.float64)
        if self.look_at is not None:
            self.look_at = np.array(self.look_at, dtype=np.float64)
        # FOV is a float, so we don't need to convert it,
        # but we could force it: self.fov = float(self.fov)

    def to_dict(self):
        """
        Converts arrays back to lists for JSON saving.
        """
        return {
            "position": self.position.tolist(),
            "wxyz": self.wxyz.tolist(),
            "fov": self.fov,
            "look_at": self.look_at.tolist(),
        }


@dataclass
class SceneItem:
    name: str
    asset_path: str
    collision_path: Optional[str]
    task_path: Optional[str]
    # We type hint as np.ndarray, but accept List in init
    position: np.ndarray
    wxyz: np.ndarray
    scale: float = 1.0
    type: str = "object"

    def __post_init__(self):
        """
        This runs automatically after __init__.
        It converts lists (from JSON) into numpy arrays.
        """
        self.position = np.array(self.position, dtype=np.float64).copy()
        self.wxyz = np.array(self.wxyz, dtype=np.float64).copy()
        # print("self.wyxz", self.wxyz)

    def to_dict(self):
        """
        Helper to convert arrays BACK to lists for JSON saving.
        """
        return {
            "name": self.name,
            "asset_path": self.asset_path,
            "collision_path": self.collision_path,
            "task_path": self.task_path,
            "position": self.position.tolist(),  # Convert to list
            "wxyz": self.wxyz.tolist(),  # Convert to list
            "scale": self.scale,
            "type": self.type,
        }


class ConfigSceneManager:
    def __init__(self, server: viser.ViserServer):
        self.server = server
        # Maps object name -> Viser SceneNodeHandle
        self.active_handles: Dict[str, Any] = {}
        # Maps object name -> SceneItem (data)
        self.items_data: Dict[str, SceneItem] = {}
        self.saved_camera_states: Dict[str, Dict[str, Any]] = {}
        self.desired_camera_state: Optional[CameraState] = None
        self.objects: Dict[str, ObjectHandles] = {}
        self.current_display_mode: Literal["visual", "collision", "both"] = "visual"
        self.click_to_focus_enabled = True
        self.current_menu_handle = None
        # https://www.roboticplus.com/index/img/laoding.svg
        self.add_logo()

        self.current_task: Dict[str, Any] = {}

    def add_logo(self):
        self.server.gui.add_html(
            f"""
    <div style="position: fixed; top: 10px; left: 10px; z-index: 1000;">
        <img src=https://upload.wikimedia.org/wikipedia/commons/e/e1/GitLab_logo.svg height="60" style="border-radius: 5px;" />

    </div>
    """
        )

    def set_display_mode(self, mode: Literal["visual", "collision", "both"]):
        """Toggles visibility of visual vs collision meshes globally."""
        self.current_display_mode = mode

        # print(f"Switching view to: {mode}")
        # print(self.objects.keys())
        for handles in self.objects.values():
            if hasattr(handles.visual, "show_visual"):

                if mode == "visual":
                    handles.visual.show_visual = True
                    handles.collision.show_visual = False

                elif mode == "collision":
                    handles.visual.show_visual = False
                    for m in handles.collision._meshes:
                        m.opacity = 1
                    handles.collision.show_visual = True

                elif mode == "both":
                    handles.visual.show_visual = True
                    handles.collision.show_visual = True
                    for m in handles.collision._meshes:
                        m.opacity = 0.75
                        # m.wireframe = True

            else:

                if mode == "visual":
                    handles.visual.visible = True
                    handles.collision.visible = False

                elif mode == "collision":
                    handles.visual.visible = False
                    handles.collision.visible = True
                    handles.collision.opacity = 1.0
                elif mode == "both":
                    handles.visual.visible = True
                    handles.collision.visible = True
                    # Optional: make visual transparent in 'both' mode?
                    handles.collision.opacity = 0.75

    def set_orthographic(self, client: viser.ClientHandle, enable: bool):
        """
        Toggles between Perspective and Simulated Orthographic view for a specific client.
        """
        client_id = client.client_id

        if enable:
            # === SAVE STATE ===
            # We save the current state before messing with it
            self.saved_camera_states[client_id] = {
                "position": np.array(client.camera.position),
                "fov": client.camera.fov,
                "wxyz": np.array(client.camera.wxyz),
                "look_at": np.array(client.camera.look_at),
            }

            # === CALCULATE ORTHO ===
            # 1. Get current position
            current_pos = np.array(client.camera.position)

            # 2. Define our "Ortho Factor" (Higher = flatter view)
            # 100x distance usually looks perfectly flat.
            FACTOR = 2.0

            # 3. Move camera 100x further away from the origin
            # (Assumes user is looking roughly at the scene center)
            new_pos = current_pos * FACTOR

            # 4. Zoom in (narrow FOV) by the same factor
            # Standard FOV is ~0.8 rad. We reduce it to ~0.008.
            new_fov = client.camera.fov / 2.0

            # 5. Apply
            client.camera.position = new_pos
            client.camera.fov = new_fov

            print(f"Client {client_id}: Switched to Orthographic.")

        else:
            # === RESTORE PERSPECTIVE ===
            state = self.saved_camera_states.get(client_id)
            if state:
                client.camera.position = state["position"]
                client.camera.fov = state["fov"]
                client.camera.wxyz = state["wxyz"]
                client.camera.look_at = state["look_at"]
                print(f"Client {client_id}: Restored Perspective.")
            else:
                print("No saved state to restore!")

    def get_object_pose(self, name: str):
        if name in self.active_handles:

            return np.hstack(
                [
                    self.active_handles[name].wxyz[[1, 2, 3, 0]],
                    self.active_handles[name].position,
                ]
            )
        else:
            return None

    def set_object_state(self, name: str, state: np.ndarray):
        if name not in self.active_handles:
            # print(f"Error: Object '{name}' not found.")
            return
        self.objects[name].visual.update_cfg(state)
        if self.objects[name].collision:
            self.objects[name].collision.update_cfg(state)

    def set_object_pose(self, name: str, pose: np.ndarray):
        """
        Updates the pose of an object using a 4x4 Numpy Transformation Matrix.

        Args:
            name: The name of the object in the scene.
            T_world_obj: A 4x4 numpy array representing [R | t].
        """
        if name not in self.active_handles:
            # print(f"Error: Object '{name}' not found.")
            return

        # 1. Decompose the 4x4 Matrix
        # T = [[R, t], [0, 1]]
        if pose.shape == (4, 4):
            R = pose[:3, :3]
            t = pose[:3, 3]
            from viser.transforms import SO3

            wxyz = SO3.from_matrix(R).wxyz
            position = t

        # --- CASE 2: 7D Vector [qw, qx, qy, qz, tx, ty, tz] ---
        elif pose.shape == (7,) or pose.size == 7:
            pose = pose.flatten()  # Ensure it's 1D
            wxyz = pose[[3, 0, 1, 2]]  # First 4 are Quaternion
            position = pose[4:7]  # Last 3 are Translation

        else:
            print(f"Error: Invalid pose shape {pose.shape}. Expected (4,4) or (7,)")
            return

        # 3. Update the Viser Handle (Visuals)
        handle = self.active_handles[name]
        handle.position = position
        handle.wxyz = wxyz

        # --- UPDATE INTERNAL STATE (For Saving) ---
        if name in self.items_data:
            self.items_data[name].position = position
            self.items_data[name].wxyz = wxyz

    def focus_on_object(self, name: str):
        """Sets the camera rotation center to the object's position."""
        if name not in self.active_handles:
            print(f"Object {name} not found.")
            return

        # Get object position
        # Note: If you use TransformControls, the handle is the control itself
        target_pos = self.active_handles[name].position

        # Apply to all clients
        for client in self.server.get_clients().values():
            # Animate the 'look_at' point to the object
            client.camera.look_at = target_pos
            print(f"Camera pivot set to {name} at {target_pos}")

    def save_scene(self, filepath: str):
        """Saves current scene state to JSON."""
        output_data = {}

        # 1. SAVE OBJECTS
        obj_list = []
        for name, handle in self.active_handles.items():
            original = self.items_data.get(name)
            if original:
                # Update our data object with live values from Viser
                print(name, handle.position)
                original.position = handle.position
                original.wxyz = handle.wxyz
                obj_list.append(original.to_dict())

        output_data["objects"] = obj_list

        # 2. SAVE CAMERA (from the first client)
        clients = self.server.get_clients()
        if clients:
            client = next(iter(clients.values()))

            # Create CameraState from live client data
            cam_state = CameraState(
                position=np.array(client.camera.position),
                wxyz=np.array(client.camera.wxyz),
                fov=client.camera.fov,
                look_at=client.camera.look_at,
            )
            output_data["camera"] = cam_state.to_dict()
            print("Camera saved.")

        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Saved to {filepath}")

    def load_scene(self, filepath: str):
        """Loads scene from JSON."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print("File not found.")
            return

        # 1. LOAD OBJECTS
        self.clear_scene()
        for obj_dict in data.get("objects", []):
            # The __post_init__ inside SceneItem converts lists -> np.array

            item = from_dict(
                data_class=SceneItem,
                data=obj_dict,
                config=Config(type_hooks={np.ndarray: np.array}),
            )  #
            # SceneItem(**obj_dict)
            # print(item.position)
            self.items_data[item.name] = item
            self._spawn_object(item)

        # 2. LOAD CAMERA
        if "camera" in data:
            cam_dict = data["camera"]
            # The __post_init__ inside CameraState converts lists -> np.array
            # state = CameraState(**cam_dict)
            state = from_dict(
                data_class=CameraState,
                data=cam_dict,
                config=Config(type_hooks={np.ndarray: np.array}),
            )  #

            self.desired_camera_state = state

            # Apply to all currently connected clients
            for client in self.server.get_clients().values():
                self.apply_camera(client)

    def apply_camera(self, client: viser.ClientHandle):
        """Applies the stored camera state to a specific client."""
        if self.desired_camera_state:
            # We can pass these directly because they are now Numpy arrays
            client.camera.position = self.desired_camera_state.position
            client.camera.wxyz = self.desired_camera_state.wxyz
            client.camera.fov = self.desired_camera_state.fov
            client.camera.look_at = self.desired_camera_state.look_at

    def _spawn_object(self, item: SceneItem):
        """Helper to actually put the object in Viser."""

        # 1. We create a "Transform Control" first.
        # This gives us the Gizmo (arrows) to move the object.
        # We name it matching the object so we can find it later.
        if item.type == "object":
            controls = self.server.scene.add_transform_controls(
                name=item.name, position=item.position, wxyz=item.wxyz, scale=item.scale
            )
        elif item.type == "robot":
            controls = self.server.scene.add_transform_controls(
                name="/" + item.name,
                position=item.position,
                wxyz=item.wxyz,
                scale=item.scale,
            )

        # print(item.name)
        # print(item.wxyz, item.position)
        # 2. We attach the actual 3D asset as a CHILD of the controls.
        # When the controls move, the asset moves.
        # NOTE: You can replace add_glb with add_mesh_simple or others depending on file type.
        mesh = None
        mesh_col = None
        if item.type == "object":
            try:
                # print("asdfasdf")
                # Attempt to load GLB/GLTF
                trimesh_viz = trimesh.load(item.asset_path)
                mesh = self.server.scene.add_mesh_trimesh(
                    f"{item.name}/mesh", trimesh_viz
                )

            except Exception:
                # Fallback for demo if file doesn't exist: Create a Cube
                # print(f"Asset {item.asset_path} not found, using placeholder.")
                # mesh_viz = trimesh.load(item.asset_path)
                mesh = self.server.scene.add_mesh_simple(
                    name=f"{item.name}/mesh",
                    vertices=np.array(
                        [
                            [-1, -1, -1],
                            [1, -1, -1],
                            [1, 1, -1],
                            [-1, 1, -1],
                            [-1, -1, 1],
                            [1, -1, 1],
                            [1, 1, 1],
                            [-1, 1, 1],
                        ]
                    )
                    * 0.2,
                    faces=np.array(
                        [
                            [0, 1, 2],
                            [0, 2, 3],
                            [4, 5, 6],
                            [4, 6, 7],
                            [0, 4, 7],
                            [0, 7, 3],
                            [1, 5, 6],
                            [1, 6, 2],
                            [0, 1, 5],
                            [0, 5, 4],
                            [3, 2, 6],
                            [3, 6, 7],
                        ]
                    ),
                    color=(100, 200, 255),
                )
            try:
                trimesh_col = trimesh.load(
                    item.collision_path, process=False, force="mesh"
                )

            except Exception:
                print(f"Collsiion {item.asset_path} not found, using convexhull.")
                trimesh_col = trimesh.convex.convex_hull(
                    trimesh_viz.dump(concatenate=True).vertices
                )
            mesh_col = self.server.scene.add_mesh_simple(
                f"{item.name}/collision",
                trimesh_col.vertices,
                trimesh_col.faces,
                # opacity=1.0,  # <--- CRITICAL: 1.0 fixes transparency artifacts
                # wireframe=False,  # <--- Explicitly ensure faces are rendered,
                flat_shading=True,  # Tells Viser to not smooth normals (crisper edges)
                # opacity=0.99999,  # <--- INVISIBLE
                # visible=True,
                color=np.random.randint(0, 256, size=3),
            )

            # mesh_col.visible=False
            # mesh. .interaction_mode = "click
            # 3. Store the CONTROLS handle, because that is what holds the position data.
            @mesh_col.on_click
            def _(event: viser.SceneNodePointerEvent):
                # Check if our "mode" is enabled
                if self.click_to_focus_enabled:
                    # pos = self.items_data[item.name].position + np.array([0, 0, 0.5])

                    # Create a floating "Delete" button
                    self.open_menu(event.client, item.name)
                    client = event.client

                    # Get the position of the parent controls (where the object actually is)
                    # We use controls.position because that updates when you drag the gizmo.
                    target_pos = controls.position

                    # Animate the camera look_at to this position
                    client.camera.look_at = target_pos

                    print(f"Clicked {item.name}. Camera pivot set to: {target_pos}")

            if self.current_display_mode == "visual":
                mesh_col.visible = False
            elif self.current_display_mode == "collision":
                mesh.visible = False

        elif item.type == "robot":
            try:
                # print("asdfasdf")
                # Attempt to load GLB/GLTF
                # print(item.name)
                mesh = ViserUrdf(
                    self.server,
                    URDF.load(item.asset_path),
                    root_node_name=f"/{item.name}/mesh",
                )
                # print("asdfsdf")
            except Exception:
                print(f"Robot {item.asset_path} cannot be loaded.")
            try:
                # print("asdfasdf")
                # Attempt to load GLB/GLTF
                mesh_col = ViserUrdf(
                    self.server,
                    URDF.load(item.collision_path),
                    root_node_name=f"/{item.name}/collision",
                )
                for mh in mesh_col._meshes:

                    @ (mh).on_click
                    def _(event: viser.SceneNodePointerEvent):
                        # Check if our "mode" is enabled
                        if self.click_to_focus_enabled:
                            # pos = self.items_data[item.name].position + np.array([0, 0, 0.5])

                            # Create a floating "Delete" button
                            self.open_menu(event.client, item.name)
                            client = event.client

                            # Get the position of the parent controls (where the object actually is)
                            # We use controls.position because that updates when you drag the gizmo.
                            target_pos = controls.position

                            # Animate the camera look_at to this position
                            client.camera.look_at = target_pos

                            print(
                                f"Clicked {item.name}. Camera pivot set to: {target_pos}"
                            )

            except Exception:
                print(f"Robot {item.collision_path} cannot be loaded.")

                # mesh_col.v
            if self.current_display_mode == "visual":
                if mesh_col:
                    mesh_col.show_visual = False
                # mesh_col.visible = False
            elif self.current_display_mode == "collision":
                mesh.show_visual = False
        self.objects[item.name] = ObjectHandles(controls, mesh, mesh_col)

        self.active_handles[item.name] = controls

    def clear_scene(self):
        # print(self.active_handles.keys())
        for name, handle in self.active_handles.items():
            self.server.scene.remove_by_name(name)

        self.active_handles.clear()
        # print(self.active_handles.keys())
        self.items_data.clear()
        import time

        time.sleep(0.05)

    def toggle_gizmos(self, name: str, status: bool):
        self.active_handles[name].disable_rotations = not status
        self.active_handles[name].disable_sliders = not status
        self.active_handles[name].disable_axes = not status

    def open_menu(self, client: viser.ClientHandle, name: str):
        # nonlocal self.cur

        # If a menu is already open, close it first to avoid clutter
        if self.current_menu_handle is not None:
            self.current_menu_handle.remove()
            self.current_menu_handle = None

        # Create a new folder for this object
        # We use client.gui (not server.gui) so it's private to this user
        menu = client.gui.add_folder(f"Object in Focus: {name}")
        self.current_menu_handle = menu

        with menu:
            # Add controls inside the folder
            btn_color = client.gui.add_button("Toggle Gizmo")
            btn_task = client.gui.add_button("Load Task")

            @btn_task.on_click
            def _(_):
                if hasattr(self.items_data[name], "task_path"):
                    # print()
                    try:
                        from waypointpath import Trajectory
                        # print(self.items_data[name].task_path)
                        traj = Trajectory.load(self.items_data[name].task_path)

                        # traj = np.loadtxt(self.items_data[name].task_path)
                        # print(traj.points)
                        self.server.scene.add_spline_catmull_rom(
                            f"{name}/task",
                            points=traj.points,
                            curve_type="chordal",
                            segments=len(traj.points),
                            tension=0.1,
                            line_width=3,
                            color=(255, 0, 0),
                        )
                        with open(self.items_data[name].task_path, "r") as file:
                            # data = json.load(file)
                            self.current_task[name] = json.load(file)
                        # print(self.current_task)
                    except Exception:
                        1

            @btn_color.on_click
            def _(_):
                self.active_handles[name].disable_rotations = not self.active_handles[
                    name
                ].disable_rotations
                self.active_handles[name].disable_sliders = not self.active_handles[
                    name
                ].disable_sliders
                self.active_handles[name].disable_axes = not self.active_handles[
                    name
                ].disable_axes

                # np.random.randint(0, 255, size=3)

            # btn_delete = client.gui.add_button("Delete Object", color="red")
            # @btn_delete.on_click
            # def _(_):
            #     cube.remove()
            #     menu.remove() # Close menu after deleting

            btn_close = client.gui.add_button("Close Menu")

            @btn_close.on_click
            def _(_):
                menu.remove()

    def add_bounding_box(
        self,
        name: str,
        bounds: tuple[float, float, float, float, float, float],
        color: tuple[int, int, int] = (255, 100, 100),
    ):
        """
        Draws a clean wireframe bounding box from min/max coordinates.
        bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        # 1. Define the 8 corners of the box
        corners = np.array(
            [
                [xmin, ymin, zmin],  # 0: Bottom-left-front
                [xmax, ymin, zmin],  # 1: Bottom-right-front
                [xmax, ymax, zmin],  # 2: Bottom-right-back
                [xmin, ymax, zmin],  # 3: Bottom-left-back
                [xmin, ymin, zmax],  # 4: Top-left-front
                [xmax, ymin, zmax],  # 5: Top-right-front
                [xmax, ymax, zmax],  # 6: Top-right-back
                [xmin, ymax, zmax],  # 7: Top-left-back
            ]
        )

        # 2. Define the 12 edges by connecting corner indices
        edge_indices = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # 4 Bottom edges
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # 4 Top edges
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # 4 Vertical pillars
        ]

        # 3. Create a (12, 2, 3) array containing the start and end points of each line
        edges = corners[edge_indices]

        # 4. Add the clean lines to the scene
        return self.server.scene.add_line_segments(
            name=name, points=edges, line_width=2.0, colors=color
        )
