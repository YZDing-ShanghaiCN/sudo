"""Config dataclass models + YAML/JSON loader."""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml


@dataclass
class EEFConfig:
    name: str
    frame: str
    mesh_path: str
    position: List[float]
    wxyz: List[float]
    scale: float = 0.15


@dataclass
class RobotParams:
    wbc_config: str = "wbc_config_hb.yaml"
    ndof: int = 16


@dataclass
class RobotConfig:
    type: str = "t2da2"
    urdf_visual: str = "./assets/hb11/urdf_c.urdf"
    urdf_collision: str = "./assets/hb11/urdf_c.urdf"
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    wxyz: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    scale: float = 0.25
    initial_q: List[float] = field(default_factory=list)
    end_effectors: List[EEFConfig] = field(default_factory=list)
    params: RobotParams = field(default_factory=RobotParams)


@dataclass
class CameraConfig:
    name: str
    mount: str
    extrinsics: np.ndarray  # 4x4
    intrinsics: np.ndarray  # 3x3
    width: int
    height: int

    @property
    def fx(self) -> float:
        return float(self.intrinsics[0, 0])

    @property
    def fy(self) -> float:
        return float(self.intrinsics[1, 1])

    @property
    def fov_y(self) -> float:
        return 2.0 * math.atan(self.height / (2.0 * self.fy))

    @property
    def fov_x(self) -> float:
        return 2.0 * math.atan(self.width / (2.0 * self.fx))


@dataclass
class CameraSystemConfig:
    config_path: str = "./configs/camera_config.json"
    show_frustums: bool = True
    frustum_scale: float = 0.1


@dataclass
class ObjectConfig:
    name: str
    mesh_path: str
    collision_path: Optional[str] = None
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    wxyz: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    scale: float = 1.0
    draggable: bool = True


@dataclass
class WorkspaceConfig:
    bounds: List[float] = field(
        default_factory=lambda: [0.0, 1.25, -1.0, 1.0, 0.6, 1.8]
    )
    show_bounds: bool = True


@dataclass
class SceneConfig:
    name: str = "HB11 Workspace Check"
    robot: RobotConfig = field(default_factory=RobotConfig)
    cameras: CameraSystemConfig = field(default_factory=CameraSystemConfig)
    objects: List[ObjectConfig] = field(default_factory=list)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)


def intrinsics_to_fov(K: np.ndarray, w: int, h: int) -> Tuple[float, float]:
    fx, fy = K[0, 0], K[1, 1]
    fov_x = 2.0 * math.atan(w / (2.0 * fx))
    fov_y = 2.0 * math.atan(h / (2.0 * fy))
    return fov_x, fov_y


def load_cameras(config_path: str) -> List[CameraConfig]:
    with open(config_path, "r") as f:
        data = json.load(f)
    cameras = []
    for cam in data["cameras"]:
        cameras.append(
            CameraConfig(
                name=cam["name"],
                mount=cam["mount"],
                extrinsics=np.array(cam["extrinsics"], dtype=np.float64),
                intrinsics=np.array(cam["intrinsics"], dtype=np.float64),
                width=cam["width"],
                height=cam["height"],
            )
        )
    return cameras


def load_config(path: str) -> SceneConfig:
    config_dir = Path(path).parent

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    scene_data = data.get("scene", {})
    robot_data = data.get("robot", {})
    cameras_data = data.get("cameras", {})
    objects_data = data.get("objects", [])
    workspace_data = data.get("workspace", {})

    # Parse EEFs
    eefs = []
    for eef in robot_data.pop("end_effectors", []):
        eef["mesh_path"] = str(config_dir / eef["mesh_path"])
        eefs.append(EEFConfig(**eef))

    # Parse robot params
    params_data = robot_data.pop("params", {})
    params = RobotParams(**params_data)

    # Resolve robot paths
    for key in ("urdf_visual", "urdf_collision"):
        if key in robot_data:
            robot_data[key] = str(config_dir / robot_data[key])
    if "wbc_config" in params.__dict__:
        params.wbc_config = str(config_dir / params.wbc_config)

    robot = RobotConfig(**robot_data, end_effectors=eefs, params=params)

    # Resolve camera config path
    if "config_path" in cameras_data:
        cameras_data["config_path"] = str(config_dir / cameras_data["config_path"])
    camera_system = CameraSystemConfig(**cameras_data)

    # Parse objects
    objs = []
    for obj_data in objects_data:
        obj_data["mesh_path"] = str(config_dir / obj_data["mesh_path"])
        if obj_data.get("collision_path"):
            obj_data["collision_path"] = str(config_dir / obj_data["collision_path"])
        objs.append(ObjectConfig(**obj_data))

    workspace = WorkspaceConfig(**workspace_data)

    return SceneConfig(
        name=scene_data.get("name", "HB11 Workspace Check"),
        robot=robot,
        cameras=camera_system,
        objects=objs,
        workspace=workspace,
    )
