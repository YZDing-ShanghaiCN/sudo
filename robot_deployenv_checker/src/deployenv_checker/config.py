"""Config dataclass models + YAML/JSON loader."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


# Camera groups by mount link — metadata about the camera_config.json layout.
# Used by the GUI to organize per-camera controls under mount-based folders.
CAMERA_GROUPS: Dict[str, List[str]] = {
    "Chest": ["chest_left_camera", "chest_right_camera"],
    "Left Hand": ["left_hand_left_camera", "left_hand_right_camera", "left_hand_center_camera"],
    "Right Hand": ["right_hand_left_camera", "right_hand_right_camera", "right_hand_center_camera"],
}


@dataclass
class ToolConfig:
    mesh_path: str
    mount_link: str
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    wxyz: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    scale: float = 1.0


@dataclass
class EEFConfig:
    name: str
    frame: str
    mesh_path: str
    position: List[float]
    wxyz: List[float]
    scale: float = 0.15
    tool: Optional["ToolConfig"] = None


@dataclass
class RobotParams:
    wbc_config: str = "wbc_config_hb.yaml"
    ndof: int = 16


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
class RobotConfig:
    name: str = "robot"
    type: str = "t2da2"
    urdf_visual: str = "./assets/hb11/urdf_c.urdf"
    urdf_collision: str = "./assets/hb11/urdf_c.urdf"
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    wxyz: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    scale: float = 0.25
    base_gizmo_scale: float = 0.8  # size of the draggable base gizmo (scene units)
    initial_q: List[float] = field(default_factory=list)
    end_effectors: List[EEFConfig] = field(default_factory=list)
    params: RobotParams = field(default_factory=RobotParams)
    cameras: Optional[CameraSystemConfig] = None


@dataclass
class ObjectConfig:
    name: str
    mesh_path: str
    collision_path: Optional[str] = None
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    wxyz: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    axis_size: float = 0.8
    draggable: bool = True


@dataclass
class WorkspaceConfig:
    bounds: List[float] = field(
        default_factory=lambda: [0.0, 1.25, -1.0, 1.0, 0.6, 1.8]
    )
    show_bounds: bool = True


@dataclass
class SceneConfig:
    name: str = "Workspace Check"
    robots: List[RobotConfig] = field(default_factory=list)
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


def _parse_robot(robot_data: dict, config_dir: Path, default_name: str) -> RobotConfig:
    """Parse a single robot YAML block into a RobotConfig with paths resolved."""
    robot_data = dict(robot_data)  # don't mutate caller's dict

    # EEFs
    eefs = []
    for eef in robot_data.pop("end_effectors", []):
        eef = dict(eef)
        eef["mesh_path"] = str(config_dir / eef["mesh_path"])
        tool_data = eef.pop("tool", None)
        if tool_data is not None:
            tool_data = dict(tool_data)
            tool_data["mesh_path"] = str(config_dir / tool_data["mesh_path"])
            eef["tool"] = ToolConfig(**tool_data)
        eefs.append(EEFConfig(**eef))

    # Params
    params_data = robot_data.pop("params", {}) or {}
    params = RobotParams(**params_data)
    if params.wbc_config:
        params.wbc_config = str(config_dir / params.wbc_config)

    # Cameras (per-robot)
    cameras_data = robot_data.pop("cameras", None)
    cameras_cfg: Optional[CameraSystemConfig] = None
    if cameras_data:
        cameras_data = dict(cameras_data)
        if "config_path" in cameras_data:
            cameras_data["config_path"] = str(config_dir / cameras_data["config_path"])
        cameras_cfg = CameraSystemConfig(**cameras_data)

    # URDF paths
    for key in ("urdf_visual", "urdf_collision"):
        if key in robot_data:
            robot_data[key] = str(config_dir / robot_data[key])

    robot_data.setdefault("name", default_name)

    return RobotConfig(
        **robot_data,
        end_effectors=eefs,
        params=params,
        cameras=cameras_cfg,
    )


def load_config(path: str) -> SceneConfig:
    config_dir = Path(path).parent

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    scene_data = data.get("scene", {}) or {}
    objects_data = data.get("objects", []) or []
    workspace_data = data.get("workspace", {}) or {}

    # New schema: `robots:` list. Legacy: single `robot:` + top-level `cameras:`.
    robots_data = data.get("robots")
    if robots_data is None:
        legacy_robot = data.get("robot")
        if legacy_robot is None:
            raise ValueError(
                f"Config {path} has no 'robots:' list and no legacy 'robot:' block."
            )
        legacy_robot = dict(legacy_robot)
        # Pull top-level cameras into the legacy robot so the unified parser handles it
        if "cameras" in data and "cameras" not in legacy_robot:
            legacy_robot["cameras"] = data["cameras"]
        robots_data = [legacy_robot]

    robots: List[RobotConfig] = []
    seen_names = set()
    for i, rd in enumerate(robots_data):
        robot = _parse_robot(rd, config_dir, default_name=f"robot_{i}")
        if robot.name in seen_names:
            raise ValueError(f"Duplicate robot name: {robot.name!r}")
        seen_names.add(robot.name)
        robots.append(robot)

    # Objects
    objs = []
    for obj_data in objects_data:
        obj_data = dict(obj_data)
        obj_data["mesh_path"] = str(config_dir / obj_data["mesh_path"])
        if obj_data.get("collision_path"):
            obj_data["collision_path"] = str(config_dir / obj_data["collision_path"])
        objs.append(ObjectConfig(**obj_data))

    workspace = WorkspaceConfig(**workspace_data)

    return SceneConfig(
        name=scene_data.get("name", "Workspace Check"),
        robots=robots,
        objects=objs,
        workspace=workspace,
    )
