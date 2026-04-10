from abc import ABC, abstractmethod
import numpy as np
from pywbc import Tf
from enum import Enum, auto, Flag
import numpy.typing as npt
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Optional, Dict, Union


class ColGroup(Flag):
    NONE = 0
    T_L = 1
    T_R = 2
    L_R = 3
    W_L = 4
    W_R = 5
    ALL_SELF = T_L | T_R | L_R
    ALL_W = W_L | W_R


class FrameEnum(Enum):
    FRAME_TACTILE_L = auto()
    FRAME_TACTILE_R = auto()
    FRAME_ELBOW_R = auto()
    FRAME_ELBOW_L = auto()
    FRAME_TORSO_2 = auto()


class Kin(ABC):
    DTYPE = np.float64

    @abstractmethod
    def update_kin(
        self, q: np.ndarray, pose_base: Tf = None, update_jacobian: bool = False
    ) -> None:
        pass

    @abstractmethod
    def get_fk(self, frame: FrameEnum) -> Tf:
        pass

    @abstractmethod
    def get_jacobian(self, frame: FrameEnum) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def get_qmap(self, frame: FrameEnum) -> npt.NDArray[np.int32]:
        pass

    @abstractmethod
    def get_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_ik(self, tf_target: Tf, frame: FrameEnum):
        pass


class Col(ABC):
    DTYPE = np.float32

    @abstractmethod
    def update_pose(self, rwt: np.ndarray) -> None:
        pass

    @abstractmethod
    def collision_free_self(self, collision_group_pair: ColGroup) -> ColGroup:
        pass

    @abstractmethod
    def collision_gradient_self(self, group: ColGroup) -> tuple:
        pass


class Traj(ABC):

    def __init__(self, waypoints: Union[List[np.ndarray], np.ndarray]):
        self.waypoints = np.array(waypoints)

        if len(self.waypoints) < 2:
            raise ValueError("A trajectory requires at least two waypoints.")

        self.dimensions = self.waypoints.shape[1]
        diffs = np.diff(self.waypoints, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        self.arc_lengths = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.arc_lengths[-1]

    @abstractmethod
    def evaluate(self, s: Union[float, np.ndarray]) -> np.ndarray:
        pass


class Mop(ABC):
    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        collision_fn: Callable[[np.ndarray], bool],
        step_size: float = 0.1,
        frozen_dofs: Optional[Dict[int, float]] = None,
    ):
        self.full_bounds = np.array(bounds)
        self.full_dim = len(bounds)
        self.collision_fn = collision_fn
        self.step_size = step_size
        self.frozen_dofs = frozen_dofs if frozen_dofs is not None else {}
        self.active_indices = [
            i for i in range(self.full_dim) if i not in self.frozen_dofs
        ]
        self.active_dim = len(self.active_indices)
        self.active_bounds = self.full_bounds[self.active_indices]

    def _get_full_state(self, active_state: np.ndarray) -> np.ndarray:
        full_state = np.zeros(self.full_dim)
        full_state[self.active_indices] = active_state
        for idx, val in self.frozen_dofs.items():
            full_state[idx] = val
        return full_state

    def _extract_active_state(self, full_state: np.ndarray) -> np.ndarray:
        return full_state[self.active_indices]

    def sample_active_state(self) -> np.ndarray:
        return np.random.uniform(self.active_bounds[:, 0], self.active_bounds[:, 1])

    def is_state_valid(self, active_state: np.ndarray) -> bool:
        if np.any(active_state < self.active_bounds[:, 0]) or np.any(
            active_state > self.active_bounds[:, 1]
        ):
            return False
        full_state = self._get_full_state(active_state)
        return not self.collision_fn(full_state)

    def is_path_valid(
        self, active_state1: np.ndarray, active_state2: np.ndarray
    ) -> bool:
        dist = np.linalg.norm(active_state2 - active_state1)
        if dist < 1e-3:
            return self.is_state_valid(active_state1)

        num_steps = int(np.ceil(dist / self.step_size))
        for i in range(num_steps + 1):
            interp_state = active_state1 + (active_state2 - active_state1) * (
                i / num_steps
            )
            if not self.is_state_valid(interp_state):
                return False
        return True

    @abstractmethod
    def plan(
        self, full_start: np.ndarray, full_goal: np.ndarray, max_iterations: int = 1000
    ) -> Optional[List[np.ndarray]]:
        pass

    @abstractmethod
    def shortcut(self, waypoints_feasible: np.ndarray) -> Optional[List[np.ndarray]]:
        pass


class RobotInterface(Col, Kin, ABC):
    @abstractmethod
    def get_arm_bound(self, which_hand: FrameEnum, which_bound: str = "z_min"):
        pass

    @abstractmethod
    def set_wall(
        self,
        x_wall: Union[List[float], np.ndarray] = [0.0, 1.5],
        y_wall: Union[List[float], np.ndarray] = [-1.0, 1.0],
        z_wall: Union[List[float], np.ndarray] = [0.6, 1.8],
    ):
        pass

    @property
    @abstractmethod
    def wall(self):
        pass

    @abstractmethod
    def update_col_self(self):
        pass

    @abstractmethod
    def check_self_collision(self, colgroup: ColGroup = ColGroup.ALL_SELF) -> ColGroup:
        pass

    @abstractmethod
    def track_tcp(
        self, hand: FrameEnum, tf_target: Tf, substeps: int = 3
    ) -> np.ndarray:
        pass
