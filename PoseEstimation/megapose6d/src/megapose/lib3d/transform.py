"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# Standard Library
from typing import Tuple, Union

# Third Party
import numpy as np
import pinocchio as pin
import torch


def _make_transform_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 SE(3) homogeneous matrix from R and t."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    assert R.shape == (3, 3), f"Expected R shape (3, 3), got {R.shape}"

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _quat_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to 3x3 rotation matrix without using pin.SE3."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)

    x, y, z, w = q
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero-norm quaternion is invalid.")

    x, y, z, w = q / norm

    R = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )

    return R


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to xyzw quaternion."""
    R = np.asarray(R, dtype=np.float64)

    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q_xyzw = np.array([x, y, z, w], dtype=np.float64)
    q_xyzw /= np.linalg.norm(q_xyzw)
    return q_xyzw


class Transform:
    """A representation of an SE(3) object based on a numpy 4x4 matrix.

    This version avoids pin.SE3 construction because some Pinocchio /
    Boost.Python / NumPy combinations fail to convert numpy arrays into
    Eigen matrices.
    """

    def __init__(
        self,
        *args: Union[
            Union[pin.SE3, np.ndarray, torch.Tensor],  # T
            Union[
                pin.Quaternion,
                np.ndarray,
                torch.Tensor,
                Tuple[float, float, float, float],
            ],  # rotation
            Union[np.ndarray, torch.Tensor, Tuple[float, float, float]],  # translation
        ]
    ):
        """
        - Transform(T): pin.SE3 or (4, 4) array
        - Transform(quaternion, translation), where:
            quaternion: pin.Quaternion, 4-array representing a xyzw quaternion,
                or a 3x3 rotation matrix
            translation: 3-array
        """
        if len(args) == 1:
            arg_T = args[0]

            if isinstance(arg_T, pin.SE3):
                self._T = np.asarray(arg_T.homogeneous, dtype=np.float64).copy()

            else:
                # Robust path: accept any array-like / numpy ndarray / torch tensor
                # as long as it represents a 4x4 homogeneous transform.
                if isinstance(arg_T, torch.Tensor):
                    T = arg_T.detach().cpu().numpy()
                else:
                    T = np.asarray(arg_T)

                if T.shape != (4, 4):
                    raise ValueError(
                        f"Expected Transform(T) with shape (4, 4), "
                        f"got type={type(arg_T)}, shape={getattr(T, 'shape', None)}"
                    )

                self._T = np.asarray(T, dtype=np.float64).copy()

        elif len(args) == 2:
            rotation, translation = args

            if isinstance(rotation, pin.Quaternion):
                R = np.asarray(rotation.matrix(), dtype=np.float64)

            elif isinstance(rotation, tuple):
                rotation_np = np.asarray(rotation, dtype=np.float64)

                if rotation_np.size == 4:
                    R = _quat_xyzw_to_rotmat(rotation_np)
                else:
                    raise ValueError(f"Unsupported tuple rotation size: {rotation_np.size}")

            elif isinstance(rotation, (np.ndarray, torch.Tensor)):
                if isinstance(rotation, torch.Tensor):
                    rotation_np = rotation.detach().cpu().numpy().copy()
                else:
                    rotation_np = np.asarray(rotation)

                if rotation_np.size == 4:
                    R = _quat_xyzw_to_rotmat(rotation_np)
                elif rotation_np.size == 9:
                    assert rotation_np.shape == (3, 3), (
                        f"Expected rotation matrix shape (3, 3), got {rotation_np.shape}"
                    )
                    R = np.asarray(rotation_np, dtype=np.float64)
                else:
                    raise ValueError(f"Unsupported rotation size: {rotation_np.size}")

            else:
                raise ValueError(f"Unsupported rotation type: {type(rotation)}")

            t = np.asarray(translation, dtype=np.float64)
            self._T = _make_transform_matrix(R, t)

        else:
            raise ValueError("Transform expects either 1 or 2 arguments.")

    def __mul__(self, other: "Transform") -> "Transform":
        T = self._T @ other._T
        return Transform(T)

    def inverse(self) -> "Transform":
        R = self._T[:3, :3]
        t = self._T[:3, 3]

        T_inv = np.eye(4, dtype=np.float64)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t

        return Transform(T_inv)

    def __str__(self) -> str:
        return str(self._T)

    def toHomogeneousMatrix(self) -> np.ndarray:
        return self._T.copy()

    @property
    def translation(self) -> np.ndarray:
        return self._T[:3, 3].copy()

    @property
    def quaternion(self) -> pin.Quaternion:
        q_xyzw = _rotmat_to_quat_xyzw(self._T[:3, :3])
        x, y, z, w = q_xyzw.tolist()
        q = pin.Quaternion(w, x, y, z)
        q.normalize()
        return q

    @property
    def rotation(self) -> np.ndarray:
        return self._T[:3, :3].copy()

    @property
    def matrix(self) -> np.ndarray:
        """Returns 4x4 homogeneous matrix representation."""
        return self._T.copy()