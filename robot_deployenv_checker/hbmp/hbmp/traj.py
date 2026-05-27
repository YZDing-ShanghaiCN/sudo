import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union

from .base import Traj


class TrajLinear(Traj):
    """
    Connects waypoints with straight lines. Good for dense paths or simple
    kinematics where sharp velocity changes at waypoints are acceptable.
    """

    def __init__(self, waypoints):
        super().__init__(waypoints)

    def evaluate(self, s: Union[float, np.ndarray]) -> np.ndarray:
        s_array = np.atleast_1d(s)

        # 2. Clip 's' to [0, total_length] to prevent out-of-bounds errors
        s_clipped = np.clip(s_array, 0.0, self.total_length)

        # 3. Find which segment each 's' belongs to.
        # np.searchsorted finds the index where 's' should be inserted to maintain order.
        idx = np.searchsorted(self.arc_lengths, s_clipped, side="right")

        # Bound the index to valid segments (1 to len-1)
        idx = np.clip(idx, 1, len(self.waypoints) - 1)

        # 4. Extract the start and end distances for those segments
        s_start = self.arc_lengths[idx - 1]
        s_end = self.arc_lengths[idx]

        # 5. Extract the start and end multidimensional waypoints
        p_start = self.waypoints[idx - 1]
        p_end = self.waypoints[idx]

        # 6. Calculate the interpolation factor 't' (0.0 to 1.0)
        segment_lengths = s_end - s_start

        # Safe division: if a segment length is 0 (duplicate waypoints), t = 0
        t = np.zeros_like(s_clipped)
        valid = segment_lengths > 0
        t[valid] = (s_clipped[valid] - s_start[valid]) / segment_lengths[valid]

        # Reshape 't' so it can broadcast against the D-dimensional waypoints
        # Shape changes from (N,) to (N, 1)
        t = t[:, np.newaxis]

        # 7. Execute the linear interpolation formula
        interpolated_points = p_start + t * (p_end - p_start)

        # 8. Return a 1D array if the user passed a single float, else return 2D array
        if np.isscalar(s) or np.ndim(s) == 0:
            return interpolated_points[0]

        return interpolated_points
