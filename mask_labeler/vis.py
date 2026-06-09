"""Mask overlay helper. Extracted from RoboCamCalib robocamcalib/utils/vis_utils.py:155-169."""
from __future__ import annotations

import numpy as np


def overlay_mask(
    img: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Blend `mask` over `img` with the given color and alpha."""
    assert img.shape[:2] == mask.shape[:2], "image and mask size mismatch"
    mask = mask.astype(np.uint8)
    vis = img.copy().astype(np.float32)
    idx = np.nonzero(mask)
    vis[idx[0], idx[1], :] *= 1.0 - alpha
    vis[idx[0], idx[1], :] += [alpha * c for c in color]
    return vis.astype(np.uint8)
