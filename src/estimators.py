"""
Homography estimation utilities.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray


def compute_homography_from_points(
    points_a: NDArray[np.float32],
    points_b: NDArray[np.float32],
    ransac_reproj_thresh: float = 5,
    ransac_max_iter: int = 500,
) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.uint8]]]:
    """
    Compute homography between two point sets using OpenCV's findHomography.

    Args:
        points_a: points in first image (N, 2)
        points_b: matching points in second image (N, 2)
        ransac_reproj_thresh: RANSAC reprojection threshold
        ransac_max_iter: maximum RANSAC iterations

    Returns:
        H (3x3 float64) and status array (N, 1 uint8) or (None, None) on failure.
    """
    if points_a is None or points_b is None:
        return None, None
    if points_a.shape[0] < 4 or points_b.shape[0] < 4:
        return None, None

    H, status = cv2.findHomography(
        points_b, points_a, cv2.RANSAC, ransac_reproj_thresh, maxIters=ransac_max_iter
    )
    H_out = np.asarray(H, dtype=np.float64) if H is not None else None
    status_out = np.asarray(status, dtype=np.uint8) if status is not None else None

    return H_out, status_out
