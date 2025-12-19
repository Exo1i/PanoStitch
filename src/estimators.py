"""
Homography estimation utilities.

Manual implementation of:
- Direct Linear Transform (DLT) for homography computation
- RANSAC algorithm for robust estimation
"""

import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray


def normalize_points(
    points: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Normalize points for numerical stability in DLT.
    
    Applies a similarity transform so that the centroid is at origin
    and the average distance from origin is sqrt(2).
    
    Args:
        points: Input points (N, 2)
        
    Returns:
        Normalized points and the normalization matrix T
    """
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Shift to origin
    shifted = points - centroid
    
    # Compute average distance from origin
    avg_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    
    # Scale factor to make average distance sqrt(2)
    if avg_dist > 1e-10:
        scale = np.sqrt(2) / avg_dist
    else:
        scale = 1.0
    
    # Build normalization matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Apply normalization
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    points_h = np.hstack([points, ones])  # (N, 3) homogeneous
    normalized = (T @ points_h.T).T  # (N, 3)
    
    return normalized[:, :2], T


def dlt_homography(
    src_points: NDArray[np.float64], 
    dst_points: NDArray[np.float64]
) -> Optional[NDArray[np.float64]]:
    """
    Compute homography using Direct Linear Transform (DLT).
    
    Computes H such that dst = H @ src (in homogeneous coordinates).
    
    Args:
        src_points: Source points (N, 2)
        dst_points: Destination points (N, 2)
        
    Returns:
        3x3 homography matrix or None if computation fails
    """
    n = src_points.shape[0]
    if n < 4:
        return None
    
    # Normalize points for numerical stability
    src_norm, T_src = normalize_points(src_points)
    dst_norm, T_dst = normalize_points(dst_points)
    
    # Build the coefficient matrix A (2N x 9) - VECTORIZED
    # For each point correspondence (x,y) -> (x',y'), we have:
    # [-x, -y, -1, 0, 0, 0, x*x', y*x', x'] @ h = 0
    # [0, 0, 0, -x, -y, -1, x*y', y*y', y'] @ h = 0
    
    x = src_norm[:, 0]
    y = src_norm[:, 1]
    xp = dst_norm[:, 0]
    yp = dst_norm[:, 1]
    zeros = np.zeros(n, dtype=np.float64)
    ones = np.ones(n, dtype=np.float64)
    
    # Row 1: [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
    A_odd = np.column_stack([-x, -y, -ones, zeros, zeros, zeros, x*xp, y*xp, xp])
    # Row 2: [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
    A_even = np.column_stack([zeros, zeros, zeros, -x, -y, -ones, x*yp, y*yp, yp])
    
    # Interleave rows
    A = np.empty((2 * n, 9), dtype=np.float64)
    A[0::2] = A_odd
    A[1::2] = A_even

    
    # Solve using SVD: A @ h = 0
    try:
        _, s, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    
    # Check if the solution is reasonable (smallest singular value should be small)
    if s[-1] > 1e-2 * s[-2] and n == 4:
        # Degenerate configuration possible
        pass
    
    # Last row of Vt is the solution
    h = Vt[-1]
    H_normalized = h.reshape(3, 3)
    
    # Denormalize: H = T_dst^(-1) @ H_normalized @ T_src
    H = np.linalg.inv(T_dst) @ H_normalized @ T_src
    
    # Normalize so H[2,2] = 1
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    
    return H


def compute_reprojection_error(
    H: NDArray[np.float64],
    src_points: NDArray[np.float64],
    dst_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute reprojection error for each point pair.
    
    Projects src_points using H and computes distance to dst_points.
    H should satisfy: dst ≈ H @ src
    
    Args:
        H: 3x3 homography matrix
        src_points: Source points (N, 2)
        dst_points: Destination/target points (N, 2)
        
    Returns:
        Reprojection errors (N,)
    """
    n = src_points.shape[0]
    
    # Convert to homogeneous coordinates
    ones = np.ones((n, 1), dtype=np.float64)
    src_h = np.hstack([src_points, ones])  # (N, 3)
    
    # Project: H @ src
    projected = (H @ src_h.T).T  # (N, 3)
    
    # Convert back from homogeneous
    w = projected[:, 2:3]
    w = np.where(np.abs(w) < 1e-10, 1e-10, w)
    projected_euclidean = projected[:, :2] / w
    
    # Compute Euclidean distance
    errors = np.sqrt(np.sum((dst_points - projected_euclidean) ** 2, axis=1))
    
    return errors


def compute_symmetric_transfer_error(
    H: NDArray[np.float64],
    src_points: NDArray[np.float64],
    dst_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute symmetric transfer error (more robust).
    
    e = d(dst, H@src)^2 + d(src, H^{-1}@dst)^2
    
    Args:
        H: 3x3 homography matrix
        src_points: Source points (N, 2)
        dst_points: Destination points (N, 2)
        
    Returns:
        Symmetric errors (N,)
    """
    # Forward error
    forward_err = compute_reprojection_error(H, src_points, dst_points)
    
    # Backward error
    try:
        H_inv = np.linalg.inv(H)
        backward_err = compute_reprojection_error(H_inv, dst_points, src_points)
    except np.linalg.LinAlgError:
        backward_err = np.full(src_points.shape[0], np.inf)
    
    # Symmetric error (sum of squared errors, then sqrt for threshold comparison)
    return np.sqrt(forward_err**2 + backward_err**2)


def ransac_homography(
    src_points: NDArray[np.float32],
    dst_points: NDArray[np.float32],
    reproj_thresh: float = 5.0,
    max_iters: int = 500,
    confidence: float = 0.99,
) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.uint8]]]:
    """
    RANSAC algorithm for robust homography estimation.
    
    Finds H such that dst ≈ H @ src.
    
    Args:
        src_points: Source points (N, 2)
        dst_points: Destination points (N, 2)
        reproj_thresh: Inlier threshold (pixels)
        max_iters: Maximum number of RANSAC iterations
        confidence: Desired confidence level for early termination
        
    Returns:
        Best homography (3x3) and inlier mask (N,) or (None, None)
    """
    n = src_points.shape[0]
    if n < 4:
        return None, None
    
    src = src_points.astype(np.float64)
    dst = dst_points.astype(np.float64)
    
    best_H: Optional[NDArray[np.float64]] = None
    best_inliers: Optional[NDArray[np.bool_]] = None
    best_num_inliers = 0
    
    num_iters = max_iters
    rng = np.random.default_rng()
    
    for iteration in range(max_iters):
        if iteration >= num_iters:
            break
        
        # Randomly sample 4 point correspondences
        indices = rng.choice(n, size=4, replace=False)
        
        sample_src = src[indices]
        sample_dst = dst[indices]
        
        # Check for degenerate configuration (collinear points)
        def check_collinear(pts: NDArray[np.float64]) -> bool:
            """Check if 4 points are nearly collinear."""
            # Check all combinations of 3 points
            for i in range(4):
                p0, p1, p2 = pts[(i+0)%4], pts[(i+1)%4], pts[(i+2)%4]
                # Cross product of vectors
                v1 = p1 - p0
                v2 = p2 - p0
                cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
                # Area of triangle
                area = cross / 2
                if area < 10:  # Very small triangle
                    return True
            return False
        
        if check_collinear(sample_src) or check_collinear(sample_dst):
            continue
        
        # Compute homography from minimal sample
        H = dlt_homography(sample_src, sample_dst)
        if H is None:
            continue
        
        # Compute reprojection errors
        errors = compute_reprojection_error(H, src, dst)
        
        # Find inliers
        inliers = errors < reproj_thresh
        num_inliers = np.sum(inliers)
        
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers = inliers
            best_H = H
            
            # Update adaptive iteration count
            inlier_ratio = num_inliers / n
            if inlier_ratio > 0:
                p_all_outliers = (1 - inlier_ratio) ** 4
                if p_all_outliers < 1 - 1e-10:
                    num_iters = min(
                        max_iters,
                        int(np.ceil(np.log(1 - confidence) / np.log(p_all_outliers))) + 10
                    )
    
    if best_H is None or best_inliers is None:
        return None, None
    
    # Refine homography using all inliers (iterative refinement)
    for _ in range(3):  # A few refinement iterations
        if best_num_inliers >= 4:
            inlier_src = src[best_inliers]
            inlier_dst = dst[best_inliers]
            refined_H = dlt_homography(inlier_src, inlier_dst)
            if refined_H is not None:
                # Recompute inliers
                errors = compute_reprojection_error(refined_H, src, dst)
                new_inliers = errors < reproj_thresh
                new_num = np.sum(new_inliers)
                if new_num >= best_num_inliers:
                    best_H = refined_H
                    best_inliers = new_inliers
                    best_num_inliers = new_num
    
    # Final status array
    status = best_inliers.astype(np.uint8)
    
    return best_H, status


def compute_homography_from_points(
    points_a: NDArray[np.float32],
    points_b: NDArray[np.float32],
    ransac_reproj_thresh: float = 5,
    ransac_max_iter: int = 500,
) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.uint8]]]:
    """
    Compute homography between two point sets using manual DLT + RANSAC.
    
    Returns H such that points_a ≈ H @ points_b.

    Args:
        points_a: Points in first image (N, 2) - destination
        points_b: Matching points in second image (N, 2) - source
        ransac_reproj_thresh: RANSAC reprojection threshold
        ransac_max_iter: Maximum RANSAC iterations

    Returns:
        H (3x3 float64) and status array (N, uint8) or (None, None) on failure.
    """
    if points_a is None or points_b is None:
        return None, None
    if points_a.shape[0] < 4 or points_b.shape[0] < 4:
        return None, None

    # H maps points_b -> points_a, so src=points_b, dst=points_a
    H, status = ransac_homography(
        points_b, points_a, ransac_reproj_thresh, ransac_max_iter
    )

    return H, status
