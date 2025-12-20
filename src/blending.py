"""
Module 8: Blending
Handles image blending and final panorama creation.
"""
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from numpy.typing import NDArray
from .image import Image


def single_weights_array(size: int) -> NDArray[np.float64]:
    """Create a 1D weights array for blending."""
    if size % 2 == 1:
        a = np.linspace(0, 1, (size + 1) // 2, dtype=np.float64)
        b = np.linspace(1, 0, (size + 1) // 2, dtype=np.float64)[1:]
        return np.concatenate([a, b])
    else:
        a = np.linspace(0, 1, size // 2, dtype=np.float64)
        b = np.linspace(1, 0, size // 2, dtype=np.float64)
        return np.concatenate([a, b])


def single_weights_matrix(shape: Tuple[int, int]) -> NDArray[np.float64]:
    """Create a 2D weights matrix for blending."""
    return (
        single_weights_array(shape[0])[:, np.newaxis]
        @ single_weights_array(shape[1])[:, np.newaxis].T
    )


def get_new_corners(
    image: NDArray[np.uint8], H: NDArray[np.float64]
) -> List[NDArray[np.float64]]:
    """Get the corners of an image after applying a homography."""
    corners = np.array(
        [
            [0, 0, 1],
            [image.shape[1], 0, 1],
            [0, image.shape[0], 1],
            [image.shape[1], image.shape[0], 1],
        ],
        dtype=np.float64,
    ).T
    transformed = H @ corners
    transformed = transformed / transformed[2, :]
    return [transformed[:2, i : i + 1] for i in range(4)]


def get_offset(corners: List[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Get offset matrix so all corners are in positive coordinates."""
    top_left, top_right, bottom_left = corners[:3]
    return np.array(
        [
            [1, 0, max(0, -float(min(top_left[0], bottom_left[0])))],
            [0, 1, max(0, -float(min(top_left[1], top_right[1])))],
            [0, 0, 1],
        ],
        np.float64,
    )


def get_new_size(corners_images: List[List[NDArray[np.float64]]]) -> Tuple[int, int]:
    """Get the size of image that would contain all corners."""
    top_right_x = np.max([corners[1][0] for corners in corners_images])
    bottom_right_x = np.max([corners[3][0] for corners in corners_images])
    bottom_left_y = np.max([corners[2][1] for corners in corners_images])
    bottom_right_y = np.max([corners[3][1] for corners in corners_images])
    width = int(np.ceil(max(float(bottom_right_x), float(top_right_x))))
    height = int(np.ceil(max(float(bottom_right_y), float(bottom_left_y))))
    # Cap at conservative limits to prevent memory issues
    width = max(100, min(width, 5000))
    height = max(100, min(height, 4000))
    return width, height


def get_new_parameters(
    panorama: Optional[NDArray[np.uint8]],
    image: NDArray[np.uint8],
    H: NDArray[np.float64],
) -> Tuple[Tuple[int, int], NDArray[np.float64]]:
    """Module 6: Get the new size and offset matrix for warping."""
    corners = get_new_corners(image, H)
    if panorama is None:
        added_offset = get_offset(corners)
    else:
        corners_panorama = get_new_corners(panorama, np.eye(3, dtype=np.float64))
        all_corners = corners + corners_panorama
        xs = np.hstack([c[0] for c in all_corners]).astype(np.float64)
        ys = np.hstack([c[1] for c in all_corners]).astype(np.float64)
        min_x = float(np.min(xs))
        min_y = float(np.min(ys))
        tx = max(0, -min_x)
        ty = max(0, -min_y)
        added_offset = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    corners_image = get_new_corners(image, added_offset @ H)
    if panorama is None:
        size = get_new_size([corners_image])
    else:
        corners_panorama = get_new_corners(panorama, added_offset)
        size = get_new_size([corners_image, corners_panorama])
    logging.info(
        f"get_new_parameters: pano shape {None if panorama is None else panorama.shape[:2]}, "
        f"img shape {image.shape[:2]}, added_offset tx/ty: {added_offset[0,2]:.1f}/{added_offset[1,2]:.1f}, "
        f"computed size (h,w): {size}"
    )
    return size, added_offset


def add_image(
    panorama: Optional[NDArray[np.uint8]],
    image: Image,
    offset: NDArray[np.float64],
    weights: Optional[NDArray[np.float64]],
) -> Tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.float64]]:
    """
    Module 8: Add a new image to panorama with blending (legacy incremental method).
    """
    H = offset @ np.asarray(image.H, dtype=np.float64)
    logging.info(f"add_image: image={image.path}, H:\n{H}")
    size, added_offset = get_new_parameters(panorama, image.image, H)
    new_image = np.asarray(
        cv2.warpPerspective(image.image, added_offset @ H, size)
    ).astype(np.uint8)

    if panorama is None:
        panorama = np.zeros_like(new_image)
        weights = np.zeros(new_image.shape, dtype=np.float64)
    else:
        panorama = np.asarray(cv2.warpPerspective(panorama, added_offset, size)).astype(
            np.uint8
        )
        if weights is None:
            weights = np.zeros(panorama.shape, dtype=np.float64)
        else:
            weights = np.asarray(
                cv2.warpPerspective(weights, added_offset, size)
            ).astype(np.float64)

    image_weights = single_weights_matrix(
        (int(image.image.shape[0]), int(image.image.shape[1]))
    )
    image_weights = np.repeat(
        np.asarray(cv2.warpPerspective(image_weights, added_offset @ H, size))[
            :, :, np.newaxis
        ].astype(np.float64),
        3,
        axis=2,
    )

    normalized_weights = np.zeros_like(weights, dtype=np.float64)
    if weights is None:
        weights = np.zeros_like(image_weights, dtype=np.float64)
    normalized_weights = np.divide(
        weights, (weights + image_weights), where=(weights + image_weights) != 0
    )
    panorama = np.where(
        np.logical_and(
            np.repeat(np.sum(panorama, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
            np.repeat(np.sum(new_image, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
        ),
        0,
        new_image * (1 - normalized_weights) + panorama * normalized_weights,
    ).astype(np.uint8)

    new_weights = (weights + image_weights) / np.maximum(
        (weights + image_weights).max(), 1e-10
    )

    return panorama, (added_offset @ offset).astype(np.float64), new_weights


def simple_blending(images: List[Image]) -> NDArray[np.uint8]:
    """
    Memory-safe custom multi-band blending with fallback.
    Fixed: GaussianBlur now works on float32 (converted from float16).
    """
    if len(images) == 0:
        raise ValueError("No images to blend")
    if len(images) == 1:
        return images[0].image.copy()

    # Compute canvas (same as before)
    corners_images = [get_new_corners(img.image, img.H) for img in images]
    all_x = np.hstack([c[0] for corners in corners_images for c in corners])
    all_y = np.hstack([c[1] for corners in corners_images for c in corners])
    min_x, min_y = float(np.min(all_x)), float(np.min(all_y))
    max_x, max_y = float(np.max(all_x)), float(np.max(all_y))
    orig_width = int(np.ceil(max_x - min_x))
    orig_height = int(np.ceil(max_y - min_y))

    # Memory safety
    MAX_WIDTH, MAX_HEIGHT = 8000, 5000
    MAX_IMAGES_MULTI = 5

    use_multi_band = len(images) <= MAX_IMAGES_MULTI and orig_width <= 10000 and orig_height <= 6000

    if not use_multi_band:
        logging.info(f"Falling back to simple blending for {len(images)} images (memory-safe)")
        return _fallback_simple_blending(images, orig_width, orig_height, min_x, min_y)

    # Multi-band path
    scale_factor = 1.0
    width, height = orig_width, orig_height
    if width > MAX_WIDTH or height > MAX_HEIGHT:
        scale_factor = min(MAX_WIDTH / width, MAX_HEIGHT / height)
        width = int(width * scale_factor)
        height = int(height * scale_factor)
        logging.info(f"Downscaling multi-band by {scale_factor:.3f} → {width}x{height}")

    canvas_size = (width, height)
    scale_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]], np.float64)
    offset_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], np.float64)
    added_offset = scale_matrix @ offset_matrix

    num_levels = 5
    feather_amount = 81

    # Custom pyrDown/pyrUp - FIXED: convert to float32 for GaussianBlur
    def my_pyrDown(img: np.ndarray) -> np.ndarray:
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=0.85)
        return blurred[::2, ::2]

    def my_pyrUp(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        h, w = img.shape[:2]
        upsampled = np.zeros((h*2, w*2, img.shape[2]), dtype=np.float32)
        upsampled[::2, ::2] = img
        blurred = cv2.GaussianBlur(upsampled, (5, 5), sigmaX=0.85) * 4.0
        if (blurred.shape[0], blurred.shape[1]) != (target_h, target_w):
            blurred = cv2.resize(blurred, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return blurred.astype(np.float16)  # Back to float16 to save memory

    # Warp and weights in float16
    warped_images = []
    weight_maps = []
    for image in images:
        H = added_offset @ image.H
        warped = cv2.warpPerspective(image.image, H, canvas_size, flags=cv2.INTER_LINEAR).astype(np.float16)

        gray = cv2.cvtColor(image.image, cv2.COLOR_BGR2GRAY)
        mask = (gray > 5).astype(np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask, H, canvas_size)

        dist = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 5)
        dist = dist / (dist.max() + 1e-8)
        dist = cv2.GaussianBlur(dist, (feather_amount, feather_amount), 0)

        weights = np.repeat(dist[..., np.newaxis], 3, axis=2).astype(np.float16) + 1e-8

        warped_images.append(warped)
        weight_maps.append(weights)

    # Build pyramids
    def build_laplacian_pyramid(img: np.ndarray, levels: int):
        pyramid = []
        sizes = []
        current = img
        for _ in range(levels - 1):
            sizes.append((current.shape[0], current.shape[1]))
            down = my_pyrDown(current)
            up = my_pyrUp(down, current.shape[0], current.shape[1])
            lap = current - up
            pyramid.append(lap.astype(np.float16))
            current = down
        pyramid.append(current)
        return pyramid, sizes

    def build_gaussian_pyramid(wt: np.ndarray, levels: int):
        pyramid = []
        current = wt
        for _ in range(levels):
            pyramid.append(current)
            if len(pyramid) == levels:
                break
            current = my_pyrDown(current)
        return pyramid

    laplacian_pyramids = []
    weight_pyramids = []
    reconstruction_sizes = None

    for i, (img, wt) in enumerate(zip(warped_images, weight_maps)):
        lap_pyr, sizes = build_laplacian_pyramid(img, num_levels)
        laplacian_pyramids.append(lap_pyr)
        if i == 0:
            reconstruction_sizes = sizes
        
        wt_pyr = build_gaussian_pyramid(wt, num_levels)
        weight_pyramids.append(wt_pyr)

    # Blend levels
    blended_pyramid = []
    for level in range(num_levels):
        ref_shape = laplacian_pyramids[0][level].shape
        blended = np.zeros(ref_shape, dtype=np.float32)
        weight_sum = np.zeros(ref_shape, dtype=np.float32)

        for lap_pyr, wt_pyr in zip(laplacian_pyramids, weight_pyramids):
            blended += lap_pyr[level].astype(np.float32) * wt_pyr[level].astype(np.float32)
            weight_sum += wt_pyr[level].astype(np.float32)

        mask = weight_sum > 1e-6
        if np.any(mask):
            blended[mask] /= weight_sum[mask]

        blended_pyramid.append(blended)

    # Reconstruct
    result = blended_pyramid[-1].copy()
    for i in range(num_levels - 2, -1, -1):
        target_h, target_w = reconstruction_sizes[i]
        result = my_pyrUp(result, target_h, target_w) + blended_pyramid[i]

    if result.shape[:2] != (height, width):
        result = cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)

    panorama = np.clip(result, 0, 255).astype(np.uint8)
    logging.info("Multi-band blending complete.")
    return panorama

def _fallback_simple_blending(images: List[Image], orig_w: int, orig_h: int, min_x: float, min_y: float) -> NDArray[np.uint8]:
    """Simple distance-transform blending — very low memory, good quality."""
    # Strong downscale for fallback
    max_w, max_h = 6000, 3000
    scale = min(max_w / orig_w, max_h / orig_h)
    width = int(orig_w * scale)
    height = int(orig_h * scale)
    canvas_size = (width, height)

    scale_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float64)
    offset_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], np.float64)
    added_offset = scale_matrix @ offset_matrix

    panorama = np.zeros((height, width, 3), dtype=np.float64)
    weights = np.zeros((height, width, 3), dtype=np.float64)

    for img in images:
        H = added_offset @ img.H
        warped = cv2.warpPerspective(img.image, H, canvas_size)

        gray = cv2.cvtColor(img.image, cv2.COLOR_BGR2GRAY)
        mask = (gray > 10).astype(np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask, H, canvas_size)
        dist = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 5)
        if dist.max() > 0:
            dist /= dist.max()
        dist = cv2.GaussianBlur(dist, (101, 101), 0)
        img_weights = np.repeat(dist[..., np.newaxis], 3, axis=2) + 1e-8

        panorama += warped.astype(np.float64) * img_weights
        weights += img_weights

    mask = weights > 1e-6
    panorama[mask] /= weights[mask]
    return np.clip(panorama, 0, 255).astype(np.uint8)
