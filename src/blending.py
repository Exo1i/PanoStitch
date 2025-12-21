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


def single_weights_array(size: int) -> NDArray[np.float32]:
    """Create a 1D weights array for blending."""
    if size % 2 == 1:
        a = np.linspace(0, 1, (size + 1) // 2, dtype=np.float32)
        b = np.linspace(1, 0, (size + 1) // 2, dtype=np.float32)[1:]
        return np.concatenate([a, b])
    else:
        a = np.linspace(0, 1, size // 2, dtype=np.float32)
        b = np.linspace(1, 0, size // 2, dtype=np.float32)
        return np.concatenate([a, b])


def single_weights_matrix(shape: Tuple[int, int]) -> NDArray[np.float32]:
    """Create a 2D weights matrix for blending."""
    return (
        single_weights_array(shape[0])[:, np.newaxis]
        @ single_weights_array(shape[1])[:, np.newaxis].T
    ).astype(np.float32)


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
    ).astype(np.float32)


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

    # When a panorama already exists, compute an offset that keeps both
    # the transformed image and the current panorama within positive coordinates
    if panorama is None:
        added_offset = get_offset(corners)
    else:
        # Corners of the panorama in its current coordinate system
        corners_panorama = get_new_corners(panorama, np.eye(3, dtype=np.float64))
        # Combine corners of image and panorama to compute global minimums
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

    import logging

    # Debug info for new parameters
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
    weights: Optional[NDArray[np.float32]],
) -> Tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.float32]]:
    """
    Module 8: Add a new image to panorama with blending (memory-optimized).

    Args:
        panorama: Existing panorama
        image: Image to add
        offset: Offset already applied to panorama
        weights: Weights matrix of the panorama

    Returns:
        Updated panorama, offset, and weights
    """
    H = (offset @ np.asarray(image.H, dtype=np.float32)).astype(np.float32)
    logging.info(f"add_image: image={image.path}, H:\n{H}")
    size, added_offset = get_new_parameters(panorama, image.image, H)
    added_offset = added_offset.astype(np.float32)
    
    new_image = cv2.warpPerspective(image.image, added_offset @ H, size).astype(np.uint8)
    logging.info(
        f"add_image: new_image shape {new_image.shape}, min/max: {new_image.min()}/{new_image.max()}"
    )

    if panorama is None:
        panorama = np.zeros_like(new_image)
        weights = np.zeros((new_image.shape[0], new_image.shape[1]), dtype=np.float32)
    else:
        panorama = cv2.warpPerspective(panorama, added_offset, size).astype(np.uint8)
        # weights may be None; if so, initialize it
        if weights is None:
            weights = np.zeros((panorama.shape[0], panorama.shape[1]), dtype=np.float32)
        else:
            # Squeeze to 2D if needed, warp, then keep as 2D
            weights_2d = weights.squeeze() if weights.ndim == 3 else weights
            weights = cv2.warpPerspective(weights_2d, added_offset, size).astype(np.float32)

    # Compute weights efficiently without repeated expansions
    image_weights_2d = single_weights_matrix(
        (int(image.image.shape[0]), int(image.image.shape[1]))
    )
    image_weights = cv2.warpPerspective(image_weights_2d, added_offset @ H, size)
    image_weights = image_weights.astype(np.float32)
    
    logging.info(
        f"add_image: image_weights shape {image_weights.shape}, min/max: {image_weights.min()}/{image_weights.max()}"
    )

    # Blending using float32 to save memory
    panorama_float = panorama.astype(np.float32)
    new_image_float = new_image.astype(np.float32)
    
    # Expand weights to 3D for broadcasting
    weights_3d = np.expand_dims(weights, axis=2)
    image_weights_3d = np.expand_dims(image_weights, axis=2)
    
    denom = weights_3d + image_weights_3d
    safe_denom = np.where(denom > 0, denom, 1.0)
    
    # Efficient blending: only compute where both images exist or where new image exists
    blended = np.where(
        denom > 0,
        (panorama_float * weights_3d + new_image_float * image_weights_3d) / safe_denom,
        np.where(np.sum(new_image_float, axis=2, keepdims=True) > 0, new_image_float, 0)
    )

    panorama = np.clip(blended, 0, 255).astype(np.uint8)
    new_weights = denom / np.maximum(denom.max(), 1e-10)
    new_weights = new_weights.squeeze()  # Return as 2D
    
    logging.info(
        f"add_image: new_weights shape {new_weights.shape}, min/max: {new_weights.min()}/{new_weights.max()}"
    )
    logging.info(
        f"add_image: after blend panorama min/max: {panorama.min()}/{panorama.max()}"
    )
    return panorama, (added_offset @ offset).astype(np.float64), new_weights.astype(np.float32)


def simple_blending(images: List[Image]) -> NDArray[np.uint8]:
    """
    Module 8: Build a panorama using simple blending (memory-optimized).

    Args:
        images: Images to stitch

    Returns:
        Final panorama
    """
    # Compute global canvas size and offset by transforming all image corners
    corners_images = [get_new_corners(img.image, img.H) for img in images]

    # Gather all x and y coordinates
    all_x = np.hstack([c[0] for corners in corners_images for c in corners])
    all_y = np.hstack([c[1] for corners in corners_images for c in corners])

    min_x, min_y = float(np.min(all_x)), float(np.min(all_y))
    max_x, max_y = float(np.max(all_x)), float(np.max(all_y))

    width = int(np.ceil(max_x - min_x))
    height = int(np.ceil(max_y - min_y))

    logging.info(f"Required canvas size: {width}x{height}")

    # Cap at memory-safe limits
    max_width = 12000
    max_height = 12000
    scale_factor = 1.0

    if width > max_width or height > max_height:
        scale_factor = min(max_width / width, max_height / height)
        width = int(width * scale_factor)
        height = int(height * scale_factor)
        logging.info(f"Canvas exceeds limits, scaling by {scale_factor:.3f}")

    width = max(1, width)
    height = max(1, height)

    logging.info(f"Final canvas size: {width}x{height}")

    # Create scaling and offset transformation
    scale_matrix = np.array(
        [[scale_factor, 0.0, 0.0], [0.0, scale_factor, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    offset_matrix = np.array(
        [[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    added_offset = scale_matrix @ offset_matrix

    # Use float32 for panorama and weights to save memory (50% reduction)
    panorama_float = np.zeros((height, width, 3), dtype=np.float32)
    weights_sum = np.zeros((height, width, 1), dtype=np.float32)

    # Warp every image to the global canvas and composite
    for image in images:
        H = (added_offset @ image.H).astype(np.float32)
        size = (width, height)
        
        # Warp image and weights in one operation
        warped = cv2.warpPerspective(image.image, H, size, borderMode=cv2.BORDER_CONSTANT)
        
        # Compute weights efficiently
        image_weights_2d = single_weights_matrix(
            (int(image.image.shape[0]), int(image.image.shape[1]))
        )
        image_weights = cv2.warpPerspective(image_weights_2d, H, size, borderMode=cv2.BORDER_CONSTANT)
        
        # Convert warped to float32 for accurate blending (avoids intermediate uint8 conversion)
        warped_float = warped.astype(np.float32)
        
        # Expand weights to 3D and accumulate
        image_weights_3d = np.expand_dims(image_weights, axis=2)
        
        # Blend using weighted average formula
        denom = weights_sum + image_weights_3d
        # Avoid division by zero
        safe_denom = np.where(denom > 0, denom, 1.0)
        
        panorama_float = np.where(
            denom > 0,
            (panorama_float * weights_sum + warped_float * image_weights_3d) / safe_denom,
            panorama_float
        )
        
        weights_sum += image_weights_3d

    # Convert back to uint8 with clipping
    panorama = np.clip(panorama_float, 0, 255).astype(np.uint8)
    
    return panorama