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
    weights: Optional[NDArray[np.float64]],
) -> Tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.float64]]:
    """
    Module 8: Add a new image to panorama with blending.

    Args:
        panorama: Existing panorama
        image: Image to add
        offset: Offset already applied to panorama
        weights: Weights matrix of the panorama

    Returns:
        Updated panorama, offset, and weights
    """
    H = offset @ np.asarray(image.H, dtype=np.float64)
    import logging

    logging.info(f"add_image: image={image.path}, H:\n{H}")
    size, added_offset = get_new_parameters(panorama, image.image, H)
    new_image = np.asarray(
        cv2.warpPerspective(image.image, added_offset @ H, size)
    ).astype(np.uint8)
    logging.info(
        f"add_image: new_image shape {new_image.shape}, min/max: {new_image.min()}/{new_image.max()}"
    )

    if panorama is None:
        panorama = np.zeros_like(new_image)
        weights = np.zeros(new_image.shape, dtype=np.float64)
    else:
        panorama = np.asarray(cv2.warpPerspective(panorama, added_offset, size)).astype(
            np.uint8
        )
        # weights may be None; if so, initialize it
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
    logging.info(
        f"add_image: image_weights shape {image_weights.shape}, min/max: {image_weights.min()}/{image_weights.max()}"
    )

    normalized_weights = np.zeros_like(weights, dtype=np.float64)
    # weights and image_weights are float arrays; ensure no None
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
    logging.info(
        f"add_image: new_weights shape {new_weights.shape}, min/max: {new_weights.min()}/{new_weights.max()}"
    )

    logging.info(
        f"add_image: after blend panorama min/max: {panorama.min()}/{panorama.max()}"
    )
    return panorama, (added_offset @ offset).astype(np.float64), new_weights


def simple_blending(images: List[Image]) -> NDArray[np.uint8]:
    """
    Module 8: Build a panorama using simple blending.

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
        dtype=np.float64,
    )
    offset_matrix = np.array(
        [[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    added_offset = scale_matrix @ offset_matrix

    panorama = np.zeros((height, width, 3), dtype=np.uint8)
    weights = np.zeros((height, width, 3), dtype=np.float64)

    # Warp every image to the global canvas and composite
    for image in images:
        H = added_offset @ image.H
        size = (width, height)
        warped = np.asarray(cv2.warpPerspective(image.image, H, size)).astype(np.uint8)

        image_weights = single_weights_matrix(
            (int(image.image.shape[0]), int(image.image.shape[1]))
        )
        image_weights = np.repeat(
            np.asarray(cv2.warpPerspective(image_weights, H, size))[
                :, :, np.newaxis
            ].astype(np.float64),
            3,
            axis=2,
        )

        # Normalize and blend
        denom = weights + image_weights
        normalized_weights = np.divide(weights, denom, where=denom != 0)
        # convert panorama to float for accurate blending, then cast back
        panorama_float = panorama.astype(np.float64)
        blended = np.where(
            np.logical_and(
                np.repeat(np.sum(panorama, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
                np.repeat(np.sum(warped, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
            ),
            0,
            warped.astype(np.float64) * (1 - normalized_weights)
            + panorama_float * normalized_weights,
        )

        panorama[:, :, :] = np.clip(blended, 0, 255).astype(np.uint8)

        weights = (weights + image_weights) / np.maximum(
            (weights + image_weights).max(), 1e-10
        )

    return panorama
