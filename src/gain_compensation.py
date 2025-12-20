"""
Module 7: Gain Compensation
Handles exposure/color balancing across images.
"""

import numpy as np
from typing import List
from numpy.typing import NDArray

from .image import Image
from .matching import PairMatch


def set_gain_compensations(
    images: List[Image],
    pair_matches: List[PairMatch],
    sigma_n: float = 10.0,
    sigma_g: float = 0.1,
) -> None:
    """
    Module 7: Compute gain compensation for each image to balance exposure.

    This solves a linear system to find optimal gain values that minimize
    intensity differences in overlapping regions.

    Args:
        images: Images of the panorama
        pair_matches: Pair matches between images
        sigma_n: Standard deviation of normalized intensity error
        sigma_g: Standard deviation of the gain
    """
    coefficients: List[List[NDArray[np.float64]]] = []
    results: List[NDArray[np.float64]] = []

    for k, image in enumerate(images):
        coefs_list: List[NDArray[np.float64]] = [
            np.zeros(3, dtype=np.float64) for _ in range(len(images))
        ]
        result = np.zeros(3, dtype=np.float64)

        for pair_match in pair_matches:
            if pair_match.area_overlap is None:
                continue

            if pair_match.image_a == image:
                coefs_list[k] += pair_match.area_overlap * (
                    (2 * pair_match.Iab**2 / sigma_n**2) + (1 / sigma_g**2)
                )
                i = images.index(pair_match.image_b)
                coefs_list[i] -= (
                    (2 / sigma_n**2)
                    * pair_match.area_overlap
                    * pair_match.Iab
                    * pair_match.Iba
                )
                result += pair_match.area_overlap / sigma_g**2
            elif pair_match.image_b == image:
                coefs_list[k] += pair_match.area_overlap * (
                    (2 * pair_match.Iba**2 / sigma_n**2) + (1 / sigma_g**2)
                )
                i = images.index(pair_match.image_a)
                coefs_list[i] -= (
                    (2 / sigma_n**2)
                    * pair_match.area_overlap
                    * pair_match.Iab
                    * pair_match.Iba
                )
                result += pair_match.area_overlap / sigma_g**2

        coefficients.append(coefs_list)
        results.append(result)

    coefficients_array = np.array(coefficients)
    results_array = np.array(results)
    gains = np.zeros_like(results_array)

    for channel in range(coefficients_array.shape[2]):
        # coefs is a 2D matrix for the current channel
        coefs_matrix: NDArray[np.float64] = coefficients_array[:, :, channel]
        res = results_array[:, channel]
        gains[:, channel] = np.linalg.solve(coefs_matrix, res)

    # Fix: Find max pixel value across all images properly
        median_gain = np.median(gains[gains > 0])  # Ignore any zero gains
        if median_gain > 0:
            gains /= median_gain

    for i, image in enumerate(images):
        image.gain = gains[i].astype(np.float32)
