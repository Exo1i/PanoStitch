"""
Module 5 & 6: Homography Estimation and Panorama Assembly
Handles connected components, homography building, and panorama warping.
"""

import numpy as np
from typing import List
from numpy.typing import NDArray

from .image import Image
from .matching import PairMatch


def find_connected_components(pair_matches: List[PairMatch]) -> List[List[Image]]:
    """
    Find connected components (separate panoramas) in the image set.

    Args:
        pair_matches: List of pair matches

    Returns:
        List of connected components (each is a list of Images)
    """
    connected_components: List[List[Image]] = []
    pair_matches_to_check = pair_matches.copy()
    component_id = 0

    while len(pair_matches_to_check) > 0:
        pair_match = pair_matches_to_check.pop(0)
        connected_component = {pair_match.image_a, pair_match.image_b}
        size = len(connected_component)
        stable = False

        while not stable:
            i = 0
            while i < len(pair_matches_to_check):
                pair_match = pair_matches_to_check[i]
                if (
                    pair_match.image_a in connected_component
                    or pair_match.image_b in connected_component
                ):
                    connected_component.add(pair_match.image_a)
                    connected_component.add(pair_match.image_b)
                    pair_matches_to_check.pop(i)
                else:
                    i += 1

            stable = size == len(connected_component)
            size = len(connected_component)

        connected_components.append(list(connected_component))
        for image in connected_component:
            image.component_id = component_id
        component_id += 1

    return connected_components


def build_homographies(
    connected_components: List[List[Image]], pair_matches: List[PairMatch]
) -> None:
    """
    Module 6: Build homographies for each image in each connected component.

    This establishes the transformation from each image to a common reference frame.

    Args:
        connected_components: The connected components
        pair_matches: The valid pair matches
    """
    for connected_component in connected_components:
        component_matches = [
            pm for pm in pair_matches if pm.image_a in connected_component
        ]

        images_added = set()

        # Choose the most connected image as reference (central image)
        # This minimizes the maximum chain length and reduces error accumulation
        connection_counts = {}
        for img in connected_component:
            connection_counts[img] = sum(
                1 for pm in component_matches if pm.contains(img)
            )

        # Select image with most connections as reference
        reference_image = max(connection_counts.items(), key=lambda x: x[1])[0]
        reference_image.H = np.eye(3, dtype=np.float64)
        images_added.add(reference_image)

        # Incrementally add remaining images
        while len(images_added) < len(connected_component):
            for pair_match in component_matches:
                if (
                    pair_match.image_a in images_added
                    and pair_match.image_b not in images_added
                ):
                    pair_match.compute_homography()
                    if pair_match.H is not None:
                        pair_match.image_b.H = np.asarray(
                            pair_match.image_a.H
                            @ np.asarray(pair_match.H, dtype=np.float64),
                            dtype=np.float64,
                        )
                        images_added.add(pair_match.image_b)
                    break
                if (
                    pair_match.image_a not in images_added
                    and pair_match.image_b in images_added
                ):
                    pair_match.compute_homography()
                    if pair_match.H is not None:
                        pair_match.image_a.H = np.asarray(
                            pair_match.image_b.H
                            @ np.linalg.inv(np.asarray(pair_match.H, dtype=np.float64)),
                            dtype=np.float64,
                        )
                        images_added.add(pair_match.image_a)
                    break
