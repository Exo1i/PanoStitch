"""
PanoStitch - Main Stitcher Class
Coordinates the entire panorama stitching pipeline.
"""

import logging
import numpy as np
from typing import List, Optional
from numpy.typing import NDArray

from .image import Image
from .matching import MultiImageMatches
from .homography import find_connected_components, build_homographies
from .gain_compensation import set_gain_compensations
from .blending import add_image, simple_blending


class PanoStitch:
    """Main panorama stitching class that coordinates all modules."""

    def __init__(
        self,
        resize_size: Optional[int] = None,
        ratio: float = 0.75,
        gain_sigma_n: float = 10.0,
        gain_sigma_g: float = 0.1,
        use_harris: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize PanoStitch.

        Args:
            resize_size: Maximum dimension to resize images (None to keep original)
            ratio: Lowe's ratio test threshold
            gain_sigma_n: Sigma_n for gain compensation
            gain_sigma_g: Sigma_g for gain compensation
            use_harris: Whether to use Harris corner detection (True) or SIFT (False)
            verbose: Whether to print progress
        """
        self.resize_size = resize_size
        self.ratio = ratio
        self.gain_sigma_n = gain_sigma_n
        self.gain_sigma_g = gain_sigma_g
        self.use_harris = use_harris
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.INFO, format="%(message)s")

    def stitch(self, image_paths: List[str]) -> List[NDArray[np.uint8]]:
        """
        Main stitching pipeline that executes all 8 modules.

        Pipeline:
        1. Load images
        2. Compute features (Module 2 & 3: Harris + Descriptors)
        3. Match images (Module 4: Feature Matching)
        4. Find connected components
        5. Build homographies (Module 5: DLT + RANSAC)
        6. Compute gain compensations (Module 7)
        7. Blend images (Module 8)

        Args:
            image_paths: List of paths to images

        Returns:
            List of panoramas (one per connected component)
        """
        # Module 1: Load images
        logging.info(f"Loading {len(image_paths)} images...")
        images = [Image(path, self.resize_size) for path in image_paths]

        # Module 2 & 3: Compute features
        detector_name = "Harris" if self.use_harris else "SIFT"
        logging.info(f"Computing features using {detector_name}...")
        for image in images:
            image.compute_features(use_harris=self.use_harris)

        # Module 4: Match images
        logging.info("Matching images...")
        matcher = MultiImageMatches(images, ratio=self.ratio)
        pair_matches = matcher.get_pair_matches()
        pair_matches.sort(key=lambda pm: len(pm.matches), reverse=True)

        logging.info(f"Found {len(pair_matches)} valid pair matches")

        # Find connected components
        logging.info("Finding connected components...")
        connected_components = find_connected_components(pair_matches)
        logging.info(f"Found {len(connected_components)} connected component(s)")

        # Module 5 & 6: Build homographies
        logging.info("Building homographies...")
        build_homographies(connected_components, pair_matches)

        # Debug: print H matrices per image
        for image in [img for comp in connected_components for img in comp]:
            logging.info(f"H for {image.path}:\n{image.H}")

        # Module 7: Compute gain compensations
        logging.info("Computing gain compensations...")
        for connected_component in connected_components:
            component_matches = [
                pm for pm in pair_matches if pm.image_a in connected_component
            ]
            set_gain_compensations(
                connected_component,
                component_matches,
                sigma_n=self.gain_sigma_n,
                sigma_g=self.gain_sigma_g,
            )

        # Apply gain
        for image in images:
            image.image = (image.image * image.gain[np.newaxis, np.newaxis, :]).astype(
                np.uint8
            )

        # Debug: print each image gain summary
        for image in images:
            logging.info(
                f"Image: {image.path}, gain: {image.gain}, min/max after gain: {image.image.min()}/{image.image.max()}"
            )

        # Module 8: Blend images
        logging.info("Blending images...")
        results = []
        # for component in connected_components:
        #     panorama = None
        #     offset = np.eye(3, dtype=np.float64)
        #     weights = None
            
        #     for image in component:
        #         panorama, offset, weights = add_image(
        #             panorama, image, offset, weights
        #         )
            
        #     results.append(panorama)
            
        results = [simple_blending(component) for component in connected_components]


        logging.info("✓ Stitching complete!")
        return results
