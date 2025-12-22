"""
Module 4: Feature Matching
Handles matching features between pairs of images.
"""

import cv2
import numpy as np
from typing import List
from numpy.typing import NDArray

from .image import Image
from .estimators import compute_homography_from_points


class PairMatch:
    """Represents a match between two images."""

    def __init__(
        self, image_a: Image, image_b: Image, matches: List[cv2.DMatch]
    ) -> None:
        """
        Create a new PairMatch object.

        Args:
            image_a: First image of the pair
            image_b: Second image of the pair
            matches: List of matches between image_a and image_b
        """
        self.image_a = image_a
        self.image_b = image_b
        self.matches = matches
        self.H: NDArray[np.float64] | None = None
        self.status: NDArray[np.uint8] | None = None
        self.overlap: NDArray[np.uint8] | None = None
        self.area_overlap: int | None = None
        self._Iab: NDArray[np.float64] | None = None
        self._Iba: NDArray[np.float64] | None = None
        self.matchpoints_a: NDArray[np.float32] | None = None
        self.matchpoints_b: NDArray[np.float32] | None = None

    def compute_homography(
        self, ransac_reproj_thresh: float = 5, ransac_max_iter: int = 500
    ) -> None:
        """
        Module 5: Compute homography between the two images.

        TODO: Replace with custom implementation:
        - Implement Direct Linear Transform (DLT)
        - Implement RANSAC algorithm

        Current: Uses OpenCV's findHomography

        Args:
            ransac_reproj_thresh: reprojection threshold for RANSAC
            ransac_max_iter: maximum iterations for RANSAC
        """
        if self.image_a.keypoints is None or self.image_b.keypoints is None:
            return

        # Create numpy arrays of match points
        self.matchpoints_a = np.array(
            [self.image_a.keypoints[match.queryIdx].pt for match in self.matches],
            dtype=np.float32,
        )
        self.matchpoints_b = np.array(
            [self.image_b.keypoints[match.trainIdx].pt for match in self.matches],
            dtype=np.float32,
        )

        # Use the homography utility (module 5) to compute H and status
        H, status = compute_homography_from_points(
            self.matchpoints_a,
            self.matchpoints_b,
            ransac_reproj_thresh,
            ransac_max_iter,
        )
        self.H = H
        self.status = status

    def set_overlap(self) -> None:
        """Compute and set the overlap region between the two images."""
        if self.H is None:
            self.compute_homography()

        if self.H is None:
            return

        mask_a = np.ones_like(self.image_a.image[:, :, 0], dtype=np.uint8)
        mask_b = cv2.warpPerspective(
            np.ones_like(self.image_b.image[:, :, 0], dtype=np.uint8),
            self.H,
            mask_a.shape[::-1],
        )
        self.overlap = np.asarray(mask_a * mask_b, dtype=np.uint8)
        self.area_overlap = int(self.overlap.sum())

    def is_valid(self, alpha: float = 8, beta: float = 0.3) -> bool:
        """
        Check if the pair match is valid.

        Args:
            alpha: alpha parameter for comparison
            beta: beta parameter for comparison

        Returns:
            True if valid, False otherwise
        """
        if self.overlap is None:
            self.set_overlap()

        if self.status is None:
            self.compute_homography()

        if self.overlap is None or self.status is None or self.matchpoints_a is None:
            return False

        matches_in_overlap = self.matchpoints_a[
            self.overlap[
                self.matchpoints_a[:, 1].astype(np.int64),
                self.matchpoints_a[:, 0].astype(np.int64),
            ]
            == 1
        ]

        return int(self.status.sum()) > alpha + beta * matches_in_overlap.shape[0]

    def contains(self, image: Image) -> bool:
        """Check if the given image is in this pair match."""
        return self.image_a == image or self.image_b == image

    @property
    def Iab(self) -> NDArray[np.float64]:
        if self._Iab is None:
            self.set_intensities()
        assert self._Iab is not None
        return self._Iab

    @property
    def Iba(self) -> NDArray[np.float64]:
        if self._Iba is None:
            self.set_intensities()
        assert self._Iba is not None
        return self._Iba

    def set_intensities(self) -> None:
        """Compute intensities in overlap region for gain compensation."""
        if self.overlap is None:
            self.set_overlap()

        if self.H is None or self.overlap is None:
            return

        inverse_overlap = cv2.warpPerspective(
            self.overlap, np.linalg.inv(self.H), self.image_b.image.shape[1::-1]
        )

        overlap_sum = self.overlap.sum()
        inverse_overlap_sum = inverse_overlap.sum()

        if overlap_sum == 0:
            self._Iab = np.zeros(3, dtype=np.float64)
            self._Iba = np.zeros(3, dtype=np.float64)
            return

        self._Iab = (
            np.sum(
                self.image_a.image
                * np.repeat(self.overlap[:, :, np.newaxis], 3, axis=2),
                axis=(0, 1),
            )
            / overlap_sum
        )
        self._Iba = (
            np.sum(
                self.image_b.image
                * np.repeat(inverse_overlap[:, :, np.newaxis], 3, axis=2),
                axis=(0, 1),
            )
            / inverse_overlap_sum
        )


class MultiImageMatches:
    """
    Module 4: Handles feature matching between multiple images.

    TODO: Replace compute_matches() with custom implementation:
    - Implement brute-force descriptor matching
    - Compute pairwise Euclidean distances
    - Apply k-NN search (k=2)
    - Apply Lowe's ratio test
    """

    def __init__(self, images: List[Image], ratio: float = 0.75, use_dnn: bool = False) -> None:
        """
        Create a new MultiImageMatches object.

        Args:
            images: images to compare
            ratio: ratio for Lowe's ratio test
            use_dnn: use DISK+LightGlue deep learning matcher
        """
        self.images = images
        self.matches: dict[str, dict[str, List[cv2.DMatch]]] = {
            image.path: {} for image in images
        }
        self.ratio = ratio
        self.use_dnn = use_dnn
        self.deep_matcher = None
        
        if use_dnn:
            from deep_matcher import DeepMatcher
            self.deep_matcher = DeepMatcher(method="disk+lightglue")

    def get_matches(self, image_a: Image, image_b: Image) -> List[cv2.DMatch]:
        """Get matches for the given images."""
        if image_b.path not in self.matches[image_a.path]:
            matches = self.compute_matches(image_a, image_b)
            self.matches[image_a.path][image_b.path] = matches
        return self.matches[image_a.path][image_b.path]

    def get_pair_matches(self, max_images: int = 6) -> List[PairMatch]:
        """
        Get valid pair matches for all images.

        Args:
            max_images: Maximum number of matches per image

        Returns:
            List of valid PairMatch objects
        """
        pair_matches: List[PairMatch] = []
        for i, image_a in enumerate(self.images):
            possible_matches = sorted(
                self.images[:i] + self.images[i + 1 :],
                key=lambda image: len(self.get_matches(image_a, image)),
                reverse=True,
            )[:max_images]

            for image_b in possible_matches:
                if self.images.index(image_b) > i:
                    pair_match = PairMatch(
                        image_a, image_b, self.get_matches(image_a, image_b)
                    )
                    if pair_match.is_valid():
                        pair_matches.append(pair_match)

        return pair_matches

    def compute_matches(self, image_a: Image, image_b: Image) -> List[cv2.DMatch]:
        """
        Module 4: Compute matches between two images.

        Args:
            image_a: First image
            image_b: Second image

        Returns:
            List of valid matches
        """
        if self.use_dnn and self.deep_matcher is not None:
            return self._compute_dnn_matches(image_a, image_b)
        else:
            return self._compute_sift_matches(image_a, image_b)

    def _compute_dnn_matches(self, image_a: Image, image_b: Image) -> List[cv2.DMatch]:
        """Compute matches using DISK+LightGlue."""
        import cv2 as cv
        
        # Get original image dimensions before DeepMatcher resizing
        orig_img_a = cv.imread(image_a.path)
        orig_img_b = cv.imread(image_b.path)
        
        if orig_img_a is None or orig_img_b is None:
            return []
            
        orig_h_a, orig_w_a = orig_img_a.shape[:2]
        orig_h_b, orig_w_b = orig_img_b.shape[:2]
        
        # Get the loaded image dimensions (after pipeline resize)
        loaded_h_a, loaded_w_a = image_a.image.shape[:2]
        loaded_h_b, loaded_w_b = image_b.image.shape[:2]
        
        # DeepMatcher resizes to max_size=1024, compute what it would resize to
        max_size = self.deep_matcher.max_size
        
        if max(orig_h_a, orig_w_a) > max_size:
            scale_a_dnn = max_size / max(orig_h_a, orig_w_a)
            dnn_h_a = int(orig_h_a * scale_a_dnn)
            dnn_w_a = int(orig_w_a * scale_a_dnn)
        else:
            dnn_h_a, dnn_w_a = orig_h_a, orig_w_a
            
        if max(orig_h_b, orig_w_b) > max_size:
            scale_b_dnn = max_size / max(orig_h_b, orig_w_b)
            dnn_h_b = int(orig_h_b * scale_b_dnn)
            dnn_w_b = int(orig_w_b * scale_b_dnn)
        else:
            dnn_h_b, dnn_w_b = orig_h_b, orig_w_b
        
        # Get matches from DeepMatcher (in DNN-resized coordinate space)
        mkpts0, mkpts1 = self.deep_matcher.match(image_a.path, image_b.path)
        
        if len(mkpts0) == 0:
            return []
        
        # Scale keypoints from DNN space to loaded image space
        scale_x_a = loaded_w_a / dnn_w_a
        scale_y_a = loaded_h_a / dnn_h_a
        scale_x_b = loaded_w_b / dnn_w_b
        scale_y_b = loaded_h_b / dnn_h_b
        
        # Convert matched keypoints to cv2.KeyPoint and store in images
        image_a.keypoints = [cv.KeyPoint(x=pt[0] * scale_x_a, y=pt[1] * scale_y_a, size=1) for pt in mkpts0]
        image_b.keypoints = [cv.KeyPoint(x=pt[0] * scale_x_b, y=pt[1] * scale_y_b, size=1) for pt in mkpts1]
        
        # Create DMatch objects (1:1 mapping since DeepMatcher returns correspondences)
        matches = [cv.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(mkpts0))]
        
        return matches

    def _compute_sift_matches(self, image_a: Image, image_b: Image) -> List[cv2.DMatch]:
        """Compute matches using SIFT + BruteForce."""
        if image_a.features is None or image_b.features is None:
            return []

        matcher = cv2.DescriptorMatcher_create("BruteForce")  # type: ignore
        raw_matches = matcher.knnMatch(image_a.features, image_b.features, 2)
        matches: List[cv2.DMatch] = []

        for m, n in raw_matches:
            if m.distance < n.distance * self.ratio:
                matches.append(m)

        return matches

