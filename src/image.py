"""
Module 1: Image Data Structure
Handles image loading, storage, and basic properties.
"""

import cv2
import numpy as np
from typing import Optional, List
from numpy.typing import NDArray
from src.sift_implementation import computeKeypointsAndDescriptors
from src.harris_detector import HarrisCornerDetector
from src.feature_descriptor import FeatureDescriptor


class Image:
    """Represents an image with its features and transformations."""

    def __init__(self, path: str, size: Optional[int] = None) -> None:
        """
        Image constructor.

        Args:
            path: path to the image
            size: maximum dimension to resize the image to (None to keep original size)
        """
        self.path = path
        img = cv2.imread(path)

        if img is None:
            raise ValueError(f"Could not load image from {path}")

        # Force to a numpy uint8 array for consistent typing
        self.image: NDArray[np.uint8] = np.asarray(img).astype(np.uint8)

        # Optional resizing
        if size is not None:
            h, w = self.image.shape[:2]
            if max(w, h) > size:
                if w > h:
                    self.image = np.asarray(
                        cv2.resize(self.image, (size, int(h * size / w)))
                    ).astype(np.uint8)
                else:
                    self.image = np.asarray(
                        cv2.resize(self.image, (int(w * size / h), size))
                    ).astype(np.uint8)

        self.keypoints: Optional[List[cv2.KeyPoint]] = None
        self.features: Optional[NDArray[np.float32]] = None
        self.H: NDArray[np.float64] = np.eye(3, dtype=np.float64)  # Homography matrix
        self.component_id: int = 0
        self.gain: NDArray[np.float32] = np.ones(3, dtype=np.float32)

    def compute_features(self, use_harris: bool = True) -> None:
        """
            Module 2 & 3: Compute the features and keypoints.

            Args:
                use_harris: If True, use Harris corner detection + Custom HOG Descriptor; else use custom SIFT
        """
        if use_harris:
            # Module 2: Harris Corner Detection
            harris_detector = HarrisCornerDetector(k=0.04, threshold=0.01, window_size=5, nms_size=5)
            # Normalize image for Harris
            gray_norm = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            keypoints_coords, _ = harris_detector.detect(gray_norm)

            # Convert to OpenCV KeyPoint format
            keypoints = [
                cv2.KeyPoint(float(x), float(y), 16.0)
                for x, y in keypoints_coords
            ]

            # Module 3: Custom Feature Descriptors
            # Use HOG-based descriptor
            feature_extractor = FeatureDescriptor(patch_size=16, num_bins=8, num_cells=4)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            valid_keypoints, features = feature_extractor.compute_descriptors(gray, keypoints_coords)

            # Update keypoints to only include valid ones
            self.keypoints = [
                cv2.KeyPoint(float(x), float(y), 16.0)
                for x, y in valid_keypoints
            ]
            self.features = features


        else:
            # Use our SIFT implementation
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            keypoints, features = computeKeypointsAndDescriptors(gray)
            self.keypoints = keypoints
            self.features = features

