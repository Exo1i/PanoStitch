"""
Module 1: Image Data Structure
Handles image loading, storage, and basic properties.
"""

import cv2
import numpy as np
from typing import Optional, List
from numpy.typing import NDArray


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
            use_harris: If True, use Harris corner detection; else use SIFT

        TODO Module 3: Replace SIFT descriptors with custom 8D descriptors
        """
        if use_harris:
            # Module 2: Harris Corner Detection
            keypoints = self._harris_corner_detection()

            # Module 3: Still using SIFT descriptors for now (TODO)
            descriptor = cv2.SIFT_create()  # type: ignore
            keypoints, features = descriptor.compute(self.image, keypoints)

            self.keypoints = keypoints
            self.features = features
        else:
            # Original SIFT implementation
            descriptor = cv2.SIFT_create()  # type: ignore
            keypoints, features = descriptor.detectAndCompute(self.image, None)
            self.keypoints = keypoints
            self.features = features

    def _harris_corner_detection(
        self,
        k: float = 0.04,
        threshold: float = 0.01,
        nms_window: int = 5,
        max_keypoints: int = 1000,
    ) -> List[cv2.KeyPoint]:
        """
        Module 2: Harris Corner Detection Implementation

        Args:
            k: Harris detector free parameter (typically 0.04-0.06)
            threshold: Threshold for corner response (relative to max response)
            nms_window: Window size for non-maximum suppression
            max_keypoints: Maximum number of keypoints to return

        Returns:
            List of detected corner keypoints
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 1. Compute gradients using Sobel operator
        Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # 2. Compute products of gradients
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # 3. Apply Gaussian smoothing to gradient products
        sigma = 2.0
        kernel_size = 7
        Sxx = cv2.GaussianBlur(np.asarray(Ixx), (kernel_size, kernel_size), sigma).astype(np.float32)  # type: ignore
        Syy = cv2.GaussianBlur(np.asarray(Iyy), (kernel_size, kernel_size), sigma).astype(np.float32)  # type: ignore
        Sxy = cv2.GaussianBlur(np.asarray(Ixy), (kernel_size, kernel_size), sigma).astype(np.float32)  # type: ignore

        # 4. Compute Harris corner response
        # R = det(M) - k * trace(M)^2
        # where M = [[Sxx, Sxy], [Sxy, Syy]]
        det_M = Sxx * Syy - Sxy * Sxy
        trace_M = Sxx + Syy
        R = det_M - k * (trace_M**2)

        # 5. Threshold corner response
        R_threshold = threshold * R.max()
        corner_mask = R > R_threshold

        # 6. Non-maximum suppression
        corner_mask = self._non_maximum_suppression(R, corner_mask, nms_window)

        # 7. Extract keypoint coordinates
        y_coords, x_coords = np.where(corner_mask)
        responses = R[y_coords, x_coords]

        # 8. Sort by response and keep top keypoints
        if len(responses) > max_keypoints:
            top_indices = np.argpartition(responses, -max_keypoints)[-max_keypoints:]
            x_coords = x_coords[top_indices]
            y_coords = y_coords[top_indices]
            responses = responses[top_indices]

        # 9. Convert to OpenCV KeyPoint format
        keypoints = [
            cv2.KeyPoint(float(x), float(y), 31.0, -1, float(r))
            for x, y, r in zip(x_coords, y_coords, responses)
        ]

        return keypoints

    def _non_maximum_suppression(
        self, response: NDArray[np.float32], mask: NDArray[np.bool_], window: int
    ) -> NDArray[np.bool_]:
        """
        Apply non-maximum suppression to corner responses.

        Args:
            response: Harris corner response map
            mask: Boolean mask of potential corners
            window: Window size for NMS

        Returns:
            Filtered boolean mask after NMS
        """
        result = np.zeros_like(mask)
        h, w = response.shape
        pad = window // 2

        # Get coordinates of potential corners
        y_coords, x_coords = np.where(mask)

        for y, x in zip(y_coords, x_coords):
            # Extract local window
            y_start = max(0, y - pad)
            y_end = min(h, y + pad + 1)
            x_start = max(0, x - pad)
            x_end = min(w, x + pad + 1)

            local_window = response[y_start:y_end, x_start:x_end]

            # Check if current point is maximum in window
            if response[y, x] == local_window.max():
                result[y, x] = True

        return result
