"""
Module 2: Harris Corner Detector
Input: Preprocessed grayscale images
Output: List of keypoint coordinates (x, y) for each image
Implementation: From Scratch

Method:
1. Compute image gradients using Sobel operators (Ix, Iy)
2. Calculate structure tensor: M = [[Ixx, Ixy], [Ixy, Iyy]]
3. Apply Gaussian smoothing to structure tensor
4. Compute Harris response: R = det(M) - k × trace(M)²
5. Threshold response and apply non-maximum suppression
"""

import cv2
import numpy as np


class HarrisCornerDetector:
    """
    Harris Corner Detector implementation from scratch.
    """
    
    def __init__(self, k=0.04, threshold=0.01, window_size=3, nms_size=5):
        """
        Initialize Harris Corner Detector.
        
        Args:
            k (float): Harris detector free parameter (typically 0.04-0.06)
            threshold (float): Threshold for corner response (relative to max response)
            window_size (int): Size of the Gaussian window for smoothing
            nms_size (int): Size of non-maximum suppression window
        """
        self.k = k
        self.threshold = threshold
        self.window_size = window_size
        self.nms_size = nms_size
    
    def compute_gradients(self, image):
        """
        Compute image gradients using Sobel operators.
        
        Args:
            image (np.ndarray): Grayscale image (normalized to [0, 1])
            
        Returns:
            tuple: (Ix, Iy) - gradients in x and y directions
        """
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
        
        sobel_y = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float32)
        
        # Compute gradients using convolution
        Ix = cv2.filter2D(image, -1, sobel_x)
        Iy = cv2.filter2D(image, -1, sobel_y)
        
        return Ix, Iy
    
    def compute_structure_tensor(self, Ix, Iy):
        """
        Calculate structure tensor components.
        
        Args:
            Ix (np.ndarray): Gradient in x direction
            Iy (np.ndarray): Gradient in y direction
            
        Returns:
            tuple: (Ixx, Iyy, Ixy) - structure tensor components
        """
        # Compute products of derivatives
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        
        return Ixx, Iyy, Ixy
    
    def apply_gaussian_smoothing(self, Ixx, Iyy, Ixy):
        """
        Apply Gaussian smoothing to structure tensor components.
        
        Args:
            Ixx, Iyy, Ixy (np.ndarray): Structure tensor components
            
        Returns:
            tuple: (Sxx, Syy, Sxy) - smoothed structure tensor components
        """
        # Apply Gaussian blur to each component
        Sxx = cv2.GaussianBlur(Ixx, (self.window_size, self.window_size), 0)
        Syy = cv2.GaussianBlur(Iyy, (self.window_size, self.window_size), 0)
        Sxy = cv2.GaussianBlur(Ixy, (self.window_size, self.window_size), 0)
        
        return Sxx, Syy, Sxy
    
    def compute_harris_response(self, Sxx, Syy, Sxy):
        """
        Compute Harris corner response.
        R = det(M) - k × trace(M)²
        where M = [[Sxx, Sxy], [Sxy, Syy]]
        
        Args:
            Sxx, Syy, Sxy (np.ndarray): Smoothed structure tensor components
            
        Returns:
            np.ndarray: Harris response map
        """
        # Compute determinant: det(M) = Sxx * Syy - Sxy²
        det_M = Sxx * Syy - Sxy * Sxy
        
        # Compute trace: trace(M) = Sxx + Syy
        trace_M = Sxx + Syy
        
        # Compute Harris response: R = det(M) - k * trace(M)²
        R = det_M - self.k * (trace_M ** 2)
        
        return R
    
    def apply_threshold(self, R):
        """
        Apply threshold to Harris response.
        
        Args:
            R (np.ndarray): Harris response map
            
        Returns:
            np.ndarray: Binary mask of corner candidates
        """
        # Threshold as percentage of maximum response
        threshold_value = self.threshold * R.max()
        
        # Create binary mask
        corner_mask = R > threshold_value
        
        return corner_mask
    
    def non_maximum_suppression(self, R, corner_mask):
        """
        Apply non-maximum suppression to find local maxima.
        
        Args:
            R (np.ndarray): Harris response map
            corner_mask (np.ndarray): Binary mask of corner candidates
            
        Returns:
            list: List of keypoint coordinates [(x, y), ...]
        """
        keypoints = []
        
        # Get coordinates of corner candidates
        corner_coords = np.argwhere(corner_mask)
        
        # For each corner candidate, check if it's a local maximum
        for coord in corner_coords:
            y, x = coord
            
            # Define neighborhood bounds
            y_min = max(0, y - self.nms_size // 2)
            y_max = min(R.shape[0], y + self.nms_size // 2 + 1)
            x_min = max(0, x - self.nms_size // 2)
            x_max = min(R.shape[1], x + self.nms_size // 2 + 1)
            
            # Extract neighborhood
            neighborhood = R[y_min:y_max, x_min:x_max]
            
            # Check if current point is the maximum in the neighborhood
            if R[y, x] == neighborhood.max():
                keypoints.append((x, y))  # Store as (x, y) for consistency
        
        return keypoints
    
    def detect(self, image):
        """
        Detect Harris corners in an image.
        
        Args:
            image (np.ndarray): Grayscale image (normalized to [0, 1])
            
        Returns:
            list: List of keypoint coordinates [(x, y), ...]
        """
        # Ensure image is in correct format
        if image.dtype != np.float32 and image.dtype != np.float64:
            image = image.astype(np.float32)
        
        # If image is not normalized, normalize it
        if image.max() > 1.0:
            image = image / 255.0
        
        print("Step 1: Computing gradients...")
        Ix, Iy = self.compute_gradients(image)
        
        print("Step 2: Computing structure tensor...")
        Ixx, Iyy, Ixy = self.compute_structure_tensor(Ix, Iy)
        
        print("Step 3: Applying Gaussian smoothing...")
        Sxx, Syy, Sxy = self.apply_gaussian_smoothing(Ixx, Iyy, Ixy)
        
        print("Step 4: Computing Harris response...")
        R = self.compute_harris_response(Sxx, Syy, Sxy)
        
        print("Step 5: Applying threshold...")
        corner_mask = self.apply_threshold(R)
        
        print("Step 6: Non-maximum suppression...")
        keypoints = self.non_maximum_suppression(R, corner_mask)
        
        print(f"Detected {len(keypoints)} corners")
        
        return keypoints, R
    
    def visualize_keypoints(self, image, keypoints, radius=3, color=(0, 255, 0)):
        """
        Visualize detected keypoints on the image.
        
        Args:
            image (np.ndarray): Original image
            keypoints (list): List of keypoint coordinates
            radius (int): Radius of keypoint circles
            color (tuple): Color of keypoint circles (BGR)
            
        Returns:
            np.ndarray: Image with keypoints drawn
        """
        # Convert to uint8 if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            vis_image = (image * 255).astype(np.uint8)
        else:
            vis_image = image.copy()
        
        # Convert to BGR if grayscale
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Draw keypoints
        for x, y in keypoints:
            cv2.circle(vis_image, (int(x), int(y)), radius, color, -1)
        
        return vis_image


if __name__ == "__main__":
    # Example usage
    from preprocessing import ImagePreprocessor
    
    # Preprocess image
    preprocessor = ImagePreprocessor(target_width=800)
    color_img, gray_img = preprocessor.preprocess("imgs/boat/boat1.jpg")
    
    # Detect Harris corners
    detector = HarrisCornerDetector(k=0.04, threshold=0.01, window_size=5, nms_size=5)
    keypoints, response = detector.detect(gray_img)
    
    # Visualize results
    result = detector.visualize_keypoints(color_img, keypoints)
    
    # Save result
    cv2.imwrite("results/harris_corners.jpg", (result * 255).astype(np.uint8))
    print(f"Result saved to results/harris_corners.jpg")
