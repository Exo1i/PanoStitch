"""
Module 3: Feature Description
Input: Keypoints from feature detection
Output: Feature descriptor vectors (N × 8 dimensional)
Implementation: From Scratch

Method:
1. Extract 16×16 pixel patch around each keypoint
2. Compute gradient magnitude and orientation
3. Create Histogram of Oriented Gradients (8 bins)
4. Weight histogram by gradient magnitude
5. Normalize descriptor to unit length

Advantages:
- Invariant to rotation
- Robust to illumination changes
- Compact representation
"""

import cv2
import numpy as np


class FeatureDescriptor:
    """
    Feature Descriptor using Histogram of Oriented Gradients (HOG).
    """
    
    def __init__(self, patch_size=16, num_bins=8):
        """
        Initialize Feature Descriptor.
        
        Args:
            patch_size (int): Size of the patch to extract around keypoint (default: 16)
            num_bins (int): Number of orientation bins for histogram (default: 8)
        """
        self.patch_size = patch_size
        self.num_bins = num_bins
        self.bin_width = 360.0 / num_bins  # degrees per bin
    
    def extract_patch(self, image, x, y):
        """
        Extract a patch around a keypoint.
        
        Args:
            image (np.ndarray): Input grayscale image
            x, y (int): Keypoint coordinates
            
        Returns:
            np.ndarray: Extracted patch, or None if out of bounds
        """
        half_size = self.patch_size // 2
        
        # Calculate patch boundaries
        y_min = int(y - half_size)
        y_max = int(y + half_size)
        x_min = int(x - half_size)
        x_max = int(x + half_size)
        
        # Check if patch is within image boundaries
        if (y_min < 0 or y_max > image.shape[0] or 
            x_min < 0 or x_max > image.shape[1]):
            return None
        
        # Extract patch
        patch = image[y_min:y_max, x_min:x_max]
        
        return patch
    
    def compute_gradients(self, patch):
        """
        Compute gradient magnitude and orientation for a patch.
        
        Args:
            patch (np.ndarray): Image patch
            
        Returns:
            tuple: (magnitude, orientation) in degrees [0, 360)
        """
        # Sobel operators for gradient computation
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
        
        sobel_y = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float32)
        
        # Compute gradients
        grad_x = cv2.filter2D(patch, -1, sobel_x)
        grad_y = cv2.filter2D(patch, -1, sobel_y)
        
        # Compute magnitude and orientation
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x) * 180.0 / np.pi  # Convert to degrees
        
        # Ensure orientation is in [0, 360) range
        orientation = orientation % 360.0
        
        return magnitude, orientation
    
    def compute_histogram(self, magnitude, orientation):
        """
        Compute weighted histogram of oriented gradients.
        
        Args:
            magnitude (np.ndarray): Gradient magnitudes
            orientation (np.ndarray): Gradient orientations in degrees
            
        Returns:
            np.ndarray: Histogram of oriented gradients (num_bins,)
        """
        histogram = np.zeros(self.num_bins, dtype=np.float32)
        
        # Flatten arrays for easier processing
        mag_flat = magnitude.flatten()
        ori_flat = orientation.flatten()
        
        # Compute histogram with linear interpolation between bins
        for i in range(len(mag_flat)):
            mag = mag_flat[i]
            ori = ori_flat[i]
            
            # Find which bin this orientation belongs to
            bin_idx = ori / self.bin_width
            
            # Linear interpolation between two adjacent bins
            bin_lower = int(np.floor(bin_idx)) % self.num_bins
            bin_upper = int(np.ceil(bin_idx)) % self.num_bins
            
            # Weight for interpolation
            weight_upper = bin_idx - np.floor(bin_idx)
            weight_lower = 1.0 - weight_upper
            
            # Add weighted magnitude to histogram bins
            histogram[bin_lower] += mag * weight_lower
            if bin_lower != bin_upper:
                histogram[bin_upper] += mag * weight_upper
        
        return histogram
    
    def normalize_descriptor(self, descriptor):
        """
        Normalize descriptor to unit length.
        
        Args:
            descriptor (np.ndarray): Feature descriptor
            
        Returns:
            np.ndarray: Normalized descriptor
        """
        # Compute L2 norm
        norm = np.linalg.norm(descriptor)
        
        # Avoid division by zero
        if norm > 0:
            descriptor = descriptor / norm
        
        # Clip values to avoid numerical instability (optional)
        descriptor = np.clip(descriptor, 0, 0.2)
        
        # Re-normalize after clipping
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm
        
        return descriptor
    
    def compute_descriptor(self, image, keypoint):
        """
        Compute feature descriptor for a single keypoint.
        
        Args:
            image (np.ndarray): Grayscale image
            keypoint (tuple): Keypoint coordinates (x, y)
            
        Returns:
            np.ndarray: Feature descriptor (num_bins,), or None if failed
        """
        x, y = keypoint
        
        # Extract patch around keypoint
        patch = self.extract_patch(image, x, y)
        if patch is None:
            return None
        
        # Ensure patch has correct size
        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            return None
        
        # Compute gradients
        magnitude, orientation = self.compute_gradients(patch)
        
        # Compute histogram
        histogram = self.compute_histogram(magnitude, orientation)
        
        # Normalize descriptor
        descriptor = self.normalize_descriptor(histogram)
        
        return descriptor
    
    def compute_descriptors(self, image, keypoints):
        """
        Compute feature descriptors for multiple keypoints.
        
        Args:
            image (np.ndarray): Grayscale image
            keypoints (list): List of keypoint coordinates [(x, y), ...]
            
        Returns:
            tuple: (valid_keypoints, descriptors)
                - valid_keypoints: List of keypoints with valid descriptors
                - descriptors: np.ndarray of shape (N, num_bins)
        """
        valid_keypoints = []
        descriptors = []
        
        print(f"Computing descriptors for {len(keypoints)} keypoints...")
        
        for i, kp in enumerate(keypoints):
            descriptor = self.compute_descriptor(image, kp)
            
            if descriptor is not None:
                valid_keypoints.append(kp)
                descriptors.append(descriptor)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(keypoints)} keypoints")
        
        # Convert to numpy array
        if len(descriptors) > 0:
            descriptors = np.array(descriptors, dtype=np.float32)
        else:
            descriptors = np.array([])
        
        print(f"Successfully computed {len(valid_keypoints)} descriptors")
        
        return valid_keypoints, descriptors
    
    def visualize_descriptor(self, descriptor):
        """
        Create a visual representation of a descriptor.
        
        Args:
            descriptor (np.ndarray): Feature descriptor
            
        Returns:
            np.ndarray: Visualization image
        """
        # Create a simple bar chart visualization
        height = 100
        width = self.num_bins * 20
        vis = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Normalize descriptor for visualization
        desc_norm = descriptor / (descriptor.max() + 1e-8)
        
        # Draw bars
        bar_width = width // self.num_bins
        for i in range(self.num_bins):
            bar_height = int(desc_norm[i] * (height - 10))
            x1 = i * bar_width
            x2 = (i + 1) * bar_width - 2
            y1 = height - bar_height - 5
            y2 = height - 5
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 128, 255), -1)
        
        return vis


if __name__ == "__main__":
    # Example usage
    from preprocessing import ImagePreprocessor
    from harris_detector import HarrisCornerDetector
    
    # Preprocess image
    print("Preprocessing image...")
    preprocessor = ImagePreprocessor(target_width=800)
    color_img, gray_img = preprocessor.preprocess("imgs/boat/boat1.jpg")
    
    # Detect Harris corners
    print("\nDetecting Harris corners...")
    detector = HarrisCornerDetector(k=0.04, threshold=0.01, window_size=5, nms_size=5)
    keypoints, _ = detector.detect(gray_img)
    
    # Compute descriptors
    print("\nComputing descriptors...")
    descriptor = FeatureDescriptor(patch_size=16, num_bins=8)
    valid_keypoints, descriptors = descriptor.compute_descriptors(gray_img, keypoints)
    
    print(f"\nSummary:")
    print(f"  Total keypoints detected: {len(keypoints)}")
    print(f"  Valid keypoints with descriptors: {len(valid_keypoints)}")
    print(f"  Descriptor shape: {descriptors.shape}")
    print(f"  Descriptor dimension: {descriptors.shape[1] if len(descriptors) > 0 else 0}")
    
    # Visualize some descriptors
    if len(descriptors) > 0:
        print("\nVisualizing first descriptor...")
        vis = descriptor.visualize_descriptor(descriptors[0])
        cv2.imwrite("results/descriptor_example.jpg", vis)
        print("Descriptor visualization saved to results/descriptor_example.jpg")
