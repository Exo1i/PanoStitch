"""
Module 3: Feature Description
Input: Keypoints from feature detection
Output: Feature descriptor vectors (N × 128 dimensional)
Implementation: From Scratch

Method:
1. Extract 16×16 pixel patch around each keypoint
2. Divide patch into 4×4 cells (4x4 pixels each)
3. For each cell, compute gradient orientation histogram (8 bins)
4. Concatenate all histograms (4×4×8 = 128 dimensions)
5. Normalize descriptor to unit length

This produces 128D descriptors similar to SIFT for better matching.
"""

import cv2
import numpy as np


class FeatureDescriptor:
    """
    Feature Descriptor using Histogram of Oriented Gradients (HOG).
    Uses 4x4 spatial cells with 8 orientation bins = 128D descriptor.
    """
    
    def __init__(self, patch_size=16, num_bins=8, num_cells=4):
        """
        Initialize Feature Descriptor.
        
        Args:
            patch_size (int): Size of the patch to extract around keypoint (default: 16)
            num_bins (int): Number of orientation bins for histogram (default: 8)
            num_cells (int): Number of cells in each dimension (default: 4, gives 4x4=16 cells)
        """
        self.patch_size = patch_size
        self.num_bins = num_bins
        self.num_cells = num_cells
        self.cell_size = patch_size // num_cells
        self.bin_width = 360.0 / num_bins  # degrees per bin
        self.descriptor_size = num_cells * num_cells * num_bins  # 4*4*8 = 128
    
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
    
    def compute_cell_histogram(self, magnitude, orientation, cell_y, cell_x):
        """
        Compute weighted histogram for a single cell.
        
        Args:
            magnitude (np.ndarray): Gradient magnitudes for entire patch
            orientation (np.ndarray): Gradient orientations for entire patch
            cell_y, cell_x (int): Cell indices (0 to num_cells-1)
            
        Returns:
            np.ndarray: Histogram for this cell (num_bins,)
        """
        # Calculate cell boundaries
        y_start = cell_y * self.cell_size
        y_end = (cell_y + 1) * self.cell_size
        x_start = cell_x * self.cell_size
        x_end = (cell_x + 1) * self.cell_size
        
        # Extract cell data
        cell_mag = magnitude[y_start:y_end, x_start:x_end].flatten()
        cell_ori = orientation[y_start:y_end, x_start:x_end].flatten()
        
        # Compute histogram with linear interpolation
        histogram = np.zeros(self.num_bins, dtype=np.float32)
        
        for i in range(len(cell_mag)):
            mag = cell_mag[i]
            ori = cell_ori[i]
            
            # Find bin index
            bin_idx = ori / self.bin_width
            
            # Linear interpolation between bins
            bin_lower = int(np.floor(bin_idx)) % self.num_bins
            bin_upper = (bin_lower + 1) % self.num_bins
            
            weight_upper = bin_idx - np.floor(bin_idx)
            weight_lower = 1.0 - weight_upper
            
            histogram[bin_lower] += mag * weight_lower
            histogram[bin_upper] += mag * weight_upper
        
        return histogram
    
    def normalize_descriptor(self, descriptor):
        """
        Normalize descriptor to unit length with clipping.
        
        Args:
            descriptor (np.ndarray): Feature descriptor
            
        Returns:
            np.ndarray: Normalized descriptor
        """
        # Compute L2 norm
        norm = np.linalg.norm(descriptor)
        
        # Avoid division by zero
        if norm > 1e-7:
            descriptor = descriptor / norm
        
        # Clip values to 0.2 (SIFT-style thresholding)
        descriptor = np.clip(descriptor, 0, 0.2)
        
        # Re-normalize after clipping
        norm = np.linalg.norm(descriptor)
        if norm > 1e-7:
            descriptor = descriptor / norm
        
        return descriptor
    
    def compute_descriptor(self, image, keypoint):
        """
        Compute 128D feature descriptor for a single keypoint.
        
        Args:
            image (np.ndarray): Grayscale image
            keypoint (tuple): Keypoint coordinates (x, y)
            
        Returns:
            np.ndarray: Feature descriptor (128,), or None if failed
        """
        x, y = keypoint
        
        # Extract patch around keypoint
        patch = self.extract_patch(image, x, y)
        if patch is None:
            return None
        
        # Ensure patch has correct size
        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            return None
        
        # Convert to float if needed
        if patch.dtype != np.float32:
            patch = patch.astype(np.float32)
        
        # Compute gradients
        magnitude, orientation = self.compute_gradients(patch)
        
        # Build descriptor from 4x4 cells
        descriptor = []
        for cell_y in range(self.num_cells):
            for cell_x in range(self.num_cells):
                histogram = self.compute_cell_histogram(magnitude, orientation, cell_y, cell_x)
                descriptor.extend(histogram)
        
        descriptor = np.array(descriptor, dtype=np.float32)
        
        # Normalize descriptor
        descriptor = self.normalize_descriptor(descriptor)
        
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
                - descriptors: np.ndarray of shape (N, 128)
        """
        valid_keypoints = []
        descriptors = []
        
        print(f"Computing 128D descriptors for {len(keypoints)} keypoints...")
        
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
        
        print(f"Successfully computed {len(valid_keypoints)} descriptors ({self.descriptor_size}D)")
        
        return valid_keypoints, descriptors
    
    def visualize_descriptor(self, descriptor):
        """
        Create a visual representation of a descriptor.
        
        Args:
            descriptor (np.ndarray): Feature descriptor
            
        Returns:
            np.ndarray: Visualization image
        """
        # Reshape to 4x4 cells x 8 bins
        desc_reshaped = descriptor.reshape(self.num_cells, self.num_cells, self.num_bins)
        
        # Create visualization
        cell_vis_size = 40
        vis_size = self.num_cells * cell_vis_size
        vis = np.ones((vis_size, vis_size, 3), dtype=np.uint8) * 255
        
        # Draw each cell's histogram as a small bar chart
        for cy in range(self.num_cells):
            for cx in range(self.num_cells):
                cell_hist = desc_reshaped[cy, cx]
                cell_hist_norm = cell_hist / (cell_hist.max() + 1e-8)
                
                # Draw mini bar chart
                bar_width = cell_vis_size // self.num_bins
                for b in range(self.num_bins):
                    bar_height = int(cell_hist_norm[b] * (cell_vis_size - 4))
                    x1 = cx * cell_vis_size + b * bar_width
                    x2 = x1 + bar_width - 1
                    y2 = (cy + 1) * cell_vis_size - 2
                    y1 = y2 - bar_height
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
    descriptor = FeatureDescriptor(patch_size=16, num_bins=8, num_cells=4)
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
