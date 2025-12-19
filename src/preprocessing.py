"""
Module 1: Image Preprocessing
Input: Raw images from camera
Output: Preprocessed images ready for feature detection
Implementation: Using OpenCV2 functions
"""

import cv2
import numpy as np


class ImagePreprocessor:
    """
    Handles image preprocessing operations including resizing and normalization.
    """
    
    def __init__(self, target_width=800):
        """
        Initialize the preprocessor.
        
        Args:
            target_width (int): Target width for resizing images (default: 800px)
        """
        self.target_width = target_width
    
    def preprocess(self, image_path, normalize=True, equalize_hist=False):
        """
        Preprocess a single image.
        
        Args:
            image_path (str): Path to the input image
            normalize (bool): Whether to normalize pixel values to [0, 1]
            equalize_hist (bool): Whether to apply histogram equalization
            
        Returns:
            tuple: (preprocessed_color_image, grayscale_image)
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Resize image to consistent dimensions
        img_resized = self.resize_image(img)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Optional: Histogram equalization for better contrast
        if equalize_hist:
            gray = cv2.equalizeHist(gray)
        
        # Normalize pixel values to [0, 1] range
        if normalize:
            img_normalized = img_resized.astype(np.float32) / 255.0
            gray_normalized = gray.astype(np.float32) / 255.0
        else:
            img_normalized = img_resized
            gray_normalized = gray
        
        return img_normalized, gray_normalized
    
    def resize_image(self, image):
        """
        Resize image to target width while maintaining aspect ratio.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Resized image
        """
        height, width = image.shape[:2]
        
        # Calculate new height to maintain aspect ratio
        aspect_ratio = height / width
        new_height = int(self.target_width * aspect_ratio)
        
        # Resize image
        resized = cv2.resize(image, (self.target_width, new_height), 
                           interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def preprocess_batch(self, image_paths, normalize=True, equalize_hist=False):
        """
        Preprocess multiple images.
        
        Args:
            image_paths (list): List of paths to input images
            normalize (bool): Whether to normalize pixel values
            equalize_hist (bool): Whether to apply histogram equalization
            
        Returns:
            tuple: (list of color images, list of grayscale images)
        """
        color_images = []
        gray_images = []
        
        for path in image_paths:
            try:
                color_img, gray_img = self.preprocess(path, normalize, equalize_hist)
                color_images.append(color_img)
                gray_images.append(gray_img)
                print(f"Preprocessed: {path}")
            except Exception as e:
                print(f"Error preprocessing {path}: {str(e)}")
        
        return color_images, gray_images
    
    def color_correction(self, image, reference_image=None):
        """
        Optional: Apply basic color correction.
        
        Args:
            image (np.ndarray): Input image
            reference_image (np.ndarray): Optional reference for color matching
            
        Returns:
            np.ndarray: Color corrected image
        """
        # Simple color correction using histogram matching
        if reference_image is not None:
            # Match histogram to reference
            corrected = np.zeros_like(image)
            for i in range(3):  # For each color channel
                corrected[:, :, i] = self._match_histograms(
                    image[:, :, i], reference_image[:, :, i]
                )
            return corrected
        else:
            # Simple normalization per channel
            corrected = np.zeros_like(image)
            for i in range(3):
                channel = image[:, :, i]
                corrected[:, :, i] = cv2.normalize(channel, None, 0, 255, 
                                                  cv2.NORM_MINMAX)
            return corrected
    
    def _match_histograms(self, source, reference):
        """
        Match histogram of source to reference.
        
        Args:
            source (np.ndarray): Source channel
            reference (np.ndarray): Reference channel
            
        Returns:
            np.ndarray: Matched channel
        """
        # Compute histograms
        src_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
        
        # Compute CDFs
        src_cdf = src_hist.cumsum()
        ref_cdf = ref_hist.cumsum()
        
        # Normalize CDFs
        src_cdf = src_cdf / src_cdf[-1]
        ref_cdf = ref_cdf / ref_cdf[-1]
        
        # Create lookup table
        lookup = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # Find closest match in reference CDF
            lookup[i] = np.argmin(np.abs(ref_cdf - src_cdf[i]))
        
        # Apply lookup table
        return lookup[source.astype(np.uint8)]


if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor(target_width=800)
    
    # Test with single image
    try:
        color_img, gray_img = preprocessor.preprocess(
            "imgs/boat/boat1.jpg",
            normalize=True,
            equalize_hist=False
        )
        print(f"Color image shape: {color_img.shape}")
        print(f"Gray image shape: {gray_img.shape}")
        print(f"Color image range: [{color_img.min():.3f}, {color_img.max():.3f}]")
        print(f"Gray image range: [{gray_img.min():.3f}, {gray_img.max():.3f}]")
    except Exception as e:
        print(f"Error: {e}")
