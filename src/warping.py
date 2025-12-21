from importlib.util import source_hash
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from numpy.typing import NDArray

from .image import Image

############################WARPING#####################
#########################################################

def create_translation_matrix(tx, ty):
    """
    Create a 3x3 translation matrix.
    
    Args:
        tx, ty: Translation in x and y directions
    
    Returns:
        T: 3x3 translation matrix
    """
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)
    return T
def apply_homography_to_point(x,y,H):
    """
    Apply homography matrix H to a single point (x, y).
    
    Args:
        x, y: Coordinates of the point
        H: 3x3 homography matrix
    
    Returns:
        x_new, y_new: Transformed coordinates
    """
    point =np.array([x,y,1.0])
    transformed = H @point
    w=transformed[2]
    x_new =transformed[0]/w
    y_new = transformed[1]/w
    
    return x_new,y_new
    
    
def get_transformed_corners(image_shape,H):
    """
    Transform the four corners of an image using homography H.
    
    Args:
        image_shape: (height, width) or (height, width, channels)
        H: 3x3 homography matrix
    
    Returns:
        corners: List of [x, y] transformed corner coordinates
    """
    height,width =image_shape[:2]
    corners = np.array([
        [0,0],
        [width,0],
        [0,height],
        [width,height]
    ],dtype=np.float64)
    
    transformed_corners=[]
    for corner in corners:
        x,y= corner
        x_new,y_new=apply_homography_to_point(x,y,H)
        transformed_corners.append([x_new,y_new])
    return transformed_corners
    
def compute_canvas_size (source_shape,reference_shape,H):
    """
    Compute the size of canvas needed to fit both warped source and reference images.
    
    Args:
        source_shape: (height, width) of source image to be warped
        reference_shape: (height, width) of reference image
        H: 3x3 homography matrix
    
    Returns:
        canvas_size: (canvas_height, canvas_width)
        offset: (x_offset, y_offset) to handle negative coordinates
    """
    
    source_corners=get_transformed_corners(source_shape,H)
    ref_height, ref_width = reference_shape[:2]
    reference_corners = np.array([
        [0, 0],
        [ref_width, 0],
        [0, ref_height],
        [ref_width, ref_height]
    ], dtype=np.float64)
    all_corners =np.vstack([source_corners,reference_corners])
    min_x = np.min (all_corners[:,0])
    max_x = np.max (all_corners[:,0])
    min_y = np.min (all_corners[:,1])
    max_y = np.max (all_corners[:,1])
    
    canvas_width =int (np.ceil(max_x-min_x))  
    canvas_height =int (np.ceil(max_y-min_y))
    x_offset = -min_x if min_x < 0 else 0
    y_offset = -min_y if min_y < 0 else 0
    
    return (canvas_height, canvas_width), (x_offset, y_offset)

## May be changed 

def bilinear_interpolation (image,x,y):
    """
    Perform bilinear interpolation to get pixel value at non-integer coordinates.
    
    Args:
        image: Input image (H x W) for grayscale or (H x W x C) for color
        x, y: Float coordinates where we want to sample
    
    Returns:
        interpolated_value: Pixel value(s) at (x, y)
    """
    
    height,width =image.shape[:2]
    # Check bounds
    if x < 0 or x >= width - 1 or y < 0 or y >= height - 1:
        # Return black for out of bounds
        if len(image.shape) == 3:  # Color image
            return np.zeros(image.shape[2])
        else:  # Grayscale
            return 0
    # Get integer parts (floor)
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    # Get fractional parts
    dx = x - x0
    dy = y - y0
    
    # Get the four neighboring pixels
    # Q11 = (x0, y0), Q21 = (x1, y0), Q12 = (x0, y1), Q22 = (x1, y1)
    Q11 = image[y0, x0]
    Q21 = image[y0, x1]
    Q12 = image[y1, x0]
    Q22 = image[y1, x1]
    
    # Perform bilinear interpolation
    # f(x,y) = (1-dx)(1-dy)Q11 + dx(1-dy)Q21 + (1-dx)dy*Q12 + dx*dy*Q22
    interpolated = (1 - dx) * (1 - dy) * Q11 + \
                   dx * (1 - dy) * Q21 + \
                   (1 - dx) * dy * Q12 + \
                   dx * dy * Q22
    
    return interpolated
    
def warp_image_from_scratch(source,H,reference_shape):
    """
    Warp source image using homography matrix H.
    Creates a canvas large enough to fit both warped source and reference image.
    
    Args:
        source: Source image to be warped (H x W x 3) or (H x W)
        H: 3x3 homography matrix that maps source to reference frame
        reference_shape: (height, width) of the reference image
    
    Returns:
        warped: Warped image on the computed canvas
        offset: (x_offset, y_offset) used for canvas positioning
    """ 
    
    
def warp_image(source, H, reference_shape, use_opencv=True):
    """
    Warp source image using homography matrix H.
    
    Args:
        source: Source image
        H: Homography matrix
        reference_shape: Reference image shape
        use_opencv: If True, use cv2.warpPerspective (fast)
                   If False, use our implementation 
    """
    canvas_size, offset = compute_canvas_size(source.shape, reference_shape, H)
    canvas_height, canvas_width = canvas_size
    x_offset, y_offset = offset
    
    T = create_translation_matrix(x_offset, y_offset)
    H_adjusted = T @ H
    
    if use_opencv:
        # Fast version using OpenCV
        warped = cv2.warpPerspective(source, H_adjusted, 
                                     (canvas_width, canvas_height))
    else:
        # implementation from scratch
        warped = warp_image_from_scratch(source, H_adjusted, canvas_size)
    
    return warped, offset