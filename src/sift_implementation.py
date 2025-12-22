from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import numpy as np
import logging

####################
# Global variables #
####################

logger = logging.getLogger(__name__)
float_tolerance = 1e-7

#################
# Main function #
#################

def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    """Compute SIFT keypoints and descriptors for an input image
    """
    image = image.astype('float32')
    base_image = generateBaseImage(image, sigma, assumed_blur)
    num_octaves = computeNumberOfOctaves(base_image.shape)
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals)
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    dog_images = generateDoGImages(gaussian_images)
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    keypoints = removeDuplicateKeypoints(keypoints)
    keypoints = convertKeypointsToInputImageSize(keypoints)
    descriptors = generateDescriptors(keypoints, gaussian_images)
    return keypoints, descriptors

#########################
# Image pyramid related #
#########################

def generateBaseImage(image, sigma, assumed_blur):
    """Generate base image from input image by upsampling by 2 in both directions and blurring
    """
    logger.debug('Generating base image...')
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)  # the image blur is now sigma instead of assumed_blur

def computeNumberOfOctaves(image_shape):
    """Compute number of octaves in image pyramid as function of base image shape (OpenCV default)
    """
    return int(round(log(min(image_shape)) / log(2) - 1))

def generateGaussianKernels(sigma, num_intervals):
    """Generate list of gaussian kernels at which to blur the input image. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper.
    """
    logger.debug('Generating scales...')
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels

def generateGaussianImages(image, num_octaves, gaussian_kernels):
    """Generate scale-space pyramid of Gaussian images
    """
    logger.debug('Generating Gaussian images...')
    gaussian_images = []

    for octave_index in range(num_octaves):
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(image)  # first image in octave already has the correct blur
        for gaussian_kernel in gaussian_kernels[1:]:
            image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[-3]
        image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
    return array(gaussian_images, dtype=object)

def generateDoGImages(gaussian_images):
    """Generate Difference-of-Gaussians image pyramid
    """
    logger.debug('Generating Difference-of-Gaussian images...')
    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
        dog_images.append(dog_images_in_octave)
    return array(dog_images, dtype=object)

###############################
# Scale-space extrema related #
###############################

def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
    """Find pixel positions of all scale-space extrema in the image pyramid - VECTORIZED VERSION
    """
    logger.debug('Finding scale-space extrema...')
    threshold = floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # Vectorized extrema detection
            first_image = first_image.astype(np.float32)
            second_image = second_image.astype(np.float32)
            third_image = third_image.astype(np.float32)
            
            # Get the center values (excluding border)
            h, w = second_image.shape
            b = image_border_width
            center = second_image[b:h-b, b:w-b]
            
            # Check if absolute value is above threshold
            abs_above_threshold = np.abs(center) > threshold
            
            # For maxima: center >= all 26 neighbors
            # For minima: center <= all 26 neighbors
            
            # Build all 26 neighbor comparisons
            # Same layer (8 neighbors)
            is_max = abs_above_threshold.copy()
            is_min = abs_above_threshold.copy()
            
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    neighbor = second_image[b+di:h-b+di, b+dj:w-b+dj]
                    is_max = is_max & (center >= neighbor)
                    is_min = is_min & (center <= neighbor)
            
            # Adjacent layers (9 neighbors each)
            for layer_img in [first_image, third_image]:
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        neighbor = layer_img[b+di:h-b+di, b+dj:w-b+dj]
                        is_max = is_max & (center >= neighbor)
                        is_min = is_min & (center <= neighbor)
            
            # Combine maxima and minima, require positive for max, negative for min
            is_extremum = (is_max & (center > 0)) | (is_min & (center < 0))
            
            # Get coordinates of extrema
            extrema_coords = np.argwhere(is_extremum)
            
            for coord in extrema_coords:
                i, j = coord[0] + b, coord[1] + b
                localization_result = localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                if localization_result is not None:
                    keypoint, localized_image_index = localization_result
                    keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                    for keypoint_with_orientation in keypoints_with_orientations:
                        keypoints.append(keypoint_with_orientation)
    return keypoints

def localizeExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
    """
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
        pixel_cube = stack([first_image[i-1:i+2, j-1:j+2],
                            second_image[i-1:i+2, j-1:j+2],
                            third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        gradient = computeGradientAtCenterPixel(pixel_cube)
        hessian = computeHessianAtCenterPixel(pixel_cube)
        extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        # make sure the new pixel_cube will lie entirely within the image
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * dot(gradient, extremum_update)
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # Contrast check passed -- construct and return OpenCV KeyPoint object
            keypoint = KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    return None

def computeGradientAtCenterPixel(pixel_array):
    """Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    """
    # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
    # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return array([dx, dy, ds])

def computeHessianAtCenterPixel(pixel_array):
    """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    """
    # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

#########################
# Keypoint orientations #
#########################

def computeKeypointsWithOrientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """Compute orientations for each keypoint - VECTORIZED VERSION
    """
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / float32(2 ** (octave_index + 1))
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    
    # Get keypoint location in this octave
    kp_y = int(round(keypoint.pt[1] / float32(2 ** octave_index)))
    kp_x = int(round(keypoint.pt[0] / float32(2 ** octave_index)))
    
    # Define region bounds (ensure within image)
    y_min = max(1, kp_y - radius)
    y_max = min(image_shape[0] - 2, kp_y + radius)
    x_min = max(1, kp_x - radius)
    x_max = min(image_shape[1] - 2, kp_x + radius)
    
    if y_min >= y_max or x_min >= x_max:
        return keypoints_with_orientations
    
    # Extract region and compute gradients vectorized
    region = gaussian_image[y_min:y_max+1, x_min:x_max+1]
    
    # Gradient computation using slicing (central differences)
    dx = gaussian_image[y_min:y_max+1, x_min+1:x_max+2] - gaussian_image[y_min:y_max+1, x_min-1:x_max]
    dy = gaussian_image[y_min-1:y_max, x_min:x_max+1] - gaussian_image[y_min+1:y_max+2, x_min:x_max+1]
    
    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
    
    # Compute weights for all pixels in region
    y_coords, x_coords = np.meshgrid(
        np.arange(y_min, y_max + 1) - kp_y,
        np.arange(x_min, x_max + 1) - kp_x,
        indexing='ij'
    )
    weights = np.exp(weight_factor * (y_coords ** 2 + x_coords ** 2))
    
    # Compute histogram indices
    histogram_indices = np.round(gradient_orientation * num_bins / 360.).astype(int) % num_bins
    
    # Weighted magnitudes
    weighted_magnitudes = weights * gradient_magnitude
    
    # Build histogram using np.add.at for accumulation
    raw_histogram = np.zeros(num_bins)
    np.add.at(raw_histogram, histogram_indices.ravel(), weighted_magnitudes.ravel())

    # Smooth histogram
    smooth_histogram = np.zeros(num_bins)
    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 
                               4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + 
                               raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    
    orientation_max = max(smooth_histogram)
    orientation_peaks = where(logical_and(smooth_histogram > roll(smooth_histogram, 1), smooth_histogram > roll(smooth_histogram, -1)))[0]
    
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0
            new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations

##############################
# Duplicate keypoint removal #
##############################

def compareKeypoints(keypoint1, keypoint2):
    """Return True if keypoint1 is less than keypoint2
    """
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def removeDuplicateKeypoints(keypoints):
    """Sort keypoints and remove duplicate keypoints
    """
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints

#############################
# Keypoint scale conversion #
#############################

def convertKeypointsToInputImageSize(keypoints):
    """Convert keypoint point, size, and octave to input image size
    """
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints

#########################
# Descriptor generation #
#########################

def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / float32(1 << octave) if octave >= 0 else float32(1 << -octave)
    return octave, layer, scale

def generateDescriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """Generate descriptors for each keypoint - FULLY VECTORIZED VERSION
    """
    logger.debug('Generating descriptors...')
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = round(scale * array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = cos(deg2rad(angle))
        sin_angle = sin(deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        histogram_tensor = zeros((window_width + 2, window_width + 2, num_bins))

        # Descriptor window size
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5))
        half_width = int(min(half_width, sqrt(num_rows ** 2 + num_cols ** 2)))

        # Create meshgrid for all row/col combinations
        rows = np.arange(-half_width, half_width + 1)
        cols = np.arange(-half_width, half_width + 1)
        row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
        
        # Rotate coordinates
        row_rot = col_grid * sin_angle + row_grid * cos_angle
        col_rot = col_grid * cos_angle - row_grid * sin_angle
        
        # Compute bin positions
        row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
        col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
        
        # Valid bin mask
        valid_bin_mask = (row_bin > -1) & (row_bin < window_width) & (col_bin > -1) & (col_bin < window_width)
        
        # Window coordinates in image
        window_row = (point[1] + row_grid).astype(int)
        window_col = (point[0] + col_grid).astype(int)
        
        # Valid image bounds mask
        valid_img_mask = (window_row > 0) & (window_row < num_rows - 1) & (window_col > 0) & (window_col < num_cols - 1)
        
        # Combined mask
        valid_mask = valid_bin_mask & valid_img_mask
        
        if not np.any(valid_mask):
            descriptors.append(zeros(window_width * window_width * num_bins))
            continue
        
        # Get valid coordinates
        valid_wr = window_row[valid_mask]
        valid_wc = window_col[valid_mask]
        
        # Vectorized gradient computation
        dx = gaussian_image[valid_wr, valid_wc + 1] - gaussian_image[valid_wr, valid_wc - 1]
        dy = gaussian_image[valid_wr - 1, valid_wc] - gaussian_image[valid_wr + 1, valid_wc]
        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
        
        # Get rotated coordinates and bins for valid pixels
        valid_row_rot = row_rot[valid_mask]
        valid_col_rot = col_rot[valid_mask]
        valid_row_bin = row_bin[valid_mask]
        valid_col_bin = col_bin[valid_mask]
        
        # Vectorized weight computation
        weight = np.exp(weight_multiplier * ((valid_row_rot / hist_width) ** 2 + (valid_col_rot / hist_width) ** 2))
        
        # Orientation bin
        orientation_bin = (gradient_orientation - angle) * bins_per_degree
        
        # Trilinear interpolation - all vectorized
        row_bin_floor = np.floor(valid_row_bin).astype(int)
        col_bin_floor = np.floor(valid_col_bin).astype(int)
        orientation_bin_floor = np.floor(orientation_bin).astype(int)
        
        row_fraction = valid_row_bin - row_bin_floor
        col_fraction = valid_col_bin - col_bin_floor
        orientation_fraction = orientation_bin - orientation_bin_floor
        
        # Handle negative orientation bins
        orientation_bin_floor = orientation_bin_floor % num_bins
        
        magnitude = weight * gradient_magnitude
        
        # Compute all 8 trilinear interpolation weights
        c1 = magnitude * row_fraction
        c0 = magnitude * (1 - row_fraction)
        c11 = c1 * col_fraction
        c10 = c1 * (1 - col_fraction)
        c01 = c0 * col_fraction
        c00 = c0 * (1 - col_fraction)
        c111 = c11 * orientation_fraction
        c110 = c11 * (1 - orientation_fraction)
        c101 = c10 * orientation_fraction
        c100 = c10 * (1 - orientation_fraction)
        c011 = c01 * orientation_fraction
        c010 = c01 * (1 - orientation_fraction)
        c001 = c00 * orientation_fraction
        c000 = c00 * (1 - orientation_fraction)
        
        # Use np.add.at for vectorized histogram accumulation
        r0 = row_bin_floor + 1
        r1 = row_bin_floor + 2
        c0_idx = col_bin_floor + 1
        c1_idx = col_bin_floor + 2
        o0 = orientation_bin_floor
        o1 = (orientation_bin_floor + 1) % num_bins
        
        # Accumulate contributions to histogram
        np.add.at(histogram_tensor, (r0, c0_idx, o0), c000)
        np.add.at(histogram_tensor, (r0, c0_idx, o1), c001)
        np.add.at(histogram_tensor, (r0, c1_idx, o0), c010)
        np.add.at(histogram_tensor, (r0, c1_idx, o1), c011)
        np.add.at(histogram_tensor, (r1, c0_idx, o0), c100)
        np.add.at(histogram_tensor, (r1, c0_idx, o1), c101)
        np.add.at(histogram_tensor, (r1, c1_idx, o0), c110)
        np.add.at(histogram_tensor, (r1, c1_idx, o1), c111)

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()
        # Threshold and normalize
        threshold = norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(norm(descriptor_vector), float_tolerance)
        # Convert to OpenCV format
        descriptor_vector = round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return array(descriptors, dtype='float32')