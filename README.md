# PanoStitch: Panoramic Image Stitching

A modular implementation of panoramic image stitching with both automatic (OpenCV-based) and manual (from-scratch) implementations of key computer vision algorithms.

## Overview

PanoStitch creates seamless panoramas from multiple overlapping images by detecting features, matching them across images, estimating geometric transformations, and blending the results. The project is structured as a pipeline of 8 independent modules, allowing for incremental replacement of OpenCV functions with custom implementations.

## Features

- **Dual Feature Detection**: Harris corner detection or SIFT keypoints
- **Robust Feature Matching**: Brute-force matching with Lowe's ratio test
- **Homography Estimation**: RANSAC-based outlier rejection with DLT solver
- **Smart Reference Selection**: Automatically chooses central image to minimize transformation errors
- **Exposure Balancing**: Gain compensation for consistent lighting across images
- **Multi-band Blending**: Weighted blending for seamless panorama composition
- **Memory-Efficient**: Automatic canvas size optimization and image resizing

## Installation

### Requirements

- Python 3.8+
- OpenCV (cv2)
- NumPy

### Setup

```bash
pip install opencv-python numpy
```

## Usage

### Basic Usage

```bash
# Stitch specific images
python panostitch.py imgs/boat/boat1.jpg imgs/boat/boat2.jpg imgs/boat/boat3.jpg

# Stitch all images in a directory
python panostitch.py imgs/boat/
```

### Configuration

Edit `panostitch.py` to customize parameters:

```python
stitcher = PanoStitch(
    resize_size=800,        # Maximum image dimension (None for original size)
    ratio=0.75,             # Lowe's ratio test threshold
    gain_sigma_n=10.0,      # Gain compensation noise tolerance
    gain_sigma_g=0.1,       # Gain compensation gain tolerance
    use_harris=True,        # Use Harris corners (False for SIFT)
    verbose=True            # Print progress information
)
```

## Architecture

### Pipeline Overview

```
Input Images
    |
    v
[1] Image Loading
    |
    v
[2] Feature Detection (Harris/SIFT)
    |
    v
[3] Feature Description (SIFT descriptors)
    |
    v
[4] Feature Matching (Brute-force + Lowe's ratio)
    |
    v
[5] Homography Estimation (RANSAC + DLT)
    |
    v
[6] Homography Assembly (Connected components + chaining)
    |
    v
[7] Gain Compensation (Exposure balancing)
    |
    v
[8] Image Blending (Weighted composition)
    |
    v
Output Panorama
```

### Module Structure

```
src/
├── image.py              # Modules 1-3: Image loading and feature computation
├── matching.py           # Module 4: Feature matching between image pairs
├── estimators.py         # Module 5: Homography estimation (DLT + RANSAC)
├── homography.py         # Module 6: Homography chaining and assembly
├── gain_compensation.py  # Module 7: Exposure balancing
├── blending.py           # Module 8: Image warping and blending
└── stitcher.py          # Pipeline coordinator
```

## Algorithm Implementation Details

### Module 1: Image Loading

**Implementation**: Custom
**File**: `src/image.py`

- Loads images using OpenCV
- Optional resizing to reduce memory consumption
- Maintains image metadata (path, dimensions, transformations)

### Module 2: Harris Corner Detection

**Implementation**: Custom
**File**: `src/image.py::_harris_corner_detection()`

**Algorithm**:

1. Convert image to grayscale
2. Compute image gradients using Sobel operator:
   ```
   Ix = ∂I/∂x
   Iy = ∂I/∂y
   ```
3. Compute structure tensor components:

   ```
   Sxx = G * (Ix * Ix)
   Syy = G * (Iy * Iy)
   Sxy = G * (Ix * Iy)
   ```

   where G is a Gaussian kernel (σ=2.0, size=7x7)

4. Calculate Harris corner response:

   ```
   R = det(M) - k * trace(M)²
   M = [[Sxx, Sxy], [Sxy, Syy]]
   k = 0.04 (empirically determined)
   ```

5. Apply threshold: `R > threshold * R_max`
6. Non-maximum suppression in 5x5 window
7. Select top N keypoints by response strength

**Parameters**:

- `k`: 0.04 (Harris free parameter)
- `threshold`: 0.01 (relative to maximum response)
- `nms_window`: 5x5 pixels
- `max_keypoints`: 1000

**Performance**: Detects ~500-1000 corners per 800px image in ~50-100ms

### Module 3: Feature Descriptors

**Implementation**: OpenCV SIFT (custom implementation TODO)
**File**: `src/image.py::compute_features()`

Currently uses SIFT descriptors (128-dimensional) computed at Harris corner locations or SIFT keypoints.

**Planned Custom Implementation**: Implement a from-scratch SIFT-like descriptor (128-dimensional)

- Extract a 16×16 oriented patch around the keypoint (relative to dominant orientation and scale)
- Divide the patch into a 4×4 grid of 16 sub-blocks (each 4×4 pixels)
- For each sub-block compute an 8-bin histogram of gradient orientations (0°–360°), weighted by gradient magnitude
- Apply a Gaussian weighting window centered on the keypoint for spatial weighting
- Concatenate the 16 histograms into a 128-dimensional vector, normalize to unit length
- Apply value clamping (e.g., limit values > 0.2) and renormalize to improve illumination robustness

### Module 4: Feature Matching

**Implementation**: OpenCV BruteForce matcher (custom implementation TODO)
**File**: `src/matching.py::MultiImageMatches`

**Algorithm**:

1. For each image pair, compute descriptor distances
2. Find 2 nearest neighbors for each descriptor
3. Apply Lowe's ratio test:
   ```
   if distance(match1) < ratio * distance(match2):
       accept match1
   ```
4. Filter pairs with insufficient matches (minimum 10 required)

**Parameters**:

- `ratio`: 0.75 (Lowe's ratio threshold)
- `min_matches`: 10

**Optimization Strategy** (for custom implementation):

```python
# Vectorized distance computation
desc1_sq = np.sum(desc1**2, axis=1, keepdims=True)
desc2_sq = np.sum(desc2**2, axis=1, keepdims=True)
dist_matrix = np.sqrt(desc1_sq + desc2_sq.T - 2 * (desc1 @ desc2.T))

# Fast k-NN using argpartition (O(n) instead of O(n log n))
k_smallest = np.argpartition(dist_matrix, kth=1, axis=1)[:, :2]
```

### Module 5: Homography Estimation

**Implementation**: OpenCV findHomography (custom implementation TODO)
**File**: `src/estimators.py::compute_homography_from_points()`

**Current**: Wrapper around `cv2.findHomography()` using RANSAC method

**Planned Custom Implementation**:

**Direct Linear Transform (DLT)**:

1. Given point correspondences (x, y) ↔ (x', y')
2. Build matrix A (8 rows per correspondence):
   ```
   [-x, -y, -1,  0,  0,  0, x'x, x'y, x']
   [ 0,  0,  0, -x, -y, -1, y'x, y'y, y']
   ```
3. Solve Ah = 0 using SVD: A = UΣV^T
4. Solution h is last column of V
5. Reshape to 3x3 homography matrix H

**RANSAC (Random Sample Consensus)**:

1. Repeat for N iterations:
   - Randomly select 4 point pairs
   - Compute homography H using DLT
   - Count inliers: points where reprojection error < threshold
2. Select H with maximum inliers
3. Refine H using all inliers

**Parameters**:

- `ransacReprojThreshold`: 5.0 pixels
- `maxIters`: 2000
- `confidence`: 0.995

### Module 6: Homography Assembly

**Implementation**: Custom
**File**: `src/homography.py`

**Connected Components Algorithm**:

1. Build graph where nodes are images and edges are valid matches
2. Find connected components using depth-first search
3. Each component will produce a separate panorama

**Reference Frame Selection**:

- Counts connections (valid matches) for each image
- Selects most connected image as reference (identity transformation)
- Reduces maximum chain length and error accumulation

**Homography Chaining**:

1. Set reference image: H_ref = I (identity)
2. For each connected image:
   - If image A has H_A and matches to image B:
     ```
     H_B = H_A * H_AB
     ```
   - Where H_AB is pairwise homography from B to A

**Critical Improvement**: By choosing the central (most connected) image as reference rather than an arbitrary endpoint, the algorithm reduced canvas size by 3.4x and improved alignment quality.

### Module 7: Gain Compensation

**Implementation**: Custom
**File**: `src/gain_compensation.py::set_gain_compensations()`

**Algorithm**:

1. For each overlapping image pair:

   - Identify overlap region using homographies
   - Sample pixel intensities in overlap
   - Build linear equations: `gain_A * I_A = gain_B * I_B`

2. Construct coefficient matrix N and residual vector:

   ```
   N[i,i] += sum of squared intensities in image i
   N[i,j] -= sum of I_i * I_j in overlap(i,j)
   ```

3. Solve linear system separately for each color channel:

   ```
   N * gains = 0
   subject to: sum(gains) = num_images
   ```

4. Apply computed gains to normalize exposure

**Parameters**:

- `sigma_n`: 10.0 (noise tolerance)
- `sigma_g`: 0.1 (gain smoothness)

**Effect**: Balances brightness across images, preventing visible seams due to lighting differences.

### Module 8: Image Blending

**Implementation**: Custom
**File**: `src/blending.py::simple_blending()`

**Algorithm**:

1. **Global Canvas Computation**:

   - Transform all image corners using their homographies
   - Compute global bounding box: [x_min, x_max] × [y_min, y_max]
   - Calculate required canvas size
   - Apply memory-safe limits (max 12000×12000)

2. **Scaling Optimization**:

   ```python
   if width > max_width or height > max_height:
       scale = min(max_width/width, max_height/height)
       transform = scale_matrix @ offset_matrix @ H
   ```

3. **Image Warping**:

   - Warp each image to global canvas using combined transformation
   - Warp weight matrix using same transformation

4. **Weighted Blending**:

   ```python
   normalized_weights = weights / (weights + image_weights)
   panorama = new_image * (1 - normalized_weights) +
              panorama * normalized_weights
   ```

5. **Weight Accumulation**:
   - Normalize accumulated weights to [0, 1]
   - Prevents overflow and maintains proper averaging

**Weight Function**:

```python
def single_weights_matrix(size):
    h, w = size
    weights = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            dist_x = min(j, w - j - 1)
            dist_y = min(i, h - i - 1)
            weights[i, j] = min(dist_x, dist_y)

    return weights / weights.max()
```

**Canvas Size Management**:

- Original approach: Fixed 5000×4000 limit caused black output
- Dynamic approach: No limits caused 97% RAM usage (16GB)
- Final approach: 12000×12000 limit with scaling (~1.7GB memory)

## Performance Benchmarks

### Feature Detection Comparison (5 images, 800px)

| Metric             | Harris    | SIFT      |
| ------------------ | --------- | --------- |
| Valid pair matches | 5         | 7         |
| Canvas size        | 4516×1732 | 3691×1261 |
| Coverage           | 51.0%     | 58.8%     |
| Processing time    | ~10s      | ~10s      |

**Analysis**: SIFT provides more robust matching due to scale-invariance and descriptor quality, resulting in smaller canvas and better coverage. Harris corners are sparser and less repeatable across scale changes.

### Memory Usage

| Configuration           | Peak RAM | Canvas Size |
| ----------------------- | -------- | ----------- |
| No limits (5 images)    | 15.5 GB  | 77150×31250 |
| 12000px limit + scaling | 1.7 GB   | 12000×4980  |
| 800px input resize      | 1.2 GB   | 3691×1261   |

**Recommendation**: Use `resize_size=800` for 6+ images to keep memory under 2GB.

## Known Limitations

1. **Harris vs SIFT**: Harris detection produces fewer matches, especially with scale variation. SIFT is more robust for general panoramas.

2. **Linear Homography Chaining**: Sequential chaining accumulates errors. For 10+ images, bundle adjustment would provide better results.

3. **Memory Constraints**: Very large panoramas (20+ megapixels) require downsampling or tiled processing.

4. **Descriptor Implementation**: Currently using SIFT descriptors. Planned work: implement custom-from-scratch SIFT descriptors (128D) for educational purposes;

5. **Blending Quality**: Simple weighted blending is used. Multi-band blending would reduce ghosting artifacts.

## Testing

### Test Datasets

The `imgs/` directory contains sample datasets:

- `imgs/boat/`: 5 images of boats with ~30% overlap
- `imgs/dam/`: Alternative test set

### Running Tests

```bash
# Test with 2 images (basic functionality)
python panostitch.py imgs/boat/boat1.jpg imgs/boat/boat2.jpg

# Test with 5 images (full pipeline)
python panostitch.py imgs/boat/boat1.jpg imgs/boat/boat2.jpg imgs/boat/boat3.jpg imgs/boat/boat4.jpg imgs/boat/boat5.jpg

# Test all images in directory
python panostitch.py imgs/boat/
```

### Validation

Results are saved to `results/panorama_0.jpg`. Validate by checking:

- No black regions (min/max pixel values)
- Proper coverage (>50% non-zero pixels)
- Seamless blending at boundaries
- Correct aspect ratio

## Troubleshooting

### Black Output Image

**Cause**: Canvas size limits too restrictive for homography transformations
**Solution**: Increase resize_size or check homography quality

### Memory Exhaustion

**Cause**: Canvas size exceeds available RAM
**Solution**: Reduce `resize_size` to 600-800px or use fewer images

### Poor Matching

**Cause**: Insufficient features or low overlap
**Solution**:

- Use SIFT instead of Harris (`use_harris=False`)
- Ensure 30%+ overlap between adjacent images
- Reduce `ratio` threshold to 0.65 for more matches

### Stretched/Distorted Output

**Cause**: Bad homography estimates from outliers
**Solution**: Verify RANSAC threshold is appropriate (default 5.0 pixels)

## Future Improvements

1. **Custom Descriptor Implementation**: Implement a custom-from-scratch SIFT descriptor (16×16 → 4×4×8 = 128D) as an alternative to OpenCV's SIFT;
2. **Custom Matching**: Vectorized brute-force implementation for educational purposes
3. **Bundle Adjustment**: Global optimization of camera parameters
4. **Multi-band Blending**: Laplacian pyramid blending for better seam hiding
5. **GPU Acceleration**: CUDA kernels for feature matching and warping
6. **Cylindrical Projection**: Better for 360-degree panoramas

## Development Team Notes

For team members implementing custom algorithms, see `README_MODULAR.md` for detailed implementation guidelines, function signatures, and testing strategies.

## References

1. Brown, M., & Lowe, D. G. (2007). "Automatic Panoramic Image Stitching using Invariant Features". International Journal of Computer Vision, 74(1), 59-73.

2. Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints". International Journal of Computer Vision, 60(2), 91-110.

3. Harris, C., & Stephens, M. (1988). "A Combined Corner and Edge Detector". Alvey Vision Conference, 15(50), 10-5244.

4. Fischler, M. A., & Bolles, R. C. (1981). "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography". Communications of the ACM, 24(6), 381-395.

## License

See LICENSE file for details.

## Contributors

Developed as a modular educational project for computer vision coursework, demonstrating both high-level OpenCV APIs and low-level algorithm implementations.
