# Image Processing Project

## Phase 1 - Proposal

### Team Members

| Name                  | ID      | Section |
| --------------------- | ------- | ------- |
| Hussien Mohamed       | 9230345 | 1       |
| Mohamed               | 9230245 | 1       |
| Amira Khaled Ahmed    | 9230513 | 1       |
| Abdulrahman Medhat    | 9231026 | 2       |
| Youssef Mohamed Noser | 9231026 | 2       |

---

## Panoramic Image Stitching

### Project Idea and Need

#### The Problem

Capturing wide-angle scenes requires expensive specialized equipment or taking multiple photos that need manual alignment. Current solutions are either costly or time-consuming.

#### Our Solution

Our proposed system automatically generates panoramas by stitching multiple overlapping images into a wide-angle photograph. This is achieved through the application of classical computer vision techniques.

### Real-World Applications

- **Virtual Tours:** Real estate, museums, and tourism
- **Surveillance Systems:** Wide field-of-view monitoring
- **Medical Imaging:** Comprehensive views from multiple scans
- **Photography:** Creating immersive visual experiences
- **Robotics:** Environmental mapping for autonomous vehicles

#### Why This Matters

Our system makes panoramic photography accessible to everyone by using standard camera images instead of expensive wide-angle lenses, while automating the tedious manual alignment process.

---

### Modules Overview

| **Block Name**            | **Input**                            | **Output**                           | **Primary Methods**                                                                                                                                |
| ------------------------- | ------------------------------------ | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Image Preprocessing**   | RGB images from camera               | Normalized images                    | Convert to grayscale, optional histogram equalization                                                                                              |
| **Feature Detection**     | Preprocessed grayscale image         | List of corner keypoints (x, y)      | Harris Corner Detection [FROM SCRATCH]                                                                                                             |
| **Feature Description**   | Keypoints (x, y) and grayscale image | Feature descriptor vectors (N × 8)   | Extract 16×16 patches, Compute gradient magnitude & orientation, Histogram of Oriented Gradients (8 bins), Normalize to unit length [FROM SCRATCH] |
| **Feature Matching**      | Descriptor sets from two images      | Matched point pairs (x₁,y₁ ↔ x₂,y₂)  | Nearest Neighbor matching, Lowe's Ratio Test, RANSAC [FROM SCRATCH]                                                                                |
| **Homography Estimation** | Matched point correspondences        | 3×3 homography matrix H              | Direct Linear Transform (DLT), Build 2N×9 matrix A, Solve using SVD, Normalize H[2,2]= [FROM SCRATCH]                                              |
| **Image Warping**         | Source image + homography matrix H   | Warped image aligned to target frame | Inverse warping with H⁻¹, Bilinear interpolation, Compute output canvas size [FROM SCRATCH]                                                        |
| **Image Blending**        | Two aligned images with overlap      | Seamless panoramic image             | Detect overlap region, Linear alpha blending, Copy non-overlapping regions                                                                         |

---

### Detailed Module Descriptions

#### Module 1: Image Preprocessing

**Input:** Raw images from camera
**Method:**

- Resize images to consistent dimensions (target: 800px width)
- Normalize pixel values to [0, 1] range
  **Optional:** Color correction and brightness normalization
  **Output:** Preprocessed images ready for feature detection
  **Implementation:** Using OpenCV2 functions

---

#### Module 2: Feature Detection (Harris Corner Detector)

**Input:** Preprocessed grayscale images
**Method:**

- Compute image gradients using Sobel operators (Ix, Iy)
- Calculate structure tensor: M = [[Ixx, Ixy], [Ixy, Iyy]]
- Apply Gaussian smoothing to structure tensor
- Compute Harris response: R = det(M) - k × trace(M)²
- Threshold response and apply non-maximum suppression
  **Output:** List of keypoint coordinates (x, y) for each image
  **Implementation:** From Scratch

---

#### Module 3: Feature Description

**Input:** Keypoints from feature detection
**Method:**

- Extract 16×16 pixel patch around each keypoint
- Compute gradient magnitude and orientation
- Create Histogram of Oriented Gradients (8 bins)
- Weight histogram by gradient magnitude
- Normalize descriptor to unit length
  **Output:** Feature descriptor vectors (N × 8 dimensional)
  **Implementation:** From Scratch
  **Advantages:**
- Invariant to rotation
- Robust to illumination changes
- Compact representation

---

#### Module 4: Feature Matching

**Input:** Feature descriptors from two images
**Method:**

1. **Nearest Neighbor Matching:**

   - For each descriptor in image 1, find closest match in image 2
   - Compute Euclidean distance between descriptors

2. **Lowe's Ratio Test:**

   - Keep matches where distance to nearest neighbor / distance to second nearest < 0.8

3. **RANSAC Outlier Rejection:**

   - Randomly sample 4 point correspondences
   - Compute homography and count inliers
   - Iterate 1000 times, keep best model
   - Inlier threshold: 5 pixels
     **Output:** Robust set of corresponding point pairs
     **Implementation:** From Scratch

---

#### Module 5: Homography Estimation (DLT)

**Input:** Matched point correspondences
**Method:**

- Direct Linear Transform (DLT) algorithm
  **Output:** 3×3 homography transformation matrix
  **Implementation:** From Scratch

---

#### Module 6: Image Warping

**Input:** Source image and homography matrix
**Output:** Warped and aligned images in common coordinate frame
**Implementation:** From Scratch
**Method:**

1. **Inverse Warping:**

   - For each pixel in output image, apply inverse homography to find source coordinates
   - Use bilinear interpolation to sample source image

2. **Bilinear Interpolation:**

   - Find four neighboring pixels
   - Compute weighted average based on distance

3. **Canvas Size Calculation:**

   - Transform corner points to find bounding box
   - Apply translation to handle negative coordinates

---

#### Module 7: Image Blending

**Input:** Aligned and warped images
**Method:**

1. Detect overlap region between images
2. Apply linear feathering in overlap area
3. Copy non-overlapping regions directly
   **Output:** Final seamless panoramic image
   **Implementation:** From Scratch (basic version)

---

### Non-Primitive Functions (Library Usage)

#### Needed (Won’t be implemented):

- Image I/O: `cv2.imread()`, `cv2.imwrite()`
- Color conversion: `cv2.cvtColor()`
- Array operations: NumPy functions (`np.dot`, `np.linalg.norm`, etc.)
- Matrix operations: `np.linalg.svd()`, `np.linalg.inv()`
- `cv2.resize()` for preprocessing
- `cv2.GaussianBlur()` for smoothing (may implement from scratch if required)
- Interpolation helpers if bilinear implementation is too slow
- `cv2.warpPerspective()` for image warping when combining multiple images into a panorama.

#### Will Implement:

- Feature detectors: `cv2.SIFT`, `cv2.ORB`, etc.
- Feature matching: `cv2.BFMatcher`, `cv2.FlannBasedMatcher`
- Homography: `cv2.findHomography()`

---

### Pretrained DNN for Comparison

We will compare our implementation with the pretrained DNNs:

- **SuperPoint:** Deep learning feature detector and descriptor
- **SuperGlue:** Neural network for feature matching
- **ORB (as baseline):** Fast feature detector

#### Comparison Metrics

We will compare our implementation with the pretrained DNN on:

1. Number of detected features
2. Number of correct matches
3. Number of inliers after RANSAC
4. Processing time

---

### Primary References

1. **Brown, M., & Lowe, D. G. (2007).** "Automatic Panoramic Image Stitching using Invariant Features." _International Journal of Computer Vision_, 74(1), 59-73.

   - Core paper on panorama stitching
   - Describes complete pipeline
   - Provides insights on blending techniques

2. **Harris, C., & Stephens, M. (1988).** "A Combined Corner and Edge Detector." _Alvey Vision Conference_, 15(50), 10-5244.

   - Original Harris corner detector paper
   - Mathematical derivation of corner response
   - Parameter selection guidelines

3. **Lowe, D. G. (2004).** "Distinctive Image Features from Scale-Invariant Keypoints." _International Journal of Computer Vision_, 60(2), 91-110.

   - SIFT descriptor (inspiration for our descriptor)
   - Ratio test for matching
   - Feature detection principles
