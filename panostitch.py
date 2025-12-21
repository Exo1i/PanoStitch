"""
PanoStitch - Panoramic Image Stitching Pipeline
A modular implementation for creating panoramas from multiple overlapping images.

Project Structure:
======================

src/
├── image.py              - Module 1: Image loading and storage
│                          - Module 2: Harris Corner Detection (TODO)
│                          - Module 3: Feature Descriptors (TODO)
│
├── matching.py           - Module 4: Feature Matching (TODO)
│                          - Implement brute-force matching
│                          - Implement Lowe's ratio test
│
├── homography.py         - Module 5 & 6: Homography Estimation and Panorama Assembly (TODO)
│                          - Implement DLT (Direct Linear Transform)
│                          - Implement RANSAC
│
├── gain_compensation.py  - Module 7: Gain Compensation
│                          - Balance exposure across images
│
├── blending.py           - Module 8: Image Blending
│                          - Combine images into final panorama
│
└── stitcher.py          - Main pipeline coordinator

Usage:
======
python panostitch.py <image1> <image2> [image3] ...
python panostitch.py <directory>

Each module has TODO comments indicating what needs to be replaced with custom implementations.
"""

import cv2
import sys
from pathlib import Path

from src.stitcher import PanoStitch


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python panostitch.py <image1> <image2> [image3] ...")
        print("   or: python panostitch.py <directory>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    # Check if input is directory or individual files
    if input_path.is_dir():
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = [
            str(f) for f in input_path.iterdir() if f.suffix.lower() in valid_extensions
        ]
    else:
        image_paths = sys.argv[1:]

    if not image_paths:
        print("Error: No valid images found!")
        sys.exit(1)

    print(f"\n=== PanoStitch - Panorama Stitching ===\n")

    # Create stitcher
    stitcher = PanoStitch(
        resize_size=800,  # Resize images to reduce memory usage and canvas size
        ratio=0.75,
        gain_sigma_n=10.0,
        gain_sigma_g=0.1,
        use_harris=False,  # Use Harris corner detection instead of SIFT
        verbose=True,
    )

    # Stitch images
    panoramas = stitcher.stitch(image_paths)

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    for i, panorama in enumerate(panoramas):
        output_path = output_dir / f"panorama_{i}.jpg"
        # Ensure proper color handling: panorama is in BGR format from OpenCV
        # cv2.imwrite expects BGR, so write directly without conversion
        success = cv2.imwrite(str(output_path), panorama)
        if success:
            print(f"✓ Saved: {output_path}")
        else:
            print(f"✗ Failed to save: {output_path}")

    print(f"\n✓ Created {len(panoramas)} panorama(s)")
