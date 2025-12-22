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
import argparse
from pathlib import Path

from src.stitcher import PanoStitch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PanoStitch - Panoramic Image Stitching Pipeline")
    parser.add_argument("input", nargs="+", help="Image files or directory containing images")
    parser.add_argument("--dnn", action="store_true", help="Use DISK+LightGlue deep learning matcher instead of SIFT")
    parser.add_argument("--harris", action="store_true", help="Use Harris corner detection (only without --dnn)")
    parser.add_argument("--no-cylindrical", action="store_true", help="Disable cylindrical warping (use standard homography)")
    parser.add_argument("--focal-length", "-f", type=float, default=1200.0, help="Focal length for cylindrical warping (default: 1200.0)")
    parser.add_argument("--resize", type=int, default=800, help="Max dimension to resize images (default: 800)")
    args = parser.parse_args()

    input_path = Path(args.input[0])

    # Check if input is directory or individual files
    if len(args.input) == 1 and input_path.is_dir():
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = [
            str(f) for f in input_path.iterdir() if f.suffix.lower() in valid_extensions
        ]
    else:
        image_paths = args.input

    if not image_paths:
        print("Error: No valid images found!")
        sys.exit(1)

    print(f"\n=== PanoStitch - Panorama Stitching ===\n")
    
    matcher_name = "DISK+LightGlue (DNN)" if args.dnn else ("Harris" if args.harris else "SIFT")
    print(f"Using matcher: {matcher_name}")

    # Create stitcher
    stitcher = PanoStitch(
        resize_size=args.resize,
        ratio=0.75,
        gain_sigma_n=10.0,
        gain_sigma_g=0.1,
        use_harris=args.harris and not args.dnn,
        use_dnn=args.dnn,
        verbose=True,
        focal_length=args.focal_length,
        use_cylindrical=not args.no_cylindrical,
    )

    # Stitch images and get source directory
    panoramas, source_dir = stitcher.stitch(image_paths)

    # Save results inside the source folder
    if source_dir:
        output_dir = source_dir
    else:
        output_dir = Path("results")
    
    output_dir.mkdir(exist_ok=True, parents=True)

    suffix = "_dnn" if args.dnn else ""
    for i, panorama in enumerate(panoramas):
        output_path = output_dir / f"panorama_{i}{suffix}.jpg"
        success = cv2.imwrite(str(output_path), panorama)
        if success:
            print(f"✓ Saved: {output_path}")
        else:
            print(f"✗ Failed to save: {output_path}")

    print(f"\n✓ Created {len(panoramas)} panorama(s)")

