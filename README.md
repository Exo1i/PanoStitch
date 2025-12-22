# PanoStitch

A modular panoramic image stitching pipeline featuring both classic algorithms and modern Deep Learning integrations.

## Features

- **Feature Detection**: Custom Harris Corner Detector or SIFT.
- **Feature Description**: Custom 128D HOG-based descriptors.
- **Matching**: Vectorized Brute-force matcher with Lowe's ratio test.
- **Deep Learning**: Integration with DISK + LightGlue for robust matching.
- **Homography**: RANSAC with Direct Linear Transform (DLT).
- **Processing**: Gain compensation for exposure correction and weighted blending.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Recommended)
Stitch all images in a directory using the standard SIFT pipeline:
```bash
python panostitch.py imgs/boat/
```

### Comparing Methods

**1. Custom "From Scratch" Implementation (Harris + HOG Descriptors)**
```bash
python panostitch.py imgs/boat/ --harris
```

**2. Deep Learning Pipeline (DISK + LightGlue)**
```bash
python panostitch.py imgs/boat/ --dnn
```

## Team Members

| Name | ID | Section |
|------|----|---------|
| Hussien Mohamed | 9230345 | 1 |
| Mohamed | 9230245 | 1 |
| Amira Khaled Ahmed | 9230513 | 1 |
| Abdulrahman Medhat | 9231026 | 2 |
| Youssef Mohamed Noser | 9231026 | 2 |
