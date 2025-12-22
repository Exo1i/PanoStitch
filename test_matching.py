"""
Test script to verify Module 4 (Feature Matching) implementations.
Compares OpenCV baseline with custom vectorized implementation.
"""

import sys
import time
import numpy as np
from pathlib import Path
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.image import Image
from src.matching import MultiImageMatches, PairMatch


def compare_implementations(img_a: Image, img_b: Image, ratio: float = 0.75):
    """Compare OpenCV and vectorized implementations."""
    
    matcher = MultiImageMatches([img_a, img_b], ratio=ratio)
    
    print(f"\n{'='*60}")
    print(f"Comparing Feature Matching Implementations")
    print(f"{'='*60}")
    print(f"Image A: {img_a.path}")
    print(f"  - Keypoints: {len(img_a.keypoints) if img_a.keypoints else 0}")
    print(f"  - Descriptors shape: {img_a.features.shape if img_a.features is not None else None}")
    print(f"Image B: {img_b.path}")
    print(f"  - Keypoints: {len(img_b.keypoints) if img_b.keypoints else 0}")
    print(f"  - Descriptors shape: {img_b.features.shape if img_b.features is not None else None}")
    print(f"Ratio threshold: {ratio}")
    print(f"{'='*60}\n")
    
    # Test OpenCV implementation
    print("Testing OpenCV implementation...")
    start = time.time()
    opencv_matches = matcher._compute_matches_opencv(img_a, img_b)
    opencv_time = time.time() - start
    print(f"  ✓ OpenCV: {len(opencv_matches)} matches in {opencv_time:.4f}s")
    
    # Test Vectorized implementation
    print("Testing Vectorized implementation...")
    start = time.time()
    vectorized_matches = matcher._compute_matches_vectorized(img_a, img_b)
    vectorized_time = time.time() - start
    print(f"  ✓ Vectorized: {len(vectorized_matches)} matches in {vectorized_time:.4f}s")
    
    # Test Loop implementation (only if small number of features)
    if img_a.features is not None and len(img_a.features) < 500:
        print("Testing Loop implementation (slow)...")
        start = time.time()
        loop_matches = matcher._compute_matches_loop(img_a, img_b)
        loop_time = time.time() - start
        print(f"  ✓ Loop: {len(loop_matches)} matches in {loop_time:.4f}s")
    else:
        loop_matches = None
        print("  ⚠ Skipping loop implementation (too many features)")
    
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    # Compare match counts
    print(f"\nMatch Counts:")
    print(f"  OpenCV:     {len(opencv_matches)}")
    print(f"  Vectorized: {len(vectorized_matches)}")
    if loop_matches is not None:
        print(f"  Loop:       {len(loop_matches)}")
    
    # Compare queryIdx and trainIdx
    opencv_pairs = set((m.queryIdx, m.trainIdx) for m in opencv_matches)
    vectorized_pairs = set((m.queryIdx, m.trainIdx) for m in vectorized_matches)
    
    common = opencv_pairs & vectorized_pairs
    only_opencv = opencv_pairs - vectorized_pairs
    only_vectorized = vectorized_pairs - opencv_pairs
    
    print(f"\nMatch Overlap:")
    print(f"  Common matches:       {len(common)}")
    print(f"  Only in OpenCV:       {len(only_opencv)}")
    print(f"  Only in Vectorized:   {len(only_vectorized)}")
    
    # Jaccard similarity
    if len(opencv_pairs | vectorized_pairs) > 0:
        jaccard = len(common) / len(opencv_pairs | vectorized_pairs)
        print(f"  Jaccard similarity:   {jaccard:.2%}")
    
    # Compare distances for common matches
    if len(common) > 0:
        opencv_dist_map = {(m.queryIdx, m.trainIdx): m.distance for m in opencv_matches}
        vectorized_dist_map = {(m.queryIdx, m.trainIdx): m.distance for m in vectorized_matches}
        
        dist_diffs = []
        for pair in common:
            diff = abs(opencv_dist_map[pair] - vectorized_dist_map[pair])
            dist_diffs.append(diff)
        
        print(f"\nDistance Comparison (for common matches):")
        print(f"  Max distance difference:  {max(dist_diffs):.6f}")
        print(f"  Mean distance difference: {np.mean(dist_diffs):.6f}")
        print(f"  Distances match exactly:  {sum(d < 1e-5 for d in dist_diffs)}/{len(dist_diffs)}")
    
    # Analyze discrepancies
    if len(only_opencv) > 0 or len(only_vectorized) > 0:
        print(f"\n{'='*60}")
        print("DISCREPANCY ANALYSIS")
        print(f"{'='*60}")
        
        if len(only_opencv) > 0:
            print(f"\nMatches only in OpenCV ({len(only_opencv)} total):")
            # Sample a few to analyze
            for i, (q, t) in enumerate(list(only_opencv)[:5]):
                if img_a.features is not None and img_b.features is not None:
                    desc_a = img_a.features[q].astype(np.float64)
                    desc_b = img_b.features[t].astype(np.float64)
                    dist = np.sqrt(np.sum((desc_a - desc_b) ** 2))
                    print(f"  [{i}] queryIdx={q}, trainIdx={t}, computed_dist={dist:.4f}")
        
        if len(only_vectorized) > 0:
            print(f"\nMatches only in Vectorized ({len(only_vectorized)} total):")
            for i, (q, t) in enumerate(list(only_vectorized)[:5]):
                if img_a.features is not None and img_b.features is not None:
                    desc_a = img_a.features[q].astype(np.float64)
                    desc_b = img_b.features[t].astype(np.float64)
                    dist = np.sqrt(np.sum((desc_a - desc_b) ** 2))
                    print(f"  [{i}] queryIdx={q}, trainIdx={t}, computed_dist={dist:.4f}")
    
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    
    if len(common) == len(opencv_matches) == len(vectorized_matches):
        print("✅ PASS: Implementations produce identical matches!")
    elif jaccard > 0.95:
        print(f"⚠️  CLOSE: {jaccard:.1%} match overlap - minor differences")
    elif jaccard > 0.8:
        print(f"⚠️  WARNING: {jaccard:.1%} match overlap - some differences")
    else:
        print(f"❌ FAIL: Only {jaccard:.1%} match overlap - significant differences")
    
    return opencv_matches, vectorized_matches


def main():
    # Find test images
    img_dir = Path("./imgs/boat")
    if not img_dir.exists():
        img_dir = Path("./imgs/dam")
    
    if not img_dir.exists():
        print("No test images found. Please provide image directory.")
        sys.exit(1)
    
    # Load first two images
    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if len(img_files) < 2:
        print(f"Need at least 2 images in {img_dir}")
        sys.exit(1)
    
    print(f"Loading images from {img_dir}...")
    img_a = Image(str(img_files[0]))
    img_b = Image(str(img_files[1]))
    
    # Compute features (required before matching!)
    print("Computing features...")
    img_a.compute_features(use_harris=True)  # Use Harris (faster)
    img_b.compute_features(use_harris=True)
    
    # Run comparison
    compare_implementations(img_a, img_b)
    
    # Test 2: Check RANSAC inlier ratio
    print(f"\n{'='*60}")
    print("TEST 2: RANSAC Inlier Ratio")
    print(f"{'='*60}")
    
    matcher = MultiImageMatches([img_a, img_b], ratio=0.75)
    matches = matcher._compute_matches_vectorized(img_a, img_b)
    
    pair = PairMatch(img_a, img_b, matches)
    pair.compute_homography()
    
    if pair.status is not None:
        inliers = int(pair.status.sum())
        total = len(matches)
        ratio = inliers / total if total > 0 else 0
        print(f"Total matches: {total}")
        print(f"Inliers (after RANSAC): {inliers}")
        print(f"Inlier ratio: {ratio:.1%}")
        
        if ratio > 0.5:
            print("✅ PASS: Inlier ratio > 50%")
        else:
            print("❌ FAIL: Inlier ratio < 50% - possible bad matches or homography")
        
        if pair.H is not None:
            print(f"\nHomography matrix:\n{pair.H}")
    else:
        print("❌ FAIL: Could not compute homography")
    
    # Test 3: Visualize matches
    print(f"\n{'='*60}")
    print("TEST 3: Visual Match Inspection")
    print(f"{'='*60}")
    
    if img_a.keypoints and img_b.keypoints and pair.status is not None:
        # Draw only inlier matches
        inlier_matches = [m for m, s in zip(matches, pair.status.flatten()) if s]
        
        vis = cv2.drawMatches(
            img_a.image, img_a.keypoints,
            img_b.image, img_b.keypoints,
            inlier_matches[:50],  # Draw up to 50 matches
            None,
            matchColor=(0, 255, 0),  # Green for inliers
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite("debug_matches.jpg", vis)
        print(f"✓ Saved debug_matches.jpg with {len(inlier_matches[:50])} inlier matches")
        print("  → Open this file to visually verify match quality")
    else:
        print("⚠ Could not create visualization")
    
    # Test 4: Simple 2-image panorama
    print(f"\n{'='*60}")
    print("TEST 4: Simple 2-Image Panorama Stitching")
    print(f"{'='*60}")
    
    if pair.H is not None:
        # Warp image_b to image_a's coordinate system
        h_a, w_a = img_a.image.shape[:2]
        h_b, w_b = img_b.image.shape[:2]
        
        # Compute corners of image_b after transformation
        corners_b = np.array([
            [0, 0, 1],
            [w_b, 0, 1],
            [w_b, h_b, 1],
            [0, h_b, 1]
        ], dtype=np.float64).T
        
        transformed_corners = pair.H @ corners_b
        transformed_corners = transformed_corners[:2] / transformed_corners[2]
        
        # Compute canvas size
        all_corners = np.hstack([
            np.array([[0, w_a, w_a, 0], [0, 0, h_a, h_a]]),
            transformed_corners
        ])
        
        min_x, min_y = np.floor(all_corners.min(axis=1)).astype(int)
        max_x, max_y = np.ceil(all_corners.max(axis=1)).astype(int)
        
        # Translation to handle negative coordinates
        translation = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ], dtype=np.float64)
        
        canvas_w = max_x - min_x
        canvas_h = max_y - min_y
        
        print(f"Canvas size: {canvas_w} x {canvas_h}")
        
        # Warp both images
        warped_a = cv2.warpPerspective(
            img_a.image, translation, (canvas_w, canvas_h)
        )
        warped_b = cv2.warpPerspective(
            img_b.image, translation @ pair.H, (canvas_w, canvas_h)
        )
        
        # Simple blending: take non-zero pixels
        result = warped_a.copy()
        mask_b = (warped_b.sum(axis=2) > 0)
        mask_a = (warped_a.sum(axis=2) > 0)
        
        # Where only B exists, use B
        only_b = mask_b & ~mask_a
        result[only_b] = warped_b[only_b]
        
        # Where both exist, average (simple blend)
        both = mask_a & mask_b
        result[both] = ((warped_a[both].astype(np.float32) + warped_b[both].astype(np.float32)) / 2).astype(np.uint8)
        
        cv2.imwrite("debug_panorama.jpg", result)
        print(f"✓ Saved debug_panorama.jpg")
        print("  → Open this file to verify stitching quality")
        print("\n✅ All tests completed!")
    else:
        print("❌ Cannot create panorama - homography failed")


if __name__ == "__main__":
    main()
