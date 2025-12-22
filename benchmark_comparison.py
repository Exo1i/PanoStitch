import os
import cv2
import time
import json
import torch
import numpy as np
import argparse
from pathlib import Path
from src.deep_matcher import DeepMatcher

# Standard SIFT Baseline
class SIFTMatcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def match(self, img_path1, img_path2):
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return np.array([]), np.array([])

        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        if not kp1 or not kp2 or des1 is None or des2 is None:
            return np.array([]), np.array([])
            
        # KNN match with ratio test
        raw_matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in raw_matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        mkpts0 = np.array([kp1[m.queryIdx].pt for m in good_matches])
        mkpts1 = np.array([kp2[m.trainIdx].pt for m in good_matches])
        
        return mkpts0, mkpts1

def compute_inliers(mkpts0, mkpts1, threshold=3.0):
    if len(mkpts0) < 4:
        return 0, 0.0, None
        
    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, threshold)
    if mask is None:
        return 0, 0.0, None
        
    inliers = int(mask.sum())
    return inliers, H, mask

def draw_matches(img1_path, img2_path, mkpts0, mkpts1, mask, out_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return

    # Draw matches
    # Create a new image side-by-side
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    
    for i in range(len(mkpts0)):
        if mask is not None and mask[i]:
            pt1 = (int(mkpts0[i][0]), int(mkpts0[i][1]))
            pt2 = (int(mkpts1[i][0]) + w1, int(mkpts1[i][1]))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(vis, pt1, 2, (0, 255, 0), -1)
            cv2.circle(vis, pt2, 2, (0, 255, 0), -1)
            
    cv2.imwrite(str(out_path), vis)

def run_benchmark(image_dir, output_dir):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Identify scenes (subdirectories) or just a single scene
    scenes = [d for d in image_dir.iterdir() if d.is_dir()]
    if not scenes:
        scenes = [image_dir]
        
    methods = ["SIFT", "LoFTR", "DISK+LightGlue"]
    results = []
    
    print("Initializing models...")
    # Initialize matchers
    matchers = {}
    for m in methods:
        if m == "SIFT":
            matchers[m] = SIFTMatcher()
        else:
            matchers[m] = DeepMatcher(method=m)
            
    for scene in scenes:
        print(f"\nProcessing Scene: {scene.name}")
        images = sorted([str(p) for p in scene.glob("*") if p.suffix.lower() in {'.jpg', '.png', '.jpeg'}])
        
        if len(images) < 2:
            print(f"Skipping {scene.name}: Need at least 2 images")
            continue

        pairs = []
        if len(images) <= 6:
            import itertools
            pairs = list(itertools.combinations(images, 2))
        else:
            # Sequential pairs
            for i in range(len(images) - 1):
                pairs.append((images[i], images[i+1]))
            
        print(f"Benchmarking {len(pairs)} pairs in {scene.name}...")

        for img1_path, img2_path in pairs:
            pair_name = f"{scene.name}_{Path(img1_path).stem}_vs_{Path(img2_path).stem}"
            print(f"  Processing Pair: {pair_name}")
            
            for m_name in methods:
                matcher = matchers[m_name]
                
                start_time = time.time()
                try:
                    mkpts0, mkpts1 = matcher.match(img1_path, img2_path)
                    match_time = time.time() - start_time
                    
                    num_putative = len(mkpts0)
                    inliers, H, mask = compute_inliers(mkpts0, mkpts1)
                    
                    inlier_ratio = inliers / num_putative if num_putative > 0 else 0
                    
                    print(f"    [{m_name}] Time: {match_time:.3f}s, Matches: {num_putative}, Inliers: {inliers} ({inlier_ratio:.2%})")
                    
                    results.append({
                        "method": m_name,
                        "scene": scene.name,
                        "pair": pair_name,
                        "time_seconds": match_time,
                        "putative_matches": num_putative,
                        "inliers": inliers,
                        "inlier_ratio": inlier_ratio
                    })
                    
                    # Visualize (save only if inliers > 0)
                    if inliers > 0:
                        vis_name = f"matches_{m_name.replace('+','_')}_{pair_name}.jpg"
                        draw_matches(img1_path, img2_path, mkpts0, mkpts1, mask, output_dir / vis_name)
                        
                except Exception as e:
                    print(f"    [{m_name}] FAILED: {e}")
                    results.append({
                        "method": m_name,
                        "scene": scene.name,
                        "pair": pair_name,
                        "error": str(e)
                    })

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir / 'metrics.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="Directory of images")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()
    
    run_benchmark(args.image_dir, args.output)
