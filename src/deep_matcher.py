import torch
import cv2
import kornia
import numpy as np
from kornia.feature import LoFTR, DISK, LightGlue
import warnings

# Suppress some kornia warnings if necessary
warnings.filterwarnings("ignore")

class DeepMatcher:
    def __init__(self, method="loftr", device="cpu", max_size=1024):
        self.method = method.lower()
        self.device = torch.device(device)
        self.matcher = None
        self.extractor = None
        self.max_size = max_size

        if self.method == "loftr":
            self.matcher = LoFTR(pretrained="outdoor").to(self.device)
        elif self.method == "disk+lightglue":
            self.extractor = DISK.from_pretrained("depth").to(self.device)
            self.matcher = LightGlue(features="disk").to(self.device)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _resize_image(self, img):
        h, w = img.shape[:2]
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    def match(self, img_path1, img_path2):
        """
        Match two images and return keypoints and matches.
        Returns:
            mkpts0: (N, 2) numpy array of keypoints in image 1
            mkpts1: (N, 2) numpy array of corresponding keypoints in image 2
        """
        img1_raw = cv2.imread(img_path1)
        img2_raw = cv2.imread(img_path2)

        if img1_raw is None or img2_raw is None:
            raise ValueError("Could not load images")

        img1_raw = self._resize_image(img1_raw)
        img2_raw = self._resize_image(img2_raw)

        if self.method == "loftr":
            img1 = self._preprocess_gray(img1_raw)
            img2 = self._preprocess_gray(img2_raw)
            return self._match_loftr(img1, img2)
        elif self.method == "disk+lightglue":
            img1 = self._preprocess_rgb(img1_raw)
            img2 = self._preprocess_rgb(img2_raw)
            return self._match_disk_lightglue(img1, img2)

    def _preprocess_gray(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = kornia.image_to_tensor(img, False).float() / 255.0
        img = img.to(self.device)
        return img

    def _preprocess_rgb(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = kornia.image_to_tensor(img, False).float() / 255.0
        img = img.to(self.device)
        return img

    def _match_loftr(self, img1, img2):
        batch = {"image0": img1, "image1": img2}
        with torch.no_grad():
            out = self.matcher(batch)
        
        mkpts0 = out["keypoints0"].cpu().numpy()
        mkpts1 = out["keypoints1"].cpu().numpy()
        
        return mkpts0, mkpts1

    def _match_disk_lightglue(self, img1, img2):
        with torch.no_grad():
            features1 = self.extractor(img1, n=2048, window_size=5, score_threshold=0.0, pad_if_not_divisible=True)
            features2 = self.extractor(img2, n=2048, window_size=5, score_threshold=0.0, pad_if_not_divisible=True)
            
            h1, w1 = img1.shape[2:]
            h2, w2 = img2.shape[2:]
            
            kpts1_px = features1[0].keypoints.unsqueeze(0)
            desc1 = features1[0].descriptors.unsqueeze(0)
            kpts2_px = features2[0].keypoints.unsqueeze(0)
            desc2 = features2[0].descriptors.unsqueeze(0)
            
            kpts1_norm = kpts1_px.clone()
            kpts1_norm[:, :, 0] /= w1
            kpts1_norm[:, :, 1] /= h1
            kpts2_norm = kpts2_px.clone()
            kpts2_norm[:, :, 0] /= w2
            kpts2_norm[:, :, 1] /= h2
            
            size1 = torch.tensor([[h1, w1]], dtype=torch.int32, device=self.device)
            size2 = torch.tensor([[h2, w2]], dtype=torch.int32, device=self.device)

            out = self.matcher({
                "image0": {"keypoints": kpts1_norm, "descriptors": desc1, "image_size": size1}, 
                "image1": {"keypoints": kpts2_norm, "descriptors": desc2, "image_size": size2}
            })
            
            m0 = out["matches0"][0].cpu().numpy()
            
            valid = m0 > -1
            m0_indices = np.where(valid)[0]
            m1_indices = m0[valid]
            
            mkpts0 = kpts1_px[0].cpu().numpy()[m0_indices]
            mkpts1 = kpts2_px[0].cpu().numpy()[m1_indices]
            
            return mkpts0, mkpts1
