import torch
from torch.utils.data import Dataset
import numpy as np
import os
from glob import glob
import cv2
from perlin_noise import PerlinNoise

LST_MIN, LST_MAX = 293.0, 323.0
ELEV_MAX, DIST_MAX = 50.0, 5000.0

class UHIDataset(Dataset):
    def __init__(self, patch_dir, mode='train'):
        """ mode: 'train' (80%), 'val' (20%), 'test' (100%) """
        self.files = sorted(glob(os.path.join(patch_dir, "*.npy")))
        if len(self.files) == 0:
            print(f"⚠️ Không tìm thấy file tại {patch_dir}")

        split_idx = int(len(self.files) * 0.8)
        if mode == 'train':
            self.files = self.files[:split_idx]
        elif mode == 'val':
            self.files = self.files[split_idx:]
        # Nếu mode == 'test', giữ nguyên 100% self.files

    def __len__(self): return len(self.files)

    def normalize(self, img):
        img[:,:,0] = (np.clip(img[:,:,0], LST_MIN, LST_MAX) - LST_MIN) / (LST_MAX - LST_MIN)
        img[:,:,1:4] = (img[:,:,1:4] + 1) / 2.0
        img[:,:,4] = np.clip(img[:,:,4], 0, 1)    
        img[:,:,5] = np.clip(img[:,:,5], 0.9, 1.0) 
        img[:,:,6] = np.clip(img[:,:,6] / DIST_MAX, 0, 1) 
        img[:,:,7] = np.clip(img[:,:,7] / ELEV_MAX, 0, 1) 
        img[:,:,8] = np.log1p(img[:,:,8]) / 10.0          
        return img

    def create_cloud_mask(self, h, w):
        strategy = np.random.choice(['squares', 'freeform', 'perlin'])
        mask = np.ones((h, w), dtype=np.float32)

        if strategy == 'squares':
            for _ in range(np.random.randint(3, 10)):
                cs = np.random.randint(15, 45)
                cx, cy = np.random.randint(0, h), np.random.randint(0, w)
                y1, y2 = np.clip([cy - cs//2, cy + cs//2], 0, h)
                x1, x2 = np.clip([cx - cs//2, cx + cs//2], 0, w)
                mask[y1:y2, x1:x2] = 0.0
        elif strategy == 'freeform':
            for _ in range(np.random.randint(16)):
                start_x, start_y = np.random.randint(w), np.random.randint(h)
                for _ in range(1 + np.random.randint(5)):
                    angle = 2 * np.pi * np.random.rand()
                    length, brush_size = np.random.randint(10, 40), np.random.randint(5, 20)
                    end_x = int(start_x + length * np.cos(angle))
                    end_y = int(start_y + length * np.sin(angle))
                    cv2.line(mask, (start_x, start_y), (end_x, end_y), 0.0, brush_size)
                    start_x, start_y = end_x, end_y
        else:
            scale = np.random.uniform(20, 50)
            noise = PerlinNoise(octaves=4, seed=np.random.randint(0, 1000))
            pic = [[noise([i/scale, j/scale]) for j in range(w)] for i in range(h)]
            mask = np.array(pic)
            if mask.max() > mask.min():
                mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = (mask < 0.4).astype(np.float32) 
            mask = 1.0 - mask
            
        return mask

    def __getitem__(self, idx):
        try:
            data = np.load(self.files[idx]).astype(np.float32) 
        except:
            data = np.zeros((128, 128, 9), dtype=np.float32)

        norm_data = self.normalize(data)
        target = norm_data[:,:,0].copy()
        mask = self.create_cloud_mask(128, 128)

        input_img = norm_data.copy()
        input_img[:,:,0:6] = input_img[:,:,0:6] * mask[:,:,np.newaxis]
        final_input = np.concatenate([input_img, mask[:,:,np.newaxis]], axis=-1)

        input_tensor = torch.from_numpy(final_input).permute(2, 0, 1)
        target_tensor = torch.from_numpy(target).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return input_tensor, target_tensor, mask_tensor