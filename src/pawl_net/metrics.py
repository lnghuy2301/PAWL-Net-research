"""Streaming cloud-only errors and patch-averaged blended-image SSIM."""

import math

import numpy as np
from skimage.metrics import structural_similarity


class ReconstructionMetrics:
    """Constant-memory MAE/RMSE/R², with numerically stable target variance."""

    def __init__(self):
        self.count = 0
        self.absolute_error = self.squared_error = 0.0
        self.target_mean = self.target_m2 = 0.0
        self.ssim_sum, self.ssim_count = 0.0, 0

    def update(self, prediction, target, mask, *, valid=None, data_range=30.0):
        prediction, target, mask = map(np.asarray, (prediction, target, mask))
        if prediction.ndim == 2:
            prediction, target, mask = prediction[None], target[None], mask[None]
            valid = None if valid is None else np.asarray(valid)[None]
        if prediction.shape != target.shape or target.shape != mask.shape:
            raise ValueError("Prediction, target and mask must have matching (B, H, W) shapes")
        valid = np.ones_like(mask, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        if valid.shape != mask.shape:
            raise ValueError("Validity mask shape must match target")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("Metrics require finite predictions and targets")
        for pred, truth, observed, known in zip(prediction, target, mask, valid):
            cloud = (observed == 0) & known
            n = int(cloud.sum())
            if not n:
                continue
            y, p = truth[cloud].astype(np.float64), pred[cloud].astype(np.float64)
            error = p - y
            self.absolute_error += np.abs(error).sum()
            self.squared_error += np.square(error).sum()
            mean = float(y.mean())
            delta = mean - self.target_mean
            total = self.count + n
            self.target_m2 += np.square(y - mean).sum() + delta**2 * self.count * n / total
            self.target_mean += delta * n / total
            self.count = total
            # SSIM requires a complete reference map. Do not impute unknown truth.
            if known.all() and min(truth.shape) >= 7:
                blended = np.where(observed == 1, truth, pred)
                self.ssim_sum += structural_similarity(truth, blended, data_range=data_range)
                self.ssim_count += 1

    def compute(self):
        if not self.count:
            raise ValueError("No cloud pixels with reference targets were evaluated")
        r2 = None
        if self.count >= 2:
            r2 = (
                1 - self.squared_error / self.target_m2
                if self.target_m2 > 0
                else (1.0 if self.squared_error == 0 else 0.0)
            )
        return {
            "cloud_pixels": self.count,
            "mae_celsius": float(self.absolute_error / self.count),
            "rmse_celsius": math.sqrt(self.squared_error / self.count),
            "r2": None if r2 is None else float(r2),
            "ssim": float(self.ssim_sum / self.ssim_count) if self.ssim_count else None,
            "ssim_patches": self.ssim_count,
        }
