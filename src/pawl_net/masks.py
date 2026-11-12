"""Synthetic masks: 0 is hidden and 1 is observed, as in the original code."""

import cv2
import numpy as np
from perlin_noise import PerlinNoise

STRATEGIES = ("squares", "freeform", "perlin")


def create_cloud_mask(height, width, *, rng=None, strategy="random", cloud_fraction=None):
    """Generate a mask using an explicit random generator when reproducibility matters.

    The default preserves the original choice of one morphology per patch, without
    coverage rejection. Optional cloud_fraction=(low, high) uses bounded rejection;
    it raises if no suitable mask is obtained rather than silently changing shape.
    """
    if min(height, width) < 1:
        raise ValueError("Mask dimensions must be positive")
    if strategy not in (*STRATEGIES, "random"):
        raise ValueError(f"Unknown mask strategy: {strategy}")
    if cloud_fraction is not None:
        low, high = cloud_fraction
        if not 0 < low <= high < 1:
            raise ValueError("cloud_fraction must satisfy 0 < low <= high < 1")
    rng = rng if rng is not None else np.random.default_rng()
    for _ in range(100):
        chosen = rng.choice(STRATEGIES) if strategy == "random" else strategy
        mask = _generate(height, width, rng, chosen)
        if cloud_fraction is None or low <= float(1 - mask.mean()) <= high:
            return mask
    raise RuntimeError("Could not generate a cloud mask within the requested coverage interval")


def _generate(height, width, rng, strategy):
    mask = np.ones((height, width), dtype=np.float32)
    if strategy == "squares":
        for _ in range(rng.integers(3, 10)):
            size = rng.integers(15, 45)
            cx, cy = rng.integers(width), rng.integers(height)
            y1, y2 = np.clip([cy - size // 2, cy + size // 2], 0, height)
            x1, x2 = np.clip([cx - size // 2, cx + size // 2], 0, width)
            mask[y1:y2, x1:x2] = 0
    elif strategy == "freeform":
        for _ in range(rng.integers(16)):
            x, y = int(rng.integers(width)), int(rng.integers(height))
            for _ in range(1 + rng.integers(5)):
                angle = 2 * np.pi * rng.random()
                length, brush = rng.integers(10, 40), int(rng.integers(5, 20))
                end_x, end_y = int(x + length * np.cos(angle)), int(y + length * np.sin(angle))
                cv2.line(mask, (x, y), (end_x, end_y), 0.0, brush)
                x, y = end_x, end_y
    else:
        scale = rng.uniform(20, 50)
        noise = PerlinNoise(octaves=4, seed=int(rng.integers(1, 1000)))
        field = np.array(
            [[noise([i / scale, j / scale]) for j in range(width)] for i in range(height)]
        )
        if field.max() > field.min():
            field = (field - field.min()) / (field.max() - field.min())
        mask = (field >= 0.4).astype(np.float32)
    return mask
