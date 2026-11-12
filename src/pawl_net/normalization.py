"""Physical scaling shared by datasets, notebooks and inference."""

import numpy as np

LST_MIN, LST_MAX = 293.0, 323.0
ELEV_MAX, DIST_MAX = 50.0, 5000.0
CHANNELS = (
    "LST_Kelvin",
    "NDVI",
    "NDBI",
    "MNDWI",
    "Albedo",
    "Emissivity",
    "Dist_Water",
    "Elevation",
    "Population",
)


def normalize_image(image):
    """Return float32 HWC data without modifying the input.

    Scaling matches the original implementation, including clipping invalid LST
    to LST_MIN. NoData validity must therefore be tracked BEFORE normalization.
    """
    image = np.array(image, dtype=np.float32, copy=True)
    if image.ndim != 3 or image.shape[-1] != len(CHANNELS):
        raise ValueError(f"Expected (H, W, 9) data, got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("Patch contains NaN or infinite values; clean NoData before training")
    if np.any(image[:, :, 8] < 0):
        raise ValueError("Population density cannot be negative")
    image[:, :, 0] = (np.clip(image[:, :, 0], LST_MIN, LST_MAX) - LST_MIN) / (LST_MAX - LST_MIN)
    image[:, :, 1:4] = (image[:, :, 1:4] + 1) / 2.0
    image[:, :, 4] = np.clip(image[:, :, 4], 0, 1)
    image[:, :, 5] = np.clip(image[:, :, 5], 0.9, 1.0)
    image[:, :, 6] = np.clip(image[:, :, 6] / DIST_MAX, 0, 1)
    image[:, :, 7] = np.clip(image[:, :, 7] / ELEV_MAX, 0, 1)
    image[:, :, 8] = np.log1p(image[:, :, 8]) / 10.0
    return image


def to_celsius(normalized):
    return normalized * (LST_MAX - LST_MIN) + (LST_MIN - 273.15)


def make_input(normalized, mask):
    """Mask the first six channels; retain the three spatial channels."""
    mask = np.asarray(mask, dtype=np.float32)
    if normalized.ndim != 3 or normalized.shape[-1] != 9 or mask.shape != normalized.shape[:2]:
        raise ValueError("Expected nine data channels and a spatially matching mask")
    if not np.isin(mask, [0.0, 1.0]).all():
        raise ValueError("Mask must be binary (0=cloud, 1=observed)")
    masked = normalized.copy()
    masked[:, :, :6] *= mask[:, :, None]
    return np.concatenate([masked, mask[:, :, None]], axis=-1)
