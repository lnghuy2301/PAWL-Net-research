"""Sliding-window inference for nine-channel scenes of arbitrary spatial size."""

import numpy as np
import torch

from pawl_net.normalization import make_input, normalize_image, to_celsius


def reconstruct_scene(model, image, mask, *, device="cpu", patch_size=128, stride=64, channels=10):
    """Return LST in °C, preserving original observed pixels exactly.

    Caller supplies a real or synthetic binary mask. This function performs no
    quantitative evaluation because real cloud gaps have no same-date reference.
    """
    if patch_size < 16 or patch_size % 16 or not 0 < stride <= patch_size:
        raise ValueError("Require patch_size divisible by 16 and 0 < stride <= patch_size")
    if channels not in {2, 10}:
        raise ValueError("channels must be 2 or 10")
    normalized = normalize_image(image)
    inputs = make_input(normalized, mask)
    height, width = image.shape[:2]
    padded_h = max(
        patch_size, patch_size + int(np.ceil(max(0, height - patch_size) / stride)) * stride
    )
    padded_w = max(
        patch_size, patch_size + int(np.ceil(max(0, width - patch_size) / stride)) * stride
    )
    padded = np.pad(inputs, ((0, padded_h - height), (0, padded_w - width), (0, 0)), mode="edge")
    totals = np.zeros((padded_h, padded_w), dtype=np.float64)
    counts = np.zeros_like(totals)
    was_training = model.training
    model.to(device).eval()
    try:
        with torch.inference_mode():
            for y in range(0, padded_h - patch_size + 1, stride):
                for x in range(0, padded_w - patch_size + 1, stride):
                    patch = padded[y : y + patch_size, x : x + patch_size]
                    tensor = torch.from_numpy(patch.copy()).permute(2, 0, 1)[None].to(device)
                    if channels == 2:
                        tensor = tensor[:, [0, 9]]
                    prediction = model(tensor)[0, 0].cpu().numpy()
                    totals[y : y + patch_size, x : x + patch_size] += prediction
                    counts[y : y + patch_size, x : x + patch_size] += 1
    finally:
        model.train(was_training)
    prediction_celsius = to_celsius((totals / counts)[:height, :width])
    return np.where(np.asarray(mask) == 1, image[:, :, 0] - 273.15, prediction_celsius)
