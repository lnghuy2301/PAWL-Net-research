"""Pixel, Laplacian and uncertainty-weighted objectives."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def region_mean(values, region=None):
    if region is None:
        return values.mean()
    weights = region.to(dtype=values.dtype).expand_as(values)
    # An empty cloud region contributes zero instead of a divide-by-zero error.
    return (values * weights).sum() / weights.sum().clamp_min(1)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps

    def forward(self, prediction, target, region=None):
        return region_mean(torch.sqrt((prediction - target).square() + self.eps**2), region)


class EdgeLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("k", kernel[None, None])
        self.loss = CharbonnierLoss(eps)

    def forward(self, prediction, target, region=None):
        # Convolve full maps before selecting the region, preserving spatial neighbors.
        return self.loss(
            F.conv2d(prediction, self.k, padding=1),
            F.conv2d(target, self.k, padding=1),
            region,
        )


class AutomaticWeightedLoss(nn.Module):
    """params[i] is log(sigma_i**2), not sigma_i itself."""

    def __init__(self, num_losses=2):
        super().__init__()
        if num_losses < 1:
            raise ValueError("num_losses must be positive")
        self.params = nn.Parameter(torch.zeros(num_losses))

    def forward(self, *losses):
        if len(losses) != len(self.params):
            raise ValueError("Number of losses must match the learned weights")
        return sum(
            0.5 * torch.exp(-log_variance) * loss + 0.5 * log_variance
            for log_variance, loss in zip(self.params, losses)
        )
