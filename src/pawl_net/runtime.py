"""Portable runtime setup and reproducible data loaders."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from pawl_net.dataset import UHIDataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    """Top-level function is picklable by Windows DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; set training.device to auto or cpu")
    return device


def make_loader(config, mode):
    settings, data = config["training"], config.get("data", {})
    directory = config["paths"]["test_patch_dir" if mode == "test" else "train_patch_dir"]
    dataset = UHIDataset(
        directory,
        mode=mode,
        seed=settings["seed"],
        train_fraction=data.get("train_fraction", 0.8),
        mask_strategy=data.get("mask_strategy", "random"),
        cloud_fraction=data.get("cloud_fraction"),
    )
    generator = torch.Generator().manual_seed(settings["seed"])
    return DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=mode == "train",
        num_workers=settings.get("num_workers", 0),
        pin_memory=resolve_device(settings["device"]).type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
