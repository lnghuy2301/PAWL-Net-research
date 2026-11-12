"""Nine-channel NumPy patches and on-the-fly synthetic cloud masking."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pawl_net.masks import create_cloud_mask
from pawl_net.normalization import make_input, normalize_image


class UHIDataset(Dataset):
    def __init__(
        self,
        patch_dir,
        mode="train",
        *,
        seed=42,
        train_fraction=0.8,
        mask_strategy="random",
        cloud_fraction=None,
    ):
        if mode not in {"train", "val", "test"}:
            raise ValueError("mode must be train, val or test")
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must be between zero and one")
        files = sorted(Path(patch_dir).glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy patches in {patch_dir}; see docs/data.md")
        split = int(len(files) * train_fraction)
        self.files = files[:split] if mode == "train" else files[split:] if mode == "val" else files
        if not self.files:
            raise ValueError(f"The {mode} partition is empty ({len(files)} source patches)")
        self.mode, self.seed = mode, int(seed)
        self.mask_strategy, self.cloud_fraction = mask_strategy, cloud_fraction

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        try:
            data = np.load(path, allow_pickle=False)
            norm = normalize_image(data)
        except (OSError, ValueError, EOFError) as exc:
            raise ValueError(f"Cannot load patch {path}: {exc}") from exc
        # Stable per-patch evaluation masks; training uses worker-seeded NumPy RNG.
        rng = np.random.default_rng(
            np.random.randint(0, 2**32, dtype=np.uint32)
            if self.mode == "train"
            else self.seed + index
        )
        mask = create_cloud_mask(
            *data.shape[:2],
            rng=rng,
            strategy=self.mask_strategy,
            cloud_fraction=self.cloud_fraction,
        )
        inputs = torch.from_numpy(make_input(norm, mask)).permute(2, 0, 1)
        target = torch.from_numpy(norm[:, :, 0].copy()).unsqueeze(0)
        return inputs, target, torch.from_numpy(mask).unsqueeze(0)
