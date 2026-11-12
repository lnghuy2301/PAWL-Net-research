"""Tile annual (12, H, W, 9) arrays into temporally separated NumPy patches."""

import argparse
import json
from pathlib import Path

import numpy as np

from pawl_net.config import DEFAULT_CONFIG, load_config


def create_patches(source, output_dir, *, patch_size=128, stride=64, min_valid_fraction=0.1):
    source, output_dir = Path(source), Path(output_dir)
    if patch_size < 1 or stride < 1 or stride > patch_size:
        raise ValueError("Require 0 < stride <= patch_size")
    if not 0 <= min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in [0, 1]")
    try:
        city, year = source.stem.rsplit("_", 1)
        int(year)
    except ValueError as exc:
        raise ValueError("Annual array filename must be City_YYYY.npy") from exc
    data = np.load(source, mmap_mode="r", allow_pickle=False)
    if data.ndim != 4 or data.shape[0] != 12 or data.shape[-1] != 9:
        raise ValueError(f"Expected (12, H, W, 9) array: {source} has {data.shape}")
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"saved": 0, "existing": 0, "rejected": 0}
    for month in range(12):
        for y in range(0, data.shape[1] - patch_size + 1, stride):
            for x in range(0, data.shape[2] - patch_size + 1, stride):
                destination = output_dir / f"{city}_{year}_m{month + 1:02d}_y{y}_x{x}.npy"
                if destination.exists():
                    counts["existing"] += 1
                    continue
                patch = np.array(
                    data[month, y : y + patch_size, x : x + patch_size], dtype=np.float32
                )
                lst = patch[:, :, 0]
                if not np.isfinite(patch).all() or np.mean(lst > 0) < min_valid_fraction:
                    counts["rejected"] += 1
                    continue
                np.save(destination, patch, allow_pickle=False)
                counts["saved"] += 1
    return counts


def prepare_directory(input_dir, config, *, patch_size=128, stride=64):
    files = sorted(Path(input_dir).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No annual arrays in {input_dir}")
    summary = {}
    for source in files:
        try:
            year = int(source.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Expected City_YYYY.npy, got {source.name}") from exc
        if not 2015 <= year <= 2024:
            raise ValueError(f"Year {year} outside the study period 2015–2024")
        split = "train_patch_dir" if year <= 2022 else "test_patch_dir"
        summary[source.name] = create_patches(
            source, config["paths"][split], patch_size=patch_size, stride=stride
        )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    summary = prepare_directory(
        args.input_dir or config["project_root"] / "data" / "raw",
        config,
        patch_size=args.patch_size,
        stride=args.stride,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
