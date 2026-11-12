"""Profile local research patches without modifying or normalizing them."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pawl_net.config import DEFAULT_CONFIG, load_config
from pawl_net.normalization import CHANNELS, LST_MAX, LST_MIN


def inspect_patches(directory):
    files = sorted(Path(directory).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy patches in {directory}")
    summary = {
        "patches": len(files),
        "readable_patches": 0,
        "finite_patches": 0,
        "lst_entirely_below_293": 0,
        "lst_entirely_above_323": 0,
        "over_70_percent_source_nodata": 0,
        "zero_population_patches": 0,
        "source_nodata_pixels": 0,
        "pixels": 0,
        "invalid_files": [],
    }
    counts, shapes = Counter(), Counter()
    minimum, maximum = np.full(9, np.inf), np.full(9, -np.inf)
    for path in files:
        try:
            array = np.load(path, allow_pickle=False)
            if array.ndim != 3 or array.shape[-1] != 9:
                raise ValueError(f"Expected (H, W, 9); found {array.shape}")
        except (OSError, ValueError, EOFError) as exc:
            summary["invalid_files"].append({"file": path.name, "error": str(exc)})
            continue
        lst = array[:, :, 0]
        nodata = (lst == 0) | ~np.isfinite(lst)
        summary["readable_patches"] += 1
        summary["finite_patches"] += int(np.isfinite(array).all())
        summary["lst_entirely_below_293"] += int(np.nanmax(lst) < LST_MIN)
        summary["lst_entirely_above_323"] += int(np.nanmin(lst) > LST_MAX)
        summary["over_70_percent_source_nodata"] += int(nodata.mean() > 0.7)
        summary["zero_population_patches"] += int(np.all(array[:, :, 8] == 0))
        summary["source_nodata_pixels"] += int(nodata.sum())
        summary["pixels"] += lst.size
        minimum = np.fmin(minimum, np.nanmin(array, axis=(0, 1)))
        maximum = np.fmax(maximum, np.nanmax(array, axis=(0, 1)))
        counts[path.stem.split("_m", 1)[0]] += 1
        shapes[str(array.shape)] += 1
    summary["readable_city_year"] = dict(counts)
    summary["listed_city_year"] = dict(Counter(path.stem.split("_m", 1)[0] for path in files))
    summary["shapes"] = dict(shapes)
    summary["channel_ranges"] = {
        name: {
            "minimum": float(low) if np.isfinite(low) else None,
            "maximum": float(high) if np.isfinite(high) else None,
        }
        for name, low, high in zip(CHANNELS, minimum, maximum)
    }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--patch-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error(f"Report already exists: {args.output}")
    directory = args.patch_dir or load_config(args.config)["paths"]["test_patch_dir"]
    summary = inspect_patches(directory)
    rendered = json.dumps(summary, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
