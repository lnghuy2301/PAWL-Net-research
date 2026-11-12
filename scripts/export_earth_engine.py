"""Export precomputed Earth Engine annual assets; upstream asset creation is external."""

import argparse
import os
from pathlib import Path

from pawl_net.config import PROJECT_ROOT
from pawl_net.normalization import CHANNELS


def export_annual_assets(project, asset_root, roi_asset, years, output_dir, *, authenticate=False):
    import ee
    import geemap
    import numpy as np

    if authenticate:
        ee.Authenticate()
    ee.Initialize(project=project)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = ee.FeatureCollection(roi_asset).getInfo()["features"]
    if not features:
        raise ValueError("ROI asset contains no features")
    for year in years:
        if not 2015 <= year <= 2024:
            raise ValueError("Study years are 2015–2024")
        prefix = "Train" if year <= 2022 else "Val"
        annual = ee.Image(f"{asset_root.rstrip('/')}/{prefix}_L8_Year_{year}")
        for feature in features:
            city = feature["properties"]["city_name"]
            if Path(city).name != city or city in {".", ".."}:
                raise ValueError("city_name must be a single filename component")
            destination = output_dir / f"{city}_{year}.npy"
            if destination.exists():
                print(f"Skipping existing array: {destination.name}")
                continue
            region = ee.Geometry(feature["geometry"])
            months = []
            for month in range(1, 13):
                bands = [f"y{year}_m{month:02d}_{channel}" for channel in CHANNELS]
                data = geemap.ee_to_numpy(annual.select(bands).unmask(0), region=region, scale=30)
                if data is None or data.ndim != 3 or data.shape[-1] != len(CHANNELS):
                    raise ValueError(f"Invalid export for {city}, {year}-{month:02d}")
                months.append(data.astype(np.float32))
            np.save(destination, np.stack(months), allow_pickle=False)
            print(f"Saved {destination.name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("GEE_PROJECT_ID"))
    parser.add_argument("--asset-root", default=os.environ.get("GEE_ASSET_ROOT"))
    parser.add_argument("--roi-asset", default=os.environ.get("GEE_ROI_ASSET"))
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2015, 2025)))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--authenticate", action="store_true")
    args = parser.parse_args(argv)
    if not args.project or not args.asset_root:
        parser.error("Set --project and --asset-root, or GEE_PROJECT_ID and GEE_ASSET_ROOT")
    export_annual_assets(
        args.project,
        args.asset_root,
        args.roi_asset or f"{args.asset_root.rstrip('/')}/00_Study_ROIs",
        args.years,
        args.output_dir,
        authenticate=args.authenticate,
    )


if __name__ == "__main__":
    main()
