"""Random Forest baseline with sampled training pixels and streaming evaluation."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from pawl_net.config import DEFAULT_CONFIG, load_config, serializable_config
from pawl_net.metrics import ReconstructionMetrics
from pawl_net.normalization import to_celsius
from pawl_net.runtime import make_loader, seed_everything


def train_random_forest(config, checkpoint):
    if Path(checkpoint).exists():
        raise FileExistsError(f"Refusing to overwrite {checkpoint}")
    seed_everything(config["training"]["seed"])
    settings = config["random_forest"]
    sample_rate, n_features = settings["sample_rate"], settings.get("n_features", 10)
    if not 0 < sample_rate <= 1 or n_features not in {9, 10}:
        raise ValueError("RF needs 0 < sample_rate <= 1 and n_features in {9, 10}")
    rng = np.random.default_rng(config["training"]["seed"])
    features, targets = [], []
    for inputs, truth, _ in tqdm(make_loader(config, "train"), desc="Sampling RF pixels"):
        pixels = inputs.numpy().transpose(0, 2, 3, 1).reshape(-1, 10)
        values = truth.numpy().reshape(-1)
        indices = rng.choice(len(values), max(1, int(len(values) * sample_rate)), replace=False)
        features.append(pixels[indices, :n_features])
        targets.append(values[indices])
    model = RandomForestRegressor(
        n_estimators=settings["n_estimators"],
        max_depth=settings["max_depth"],
        n_jobs=settings.get("n_jobs", -1),
        random_state=config["training"]["seed"],
    )
    model.fit(np.concatenate(features), np.concatenate(targets))
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, checkpoint)
    checkpoint.with_suffix(".json").write_text(
        json.dumps(
            {"config": serializable_config(config), "training_pixels": sum(map(len, targets))},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint


def evaluate_random_forest(config, checkpoint):
    seed_everything(config["training"]["seed"])
    # joblib uses pickle: load only checkpoints from a trusted research source.
    model = joblib.load(checkpoint)
    features = model.n_features_in_
    if features not in {9, 10}:
        raise ValueError(f"Expected nine or ten RF features, got {features}")
    metrics = ReconstructionMetrics()
    loader = make_loader(config, "test")
    for inputs, targets, masks in tqdm(loader, desc="Evaluating RF"):
        batch, _, height, width = inputs.shape
        pixels = inputs.numpy().transpose(0, 2, 3, 1).reshape(-1, 10)
        prediction = model.predict(pixels[:, :features]).reshape(batch, height, width)
        metrics.update(
            to_celsius(prediction), to_celsius(targets[:, 0].numpy()), masks[:, 0].numpy()
        )
    return {
        "experiment": "random-forest",
        "checkpoint": str(Path(checkpoint).resolve()),
        "patches": len(loader.dataset),
        "n_features": int(features),
        "metrics": metrics.compute(),
        "protocol": {
            "mask_seed": config["training"]["seed"],
            "ssim_data_range_celsius": 30.0,
            "masks": "deterministic synthetic; same as deep learning evaluation",
            "targets": "legacy clipped LST, including zero-valued source NoData",
        },
        "config": serializable_config(config),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.action == "train":
        print(train_random_forest(config, args.checkpoint))
    else:
        if args.output and args.output.exists():
            parser.error(f"Report already exists: {args.output}")
        result = evaluate_random_forest(config, args.checkpoint)
        rendered = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)


if __name__ == "__main__":
    main()
