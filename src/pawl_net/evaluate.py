"""Evaluate checkpoint weights using deterministic per-patch synthetic clouds."""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from pawl_net.checkpoints import load_model_weights
from pawl_net.config import DEFAULT_CONFIG, load_config, serializable_config
from pawl_net.experiments import EXPERIMENTS, build_model, select_channels
from pawl_net.metrics import ReconstructionMetrics
from pawl_net.normalization import to_celsius
from pawl_net.runtime import make_loader, resolve_device, seed_everything


def evaluate(config, experiment_name="pawl", *, checkpoint=None):
    seed_everything(config["training"]["seed"])
    device = resolve_device(config["training"]["device"])
    experiment = EXPERIMENTS[experiment_name]
    checkpoint = (
        Path(checkpoint) if checkpoint else config["paths"]["ckpt_dir"] / experiment.best_checkpoint
    )
    model = build_model(experiment, config, pretrained=False).to(device)
    load_model_weights(model, checkpoint, device)
    model.eval()
    loader = make_loader(config, "test")
    metrics = ReconstructionMetrics()
    with torch.inference_mode():
        for inputs, targets, masks in tqdm(loader, desc=f"Evaluating {experiment_name}"):
            prediction = model(select_channels(inputs, experiment).to(device))[:, 0].cpu().numpy()
            metrics.update(
                to_celsius(prediction), to_celsius(targets[:, 0].numpy()), masks[:, 0].numpy()
            )
    return {
        "experiment": experiment_name,
        "checkpoint": str(checkpoint.resolve()),
        "patches": len(loader.dataset),
        "metrics": metrics.compute(),
        "protocol": {
            "mask_seed": config["training"]["seed"],
            "ssim_data_range_celsius": 30.0,
            "ssim_reference": "full patch, observed ground truth blended with cloud predictions",
            "targets": "legacy clipped LST, including zero-valued source NoData",
            "masks": "synthetic; deterministic by sorted patch index",
        },
        "config": serializable_config(config),
        "torch_version": str(torch.__version__),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="pawl")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report (refuses to overwrite)")
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error(f"Report already exists: {args.output}")
    result = evaluate(load_config(args.config), args.experiment, checkpoint=args.checkpoint)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
