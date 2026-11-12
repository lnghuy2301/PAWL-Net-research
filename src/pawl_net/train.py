"""Shared training entry point for PAWL-Net, Swin and U-Net ablations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pawl_net.config import DEFAULT_CONFIG, load_config, serializable_config
from pawl_net.experiments import EXPERIMENTS, build_model, select_channels
from pawl_net.losses import AutomaticWeightedLoss, CharbonnierLoss, EdgeLoss, region_mean
from pawl_net.runtime import make_loader, resolve_device, seed_everything


def train(config, experiment_name="pawl", *, run_name=None, resume=None):
    experiment = EXPERIMENTS[experiment_name]
    settings = config["training"]
    seed_everything(settings["seed"])
    device = resolve_device(settings["device"])
    run_name = run_name or experiment_name
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a single directory name")
    run_dir = config["paths"]["ckpt_dir"] / "runs" / run_name
    log_dir = config["paths"]["log_dir"] / run_name
    best_path, last_path = (
        run_dir / experiment.best_checkpoint,
        run_dir / experiment.last_checkpoint,
    )
    if (
        best_path.exists() or last_path.exists() or (log_dir / "history.jsonl").exists()
    ) and not resume:
        raise FileExistsError(
            f"Run {run_name!r} already exists; use --resume or another --run-name"
        )
    train_loader, val_loader = make_loader(config, "train"), make_loader(config, "val")
    pretrained = bool(config["model"].get("pretrained", True)) and resume is None
    model = build_model(experiment, config, pretrained=pretrained).to(device)
    pixel = CharbonnierLoss(settings.get("charbonnier_eps", 1e-6)).to(device)
    edge = EdgeLoss(settings.get("charbonnier_eps", 1e-6)).to(device)
    auto_loss = AutomaticWeightedLoss().to(device)
    groups = [{"params": model.parameters()}]
    if experiment.loss == "auto":
        groups.append({"params": auto_loss.parameters(), "lr": settings.get("loss_lr", 1e-3)})
    optimizer = torch.optim.AdamW(groups, lr=settings["lr"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    use_amp = bool(settings.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best_val, counter = 0, float("inf"), 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        if "model_state_dict" not in checkpoint or "optimizer_state_dict" not in checkpoint:
            raise ValueError("Resume requires a last checkpoint, not a raw best state dictionary")
        if checkpoint.get("experiment", experiment_name) != experiment_name:
            raise ValueError("Checkpoint experiment does not match --experiment")
        if "config" in checkpoint:
            previous = checkpoint["config"]
            for key in (
                "seed",
                "batch_size",
                "num_workers",
                "amp",
                "loss_region",
                "charbonnier_eps",
            ):
                if previous["training"].get(key) != settings.get(key):
                    raise ValueError(f"Resume changes training.{key}; create a new run instead")
            if previous.get("data") != config.get("data"):
                raise ValueError("Resume changes the data protocol; create a new run instead")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "loss_weights" in checkpoint:
            auto_loss.load_state_dict(checkpoint["loss_weights"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch, best_val = checkpoint["epoch"] + 1, checkpoint["best_val_loss"]
        counter = checkpoint.get("early_stop_counter", 0)
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            numpy_state = checkpoint["numpy_rng_state"]
            np.random.set_state(
                (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
            )
            train_loader.generator.set_state(checkpoint["loader_rng_state"].cpu())
            val_loader.generator.set_state(checkpoint["val_loader_rng_state"].cpu())
            if device.type == "cuda" and "cuda_rng_state" in checkpoint:
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in checkpoint["cuda_rng_state"]]
                )
    if start_epoch >= settings["epochs"] or counter >= settings["patience"]:
        print("Checkpoint has already reached the epoch or early stopping limit.")
        return best_path
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, settings["epochs"]):
        model.train()
        loss_sum, samples = 0.0, 0
        for inputs, targets, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs = select_channels(inputs, experiment).to(device)
            targets, masks = targets.to(device), masks.to(device)
            region = 1 - masks if settings.get("loss_region", "full") == "cloud" else None
            if region is not None and not torch.any(region):
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = (
                    auto_loss(pixel(outputs, targets, region), edge(outputs, targets, region))
                    if (experiment.loss == "auto")
                    else region_mean((outputs - targets).abs(), region)
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(inputs)
            samples += len(inputs)
        if not samples:
            raise ValueError("No cloud training pixels in this epoch; review mask settings")
        model.eval()
        val_sum, val_samples = 0.0, 0
        with torch.inference_mode():
            for inputs, targets, _ in val_loader:
                inputs, targets = select_channels(inputs, experiment).to(device), targets.to(device)
                val_sum += F.l1_loss(model(inputs), targets).item() * len(inputs)
                val_samples += len(inputs)
        val_loss = val_sum / val_samples
        if not np.isfinite(val_loss):
            raise FloatingPointError("Non-finite validation loss")
        scheduler.step()
        if val_loss < best_val:
            best_val, counter = val_loss, 0
            torch.save(model.state_dict(), best_path)
        else:
            counter += 1
        numpy_state = np.random.get_state()
        state = {
            "epoch": epoch,
            "experiment": experiment_name,
            "model_state_dict": model.state_dict(),
            "loss_weights": auto_loss.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val,
            "early_stop_counter": counter,
            "config": serializable_config(config),
            "torch_version": str(torch.__version__),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "loader_rng_state": train_loader.generator.get_state(),
            "val_loader_rng_state": val_loader.generator.get_state(),
        }
        if device.type == "cuda":
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        torch.save(state, last_path)
        row = {"epoch": epoch + 1, "train_loss": loss_sum / samples, "val_l1": val_loss}
        if experiment.loss == "auto":
            row["log_variances"] = auto_loss.params.detach().cpu().tolist()
        with (log_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(f"Validation L1: {val_loss:.6f}; best: {best_val:.6f}")
        if counter >= settings["patience"]:
            print("Early stopping.")
            break
    return best_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="pawl")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path, help="Explicit last checkpoint to resume")
    args = parser.parse_args(argv)
    path = train(
        load_config(args.config), args.experiment, run_name=args.run_name, resume=args.resume
    )
    print(f"Best checkpoint: {path}")


if __name__ == "__main__":
    main()
