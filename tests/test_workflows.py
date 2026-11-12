import importlib

import numpy as np
import pytest
import torch

from pawl_net.config import load_config
from pawl_net.preprocessing import create_patches, prepare_directory


def test_preprocessing_routes_temporal_holdout_and_supports_city_underscores(tmp_path):
    config = load_config()
    raw = tmp_path / "raw"
    raw.mkdir()
    annual = np.zeros((12, 20, 20, 9), dtype=np.float32)
    annual[:, :, :, 0] = 300
    for year in (2022, 2023):
        np.save(raw / f"Can_Tho_{year}.npy", annual)
    config["paths"]["train_patch_dir"] = tmp_path / "train"
    config["paths"]["test_patch_dir"] = tmp_path / "test"
    summary = prepare_directory(raw, config, patch_size=16, stride=8)
    assert summary["Can_Tho_2022.npy"]["saved"] == 12
    assert summary["Can_Tho_2023.npy"]["saved"] == 12
    assert len(list((tmp_path / "train").glob("*.npy"))) == 12
    assert all("2023" in p.name for p in (tmp_path / "test").glob("*.npy"))
    assert (
        create_patches(raw / "Can_Tho_2022.npy", tmp_path / "train", patch_size=16, stride=8)[
            "existing"
        ]
        == 12
    )


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 1, 1)

    def forward(self, tensor):
        return torch.sigmoid(self.conv(tensor))


def test_cpu_training_resume_and_evaluation(tmp_path, patch_dir, monkeypatch):
    train_module = importlib.import_module("pawl_net.train")
    evaluate_module = importlib.import_module("pawl_net.evaluate")
    for module in (train_module, evaluate_module):
        monkeypatch.setattr(module, "build_model", lambda *args, **kwargs: TinyModel())
    config = load_config()
    config["paths"].update(
        train_patch_dir=patch_dir,
        test_patch_dir=patch_dir,
        ckpt_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
    )
    config["training"].update(device="cpu", epochs=1, batch_size=2)
    config["data"]["mask_strategy"] = "squares"
    best = train_module.train(config, run_name="resumed")
    last = best.with_name("last_unet.pth")
    assert best.exists() and last.exists()
    original_bytes = best.read_bytes()
    with pytest.raises(FileExistsError):
        train_module.train(config, run_name="resumed")
    assert best.read_bytes() == original_bytes
    config["training"]["epochs"] = 2
    train_module.train(config, run_name="resumed", resume=last)
    train_module.train(config, run_name="continuous")
    resumed = torch.load(last, weights_only=True)
    continuous = torch.load(
        tmp_path / "checkpoints/runs/continuous/last_unet.pth", weights_only=True
    )
    assert resumed["epoch"] == 1
    for key, value in resumed["model_state_dict"].items():
        torch.testing.assert_close(value, continuous["model_state_dict"][key], rtol=0, atol=0)
    result = evaluate_module.evaluate(config, checkpoint=best)
    assert result["patches"] == 10
    assert result["metrics"]["cloud_pixels"] > 0


def test_random_forest_train_and_evaluate(tmp_path, patch_dir):
    from pawl_net.baselines import evaluate_random_forest, train_random_forest

    config = load_config()
    config["paths"].update(train_patch_dir=patch_dir, test_patch_dir=patch_dir)
    config["training"].update(device="cpu", batch_size=2)
    config["data"]["mask_strategy"] = "squares"
    config["random_forest"].update(n_estimators=2, max_depth=3, n_jobs=1)
    checkpoint = tmp_path / "forest.joblib"
    train_random_forest(config, checkpoint)
    assert checkpoint.with_suffix(".json").exists()
    with pytest.raises(FileExistsError):
        train_random_forest(config, checkpoint)
    result = evaluate_random_forest(config, checkpoint)
    assert result["patches"] == 10
    assert result["n_features"] == 10
    assert np.isfinite(result["metrics"]["rmse_celsius"])
