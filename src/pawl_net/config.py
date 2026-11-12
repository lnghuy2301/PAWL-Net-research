"""Load YAML settings with paths anchored to the repository, independent of cwd."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path=None):
    """Resolve project_root relative to YAML; resolve all paths relative to that root."""
    config_path = Path(path or DEFAULT_CONFIG).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {config_path}")
    root = (config_path.parent / config.get("project_root", "..")).resolve()
    for key in ("train_patch_dir", "test_patch_dir", "ckpt_dir", "log_dir"):
        try:
            value = Path(config["paths"][key]).expanduser()
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Missing configuration: paths.{key}") from exc
        config["paths"][key] = value.resolve() if value.is_absolute() else (root / value).resolve()
    config["config_path"] = str(config_path)
    config["project_root"] = root
    training = config["training"]
    for key in ("batch_size", "epochs", "patience"):
        if training[key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if training.get("num_workers", 0) < 0:
        raise ValueError("training.num_workers cannot be negative")
    if training.get("loss_region", "full") not in {"full", "cloud"}:
        raise ValueError("training.loss_region must be 'full' or 'cloud'")
    return config


def serializable_config(config):
    """Convert filesystem paths for checkpoint/JSON metadata."""
    if isinstance(config, dict):
        return {key: serializable_config(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        return [serializable_config(value) for value in config]
    return str(config) if isinstance(config, Path) else config
