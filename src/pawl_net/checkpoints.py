"""Load both original raw state dictionaries and new training checkpoints."""

from pathlib import Path

import torch


def load_model_weights(model, path, device="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return checkpoint
