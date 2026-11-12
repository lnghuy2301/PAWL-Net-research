# Contributing

Use a dedicated Python environment and install `python -m pip install -e ".[dev]"`. Run `python -m pytest -q`, `ruff check .`, and `ruff format --check .` before sharing changes.

Keep model architecture/state-dictionary names compatible with existing checkpoints unless an intentional versioned migration is documented. Record changes to normalization, data splits, masks, loss regions, or metrics in `docs/reproducibility.md`; they affect scientific comparability.

Use synthetic fixtures for tests. Keep data, credentials, model weights, generated notebook outputs, and machine-specific paths outside versioned source files. Configure local paths in `configs/config.local.yaml` and Earth Engine settings through environment variables.

This project has not selected a license or published a contribution policy for rights transfer. Coordinate substantive external contributions with the project owner before redistribution.
