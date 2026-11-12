# Restructuring validation

Checks were performed locally on 9 November 2026. No repository was published, no remote messages were sent, and no manuscript experiment was retrained.

## Code checks

- **24 regression tests passed** on CPU, including U-Net shape/range checks, loss gradients, channel masking, deterministic test masks, cloud-only streaming metrics, temporal patch routing, arbitrary-size scene coverage, Random Forest training/evaluation, and a small training/resume/evaluation workflow.
- The resume regression compares final model tensors from interrupted versus uninterrupted synthetic training exactly.
- `ruff check .` and `ruff format --check .` passed.
- Python package metadata and wheel build passed with dependencies/build isolation disabled for the local packaging check.
- Both notebook documents pass nbformat schema validation. All 11 default code cells across the two notebooks executed successfully; research-data export, full scene inference, and timing remain opt-in.
- Citation metadata passed the [official CFF 1.2.0 schema](https://raw.githubusercontent.com/citation-file-format/citation-file-format/1.2.0/schema.json), all three ORCID check digits were verified, and local Markdown links checked.

The GitHub Actions workflow has been prepared but has not run remotely. Its Python 3.10/3.12 jobs are prospective; the local tests used Python 3.11.15 on Windows, PyTorch 2.5.1, NumPy 1.26.4, scikit-learn 1.8.0, timm 1.0.31, and OpenCV 4.14.0.

## Existing model artifacts

All five best deep-learning state dictionaries loaded successfully into the reorganized architectures. A forward pass on a synthetic input produced a finite `(1, 1, 128, 128)` output for each:

| Experiment | Checkpoint | Trainable parameters |
| --- | --- | ---: |
| PAWL-Net | `best_unet.pth` | 31,047,553 |
| Swin baseline | `best_model.pth` | 39,878,907 |
| 2-channel + L1 | `best_2ch_l1.pth` | 31,042,945 |
| 2-channel + AutoLoss | `best_2ch_auto.pth` | 31,042,945 |
| 10-channel + L1 | `best_10ch_l1.pth` | 31,047,553 |

The two last-training checkpoints were read with `weights_only=True`; their saved model/loss/optimizer/scheduler fields were inspected. These legacy snapshots lack full RNG/scaler/early-stopping state.

The original Random Forest artifact loaded with its recorded scikit-learn version, 1.8.0. It has 100 trees and ten input features; a small prediction produced finite values. `requirements/rf-checkpoint.txt` records that version for artifact compatibility. This does not recover every historical environment detail.

These checks establish loading and basic execution, not reproduction of reported test metrics. The original eight model artifacts and archive were hashed after relocation and checked for unchanged bytes during final verification. Local manifests and detailed check outputs are kept under `.local/`.

## Data and Git limitations

All 8,455 archive entries were inspected. There are 8,131 readable finite `(128, 128, 9)` arrays and 324 zero-byte entries in the supplied local archive. They are retained in place; no synthetic replacement, scale correction, or silent subset evaluation was used. See [data.md](data.md) and [the input profile](../reports/local_data_profile.json).

Training arrays and annual source scenes were absent. No full test-set metric run, new training of research checkpoints, Earth Engine authentication/export, CUDA execution, or manuscript timing benchmark was performed.

The pre-existing `.git` contains corrupt/empty objects and invalid references. It was inspected but not repaired, reset, or replaced. Git history requires separate attention before future publication; the source tree and local backups remain available independently of that history.
