# Checkpoints

Existing files retain their original names:

| File | Role |
| --- | --- |
| `best_unet.pth`, `last_unet.pth` | PAWL-Net / 10-channel U-Net + AutoLoss |
| `best_model.pth`, `last_model.pth` | Swin encoder + convolutional decoder |
| `best_2ch_l1.pth` | 2-channel U-Net + L1 |
| `best_2ch_auto.pth` | 2-channel U-Net + AutoLoss |
| `best_10ch_l1.pth` | 10-channel U-Net + L1 |
| `random_forest_model.joblib` | Original pixel-wise Random Forest |

New training output goes in `runs/<run-name>/` and does not overwrite these research files. Weight files are local artifacts, not included with a source-only checkout.
