# Data schema and availability

The project owner identifies the supplied code and preprocessing workflow as the basis of the final manuscript. Data are preserved as supplied; the restructuring does not infer a new scaling formula or change research targets.

## Availability

The original workspace contains a local `S128_Test.rar` archive and model checkpoints. Its training-patch folder is empty. Annual NumPy arrays and the code that originally generated the Earth Engine annual assets are not included. The archive and checkpoints remain local and are excluded from Git.

The test archive is preserved in `data/archives/` and extracted to `data/patches/test/`. Source-code users must supply their own compatible data and weights. No public dataset or weight-download URL is claimed.

## Array schema

Annual arrays: `(12, H, W, 9)`, filenames `City_YYYY.npy`.
Patches: `(128, 128, 9)`, float32, filenames such as `CanTho_2023_m01_y64_x384.npy`.

| Index | Export band | Source units | Historical normalization |
| ---: | --- | --- | --- |
| 0 | `LST_Kelvin` | Kelvin, as labelled in the export | `(clip(x, 293, 323) - 293) / 30` |
| 1 | `NDVI` | Dimensionless | `(x + 1) / 2` |
| 2 | `NDBI` | Dimensionless | `(x + 1) / 2` |
| 3 | `MNDWI` | Dimensionless | `(x + 1) / 2` |
| 4 | `Albedo` | Dimensionless | `clip(x, 0, 1)` |
| 5 | `Emissivity` | Dimensionless | `clip(x, 0.9, 1.0)` |
| 6 | `Dist_Water` | Metres | `clip(x / 5000, 0, 1)` |
| 7 | `Elevation` | Metres | `clip(x / 50, 0, 1)` |
| 8 | `Population` | Density; upstream unit metadata not supplied | `log1p(x) / 10` |

A tenth channel is created at load time: binary visibility, where 1 is observed and 0 is hidden. The loader masks the first six data channels; spatial channels remain present. Conversion of a normalized target to Celsius is `x * 30 + 19.85`.

The fifth band is **Albedo** in the export notebook and PDF Table 1. The legacy Vietnamese method notes and a manuscript figure use FVC terminology. No data channel has been renamed or replaced to resolve that textual discrepancy.

## Local archive inspection

The local archive lists 8,455 unique filenames. All entries were extracted and inspected. The supplied archive contains **324 zero-byte entries**; 8,131 entries contain readable NumPy arrays. Empty entries were confirmed with both the Windows extractor and the [official UnRAR tool](https://www.rarlab.com/rar_add.htm); the archive's own listing reports zero size for those entries. They were retained, not replaced or silently removed.

The original dataset loader caught load failures and substituted a `(128, 128, 9)` zero array. The cleaned loader raises a filename-specific error instead. Consequently, an evaluation against this exact local extraction stops at the first empty entry until a complete source copy is supplied. Running only the readable files would change the evaluated test set and is not treated as reproduction of the manuscript.

This finding describes the copy currently present in the workspace. The original notebook's saved output reports scanning 8,455 patches; the present inspection does not establish whether those historical runs encountered the same empty entries.

[reports/local_data_profile.json](../reports/local_data_profile.json) records the complete local inspection, including empty filenames, source channel ranges, and patch counts. This is an input profile, not a new model evaluation.

The readable arrays include LST values below the historical 293 K lower clipping threshold, and the inspected population channel is zero throughout. Both are recorded without changing the data. Values outside the clipping interval are handled by the original normalization rule. Source zero-valued NoData is distinct from synthetic clouds, but the legacy target convention clips those values as well; see [reproducibility.md](reproducibility.md).

## Preprocessing and split

`scripts/export_earth_engine.py` exports existing annual assets using configurable project and asset IDs. It does not recreate the upstream Landsat processing or apply undocumented scale/offset corrections. Configure these IDs through flags or shell environment variables; `.env.example` lists the names.

`scripts/prepare_patches.py` creates 128-pixel patches with stride 64, routes 2015–2022 to training/validation and 2023–2024 to testing, skips existing output, and rejects patches with less than 10% nonzero LST or nonfinite values. The source notebook already rejected heavily empty patches; nonfinite-value rejection is explicit in the cleaned utility.

The loader retains the original sorted-file 80/20 internal training/validation split. Because neighboring patches overlap, this split does not guarantee scene-level or spatial separation. The test period is temporally held out when folders contain the intended years; the loader does not infer or validate acquisition dates from external metadata.

Run a fresh input audit without modifying data:

```text
python scripts/inspect_data.py --output outputs/logs/test-input-profile.json
```

For a source-only checkout, populate `data/raw/` or `data/patches/` before running the corresponding workflows.
