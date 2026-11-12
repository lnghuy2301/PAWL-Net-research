# Reproducibility and scientific provenance

This cleanup preserves the supplied model architectures and default research settings while making paths, entry points, and evaluation behavior explicit. It is not a retraining or independent confirmation of the manuscript results.

## Manuscript versus executable source

| Topic | Supplied material | Repository decision |
| --- | --- | --- |
| Fifth data channel | PDF Table 1 and export notebook say Albedo; `method.txt` and a figure label say FVC | Document Albedo; preserve the actual exported channel order |
| Loss region | PDF says cloud-only; original scripts pass full maps to pixel/edge loss | Default `loss_region: full` preserves original training; `cloud` is opt-in |
| Pixel epsilon | PDF states `1e-3`; original code uses `1e-6` | Default `charbonnier_eps: 1e-6` |
| Edge objective | PDF Eq. 3 uses absolute Laplacian difference; code applies Charbonnier to Laplacian maps | Preserve the code's smoothed edge objective; cloud-only selection does not remove this distinction |
| Mask geometry | PDF describes a combined generator with cubic Bézier strokes; code chooses one morphology and uses line strokes | Preserve squares/freeform lines/Perlin; do not describe line strokes as Bézier curves |
| Mask coverage | PDF describes 10%–70%; original generator has no coverage rejection and may produce no clouds | Default coverage is unrestricted; optional `data.cloud_fraction: [0.1, 0.7]` adds bounded rejection |
| Optical masking | Original loader masks channels 0–5 and retains spatial layers | Preserve this convention for compatibility |
| Random Forest | PDF: 7% sample, nine features; script call: 3% sample, ten features; log text incorrectly says 5% | Default 3% and ten features; both are explicit settings |
| Optimizer setup | Original YAML: LR `2e-4`, 50 epochs, patience 10; method notes: LR `1e-4`, patience 20 | Preserve executable YAML settings; AutoLoss group LR is `1e-3` |
| Validation | Original scripts compute full-map L1 on dynamically masked validation data | Retain full-map L1; stabilize masks and weight batches by sample count |
| Population scaling | PDF table describes Min-Max; actual code uses `log1p(population)/10` | Preserve logarithmic scaling, which is not guaranteed to lie in [0,1] |
| LSE scaling | PDF table broadly says [0,1]; code clips to [0.9,1] without rescaling | Preserve the code |
| Swin baseline | Code uses a timm Swin-Tiny encoder with CNN upsampling blocks | Describe that architecture explicitly; pretrained dataset/version is not recoverable from the original unpinned requirements |
| Initialization | PDF algorithm mentions He initialization; supplied U-Net uses PyTorch defaults | Preserve PyTorch defaults and checkpoint parameter names |
| Geographic/figure labels | PDF uses Long Xuyen in most sections, An Giang in conclusion; 128-pixel captions accompany 512-pixel plots | Describe the study cities consistently and derive plot dimensions from actual arrays |

Changing any of these settings defines a new experiment. Enabling cloud-only loss and coverage rejection alone does not reproduce the PDF's entire protocol.

## Evaluation changes

The original loader regenerated random masks on every access, including test access. The historical result log contains different evaluated pixel counts across runs and models. Those runs cannot be reconstructed exactly without their seeds or saved masks.

The reorganized loader uses `seed + sorted_patch_index` for validation/test masks. All experiments use the same dataset and mask settings, so their current evaluations are comparable when their patch lists match. Training keeps dynamic masks. Reordering or adding patch filenames changes the indexed masks; retain a fixed data manifest for a formal experiment.

MAE, RMSE, and R² accumulate over hidden pixels in constant memory. SSIM is averaged over patches with cloud pixels, after replacing observed pixels by the target, with a fixed 30 °C data range. This preserves the original deep-learning evaluator's definition. The original Random Forest evaluator instead used a per-patch target range; its historical SSIM is therefore not guaranteed to match the shared evaluator. SSIM is a full blended-image metric, not a metric restricted to cloud pixels.

## NoData and input quality

The original preprocessing accepted patches with as little as 10% nonzero LST. The original loader then clipped zero-valued source LST to 293 K and treated it as a valid zero-normalized target. That convention is retained for checkpoint compatibility and is explicitly recorded in evaluation JSON. It can bias historical and future metrics. Real cloud gaps do not have known same-date targets and must not be claimed as independently validated reconstruction pixels.

The streaming metrics utility accepts a separate validity mask, but the default loader/evaluator intentionally follows the historical protocol. A scientifically revised protocol should preserve source validity before clipping, score only synthetically hidden valid reference pixels, and revisit full-image SSIM for incomplete reference maps. Such a revision requires new metrics and potentially retraining, rather than substituting old numbers.

The local archive also contains unusually low nonzero LST values, as detailed in [data.md](data.md). No scale/offset correction has been guessed or applied.

## Training state and compatibility

Existing checkpoint filenames and layer keys remain compatible. Evaluation constructs Swin without downloading pretrained weights. Fresh Swin training copies the original patch-embedding bias as well as its RGB weights; this fixes an initialization omission in the old code and can change newly trained models.

New runs are isolated under `outputs/checkpoints/runs/<run-name>/`; existing research checkpoints are preserved. Explicit `--resume` restores model, optimizer, scheduler, loss weights, scaler, early stopping counter, and saved PyTorch/NumPy/loader RNG states when available. Legacy checkpoints lack some of these fields, so their resume cannot be bit-exact. Cross-device/library-version reproducibility is not guaranteed.

The original dependency file was unpinned and the copied `venv` was a Linux environment unusable on Windows. The package now declares supported dependency ranges. They are compatibility ranges, not a historical environment lock. Record a fresh environment export alongside any new scientific run.

## Scope of verification

Regression tests use synthetic fixtures and cover scaling, channel masking, stable evaluation masks, splits, model outputs, loss gradients, streaming metrics, temporal preprocessing, sliding-window coverage, and a small CPU training/resume/evaluation workflow. Local checkpoint checks and data inspection are documented separately in [validation.md](validation.md). Manuscript tables remain labelled as reported results.
