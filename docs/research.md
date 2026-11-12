# Research context

The supplied 15-page PDF is titled **PAWL-Net: A Physics-Aware Inpainting Network with Automatic Loss Weighting for Cloud-Obscured Land Surface Temperature Reconstruction**. This document is the source for the author metadata and reported tables in this repository. The PDF is kept outside the source tree and has not been copied for distribution.

| Author, as printed | Affiliation, as printed | ORCID |
| --- | --- | --- |
| Gia Huy Le-Nguyen | Department of Computer Science, School of Information and Communication Technology, Can Tho University | 0009-0001-1154-1035 |
| Trung Nguyen Ly | Department of Land Resource, College of Environment and Natural Resources, Can Tho University | 0009-0009-4480-264X |
| Nguyen Thai-Nghe | Department of Information Systems, School of Information and Communication Technology, Can Tho University | 0000-0002-9127-2778 |

The third author is marked as corresponding author in the PDF. No email, DOI, publication venue, or publication year is provided in the supplied document, and none has been inferred from filesystem timestamps. Citation metadata uses a provisional unpublished category until bibliographic details are confirmed.

## Method represented by the code

Nine data channels are normalized, the first six are masked to simulate cloud obstruction, and the binary visibility mask is appended. PAWL-Net uses a standard U-Net with encoder widths 64, 128, 256, and 512, a 1024-channel bottleneck, transposed-convolution upsampling, concatenated skip connections, and a sigmoid LST output.

The automatic weighting module learns log variances `s_i = log(sigma_i²)`:

```text
L_total = sum_i [0.5 * exp(-s_i) * L_i + 0.5 * s_i]
```

The two components are Charbonnier pixel loss and a Laplacian structural loss. Learnable parameters start at zero, corresponding to unit variance. They are log variances, not direct standard deviations.

Baselines include a Swin-Tiny encoder with a convolutional decoder, and pixel-wise Random Forest. Four U-Net ablations compare 2 versus 10 channels and L1 versus automatic loss weighting. The 10-channel AutoLoss case is the proposed PAWL-Net configuration.

## Reported ablations

Values below are transcribed from manuscript Table 3. They have not been recomputed during restructuring.

| Configuration | MAE (°C) | RMSE (°C) | R² | SSIM |
| --- | ---: | ---: | ---: | ---: |
| 2-channel + L1 | 0.4644 | 1.2155 | 0.9619 | 0.9669 |
| 2-channel + AutoLoss | 0.4533 | 1.1769 | 0.9647 | 0.9675 |
| 10-channel + L1 | 0.4669 | 1.2061 | 0.9630 | 0.9664 |
| 10-channel + AutoLoss | 0.4504 | 1.1815 | 0.9641 | 0.9679 |

The two-channel AutoLoss run has the best reported RMSE and R² among these ablations. The ten-channel AutoLoss run has the best reported MAE and SSIM. The table therefore does not support claiming that extra channels improve every metric.

Differences between manuscript descriptions, legacy notes, and executable code are recorded in [reproducibility.md](reproducibility.md). The original Vietnamese method notes are preserved in [legacy_method_vi.md](legacy_method_vi.md) as a historical source.
