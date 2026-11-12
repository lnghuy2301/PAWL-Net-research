# Figure provenance

`reconstruction_comparison.png` is extracted byte-for-byte from the saved PNG output of cell 4, counting from zero, in the original `UNet_Inpainting.ipynb`.

The source cell selects March 2024 from `CanTho_2024.npy`, crops a central 512 × 512 region, combines synthetic cloud shapes with the source availability mask, and compares the original PAWL-Net and Swin checkpoints. Panel order is reference LST, masked input, PAWL-Net, and Swin. Pixel axes and color scale are retained unchanged.

The original notebook with its code and all outputs remains in `.local/original_files/`. The source annual array was not included in the project folder, so this figure has not been independently regenerated. No new usage license is assigned to the figure.
