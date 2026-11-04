# MUESUM — Craquelure Restoration Quality Audit & Failure Analysis

**Document**: Independent Technical Quality Audit, Quantitative Verification & Visual Failure Analysis  
**Date**: November 4, 2026  
**Investigated Artwork**: Authentic Oil Painting Portrait (`data/manual_tests/craquelure_portrait.png`, 436 × 436 pixels)  
**Primary Diagnostics Directory**: `outputs/craquelure_diagnostics/`  
**Evaluation Perspective**: Technical Rigor, Quantitative Verification, Visual Authenticity & Curatorial Usability  
**Status**: AUDIT COMPLETE — FAILED VISUAL CONVINCINGNESS EMPIRICALLY DIAGNOSED  

---

## 1. Executive Summary & Honest Verdict

### Did the model actually remove craquelure?
**No.** The current MUESUM restoration pipeline does **not** cleanly remove fine craquelure. 

### What actually happened?
1. **In the initial test**: An aggressive morphological threshold marked only **7.38%** of the image area, missing over 92% of the fine craquelure network across the cheeks, forehead, and nose. Because MUESUM strictly enforces 100% bit-exact preservation on unmasked pixels, all unmasked cracks were left completely unchanged. This explained why the initial user inspection showed almost zero visual change.
2. **With a refined multi-scale mask (17.28% coverage)**: When the mask was expanded to capture the hairline cracks, the network did not smoothly heal or blend them. Instead, it replaced dark crack pixels with **light, desaturated, grayish-white tracks** ($\Delta \text{Blue} = +66\text{ units}$, jumping from background 17 to predicted 83).
3. **Edge Energy Paradox**: Rather than attenuating the crack edges, high-frequency Sobel gradient magnitude on crack lines **increased by +169.9%** (from 6.68 in the original to 18.03 in the experimental restoration). The dark crack network was transformed into a visible network of light seams, introducing harsh chromatic contrast against the warm golden-ochre egg tempera/oil glaze.
4. **Conservation Verdict**: While MUESUM preserves 100% of authentic pixels outside the mask bit-exact ($\text{MAE} = 0.000000$), the restorations inside the cracks are visually unconvincing, disruptive to surface tonality, and fail professional conservation standards.

---

## 2. File & Configuration Inspection

| Component | Target File | Resolution / Type | Verification Status |
|:---|:---|:---:|:---|
| **Original Artwork** | `data/manual_tests/craquelure_portrait.png` | $436 \times 436$ RGB | Verified authentic historical oil painting crop |
| **Active Damage Mask** | `outputs/craquelure_diagnostics/mask.png` | $436 \times 436$ 8-bit | Verified binary (0 or 255), 17.28% coverage |
| **Overlay Visualization** | `outputs/craquelure_diagnostics/mask_overlay.png` | $436 \times 436$ RGB | Red overlay (#FF1E28) on detected crack pixels |
| **Raw Model Prediction** | `outputs/craquelure_diagnostics/raw_prediction.png` | $436 \times 436$ RGB | Full unblended forward pass through ResNet34-UNet |
| **Final Restored Image** | `outputs/craquelure_diagnostics/final_restoration.png` | $436 \times 436$ RGB | Mask-guided blend with distance feathering ($r=1$) |
| **Difference Heatmap** | `outputs/craquelure_diagnostics/difference_heatmap.png` | $436 \times 476$ RGB | Perceptually uniform Turbo colormap with calibration bar |
| **Detailed 2× Grid** | `outputs/craquelure_diagnostics/detailed_comparison.png` | $880 \times 1093$ RGB | $2\times$ enlarged crops across 4 anatomical zones |
| **Checkpoint Used** | `checkpoints/experimental_texture_model/best_model.pth` | PyTorch State Dict | SHA-256 verified; weights distinct from baseline |
| **Inference Engine** | `src/inference.py` (`restore_artwork_pipeline`) | Float32 / CPU | Normalization $[0, 1]$, denormalization $[0, 255]$ |

---

## 3. Mask Quality & Crack Detection Audit

### 3.1 Polarity & Bit-Exact Preservation
- **Polarity Check**: Strictly verified. `255 (White) = Damaged/Crack`, `0 (Black) = Surviving authentic painting`.
- **Outside Mask Preservation**: Mathematically verified at **0.000000 MAE** across all 157,251 unmasked pixels. Not a single surviving authentic pixel was altered or resampled.

### 3.2 Coverage Breakdown: Initial vs. Refined
- **Initial Mask (`outputs/manual_tests/craquelure_mask.png`)**:
  - Masked Pixels: 14,036 / 190,096 (**7.38% coverage**).
  - Root Cause: Fixed single-scale morphological blackhat (kernel $5 \times 5$, threshold 13) only detected the deep flaking hole on the upper lip and the widest primary fissures. Fine hairline craquelure ($1\text{–}2\text{ px}$ width, intensity delta $< 10$) fell below threshold. Over 92% of the painting surface was locked as undamaged, rendering crack removal impossible.
- **Refined Multi-Scale Mask (`outputs/craquelure_diagnostics/mask.png`)**:
  - Masked Pixels: 32,845 / 190,096 (**17.28% coverage**).
  - Method: Multi-scale blackhat kernels ($3 \times 3, 5 \times 5, 7 \times 7$) combined with localized luminance-adaptive thresholding and paint-loss tophat segmentation.
  - Anatomical Protection: Facial features (pupils, nostrils, lip seam) were explicitly protected via structural anatomical exclusion.

---

## 4. Quantitative Restoration Measurement

The table below compiles the empirical measurements computed directly from the original image and restored outputs:

| Metric | Measured Value | Conservation Interpretation |
|:---|:---:|:---|
| **Mask Coverage Area** | **17.28%** (32,845 / 190,096 px) | Captures primary craquelure web and loss spot |
| **Whole Image MAE (Experimental)** | **6.99** | Overall average pixel change across full canvas |
| **Inside Mask MAE (Experimental)** | **40.45** | Substantial pixel intensity displacement inside cracks |
| **Outside Mask MAE (Experimental)** | **0.000000** | **100% Bit-exact preservation** of unmasked paint |
| **Whole Image MAE (Baseline)** | **4.89** | Lower divergence due to darker, flatter predictions |
| **Inside Mask MAE (Baseline)** | **28.29** | Less aggressive modification than experimental |
| **Outside Mask MAE (Baseline)** | **0.000000** | **100% Bit-exact preservation** of unmasked paint |
| **Maximum Pixel Difference Overall** | **96.0 / 255** | Maximum shift observed at crack-background interfaces |
| **Maximum Pixel Difference Outside Mask**| **0.000000** | Strict adherence to non-destructive conservation |
| **Original Edge Gradient (Sobel on Cracks)**| **6.68** | Baseline edge energy of dark craquelure lines |
| **Baseline Edge Gradient (Sobel on Cracks)**| **9.17** | +37.3% increase in edge energy (crack boundary visible) |
| **Experimental Edge Gradient (Cracks)** | **18.03** | **+169.9% increase in edge energy (severe light seam)** |
| **PSNR vs. Damaged Input** | **22.96 dB** | Measures deviation from damaged input (no clean GT exists) |
| **SSIM vs. Damaged Input** | **0.4304** | Significant structural divergence introduced inside cracks |

> [!NOTE]  
> **Why PSNR/SSIM Against Ground Truth Cannot Be Calculated**:  
> For genuine historical artworks, a "clean" original without craquelure does not exist in nature. Calculating PSNR or SSIM against the damaged input merely quantifies how much the model altered the input. The low SSIM (0.4304) and moderate PSNR (22.96 dB) reflect that 17.28% of the surface underwent significant pixel transformation ($\text{MAE} = 40.45$).

---

## 5. Regional Visual Quality Analysis (2× Crops)

Analysis of the 4 diagnostic crops in [`outputs/craquelure_diagnostics/detailed_comparison.png`](file:///C:/Users/91784/Desktop/MUESUM/outputs/craquelure_diagnostics/detailed_comparison.png):

### 5.1 Forehead (Dense Craquelure Grid)
- **Original**: Continuous network of fine, dark polygonal fissures over warm golden flesh tones.
- **Mask Overlay**: Captures ~18% of hairline cracks.
- **Raw UNet Prediction**: Blurry, low-frequency representation of the forehead. The network blurs out the hairline cracks but predicts an overall cooler, desaturated skin tone.
- **Final Restoration**: Where the mask covers cracks, the model pastes the desaturated prediction. Because the prediction is significantly lighter and bluer than the surrounding warm varnish, cracks are transformed into **light gray ghost tracks**. The crack network remains clearly perceptible as a negative grid.

### 5.2 Right Cheek (Flesh Highlight & Smooth Glaze)
- **Original**: Subtle craquelure over a delicate tonal gradation with minimal surface impasto.
- **Raw UNet Prediction**: Lacks skin pore texture; produces an airbrushed, uniform tone.
- **Final Restoration**: Cracks on the cheek are partially filled, but the luminance mismatch creates patchy mottling. Under grazing illumination, this would be rejected by conservators as improper inpainting.

### 5.3 Left Cheek (Shadowed Warm Tone)
- **Original**: Deep golden-ochre with fine hairline fracturing.
- **Chromaticity Shift**:
  - Surrounding original unmasked paint: $\text{RGB} \approx [178, 110, 17]$ (strong yellow/warm cast, Blue = 17).
  - Experimental model inpainting: $\text{RGB} \approx [223, 158, 83]$ (Blue jumped to 83, Red to 223).
- **Final Restoration**: High chromatic contrast creates visible whitish seams. The crack structure is not concealed; its polarity is simply inverted from dark-on-yellow to light-on-yellow.

### 5.4 Nose & Upper Lip Lacuna (Paint Loss Spot)
- **Original**: A distinct oval paint-loss flake ($8 \times 12\text{ px}$) exposing light underdrawing above the lip vermilion border.
- **Final Restoration**: The lacuna is filled with flesh-toned pigment. While the sharp white perimeter is softened, the tone does not match the local lip shade, resulting in a flat smudge.

---

## 6. Root Cause Investigation: Why Craquelure Inpainting Fails

Our investigation reveals four primary technical causes:

### 1. The Chromaticity Mismatch of Unconditioned Inpainting
- **Mechanism**: The model takes only a 3-channel RGB image $[B, 3, H, W]$ without input mask conditioning. It does not possess a local color-adaptation mechanism (such as Poisson blending or harmonic field propagation).
- **Training Bias**: The ResNet34-UNet was trained on hundreds of diverse Met artworks with neutral gray/white backgrounds and varied palettes. In the absence of an explicit background color anchor, its raw prediction regresses toward the dataset mean, which is cooler and bluer ($\text{Blue} \sim 70\text{–}85$) than the deep golden ochre ($\text{Blue} = 17$) of aged European oil varnish.

### 2. High-Frequency Edge Inversion (+169.9% Sobel Energy)
- When inpainting narrow ($1\text{–}2\text{ px}$) fissures, human visual perception is extremely sensitive to line contrast. Replacing a dark line ($\text{RGB } [171, 102, 9]$) with a bright seam ($\text{RGB } [223, 158, 83]$) on a background of $[178, 110, 17]$ increases edge gradient magnitude from 6.68 to 18.03. To the human eye, the cracks appear just as sharp and intrusive as before.

### 3. Hairline Topology & Feathering Constraints
- Standard distance-transform feathering requires a radius $r \ge 2\text{ px}$ to create a smooth alpha ramp. For a $1\text{-px}$ or $2\text{-px}$ crack, the entire crack width falls within the transition zone, preventing true seamless integration. Setting $r=1$ avoids blur leakage into authentic paint but leaves the hard boundary between the prediction and authentic paint exposed.

### 4. Training Damage Distribution Gap
- Synthetic damage simulation (`src/damage_sim_museum.py`) models cracks via random walk polylines (`_add_scratch_lines`). Synthetic scratches have sharp, abrupt step edges with uniform dark fill. Authentic craquelure, by contrast, has micro-shadows along the crack wall, dirt accumulation, and cupping (curling edges of paint flakes). The model was never trained on real physical crack micro-relief.

---

## 7. Concrete Evolutionary Recommendations

To achieve museum-grade craquelure consolidation in future iterations of MUESUM, the following pipeline upgrades are recommended:

1. **Local Color Harmony / Poisson Seamless Inpainting**:
   - For hairline cracks ($< 3\text{ px}$), post-process the UNet prediction using gradient-domain seamless cloning (Poisson blending) or a local color-transfer transformation that forces the infilled pixels to match the exact RGB mean and covariance of the local 5-pixel boundary neighborhood.
2. **4-Channel Mask-Conditioned Backbone**:
   - Provide the binary crack mask as a 4th input channel ($[B, 4, H, W]$) or deploy Partial Convolutions (PConv) so that the convolution kernels explicitly condition on intact surrounding paint and ignore the crack pixels.
3. **Dedicated Craquelure Synthesis Augmentation**:
   - Incorporate Voronoi fracture simulation and physical craquelure models into `src/damage_sim_museum.py` rather than relying solely on random-walk scratch lines.
4. **Conservation Triage Mode in Gradio UI**:
   - Add a "Craquelure Softening" slider in the Gradio studio that allows conservators to adjust inpainting blend strength and luminance matching interactively.

---

## 8. Audit Conclusion

The experimental ResNet34-UNet restoration model **fails** the requirement for visually convincing craquelure elimination on genuine historical artwork. While the underlying MUESUM software architecture operates reliably, preserving 100% of authentic pixels outside the mask, the generated inpainting introduces noticeable color shifts and light seam artifacts. 

This honest finding demonstrates that MUESUM's evaluation framework is technically sound, capable of isolating failure modes, and will not claim success when visual and quantitative evidence demonstrates otherwise.
