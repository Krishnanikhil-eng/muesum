# MUESUM — Failed Restoration Technical Diagnosis & Root Cause Analysis

**Document**: Root Cause Investigation, Controlled Experiment & Architectural Recommendations  
**Date**: November 4, 2026  
**Investigated Artifact**: `data/manual_tests/damaged_portrait.png` (Dimensions: 626 × 417 pixels)  
**Experiment Outputs**: `outputs/manual_tests/` and `outputs/diagnostics/`  
**Status**: INVESTIGATION COMPLETE — ROOT CAUSES EMPIRICALLY ISOLATED  

---

## 1. Executive Summary & Primary Diagnosis

The real-world portrait restoration experiment on `damaged_portrait.png` revealed a stark visual failure: rather than reconstructing believable facial features (such as the occluded right eye, eyebrow, and cheekbone), the network filled the missing region with a **smooth, blurry, low-detail wash**.

Our rigorous technical audit reveals that this failure is **not** an operational error or a bug in mask feathering, but the direct consequence of four fundamental architectural and training boundaries:

1. **Receptive Field & Scale Mismatch**: The model was trained on $256 \times 256$ crops degraded by small synthetic blemishes (scratches $1\text{–}3\text{ px}$, mold spots $r=2\text{–}15\text{ px}$, and translucent stains). In this test image, the damage mask occludes **74.19% of the entire artwork** ($193,656\text{ pixels}$), spanning over $250\text{ pixels}$ continuously across the right face. Standard $3 \times 3$ convolutions cannot propagate high-frequency edge information across such vast distances.
2. **Absence of Input Mask Conditioning (3-Channel Input)**: The ResNet34-UNet architecture accepts a standard 3-channel RGB tensor $[B, 3, H, W]$. It **does not take the damage mask as an input channel**. Consequently, the network receives the gray peeling plaster as actual image content, rather than an unpainted void requiring semantic inpainting.
3. **L1 Regression Conditional Mean Collapse**: When trained with L1 / L2 reconstruction losses on large missing areas, deterministic neural networks naturally minimize expected error by predicting the conditional mean of the training distribution—resulting in smooth, desaturated, "airbrushed" averages without high-frequency structural detail.
4. **Task Domain Boundary (Surface Restoration vs. Semantic Hallucination)**: MUESUM was engineered as a *restoration* filter for cultural heritage conservation (preserving surviving authentic canvas while cleaning surface defects), **not** a generative *semantic hallucination engine* capable of inventing unseen anatomical features (e.g., a human eye) from scratch.

---

## 2. In-Depth Investigation of Potential Problems

### 2.1 Mask Correctness & Alignment Audit
- **Mask Polarity**: Verified. `src/inference.py` line 248 strictly binarizes integer masks: `(mask > 127).astype(np.float32)`. `1.0 / 255 = damaged region`, `0.0 / 0 = surviving authentic artwork`.
- **Preservation Outside Mask**: Verified 100% bit-exact ($\text{MAE} = 0.000000, \text{MaxDiff} = 0$). In `src/feathering.py`, `alpha[~dmg_bool] = 0.0`, and surviving pixels are locked via direct array assignment (`blended_final[undmg_bool] = orig_rgb[undmg_bool]`).
- **Mask Size**: **The mask is excessively large (74.19% of image area).** While accurately identifying all peeling plaster, concrete exposure, and torn edges, masking 74% of a human face forces the model into large-hole semantic synthesis rather than local inpainting.
- **Diagnostic Visualization**: Generated [`outputs/diagnostics/mask_visualization.png`](file:///C:/Users/91784/Desktop/MUESUM/outputs/diagnostics/mask_visualization.png) and [`outputs/diagnostics/mask_overlay.png`](file:///C:/Users/91784/Desktop/MUESUM/outputs/diagnostics/mask_overlay.png).

### 2.2 Model Architecture & Training Limitations
- **Synthetic Damage Simulator (`src/damage_sim_museum.py`)**:
  - Training damages: `_add_scratch_lines` (width 1–3 px), `_add_dust_and_specks` (radius 1–4 px), `_add_mold_spots` (radius 2–15 px), and `_add_stain` (translucent color modulation where underlying artwork remains visible).
  - The model was **never trained** on massive, solid plaster delamination occluding over 70% of a composition.
- **Patch Size ($256 \times 256$)**:
  - Models trained on local $256 \times 256$ crops learn localized texture synthesis (canvas weave, paper grain, craquelure filling). They cannot learn global facial symmetry (e.g., using a left eye at $(x_1, y_1)$ to place a right eye at $(x_2, y_2)$ across a 200-pixel void).
- **Missing Generative Prior**:
  - ResNet34-UNet lacks an autoregressive or diffusion-based semantic prior. It cannot invent a missing human eye or lips out of thin air.

### 2.3 Inference Pipeline & Blending Dynamics
- **Checkpoints**: Verified. Both `baseline_run/best_model.pth` and `experimental_texture_model/best_model.pth` loaded cleanly with distinct weights (240/240 differing tensors).
- **Model Output Normalization**: Input normalized to $[0, 1]$; raw UNet output denormalized via `tensor_to_uint8`.
- **Feathering Behavior**: Interior distance-transform feathering ($r=3$) operates strictly *inside* the damage mask. It does not bleed into undamaged areas. However, because the raw UNet prediction inside the mask is blurry, feathering smoothly transitions the sharp authentic edge into that blurry prediction, creating a soft transition zone.

---

## 3. Controlled Empirical Experiment: Damage Scale Breakdown

To isolate the exact point where restoration quality fails, we executed a controlled benchmark on an identical $256 \times 256$ crop of the portrait's facial region across three damage scales:
1. **Small Damage Mask (2.30% area, 1,509 px)**: Thin scratches ($3\text{–}4\text{ px}$) across forehead and cheek.
2. **Medium Damage Mask (7.23% area, 4,739 px)**: Moderate circular/elliptical flaking patch ($r = 25\text{–}35\text{ px}$) on the cheek.
3. **Large Damage Mask (55.08% area, 36,096 px)**: Massive missing block covering the right half of the face.

### Quantitative Sharpness & Variance Breakdown

| Damage Scale | Mask Area (%) | Baseline Sobel Grad | Experimental Sobel Grad | Inpainted Std | Visual Restoration Quality |
|:---|:---:|:---:|:---:|:---:|:---|
| **Small (Scratches)** | 2.30% | **12.40** | **11.76** | 35.90 | **Sharp & High-Fidelity**: Seamless line inpainting; surrounding pixels inform receptive field. |
| **Medium (Flaking)** | 7.23% | **3.65** | **3.63** | 57.14 | **Soft / Textural**: Plausible local color fill; fine skin micro-texture begins to degrade. |
| **Large (Missing Half)** | 55.08% | **1.04** | **1.01** | 25.16 | **Complete Collapse**: Flat, blurry, featureless wash. 92% loss in edge gradient energy. |

*Visual Evidence*: Inspect [`outputs/diagnostics/model_comparison.png`](file:///C:/Users/91784/Desktop/MUESUM/outputs/diagnostics/model_comparison.png) for side-by-side grid comparison across all three scales.

---

## 4. Root Cause Summary Matrix

```
Damage Type         Receptive Field   Model Capability                Result
─────────────────────────────────────────────────────────────────────────────
Narrow Scratch      Fits in kernel    Local texture interpolation     SHARP / EXCELLENT
Small Craquelure    Fits in kernel    High-frequency fill             SHARP / EXCELLENT
Mold Spot / Stain   Local context     Color/tone correction           PLAUSIBLE / GOOD
Massive Lacuna      Exceeds kernel    L1 conditional mean collapse    BLURRY / FAILED
Missing Anatomy     No semantic prior Cannot invent eye/features      BLURRY / FAILED
```

---

## 5. Architectural Recommendations (Prioritized)

To address large-scale real-world degradation without compromising MUESUM's core conservation guarantees, we recommend the following evolutionary roadmap:

### Recommendation 1 (Immediate / High Impact): Mask-Conditioned Network Input
- **Current Limitation**: ResNet34-UNet accepts $[B, 3, H, W]$ RGB. It sees the gray plaster as the actual image.
- **Solution**: Upgrade the input layer to **4 channels**: $[B, 4, H, W]$ ($RGB + Mask$) or adopt **Partial Convolutions (PConv)**. By explicitly zeroing masked pixels and feeding the mask into every layer, convolutions never sample degraded plaster pixels, preventing color contamination.

### Recommendation 2 (Architecture): Large-Receptive-Field Backbone (LaMa / FFC)
- **Current Limitation**: Standard $3 \times 3$ convolutional kernels have localized receptive fields that cannot bridge hundreds of missing pixels.
- **Solution**: Incorporate **Fast Fourier Convolutions (FFCs)**, as pioneered in the *LaMa (Large Mask Inpainting)* architecture. FFCs operate in the frequency domain, providing an image-wide receptive field from the very first layers. This allows the model to transfer textures across large occlusions.

### Recommendation 3 (Long-Term / Comprehensive): Two-Tier Hybrid Architecture
In professional museum conservation, surface retouching is strictly distinguished from anatomical reconstruction:
1. **Tier 1 (MUESUM Local Inpainting Engine — Active)**:
   - Reserved for physical damage, scratches, craquelure, tears, and stains ($< 25\text{ px}$ width).
   - Preserves 100% bit-exact substrate with interior distance-transform feathering.
2. **Tier 2 (Semantic Generative Prior — Hybrid Integration)**:
   - Activated only when large semantic anatomy is missing ($> 30\%$ area loss, missing facial features).
   - Uses a latent diffusion or control-net inpainting model (conditioned on period style and portrait symmetry) to propose plausible anatomical hypotheses.
   - Outputs are then passed through MUESUM's curatorial mask-guided compositing engine to guarantee that surviving authentic artwork remains mathematically untouched.
