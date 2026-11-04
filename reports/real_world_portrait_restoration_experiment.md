# MUESUM — Real-World Artwork Restoration Experiment Report

**Experiment Title**: Real-World Historical Portrait Photograph Restoration  
**Date**: November 4, 2026  
**Input Image**: `data/manual_tests/damaged_portrait.png` (Dimensions: 626 × 417 pixels)  
**Curatorial Medium**: Weathered Historical Portrait Photograph on Deteriorated Plaster Substrate  
**Status**: COMPLETE — ALL ASSETS GENERATED & VERIFIED  

---

## 1. Executive Summary

This experiment evaluated the MUESUM neural restoration pipeline on a real-world, non-synthetic damaged artifact: a vintage portrait photograph degraded by extensive peeling plaster, torn paper delamination, and surface loss.

- **Total Damage Coverage**: **74.19%** of the artwork surface was covered by gray plaster, flaking edges, or exposed concrete.
- **Authentic Artwork Preserved**: **25.81%** surviving substrate (left eye, eyebrow, nose bridge, lips, hair fragments).
- **Conservation Guarantee**: **100% Bit-Exact Mathematical Preservation** ($\text{MAE} = 0.000000, \text{MaxDiff} = 0$) across all surviving authentic pixels.
- **Visual Outcome**: Both models successfully synthesized coherent skin tones, hair mass, and background continuity across the missing right half of the face. The **Experimental Texture Model** demonstrated superior gradient sharpness and organic skin grain over the baseline model.

---

## 2. Experimental Setup & Pipeline Parameters

| Parameter | Configuration Setting | Curatorial Rationale |
|:---|:---|:---|
| **Input Image** | `data/manual_tests/damaged_portrait.png` | Real photographic portrait with severe plaster loss |
| **Input Resolution** | 626 × 417 pixels (Native RGB) | Evaluated at native full resolution ($\le 1024\text{ px}$) |
| **Damage Mask** | `outputs/manual_tests/damage_mask.png` | High-precision binary mask (White: 255, Black: 0) |
| **Mask Coverage** | 74.19% (193,656 damaged pixels) | Large-scale lacunae covering the entire right facial hemisphere |
| **Blend Mode** | `mask_guided` | Strict non-destructive curatorial preservation mode |
| **Feathering Mode** | `feathered` | Distance-transform interior smoothing |
| **Feather Radius** | 3 pixels | Eliminates jagged paper tear lines without altering authentic paint |
| **Checkpoints Tested**| 1. Baseline Run (`checkpoints/baseline_run/best_model.pth`)<br>2. Experimental Model (`checkpoints/experimental_texture_model/best_model.pth`) | Comparative evaluation of pure L1 vs Multi-Term Composite Loss |

---

## 3. Generated Output Deliverables

All generated outputs have been saved to `outputs/manual_tests/`:

1. **`original.png`** (626 × 417, 594.4 KB): Unmodified original damaged input photograph.
2. **`damage_mask.png`** (626 × 417, 3.8 KB): Binary curatorial damage mask isolating peeling plaster and lacunae.
3. **`restored_baseline.png`** (626 × 417, 364.1 KB): Restored output from the baseline ResNet34-UNet (Epoch 8).
4. **`restored_experimental.png`** (626 × 417, 369.1 KB): Restored output from the Experimental Texture Model.
5. **`final_restored.png`** (626 × 417, 369.1 KB): Selected production restoration (Experimental Texture Model).
6. **`comparison.png`** (1878 × 964, 1942.8 KB): Comprehensive 6-panel curatorial audit comparison board with statistics.

---

## 4. Empirical Evaluation & Visual Analysis

### 4.1 Quantitative Diagnostics

| Diagnostic Metric | Baseline Model | Experimental Texture Model | Curatorial Verification |
|:---|:---:|:---:|:---:|
| **Processing Latency** | 1977.9 ms | 2247.9 ms | Sequential CPU execution |
| **Undamaged Pixel MAE** | **0.000000** | **0.000000** | Bit-exact preservation confirmed |
| **Undamaged MaxDiff** | **0** | **0** | Zero intensity drift |
| **Inpainted Mean Color (RGB)**| (149.2, 135.2, 118.8) | (144.1, 133.1, 118.3) | Harmonized with surviving skin tones |
| **Inpainted Texture Std** | 47.48 | **50.50** | Higher spatial variance in experimental |
| **Face Gradient Sharpness**| 0.928 | **0.953** | Higher edge and feature definition |

### 4.2 Curatorial Observations

1. **Reconstruction of Missing Anatomy**:
   The model faced an extreme inpainting task: the right eye, right cheek, temple, and right jawline were 100% occluded by peeling plaster. The network successfully inferred the oval contour of the woman's face, synthesized believable facial flesh tones matching the left cheek, and bridged the gap between the head and the surviving right hair lock.
2. **Boundary Seam Softening**:
   The sharp, white-edged paper tear that severed the face adjacent to the lips was completely dissolved by the interior distance-transform feathering ($r=3$). No harsh transition or chromatic aberration is visible along the seam boundary.
3. **Model Comparison**:
   - The **Baseline Model** produced a plausible but visibly smoothed reconstruction characteristic of pure L1 optimization (a slight "airbrushed" appearance).
   - The **Experimental Texture Model** incorporated high-frequency Sobel gradients and VGG perceptual features, producing richer textural shading across the synthesized cheek and hairline.
4. **Scientific Authenticity Disclaimer**:
   In accordance with conservation ethics, this restoration represents a **plausible digital hypothesis**. Because more than 74% of the original information was physically destroyed, the neural network cannot reconstruct the true historical individuality of the subject's right eye or micro-pores; rather, it synthesizes a coherent, aesthetically harmonious approximation.

