# MUESUM — Phase F.3: Implementation Report

**Document ID**: `reports/phase_f3_implementation_report.md`  
**Date**: November 3, 2026  
**Milestone**: Phase F.3 (Inference Pipeline & Gradio Studio Implementation)  
**System Hardware**: Intel Core i5-1235U (10 Cores), 16 GB RAM, CPU Execution  
**Model Checkpoint**: ResNet34-UNet Epoch 8 Best Checkpoint (`checkpoints/baseline_run/best_model.pth`)  
**Status**: **COMPLETED & VERIFIED (All Tests Passed)**

---

## 1. Executive Summary

Phase F.3 has successfully bridged the gap between raw neural research and an operational, curatorial-grade restoration system. The approved mask-guided blending architecture and interactive web studio have been implemented and rigorously tested.

### Core Achievements
1. **Reusable Core Inference Engine (`src/inference.py`)**:
   - Implemented `restore_artwork_pipeline()` with strict validation, aspect-preserving neural scaling, and full-resolution composition.
   - Built `ModelManager` singleton cache to avoid reloading the 339 MB checkpoint across sequential inference calls.
   - Preserved 100% backward compatibility via the legacy `unet_restore()` wrapper.
2. **Mathematical Bit-Exact Preservation**:
   - Ensured that 100% of authentic surviving artwork is bit-exact ($\text{MaxDiff} = 0$, $\text{MAE} = 0.000000$).
3. **Interactive Gradio Studio (`app/demo.py`)**:
   - Modernized UI supporting dual workflows: File Upload (Image + Mask) and Interactive Brush Canvas (`gr.ImageEditor`).
   - Integrated Before/After interactive slider (`gr.ImageSlider`), comparative diagnostic tabs, and full-resolution PNG download.
4. **Comprehensive Test Suite (`tests/test_inference_pipeline.py`)**:
   - 8 unit tests created and verified (0.50s execution time), covering bit-exact preservation, dimension checks, missing-mask rejection, polarity normalization, and model caching.
   - Passed existing full diagnostic suite in `test_pipeline.py` (6/6 tests passed).

---

## 2. Inventory of Modified & Created Files

| File Path | Action | Role & Functional Description |
| :--- | :---: | :--- |
| `src/inference.py` | **Modified** | Re-engineered into a high-performance restoration engine. Added `restore_artwork_pipeline()`, `ModelManager`, `_standardize_image()`, `_standardize_mask()`, and updated `unet_restore()`. |
| `app/demo.py` | **Modified** | Completely modernized Gradio studio. Added dual tabs, `gr.ImageEditor` for mask painting, `gr.ImageSlider`, diagnostic cards, and export button. |
| `tests/test_inference_pipeline.py` | **Created** | Comprehensive unit test suite with lightweight mock models covering all 6 mandatory test cases plus resizing and wrapper regression. |
| `scripts/verify_representative_artwork.py` | **Created** | Standalone benchmark verification script running the full pipeline on anchor artwork `247010` at 1024 max dimension. |

---

## 3. Key Implementation Decisions & Technical Contracts

### 3.1 Model Caching via `ModelManager` Singleton
- **Problem**: Loading the 339 MB checkpoint from disk takes ~0.9s on CPU and fragments memory when called in a loop or web server.
- **Solution**: Implemented `ModelManager` class with class-level attributes `_model`, `_ckpt_path`, and `_device`. Subsequent calls retrieve the resident PyTorch model in memory with zero file I/O overhead.
- **Cache Invalidation**: Includes `ModelManager.clear_cache()` for explicit memory flushing when needed.

### 3.2 Strict Input & Mask Validation Rules
- **Rule 1 — No Silent Mask Fabrication**:
  If `blend_mode == "mask_guided"` and `damage_mask is None`, the pipeline raises an explicit `ValueError`:
  > *"Mask-guided restoration requires a damage mask, but none was provided. MUESUM does not fabricate or guess damage masks automatically. Either provide an explicit damage mask, use the interactive brush tool, or set blend_mode='raw_unet'."*
- **Rule 2 — Exact Dimension Matching**:
  Image $[H, W]$ and Mask $[H, W]$ must match exactly. Any mismatch raises `ValueError(f"Dimension mismatch: damage mask dimensions ({w_m}x{h_m}) do not match image dimensions ({w_i}x{h_i}).")`.
- **Rule 3 — Polarity Standardization**:
  Mask polarity is standardized and documented: $1.0$ (or $255$) = damaged pixels to repair; $0.0$ (or $0$) = pristine artwork. The pipeline validates that masks contain no NaNs/Infs and fall within valid numeric ranges ($[0, 1]$ for floats, $[0, 255]$ for integers).

### 3.3 Aspect-Preserving Scaling with Full-Resolution Composition
- For memory safety on CPU, large artworks (e.g. $3883 \times 3614$ px) are scaled down to `max_dimension=1024` during the neural network forward pass.
- After unpadding, the raw neural prediction is bicubically upscaled back to the **original full resolution** ($3883 \times 3614$ px).
- The final composition is executed at the **original full resolution** using the **original unscaled mask**:
  $$\text{Final}_{\text{orig}}[\sim M_{\text{orig}}] = \text{Original}_{\text{orig}}[\sim M_{\text{orig}}]$$
  $$\text{Final}_{\text{orig}}[M_{\text{orig}}] = \text{Restored}_{\text{orig}}[M_{\text{orig}}]$$
- **Result**: Guarantees that untouched canvas/paper pixels undergo zero bicubic degradation or color interpolation.

### 3.4 Memory Management & CPU Optimization
- Uses `torch.inference_mode()` instead of `torch.no_grad()` for optimized runtime execution without autograd graph tracking.
- Output tensors are detached to CPU immediately and converted to NumPy uint8.
- Temporary tensors (`inp_tensor`, `out_tensor`, `eval_padded`) are explicitly deleted with `del` at the end of each pass.

---

## 4. Gradio Studio Architecture (`app/demo.py`)

### 4.1 Component Compatibility (Gradio 6.31.0)
The web demo leverages modern Gradio 6 components:
- `gr.ImageEditor`: Configured with red brush tool (`#ff2222`) and eraser. When users paint directly on the artwork, the alpha channel of the brush layers is extracted into an exact binary mask.
- `gr.ImageSlider`: Displays an interactive Before/After comparison widget where users slide between the damaged original and the restored artwork.
- `gr.DownloadButton`: Delivers full-fidelity PNG export without UI re-renders.

### 4.2 User Workflows Supported
1. **Workflow A (Archival / Precomputed Mask)**:
   Conservators upload a high-resolution damaged artwork and a paired binary damage mask.
2. **Workflow B (Interactive Conservation / Brush Tool)**:
   For artworks without a digital mask, conservators upload the damaged image into the interactive canvas and paint directly over cracks, mold, or tears. The system automatically binarizes the painted brush strokes as the restoration mask.
3. **Workflow C (Baseline Comparison / Raw UNet)**:
   Users select "Raw UNet Inpainting" to visually inspect the difference between unguided convolutions and mask-guided blending.

---

## 5. Benchmark Performance on Representative Artwork

Verified on Metropolitan Museum Anchor Artwork `247010` (*Wall painting from Room H of the Villa of P. Fannius Synistor*):
- **Native Dimensions**: $3883 \times 3614$ pixels (~14.0 Megapixels).
- **Neural Evaluation Scale**: $1024 \times 953$ pixels.
- **Damage Coverage**: $27.00\%$ physical damage area.
- **Run 1 (Cold Start — Model Load + Inference)**:
  - Total Wall Time: **$3.79\text{ s}$**
  - Pipeline Execution Latency: **$3409.0\text{ ms}$**
- **Run 2 (Warm Start — Cached Model in Memory)**:
  - Total Wall Time: **$3.24\text{ s}$**
  - Pipeline Execution Latency: **$2834.4\text{ ms}$**
- **Preservation Verification**:
  - `undamaged_max_diff`: **`0`**
  - `undamaged_mae`: **`0.000000`**
  - `is_bit_exact`: **`True`**
- **System Memory**: RAM utilization remained stable between $10.84\text{ GB}$ and $12.10\text{ GB}$ with over $3.59\text{ GB}$ available headroom on CPU.

---

## 6. Known Limitations & Recommendations for Phase F.4

1. **Resolution Bounds for Inpainted Patches**:
   While undamaged pixels remain at native resolution (e.g. $3883 \times 3614$), the infilled patches inside large damage regions are upsampled from the 1024-pixel neural output. For gigapixel-scale physical inpainting, tiled/patch-based native inference can be introduced as an optional high-resolution mode.
2. **Lack of Autonomous Defect Segmentation**:
   The current pipeline is semi-automated: it requires either an archival mask file or interactive brush input. Integrating an automated damage-detection module upstream is recommended for fully autonomous batch pipelines.

