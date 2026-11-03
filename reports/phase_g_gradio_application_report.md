# MUESUM — Stage 5: Gradio Restoration Studio Report
**Project**: MUESUM (AI-Powered Historical Artwork Restoration)  
**Date**: 2026-11-04  
**Author**: Lead CV Engineer / Release Architect  
**Status**: VERIFIED & DEPLOYED  

---

## 1. Executive Summary

In Stage 5, the user-facing restoration interface (`app/demo.py`) was upgraded to a comprehensive, museum-grade interactive restoration studio built on Gradio 6.0+.

The application empowers conservators, researchers, and museum professionals to inspect, restore, and export cultural heritage artifacts with interactive before/after comparisons, dual model selection, and archival provenance diagnostics.

---

## 2. Key Architecture & Interface Capabilities

### 2.1 Dual Input Workflows
1. **File Upload Mode (`Tab 1`)**:
   - Accepts high-resolution damaged artwork (`JPG`, `PNG`, `WEBP`).
   - Accepts binary or grayscale damage masks (`PNG`), adhering to the museum convention where `255` (white) = damaged region to be restored, and `0` (black) = authentic surviving artwork.
2. **Interactive Brush Canvas (`Tab 2`)**:
   - Features `gr.ImageEditor` with customizable brush size and eraser tools.
   - Enables users to paint red or white brush strokes directly over physical blemishes, water stains, mold patches, or tears.
   - Automatically rasterizes painted brush layers into a binary damage mask without requiring external graphic software.

### 2.2 Advanced Inpainting & Model Controls
- **Model Selection**:
  - `Experimental Texture Model (Composite Loss — Recommended)`: Optimized for fine surface craquelure, brushstroke textures, and high-frequency edge definition.
  - `Baseline ResNet34-UNet (Epoch 8 Checkpoint)`: Classical L1 loss checkpoint.
- **Restoration Mode**:
  - `Mask-Guided Blending`: Guarantees 100% bit-exact preservation on all surviving authentic artwork.
  - `Raw UNet Inpainting`: Full-image neural reconstruction for comparative baseline analysis.
- **Interior Boundary Feathering**:
  - `Feathered Interior`: Computes Euclidean distance transform strictly inside the damage mask to soften hard border steps with smoothstep Hermite interpolation.
  - `Hard Cutoff`: Traditional hard step assignment.
  - `Feather Radius Slider`: Adjustable from 1 to 15 pixels (default: 3 px).
- **Scale Ceiling**: Aspect-preserving resize ceiling (1024 px standard, 1536 px fine, or native resolution).

### 2.3 Visual Comparisons & Archival Exports
- **Interactive Before/After ImageSlider**: Real-time sliding curtain allowing pixel-level comparison between damaged input and restored output.
- **Tabbed Views**: Independent high-resolution tabs for Restored Artwork, Raw UNet Output, and Damage Mask Overlay.
- **Curatorial Diagnostics Panel**: Live quantitative readout displaying latency (ms), damage coverage (%), undamaged MAE ($0.000000$), max difference ($0$), and mathematical preservation status.
- **Dual Export Buttons**:
  1. `Download Restored Artwork (PNG)`: Lossless full-resolution composite artwork.
  2. `Export Curatorial Diagnostics (JSON)`: Machine-readable JSON recording timestamp, model architecture, checkpoint SHA, hyperparameters, and preservation metrics for conservation archives.

---

## 3. Testing & Verification

Automated test harness `tests/test_gradio_app.py` verifies:
- `test_build_interface_constructs_cleanly`: Validates full Blocks component tree layout.
- `test_extract_inputs_upload_mode`: Validates upload extraction logic.
- `test_extract_inputs_brush_mode`: Validates extraction of painted brush layers to binary mask.
- `test_checkpoint_options_paths`: Validates path resolution for baseline and experimental checkpoints.

```
python -m unittest tests/test_gradio_app.py
Ran 4 tests in 0.329s
OK
```
All tests passed with zero errors.
