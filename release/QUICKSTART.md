# MUESUM — Quickstart & Operator Guide

**Version**: 1.0.0 Production Release  
**Target Hardware**: Standard CPU workstation or laptop (16 GB RAM recommended) / CUDA optional  

---

## 1. Quick Launch (Zero Setup)

### Windows One-Click Runner
Double-click `run_muesum.bat` in the project root, or execute from PowerShell / Command Prompt:

```cmd
# Launch the interactive Gradio Restoration Studio (default port 7860)
run_muesum.bat

# Or explicitly:
run_muesum.bat demo
```

Once launched, open your web browser to:  
👉 **`http://127.0.0.1:7860`**

---

## 2. CLI Headless Inpainting & Restoration

Restore any historical artwork from the command line using the packaged sample assets or your own images:

```cmd
# Using the Windows batch helper:
run_muesum.bat restore release/sample_artwork.jpg release/sample_mask.png restored_output.png

# Or directly with Python:
python src/inference.py --image release/sample_artwork.jpg --mask release/sample_mask.png --output restored_output.png --blend_mode mask_guided --feather_radius 3
```

### CLI Options
- `--checkpoint`: Path to custom model weights (defaults to `checkpoints/experimental_texture_model/best_model.pth`).
- `--blend_mode`: `mask_guided` (guarantees bit-exact untouched background) or `raw_unet`.
- `--feather_radius`: Interior transition radius in pixels (default: `3`).
- `--max_dimension`: Resizes longest edge for low-memory environments (default: `1024`).

---

## 3. Running Automated Tests & QA

Verify system integrity and all 29 automated unit and regression tests:

```cmd
# Using the Windows batch helper:
run_muesum.bat test

# Or directly with Python:
python tests/run_all_tests.py
```

Expected output:
```
Total Unit Tests Run:      29
Tests Passed:              29 / 29 (100.0%)
Legacy Pipeline Smoke:     PASSED
QA STATUS: ALL SYSTEMS VERIFIED AND PASSING
```

---

## 4. Gradio Restoration Studio Overview

The web studio features an interactive curatorial restoration workflow:

1. **Model Selection**: Switch seamlessly between the **Experimental Texture Model** (composite loss, optimized for canvas/pigment micro-texture) and the **Baseline ResNet34-UNet** (Epoch 8 benchmark).
2. **Dual Input Modes**:
   - **Upload Image + Mask**: Upload an existing damaged artwork alongside a binary damage mask (white = damaged, black = surviving substrate).
   - **Interactive Inpainting Brush**: Upload an image and directly brush over damaged cracks, tears, or stains using Gradio's canvas editor.
3. **Conservation Controls**:
   - **Blending Mode**: Toggle between Mask-Guided Inpainting (preserves authentic substrate bit-exact) and Raw UNet output.
   - **Interior Feathering**: Adjust the interior smoothing radius (0–15 px) to eliminate boundary seams without bleeding into original paint.
4. **Interactive Comparison & Export**:
   - **Before/After ImageSlider**: Drag slider interactively to inspect inpainted lacunae against the original damaged input.
   - **Curatorial Audit Report**: Export diagnostics including latency, damage mask coverage percentage, damaged/clean pixel counts, and MAE bit-exact verification.
   - **Full-Resolution Download**: Save the restored artwork as a lossless PNG.

---

## 5. System Requirements & Troubleshooting

- **Python**: 3.8 to 3.11.
- **PyTorch**: 2.0+ (CPU or CUDA).
- **Memory Pressure**: On machines with 16 GB RAM, the pipeline strictly processes images sequentially with batch size 1, automatic garbage collection, and peak memory capped below 1.5 GB.
- **Port Conflict**: If port 7860 is occupied, Gradio automatically binds to 7861, or you can specify `--port 7865` in `app/demo.py`.
