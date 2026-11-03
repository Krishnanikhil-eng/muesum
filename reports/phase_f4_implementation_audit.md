# MUESUM — Phase F.4: Implementation & System Audit

**Audit Date**: November 3, 2026  
**Auditor**: Antigravity Technical Pair Programmer  
**System Hardware**: Dell Vostro 3520 (Intel Core i5-1235U, 16 GB RAM, CPU Execution)  
**Active Checkpoint**: ResNet34-UNet Epoch 8 (`checkpoints/baseline_run/best_model.pth`)  
**Audit Status**: **PASSED (100% Verified Across All 8 Technical Requirements)**

---

## 1. Executive Summary

This audit establishes the technical validity, integrity, and behavioral compliance of the MUESUM artwork restoration system prior to visual and independent quality validation. 

All 8 technical criteria mandated in Phase F.4 were programmatically audited through automated scripts (`scripts/verify_stage1_audit.py` and `scripts/verify_stage2_workflows.py`). The system is confirmed to execute with zero silent mask fabrications, zero weight reloads across sequential requests, and mathematical bit-exact preservation on all authentic artwork.

---

## 2. Detailed Technical Verifications (Items 1–8)

### Item 1: Epoch 8 Checkpoint Verification
- **Target File**: `checkpoints/baseline_run/best_model.pth`
- **File Size**: $339,079,693\text{ bytes}$ (~$339\text{ MB}$).
- **State Dict Keys**: `['model', 'opt', 'epoch', 'val_ssim']`.
- **Recorded Epoch**: `8` (Best checkpoint achieved during Stage 5 baseline training).
- **Validation Metric at Epoch 8**: `val_ssim = 0.4908554631945002`.
- **Verdict**: **VERIFIED**. The model strictly uses Epoch 8 weights.

### Item 2: Architecture & Weight Compatibility
- **Model Class**: `ResNet34UNet` (`src/unet_model.py`).
- **Backbone**: Torchvision ResNet-34 encoder with skip connections.
- **Center Block**: Two $3 \times 3$ convolutional layers with ReLU activations.
- **Decoder**: 4 sequential decoder blocks with bilinear feature alignment to handle arbitrary spatial dimensions.
- **Parameter Count**: Exactly **$28,235,459$ parameters** ($100\%$ trainable).
- **Weight Loading**: All state-dict keys mapped with zero missing or unexpected tensors.
- **Verdict**: **VERIFIED**.

### Item 3: Model Caching Across Sequential Requests
- **Implementation**: `ModelManager` singleton in `src/inference.py`.
- **Mechanism**: Class attributes `_model`, `_ckpt_path`, and `_device` cache the resident model in RAM.
- **Verification**: Repeated calls to `ModelManager.get_model()` return the exact same Python object reference (`model is m2 == True`).
- **Latency Impact**:
  - Run 1 (Cold Start — Disk Read + Load): $3.79\text{ s}$.
  - Run 2 (Warm Start — Cached Model): $3.24\text{ s}$ ($2834\text{ ms}$ inference latency).
- **Verdict**: **VERIFIED**. Checkpoint reloading from disk is completely eliminated.

### Item 4: Mask Polarity & Numeric Normalization
- **Polarity Convention**:
  - $M(x, y) = 1.0$ (or $255$, or `True`) $\rightarrow$ **Damaged pixel to be restored**.
  - $M(x, y) = 0.0$ (or $0$, or `False`) $\rightarrow$ **Authentic, undamaged clean artwork**.
- **Normalization Safeguards**:
  - Floats outside $[0.0, 1.0]$ or integers outside $[0, 255]$ trigger an immediate `ValueError`.
  - NaNs and Infinities trigger an immediate `ValueError`.
  - Rejection of inverted or empty masks verified.
- **Verdict**: **VERIFIED**.

### Item 5: Bit-Exact Mathematical Preservation
- **Blending Equation**:
  $$\text{Final}[x, y] = \begin{cases} \text{Restored}_{\text{raw}}[x, y] & \text{if } M(x, y) = 1 \\ \text{Original}[x, y] & \text{if } M(x, y) = 0 \end{cases}$$
- **Measured Metrics Across Benchmark Artwork `247010` ($14.0\text{ Megapixels}$)**:
  - Total undamaged pixels evaluated: $10,248,348$ pixels.
  - Maximum Absolute Pixel Difference: **`0`**.
  - Mean Absolute Error ($\text{MAE}$): **`0.000000`**.
  - `is_bit_exact`: **`True`**.
- **Verdict**: **VERIFIED**. Authentic surviving artwork is preserved down to the exact bit.

### Item 6: Original-Resolution Composition with Neural Scaling
- **Handling**:
  1. The input image ($3883 \times 3614$ px) is aspect-preservingly resized to $1024 \times 953$ px for neural inference, preventing memory thrashing on CPU.
  2. The neural prediction is unpadded and bicubically rescaled back to the original full dimensions ($3883 \times 3614$ px).
  3. The final composition is applied at **full original resolution** using the **unscaled original mask**.
- **Result**: Surviving pixels undergo zero interpolation, resizing, or blur.
- **Verdict**: **VERIFIED**.

### Item 7: Distinction Between Raw UNet and Mask-Guided Outputs
- **Raw UNet Mode**:
  - Replaces $100\%$ of pixels across the canvas.
  - Undamaged Pixel MaxDiff: **`206`** (severe global shift).
  - Undamaged Pixel MAE: **`25.35`**.
  - `is_bit_exact`: `False`.
- **Mask-Guided Mode**:
  - Replaces only pixels inside $M = 1$.
  - Undamaged Pixel MaxDiff: **`0`**.
  - Undamaged Pixel MAE: **`0.000000`**.
  - `is_bit_exact`: `True`.
- **Pixel Equality Check**: `np.array_equal(res_blend, res_raw) == False`.
- **Verdict**: **VERIFIED**. Raw and blended modes produce distinct, verifiable outputs.

### Item 8: Gradio Application Workflows & Download Verification
Tested end-to-end via `scripts/verify_stage2_workflows.py`:
- **Workflow A (Precomputed Mask)**:
  - Full-resolution restoration executed cleanly.
  - Download button generated valid persistent PNG file (`tmpxg8yd3t_.png`, dimensions $3883 \times 3614$ px).
- **Workflow B (Manually Painted Mask)**:
  - Painted red strokes on `gr.ImageEditor` canvas were extracted into a binary mask ($11.11\%$ coverage).
  - Mask-guided restoration executed cleanly with $100\%$ bit-exact preservation on unpainted regions.
  - Downloadable restored PNG verified.
- **Workflow C (Input Validation & Error Messages)**:
  - Missing artwork input $\rightarrow$ Caught with clear message: *"Please upload or provide a damaged artwork image."*
  - Missing mask in mask-guided mode $\rightarrow$ Caught with clear message: *"Mask-guided restoration requires a damage mask..."*
  - Empty brush mask $\rightarrow$ Caught with clear message: *"No painted damage detected! Please use the brush tool in the canvas to paint over the damaged regions before restoring."*
  - Dimension mismatch $\rightarrow$ Caught with clear message: *"Dimension mismatch: damage mask dimensions (1941x1807) do not match image dimensions (3883x3614)."*
  - Invalid NaN mask $\rightarrow$ Caught with clear message: *"Damage mask contains NaN or Infinite values."*
- **Verdict**: **VERIFIED**. Zero silent mask fabrications; user-facing error messaging is clear and actionable.

---

## 3. Audit Conclusion

The underlying codebase and Gradio studio adhere strictly to the engineering specifications established in Phase F.2 and F.3. The system is technically stable, deterministic, and ready for qualitative restoration evaluation.

