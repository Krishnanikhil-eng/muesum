# MUESUM — Model Card
**Model Name**: MUESUM ResNet34-UNet Artwork Inpainting & Restoration Engine  
**Version**: 2.0  
**Format**: PyTorch (`.pth`)  
**Standard**: Follows Mitchell et al. (2019), *Model Cards for Model Reporting*  

---

## 1. Model Details

- **Model Architecture**: ResNet34-UNet (Deep convolutional encoder with residual bottleneck skip connections to an upsampling decoder).
- **Parameters**: **28,235,459 total trainable parameters**.
- **Model Variants**:
  1. **Baseline Model (`checkpoints/baseline_run/best_model.pth`)**: Trained for 10 epochs using L1 loss; checkpoint selected at Epoch 8 (`val_ssim: 0.4908`).
  2. **Experimental Texture Model (`checkpoints/experimental_texture_model/best_model.pth`)**: Fine-tuned from Epoch 8 weights using Multi-Term Composite Loss (Masked L1 + Sobel Edge + TV + VGG Perceptual; `val_ssim: 0.5544`).
- **Input Format**: 3-channel RGB image normalized to $[0.0, 1.0]$.
- **Output Format**: 3-channel RGB image in $[0.0, 1.0]$ representing restored substrate.
- **License**: Creative Commons Zero v1.0 Universal (CC0-1.0).

---

## 2. Intended Use

### 2.1 Primary Intended Applications
- **Digital Conservation**: Virtual restoration of aged, damaged, or torn cultural heritage artifacts for digital archives and museum exhibits.
- **Curatorial Research**: Visual hypothesis testing of artwork appearance prior to physical degradation.
- **Public Interactive Education**: Interactive before/after museum kiosks and research exhibitions.

### 2.2 Out-of-Scope & Unintended Uses
- **Physical Art Authentication**: MUESUM is a digital inpainting system and must never be used to authenticate physical antiquities or forge provenance records.
- **Automated Physical Inpainting**: Generating toolpaths for direct automated repainting of physical canvases without expert conservator oversight.
- **Deceptive Restoration**: Claiming neural restorations represent verified historical truth without curatorial disclosure.

---

## 3. Factors, Metrics & Quantitative Evaluation

### 3.1 Evaluated Factors
- **Artwork Category**: 10 distinct heritage medium domains (Paintings, Drawings, Photographs, Sculptures, Bronzes, Textiles, Books, Manuscripts, Ceramics, Metalwork).
- **Degradation Severity**: Mask damage coverage ranging from $4.3\%$ to $36.8\%$ of surface area.
- **Resolution**: Native resolutions from $600\text{ px}$ to $4000\text{ px}$.

### 3.2 Evaluation Metrics
- **Peak Signal-to-Noise Ratio (PSNR)**: Logarithmic measure of pixel fidelity.
- **Structural Similarity Index Measure (SSIM)**: Luminance, contrast, and structural texture correlation.
- **Mean Absolute Error (MAE)**: Average absolute per-pixel deviation.
- **Bit-Exact Preservation MAE**: Measured strictly on surviving authentic pixels where mask $= 0$.

### 3.3 Quantitative Performance Summary (Independent 40-Artwork Holdout)
| Metric | Damaged Input | Baseline Model (Epoch 8) | Experimental Texture Model |
|---|:---:|:---:|:---:|
| **Mean PSNR (dB)** | 23.53 dB | 25.71 dB | **25.88 dB** |
| **Mean SSIM** | 0.8651 | 0.8988 | **0.9022** |
| **Mean MAE** | 10.42 | 6.81 | **6.64** |
| **Bit-Exact Preservation** | — | **100% Bit-Exact (0 diff)** | **100% Bit-Exact (0 diff)** |
| **Undamaged Substrate MAE**| — | **0.000000** | **0.000000** |

---

## 4. Ethical Considerations & Curatorial Safeguards

1. **Conservation Invariance Principle**:
   - Digital inpainting in cultural heritage must adhere to strict non-invasive principles: surviving historical pigment must remain completely untouched. MUESUM enforces this mathematically via mask-guided blending, guaranteeing that undamaged regions are bit-for-bit identical to the original image ($MAE = 0.000000$).
2. **Provenance Tracking**:
   - The Gradio studio exports a cryptographically verifiable JSON provenance file recording model architecture, checkpoint version, input parameters, and exact pixel preservation statistics.
3. **No Autonomous Mask Fabrication**:
   - The model does not invent or guess where damage exists; human conservator oversight via explicit mask files or interactive painting is mandatory.
