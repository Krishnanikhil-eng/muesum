# Historical Artwork Restoration: Baseline Evaluation Methodology

## 1. Executive Overview & Experimental Setup
This document details the scientific evaluation methodology applied to the baseline **ResNet34-UNet** historical artwork restoration model trained in Stage 5, Phase D.

- **Evaluated Checkpoint**: `checkpoints/baseline_run/best_model.pth`
  - Selected strictly on validation performance: **Epoch 8** (`val_ssim: 0.490855`, `val_psnr: 19.5044 dB`).
- **Test Dataset**: `data/damaged/test/` (200 museum artwork images from The Metropolitan Museum of Art Collection).
- **Split File**: `data/splits/test_ids.csv` (100% disjoint from training and validation sets; zero data leakage).
- **Clean Ground Truth**: `data/clean/` (original museum public domain artwork images).
- **Hardware Platform**: Intel Core i5-1235U CPU (10 physical cores / 12 logical threads), 16 GB RAM, PyTorch 2.11.0+cpu.

---

## 2. Multi-Tier Evaluation Protocol

To balance spatial accuracy, global composition coherence, and hardware memory safety, three sequential evaluation tiers were conducted:

### Evaluation Tier 1: 256×256 Center-Crop Benchmark ($N = 200$)
- **Objective**: Direct apples-to-apples performance comparison against the validation training protocol.
- **Protocol**:
  1. For each test artwork, extract a deterministic center crop of size $256 \times 256$ pixels from the clean image, simulated damaged image, and binary damage mask.
  2. Compute initial damage metrics: $\text{PSNR}_{\text{dmg}}$ and $\text{SSIM}_{\text{dmg}}$ between clean and damaged crops.
  3. Feed the damaged crop into `ResNet34UNet` (`torch.no_grad()`).
  4. Compute restoration metrics: $\text{PSNR}_{\text{res}}$ and $\text{SSIM}_{\text{res}}$ between clean and restored crops.
  5. Calculate restoration improvement: $\Delta\text{PSNR} = \text{PSNR}_{\text{res}} - \text{PSNR}_{\text{dmg}}$ and $\Delta\text{SSIM} = \text{SSIM}_{\text{res}} - \text{SSIM}_{\text{dmg}}$.
  6. Measure inference latency per crop.

### Evaluation Tier 2: 1024-Pixel Resized Benchmark ($N = 200$)
- **Objective**: Full-artwork holistic composition restoration across the entire test corpus while maintaining safe memory bounds.
- **Protocol**:
  1. Rescale each artwork with aspect-ratio preservation so that $\max(\text{Width}, \text{Height}) = 1024\text{ px}$.
  2. Apply identical bilinear interpolation to clean and damaged images; apply nearest-neighbor interpolation to binary damage masks.
  3. Pad dimensions to multiples of 32 using reflection padding (`pad_to_multiple_of_32`).
  4. Perform forward inference through `ResNet34UNet`.
  5. Unpad the output to exact resized dimensions (`unpad`).
  6. Compute all six primary metrics ($\text{PSNR}_{\text{dmg}}$, $\text{SSIM}_{\text{dmg}}$, $\text{PSNR}_{\text{res}}$, $\text{SSIM}_{\text{res}}$, $\Delta\text{PSNR}$, $\Delta\text{SSIM}$), damage coverage, and latency.

### Evaluation Tier 3: Original-Resolution Benchmark ($N = 10$)
- **Objective**: Measure real-world latency, memory footprint, and visual fidelity on native multi-megapixel artwork files.
- **Selection Criteria**: Pre-selected based strictly on metadata categories and curatorial departments before inference (zero performance cherry-picking):
  1. **Anchor Sample**: `247010` (Greek & Roman Wall Painting, Boscoreale, 14.03 MP)
  2. **Painting**: `48953` (Asian Art, 9.40 MP)
  3. **Codex / Manuscript**: `451301` (Islamic Art, 11.22 MP)
  4. **Textile**: `451019` (Woven Tapestry Fragment, Islamic Art, 10.86 MP)
  5. **Drawing**: `336663` (Allegorical Figure, Drawings & Prints, 3.11 MP)
  6. **Sculpture**: `466379` (Enthroned King, Medieval Art, 11.07 MP)
  7. **Print**: `403800` (Ancient Philosophers, Drawings & Prints, 11.53 MP)
  8. **Book**: `354833` (Works of Virgil, Drawings & Prints, 11.76 MP)
  9. **Photograph**: `283234` (Sculpture Study, Photographs, 11.29 MP)
  10. **Costume**: `314620` (Four-Cornered Hat, Rockefeller Wing, 14.52 MP)
- **Protocol**:
  - Run sequentially with explicit garbage collection between samples.
  - Measure per-image latency and RAM differential.
  - Save full side-by-side composite visual comparisons `[Original Clean | Damaged Input | Model Restored]`.

---

## 3. Mathematical Metric Definitions & Verification

### Peak Signal-to-Noise Ratio (PSNR)
$$\text{MSE} = \frac{1}{3 \cdot H \cdot W} \sum_{c=1}^3 \sum_{i=1}^H \sum_{j=1}^W (I_1(c, i, j) - I_2(c, i, j))^2$$
$$\text{PSNR} = 20 \log_{10}\left(\frac{255.0}{\sqrt{\text{MSE}}}\right) \quad (\text{dB})$$
- Inputs: 8-bit RGB images cast to `float64` in $[0, 255]$.
- Identical images ($\text{MSE} = 0$) clamp to $100.0\text{ dB}$.

### Structural Similarity Index Measure (SSIM)
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
- Parameters: Gaussian window ($11 \times 11$, $\sigma = 1.5$), $L = 255.0$, $K_1 = 0.01$, $K_2 = 0.03$.
- Evaluated independently on R, G, B channels and averaged: $\text{SSIM} = \frac{1}{3}(\text{SSIM}_R + \text{SSIM}_G + \text{SSIM}_B)$.
- Scale: Strictly fractional in $[0.0, 1.0]$.

---

## 4. Methodological Insights & Limitations
1. **Unmasked Whole-Image Evaluation**:
   In our baseline evaluation, `ResNet34UNet` processes the entire damaged image and produces a complete reconstructed image. Because simulated damage covers $\sim 19.7\%$ of an artwork's canvas on average, $\sim 80.3\%$ of the damaged image is identical to the ground truth. Passing clean pixels through the baseline UNet introduces minor global smoothing and color shifts, which lower whole-image PSNR compared to an input where $80\%$ of pixels are bit-exact.
2. **Post-Processing Recommendation**:
   In production museum restoration workflows, mask-guided alpha blending (or Poisson blending) is recommended:
   $$I_{\text{final}} = I_{\text{damaged}} \odot (1 - M) + I_{\text{restored}} \odot M$$
   This preserves 100% of the authentic surviving artwork while reconstructing only the verified loss regions.

