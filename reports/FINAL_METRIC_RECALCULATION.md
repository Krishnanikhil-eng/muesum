# MUESUM — Independent Metric Recalculation & Statistical Audit

**Audit Date**: November 4, 2026  
**Auditor**: Lead Computer Vision & Evaluation Auditor (MUESUM Autonomous Task Force)  
**Dataset Under Audit**: `reports/final_independent_evaluation/holdout_metrics.csv`  
**Artwork Corpus**: 40 Independent Holdout Artworks from The Metropolitan Museum of Art (Zero ID overlap)  
**Status**: VERIFIED & RECONCILED  

---

## 1. Executive Summary & Verification Finding

Every individual metric record in `reports/final_independent_evaluation/holdout_metrics.csv` has been independently recalculated from the raw evaluation data using Python 3.11 with Pandas and NumPy.

### Key Recalculation Findings:
1. **Overall Mean PSNR & SSIM**:
   - **Damaged Input**: Mean PSNR = **23.52 dB** (23.5242 dB), Mean SSIM = **0.8919**
   - **Baseline Model (Epoch 8)**: Mean PSNR = **25.71 dB** (25.7055 dB), Mean SSIM = **0.8988**
   - **Experimental Texture Model**: Mean PSNR = **25.88 dB** (25.8782 dB), Mean SSIM = **0.9022**
2. **Mean Delta Improvements**:
   - **Experimental vs. Damaged**: **+2.35 dB** (2.3540 dB) arithmetic mean improvement.
   - **Baseline vs. Damaged**: **+2.18 dB** (2.1813 dB) arithmetic mean improvement.
   - **Experimental vs. Baseline**: **+0.17 dB** (0.1727 dB) arithmetic mean improvement across all 40 artworks.
3. **Discrepancy in Previous Summary Table**:
   - An earlier report draft presented approximate category values that did not strictly reflect the arithmetic mean of the 4 samples per category. 
   - This document provides the authoritative, exact mathematical breakdown per category and explains the aggregation methodology.

---

## 2. Aggregation Methodology

In computer vision and image processing literature, two primary methods exist for aggregating PSNR across an image corpus:
1. **Per-Image Arithmetic Mean (Standard Evaluation Protocol)**:
   \[
   \overline{\text{PSNR}} = \frac{1}{N} \sum_{i=1}^N \text{PSNR}(I_i, \hat{I}_i)
   \]
   This protocol evaluates each artwork as an independent conservation subject and calculates sample-level restoration quality.
2. **Logarithmic Pooled MSE**:
   \[
   \text{PSNR}_{\text{pooled}} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\frac{1}{N}\sum_{i=1}^N \text{MSE}_i}\right)
   \]

**MUESUM Evaluation Standard**: MUESUM adheres to the **Per-Image Arithmetic Mean Protocol (1)** because artworks vary in resolution, aspect ratio, and curatorial medium; pooling MSE disproportionately weights massive pixel rasters over delicate small-format drawings.

---

## 3. Comprehensive Category-Wise Recalculation Table

All 10 curatorial categories contain exactly 4 public-domain artworks evaluated at 1024 max dimension with mask-guided interior feathering ($r=3$):

| Artwork Classification | Samples | Damaged PSNR (dB) | Baseline PSNR (dB) | Experimental PSNR (dB) | Exp vs Damaged $\Delta$PSNR | Exp vs Base $\Delta$PSNR | Damaged SSIM | Baseline SSIM | Experimental SSIM | Exp vs Damaged $\Delta$SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Books** | 4 | 22.35 | 25.41 | **24.60** | **+2.25 dB** | -0.80 dB | 0.8773 | 0.8958 | **0.9012** | +0.0239 |
| **Bronzes** | 4 | 25.89 | 27.93 | **28.20** | **+2.32 dB** | +0.28 dB | 0.8383 | 0.9067 | **0.9121** | +0.0738 |
| **Ceramics** | 4 | 25.85 | 29.15 | **30.18** | **+4.33 dB** | +1.04 dB | 0.9371 | 0.9174 | **0.9194** | -0.0177 |
| **Drawings** | 4 | 19.92 | 24.13 | **23.37** | **+3.45 dB** | -0.76 dB | 0.8710 | 0.8869 | **0.8858** | +0.0148 |
| **Manuscripts** | 4 | 25.67 | 27.36 | **27.02** | **+1.35 dB** | -0.34 dB | 0.9054 | 0.9004 | **0.9019** | -0.0035 |
| **Metalwork** | 4 | 21.24 | 22.94 | **22.58** | **+1.34 dB** | -0.36 dB | 0.8872 | 0.8799 | **0.8784** | -0.0088 |
| **Paintings** | 4 | 21.82 | 24.19 | **24.66** | **+2.84 dB** | +0.47 dB | 0.8530 | 0.8851 | **0.8959** | +0.0428 |
| **Photographs** | 4 | 22.82 | 25.24 | **26.45** | **+3.64 dB** | +1.22 dB | 0.9129 | 0.9015 | **0.9107** | -0.0023 |
| **Sculptures** | 4 | 24.91 | 25.50 | **25.34** | **+0.43 dB** | -0.16 dB | 0.8905 | 0.9399 | **0.9397** | +0.0492 |
| **Textiles** | 4 | 24.78 | 25.23 | **26.37** | **+1.58 dB** | +1.14 dB | 0.9464 | 0.8741 | **0.8770** | -0.0694 |
| **OVERALL (MEAN)** | **40** | **23.52** | **25.71** | **25.88** | **+2.35 dB** | **+0.17 dB** | **0.8919** | **0.8988** | **0.9022** | **+0.0103** |

---

## 4. Overall Dataset Distribution Statistics

| Metric | Mean | Median | Standard Dev | Minimum | Maximum |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Damaged Input PSNR** | 23.52 dB | 23.60 dB | 4.11 dB | 15.11 dB | 34.81 dB |
| **Baseline Model PSNR** | 25.71 dB | 25.95 dB | 4.02 dB | 17.81 dB | 37.93 dB |
| **Experimental Model PSNR** | 25.88 dB | 25.43 dB | 4.30 dB | 17.72 dB | 38.63 dB |
| **Damaged Input SSIM** | 0.8919 | 0.9040 | 0.0673 | 0.7295 | 0.9849 |
| **Baseline Model SSIM** | 0.8988 | 0.9027 | 0.0420 | 0.8186 | 0.9649 |
| **Experimental Model SSIM** | 0.9022 | 0.9133 | 0.0420 | 0.8161 | 0.9670 |
| **Exp vs Damaged $\Delta$PSNR** | **+2.35 dB** | **+2.57 dB** | 2.16 dB | -1.42 dB | +7.96 dB |
| **Exp vs Base $\Delta$PSNR** | **+0.17 dB** | **+0.15 dB** | 1.11 dB | -2.05 dB | +2.61 dB |

---

## 5. Domain Specialization Analysis

The recalculated category metrics reveal a clear structural pattern:
1. **High-Texture Domain Superiority (Experimental Model Advantage)**:
   - **Ceramics**: **+4.33 dB** over damaged, **+1.04 dB** over baseline.
   - **Photographs**: **+3.64 dB** over damaged, **+1.22 dB** over baseline.
   - **Textiles**: **+1.58 dB** over damaged, **+1.14 dB** over baseline.
   - **Paintings**: **+2.84 dB** over damaged, **+0.47 dB** over baseline.
   - *Reason*: Sobel edge loss ($\lambda=0.15$) and VGG perceptual features ($\lambda=0.05$) preserve fine micro-structures (craquelure, canvas weave, photographic grain, and glaze specularities) that pure L1 loss penalizes towards blurred averages.
2. **Monochromatic & Planar Media (Baseline Model Advantage)**:
   - **Books & Manuscripts**: Baseline L1 model achieves higher PSNR on printed text and ink-on-parchment (+0.80 dB and +0.34 dB over experimental).
   - *Reason*: Pure L1 loss optimizes strictly for pixel-value variance without introducing perceptual high-frequency hallucination, which is advantageous on uniform white or parchment paper backgrounds.

