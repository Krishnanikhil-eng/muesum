# 🏛️ MUESUM — AI Historical Artwork Restoration System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.0+-orange.svg)](https://gradio.app/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Tests: 23 Passed](https://img.shields.io/badge/Tests-23%2F23%20Passed-brightgreen.svg)](tests/run_all_tests.py)
[![Preservation: Bit-Exact](https://img.shields.io/badge/Preservation-100%25%20Bit--Exact-success.svg)](reports/phase_g_boundary_feathering_report.md)

**MUESUM** is an enterprise-grade artificial intelligence platform engineered for the digital conservation, repair, and inpainting of cultural heritage artifacts degraded by physical tears, craquelure, liquid stains, mold spots, and fading.

Built upon **The Metropolitan Museum of Art Open Access Collection**, MUESUM combines a deep **ResNet34-UNet** neural architecture with strict archival conservation principles, guaranteeing **100% bit-exact mathematical preservation of all surviving authentic substrate** alongside distance-transform interior seam feathering.

---

## 🌟 Key Innovations

1. **100% Bit-Exact Authentic Preservation**:
   - In mask-guided restoration mode, authentic surviving substrate is mathematically locked:
     $$\forall (x, y) \notin \mathcal{M}, \quad I_{\text{restored}}(x, y) \equiv I_{\text{original}}(x, y) \implies \text{MAE} = 0.000000, \; \text{MaxDiff} = 0$$
   - The neural network generates pixels *strictly* within the damaged region $\mathcal{M}$.
2. **Interior Boundary Seam Feathering (`src/feathering.py`)**:
   - Computes Euclidean distance transform $D(x, y)$ strictly *inside* the damage mask.
   - Smoothstep Hermite polynomial transitions ($3t^2 - 2t^3$) eliminate harsh border steps without leaking into authentic substrate.
3. **Multi-Term Composite Texture Loss (`src/composite_loss.py`)**:
   - Multi-scale objective balancing Masked L1, Sobel High-Frequency Edge Loss, Total Variation (TV), and VGG-19 Deep Perceptual activations.
   - Boosts structural similarity by **+12.96%** (SSIM $0.4908 \to 0.5544$), resolving the over-smoothing artifacts typical of pure L1 loss.
4. **Interactive Gradio 6.0+ Restoration Studio (`app/demo.py`)**:
   - Dual inputs: File Upload (Image + Mask) and Interactive Brush Canvas (paint directly on blemishes).
   - Real-time sliding curtain before/after comparisons (`ImageSlider`).
   - One-click export of full-resolution restored PNGs and archival JSON provenance reports.
5. **Rigorous Independent Holdout Validation (`reports/final_independent_evaluation/`)**:
   - 40 public-domain artworks across 10 categories acquired from Met API v1.1.
   - Quarantined with strictly 0 ID overlap against project training data.

---

## 🏛️ System Architecture

```
                                 [Damaged Artwork]
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
          [Damage Mask File]                       [Interactive Brush Tool]
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                             [Damage Mask Binary M]
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
[Neural Inpainting Branch]                                   [Authentic Pass-Through]
  ResNet34-UNet (28.2M Params)                                 Untouched Surviving Canvas
  Aspect-Preserving 1024 Neural Scale                          (M == 0)
        │                                                                 │
        ▼                                                                 │
[Interior Distance Transform]                                             │
  Hermite Smoothstep Alpha α(x, y)                                        │
        │                                                                 │
        └────────────────────────────────┬────────────────────────────────┘
                                         ▼
                     [Composite Output: I_final = (1-α)*I_orig + α*I_neural]
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
[Curatorial Diagnostics Report]                               [Interactive ImageSlider]
  MAE: 0.000000 | MaxDiff: 0                                   Full-Resolution PNG & JSON
```

---

## 📊 Benchmark & Evaluation Results

Evaluated across the 40 untouched independent holdout artworks from The Metropolitan Museum of Art (10 categories: Paintings, Drawings, Photographs, Sculptures, Bronzes, Textiles, Books, Manuscripts, Ceramics, Metalwork):

| Metric | Damaged Input | Baseline Model (Epoch 8) | Experimental Texture Model | Improvement (Exp vs Base) |
|---|:---:|:---:|:---:|:---:|
| **Mean PSNR** | 23.53 dB | 25.71 dB | **25.88 dB** | **+0.17 dB** |
| **Mean SSIM** | 0.8651 | 0.8988 | **0.9022** | **+0.0034** |
| **Mean MAE** | 10.42 | 6.81 | **6.64** | **-0.17** |
| **Undamaged Preservation** | — | **100% Bit-Exact (0 diff)**| **100% Bit-Exact (0 diff)**| **0.000000 MAE** |
| **Texture Mediums (Textiles/Photos/Ceramics)** | — | 26.53 dB | **27.67 dB** | **+1.14 dB** |

---

## 🚀 Quick Start

### 1. Installation
Clone repository and install dependencies in Python 3.10+:

```bash
git clone https://github.com/your-username/MUESUM.git
cd MUESUM
pip install -r requirements.txt
```

### 2. Launch Gradio Restoration Studio
Launch the web interface locally on `http://localhost:7860`:

```bash
python app/demo.py
# Or on Windows using one-click release launcher:
run_muesum.bat
```

### 3. CLI Inpainting & Restoration
Restore artworks directly from the command line:

```bash
# Mask-guided restoration with interior seam feathering
python src/inference.py \
  --inp_path data/independent_holdout/damaged/12464.jpg \
  --mask_path data/independent_holdout/masks/12464_mask.png \
  --out_path outputs/restored_12464.png \
  --mode mask_guided \
  --feather_mode feathered \
  --feather_radius 3

# Raw UNet unguided baseline inpainting
python src/inference.py \
  --inp_path data/independent_holdout/damaged/12464.jpg \
  --out_path outputs/raw_12464.png \
  --mode raw_unet
```

### 4. Execute Automated Test Suite
Run all 23 unit and regression tests:

```bash
python tests/run_all_tests.py
```

---

## 📁 Repository Structure

```
MUESUM/
├── app/
│   └── demo.py                       # Gradio 6.0+ Restoration Studio
├── checkpoints/
│   ├── baseline_run/                 # Baseline Epoch 8 best checkpoint (Read-Only)
│   └── experimental_texture_model/   # Composite Loss fine-tuned checkpoint & metrics
├── data/
│   ├── clean/                        # Met Museum authentic public-domain images
│   ├── damaged/                      # Multi-modal synthetically damaged sets (train/val/test)
│   ├── independent_holdout/          # 40 independent holdout artworks & masks
│   └── splits/                       # Train/val/test split manifests
├── docs/
│   ├── ARCHITECTURE.md               # Technical system design & data flow
│   ├── MODEL_CARD.md                 # Model Card following Mitchell et al.
│   ├── DATASET_CARD.md               # Dataset Card following Gebru et al.
│   └── REPRODUCIBILITY.md            # Exact step-by-step reproduction guide
├── reports/
│   ├── final_independent_evaluation/ # Holdout evaluation CSV & visual comparisons
│   └── phase_g_*.md                  # Phase engineering & audit reports
├── src/
│   ├── composite_loss.py             # Masked L1 + Sobel + TV + VGG loss
│   ├── dataset.py                    # PyTorch ArtifactDataset loader & unpadder
│   ├── feathering.py                 # Interior distance transform feathering
│   ├── inference.py                  # Reusable enterprise restoration engine
│   ├── unet_model.py                 # ResNet34-UNet backbone (28.2M params)
│   └── utils.py                      # PSNR, SSIM, and image processing helpers
├── tests/
│   ├── run_all_tests.py              # Master QA test runner
│   ├── test_composite_loss.py        # Loss formulation & gradient tests
│   ├── test_feathering.py            # Seam feathering & bit-exact tests
│   ├── test_gradio_app.py            # Gradio Blocks UI integration tests
│   └── test_inference_pipeline.py    # Restoration engine unit tests
├── requirements.txt                  # Python dependencies
└── run_muesum.bat                    # One-click Windows execution launcher
```

---

## ⚖️ Curatorial Ethics & Mathematical Guarantee

In digital cultural heritage conservation, hallucinated additions must never corrupt surviving original pigment:
1. **Mathematical Invariant**: Any pixel designated as clean substrate ($M = 0$) is completely protected from convolutional shifts and rounding errors.
2. **No Automated Fabrication**: MUESUM requires explicit damage masks (from archival maps or user brush strokes); the system never hallucinates damage boundaries autonomously.
3. **Transparent Auditing**: Every restoration run exports a cryptographic parameter JSON recording latency, resolution, checkpoint SHA, and pixel deviation metrics.

---

## 📜 License & Acknowledgements
- **Dataset**: Public domain artworks provided courtesy of [The Metropolitan Museum of Art Collection API](https://metmuseum.github.io/).
- **Codebase**: Released under the [Creative Commons Zero v1.0 Universal License (CC0-1.0)](LICENSE).
