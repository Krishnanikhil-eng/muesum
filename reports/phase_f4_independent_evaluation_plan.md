# MUESUM — Phase F.4: Independent Evaluation Strategy & Experimental Design

**Document ID**: `reports/phase_f4_independent_evaluation_plan.md`  
**Date**: November 3, 2026  
**Auditor**: Antigravity Technical Pair Programmer  
**Objective**: Protocol for establishing an independent holdout benchmark from The Metropolitan Museum of Art API.  
**Execution Status**: **PROPOSED / PENDING USER APPROVAL (Zero Downloads or Processing Initiated)**

---

## 1. Scientific Rationale for an Independent Holdout Split

In scientific machine learning and computer vision benchmarking, a test set must remain completely isolated from methodological design iterations. 

### Why the Existing 200-Image Test Set Cannot Serve as the Final Holdout
1. **Adaptive Method Selection**: The decision to implement mask-guided blending was formulated specifically after inspecting Phase E evaluation results on the 200 test artworks (which revealed that raw UNet suffered from global color shifts across undamaged regions).
2. **Post-Hoc Optimization**: Although model weights (`checkpoints/baseline_run/best_model.pth`) were never trained on the test set, designing inference post-processing and fine-tuning parameters against the same 200 images introduces an **adaptive feedback loop**.
3. **Peer-Review Credibility**: To make unbiased, reproducible claims of generalizability in academic literature or museum conservation technical reports, the final system must be evaluated on a **fresh, uninspected cohort of artworks**.

---

## 2. Dataset Design & Acquisition Protocol

### 2.1 Cohort Size & Composition
- **Target Size**: **40 new public-domain artworks** (balanced across 8 primary curatorial categories, 5 images per category):
  1. Paintings (European & American)
  2. Drawings & Watercolors
  3. Vintage Photographs (19th century)
  4. Printed Textiles
  5. Woven Textiles & Tapestries
  6. Sculptures (Stone & Marble)
  7. Bronze & Metalwork
  8. Illuminated Manuscripts & Early Printed Books

### 2.2 Strict Non-Overlap Verification
- The Met Collection API v1.1 endpoint (`https://collectionapi.metmuseum.org/publicaccess/v1/0/objects/[objectID]`) will be queried.
- Every candidate `objectID` must be checked against the existing 2,000 project IDs in `data/metadata.csv`.
- Any candidate present in `data/metadata.csv` or `data/splits/*.csv` will be immediately rejected:
  $$\mathcal{D}_{\text{independent}} \cap (\mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}) = \emptyset$$

### 2.3 Synthetic Damage Preprocessing
- Damaged counterparts and binary masks will be generated strictly using the verified `damage_sim_museum.py` engine:
  - Simulator: `simulate_damage_strong(clean_img, max_damage_types=4, seed=seed)`
  - Mask convention: $1.0$ (or $255$) = damaged pixels, $0.0$ (or $0$) = clean pixels.
  - Reproducibility: Deterministic seeds per object ID.

---

## 3. Storage, RAM & Resource Feasibility Assessment

### 3.1 Pre-Execution Storage & Disk Space Audit
- **Development Machine Free Disk Space**: **$51.11\text{ GB}$** (verified via `shutil.disk_usage`).
- **Estimated Resource Requirements for 40 New Artworks**:
  - 40 Clean reference images (JPEG, high-res): $\approx 90\text{ MB}$
  - 40 Damaged reference images (JPEG, high-res): $\approx 90\text{ MB}$
  - 40 Binary masks (PNG, 1-channel): $\approx 15\text{ MB}$
  - Metadata CSV and diagnostic logs: $< 1\text{ MB}$
  - **Total Required Storage**: **$\approx 195\text{ MB}$ ($0.19\text{ GB}$)**.
  - **Disk Safety Margin**: Consumes $< 0.4\%$ of available free disk space. Zero disk pressure risk.

### 3.2 Processing Duration & Memory Overhead
- **Download Time (API v1.1 Rate-Limited)**: $\approx 60\text{ s}$ ($\sim 1.5\text{ s}$ per object with polite backoff).
- **Damage Simulation Time**: $\approx 40\text{ s}$ on CPU ($\sim 1.0\text{ s}$ per image).
- **Inference Evaluation Time**: $\approx 100\text{ s}$ on CPU ($\sim 2.5\text{ s}$ per image at 1024 resolution scale).
- **RAM Headroom**: Memory consumption will remain under $12.2\text{ GB}$, leaving $> 3.5\text{ GB}$ available RAM throughout.

---

## 4. Quantitative Evaluation Metric Suite

For every artwork in the independent holdout, the following 12 metrics will be recorded:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              INDEPENDENT METRIC TAXONOMY                               │
├─────────────────────────┬──────────────────────┬───────────────────────────────────────┤
│ Domain                  │ Metric Name          │ Mathematical Scope                    │
├─────────────────────────┼──────────────────────┼───────────────────────────────────────┤
│ Whole-Image Fidelity    │ Damaged PSNR (dB)    │ Full composition baseline             │
│                         │ Restored PSNR (dB)   │ Mask-blended restored output          │
│                         │ $\Delta\text{PSNR}$  │ Net restoration gain ($\text{dB}$)    │
│                         │ Damaged SSIM         │ Structural baseline $[0, 1]$          │
│                         │ Restored SSIM        │ Mask-blended restored SSIM            │
│                         │ $\Delta\text{SSIM}$  │ Net structural gain                   │
├─────────────────────────┼──────────────────────┼───────────────────────────────────────┤
│ Damaged-Region (ROI)    │ ROI Damaged PSNR     │ Quality inside damage mask ($M=1$)    │
│                         │ ROI Restored PSNR    │ Inpainting quality inside mask        │
│                         │ ROI $\Delta\text{PSNR}$| Pure reconstruction repair ($\text{dB}$)│
├─────────────────────────┼──────────────────────┼───────────────────────────────────────┤
│ Authenticity Lock       │ Undamaged MAE        │ Error on untouched pixels ($M=0$)     │
│                         │ Undamaged MaxDiff    │ Max pixel deviation ($0$ required)    │
│                         │ `is_bit_exact`       │ Boolean confirmation of exactness     │
├─────────────────────────┼──────────────────────┼───────────────────────────────────────┤
│ Operational Diagnostics │ Mask Coverage %      │ Percentage of artwork damaged         │
│                         │ Inference Latency    │ Wall-clock execution time (ms)        │
└─────────────────────────┴──────────────────────┴───────────────────────────────────────┘
```

### Statistical Aggregation Protocol
- Results will be aggregated across all 40 artworks and reported by:
  - **Mean $\pm$ Standard Deviation**
  - **Median**
  - **Min / Max Range**
  - **Success Rate** (Percentage of artworks with $\Delta\text{PSNR} > 0\text{ dB}$)
- Metrics will be reported **both globally and grouped by curatorial category**, explicitly highlighting vulnerable media (photographs, line drawings) alongside high-performing media (sculptures, bronzes).

---

## 5. Directory Structure for Independent Evaluation

To maintain strict physical separation from training/validation/test data:

```
data/
├── independent_holdout/
│   ├── clean/             # 40 untouched clean Met artworks
│   ├── damaged/           # 40 precomputed damaged images
│   ├── masks/             # 40 paired binary damage masks
│   └── holdout_ids.csv    # Metadata, curatorial classes, Met URLs
reports/
└── independent_evaluation/
    ├── holdout_metrics.csv
    ├── category_breakdown.csv
    ├── independent_summary.md
    └── visual_comparisons/
```

---

## 6. Execution Safeguards

> [!IMPORTANT]
> **MANDATORY HOLD NOTICE**:  
> In accordance with Phase F.4 instructions:
> 1. **Zero download scripts have been triggered**.
> 2. **Zero new files have been written to `data/`**.
> 3. Execution will proceed **only after explicit user approval** of this plan.

