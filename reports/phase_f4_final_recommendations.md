# MUESUM — Phase F.4: Final Recommendations & Strategic Roadmap

**Document ID**: `reports/phase_f4_final_recommendations.md`  
**Date**: November 3, 2026  
**Auditor**: Antigravity Technical Pair Programmer  
**Target Milestone**: Post-Phase F Architectural & Training Decisions  
**System Hardware**: Dell Vostro 3520 (Intel Core i5-1235U, 16 GB RAM, CPU-Only)  
**Status**: **STRATEGIC TECHNICAL SPECIFICATION**

---

## 1. Executive Decision Classification

Based on comprehensive quantitative audits (Phase F.1), qualitative visual evaluations (Phases F.2 & F.4), and end-to-end studio verification (Phase F.3), the next development phase is classified as:

### **VERDICT: Option E — Multiple Targeted Improvements Are Necessary**

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               DEVELOPMENT VERDICT SUMMARY                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Option A (Demo-Ready):       [YES]  The current system is demonstrably successful      │
│                                     for controlled museum previews & continuous-tone   │
│                                     artifacts (bronzes, sculptures, illuminated text). │
│                                                                                        │
│ Option B (Loss Function):    [CRITICAL] L1-loss conditional-mean smoothing is the      │
│                                         primary bottleneck causing failure on graphic  │
│                                         drawings, photographs, and woven textiles.     │
│                                                                                        │
│ Option D (Damage Detection): [IMPORTANT] Semi-automated brush masking works, but true   │
│                                          mass digitization requires automated defect   │
│                                          segmentation upstream.                        │
│                                                                                        │
│ FINAL SYNTHESIS:             ===> OPTION E (Staged Multi-Component Upgrades)           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technical Evidence Supporting the Verdict

1. **Why the Current System IS Ready for Controlled Demonstrations**:
   - Mask-guided blending guarantees **100% bit-exact authentic preservation** ($\text{MAE} = 0.000000$).
   - Across continuous-tone surfaces (stone sculptures, carved reliefs, cast bronzes), the model delivers $+3.5\text{ dB}$ to $+7.6\text{ dB}$ ROI gains with high visual plausibility.
   - The interactive Gradio studio is fully operational, responsive, and robust against invalid inputs.
2. **Why the Current System CANNOT Be Declared a General Conservation Solution**:
   - In fine line drawings (Guercino `336663`), scratches across parallel cross-hatching are erased rather than reconstructed ($\Delta\text{PSNR} = -1.15\text{ dB}$).
   - In 19th-century vintage photography (albumen print `283234`), authentic silver gelatin film grain is eradicated, creating glass-smooth synthetic streaks ($\Delta\text{PSNR} = -4.10\text{ dB}$).
   - In periodic printed textiles (`157559`), repeating floral rosettes break down into gray smudges ($\Delta\text{SSIM} = -0.1566$).
   - The system is completely dependent on oracle or manual masks; it possesses no autonomous understanding of what constitutes damage versus authentic paint.

---

## 3. Prioritized Strategic Roadmap

To maximize restoration quality while respecting CPU hardware constraints (Intel Core i5, 16 GB RAM), improvements are prioritized across 4 engineering tiers:

```
Priority 1: Multi-Component Loss (VGG + Edge + TV) ──► Highest ROI / Low Complexity
Priority 2: Boundary Seam Feathering (Post-Process) ──► Zero Compute Cost / Immediate Polish
Priority 3: Automated Damage Segmentation (Upstream) ──► High Utility / Moderate Effort
Priority 4: Attention Mechanism (Transformer Decoder) ──► High Complexity / Defer to GPU
```

---

### Priority 1: Multi-Component Loss Refinement (VGG Perceptual + Sobel Edge)
- **Problem Solved**: Eliminates L1 conditional-mean collapse (Failure Modes 1 & 2); forces the network to retain fine textural statistics and directional ink strokes.
- **Formulation**:
  $$\mathcal{L}_{\text{total}} = \lambda_{\text{L1}} \mathcal{L}_{\text{L1}} + \lambda_{\text{percep}} \mathcal{L}_{\text{VGG19}} + \lambda_{\text{edge}} \mathcal{L}_{\text{Sobel}} + \lambda_{\text{tv}} \mathcal{L}_{\text{TV}}$$
  *(Note: `src/losses.py` already contains initial implementations of `SobelEdgeLoss` and `TVLoss`).*
- **Expected Quality Gain**: **Very High** (+1.5 to +3.0 dB on textured canvas; sharpens fine lines).
- **Engineering Complexity**: **Low** (loss functions exist; requires integration into training loop).
- **CPU Training Cost**: **Medium** (Fine-tuning the existing Epoch 8 checkpoint for 5–10 epochs takes $\approx 4$ hours on CPU).
- **RAM Footprint**: $\approx 6\text{ GB}$ (safely within 16 GB).
- **Estimated Timeline**: 1 day.

---

### Priority 2: Inference-Time Boundary Seam Feathering
- **Problem Solved**: Eliminates the 1-pixel step discontinuity along mask contours under high magnification (Failure Mode 5).
- **Formulation**:
  Replace hard binary thresholding with narrow-band alpha feathering along the boundary contour ring $\partial M$:
  $$M_{\text{feather}} = \text{GaussianFilter}(M, \sigma = 0.8)$$
  $$\text{Final} = \text{Restored} \odot M_{\text{feather}} + \text{Original} \odot (1 - M_{\text{feather}})$$
- **Expected Quality Gain**: **Medium** (eliminates visible loupe seams for conservators).
- **Engineering Complexity**: **Very Low** (15 lines of NumPy/OpenCV in `src/inference.py`).
- **CPU Training Cost**: **Zero** (pure inference post-processing; zero retraining).
- **RAM Footprint**: **Zero**.
- **Estimated Timeline**: 2 hours.

---

### Priority 3: Automated Damage Mask Segmentation Subsystem
- **Problem Solved**: Removes dependency on manual brush painting or precomputed oracle masks; enables batch digitization of uncurated museum archives.
- **Architecture**: A lightweight binary defect segmenter (MobileNetV3-UNet or SegNet, $< 3\text{M}$ params) trained on paired clean/damaged samples to predict $M(x, y) \in [0, 1]$.
- **Expected Utility Gain**: **High** (transforms system from semi-automated to autonomous).
- **Engineering Complexity**: **Moderate**.
- **CPU Training Cost**: **Moderate** ($\approx 3$ hours on CPU for lightweight backbone).
- **RAM Footprint**: $< 4\text{ GB}$.
- **Estimated Timeline**: 2 days.

---

### Priority 4: Attention-Based Decoder Enhancement
- **Problem Solved**: Solves long-range periodic pattern reconstruction for printed and woven textiles (Failure Mode 3).
- **Formulation**: Replace standard convolutional decoder blocks with Contextual Attention or Swin-Transformer blocks that can query distant intact textile patterns.
- **Expected Quality Gain**: High for textiles; Marginal for general paintings.
- **Engineering Complexity**: **High**.
- **CPU Training Cost**: **Prohibitive on CPU** (attention compute scaling requires GPU acceleration).
- **Recommendation**: **Defer** until cloud or dedicated GPU resources (e.g. NVIDIA T4/A100) become available.

---

## 4. Proposed Immediate Action: Phase F.5 Independent Holdout Execution

Before initiating any architectural refactoring or loss-function retraining (Priority 1), we recommend executing the **Independent Holdout Evaluation** designed in `reports/phase_f4_independent_evaluation_plan.md`:
1. Acquire 40 new non-overlapping artworks from The Met API.
2. Generate synthetic damage using the existing simulator.
3. Establish the definitive, unbiased baseline benchmark of the Epoch 8 model.
4. Use this independent score as the benchmark against which future perceptual/edge-loss models are judged.

---

## 5. Summary Matrix of Recommendations

| Initiative | Priority | Expected Impact | Compute Cost | RAM Footprint | Timeline | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Independent Holdout Benchmark (40 Met Artworks)** | **Immediate** | Scientific Validity | Negligible ($\sim 3$ min) | Safe ($< 4\text{ GB}$) | 1 hour | **Execute First (Pending Approval)** |
| **Boundary Seam Feathering** | High | Clean Borders | Zero (Inference) | Zero | 2 hours | Implement Next |
| **Perceptual + Edge Loss Fine-Tuning** | High | Solves Texture Loss | Moderate ($\sim 4$ hrs) | $\approx 6\text{ GB}$ | 1 day | Primary Model Upgrade |
| **Automated Damage Segmentation Net** | Medium | Autonomy | Moderate ($\sim 3$ hrs) | $\approx 4\text{ GB}$ | 2 days | Phase G Major Feature |
| **Transformer Attention Decoder** | Low | Pattern Periodicity | Severe on CPU | High | 4 days | Defer to GPU Environment |

