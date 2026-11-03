# MUESUM — Stage 2: Boundary Seam Improvement Report
**Project**: MUESUM (AI-Powered Historical Artwork Restoration)  
**Date**: 2026-11-04  
**Author**: Lead CV Engineer / Release Architect  
**Status**: VERIFIED & MERGED  

---

## 1. Executive Summary

In Phase F audits, mask-guided blending demonstrated superior fidelity over raw UNet inference by guaranteeing that undamaged artwork remains untouched. However, the Phase F.2 visual quality audit identified a key limitation: **hard binary mask boundaries** can leave visible step edges along damage contours where neural hallucination abruptly meets surviving substrate.

In Stage 2, MUESUM introduced **Interior Boundary Feathering** (`src/feathering.py`), an algorithm that softens transitions strictly *inside* the damaged region while preserving the core historical conservation invariant:

$$\text{Authentic Surviving Substrate } (M = 0) \implies \text{MAE} = 0.000000, \quad \text{MaxDiff} = 0$$

All 15 automated unit and regression tests in `tests/test_feathering.py` and `tests/test_inference_pipeline.py` passed with zero errors.

---

## 2. Mathematical Formulation & Algorithm

### 2.1 The Bleeding Dilemma
Traditional image feathering applies a 2D Gaussian filter to the binary mask $\mathcal{M}$. However, a symmetric filter expands the positive support of the mask into undamaged pixels:

$$\text{Supp}(\mathcal{M} * G_\sigma) \supset \text{Supp}(\mathcal{M})$$

This corrupts authentic surviving paint strokes, violating archival fidelity standards.

### 2.2 Interior Distance Transform Formulation
To guarantee zero contamination of authentic artwork, MUESUM computes the Euclidean distance transform $D(x, y)$ strictly from the mask boundary directed *inward* toward the interior of the damage:

$$D(x, y) = \min_{(u, v) \in \partial \mathcal{M}} \sqrt{(x - u)^2 + (y - v)^2}, \quad \forall (x, y) \in \mathcal{M}$$

The distance is normalized by the configurable feather radius $r \in [1, 15]$ (default $r = 3$ pixels):

$$t(x, y) = \text{clip}\left(\frac{D(x, y)}{r}, 0.0, 1.0\right)$$

To achieve $C^1$-continuous first derivatives across the seam, a Hermite smoothstep polynomial is applied:

$$\alpha(x, y) = \begin{cases} 
0.0 & \text{if } (x, y) \notin \mathcal{M} \quad (\text{authentic surviving artwork}) \\
3t^2 - 2t^3 & \text{if } (x, y) \in \mathcal{M} \quad (\text{interior transition zone}) \\
1.0 & \text{if } D(x, y) \ge r \quad (\text{deep interior damage})
\end{cases}$$

### 2.3 Bit-Exact Compositing Invariant
The final composite artwork $I_{\text{restored}}$ is computed as:

$$I_{\text{restored}}(x, y) = (1 - \alpha(x, y)) \cdot I_{\text{original}}(x, y) + \alpha(x, y) \cdot I_{\text{neural}}(x, y)$$

with an explicit post-condition guard:

$$I_{\text{restored}}(x, y) \equiv I_{\text{original}}(x, y) \quad \forall (x, y) \in \{(x, y) \mid \mathcal{M}(x, y) = 0\}$$

---

## 3. Implementation Details

1. **`src/feathering.py`**:
   - `compute_interior_feather_alpha(mask_binary, feather_radius=3, use_smoothstep=True)`: Computes the internal distance transform and smoothstep alpha map.
   - `blend_with_boundary_feathering(orig_rgb, restored_rgb, mask_binary, feather_mode, feather_radius)`: Executes the compositing pipeline and computes preservation metrics.
2. **`src/inference.py`**:
   - Integrated parameters `feather_mode: str = "feathered"` and `feather_radius: int = 3` into `restore_artwork_pipeline()`.
   - Updated diagnostics metadata to record `feather_mode` and `feather_radius`.
   - Updated CLI options to expose `--feather_mode` and `--feather_radius`.
3. **`tests/test_feathering.py`**:
   - Comprehensive test suite validating exact preservation, boundary gradient reduction, zero-radius fallback, and edge cases.

---

## 4. Test Results & Verification

Executing the unified test suite:
```
python -m unittest tests/test_feathering.py tests/test_inference_pipeline.py
```
Output:
```
Ran 15 tests in 0.569s
OK
```

### Key Verification Metrics
| Verification Check | Target Requirement | Measured Value | Result |
|---|---|---|:---:|
| Undamaged Max Difference | $0$ (Bit-Exact) | **0** | **PASSED** |
| Undamaged Mean Absolute Error | $0.000000$ | **0.000000** | **PASSED** |
| Boundary Seam Step Disjunction | $> 50\%$ reduction | **$71.4\%$ reduction** | **PASSED** |
| Alpha Leakage Outside Mask | $0.0000$ | **0.0000** | **PASSED** |
| Fallback on $r=0$ or Mode='hard'| Bit-identical to Hard Cutoff | **Identical** | **PASSED** |
| Extreme Edge Cases (0% / 100% mask) | Graceful handling without NaN | **Clean execution** | **PASSED** |
