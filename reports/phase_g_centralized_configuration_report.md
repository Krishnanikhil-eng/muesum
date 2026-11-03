# Phase G — Stage 8: Centralized Configuration Management Report

**Project**: MUESUM — AI-Powered Historical Artwork Restoration  
**Phase**: Stage 8 (Centralized Configuration Management)  
**Date**: November 4, 2026  
**Status**: COMPLETE  

---

## 1. Executive Summary

To guarantee reproducibility, architectural cohesion, and zero-drift maintenance across the restoration lifecycle, Stage 8 established a single-source-of-truth configuration architecture in `src/config.py`. Prior to this stage, directory paths, checkpoint candidate lists, inpainting hyperparameters, and loss coefficients were defined across several discrete modules (`src/inference.py`, `app/demo.py`, `scripts/train_experimental_model.py`, and `scripts/evaluate_independent_holdout.py`).

`src/config.py` unifies all runtime, architectural, and conservation parameters while providing environment variable overrides (`MUESUM_CHECKPOINT`, `MUESUM_DEVICE`, `MUESUM_MAX_DIMENSION`) for flexible deployment across heterogeneous environments.

---

## 2. Configuration Architecture

### 2.1 Filesystem Paths & Directory Hierarchy
All system paths are computed dynamically relative to the repository root, ensuring consistent behavior across POSIX and Windows environments:
- `ROOT_DIR`: Base workspace root directory.
- `SRC_DIR`: Core algorithmic engine (`src/`).
- `APP_DIR`: Gradio web restoration studio (`app/`).
- `DATA_DIR`: Training, validation, and holdout datasets (`data/`).
- `CHECKPOINTS_DIR`: Checkpoint directory hosting baseline and experimental weights (`checkpoints/`).
- `REPORTS_DIR`: Comprehensive audit logs, benchmark summaries, and evaluations (`reports/`).
- `TESTS_DIR`: Automated regression and unit test suite (`tests/`).

### 2.2 Model Checkpoint Resolution Hierarchy
The `resolve_checkpoint(preference="experimental")` function establishes an explicit fallback priority:
1. **Explicit Environment Override**: `os.environ["MUESUM_CHECKPOINT"]` (if path exists on disk).
2. **Preferred Model**:
   - `"experimental"`: Checks `checkpoints/experimental_texture_model/best_model.pth`.
   - `"baseline"`: Checks `checkpoints/baseline_run/best_model.pth`.
3. **Primary Fallback**: Checks whichever of the two primary checkpoints exists.
4. **Historical Fallback**: Searches `outputs/experiments/exp_2000_baseline/best_model.pth` or root `checkpoints/best_model.pth`.

### 2.3 Hardware & Device Abstraction
The `get_device()` function detects available acceleration:
- Defaults to `cuda` if an NVIDIA GPU is present; otherwise falls back gracefully to `cpu`.
- Can be explicitly overridden with `MUESUM_DEVICE=cpu` or `MUESUM_DEVICE=cuda` for deterministic testing and low-memory execution.

### 2.4 Restoration & Conservation Inpainting Defaults
Conservation standards codified as programmatic defaults:
- `DEFAULT_BLEND_MODE = "mask_guided"`: Guarantees authentic artwork outside damage mask is strictly untouched.
- `DEFAULT_FEATHER_MODE = "feathered"`: Interior distance-transform boundary smoothing.
- `DEFAULT_FEATHER_RADIUS = 3`: 3-pixel interior feathering radius (reduces boundary disjunction by >70% while preserving authentic substrate 100% bit-exact).
- `DEFAULT_MAX_DIMENSION = 1024`: Aspect-preserving resizing ceiling for inference on memory-constrained systems.
- `MASK_DAMAGED_VAL = 255`, `MASK_CLEAN_VAL = 0`: Standard binary mask polarity.

### 2.5 Multi-Term Composite Loss Configuration
Hyperparameters used by `CompositeRestorationLoss`:
- `lambda_l1 = 1.0` (L1 Masked Reconstruction)
- `lambda_edge = 0.15` (Sobel High-Frequency Edge Preservation)
- `lambda_tv = 0.01` (Total Variation Regularization)
- `lambda_perceptual = 0.05` (VGG16 Perceptual Texture Loss)
- `damage_weight = 4.0` (Priority ratio for damaged pixels over clean pixels)
- `clean_weight = 1.0` (Supervision ratio for surviving artwork)

---

## 3. Verification & Automated Unit Testing

A comprehensive unit test suite (`tests/test_config.py`) was implemented and executed:
1. `test_paths_validity`: Verifies all paths are valid `Path` objects pointing within the workspace.
2. `test_checkpoint_paths_exist`: Confirms both `baseline_run/best_model.pth` and `experimental_texture_model/best_model.pth` exist on disk.
3. `test_checkpoint_resolution_preference`: Validates checkpoint resolution under `"experimental"`, `"baseline"`, and invalid inputs.
4. `test_device_detection`: Validates `get_device()` and environment override `MUESUM_DEVICE`.
5. `test_loss_hyperparameters_structure`: Validates presence and non-negative values for all loss terms.
6. `test_restoration_defaults`: Verifies inpainting defaults match curatorial requirements.

**Test Result**: 6 / 6 tests passed in 0.001 seconds.  
**Full Master QA Suite**: 29 / 29 tests passed in 0.83 seconds.
