# src/config.py
"""
MUESUM — Centralized Project Configuration & Environmental Settings
===================================================================
Provides single-source-of-truth definitions for:
  - System directory paths and dataset locations.
  - Model checkpoint resolutions (Baseline & Experimental).
  - Restoration and boundary feathering hyperparameter defaults.
  - Multi-Term Composite Loss weighting coefficients.
  - Reproducible random seeds and device detection.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch

# ── 1. Filesystem Paths ───────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
APP_DIR = ROOT_DIR / "app"
DATA_DIR = ROOT_DIR / "data"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
REPORTS_DIR = ROOT_DIR / "reports"
OUTPUTS_DIR = ROOT_DIR / "outputs"
TESTS_DIR = ROOT_DIR / "tests"
DOCS_DIR = ROOT_DIR / "docs"

# Data Subdirectories
CLEAN_DATA_DIR = DATA_DIR / "clean"
DAMAGED_DATA_DIR = DATA_DIR / "damaged"
SPLITS_DIR = DATA_DIR / "splits"
HOLDOUT_DATA_DIR = DATA_DIR / "independent_holdout"

# Checkpoint Paths
BASELINE_CHECKPOINT_PATH = CHECKPOINTS_DIR / "baseline_run" / "best_model.pth"
EXPERIMENTAL_CHECKPOINT_PATH = CHECKPOINTS_DIR / "experimental_texture_model" / "best_model.pth"

# ── 2. Checkpoint Resolution ──────────────────────────────────────────────────
def resolve_checkpoint(preference: str = "experimental") -> str:
    """
    Resolve model checkpoint path based on user preference or environment override.
    Priority:
      1. MUESUM_CHECKPOINT environment variable
      2. Experimental checkpoint (if preferred and exists)
      3. Baseline checkpoint
      4. Auto-discovery fallback
    """
    env_override = os.environ.get("MUESUM_CHECKPOINT", "").strip()
    if env_override and Path(env_override).exists():
        return str(Path(env_override).resolve())

    if preference == "experimental" and EXPERIMENTAL_CHECKPOINT_PATH.exists():
        return str(EXPERIMENTAL_CHECKPOINT_PATH.resolve())

    if BASELINE_CHECKPOINT_PATH.exists():
        return str(BASELINE_CHECKPOINT_PATH.resolve())

    fallback_candidates = [
        ROOT_DIR / "checkpoints" / "best_model.pth",
        OUTPUTS_DIR / "experiments" / "exp_2000_baseline" / "best_model.pth",
        OUTPUTS_DIR / "best_model.pth"
    ]
    for c in fallback_candidates:
        if c.exists():
            return str(c.resolve())

    return str(BASELINE_CHECKPOINT_PATH)


# ── 3. Device & Hardware Configuration ────────────────────────────────────────
def get_device() -> torch.device:
    """Detect compute device, respecting MUESUM_DEVICE override."""
    env_dev = os.environ.get("MUESUM_DEVICE", "").strip().lower()
    if env_dev in ("cpu", "cuda"):
        return torch.device(env_dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 4. Restoration & Inpainting Defaults ───────────────────────────────────────
DEFAULT_BLEND_MODE: str = "mask_guided"
DEFAULT_FEATHER_MODE: str = "feathered"
DEFAULT_FEATHER_RADIUS: int = 3
DEFAULT_MAX_DIMENSION: int = 1024
DEFAULT_RANDOM_SEED: int = 42

# Mask Polarity Definition
MASK_DAMAGED_VAL: int = 255
MASK_CLEAN_VAL: int = 0

# ── 5. Loss Formulation Hyperparameters ───────────────────────────────────────
LOSS_CONFIG: Dict[str, float] = {
    "lambda_l1": 1.0,
    "lambda_edge": 0.15,
    "lambda_tv": 0.01,
    "lambda_perceptual": 0.05,
    "damage_weight": 4.0,
    "clean_weight": 1.0
}

# ── 6. Gradio Server Configuration ────────────────────────────────────────────
DEFAULT_SERVER_PORT: int = int(os.environ.get("MUESUM_PORT", "7860"))
DEFAULT_SERVER_NAME: str = "127.0.0.1"
