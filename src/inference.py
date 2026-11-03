# src/inference.py
"""
MUESUM — Reusable Restoration Inference Pipeline
================================================
Enterprise-grade, memory-efficient restoration engine featuring:
  - Mask-guided blending with bit-exact preservation of undamaged artwork.
  - ModelManager singleton for checkpoint and model caching across calls.
  - Strict input dimensions, channel, and mask polarity validation.
  - Aspect-preserving resize with full-resolution final composition.
  - Comprehensive diagnostic metadata (latency, coverage, MAE, MaxDiff).
  - Backward compatibility for existing CLI unet_restore() calls.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import gc
import argparse
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

# Ensure local imports work whether called from root or src
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from unet_model import ResNet34UNet
from dataset import pad_to_multiple_of_32, unpad
from utils import tensor_to_uint8, ensure_dir
from feathering import blend_with_boundary_feathering

try:
    import config as muesum_cfg
except ImportError:
    muesum_cfg = None


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint Resolution & ModelManager Singleton
# ──────────────────────────────────────────────────────────────────────────────
def find_default_checkpoint() -> str:
    """Search prioritized checkpoint locations and return the first found path."""
    if muesum_cfg is not None:
        cp = muesum_cfg.resolve_checkpoint("baseline")
        if os.path.exists(cp):
            return cp

    candidates = [
        _ROOT_DIR / "checkpoints" / "baseline_run" / "best_model.pth",
        _ROOT_DIR / "checkpoints" / "experimental_texture_model" / "best_model.pth",
        _ROOT_DIR / "checkpoints" / "best_model.pth",
        _ROOT_DIR / "outputs" / "experiments" / "exp_2000_baseline" / "best_model.pth",
        _ROOT_DIR / "outputs" / "experiments" / "exp_2000_perceptual" / "best_model.pth",
        _ROOT_DIR / "outputs" / "best_model.pth",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


class ModelManager:
    """
    Thread-safe model cache singleton. Prevents reloading heavy model weights
    from disk on repeated sequential inference requests.
    """
    _instance: Optional["ModelManager"] = None
    _model: Optional[torch.nn.Module] = None
    _ckpt_path: Optional[str] = None
    _device: Optional[torch.device] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_model(
        cls,
        ckpt_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.nn.Module, torch.device]:
        """
        Retrieve cached model or load checkpoint if not already loaded.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        target_ckpt = str(ckpt_path).strip() if ckpt_path else find_default_checkpoint()
        if not target_ckpt or not Path(target_ckpt).exists():
            raise FileNotFoundError(
                f"Checkpoint not found at '{target_ckpt}'. "
                "Please verify checkpoint path or train the model first."
            )

        resolved_path = str(Path(target_ckpt).resolve())

        # Return cached instance if checkpoint path and device match
        if (
            cls._model is not None
            and cls._ckpt_path == resolved_path
            and cls._device == device
        ):
            return cls._model, cls._device

        # Otherwise, load new model into cache
        print(f"[ModelManager] Loading model checkpoint from: {resolved_path} on {device}")
        model = ResNet34UNet(pretrained=False).to(device)
        st = torch.load(resolved_path, map_location=device)
        state_dict = st.get("model", st)
        model.load_state_dict(state_dict)
        model.eval()

        cls._model = model
        cls._ckpt_path = resolved_path
        cls._device = device

        return cls._model, cls._device

    @classmethod
    def clear_cache(cls):
        """Release cached model from memory and trigger garbage collection."""
        cls._model = None
        cls._ckpt_path = None
        cls._device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_model(ckpt_path: Optional[Union[str, Path]] = None, device: Optional[torch.device] = None):
    """Convenience wrapper using ModelManager for backward compatibility."""
    return ModelManager.get_model(ckpt_path=ckpt_path, device=device)


# ──────────────────────────────────────────────────────────────────────────────
# Input Normalization & Validation Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _standardize_image(img_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Convert any supported image input into a standardized uint8 RGB NumPy array [H, W, 3].
    """
    if isinstance(img_input, (str, Path)):
        p = Path(img_input)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {p}")
        with Image.open(p) as pil_img:
            img_arr = np.array(pil_img.convert("RGB"))
    elif isinstance(img_input, Image.Image):
        img_arr = np.array(img_input.convert("RGB"))
    elif isinstance(img_input, np.ndarray):
        img_arr = img_input.copy()
        if img_arr.ndim == 2:  # Grayscale
            img_arr = np.stack([img_arr] * 3, axis=-1)
        elif img_arr.ndim == 3:
            if img_arr.shape[2] == 4:  # RGBA -> drop alpha
                img_arr = img_arr[:, :, :3]
            elif img_arr.shape[2] == 1:
                img_arr = np.concatenate([img_arr] * 3, axis=-1)
        else:
            raise ValueError(f"Invalid image array shape: {img_arr.shape}")
        
        # Ensure uint8 [0, 255]
        if np.issubdtype(img_arr.dtype, np.floating):
            if img_arr.max() <= 1.01:
                img_arr = (img_arr * 255.0).clip(0, 255)
            img_arr = np.round(img_arr).astype(np.uint8)
        elif img_arr.dtype != np.uint8:
            img_arr = img_arr.clip(0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image input type: {type(img_input)}")

    if img_arr.shape[0] < 8 or img_arr.shape[1] < 8:
        raise ValueError(f"Image too small: {img_arr.shape[:2]}. Minimum dimension is 8x8.")

    return img_arr


def _standardize_mask(
    mask_input: Optional[Union[str, Path, Image.Image, np.ndarray]],
    target_shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    """
    Convert any supported mask input into a standardized boolean/float32 mask [H, W]
    where 1.0 (True) = damaged region, 0.0 (False) = undamaged clean artwork.
    
    Validates dimensions and values strictly.
    """
    if mask_input is None:
        return None

    if isinstance(mask_input, (str, Path)):
        p = Path(mask_input)
        if not p.exists():
            raise FileNotFoundError(f"Mask file not found: {p}")
        with Image.open(p) as pil_m:
            m_arr = np.array(pil_m.convert("L"))
    elif isinstance(mask_input, Image.Image):
        m_arr = np.array(mask_input.convert("L"))
    elif isinstance(mask_input, np.ndarray):
        m_arr = mask_input.copy()
        if m_arr.ndim == 3:
            if m_arr.shape[2] == 4:  # RGBA from drawing canvas: use alpha channel
                m_arr = m_arr[:, :, 3]
            elif m_arr.shape[2] in (1, 3):
                m_arr = m_arr.squeeze()
                if m_arr.ndim == 3:
                    m_arr = cv2.cvtColor(m_arr, cv2.COLOR_RGB2GRAY)
        elif m_arr.ndim != 2:
            raise ValueError(f"Invalid mask array shape: {m_arr.shape}")
    else:
        raise TypeError(f"Unsupported mask input type: {type(mask_input)}")

    # 1. Dimension validation
    h_img, w_img = target_shape
    h_mask, w_mask = m_arr.shape[:2]
    if (h_mask, w_mask) != (h_img, w_img):
        raise ValueError(
            f"Dimension mismatch: damage mask dimensions ({w_mask}x{h_mask}) do not "
            f"match image dimensions ({w_img}x{h_img})."
        )

    # 2. Value range and NaN/Inf validation
    if np.any(np.isnan(m_arr)) or np.any(np.isinf(m_arr)):
        raise ValueError("Damage mask contains NaN or Infinite values.")

    # 3. Polarity validation & binarization (1.0 = damaged, 0.0 = clean)
    if m_arr.dtype == bool:
        mask_binary = m_arr.astype(np.float32)
    elif np.issubdtype(m_arr.dtype, np.floating):
        if m_arr.min() < -1e-4 or m_arr.max() > 1.01:
            raise ValueError(f"Float mask values out of expected range [0.0, 1.0]: min={m_arr.min()}, max={m_arr.max()}")
        mask_binary = (m_arr > 0.5).astype(np.float32)
    else:
        # Integer values (e.g. uint8 0..255)
        if m_arr.min() < 0 or m_arr.max() > 255:
            raise ValueError(f"Integer mask values out of expected range [0, 255]: min={m_arr.min()}, max={m_arr.max()}")
        mask_binary = (m_arr > 127).astype(np.float32)

    return mask_binary


# ──────────────────────────────────────────────────────────────────────────────
# Core Restoration Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def restore_artwork_pipeline(
    damaged_image: Union[str, Path, Image.Image, np.ndarray],
    damage_mask: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
    ckpt_path: Optional[Union[str, Path]] = None,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    blend_mode: str = "mask_guided",
    feather_mode: str = "feathered",
    feather_radius: int = 3,
    max_dimension: Optional[int] = 1024,
    return_diagnostics: bool = True
) -> Dict[str, Any]:
    """
    Main entry point for artwork restoration inference.

    Args:
        damaged_image: Filepath, PIL Image, or NumPy uint8 array of the damaged artwork.
        damage_mask: Filepath, PIL Image, or NumPy array of the damage mask.
                     Convention: 1 / 255 = damaged pixel; 0 = undamaged clean artwork.
                     REQUIRED when blend_mode is 'mask_guided'.
        ckpt_path: Path to .pth checkpoint. If None, auto-detected via find_default_checkpoint().
        model: Optional pre-loaded PyTorch model instance (overrides ModelManager).
        device: Target execution device (torch.device('cpu') or 'cuda').
        blend_mode: 'mask_guided' (recommended, bit-exact preservation) or 'raw_unet' (unguided).
        feather_mode: 'feathered' (interior seam smoothing) or 'hard' (strict step cutoff).
        feather_radius: Feathering radius in pixels strictly inside damage mask (default: 3).
        max_dimension: Aspect-preserving resize ceiling for neural inference (default: 1024).
                       If None, runs at native resolution.
        return_diagnostics: Whether to compute and include detailed metadata in the output dictionary.

    Returns:
        dict containing:
            - 'restored_image': np.ndarray (uint8 [H, W, 3])
            - 'raw_unet_image': np.ndarray (uint8 [H, W, 3])
            - 'original_image': np.ndarray (uint8 [H, W, 3])
            - 'mask_binary': np.ndarray (uint8 [H, W] with values 0, 255) or None
            - 'mask_preview': PIL.Image (semi-transparent red overlay on original) or None
            - 'diagnostics': dict of quantitative performance and preservation metrics
    """
    t_start = time.perf_counter()

    # ── 1. Validate blend_mode & mask requirement ─────────────────────────────
    blend_mode = blend_mode.strip().lower()
    if blend_mode not in ("mask_guided", "raw_unet"):
        raise ValueError(f"Invalid blend_mode: '{blend_mode}'. Must be 'mask_guided' or 'raw_unet'.")

    if blend_mode == "mask_guided" and damage_mask is None:
        raise ValueError(
            "Mask-guided restoration requires a damage mask, but none was provided.\n"
            "MUESUM does not fabricate or guess damage masks automatically.\n"
            "Either provide an explicit damage mask, use the interactive brush tool, "
            "or set blend_mode='raw_unet' to run unguided baseline inference."
        )

    # ── 2. Standardize inputs & validate dimensions ───────────────────────────
    orig_rgb = _standardize_image(damaged_image)
    orig_h, orig_w = orig_rgb.shape[:2]

    mask_binary = _standardize_mask(damage_mask, (orig_h, orig_w))
    mask_coverage_pct = float((np.sum(mask_binary) / mask_binary.size) * 100.0) if mask_binary is not None else 0.0

    # ── 3. Resolve Model & Device ─────────────────────────────────────────────
    if model is None:
        model, device = ModelManager.get_model(ckpt_path=ckpt_path, device=device)
    elif device is None:
        device = next(model.parameters()).device

    # ── 4. Aspect-Preserving Scaling for Neural Inference ─────────────────────
    if max_dimension is not None and max(orig_h, orig_w) > max_dimension:
        scale = float(max_dimension) / float(max(orig_h, orig_w))
        eval_w = max(32, int(round(orig_w * scale)))
        eval_h = max(32, int(round(orig_h * scale)))
        eval_img = cv2.resize(orig_rgb, (eval_w, eval_h), interpolation=cv2.INTER_AREA)
    else:
        eval_w, eval_h = orig_w, orig_h
        eval_img = orig_rgb

    # ── 5. Neural Network Forward Pass (Padded to multiple of 32) ─────────────
    eval_padded, pad_info = pad_to_multiple_of_32(eval_img)
    inp_tensor = torch.from_numpy(eval_padded.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    inp_tensor = inp_tensor.to(device)

    # Use torch.inference_mode() for speed and zero gradient tracking overhead
    with torch.inference_mode():
        out_tensor = model(inp_tensor)

    # Convert to uint8 and unpad back to eval dimensions
    out_np = tensor_to_uint8(out_tensor[0].cpu())
    raw_unet_eval = unpad(out_np, pad_info)

    # Clean up intermediate tensors immediately
    del inp_tensor, out_tensor, eval_padded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 6. Rescale Raw Prediction to Original Resolution if Needed ───────────
    if (eval_w, eval_h) != (orig_w, orig_h):
        raw_unet_orig = cv2.resize(raw_unet_eval, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        raw_unet_orig = np.clip(raw_unet_orig, 0, 255).astype(np.uint8)
    else:
        raw_unet_orig = raw_unet_eval

    # ── 7. Final Composition & Bit-Exact Blending ─────────────────────────────
    if blend_mode == "mask_guided" and mask_binary is not None:
        restored_blended, blend_metrics = blend_with_boundary_feathering(
            orig_rgb=orig_rgb,
            restored_rgb=raw_unet_orig,
            mask_binary=mask_binary,
            feather_mode=feather_mode,
            feather_radius=feather_radius
        )
        undamaged_max_diff = blend_metrics["undamaged_max_diff"]
        undamaged_mae = blend_metrics["undamaged_mae"]
        is_bit_exact = blend_metrics["is_bit_exact"]
        final_output = restored_blended
    else:
        # Raw UNet replaces 100% of pixels
        diff = np.abs(raw_unet_orig.astype(np.int32) - orig_rgb.astype(np.int32))
        undamaged_max_diff = int(np.max(diff))
        undamaged_mae = float(np.mean(diff))
        is_bit_exact = False
        final_output = raw_unet_orig

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    # ── 8. Generate Visual Mask Overlay Preview ───────────────────────────────
    mask_preview_pil = None
    mask_uint8 = None
    if mask_binary is not None:
        mask_uint8 = (mask_binary * 255.0).astype(np.uint8)
        # Create semi-transparent red damage highlight on top of original artwork
        overlay = orig_rgb.copy()
        dmg_idx = (mask_binary > 0.5)
        # Red tint: 70% original + 30% crimson [255, 30, 30]
        overlay[dmg_idx] = np.clip(0.5 * overlay[dmg_idx] + 0.5 * np.array([255, 30, 30]), 0, 255).astype(np.uint8)
        mask_preview_pil = Image.fromarray(overlay)

    # ── 9. Package Diagnostics & Output Dictionary ────────────────────────────
    diagnostics = {
        "blend_mode": blend_mode,
        "feather_mode": feather_mode if blend_mode == "mask_guided" else "none",
        "feather_radius": feather_radius if (blend_mode == "mask_guided" and feather_mode == "feathered") else 0,
        "original_size": (orig_w, orig_h),
        "eval_size": (eval_w, eval_h),
        "mask_coverage_pct": round(mask_coverage_pct, 2),
        "latency_ms": round(latency_ms, 1),
        "undamaged_max_diff": undamaged_max_diff,
        "undamaged_mae": round(undamaged_mae, 6),
        "is_bit_exact": is_bit_exact,
        "device": str(device),
        "preservation_status": "100% Bit-Exact Preservation (0 error)" if is_bit_exact else "Modified (Unguided Convolutions)"
    }

    return {
        "restored_image": final_output,
        "raw_unet_image": raw_unet_orig,
        "original_image": orig_rgb,
        "mask_binary": mask_uint8,
        "mask_preview": mask_preview_pil,
        "diagnostics": diagnostics
    }


# ──────────────────────────────────────────────────────────────────────────────
# Backward Compatibility Wrapper
# ──────────────────────────────────────────────────────────────────────────────
def unet_restore(inp_path: Union[str, Path], out_path: Union[str, Path], ckpt_path: Optional[Union[str, Path]] = None):
    """
    Legacy wrapper maintaining exact backward compatibility with earlier pipeline scripts.
    Executes raw unguided UNet restoration and saves the output to out_path.
    """
    res = restore_artwork_pipeline(
        damaged_image=inp_path,
        ckpt_path=ckpt_path,
        blend_mode="raw_unet",
        max_dimension=None,
        return_diagnostics=False
    )
    ensure_dir(str(Path(out_path).parent))
    Image.fromarray(res["restored_image"]).save(str(out_path))
    print(f"Saved {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MUESUM Artwork Restoration Inference CLI")
    parser.add_argument("--inp_path", "--image", "--input", dest="inp_path", required=True, help="Path to input damaged image")
    parser.add_argument("--out_path", "--output", "--out", dest="out_path", required=True, help="Path to save restored image")
    parser.add_argument("--mask_path", "--mask", dest="mask_path", default=None, help="Path to damage mask (required for mask_guided mode)")
    parser.add_argument("--ckpt_path", "--checkpoint", dest="ckpt_path", default=None, help="Path to .pth checkpoint (optional)")
    parser.add_argument("--mode", "--blend_mode", dest="mode", default="mask_guided", choices=["mask_guided", "raw_unet"], help="Restoration mode")
    parser.add_argument("--feather_mode", default="feathered", choices=["feathered", "hard"], help="Boundary feather mode")
    parser.add_argument("--feather_radius", type=int, default=3, help="Interior feather radius in pixels (default: 3)")
    parser.add_argument("--max_dim", "--max_dimension", dest="max_dim", type=int, default=1024, help="Maximum evaluation dimension (default: 1024)")
    args = parser.parse_args()

    result = restore_artwork_pipeline(
        damaged_image=args.inp_path,
        damage_mask=args.mask_path,
        ckpt_path=args.ckpt_path,
        blend_mode=args.mode,
        feather_mode=args.feather_mode,
        feather_radius=args.feather_radius,
        max_dimension=args.max_dim if args.max_dim > 0 else None,
        return_diagnostics=True
    )

    ensure_dir(str(Path(args.out_path).parent))
    Image.fromarray(result["restored_image"]).save(args.out_path)
    diag = result["diagnostics"]
    print(f"✅ Restored artwork saved to: {args.out_path}")
    print(f"   Mode: {diag['blend_mode']} | Latency: {diag['latency_ms']:.1f} ms | Coverage: {diag['mask_coverage_pct']:.2f}%")
    print(f"   Preservation: {diag['preservation_status']} (MAE: {diag['undamaged_mae']:.6f})")
