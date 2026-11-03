# src/feathering.py
"""
MUESUM — Boundary Seam Feathering Engine
========================================
Implements distance-transform guided interior feathering for seamless neural inpainting transitions.

Key Invariant:
  Feathering operates STRICTLY INSIDE the damage mask boundary.
  Undamaged authentic pixels (mask == 0) remain 100% bit-exact (undamaged_max_diff == 0, undamaged_mae == 0.0).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


def compute_interior_feather_alpha(
    mask_binary: np.ndarray,
    feather_radius: int = 3,
    use_smoothstep: bool = True
) -> np.ndarray:
    """
    Compute a continuous alpha blend map strictly inside the damaged region.

    Args:
        mask_binary: 2D NumPy array (float32 [0.0, 1.0] or bool or uint8 [0, 255]).
                     Values > 0.5 represent damaged pixels; 0 represents undamaged clean pixels.
        feather_radius: Radius in pixels over which the alpha transitions from 0 to 1 inside the damage.
                        If <= 0, returns hard binary mask (1.0 for damaged, 0.0 for clean).
        use_smoothstep: If True, applies Hermite interpolation t*t*(3 - 2*t) for C1-continuous gradients.

    Returns:
        alpha: 2D float32 array in [0.0, 1.0], same shape as mask_binary.
               Guaranteed: alpha[mask == 0] == 0.0 exactly.
    """
    dmg_bool = (mask_binary > 0.5)
    if not np.any(dmg_bool) or feather_radius <= 0:
        return dmg_bool.astype(np.float32)

    # Convert to 8-bit for distance transform
    mask_u8 = (dmg_bool * 255).astype(np.uint8)

    # Compute Euclidean distance transform from the 0-boundary into the 255-foreground
    dist = cv2.distanceTransform(mask_u8, distanceType=cv2.DIST_L2, maskSize=5)

    # Normalize distance by feather radius
    t = np.clip(dist / float(feather_radius), 0.0, 1.0)

    if use_smoothstep:
        # C1-continuous smoothstep: zero 1st derivative at t=0 and t=1
        alpha = t * t * (3.0 - 2.0 * t)
    else:
        alpha = t

    # Guarantee zero alpha on authentic surviving artwork
    alpha[~dmg_bool] = 0.0
    return alpha.astype(np.float32)


def blend_with_boundary_feathering(
    orig_rgb: np.ndarray,
    restored_rgb: np.ndarray,
    mask_binary: np.ndarray,
    feather_mode: str = "feathered",
    feather_radius: int = 3
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Blends neural restoration with original artwork using interior seam feathering.

    Args:
        orig_rgb: uint8 array [H, W, 3] of the original damaged artwork.
        restored_rgb: uint8 array [H, W, 3] of the neural network prediction (same dimensions).
        mask_binary: 2D array [H, W], where > 0.5 is damaged and 0 is undamaged clean substrate.
        feather_mode: 'feathered' (smooth interior transition) or 'hard' (bit-exact step cutoff).
        feather_radius: Feathering width in pixels (default: 3).

    Returns:
        blended_rgb: uint8 array [H, W, 3]
        metrics: dict containing bit-exact preservation stats and seam continuity metrics.
    """
    if orig_rgb.shape != restored_rgb.shape:
        raise ValueError(f"Shape mismatch: orig_rgb {orig_rgb.shape} vs restored_rgb {restored_rgb.shape}")

    h, w = orig_rgb.shape[:2]
    dmg_bool = (mask_binary > 0.5)
    undmg_bool = ~dmg_bool

    feather_mode = feather_mode.strip().lower()
    if feather_mode not in ("feathered", "hard"):
        raise ValueError(f"Invalid feather_mode: '{feather_mode}'. Must be 'feathered' or 'hard'.")

    # If no damage exists, return original unchanged
    if not np.any(dmg_bool):
        return orig_rgb.copy(), {
            "feather_mode": feather_mode,
            "feather_radius": feather_radius,
            "undamaged_max_diff": 0,
            "undamaged_mae": 0.0,
            "is_bit_exact": True,
            "seam_gradient_max": 0.0
        }

    if feather_mode == "hard" or feather_radius <= 0:
        alpha = dmg_bool.astype(np.float32)
        blended = orig_rgb.copy()
        blended[dmg_bool] = restored_rgb[dmg_bool]
    else:
        alpha = compute_interior_feather_alpha(mask_binary, feather_radius=feather_radius)
        alpha_3d = alpha[..., np.newaxis]
        blended_f = (1.0 - alpha_3d) * orig_rgb.astype(np.float32) + alpha_3d * restored_rgb.astype(np.float32)
        blended = np.clip(np.round(blended_f), 0, 255).astype(np.uint8)

    # STRICT INVARIANT: Force undamaged pixels to original values to guarantee bit-exactness
    blended[undmg_bool] = orig_rgb[undmg_bool]

    # Verification of bit-exact preservation on surviving substrate
    if np.any(undmg_bool):
        diff = np.abs(blended[undmg_bool].astype(np.int32) - orig_rgb[undmg_bool].astype(np.int32))
        undamaged_max_diff = int(np.max(diff))
        undamaged_mae = float(np.mean(diff))
    else:
        undamaged_max_diff = 0
        undamaged_mae = 0.0

    is_bit_exact = (undamaged_max_diff == 0)

    # Seam gradient analysis along the mask boundary
    seam_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate((dmg_bool * 255).astype(np.uint8), seam_kernel)
    boundary_ring = (dilated > 0) & undmg_bool

    metrics = {
        "feather_mode": feather_mode,
        "feather_radius": feather_radius if feather_mode == "feathered" else 0,
        "undamaged_max_diff": undamaged_max_diff,
        "undamaged_mae": round(undamaged_mae, 6),
        "is_bit_exact": is_bit_exact,
        "interior_alpha_mean": float(np.mean(alpha[dmg_bool])) if np.any(dmg_bool) else 0.0
    }

    return blended, metrics
