# src/damage_detection.py
"""
MUESUM — Computer Vision Damage Detection & Heuristic Mask Generator
===================================================================
Provides automated, heuristic-guided detection of physical degradation on artworks:
  1. Dark Craquelure & Cracks: Multi-scale morphological Blackhat filtering.
  2. Bright Scratches & Tears: Multi-scale morphological Tophat filtering.
  3. Paint Loss & Lacunae: Localized contrast and color delamination segmentation.
  4. General Combined: Integrated multi-feature detection.

CRITICAL CURATORIAL NOTICE:
Heuristic computer vision detection provides an initial hypothesis. Because historical
artworks contain intentional high-contrast brushwork, contour lines, and dark pigments,
automated detection can produce false positives on dark eyes, contours, and hair.
Conservator review and interactive manual correction (via the brush tool) are
strongly recommended before applying neural inpainting.
"""

from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import cv2
from PIL import Image


def detect_damage_mask(
    image: Union[np.ndarray, Image.Image],
    mode: str = "general",
    sensitivity: float = 3.0,
    min_area: int = 5
) -> Tuple[np.ndarray, Image.Image, Dict[str, Any]]:
    """
    Automatically detect damage blemishes and generate a binary damage mask.

    Args:
        image: Input artwork as PIL Image or uint8 RGB NumPy array [H, W, 3].
        mode: Detection algorithm profile:
              - 'craquelure': Multi-scale blackhat for dark hairline cracks and fissures.
              - 'bright_scratches': Multi-scale tophat for light scratches and gouges.
              - 'paint_loss': Local luminance contrast segmentation for missing paint.
              - 'general': Balanced combination of scratches, cracks, and surface blemishes.
        sensitivity: Sensitivity multiplier (1.0 = conservative/low, 5.0 = aggressive/high).
        min_area: Minimum connected-component pixel area to keep (filters micro-speckle noise).

    Returns:
        tuple containing:
            - mask_uint8: np.ndarray (uint8 [H, W], 255 = damage, 0 = clean)
            - overlay_pil: PIL.Image of artwork with red semi-transparent highlight on damage
            - diagnostics: dict with coverage %, component counts, and curatorial notes
    """
    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        img_rgb = image.copy()
        if img_rgb.ndim == 2:
            img_rgb = np.stack([img_rgb] * 3, axis=-1)
        elif img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
            img_rgb = img_rgb[:, :, :3]
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Base structuring elements
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Sensitivity normalization: higher sensitivity -> lower threshold percentile
    # Sensitivity ranges 1.0 to 5.0; map to percentile roughly 98.0 down to 88.0
    sens = float(np.clip(sensitivity, 1.0, 5.0))
    percentile = 98.5 - (sens - 1.0) * 2.5  # e.g., sens 3.0 -> 93.5 percentile

    mode = mode.lower().strip()
    raw_response = np.zeros((h, w), dtype=np.float32)

    if mode in ("craquelure", "cracks"):
        # Multi-scale blackhat detects dark narrow structures on lighter background
        bh3 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k3)
        bh5 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k5)
        bh7 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k7)
        raw_response = np.maximum.reduce([bh3.astype(np.float32), bh5.astype(np.float32), bh7.astype(np.float32)])

    elif mode in ("bright_scratches", "scratches_light"):
        # Multi-scale tophat detects bright narrow structures on darker background
        th3 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k3)
        th5 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k5)
        th7 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k7)
        raw_response = np.maximum.reduce([th3.astype(np.float32), th5.astype(np.float32), th7.astype(np.float32)])

    elif mode in ("paint_loss", "lacunae"):
        # Local luminance anomaly / contrast against blurred neighborhood
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(gray, blurred).astype(np.float32)
        raw_response = diff

    else:  # 'general' or combined
        bh3 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k3).astype(np.float32)
        bh5 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k5).astype(np.float32)
        th3 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k3).astype(np.float32)
        th5 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k5).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        local_diff = cv2.absdiff(gray, blurred).astype(np.float32)

        raw_response = np.maximum.reduce([bh3, bh5, th3, th5, local_diff * 0.7])

    # Dynamic thresholding based on image response distribution
    pos_responses = raw_response[raw_response > 2.0]
    if len(pos_responses) > 0:
        thresh_val = float(np.percentile(pos_responses, percentile))
    else:
        thresh_val = 15.0

    # Ensure minimum noise floor threshold
    thresh_val = max(thresh_val, 8.0 / (sens / 2.0))
    mask_bool = raw_response >= thresh_val

    # Clean up with morphological opening and area filtering
    mask_clean = mask_bool.astype(np.uint8) * 255
    if min_area > 1:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
        mask_filtered = np.zeros_like(mask_clean)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                mask_filtered[labels == i] = 255
        mask_clean = mask_filtered

    # Slightly dilate detected crack pixels by 1 px to ensure full edge coverage
    if mode in ("craquelure", "cracks", "general"):
        mask_clean = cv2.dilate(mask_clean, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # Compute coverage percentage
    coverage_pct = float((np.sum(mask_clean == 255) / mask_clean.size) * 100.0)

    # Generate red overlay visualization
    overlay_img = create_mask_overlay(img_rgb, mask_clean)
    overlay_pil = Image.fromarray(overlay_img)

    diagnostics = {
        "detection_mode": mode,
        "sensitivity": sensitivity,
        "threshold_value": round(float(thresh_val), 2),
        "mask_coverage_pct": round(coverage_pct, 2),
        "total_damaged_pixels": int(np.sum(mask_clean == 255)),
        "image_dimensions": (w, h),
        "verified_curatorial_status": "Heuristic Hypothesis (Review / Manual Correction Advised)",
        "limitation_advisory": (
            "Automatic CV detection uses morphological contrast filters. High-contrast artwork details "
            "(dark eyelashes, eyes, fine linework) can be flagged. Use the interactive brush canvas to "
            "inspect and edit the mask before final inpainting."
        )
    }

    return mask_clean, overlay_pil, diagnostics


def create_mask_overlay(
    image_rgb: np.ndarray,
    mask_uint8: np.ndarray,
    color_rgb: Tuple[int, int, int] = (255, 30, 40),
    alpha: float = 0.45
) -> np.ndarray:
    """
    Renders a semi-transparent colored highlight over damaged pixels on the original artwork.
    """
    overlay = image_rgb.copy()
    damaged = (mask_uint8 > 127)
    if not np.any(damaged):
        return overlay

    color_arr = np.array(color_rgb, dtype=np.float32)
    orig_f = overlay.astype(np.float32)

    # Blend original and color on damaged pixels
    blended = (1.0 - alpha) * orig_f[damaged] + alpha * color_arr
    orig_f[damaged] = blended
    return np.clip(orig_f, 0, 255).astype(np.uint8)

