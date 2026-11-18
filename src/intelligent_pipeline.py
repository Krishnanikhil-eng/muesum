"""
intelligent_pipeline.py
========================
Intelligent restoration pipeline for MUESUM.

Ties together:
  - RestorationRouter (src/restoration_router.py) — mask-complexity analysis and pre-restoration routing
  - restore_artwork_pipeline (src/inference.py) — local ResNet34-UNet restoration
  - RestorationQualityChecker (src/quality_checker.py) — post-restoration heuristic quality checks
  - CloudInpaintingService (src/cloud_inpainting_service.py) — Cloudflare Workers AI inpainting fallback
  - blend_with_boundary_feathering (src/feathering.py) — 100% bit-exact authentic substrate compositor

Guarantees:
  - 100% bit-exact authentic canvas preservation outside the damage mask (MAE = 0.0, MaxDiff = 0)
  - At most ONE Cloud AI call per restoration action
  - Graceful fallback to local UNet if cloud is unavailable or fails
  - Transparent status feedback explaining which engine was chosen and why
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Core imports
from inference import restore_artwork_pipeline, find_default_checkpoint
from feathering import blend_with_boundary_feathering

# Router import
try:
    from restoration_router import RestorationRouter
    _ROUTER_AVAILABLE = True
except ImportError:
    _ROUTER_AVAILABLE = False
    RestorationRouter = None  # type: ignore

# Quality checker import
try:
    from quality_checker import RestorationQualityChecker
    _QC_AVAILABLE = True
except ImportError:
    _QC_AVAILABLE = False
    RestorationQualityChecker = None  # type: ignore

# Cloud service import
try:
    from cloud_inpainting_service import get_cloud_service, CloudInpaintingService
    _CLOUD_AVAILABLE = True
except ImportError:
    _CLOUD_AVAILABLE = False
    get_cloud_service = None  # type: ignore
    CloudInpaintingService = None  # type: ignore


_ENGINE_LABELS = {
    "local_unet": "Local ResNet34-UNet",
    "cloud_ai": "Cloud AI (Cloudflare SD 1.5)",
    "local_first_then_cloud": "Hybrid (Local → Cloud Standby)",
    "no_mask": "Local UNet (Raw / No Mask)",
}


def intelligent_restore(
    image: Union[Image.Image, np.ndarray],
    mask: Optional[Union[Image.Image, np.ndarray]] = None,
    ckpt_path: Optional[str] = None,
    blend_mode: str = "mask_guided",
    feather_mode: str = "feathered",
    feather_radius: int = 3,
    max_dimension: Optional[int] = 1024,
    force_engine: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run intelligent damage-aware restoration on an artwork image.

    Parameters
    ----------
    image : PIL.Image.Image or np.ndarray
        The damaged artwork image.
    mask : PIL.Image.Image or np.ndarray or None
        Binary damage mask (255 = damaged, 0 = undamaged clean canvas).
        Pass None for raw UNet mode without mask guidance.
    ckpt_path : str, optional
        Path to local ResNet34-UNet checkpoint. If None, resolved automatically.
    blend_mode : str
        "mask_guided" (default) or "raw_unet".
    feather_mode : str
        "feathered" (smoothstep interior seam) or "hard" (step cutoff).
    feather_radius : int
        Feathering radius in pixels. Default: 3.
    max_dimension : int or None
        Resize longest edge before neural inference (None = native).
    force_engine : str, optional
        Override automatic routing: "local_unet" or "cloud_ai".

    Returns
    -------
    dict with keys:
        restored_image   : np.ndarray (H, W, 3) uint8
        original_image   : np.ndarray (H, W, 3) uint8
        raw_unet_image   : np.ndarray (H, W, 3) uint8 or None
        mask_preview     : PIL.Image.Image or None
        diagnostics      : dict
        status_message   : str
        final_engine_used: str ("local_unet", "cloud_ai", "no_mask")
        quality_report   : dict
    """
    t_start = time.perf_counter()

    # Standardize image to PIL Image and uint8 numpy array
    if isinstance(image, np.ndarray):
        img_pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        img_pil = image.convert("RGB")

    orig_rgb = np.array(img_pil, dtype=np.uint8)
    h, w = orig_rgb.shape[:2]

    # Resolve checkpoint
    target_ckpt = ckpt_path.strip() if ckpt_path else find_default_checkpoint()

    # Initialize quality checker
    qc = RestorationQualityChecker() if _QC_AVAILABLE else None

    # Cloud service reference
    cloud_svc = get_cloud_service() if _CLOUD_AVAILABLE else None

    # Handle no-mask or raw UNet mode
    if mask is None or blend_mode == "raw_unet":
        local_res = restore_artwork_pipeline(
            damaged_image=img_pil,
            damage_mask=None,
            ckpt_path=target_ckpt,
            blend_mode="raw_unet",
            feather_mode=feather_mode,
            feather_radius=feather_radius,
            max_dimension=max_dimension,
            return_diagnostics=True,
        )
        return {
            "restored_image": local_res["restored_image"],
            "original_image": local_res["original_image"],
            "raw_unet_image": local_res["raw_unet_image"],
            "mask_preview": None,
            "diagnostics": local_res["diagnostics"],
            "status_message": "🏠 Local UNet Selected — Raw UNet convolution (no mask applied).",
            "final_engine_used": "no_mask",
            "quality_report": {"passed": True, "details": {}, "recommendation": "Raw mode - unguided"},
        }

    # Standardize mask to uint8 [H, W] and binary float32 [H, W]
    if isinstance(mask, np.ndarray):
        m_arr = np.squeeze(mask)
        if m_arr.shape[:2] != (h, w):
            mask_pil = Image.fromarray((m_arr > 127).astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
            mask_uint8 = np.array(mask_pil, dtype=np.uint8)
        else:
            mask_uint8 = (m_arr > 127).astype(np.uint8) * 255
            mask_pil = Image.fromarray(mask_uint8, mode="L")
    else:
        if mask.size != (w, h):
            mask_pil = mask.convert("L").resize((w, h), Image.NEAREST)
        else:
            mask_pil = mask.convert("L")
        mask_uint8 = (np.array(mask_pil) > 127).astype(np.uint8) * 255

    mask_binary = (mask_uint8 > 127).astype(np.float32)
    dmg_bool = (mask_binary > 0.5)

    # ------------------------------------------------------------------
    # Step 1: Pre-Restoration Mask Analysis & Routing
    # ------------------------------------------------------------------
    router = RestorationRouter() if _ROUTER_AVAILABLE else None
    if router is not None:
        mask_analysis = router.analyze_mask(mask_uint8)
        routing_decision = router.route(mask_analysis, image_shape=orig_rgb.shape)
    else:
        mask_analysis = {
            "total_coverage_pct": float(np.mean(dmg_bool) * 100.0),
            "largest_component_pct": float(np.mean(dmg_bool) * 100.0),
            "num_components": 1,
        }
        routing_decision = {
            "engine": "local_unet",
            "reason": "Router unavailable; defaulting to Local UNet.",
            "confidence": "low",
            "local_suitable": True,
            "cloud_recommended": False,
        }

    initial_engine = force_engine if force_engine else routing_decision["engine"]
    status_message = ""
    cloud_called = False
    final_engine_used = initial_engine
    restored_raw: Optional[np.ndarray] = None
    local_res: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Step 2: Engine Execution & Escalation Logic
    # ------------------------------------------------------------------
    # Branch A: Cloud AI requested by routing
    if initial_engine == "cloud_ai":
        if cloud_svc is not None and cloud_svc.is_available():
            cloud_called = True
            cloud_res = cloud_svc.restore(image_pil=img_pil, mask_uint8=mask_uint8)
            if cloud_res.get("success") and cloud_res.get("image") is not None:
                # Cloud succeeded: resize to original image dimensions if needed
                cloud_img_pil = cloud_res["image"].convert("RGB").resize((w, h), Image.LANCZOS)
                restored_raw = np.array(cloud_img_pil, dtype=np.uint8)
                final_engine_used = "cloud_ai"
                status_message = "☁️ Cloud AI Selected — Large/complex damage restored via Cloudflare SD 1.5 inpainting."
            else:
                # Cloud API failed -> graceful fallback to local UNet
                err_msg = cloud_res.get("error", "Unknown API error")
                status_message = f"⚠️ Cloud AI failed ({err_msg}). Falling back to Local UNet."
                final_engine_used = "local_unet"
        else:
            # Cloud credentials missing -> graceful fallback to local UNet
            status_message = "⚠️ Cloud AI recommended (complex damage), but cloud credentials not configured. Falling back to Local UNet."
            final_engine_used = "local_unet"

    # Branch B: Local UNet (either originally routed or as cloud fallback)
    if final_engine_used != "cloud_ai" or restored_raw is None:
        local_res = restore_artwork_pipeline(
            damaged_image=img_pil,
            damage_mask=mask_pil,
            ckpt_path=target_ckpt,
            blend_mode=blend_mode,
            feather_mode=feather_mode,
            feather_radius=feather_radius,
            max_dimension=max_dimension,
            return_diagnostics=True,
        )
        restored_raw = local_res["raw_unet_image"]

        # --------------------------------------------------------------
        # Step 3: Quality Checks on Local Restoration
        # --------------------------------------------------------------
        # Create candidate blend to check local quality
        local_blended, _ = blend_with_boundary_feathering(
            orig_rgb=orig_rgb,
            restored_rgb=restored_raw,
            mask_binary=mask_binary,
            feather_mode=feather_mode,
            feather_radius=feather_radius,
        )

        qc_report = (
            qc.run_all_checks(orig_rgb, local_blended, mask_binary)
            if qc is not None
            else {"passed": True, "escalate_to_cloud": False, "warnings": [], "hard_failures": []}
        )

        # Check if local output should escalate to Cloud AI fallback
        # Rule: Only call cloud if not already called, cloud is available, QC flagged escalation,
        # and caller did NOT explicitly force local_unet.
        should_escalate = (
            force_engine != "local_unet"
            and (
                qc_report.get("escalate_to_cloud", False)
                or initial_engine == "local_first_then_cloud"
            )
        )

        if should_escalate and not cloud_called and cloud_svc is not None and cloud_svc.is_available():
            # Cloud fallback escalation (at most ONE API call)
            cloud_called = True
            cloud_res = cloud_svc.restore(image_pil=img_pil, mask_uint8=mask_uint8)
            if cloud_res.get("success") and cloud_res.get("image") is not None:
                cloud_img_pil = cloud_res["image"].convert("RGB").resize((w, h), Image.LANCZOS)
                cloud_candidate_raw = np.array(cloud_img_pil, dtype=np.uint8)
                cloud_blended, _ = blend_with_boundary_feathering(
                    orig_rgb=orig_rgb,
                    restored_rgb=cloud_candidate_raw,
                    mask_binary=mask_binary,
                    feather_mode=feather_mode,
                    feather_radius=feather_radius,
                )
                qc_cloud = (
                    qc.run_all_checks(orig_rgb, cloud_blended, mask_binary)
                    if qc is not None
                    else {"passed": True, "hard_failures": []}
                )

                # Prefer cloud if it passes without hard failures
                if qc_cloud.get("passed", False) or len(qc_cloud.get("hard_failures", [])) == 0:
                    restored_raw = cloud_candidate_raw
                    final_engine_used = "cloud_ai"
                    status_message = "☁️ Cloud Fallback Used — Local UNet flagged for quality; escalated to Cloud AI."
                else:
                    # Cloud also flagged; retain local result with human review note
                    final_engine_used = "local_unet"
                    status_message = "⚠️ Local UNet flagged and Cloud AI did not improve result; conservator review required."
            else:
                # Cloud call failed
                final_engine_used = "local_unet"
                err_msg = cloud_res.get("error", "API error")
                status_message = f"⚠️ Local UNet flagged for review; Cloud AI fallback failed ({err_msg}). Local result retained."
        else:
            final_engine_used = "local_unet"
            if not status_message:
                if not qc_report.get("passed", True):
                    status_message = "⚠️ Local UNet Selected — Quality checks flagged warnings. Visual conservator review recommended."
                elif initial_engine == "local_first_then_cloud":
                    status_message = "🏠 Local UNet Accepted — Damage in hybrid range; local restoration passed quality checks."
                else:
                    status_message = "🏠 Local UNet Selected — Localized damage within ResNet34-UNet capacity."

    # ------------------------------------------------------------------
    # Step 4: Strict Preservation Guarantee Compositor (Requirement 8)
    # ------------------------------------------------------------------
    # All final outputs MUST pass through blend_with_boundary_feathering
    # to guarantee MAE = 0 and MaxDiff = 0 on undamaged authentic pixels.
    final_blended, blend_metrics = blend_with_boundary_feathering(
        orig_rgb=orig_rgb,
        restored_rgb=restored_raw,
        mask_binary=mask_binary,
        feather_mode=feather_mode,
        feather_radius=feather_radius,
    )

    # STRICT INVARIANT: Force undamaged pixels to original values to guarantee bit-exactness
    final_blended[~dmg_bool] = orig_rgb[~dmg_bool]

    # Calculate exact verification metrics
    if np.any(~dmg_bool):
        diff = np.abs(final_blended[~dmg_bool].astype(np.int32) - orig_rgb[~dmg_bool].astype(np.int32))
        undamaged_max_diff = int(np.max(diff))
        undamaged_mae = float(np.mean(diff))
    else:
        undamaged_max_diff = 0
        undamaged_mae = 0.0

    is_bit_exact = (undamaged_max_diff == 0)

    # Run final QC on composite output
    final_qc = (
        qc.run_all_checks(orig_rgb, final_blended, mask_binary)
        if qc is not None
        else {"passed": is_bit_exact, "warnings": [], "hard_failures": []}
    )

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000.0, 2)

    # Build comprehensive curatorial diagnostics
    diagnostics: Dict[str, Any] = {
        "engine": final_engine_used,
        "engine_label": _ENGINE_LABELS.get(final_engine_used, final_engine_used),
        "routing": routing_decision,
        "blend_mode": blend_mode,
        "feather_mode": feather_mode,
        "feather_radius": feather_radius if feather_mode == "feathered" else 0,
        "is_bit_exact": is_bit_exact,
        "undamaged_mae": undamaged_mae,
        "undamaged_max_diff": undamaged_max_diff,
        "mask_coverage_pct": round(mask_analysis["total_coverage_pct"], 2),
        "latency_ms": latency_ms,
        "original_size": [w, h],
        "eval_size": local_res["diagnostics"]["eval_size"] if local_res else [w, h],
        "quality_report": {
            "passed": final_qc.get("passed", True),
            "recommendation": final_qc.get("recommendation", "Review advised"),
            "warnings": final_qc.get("warnings", []),
            "hard_failures": final_qc.get("hard_failures", []),
        },
    }

    raw_output = local_res["raw_unet_image"] if local_res else restored_raw

    return {
        "restored_image": final_blended,
        "original_image": orig_rgb,
        "raw_unet_image": raw_output,
        "mask_preview": Image.fromarray(mask_uint8, mode="L"),
        "diagnostics": diagnostics,
        "status_message": status_message,
        "final_engine_used": final_engine_used,
        "quality_report": final_qc,
    }
