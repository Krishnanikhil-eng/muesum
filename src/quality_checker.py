"""
quality_checker.py — Heuristic quality-control checks for MUESUM artwork restorations.

DISCLAIMER
----------
All checks in this module are purely technical/heuristic. They detect obvious
machine-detectable failures such as NaN pixels, dimension mismatches, excessive
smoothing, or sharp colour boundaries at the inpainted edge. They do NOT and
CANNOT assess historical accuracy, artistic authenticity, or period-correct colour
palette. A restoration that passes every check here may still be artistically or
historically incorrect. All results should be reviewed by a conservator.
"""

from __future__ import annotations

import warnings
from typing import Optional

import cv2
import numpy as np


class RestorationQualityChecker:
    """
    Heuristic quality-control checks for artwork restoration outputs.

    IMPORTANT DISCLAIMER: These checks detect obvious technical failures
    (NaNs, dimension errors, excessive smoothing, boundary violations).
    They do NOT assess historical accuracy or artistic authenticity.
    A result that passes all checks may still be artistically incorrect.

    Usage
    -----
    checker = RestorationQualityChecker()
    result  = checker.run_all_checks(original, restored, mask_binary)

    Parameters
    ----------
    smoothing_ratio_threshold : float
        Fraction below which inpainted gradient is considered excessively smooth
        relative to the undamaged reference gradient. Default 0.20 (20 %).
    color_discontinuity_threshold : float
        Maximum allowed mean per-channel colour difference (in intensity units)
        at the mask boundary. Default 40.
    boundary_erosion_px : int
        Structuring-element half-size (in pixels) used for eroding and dilating
        the mask to extract interior / exterior boundary strips. Default 3.
    """

    def __init__(
        self,
        smoothing_ratio_threshold: float = 0.20,
        color_discontinuity_threshold: float = 40.0,
        boundary_erosion_px: int = 3,
    ) -> None:
        self.smoothing_ratio_threshold = smoothing_ratio_threshold
        self.color_discontinuity_threshold = color_discontinuity_threshold
        self.boundary_erosion_px = boundary_erosion_px

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_dimensions(
        self,
        original: np.ndarray,
        restored: np.ndarray,
    ) -> dict:
        """
        Verify that *restored* has exactly the same shape as *original*.

        Parameters
        ----------
        original : np.ndarray
            The source artwork image (H × W × C or H × W).
        restored : np.ndarray
            The restoration output to validate.

        Returns
        -------
        dict
            ``{"pass": bool, "issue": str | None}``

        Notes
        -----
        Shape must match in every axis (height, width, and channel count).
        A colour/grayscale mismatch is also a failure.
        """
        if original.shape == restored.shape:
            return {"pass": True, "issue": None}

        issue = (
            f"Shape mismatch: original {original.shape} vs restored {restored.shape}."
        )
        return {"pass": False, "issue": issue}

    def check_pixel_validity(self, restored: np.ndarray) -> dict:
        """
        Verify that every pixel value in *restored* is a valid finite number
        in the expected [0, 255] range.

        Parameters
        ----------
        restored : np.ndarray
            The restoration output to validate (uint8 or float).

        Returns
        -------
        dict
            ``{"pass": bool, "issue": str | None}``

        Notes
        -----
        Checks performed (in order):
        1. Any NaN values.
        2. Any Inf values (positive or negative).
        3. Any pixel < 0.
        4. Any pixel > 255.
        """
        arr = restored.astype(np.float64, copy=False)

        if np.any(np.isnan(arr)):
            return {"pass": False, "issue": "Restored image contains NaN values."}

        if np.any(np.isinf(arr)):
            return {"pass": False, "issue": "Restored image contains Inf values."}

        if np.any(arr < 0):
            count = int(np.sum(arr < 0))
            return {
                "pass": False,
                "issue": f"Restored image has {count} pixel(s) with negative values.",
            }

        if np.any(arr > 255):
            count = int(np.sum(arr > 255))
            return {
                "pass": False,
                "issue": f"Restored image has {count} pixel(s) with values > 255.",
            }

        return {"pass": True, "issue": None}

    def check_outside_mask_preservation(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        mask_binary: np.ndarray,
    ) -> dict:
        """
        Verify that undamaged pixels (mask == 0) are pixel-perfectly preserved.

        MUESUM's compositor in ``src/feathering.py`` already guarantees this by
        compositing the restoration only inside the damage mask.  This check is a
        redundant safety net that should *never* fail under normal operation.

        Parameters
        ----------
        original : np.ndarray
            The source artwork image, uint8 or float, H × W × C or H × W.
        restored : np.ndarray
            The restoration output to validate.
        mask_binary : np.ndarray
            Float32 array of shape H × W, values in [0, 1].
            ``1`` = damaged region, ``0`` = clean / undamaged region.

        Returns
        -------
        dict
            ``{"pass": bool, "undamaged_mae": float,
               "undamaged_max_diff": int, "issue": str | None}``

        Notes
        -----
        * Undamaged pixels are those where ``mask_binary < 0.5``.
        * Both MAE and MaxDiff must be **exactly 0** to pass, because the
          compositor must copy undamaged pixels without alteration.
        * When the entire image is damaged (empty undamaged set) the check
          passes trivially — there is nothing to preserve.
        """
        clean_mask = mask_binary < 0.5  # True where pixel is undamaged

        # Expand to channel axis if the arrays are colour
        if original.ndim == 3 and clean_mask.ndim == 2:
            clean_mask_3d = clean_mask[:, :, np.newaxis]
            clean_mask_3d = np.broadcast_to(clean_mask_3d, original.shape)
        else:
            clean_mask_3d = clean_mask

        if not np.any(clean_mask):
            # Entire image is damaged — nothing to preserve
            return {
                "pass": True,
                "undamaged_mae": 0.0,
                "undamaged_max_diff": 0,
                "issue": None,
            }

        orig_pixels = original.astype(np.float64)[clean_mask_3d]
        rest_pixels = restored.astype(np.float64)[clean_mask_3d]

        diff = np.abs(orig_pixels - rest_pixels)
        mae = float(np.mean(diff))
        max_diff = int(np.max(diff))

        if mae == 0.0 and max_diff == 0:
            return {
                "pass": True,
                "undamaged_mae": mae,
                "undamaged_max_diff": max_diff,
                "issue": None,
            }

        issue = (
            f"Undamaged pixels were modified: MAE={mae:.4f}, MaxDiff={max_diff}. "
            "The compositor should have preserved these exactly."
        )
        return {
            "pass": False,
            "undamaged_mae": mae,
            "undamaged_max_diff": max_diff,
            "issue": issue,
        }

    def check_excessive_smoothing(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        mask_binary: np.ndarray,
    ) -> dict:
        """
        Detect whether the inpainted region is excessively smooth compared with
        the original artwork's texture.

        Excessive smoothing is a common failure mode of deep inpainting models;
        it makes the inpainted patch look "painted over" rather than restored.

        DISCLAIMER: A low gradient inside the mask is a *warning*, not a
        definitive failure.  Flat-colour artworks (e.g. minimalist paintings,
        solid-colour decorative elements) may legitimately have near-zero
        gradients and will produce false positives here.  Always pair this
        check with visual inspection by a conservator.

        Parameters
        ----------
        original : np.ndarray
            The source artwork image (H × W × C or H × W, uint8 or float).
        restored : np.ndarray
            The restoration output.
        mask_binary : np.ndarray
            Float32 H × W mask (1 = damaged, 0 = clean).

        Returns
        -------
        dict
            ``{"pass": bool, "inpainted_gradient": float,
               "reference_gradient": float, "ratio": float,
               "issue": str | None}``

        Algorithm
        ---------
        1. Convert both images to uint8 grayscale.
        2. Compute Sobel gradient magnitude (combined x+y) for each.
        3. ``inpainted_gradient``  = mean gradient of *restored* where mask == 1.
        4. ``reference_gradient`` = mean gradient of *original* where mask == 0.
        5. ``ratio`` = inpainted / reference (clamped; 0.0 when reference is 0).
        6. Flag if ratio < ``smoothing_ratio_threshold`` (default 0.20).
        """
        damaged_mask = mask_binary >= 0.5   # True inside damage region
        clean_mask   = mask_binary <  0.5   # True in undamaged region

        def _sobel_magnitude(img_uint8: np.ndarray) -> np.ndarray:
            gray = (
                cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
                if img_uint8.ndim == 3
                else img_uint8
            )
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(sx ** 2 + sy ** 2)

        orig_u8 = _to_uint8(original)
        rest_u8 = _to_uint8(restored)

        grad_restored  = _sobel_magnitude(rest_u8)
        grad_original  = _sobel_magnitude(orig_u8)

        # Mean gradient inside the inpainted region
        if np.any(damaged_mask):
            inpainted_gradient = float(np.mean(grad_restored[damaged_mask]))
        else:
            # No damaged region — smoothing check is vacuously passing
            return {
                "pass": True,
                "inpainted_gradient": 0.0,
                "reference_gradient": 0.0,
                "ratio": 1.0,
                "issue": None,
            }

        # Mean gradient of original in the undamaged region (reference texture)
        if np.any(clean_mask):
            reference_gradient = float(np.mean(grad_original[clean_mask]))
        else:
            # Whole image is damaged — no reference texture available
            reference_gradient = 0.0

        if reference_gradient == 0.0:
            # Degenerate: original is flat-colour; no smoothing check possible
            ratio = 1.0
            return {
                "pass": True,
                "inpainted_gradient": inpainted_gradient,
                "reference_gradient": reference_gradient,
                "ratio": ratio,
                "issue": None,
            }

        ratio = inpainted_gradient / reference_gradient

        if ratio < self.smoothing_ratio_threshold:
            issue = (
                f"Inpainted region appears excessively smooth: gradient ratio "
                f"{ratio:.3f} < threshold {self.smoothing_ratio_threshold:.2f}. "
                "Note: flat-colour artworks may legitimately trigger this warning."
            )
            return {
                "pass": False,
                "inpainted_gradient": inpainted_gradient,
                "reference_gradient": reference_gradient,
                "ratio": ratio,
                "issue": issue,
            }

        return {
            "pass": True,
            "inpainted_gradient": inpainted_gradient,
            "reference_gradient": reference_gradient,
            "ratio": ratio,
            "issue": None,
        }

    def check_color_discontinuity(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        mask_binary: np.ndarray,
    ) -> dict:
        """
        Detect large colour jumps at the boundary between the inpainted region
        and the surrounding original artwork.

        A sharp colour discontinuity at the mask edge indicates the model's
        output does not blend smoothly with the surrounding artwork, which is
        visually obvious and undesirable.

        Parameters
        ----------
        original : np.ndarray
            The source artwork image (H × W × C or H × W, uint8 or float).
        restored : np.ndarray
            The restoration output (same shape/dtype as *original*).
        mask_binary : np.ndarray
            Float32 H × W mask (1 = damaged, 0 = clean).

        Returns
        -------
        dict
            ``{"pass": bool, "max_channel_diff": float, "issue": str | None}``

        Algorithm
        ---------
        * ``mask_uint8`` = thresholded mask at 0.5, cast to uint8 (0 or 255).
        * ``interior_boundary`` = mask eroded by ``boundary_erosion_px``-px
          square kernel, XOR'd with the eroded result → a strip just inside
          the damage boundary.
        * ``exterior_boundary`` = mask dilated by the same kernel, XOR'd with
          the original mask → a strip just outside.
        * Measure mean per-channel absolute difference:
          ``restored[interior_boundary]`` vs ``original[exterior_boundary]``.
        * Flag if any channel's mean > ``color_discontinuity_threshold`` (40).

        Edge cases
        ----------
        * Empty mask or no-boundary situation: returns pass=True.
        * Grayscale images are treated as single-channel.
        """
        mask_uint8 = (mask_binary >= 0.5).astype(np.uint8) * 255

        k = self.boundary_erosion_px
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * k + 1, 2 * k + 1)
        )

        eroded  = cv2.erode(mask_uint8,  kernel)
        dilated = cv2.dilate(mask_uint8, kernel)

        # Interior boundary: pixels that are in the mask but not in its erosion
        interior_boundary = (mask_uint8 > 0) & (eroded == 0)
        # Exterior boundary: pixels that are in the dilation but not in the mask
        exterior_boundary = (dilated > 0) & (mask_uint8 == 0)

        if not np.any(interior_boundary) or not np.any(exterior_boundary):
            # Mask is too small / entire image is masked — no boundary to check
            return {"pass": True, "max_channel_diff": 0.0, "issue": None}

        orig_f = original.astype(np.float64)
        rest_f = restored.astype(np.float64)

        if orig_f.ndim == 2:
            # Grayscale: add channel axis for uniform handling
            orig_f = orig_f[:, :, np.newaxis]
            rest_f = rest_f[:, :, np.newaxis]

        num_channels = orig_f.shape[2]
        channel_diffs: list[float] = []

        for c in range(num_channels):
            interior_pixels = rest_f[:, :, c][interior_boundary]
            exterior_pixels = orig_f[:, :, c][exterior_boundary]

            # Compare mean colours at each side of the boundary
            mean_interior = float(np.mean(interior_pixels))
            mean_exterior = float(np.mean(exterior_pixels))
            channel_diffs.append(abs(mean_interior - mean_exterior))

        max_channel_diff = float(max(channel_diffs))

        if max_channel_diff > self.color_discontinuity_threshold:
            issue = (
                f"Colour discontinuity detected at mask boundary: "
                f"max channel diff = {max_channel_diff:.1f} intensity units "
                f"(threshold {self.color_discontinuity_threshold:.0f})."
            )
            return {"pass": False, "max_channel_diff": max_channel_diff, "issue": issue}

        return {"pass": True, "max_channel_diff": max_channel_diff, "issue": None}

    # ------------------------------------------------------------------
    # Aggregated runner
    # ------------------------------------------------------------------

    def run_all_checks(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        mask_binary: np.ndarray,
    ) -> dict:
        """
        Run all five quality checks and produce a consolidated result with an
        escalation recommendation.

        DISCLAIMER: "Passed" here means no *technical* anomalies were detected.
        It does NOT imply historical accuracy or artistic correctness.  Every
        restoration should still be reviewed by a conservator.

        Parameters
        ----------
        original : np.ndarray
            The source artwork image (H × W × C or H × W, uint8 or float).
        restored : np.ndarray
            The restoration output to validate.
        mask_binary : np.ndarray
            Float32 H × W mask (1 = damaged, 0 = clean).

        Returns
        -------
        dict
            ::

                {
                  "passed":          bool,       # True iff ALL hard checks pass
                  "warnings":        list[str],  # Non-fatal issues
                  "hard_failures":   list[str],  # Fatal technical failures
                  "details":         dict,        # Per-check result dicts
                  "recommendation":  str,         # Human-readable guidance
                  "escalate_to_cloud": bool,
                }

        Escalation logic
        ----------------
        * ``dimensions`` or ``pixel_validity`` fail →
          ``escalate_to_cloud=True``, recommendation="Escalate to cloud AI"
        * ``outside_mask_preservation`` fails →
          ``escalate_to_cloud=False``,
          recommendation="Human review required"
          (compositor bug — cloud AI cannot fix this)
        * ``smoothing`` or ``color_discontinuity`` flagged (warnings) →
          ``escalate_to_cloud=True``,
          recommendation="Review recommended — cloud AI may produce better results"
        * All pass →
          ``escalate_to_cloud=False``,
          recommendation="Looks acceptable — visual review advised"
        """
        # --- Run checks -----------------------------------------------
        dim_result  = self.check_dimensions(original, restored)
        pxv_result  = self.check_pixel_validity(restored)

        # Downstream checks depend on matching shapes; skip gracefully if broken
        if dim_result["pass"]:
            pres_result  = self.check_outside_mask_preservation(original, restored, mask_binary)
            smooth_result = self.check_excessive_smoothing(original, restored, mask_binary)
            disc_result   = self.check_color_discontinuity(original, restored, mask_binary)
        else:
            _skip = {"pass": False, "issue": "Skipped due to dimension mismatch."}
            pres_result   = {**_skip, "undamaged_mae": 0.0, "undamaged_max_diff": 0}
            smooth_result = {**_skip, "inpainted_gradient": 0.0,
                             "reference_gradient": 0.0, "ratio": 0.0}
            disc_result   = {**_skip, "max_channel_diff": 0.0}

        details = {
            "dimensions":                dim_result,
            "pixel_validity":            pxv_result,
            "outside_mask_preservation": pres_result,
            "excessive_smoothing":       smooth_result,
            "color_discontinuity":       disc_result,
        }

        # --- Classify failures and warnings ---------------------------
        hard_failures: list[str] = []
        warnings_list: list[str] = []

        # Hard checks
        for key in ("dimensions", "pixel_validity", "outside_mask_preservation"):
            r = details[key]
            if not r["pass"] and r.get("issue") and "Skipped" not in r["issue"]:
                hard_failures.append(r["issue"])

        # Soft checks (warnings)
        for key in ("excessive_smoothing", "color_discontinuity"):
            r = details[key]
            if not r["pass"] and r.get("issue") and "Skipped" not in r["issue"]:
                warnings_list.append(r["issue"])

        passed = len(hard_failures) == 0

        # --- Escalation logic (priority order) ------------------------
        dim_failed  = not dim_result["pass"]
        pxv_failed  = not pxv_result["pass"]
        pres_failed = not pres_result["pass"] and "Skipped" not in (pres_result.get("issue") or "")
        has_warnings = len(warnings_list) > 0

        if dim_failed or pxv_failed:
            escalate_to_cloud = True
            recommendation    = "Escalate to cloud AI"
        elif pres_failed:
            escalate_to_cloud = False
            recommendation    = "Human review required"
        elif has_warnings:
            escalate_to_cloud = True
            recommendation    = "Review recommended — cloud AI may produce better results"
        else:
            escalate_to_cloud = False
            recommendation    = "Looks acceptable — visual review advised"

        return {
            "passed":           passed,
            "warnings":         warnings_list,
            "hard_failures":    hard_failures,
            "details":          details,
            "recommendation":   recommendation,
            "escalate_to_cloud": escalate_to_cloud,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Convert an array to uint8 for OpenCV operations.

    * If already uint8, returned as-is (no copy).
    * If float, assumed to be in [0, 255] and clipped/rounded.
      NaN / Inf values are replaced by 0 before clipping so that the cast
      does not raise a RuntimeWarning (NaN/Inf inputs are caught earlier by
      ``check_pixel_validity`` anyway).
    * Other integer dtypes are cast directly.
    """
    if arr.dtype == np.uint8:
        return arr
    safe = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    return np.round(np.clip(safe, 0, 255)).astype(np.uint8)
