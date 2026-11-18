"""
restoration_router.py
=====================
Intelligent damage routing module for MUESUM (Museum Artwork Restoration System).

This module analyzes a binary damage mask produced by `damage_detection.py` and
decides—*before* any restoration runs—whether to process the artwork with the
local ResNet34-UNet model or to escalate to a Cloud AI backend.

The routing logic is grounded in MUESUM's own failure analysis
(FAILED_RESTORATION_DIAGNOSIS.md):  the local UNet excels at fine scratches,
craquelure, and small flaking but degrades on large structural voids.

Typical usage
-------------
    from restoration_router import RestorationRouter
    import cv2, numpy as np

    router = RestorationRouter()

    mask = cv2.imread("damage_mask.png", cv2.IMREAD_GRAYSCALE)
    analysis  = router.analyze_mask(mask)
    decision  = router.route(analysis, image_shape=(H, W))
    status    = router.format_routing_status(decision)
    print(status)
"""

from __future__ import annotations

import numpy as np
import cv2


class RestorationRouter:
    """
    Analyses a binary damage mask and routes the restoration job to the most
    appropriate engine: the local ResNet34-UNet, Cloud AI, or a hybrid mode
    where the local model is attempted first with Cloud AI on standby.

    Parameters
    ----------
    local_max_coverage_pct : float
        If total damaged area exceeds this percentage of the image, Cloud AI
        is preferred.  Default: 35.0 (%).
    local_max_component_pct : float
        If the single largest connected damage region exceeds this percentage
        of the image, Cloud AI is preferred.  Default: 20.0 (%).
    local_max_width_ratio : float
        If the bounding-box width of the largest component exceeds this
        fraction of the image width, Cloud AI is preferred.  Default: 0.35.
    num_components_complex : int
        When the number of distinct damage regions exceeds this value a
        "complexity" flag is raised (used as a tiebreaker, not a hard rule).
        Default: 15.
    """

    def __init__(
        self,
        *,
        local_max_coverage_pct: float = 35.0,
        local_max_component_pct: float = 20.0,
        local_max_width_ratio: float = 0.35,
        num_components_complex: int = 15,
    ) -> None:
        self.local_max_coverage_pct = local_max_coverage_pct
        self.local_max_component_pct = local_max_component_pct
        self.local_max_width_ratio = local_max_width_ratio
        self.num_components_complex = num_components_complex

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_mask(self, mask_uint8: np.ndarray) -> dict:
        """
        Compute spatial statistics for a binary damage mask.

        The mask must be a 2-D ``numpy.ndarray`` of dtype ``uint8`` where
        pixel value **255** means *damaged* and **0** means *clean*.

        Parameters
        ----------
        mask_uint8 : np.ndarray
            Binary mask with shape ``(H, W)`` and dtype ``uint8``.

        Returns
        -------
        dict with keys:
            total_coverage_pct : float
                Percentage of image pixels that are damaged (0–100).
            largest_component_pct : float
                Percentage of image pixels covered by the single largest
                connected damage region.
            largest_component_bbox : tuple[int, int, int, int]
                Bounding box ``(x, y, w, h)`` of the largest component.
                ``(0, 0, 0, 0)`` when no damage is present.
            largest_component_width_px : int
                Width (``w``) of that bounding box in pixels.
            largest_component_height_px : int
                Height (``h``) of that bounding box in pixels.
            num_components : int
                Total number of distinct connected damage regions.
            avg_component_pct : float
                Mean area of a damage region expressed as % of the image.
            estimated_max_width_px : int
                The larger of ``largest_component_width_px`` and
                ``largest_component_height_px``—useful as a single
                "how big is the worst void" figure.

        Raises
        ------
        ValueError
            If ``mask_uint8`` is not a 2-D array.
        """
        if mask_uint8.ndim != 2:
            raise ValueError(
                f"mask_uint8 must be a 2-D array; got shape {mask_uint8.shape}"
            )

        total_pixels: int = mask_uint8.shape[0] * mask_uint8.shape[1]
        damaged_pixels: int = int(np.count_nonzero(mask_uint8))
        total_coverage_pct: float = (damaged_pixels / total_pixels) * 100.0

        # Connected-component analysis (8-connectivity)
        binary = (mask_uint8 > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # Label 0 is the background; components start at label 1
        num_components: int = max(0, num_labels - 1)

        # --- Per-component metrics ---
        largest_component_pct: float = 0.0
        largest_component_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
        largest_component_width_px: int = 0
        largest_component_height_px: int = 0
        avg_component_pct: float = 0.0
        estimated_damage_width_px: int = 0

        # Estimated stroke width via distance transform
        if damaged_pixels > 0:
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            estimated_damage_width_px = int(np.round(np.max(dist) * 2))

        if num_components > 0:
            # stats columns: CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH,
            #                CC_STAT_HEIGHT, CC_STAT_AREA
            component_areas = stats[1:, cv2.CC_STAT_AREA]  # skip background

            largest_idx = int(np.argmax(component_areas))  # 0-based among components
            largest_area = int(component_areas[largest_idx])
            largest_component_pct = (largest_area / total_pixels) * 100.0

            # stats rows are offset by 1 because row 0 is background
            stat_row = stats[largest_idx + 1]
            x = int(stat_row[cv2.CC_STAT_LEFT])
            y = int(stat_row[cv2.CC_STAT_TOP])
            w = int(stat_row[cv2.CC_STAT_WIDTH])
            h = int(stat_row[cv2.CC_STAT_HEIGHT])
            largest_component_bbox = (x, y, w, h)
            largest_component_width_px = w
            largest_component_height_px = h

            avg_area = float(np.mean(component_areas))
            avg_component_pct = (avg_area / total_pixels) * 100.0

        estimated_max_width_px: int = max(
            largest_component_width_px, largest_component_height_px
        )

        return {
            "total_coverage_pct": round(total_coverage_pct, 4),
            "largest_component_pct": round(largest_component_pct, 4),
            "largest_component_bbox": largest_component_bbox,
            "largest_component_width_px": largest_component_width_px,
            "largest_component_height_px": largest_component_height_px,
            "num_components": num_components,
            "avg_component_pct": round(avg_component_pct, 4),
            "estimated_max_width_px": estimated_max_width_px,
            "estimated_damage_width_px": estimated_damage_width_px,
        }

    def route(self, mask_analysis: dict, image_shape: tuple) -> dict:
        """
        Decide which restoration engine should process this artwork.

        The decision is derived from the spatial statistics in
        ``mask_analysis`` (produced by :meth:`analyze_mask`) and from the
        configurable thresholds set at construction time.

        Routing hierarchy (evaluated top-to-bottom; first match wins):

        1. **Cloud AI** — triggered by *any* of:
           - ``total_coverage_pct`` > ``local_max_coverage_pct``
           - ``largest_component_pct`` > ``local_max_component_pct``
           - bbox width / image width  > ``local_max_width_ratio``

        2. **Local UNet — fine damage** — all of:
           - ``total_coverage_pct`` < 5 %
           - ``num_components`` < ``num_components_complex``
           - bbox width < 15 % of image width

        3. **Local UNet — small-medium localized** — all of:
           - ``total_coverage_pct`` < 15 %
           - ``largest_component_pct`` < 10 %

        4. **Hybrid (local first, cloud standby)** — everything else:
           - ``total_coverage_pct`` between 15 % and ``local_max_coverage_pct``
           - OR ``largest_component_pct`` between 10 % and ``local_max_component_pct``

        Parameters
        ----------
        mask_analysis : dict
            Output of :meth:`analyze_mask`.
        image_shape : tuple
            ``(height, width)`` or ``(height, width, channels)`` of the
            original image.  Only the first two elements are used.

        Returns
        -------
        dict with keys:
            engine : str
                One of ``"local_unet"``, ``"cloud_ai"``,
                ``"local_first_then_cloud"``.
            reason : str
                A brief, plain-language explanation aimed at a conservator.
            confidence : str
                One of ``"high"``, ``"medium"``, ``"low"``.  Reflects how
                clearly the damage profile fits the routing criteria—*not* a
                measure of model accuracy.
            local_suitable : bool
                ``True`` when the local UNet is expected to handle the damage.
            cloud_recommended : bool
                ``True`` when Cloud AI is the primary or fallback engine.
        """
        img_h: int = image_shape[0]
        img_w: int = image_shape[1]

        total_cov: float = mask_analysis["total_coverage_pct"]
        largest_cov: float = mask_analysis["largest_component_pct"]
        bbox_w: int = mask_analysis["largest_component_width_px"]
        num_comp: int = mask_analysis["num_components"]

        bbox_w_ratio: float = (bbox_w / img_w) if img_w > 0 else 0.0
        damage_stroke_width = mask_analysis.get("estimated_damage_width_px", 0)

        # Thin elongated scratches/craquelure (stroke width < 25 px and low area)
        # must NOT be classified as large structural missing voids
        is_elongated_scratch = (total_cov < 5.0 and damage_stroke_width < 25)

        # --- Rule 1: Large void / structural damage → Cloud AI ---
        cloud_triggers = []
        if total_cov > self.local_max_coverage_pct:
            cloud_triggers.append(
                f"total damage ({total_cov:.1f}%) exceeds {self.local_max_coverage_pct:.0f}% threshold"
            )
        if largest_cov > self.local_max_component_pct:
            cloud_triggers.append(
                f"largest void ({largest_cov:.1f}%) exceeds {self.local_max_component_pct:.0f}% threshold"
            )
        if bbox_w_ratio > self.local_max_width_ratio and not is_elongated_scratch:
            cloud_triggers.append(
                f"void width ({bbox_w_ratio * 100:.1f}% of image) exceeds {self.local_max_width_ratio * 100:.0f}% threshold"
            )

        if cloud_triggers:
            return {
                "engine": "cloud_ai",
                "reason": (
                    "Large missing region detected — exceeds local model's "
                    "training domain. Cloud AI selected."
                ),
                "confidence": "high",
                "local_suitable": False,
                "cloud_recommended": True,
            }

        # --- Rule 2: Fine scratches / craquelure → Local UNet (high confidence) ---
        fine_width_threshold = 0.15  # 15% of image width
        if (
            total_cov < 5.0
            and num_comp < self.num_components_complex
            and (bbox_w_ratio < fine_width_threshold or is_elongated_scratch)
        ):
            return {
                "engine": "local_unet",
                "reason": (
                    "Fine scratches detected — local ResNet34-UNet operates "
                    "within trained domain."
                ),
                "confidence": "high",
                "local_suitable": True,
                "cloud_recommended": False,
            }

        # --- Rule 3: Small-medium localized damage → Local UNet (medium confidence) ---
        if total_cov < 15.0 and largest_cov < 10.0:
            # Raise confidence to "medium" if there is high component complexity
            conf = "medium" if num_comp >= self.num_components_complex else "high"
            return {
                "engine": "local_unet",
                "reason": (
                    "Localized surface damage — local model handles this well."
                ),
                "confidence": conf,
                "local_suitable": True,
                "cloud_recommended": False,
            }

        # --- Rule 4: Medium damage → Hybrid ---
        return {
            "engine": "local_first_then_cloud",
            "reason": (
                "Moderate damage — local model attempted; "
                "cloud AI on standby if quality check fails."
            ),
            "confidence": "medium",
            "local_suitable": True,
            "cloud_recommended": True,
        }

    def format_routing_status(self, routing_decision: dict) -> str:
        """
        Return a human-readable status badge suitable for display in a UI.

        Parameters
        ----------
        routing_decision : dict
            Output of :meth:`route`.

        Returns
        -------
        str
            A single-line string with an emoji prefix, the engine label, and
            the reason from the routing decision.

        Examples
        --------
        >>> router = RestorationRouter()
        >>> mask = np.zeros((512, 512), dtype=np.uint8)
        >>> mask[100:120, 50:200] = 255        # small scratch
        >>> analysis = router.analyze_mask(mask)
        >>> decision = router.route(analysis, image_shape=(512, 512))
        >>> print(router.format_routing_status(decision))
        🏠 Local UNet Selected — Fine scratches detected, operates within trained domain.
        """
        engine: str = routing_decision.get("engine", "")
        reason: str = routing_decision.get("reason", "")

        if engine == "local_unet":
            emoji = "🏠"
            label = "Local UNet Selected"
        elif engine == "cloud_ai":
            emoji = "☁️"
            label = "Cloud AI Selected"
        elif engine == "local_first_then_cloud":
            emoji = "🔀"
            label = "Hybrid Mode"
        else:
            emoji = "❓"
            label = "Unknown Engine"

        return f"{emoji} {label} — {reason}"
