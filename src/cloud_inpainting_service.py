"""
cloud_inpainting_service.py
===========================
Cloudflare Workers AI inpainting service integration for MUESUM.

This module provides a drop-in cloud fallback for artwork restoration when the
local ResNet34-UNet is insufficient (large voids, heavy damage, etc.).

Mask polarity note
------------------
MUESUM uses the convention: **white (255) = damaged area** to be restored.
Stable Diffusion 1.5 inpainting (via Cloudflare Workers AI) uses the same
polarity: **white (255) = region to inpaint**.
Therefore, NO mask inversion is needed when passing MUESUM masks to this API.

Credentials
-----------
All credentials are read exclusively from environment variables:
  - CLOUDFLARE_ACCOUNT_ID  : Your Cloudflare account identifier.
  - CLOUDFLARE_API_TOKEN   : A scoped API token with Workers AI run permission.

Tokens are never logged, printed, or included in exception messages.
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
    else:
        load_dotenv()
except ImportError:
    pass

import numpy as np
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_BASE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"
_MODEL_PATH = "@cf/runwayml/stable-diffusion-v1-5-inpainting"
_TARGET_SIZE = (512, 512)   # SD 1.5 native resolution
_TIMEOUT_S = 60             # Maximum seconds to wait for the Cloudflare API
_ENGINE_NAME = "cloudflare_sd15_inpainting"

_DEFAULT_PROMPT = (
    "restore damaged museum artwork, oil painting restoration, preserve original style"
)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_service_instance: Optional["CloudInpaintingService"] = None


def get_cloud_service() -> "CloudInpaintingService":
    """Return the module-level singleton :class:`CloudInpaintingService`.

    The instance is created lazily on first call, then cached for the lifetime
    of the process.  Re-creating it on every call would re-read env vars each
    time, but the singleton pattern matches the existing MUESUM pattern for
    the local inference model.

    Returns
    -------
    CloudInpaintingService
        The shared service instance.  Call :meth:`is_available` before use.
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = CloudInpaintingService()
    return _service_instance


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class CloudInpaintingService:
    """Cloudflare Workers AI inpainting service integration.

    Model: ``@cf/runwayml/stable-diffusion-v1-5-inpainting``

    All credentials must come from environment variables:

    * ``CLOUDFLARE_ACCOUNT_ID``
    * ``CLOUDFLARE_API_TOKEN``

    Tokens are **never** logged or exposed.

    Mask polarity
    -------------
    MUESUM encodes damaged regions as **white (255)**; undamaged areas are
    **black (0)**.  Stable Diffusion inpainting uses the identical convention:
    white pixels indicate the area that should be regenerated.  Consequently
    this service passes the mask through unchanged — no inversion is applied.

    Usage
    -----
    ::

        svc = get_cloud_service()
        if svc.is_available():
            result = svc.restore(image_pil, mask_array)
            if result["success"]:
                restored = result["image"]
    """

    def __init__(self) -> None:
        """Initialise the service by reading credentials from environment variables.

        If either ``CLOUDFLARE_ACCOUNT_ID`` or ``CLOUDFLARE_API_TOKEN`` is
        absent or empty the service marks itself unavailable so callers can
        gracefully fall back to the local model.  No exception is raised.
        """
        # Read credentials — intentionally not stored with descriptive names
        # that could be mistakenly logged.
        self._account_id: str = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self._api_token: str = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()

        self.available: bool = bool(self._account_id and self._api_token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return ``True`` only if both credentials are present and non-empty.

        Dynamically checks environment variables to support runtime configuration.

        Returns
        -------
        bool
            ``True`` when the service is ready to accept restore requests.
        """
        self._account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self._api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
        self.available = bool(self._account_id and self._api_token)
        return self.available

    def restore(
        self,
        image_pil: Image.Image,
        mask_uint8: np.ndarray,
        prompt: str = _DEFAULT_PROMPT,
    ) -> dict:
        """Run cloud inpainting via Cloudflare Workers AI.

        The method submits the artwork image and its damage mask to the
        Cloudflare ``@cf/runwayml/stable-diffusion-v1-5-inpainting`` model and
        returns the inpainted result resized back to the original dimensions.

        Mask polarity
        ~~~~~~~~~~~~~
        ``mask_uint8`` is expected in MUESUM's native format: **255 = damaged,
        0 = intact**.  Stable Diffusion inpainting uses the same convention
        (white = inpaint), so the mask is passed through without inversion.

        Parameters
        ----------
        image_pil:
            The artwork image as a PIL ``Image`` (any mode; converted to RGB
            internally).
        mask_uint8:
            Grayscale damage mask as a ``numpy`` array with dtype ``uint8``
            and values 0 or 255.  Shape must be ``(H, W)`` or ``(H, W, 1)``.
        prompt:
            Text guidance for the inpainting model.  Defaults to a generic
            artwork restoration prompt.

        Returns
        -------
        dict
            A result dictionary with the following keys:

            ``success`` : bool
                ``True`` if inpainting completed without error.
            ``image`` : PIL.Image.Image or None
                The restored image at the original input resolution, or
                ``None`` on failure.
            ``error`` : str or None
                Human-readable error description on failure, ``None`` on
                success.
            ``engine`` : str
                Always ``"cloudflare_sd15_inpainting"``.
            ``latency_ms`` : float
                Wall-clock time for the API round-trip in milliseconds.
        """
        t_start = time.perf_counter()

        # Always populate the engine key so callers can identify the source
        base_result = {
            "success": False,
            "image": None,
            "error": None,
            "engine": _ENGINE_NAME,
            "latency_ms": 0.0,
        }

        if not self.is_available():
            base_result["error"] = (
                "CloudInpaintingService is not available: missing credentials."
            )
            return base_result

        try:
            # ----------------------------------------------------------
            # 1. Prepare image
            # ----------------------------------------------------------
            original_size = image_pil.size  # (width, height)
            image_rgb = image_pil.convert("RGB")
            image_512 = image_rgb.resize(_TARGET_SIZE, Image.LANCZOS)

            img_bytes = _pil_to_png_bytes(image_512)
            img_b64 = base64.b64encode(img_bytes).decode("ascii")

            # ----------------------------------------------------------
            # 2. Prepare mask
            # ----------------------------------------------------------
            # Squeeze to 2-D if needed
            mask_2d = np.squeeze(mask_uint8)
            if mask_2d.ndim != 2:
                raise ValueError(
                    f"mask_uint8 must be 2-D (H×W); got shape {mask_uint8.shape}"
                )

            mask_pil = Image.fromarray(mask_2d.astype(np.uint8), mode="L")
            mask_512 = mask_pil.resize(_TARGET_SIZE, Image.NEAREST)

            mask_bytes = _pil_to_png_bytes(mask_512)
            # Cloudflare Workers AI inpainting requires mask as an array of 8-bit integers
            mask_int_array = [int(b) for b in mask_bytes]

            # ----------------------------------------------------------
            # 3. Build request
            # ----------------------------------------------------------
            url = f"{_API_BASE.format(account_id=self._account_id)}/{_MODEL_PATH}"

            headers = {
                # Token is kept in the header only; never echoed elsewhere.
                "Authorization": f"Bearer {self._api_token}",
                "Content-Type": "application/json",
            }

            payload = {
                "prompt": prompt,
                "image_b64": img_b64,
                "mask": mask_int_array,
                "num_steps": 20,
                "strength": 0.99,
                "guidance": 7.5,
            }

            # ----------------------------------------------------------
            # 4. Call the API (single attempt, no retries)
            # ----------------------------------------------------------
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=_TIMEOUT_S,
            )

            latency_ms = (time.perf_counter() - t_start) * 1000.0

            if response.status_code != 200:
                # Avoid including the URL (which contains account_id) in the
                # error; return only the status code and sanitised body.
                try:
                    err_body = response.json()
                    err_detail = err_body.get("errors") or err_body.get("message", "")
                except Exception:
                    err_detail = response.text[:200]

                base_result["error"] = (
                    f"Cloudflare API returned HTTP {response.status_code}: {err_detail}"
                )
                base_result["latency_ms"] = latency_ms
                return base_result

            # ----------------------------------------------------------
            # 5. Decode result
            # ----------------------------------------------------------
            # Cloudflare Workers AI can return either binary image bytes (image/png)
            # or a JSON wrapper: {"result": {"image": "<base64>"}, "success": true}
            content_type = response.headers.get("content-type", "").lower()
            result_img_512 = None

            is_binary_image = (
                isinstance(response.content, bytes)
                and len(response.content) > 0
                and (response.content.startswith(b"\x89PNG") or response.content.startswith(b"\xff\xd8") or "image" in content_type)
            )

            if is_binary_image:
                try:
                    result_img_512 = Image.open(io.BytesIO(response.content)).convert("RGB")
                except Exception as exc:
                    base_result["error"] = f"Failed to decode binary image from Cloudflare response: {exc}"
                    base_result["latency_ms"] = latency_ms
                    return base_result
            else:
                try:
                    resp_json = response.json()
                except Exception as exc:
                    base_result["error"] = f"Failed to parse API response as JSON or Image (HTTP {response.status_code}): {response.text[:200]}"
                    base_result["latency_ms"] = latency_ms
                    return base_result

                if not resp_json.get("success", False):
                    errors = resp_json.get("errors", [])
                    base_result["error"] = f"Cloudflare API reported failure: {errors}"
                    base_result["latency_ms"] = latency_ms
                    return base_result

                result = resp_json.get("result", {})
                if isinstance(result, dict) and "image" in result:
                    result_b64 = result["image"]
                    result_bytes = base64.b64decode(result_b64)
                    result_img_512 = Image.open(io.BytesIO(result_bytes)).convert("RGB")
                else:
                    base_result["error"] = f"Unexpected JSON response structure: {list(resp_json.keys())}"
                    base_result["latency_ms"] = latency_ms
                    return base_result

            # ----------------------------------------------------------
            # 6. Resize back to original dimensions (BICUBIC ≈ INTER_CUBIC)
            # ----------------------------------------------------------
            result_img = result_img_512.resize(original_size, Image.BICUBIC)

            base_result.update(
                {
                    "success": True,
                    "image": result_img,
                    "error": None,
                    "latency_ms": latency_ms,
                }
            )
            return base_result

        except requests.exceptions.Timeout:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            base_result["error"] = (
                f"Cloudflare API request timed out after {_TIMEOUT_S}s."
            )
            base_result["latency_ms"] = latency_ms
            return base_result

        except requests.exceptions.RequestException as exc:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            base_result["error"] = f"Network error contacting Cloudflare API: {exc}"
            base_result["latency_ms"] = latency_ms
            return base_result

        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            base_result["error"] = f"Unexpected error in CloudInpaintingService: {exc}"
            base_result["latency_ms"] = latency_ms
            return base_result

    def get_status_message(self) -> str:
        """Return a user-visible status badge string.

        Returns
        -------
        str
            A single-line status suitable for display in the MUESUM UI.
        """
        if self.is_available():
            return (
                "☁️ Cloud AI: Cloudflare Workers AI (SD 1.5 Inpainting) — Ready"
            )
        return (
            "☁️ Cloud AI: Unavailable — Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pil_to_png_bytes(img: Image.Image) -> bytes:
    """Encode a PIL image to PNG bytes in memory.

    Parameters
    ----------
    img:
        PIL image to encode.

    Returns
    -------
    bytes
        Raw PNG-encoded bytes.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
