"""
scripts/execute_single_real_cloud_request.py
============================================
Makes EXACTLY ONE real request to Cloudflare Workers AI for
@cf/runwayml/stable-diffusion-v1-5-inpainting using Test E (large missing region).

Strict rules:
- Exactly ONE request made (no retries).
- Checks credentials without exposing tokens.
- Inspects HTTP status, headers, and body.
- If successful: saves raw image, composites with boundary feathering, verifies MAE=0.
- If unsuccessful: stops and reports exact error code, message, and likely cause.
"""

import os
import sys
import time
import base64
import io
import json
from pathlib import Path

import requests
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
    load_dotenv()
except ImportError:
    pass

from feathering import blend_with_boundary_feathering


def run_single_cloud_request():
    print("=" * 80)
    print("MUESUM — SINGLE REAL CLOUDFLARE WORKERS AI INPAINTING REQUEST")
    print("=" * 80)

    # 1. Inspect Credentials
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()

    print(f"1. CLOUDFLARE_ACCOUNT_ID configured: {bool(account_id)}")
    if account_id:
        print(f"   Account ID character length: {len(account_id)}")
        print(f"   Account ID format: {'32-char hex' if len(account_id) == 32 else f'{len(account_id)}-char string'}")
        is_placeholder = account_id.lower() in ("test_acc", "live_probe_account", "{account_id}", "<account_id>")
        print(f"   Is placeholder: {is_placeholder}")
    else:
        print("   ❌ CLOUDFLARE_ACCOUNT_ID is NOT set in environment or .env file.")

    print(f"2. CLOUDFLARE_API_TOKEN configured: {bool(api_token)}")
    if api_token:
        print(f"   API Token character length: {len(api_token)} (Token characters concealed)")
        print(f"   Whitespace detected in token: {'Yes (Stripped)' if os.environ.get('CLOUDFLARE_API_TOKEN', '') != api_token else 'No'}")
    else:
        print("   ❌ CLOUDFLARE_API_TOKEN is NOT set in environment or .env file.")

    if not account_id or not api_token:
        print("\n[STOPPED] Real API call cannot proceed because valid Cloudflare credentials")
        print("are not configured in the environment or in a .env file.")
        print("Likely cause: The user has not yet set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN.")
        return {
            "attempted": False,
            "success": False,
            "error": "Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN environment variable.",
            "status_code": None,
        }

    # 2. Build Request to Exact Endpoint
    endpoint = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/runwayml/stable-diffusion-v1-5-inpainting"
    print(f"\n3. Target Endpoint: https://api.cloudflare.com/client/v4/accounts/{account_id[:6]}...{account_id[-4:]}/ai/run/@cf/runwayml/stable-diffusion-v1-5-inpainting")

    # 3. Prepare Image & Mask
    img_path = _ROOT / "data" / "demo_samples" / "test_E_large_missing_region.png"
    mask_path = _ROOT / "data" / "demo_samples" / "test_E_large_missing_region_mask.png"

    orig_pil = Image.open(img_path).convert("RGB")
    mask_pil = Image.open(mask_path).convert("L")
    orig_w, orig_h = orig_pil.size

    # Resize to 512x512 for SD 1.5 inpainting
    img_512 = orig_pil.resize((512, 512), Image.LANCZOS)
    mask_512 = mask_pil.resize((512, 512), Image.NEAREST)

    # Encode Image to PNG bytes -> base64
    buf_img = io.BytesIO()
    img_512.save(buf_img, format="PNG")
    img_b64 = base64.b64encode(buf_img.getvalue()).decode("ascii")

    # Encode Mask to PNG bytes -> integer array (Cloudflare official format)
    buf_mask = io.BytesIO()
    mask_512.save(buf_mask, format="PNG")
    mask_bytes = buf_mask.getvalue()
    mask_int_array = [int(b) for b in mask_bytes]

    payload = {
        "prompt": "restore museum artwork, seamless classical oil painting inpainting",
        "image_b64": img_b64,
        "mask": mask_int_array,
        "num_steps": 20,
        "strength": 0.99,
        "guidance": 7.5,
    }

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    # 4. Issue Exactly ONE HTTP Request
    print("\n4. Issuing EXACTLY ONE live HTTP request to Cloudflare Workers AI...")
    t_start = time.perf_counter()
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        latency_ms = (time.perf_counter() - t_start) * 1000.0
    except Exception as exc:
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        print(f"\n[NETWORK EXCEPTION] {exc}")
        return {
            "attempted": True,
            "success": False,
            "error": f"Network exception: {exc}",
            "status_code": None,
            "latency_ms": latency_ms,
        }

    # 5. Inspect HTTP Response
    status = response.status_code
    content_type = response.headers.get("content-type", "")
    print(f"\n5. Live API Response Received in {latency_ms:.1f} ms:")
    print(f"   HTTP Status Code : {status}")
    print(f"   Content-Type     : {content_type}")

    if status != 200:
        print(f"   Response Body    : {response.text[:300]}")
        print("\n[STOPPED] API request returned non-200 status.")
        try:
            err_json = response.json()
            errors = err_json.get("errors", [])
        except Exception:
            errors = response.text[:200]
        return {
            "attempted": True,
            "success": False,
            "status_code": status,
            "error": errors,
            "latency_ms": latency_ms,
        }

    # 6. Decode Image If Successful
    print("\n6. HTTP 200 OK received! Decoding inpainting output...")
    is_binary_image = (
        isinstance(response.content, bytes)
        and len(response.content) > 0
        and (response.content.startswith(b"\x89PNG") or response.content.startswith(b"\xff\xd8") or "image" in content_type.lower())
    )

    if is_binary_image:
        cloud_raw_512 = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        resp_json = response.json()
        b64 = resp_json["result"]["image"]
        cloud_raw_512 = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    print(f"   Successfully decoded 512x512 image from Cloudflare response.")

    # 7. Upscale to native resolution & Save Raw Generated Image
    cloud_raw = cloud_raw_512.resize((orig_w, orig_h), Image.BICUBIC)
    out_dir = _ROOT / "outputs" / "live_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "cloud_raw_genuine.png"
    cloud_raw.save(raw_path, format="PNG")
    print(f"   Raw generated image saved -> {raw_path}")

    # 8. Composite with Original Artwork using Boundary Feathering
    orig_np = np.array(orig_pil, dtype=np.uint8)
    mask_np = (np.array(mask_pil) > 127).astype(np.float32)
    cloud_np = np.array(cloud_raw, dtype=np.uint8)

    blended, metrics = blend_with_boundary_feathering(
        orig_rgb=orig_np,
        restored_rgb=cloud_np,
        mask_binary=mask_np,
        feather_mode="feathered",
        feather_radius=3
    )

    # 9. Verify Preservation Guarantee Outside Mask
    dmg_bool = (mask_np > 0.5)
    outside_diff = np.abs(blended[~dmg_bool].astype(int) - orig_np[~dmg_bool].astype(int))
    max_diff = int(np.max(outside_diff))
    mae = float(np.mean(outside_diff))

    print(f"7. Preservation Verification on Genuine Cloud Result:")
    print(f"   Outside-mask MAE     : {mae:.6f} (Must be exactly 0.000000)")
    print(f"   Outside-mask MaxDiff : {max_diff} (Must be exactly 0)")

    assert max_diff == 0, "MaxDiff must be 0"
    assert mae == 0.0, "MAE must be 0.0"

    composite_path = out_dir / "cloud_composite_genuine.png"
    Image.fromarray(blended).save(composite_path, format="PNG")
    print(f"   Composite saved -> {composite_path}")

    return {
        "attempted": True,
        "success": True,
        "status_code": 200,
        "latency_ms": latency_ms,
        "raw_path": str(raw_path),
        "composite_path": str(composite_path),
        "mae": mae,
        "max_diff": max_diff,
    }


if __name__ == "__main__":
    result = run_single_cloud_request()
    print("\nExecution Result Summary:")
    print(json.dumps(result, indent=2))

