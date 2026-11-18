"""
scripts/test_gemini_restoration.py
==================================
Test Google's Gemini API (gemini-2.5-flash-image) for historical artwork / photograph reconstruction.
Adheres strictly to project constraints:
  - 1 live API request to conserve credits.
  - Zero credential leakage.
  - Generates raw Gemini output, bit-exact composite, and 4-panel comparison panel.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feathering import blend_with_boundary_feathering

# 1. Load Environment securely
env_path = ROOT_DIR / ".env"
load_dotenv(env_path)
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("[ERROR] GEMINI_API_KEY is not defined in .env file.")
    sys.exit(1)

print(f"[SECURITY] GEMINI_API_KEY detected securely (length: {len(api_key)} chars). Credentials masked.")

# 2. Setup Google GenAI Client
from google import genai
from google.genai import types

client = genai.Client(api_key=api_key)

# 3. Load Damaged Image & Mask
img_path = ROOT_DIR / "data" / "demo_samples" / "test_E_large_missing_region.png"
mask_path = ROOT_DIR / "data" / "demo_samples" / "test_E_large_missing_region_mask.png"

if not img_path.exists() or not mask_path.exists():
    print(f"[ERROR] Required test files missing: {img_path} or {mask_path}")
    sys.exit(1)

orig_pil = Image.open(img_path).convert("RGB")
orig_w, orig_h = orig_pil.size
orig_np = np.array(orig_pil, dtype=np.uint8)

mask_pil = Image.open(mask_path).convert("L")
mask_np = np.array(mask_pil, dtype=np.uint8)
mask_binary = (mask_np > 127).astype(np.float32)
dmg_bool = (mask_binary > 0.5)

coverage_pct = float(np.mean(dmg_bool) * 100.0)
print(f"[INPUT] Image dimensions: {orig_w}x{orig_h} px")
print(f"[INPUT] Damage mask coverage: {coverage_pct:.2f}%")

# Read raw bytes for API transmission
with open(img_path, "rb") as f:
    img_bytes = f.read()

with open(mask_path, "rb") as f:
    mask_bytes = f.read()

# 4. Prepare Gemini Request
# Using gemini-2.5-flash-image
model_id = "gemini-2.5-flash-image"

restoration_prompt = (
    "Image 1 is a damaged historical photograph with physical damage (a large vertical tear, missing sections, and scratches across the faces and clothing). "
    "Image 2 is the exact binary damage mask where white (255) pixels indicate the damaged areas to reconstruct, and black (0) pixels indicate surviving original artwork.\n\n"
    "Restore this damaged historical photograph. Reconstruct missing and damaged areas using the surrounding visual context. "
    "Preserve the original people's facial identities, expressions, clothing, composition, lighting, and historical photographic appearance. "
    "Repair missing regions, large tears, scratches, and damaged faces as naturally as possible. "
    "Do not introduce unnecessary objects or change undamaged content. Produce a high quality, natural, realistic historical photograph restoration."
)

parts = [
    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
    types.Part.from_bytes(data=mask_bytes, mime_type="image/png"),
    types.Part.from_text(text=restoration_prompt)
]

config = types.GenerateContentConfig(
    response_modalities=["IMAGE", "TEXT"],
    temperature=0.3
)

output_dir = ROOT_DIR / "outputs" / "gemini_restoration"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n[API CALL] Dispatching single image restoration request to Google Gemini ({model_id})...")
t_start = time.perf_counter()

gemini_raw_pil = None
model_text_response = ""

try:
    response = client.models.generate_content(
        model=model_id,
        contents=parts,
        config=config
    )
    t_elapsed = time.perf_counter() - t_start
    print(f"[API CALL] Response received in {t_elapsed:.2f}s!")

    # Inspect response candidates
    if not response.candidates:
        print("[ERROR] No candidates returned by Gemini API.")
        if response.text:
            print("[INFO] Text in response:", response.text)
        sys.exit(1)

    candidate = response.candidates[0]
    for idx, part in enumerate(candidate.content.parts):
        # Check for image
        img = part.as_image()
        if img is not None:
            gemini_raw_pil = img
            print(f"[SUCCESS] Extracted generated image part (part {idx}), format={img.format}, size={img.size}, mode={img.mode}")
        elif part.text:
            model_text_response += part.text + "\n"

    if model_text_response.strip():
        print(f"[GEMINI TEXT OUTPUT]:\n{model_text_response.strip()}\n")

    if gemini_raw_pil is None:
        # Check inline data directly
        for idx, part in enumerate(candidate.content.parts):
            if hasattr(part, "inline_data") and part.inline_data:
                import io
                gemini_raw_pil = Image.open(io.BytesIO(part.inline_data.data))
                print(f"[SUCCESS] Extracted image from inline_data bytes! Size: {gemini_raw_pil.size}")
                break

    if gemini_raw_pil is None:
        print("[ERROR] Gemini did not return an image part. Text returned:")
        print(model_text_response)
        sys.exit(1)

except Exception as e:
    t_elapsed = time.perf_counter() - t_start
    print(f"[ERROR] Gemini API call failed after {t_elapsed:.2f}s: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Save Raw Gemini Output
raw_out_path = output_dir / "gemini_raw_output.png"
gemini_raw_pil.save(raw_out_path, format="PNG")
print(f"[SAVED] Raw Gemini output saved to: {raw_out_path}")

# 6. Composite with Substrate Preservation Guarantee
# Resize raw output to native resolution (orig_w, orig_h) if dimensions differ
raw_resized = gemini_raw_pil.convert("RGB").resize((orig_w, orig_h), Image.LANCZOS)
raw_np = np.array(raw_resized, dtype=np.uint8)

# Boundary feathering
composite_np, blend_metrics = blend_with_boundary_feathering(
    orig_rgb=orig_np,
    restored_rgb=raw_np,
    mask_binary=mask_binary,
    feather_mode="feathered",
    feather_radius=3
)

# Enforce strict bit-exact preservation on untouched pixels
composite_np[~dmg_bool] = orig_np[~dmg_bool]

# Verification metrics
diff = np.abs(composite_np[~dmg_bool].astype(np.int32) - orig_np[~dmg_bool].astype(np.int32))
undamaged_mae = float(np.mean(diff))
undamaged_max_diff = int(np.max(diff))
is_bit_exact = (undamaged_max_diff == 0)

print(f"\n[VERIFICATION] Undamaged Canvas Preservation:")
print(f"  MAE: {undamaged_mae:.6f} (target: 0.000000)")
print(f"  MaxDiff: {undamaged_max_diff} (target: 0)")
print(f"  Bit-Exact Verified: {is_bit_exact}")

composite_pil = Image.fromarray(composite_np)
composite_out_path = output_dir / "gemini_composite_restored.png"
composite_pil.save(composite_out_path, format="PNG")
print(f"[SAVED] Final composite saved to: {composite_out_path}")

# 7. Generate 4-Panel Comparison Board
# [Original Damaged] | [Damage Mask] | [Raw Gemini Output] | [Final Restored Composite]
panel_w = orig_w
panel_h = orig_h
header_h = 50
total_w = panel_w * 4
total_h = panel_h + header_h

comparison_canvas = Image.new("RGB", (total_w, total_h), color=(18, 20, 26))
draw = ImageDraw.Draw(comparison_canvas)

# Color mask overlay (red on damaged regions)
mask_rgb = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
mask_rgb[dmg_bool] = [230, 40, 40]
mask_rgb[~dmg_bool] = [20, 20, 20]
mask_display_pil = Image.fromarray(mask_rgb)

panels = [
    ("1. Original Damaged Artwork", orig_pil),
    (f"2. Damage Mask ({coverage_pct:.1f}% Area)", mask_display_pil),
    (f"3. Raw Gemini ({model_id})", raw_resized),
    ("4. Final Composite (100% Bit-Exact)", composite_pil)
]

for idx, (title, p_img) in enumerate(panels):
    x_offset = idx * panel_w
    # Paste image
    comparison_canvas.paste(p_img, (x_offset, header_h))
    # Draw header background
    draw.rectangle([x_offset, 0, x_offset + panel_w - 1, header_h], fill=(28, 32, 42))
    # Draw outline separator
    draw.rectangle([x_offset, 0, x_offset + panel_w - 1, total_h - 1], outline=(60, 68, 88), width=1)
    # Header text
    draw.text((x_offset + 15, 16), title, fill=(240, 240, 245))

four_panel_path = output_dir / "gemini_4_panel_comparison.png"
comparison_canvas.save(four_panel_path, format="PNG")
print(f"[SAVED] 4-Panel comparison saved to: {four_panel_path}")

# 8. Curatorial & Metric Summary
print("\n" + "=" * 60)
print("GEMINI RESTORATION EXPERIMENT SUMMARY")
print("=" * 60)
print(f"Model Tested:            {model_id}")
print(f"API Latency:             {t_elapsed:.2f} seconds")
print(f"Original Resolution:     {orig_w} x {orig_h} px")
print(f"Gemini Raw Resolution:   {gemini_raw_pil.size[0]} x {gemini_raw_pil.size[1]} px")
print(f"Damage Region Coverage:  {coverage_pct:.2f}%")
print(f"Bit-Exact Preservation:  {'PASS (MAE = 0.000000)' if is_bit_exact else 'FAIL'}")
print(f"Estimated Cost:          ~$0.03 (Google GenAI image generation tier)")
print("=" * 60)

