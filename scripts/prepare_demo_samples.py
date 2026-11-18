# scripts/prepare_demo_samples.py
"""
MUESUM — Demonstration Samples Generator & Validator
===================================================
Prepares and validates five standardized artwork test pairs covering:
  Test A: Ordinary JPG artwork (Standard museum painting).
  Test B: PNG artwork (Historical portrait).
  Test C: Artwork with fine scratches (Hairline physical scratches).
  Test D: Artwork with localized missing paint (Flaking lacquer / lacuna patch).
  Test E: Artwork with large missing regions (Severe plaster delamination / canvas tear).
"""

import sys
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

root_dir = Path(__file__).resolve().parent.parent
samples_dir = root_dir / "data" / "demo_samples"
samples_dir.mkdir(parents=True, exist_ok=True)

# ── Sample A: Ordinary JPG Artwork ───────────────────────────────────────────
src_a_img = root_dir / "release" / "sample_artwork.jpg"
src_a_mask = root_dir / "release" / "sample_mask.png"
dst_a_img = samples_dir / "test_A_ordinary_artwork.jpg"
dst_a_mask = samples_dir / "test_A_ordinary_artwork_mask.png"

shutil.copy2(src_a_img, dst_a_img)
shutil.copy2(src_a_mask, dst_a_mask)
print(f"[Sample A] Copied: {dst_a_img.name} (JPG) + {dst_a_mask.name}")

# ── Sample B: PNG Artwork ─────────────────────────────────────────────────────
# Convert a high-quality clean artwork into PNG with synthetic surface stain/scratch
base_clean_jpg = root_dir / "data" / "clean" / "10344.jpg"
if base_clean_jpg.exists():
    img_b = cv2.imread(str(base_clean_jpg))
    # Resize to standard 512x512 crop
    h, w = img_b.shape[:2]
    crop_b = cv2.resize(img_b, (512, 512), interpolation=cv2.INTER_AREA)
    mask_b = np.zeros((512, 512), dtype=np.uint8)
    # Add localized blemishes: scratch + stain patch
    cv2.line(mask_b, (100, 80), (380, 220), 255, 4)
    cv2.circle(mask_b, (260, 320), 30, 255, -1)
    
    # Degrade image at mask
    damaged_b = crop_b.copy()
    damaged_b[mask_b == 255] = np.clip(damaged_b[mask_b == 255] * 0.4 + 40, 0, 255).astype(np.uint8)
    
    dst_b_img = samples_dir / "test_B_renaissance_artwork.png"
    dst_b_mask = samples_dir / "test_B_renaissance_artwork_mask.png"
    cv2.imwrite(str(dst_b_img), damaged_b)
    cv2.imwrite(str(dst_b_mask), mask_b)
    print(f"[Sample B] Generated: {dst_b_img.name} (PNG) + {dst_b_mask.name}")
else:
    # Fallback to craquelure portrait
    craq_src = root_dir / "data" / "manual_tests" / "craquelure_portrait.png"
    craq_mask_src = root_dir / "outputs" / "craquelure_diagnostics" / "mask.png"
    dst_b_img = samples_dir / "test_B_renaissance_artwork.png"
    dst_b_mask = samples_dir / "test_B_renaissance_artwork_mask.png"
    shutil.copy2(craq_src, dst_b_img)
    shutil.copy2(craq_mask_src, dst_b_mask)
    print(f"[Sample B] Copied from craquelure portrait: {dst_b_img.name}")

# ── Sample C: Fine Scratches ──────────────────────────────────────────────────
# Clean oil painting crop with fine scratches (1-3 px width)
clean_ref = cv2.imread(str(root_dir / "outputs" / "manual_tests" / "original.png"))
if clean_ref is None:
    clean_ref = cv2.imread(str(dst_a_img))

ch, cw = 400, 400
crop_c = cv2.resize(clean_ref, (cw, ch), interpolation=cv2.INTER_AREA)
mask_c = np.zeros((ch, cw), dtype=np.uint8)

# Multi-directional fine scratches
cv2.line(mask_c, (50, 60), (320, 180), 255, 2)
cv2.line(mask_c, (90, 200), (280, 290), 255, 3)
cv2.line(mask_c, (220, 50), (140, 340), 255, 2)
cv2.line(mask_c, (300, 150), (370, 310), 255, 2)

damaged_c = crop_c.copy()
# Scratch appears as bright gouges / dark fissures
damaged_c[mask_c == 255] = np.clip(damaged_c[mask_c == 255] * 0.2 + 20, 0, 255).astype(np.uint8)

dst_c_img = samples_dir / "test_C_fine_scratches.png"
dst_c_mask = samples_dir / "test_C_fine_scratches_mask.png"
cv2.imwrite(str(dst_c_img), damaged_c)
cv2.imwrite(str(dst_c_mask), mask_c)
print(f"[Sample C] Generated: {dst_c_img.name} (Fine scratches, width 2-3px)")

# ── Sample D: Localized Missing Paint ─────────────────────────────────────────
# Artwork with medium flaking patches (lacunae where paint has chipped away)
crop_d = cv2.resize(clean_ref, (cw, ch), interpolation=cv2.INTER_AREA)
mask_d = np.zeros((ch, cw), dtype=np.uint8)

# Flaking irregular patches
cv2.ellipse(mask_d, (180, 160), (45, 28), 25, 0, 360, 255, -1)
cv2.circle(mask_d, (290, 260), 32, 255, -1)
cv2.ellipse(mask_d, (100, 280), (25, 40), -15, 0, 360, 255, -1)

damaged_d = crop_d.copy()
# Exposed white/gray canvas substrate under missing paint
damaged_d[mask_d == 255] = [215, 210, 200]  # Chalky gesso / primer tone

dst_d_img = samples_dir / "test_D_localized_paint_loss.png"
dst_d_mask = samples_dir / "test_D_localized_paint_loss_mask.png"
cv2.imwrite(str(dst_d_img), damaged_d)
cv2.imwrite(str(dst_d_mask), mask_d)
print(f"[Sample D] Generated: {dst_d_img.name} (Localized missing paint lacunae)")

# ── Sample E: Large Missing Regions ───────────────────────────────────────────
# Demonstrates large-scale missing regions (>50% area) and documents neural boundary limits
damaged_e_src = root_dir / "data" / "manual_tests" / "damaged_portrait.png"
mask_e_src = root_dir / "outputs" / "manual_tests" / "damage_mask.png"
dst_e_img = samples_dir / "test_E_large_missing_region.png"
dst_e_mask = samples_dir / "test_E_large_missing_region_mask.png"

if damaged_e_src.exists() and mask_e_src.exists():
    shutil.copy2(damaged_e_src, dst_e_img)
    shutil.copy2(mask_e_src, dst_e_mask)
else:
    crop_e = cv2.resize(clean_ref, (cw, ch), interpolation=cv2.INTER_AREA)
    mask_e = np.zeros((ch, cw), dtype=np.uint8)
    mask_e[:, 180:] = 255  # 55% area missing
    damaged_e = crop_e.copy()
    damaged_e[mask_e == 255] = [150, 150, 150]
    cv2.imwrite(str(dst_e_img), damaged_e)
    cv2.imwrite(str(dst_e_mask), mask_e)

print(f"[Sample E] Copied/Generated: {dst_e_img.name} (Severe large-scale missing region)")
print("\nAll 5 demonstration test samples ready in data/demo_samples/!")

