# scripts/run_controlled_damage_experiment.py
"""
Controlled Damage Scale Experiment:
Evaluates ResNet34-UNet restoration across three damage scales:
  1. Small Damage (thin scratches/craquelure, ~2.5% area)
  2. Medium Damage (moderate flaking patch, ~11% area)
  3. Large Damage (massive missing block, ~55% area)
on a 256x256 facial crop.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure src is in python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir / "app"))

from inference import restore_artwork_pipeline
from config import BASELINE_CHECKPOINT_PATH, EXPERIMENTAL_CHECKPOINT_PATH

out_diag = root_dir / "outputs" / "diagnostics"
out_diag.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(root_dir / "outputs" / "manual_tests" / "original.png"))
# 256x256 crop covering face and tear boundary
crop_y0, crop_y1 = 75, 331
crop_x0, crop_x1 = 120, 376
crop_img = orig[crop_y0:crop_y1, crop_x0:crop_x1].copy()
ch, cw = crop_img.shape[:2]

# 1. Small Mask: Thin scratches
mask_small = np.zeros((ch, cw), dtype=np.uint8)
cv2.line(mask_small, (30, 40), (140, 100), 255, 3)
cv2.line(mask_small, (50, 120), (180, 160), 255, 4)
cv2.circle(mask_small, (90, 80), 8, 255, -1)

# 2. Medium Mask: Moderate flaking patch
mask_med = np.zeros((ch, cw), dtype=np.uint8)
cv2.circle(mask_med, (110, 100), 28, 255, -1)
cv2.ellipse(mask_med, (70, 150), (20, 35), 30, 0, 360, 255, -1)

# 3. Large Mask: Massive missing block (right half)
mask_large = np.zeros((ch, cw), dtype=np.uint8)
mask_large[:, 115:] = 255

cov_small = float(np.mean(mask_small == 255) * 100)
cov_med = float(np.mean(mask_med == 255) * 100)
cov_large = float(np.mean(mask_large == 255) * 100)

print(f"Mask coverages on 256x256 crop:")
print(f"  Small:  {cov_small:.2f}% ({np.sum(mask_small == 255)} px)")
print(f"  Medium: {cov_med:.2f}% ({np.sum(mask_med == 255)} px)")
print(f"  Large:  {cov_large:.2f}% ({np.sum(mask_large == 255)} px)")

results = {}
for name, m in [("small", mask_small), ("med", mask_med), ("large", mask_large)]:
    for model_name, ckpt in [("base", BASELINE_CHECKPOINT_PATH), ("exp", EXPERIMENTAL_CHECKPOINT_PATH)]:
        res = restore_artwork_pipeline(
            damaged_image=crop_img,
            damage_mask=m,
            ckpt_path=str(ckpt),
            blend_mode="mask_guided",
            feather_mode="feathered",
            feather_radius=3,
            max_dimension=None,
            return_diagnostics=True
        )
        results[f"{name}_{model_name}"] = cv2.cvtColor(res["restored_image"], cv2.COLOR_RGB2BGR)
        diag = res["diagnostics"]
        print(f"[{name.upper()}] {model_name.upper()} Model: latency={diag.get('latency_ms')} ms, bit_exact={diag.get('is_bit_exact')}")

# Save individual crops
cv2.imwrite(str(out_diag / "crop_orig.png"), crop_img)
cv2.imwrite(str(out_diag / "crop_mask_small.png"), mask_small)
cv2.imwrite(str(out_diag / "crop_mask_med.png"), mask_med)
cv2.imwrite(str(out_diag / "crop_mask_large.png"), mask_large)

for k, img_res in results.items():
    cv2.imwrite(str(out_diag / f"crop_res_{k}.png"), img_res)

# Generate comprehensive comparison panel: outputs/diagnostics/model_comparison.png
# Layout: 4 columns x 4 rows
# Columns: [Damage Mask] | [Damaged Input Crop] | [Baseline Restoration] | [Experimental Restoration]
# Rows:
# Row 0: Original Reference / Context
# Row 1: Small Damage (Scratch)
# Row 2: Medium Damage (Flaking Patch)
# Row 3: Large Damage (Missing Half-Face)

banner_h = 36
panel_size = 256

def make_cell(img_bgr, label, is_mask=False):
    cell = np.full((panel_size + banner_h, panel_size, 3), 30, dtype=np.uint8)
    cv2.putText(cell, label, (8, 24), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(cell, (0, banner_h - 1), (panel_size, banner_h - 1), (60, 60, 60), 1)
    if is_mask:
        cell[banner_h:, :] = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR) if img_bgr.ndim == 2 else img_bgr
    else:
        cell[banner_h:, :] = img_bgr
    return cell

# Small Row
r1_c1 = make_cell(mask_small, f"Small Mask ({cov_small:.1f}%)", is_mask=True)
# Simulated damaged input
sim_dmg_small = crop_img.copy()
sim_dmg_small[mask_small == 255] = [200, 200, 200]
r1_c2 = make_cell(sim_dmg_small, "Damaged Input (Scratch)")
r1_c3 = make_cell(results["small_base"], "Baseline UNet (Sharp)")
r1_c4 = make_cell(results["small_exp"], "Experimental (Sharp)")
row1 = np.hstack([r1_c1, r1_c2, r1_c3, r1_c4])

# Medium Row
r2_c1 = make_cell(mask_med, f"Medium Mask ({cov_med:.1f}%)", is_mask=True)
sim_dmg_med = crop_img.copy()
sim_dmg_med[mask_med == 255] = [200, 200, 200]
r2_c2 = make_cell(sim_dmg_med, "Damaged Input (Flaking)")
r2_c3 = make_cell(results["med_base"], "Baseline UNet (Soft)")
r2_c4 = make_cell(results["med_exp"], "Experimental (Textured)")
row2 = np.hstack([r2_c1, r2_c2, r2_c3, r2_c4])

# Large Row
r3_c1 = make_cell(mask_large, f"Large Mask ({cov_large:.1f}%)", is_mask=True)
sim_dmg_large = crop_img.copy()
sim_dmg_large[mask_large == 255] = [190, 190, 190]
r3_c2 = make_cell(sim_dmg_large, "Damaged Input (Missing Half)")
r3_c3 = make_cell(results["large_base"], "Baseline UNet (BLURRY)")
r3_c4 = make_cell(results["large_exp"], "Experimental (BLURRY)")
row3 = np.hstack([r3_c1, r3_c2, r3_c3, r3_c4])

comp_grid = np.vstack([row1, row2, row3])

# Header Title Bar
title_h = 50
title_bar = np.full((title_h, comp_grid.shape[1], 3), 16, dtype=np.uint8)
cv2.putText(title_bar, "MUESUM — Controlled Damage Scale Breakdown Experiment (Small vs Med vs Large)", (16, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)

final_comp = np.vstack([title_bar, comp_grid])
cv2.imwrite(str(out_diag / "model_comparison.png"), final_comp)
print(f"Saved outputs/diagnostics/model_comparison.png ({final_comp.shape[1]}x{final_comp.shape[0]})")
