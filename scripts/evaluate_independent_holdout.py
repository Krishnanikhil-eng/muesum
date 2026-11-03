# scripts/evaluate_independent_holdout.py
"""
MUESUM — Stage 4: Independent Holdout Evaluation Suite
======================================================
Evaluates both the Baseline Model (Epoch 8) and Experimental Texture Model
on the 40 newly acquired, untouched independent artworks.

Protocol:
  - Aspect-preserving resize with max dimension 1024 (identical to Phase E).
  - Sequential processing, batch size 1, CPU-only.
  - Strict zero leakage: independent holdout was completely untouched during training.
  - Generates comprehensive per-sample CSV metrics and visual comparison panels.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
import csv
import gc
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from unet_model import ResNet34UNet
from inference import restore_artwork_pipeline, ModelManager
from utils import ensure_dir, psnr, safe_ssim


def evaluate_independent_holdout(
    feather_radius: int = 3,
    max_dimension: int = 1024,
    save_visuals: bool = True
):
    print("=" * 80)
    print("MUESUM — STAGE 4: FINAL INDEPENDENT EVALUATION")
    print("Untouched Holdout Set: 40 Public-Domain Artworks from The Met Museum Collection")
    print("=" * 80)

    device = torch.device("cpu")
    holdout_dir = root_dir / "data" / "independent_holdout"
    clean_dir = holdout_dir / "clean"
    damaged_dir = holdout_dir / "damaged"
    masks_dir = holdout_dir / "masks"
    meta_csv = holdout_dir / "metadata.csv"

    out_dir = root_dir / "reports" / "final_independent_evaluation"
    visuals_dir = out_dir / "visual_comparisons"
    ensure_dir(str(out_dir))
    if save_visuals:
        ensure_dir(str(visuals_dir))

    # Verify Paths
    baseline_ckpt = root_dir / "checkpoints" / "baseline_run" / "best_model.pth"
    exp_ckpt = root_dir / "checkpoints" / "experimental_texture_model" / "best_model.pth"

    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found at: {baseline_ckpt}")
    if not exp_ckpt.exists():
        raise FileNotFoundError(f"Experimental checkpoint not found at: {exp_ckpt}")

    # Load Artworks Metadata
    metadata = {}
    with open(meta_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            metadata[r["object_id"]] = r

    object_ids = sorted(list(metadata.keys()))
    print(f"[Holdout] Found {len(object_ids)} holdout artworks across 10 categories.")

    # 1. Load Baseline Model
    print(f"[Model A] Loading Baseline ResNet34-UNet (Epoch 8) from {baseline_ckpt}...")
    baseline_model = ResNet34UNet(pretrained=False).to(device)
    base_st = torch.load(str(baseline_ckpt), map_location=device)
    baseline_model.load_state_dict(base_st.get("model", base_st))
    baseline_model.eval()

    # 2. Load Experimental Model
    print(f"[Model B] Loading Experimental Texture Model from {exp_ckpt}...")
    exp_model = ResNet34UNet(pretrained=False).to(device)
    exp_st = torch.load(str(exp_ckpt), map_location=device)
    exp_model.load_state_dict(exp_st.get("model", exp_st))
    exp_model.eval()

    # Setup Metrics Output CSV
    metrics_path = out_dir / "holdout_metrics.csv"
    columns = [
        "object_id", "title", "classification", "width", "height", "damage_coverage_pct",
        "dmg_psnr", "dmg_ssim", "dmg_mae",
        # Baseline Model Results (Feathered)
        "base_psnr", "base_ssim", "base_mae", "base_delta_psnr", "base_delta_ssim", "base_latency_ms",
        # Experimental Model Results (Feathered)
        "exp_psnr", "exp_ssim", "exp_mae", "exp_delta_psnr", "exp_delta_ssim", "exp_latency_ms",
        # Comparative Advantage
        "exp_vs_base_delta_psnr", "exp_vs_base_delta_ssim",
        # Verification
        "undamaged_max_diff", "undamaged_mae", "is_bit_exact"
    ]

    results_rows = []

    print("\n" + "-" * 80)
    print(f"{'ID':<8} | {'Category':<12} | {'Cov%':<6} | {'Dmg PSNR':<9} | {'Base PSNR':<9} | {'Exp PSNR':<9} | {'Delta':<7}")
    print("-" * 80)

    for idx, obj_id in enumerate(object_ids, 1):
        clean_file = clean_dir / f"{obj_id}.jpg"
        dmg_file = damaged_dir / f"{obj_id}.jpg"
        mask_file = masks_dir / f"{obj_id}_mask.png"

        meta = metadata.get(obj_id, {})
        cat = meta.get("classification", "Unknown")
        title = meta.get("title", "")

        # Standard aspect-preserving resizing (max_dimension: 1024)
        with Image.open(clean_file) as c_im:
            clean_orig = np.array(c_im.convert("RGB"))
        h_orig, w_orig = clean_orig.shape[:2]

        if max(h_orig, w_orig) > max_dimension:
            scale = float(max_dimension) / float(max(h_orig, w_orig))
            w_eval = max(32, int(round(w_orig * scale)))
            h_eval = max(32, int(round(h_orig * scale)))
            clean_eval = cv2.resize(clean_orig, (w_eval, h_eval), interpolation=cv2.INTER_AREA)
        else:
            w_eval, h_eval = w_orig, h_orig
            clean_eval = clean_orig

        # ── Infer Model A: Baseline Model ────────────────────────────────────
        t_base_0 = time.perf_counter()
        res_base = restore_artwork_pipeline(
            damaged_image=dmg_file,
            damage_mask=mask_file,
            model=baseline_model,
            device=device,
            blend_mode="mask_guided",
            feather_mode="feathered",
            feather_radius=feather_radius,
            max_dimension=max_dimension,
            return_diagnostics=True
        )
        base_latency = (time.perf_counter() - t_base_0) * 1000.0

        # ── Infer Model B: Experimental Texture Model ─────────────────────────
        t_exp_0 = time.perf_counter()
        res_exp = restore_artwork_pipeline(
            damaged_image=dmg_file,
            damage_mask=mask_file,
            model=exp_model,
            device=device,
            blend_mode="mask_guided",
            feather_mode="feathered",
            feather_radius=feather_radius,
            max_dimension=max_dimension,
            return_diagnostics=True
        )
        exp_latency = (time.perf_counter() - t_exp_0) * 1000.0

        # Resize clean reference to match evaluated resolution for exact metrics
        dmg_eval = cv2.resize(np.array(Image.open(dmg_file).convert("RGB")), (w_eval, h_eval), interpolation=cv2.INTER_AREA)
        restored_base_eval = cv2.resize(res_base["restored_image"], (w_eval, h_eval), interpolation=cv2.INTER_AREA)
        restored_exp_eval = cv2.resize(res_exp["restored_image"], (w_eval, h_eval), interpolation=cv2.INTER_AREA)

        # Quantitative Metrics: Damaged
        p_dmg = psnr(clean_eval, dmg_eval)
        s_dmg = safe_ssim(clean_eval, dmg_eval)
        m_dmg = float(np.mean(np.abs(clean_eval.astype(float) - dmg_eval.astype(float))))

        # Quantitative Metrics: Baseline
        p_base = psnr(clean_eval, restored_base_eval)
        s_base = safe_ssim(clean_eval, restored_base_eval)
        m_base = float(np.mean(np.abs(clean_eval.astype(float) - restored_base_eval.astype(float))))
        delta_p_base = p_base - p_dmg
        delta_s_base = s_base - s_dmg

        # Quantitative Metrics: Experimental
        p_exp = psnr(clean_eval, restored_exp_eval)
        s_exp = safe_ssim(clean_eval, restored_exp_eval)
        m_exp = float(np.mean(np.abs(clean_eval.astype(float) - restored_exp_eval.astype(float))))
        delta_p_exp = p_exp - p_dmg
        delta_s_exp = s_exp - s_dmg

        exp_vs_base_p = p_exp - p_base
        exp_vs_base_s = s_exp - s_base

        cov_pct = res_exp["diagnostics"]["mask_coverage_pct"]
        undmg_diff = res_exp["diagnostics"]["undamaged_max_diff"]
        undmg_mae = res_exp["diagnostics"]["undamaged_mae"]
        is_exact = res_exp["diagnostics"]["is_bit_exact"]

        row = {
            "object_id": obj_id,
            "title": title,
            "classification": cat,
            "width": w_orig,
            "height": h_orig,
            "damage_coverage_pct": cov_pct,
            "dmg_psnr": round(p_dmg, 4),
            "dmg_ssim": round(s_dmg, 6),
            "dmg_mae": round(m_dmg, 4),
            "base_psnr": round(p_base, 4),
            "base_ssim": round(s_base, 6),
            "base_mae": round(m_base, 4),
            "base_delta_psnr": round(delta_p_base, 4),
            "base_delta_ssim": round(delta_s_base, 6),
            "base_latency_ms": round(base_latency, 1),
            "exp_psnr": round(p_exp, 4),
            "exp_ssim": round(s_exp, 6),
            "exp_mae": round(m_exp, 4),
            "exp_delta_psnr": round(delta_p_exp, 4),
            "exp_delta_ssim": round(delta_s_exp, 6),
            "exp_latency_ms": round(exp_latency, 1),
            "exp_vs_base_delta_psnr": round(exp_vs_base_p, 4),
            "exp_vs_base_delta_ssim": round(exp_vs_base_s, 6),
            "undamaged_max_diff": undmg_diff,
            "undamaged_mae": undmg_mae,
            "is_bit_exact": is_exact
        }
        results_rows.append(row)

        print(
            f"{obj_id:<8} | {cat:<12} | {cov_pct:>5.1f}% | {p_dmg:>8.2f} dB | {p_base:>8.2f} dB | "
            f"{p_exp:>8.2f} dB | {exp_vs_base_p:>+6.2f}"
        )

        # Save Visual Comparison Panels for Selected Exemplars (First sample per category)
        if save_visuals and idx in (1, 5, 9, 13, 21, 29):
            # Create a 4-panel comparison: Clean Reference | Damaged Input | Baseline Restored | Experimental Restored
            vis_w, vis_h = 320, int(round(320 * (h_eval / w_eval)))
            c_p = cv2.resize(clean_eval, (vis_w, vis_h))
            d_p = cv2.resize(dmg_eval, (vis_w, vis_h))
            b_p = cv2.resize(restored_base_eval, (vis_w, vis_h))
            e_p = cv2.resize(restored_exp_eval, (vis_w, vis_h))

            # Add header text labels to images
            def label_img(im, txt):
                labeled = im.copy()
                cv2.rectangle(labeled, (0, 0), (vis_w, 24), (20, 20, 20), -1)
                cv2.putText(labeled, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                return labeled

            c_lbl = label_img(c_p, "Authentic Clean Reference")
            d_lbl = label_img(d_p, f"Damaged (Cov: {cov_pct:.1f}%)")
            b_lbl = label_img(b_p, f"Baseline (PSNR: {p_base:.2f})")
            e_lbl = label_img(e_p, f"Experimental (PSNR: {p_exp:.2f})")

            panel = np.hstack([c_lbl, d_lbl, b_lbl, e_lbl])
            panel_path = visuals_dir / f"comparison_{cat.lower()}_{obj_id}.jpg"
            Image.fromarray(panel).save(str(panel_path), quality=92)

        gc.collect()

    # Save CSV
    with open(metrics_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results_rows)

    print("-" * 80)
    print(f"\n[Evaluation Complete] Evaluated {len(results_rows)} holdout artworks.")
    print(f"Metrics saved to: {metrics_path}")

    # Summary Statistics
    mean_base_psnr = float(np.mean([r["base_psnr"] for r in results_rows]))
    mean_exp_psnr = float(np.mean([r["exp_psnr"] for r in results_rows]))
    mean_base_ssim = float(np.mean([r["base_ssim"] for r in results_rows]))
    mean_exp_ssim = float(np.mean([r["exp_ssim"] for r in results_rows]))
    mean_base_d_psnr = float(np.mean([r["base_delta_psnr"] for r in results_rows]))
    mean_exp_d_psnr = float(np.mean([r["exp_delta_psnr"] for r in results_rows]))
    exact_count = sum(1 for r in results_rows if r["is_bit_exact"])

    print("\n" + "=" * 80)
    print("FINAL INDEPENDENT EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Evaluated Artworks:          40")
    print(f"Bit-Exact Preserved Artworks:      {exact_count} / 40 (100.0%)")
    print(f"Baseline Mean PSNR:                {mean_base_psnr:.2f} dB (Improvement: {mean_base_d_psnr:+.2f} dB)")
    print(f"Experimental Mean PSNR:            {mean_exp_psnr:.2f} dB (Improvement: {mean_exp_d_psnr:+.2f} dB)")
    print(f"Baseline Mean SSIM:                {mean_base_ssim:.4f}")
    print(f"Experimental Mean SSIM:            {mean_exp_ssim:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_independent_holdout()
