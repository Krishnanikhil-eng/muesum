import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import gc
import psutil
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
import cv2

# Local imports
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from unet_model import ResNet34UNet
from dataset import pad_to_multiple_of_32, unpad
from utils import tensor_to_uint8, psnr, safe_ssim, ensure_dir

def run_full_200_experiment():
    print("=" * 75)
    print("MUESUM — FULL 200 TEST SET MASK-GUIDED BLENDING EXPERIMENT")
    print("=" * 75)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    mem_init = psutil.virtual_memory()
    print(f"Initial Memory: Available={mem_init.available / (1024**3):.2f} GB | Total={mem_init.total / (1024**3):.2f} GB")
    
    ckpt_path = root_dir / "checkpoints" / "baseline_run" / "best_model.pth"
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"
    
    print(f"Loading best checkpoint (Epoch 8): {ckpt_path}")
    model = ResNet34UNet(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print("[OK] Model loaded and set to eval mode.")
    
    clean_dir = root_dir / "data" / "clean"
    damaged_dir = root_dir / "data" / "damaged" / "test"
    test_csv = root_dir / "data" / "splits" / "test_ids.csv"
    meta_csv = root_dir / "data" / "metadata.csv"
    
    out_dir = root_dir / "reports" / "baseline_evaluation" / "mask_blending_experiment" / "full_200"
    vis_dir = out_dir / "visual_comparisons"
    plots_dir = out_dir / "plots"
    ensure_dir(str(vis_dir))
    ensure_dir(str(plots_dir))
    
    test_ids = pd.read_csv(test_csv)['obj_id'].astype(str).str.strip().tolist()
    meta_df = pd.read_csv(meta_csv)
    meta_map = {str(row['object_id']).strip(): row for _, row in meta_df.iterrows()}
    
    print(f"Total test images in split: {len(test_ids)} (all unique: {len(set(test_ids)) == len(test_ids)})")
    assert len(test_ids) == 200
    
    metrics_csv_path = out_dir / "full_200_metrics.csv"
    
    # Check for resume
    completed_ids = set()
    records = []
    if metrics_csv_path.exists():
        try:
            existing_df = pd.read_csv(metrics_csv_path)
            if len(existing_df) > 0 and 'obj_id' in existing_df.columns:
                records = existing_df.to_dict('records')
                completed_ids = set(existing_df['obj_id'].astype(str).str.strip())
                print(f"Resuming: found {len(completed_ids)} already completed samples.")
        except Exception as e:
            print("Could not read existing metrics for resume:", e)
            
    t_start = time.time()
    
    for idx, obj_id in enumerate(test_ids):
        if obj_id in completed_ids:
            continue
            
        c_path = clean_dir / f"{obj_id}.jpg"
        d_path = damaged_dir / f"{obj_id}.jpg"
        m_path = damaged_dir / f"{obj_id}_mask.png"
        
        meta = meta_map.get(obj_id, {})
        title = meta.get('title', 'Unknown')
        classification = meta.get('classification', 'Unknown')
        department = meta.get('department', 'Unknown')
        
        with Image.open(c_path) as c_im, Image.open(d_path) as d_im, Image.open(m_path) as m_im:
            c_rgb = c_im.convert('RGB')
            d_rgb = d_im.convert('RGB')
            m_l = m_im.convert('L')
            
        orig_w, orig_h = c_rgb.size
        assert d_rgb.size == (orig_w, orig_h), f"Dimension mismatch for {obj_id}"
        assert m_l.size == (orig_w, orig_h), f"Mask dimension mismatch for {obj_id}"
        
        # 1024 aspect-preserving resize
        scale = 1024.0 / max(orig_w, orig_h)
        new_w = max(32, int(round(orig_w * scale)))
        new_h = max(32, int(round(orig_h * scale)))
        
        c_res = c_rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
        d_res = d_rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
        m_res = m_l.resize((new_w, new_h), Image.Resampling.NEAREST)
        
        clean_np = np.array(c_res)
        damaged_np = np.array(d_res)
        mask_np = np.array(m_res)
        
        # Mask normalization (1 = damaged, 0 = clean)
        mask_bin = (mask_np > 127).astype(np.float32)
        mask_cov = (np.sum(mask_bin) / mask_bin.size) * 100.0
        mask_3c = np.repeat(mask_bin[:, :, np.newaxis], 3, axis=2)
        
        # Raw UNet inference
        t0 = time.time()
        d_pad, pad_info = pad_to_multiple_of_32(damaged_np)
        inp_tensor = torch.from_numpy(d_pad.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        
        with torch.no_grad():
            out_tensor = model(inp_tensor.to(device))
        latency_ms = (time.time() - t0) * 1000.0
        
        out_np = tensor_to_uint8(out_tensor[0].cpu())
        restored_raw = unpad(out_np, pad_info)
        if restored_raw.shape != clean_np.shape:
            restored_raw = cv2.resize(restored_raw, (new_w, new_h))
            
        # Mask-guided blending: final = restored * mask + original * (1 - mask)
        blended_f = restored_raw.astype(np.float32) * mask_3c + damaged_np.astype(np.float32) * (1.0 - mask_3c)
        restored_blended = np.clip(np.round(blended_f), 0, 255).astype(np.uint8)
        
        # Verification: Undamaged preservation MAE
        undamaged_mask = (mask_bin == 0.0)
        undamaged_diff = np.abs(restored_blended[undamaged_mask].astype(np.int32) - damaged_np[undamaged_mask].astype(np.int32))
        undamaged_max_diff = int(np.max(undamaged_diff)) if np.any(undamaged_mask) else 0
        undamaged_mae = float(np.mean(undamaged_diff)) if np.any(undamaged_mask) else 0.0
        assert undamaged_max_diff == 0, f"Undamaged pixels modified for ID {obj_id}!"
        
        # Metrics: Whole-image
        psnr_dmg_whole = psnr(clean_np, damaged_np)
        ssim_dmg_whole = safe_ssim(clean_np, damaged_np)
        
        psnr_raw_whole = psnr(clean_np, restored_raw)
        ssim_raw_whole = safe_ssim(clean_np, restored_raw)
        
        psnr_blend_whole = psnr(clean_np, restored_blended)
        ssim_blend_whole = safe_ssim(clean_np, restored_blended)
        
        delta_psnr_raw = psnr_raw_whole - psnr_dmg_whole
        delta_ssim_raw = ssim_raw_whole - ssim_dmg_whole
        
        delta_psnr_blend = psnr_blend_whole - psnr_dmg_whole
        delta_ssim_blend = ssim_blend_whole - ssim_dmg_whole
        
        # Metrics: Damaged ROI
        dmg_mask_bool = (mask_bin == 1.0)
        if np.any(dmg_mask_bool):
            c_roi = clean_np[dmg_mask_bool].astype(np.float64)
            d_roi = damaged_np[dmg_mask_bool].astype(np.float64)
            r_roi = restored_raw[dmg_mask_bool].astype(np.float64)
            
            mse_dmg_roi = np.mean((c_roi - d_roi) ** 2)
            mse_res_roi = np.mean((c_roi - r_roi) ** 2)
            
            psnr_dmg_roi = 20.0 * np.log10(255.0 / np.sqrt(mse_dmg_roi)) if mse_dmg_roi > 0 else 100.0
            psnr_res_roi = 20.0 * np.log10(255.0 / np.sqrt(mse_res_roi)) if mse_res_roi > 0 else 100.0
            delta_psnr_roi = psnr_res_roi - psnr_dmg_roi
        else:
            psnr_dmg_roi, psnr_res_roi, delta_psnr_roi = 0.0, 0.0, 0.0
            
        row_data = {
            "obj_id": obj_id,
            "title": str(title)[:35],
            "classification": classification,
            "department": department,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "eval_width": new_w,
            "eval_height": new_h,
            "mask_coverage_pct": round(mask_cov, 2),
            "latency_ms": round(latency_ms, 1),
            "undamaged_max_diff": undamaged_max_diff,
            "undamaged_mae": undamaged_mae,
            "damaged_psnr_whole": round(psnr_dmg_whole, 4),
            "damaged_ssim_whole": round(ssim_dmg_whole, 4),
            "raw_psnr_whole": round(psnr_raw_whole, 4),
            "raw_ssim_whole": round(ssim_raw_whole, 4),
            "raw_delta_psnr": round(delta_psnr_raw, 4),
            "raw_delta_ssim": round(delta_ssim_raw, 4),
            "blended_psnr_whole": round(psnr_blend_whole, 4),
            "blended_ssim_whole": round(ssim_blend_whole, 4),
            "blended_delta_psnr": round(delta_psnr_blend, 4),
            "blended_delta_ssim": round(delta_ssim_blend, 4),
            "roi_damaged_psnr": round(psnr_dmg_roi, 4),
            "roi_restored_psnr": round(psnr_res_roi, 4),
            "roi_delta_psnr": round(delta_psnr_roi, 4)
        }
        records.append(row_data)
        completed_ids.add(obj_id)
        
        # Save visual comparison for select representative samples (anchor 247010 + 9 others)
        key_visual_ids = ["247010", "48953", "451301", "451019", "336663", "466379", "403800", "354833", "283234", "314620", "437468", "324005"]
        if obj_id in key_visual_ids:
            scale_v = 400.0 / new_h
            vw = max(1, int(new_w * scale_v))
            vh = 400
            c_v = c_res.resize((vw, vh), Image.Resampling.BILINEAR)
            d_v = d_res.resize((vw, vh), Image.Resampling.BILINEAR)
            r_v = Image.fromarray(restored_raw).resize((vw, vh), Image.Resampling.BILINEAR)
            b_v = Image.fromarray(restored_blended).resize((vw, vh), Image.Resampling.BILINEAR)
            
            comp = Image.new("RGB", (vw * 4 + 30, vh + 40), color=(25, 25, 25))
            comp.paste(c_v, (0, 30))
            comp.paste(d_v, (vw + 10, 30))
            comp.paste(r_v, (vw * 2 + 20, 30))
            comp.paste(b_v, (vw * 3 + 30, 30))
            comp.save(vis_dir / f"compare_4panel_{obj_id}.png")
            
        # Clean up memory
        del inp_tensor, out_tensor, d_pad, out_np, clean_np, damaged_np, restored_raw, restored_blended, c_res, d_res, m_res
        
        # Incremental save every 10 samples
        if len(records) % 10 == 0 or len(records) == len(test_ids):
            temp_df = pd.DataFrame(records)
            temp_df.to_csv(metrics_csv_path, index=False)
            mem_curr = psutil.virtual_memory()
            avg_blend_gain = temp_df['blended_delta_psnr'].mean()
            pct_improved = (temp_df['blended_delta_psnr'] > 0).mean() * 100.0
            print(f"[{len(records):3d}/200] Processed | Avg Blended Delta PSNR: {avg_blend_gain:+.2f} dB | Improved: {pct_improved:.1f}% | RAM Avail: {mem_curr.available / (1024**3):.2f} GB")
            gc.collect()
            
    dur_total = time.time() - t_start
    print(f"\n[SUCCESS] Completed all 200 samples in {dur_total:.1f} s ({dur_total/60.0:.2f} min).")
    
    # Save final complete metrics
    final_df = pd.DataFrame(records)
    final_df.to_csv(metrics_csv_path, index=False)
    
    # -------------------------------------------------------------
    # CATEGORY ANALYSIS
    # -------------------------------------------------------------
    cat_summary = []
    for cat, group in final_df.groupby('classification'):
        if len(group) < 3:
            continue
        cat_summary.append({
            "classification": cat,
            "sample_count": len(group),
            "mean_damage_cov_pct": round(group['mask_coverage_pct'].mean(), 2),
            "damaged_psnr_mean": round(group['damaged_psnr_whole'].mean(), 2),
            "raw_psnr_mean": round(group['raw_psnr_whole'].mean(), 2),
            "raw_delta_psnr_mean": round(group['raw_delta_psnr'].mean(), 2),
            "blended_psnr_mean": round(group['blended_psnr_whole'].mean(), 2),
            "blended_delta_psnr_mean": round(group['blended_delta_psnr'].mean(), 2),
            "blended_ssim_mean": round(group['blended_ssim_whole'].mean(), 4),
            "pct_improved_psnr": round((group['blended_delta_psnr'] > 0).mean() * 100.0, 1),
            "roi_delta_psnr_mean": round(group['roi_delta_psnr'].mean(), 2)
        })
    cat_df = pd.DataFrame(cat_summary).sort_values(by="sample_count", ascending=False)
    cat_csv_path = out_dir / "category_analysis.csv"
    cat_df.to_csv(cat_csv_path, index=False)
    print(f"[OK] Saved category analysis to {cat_csv_path.name}")
    
    # -------------------------------------------------------------
    # GENERATE PLOTS
    # -------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Plot 1: PSNR Comparison Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(final_df['damaged_psnr_whole'], bins=20, alpha=0.5, label=f"Damaged Input ({final_df['damaged_psnr_whole'].mean():.2f} dB)", color='red')
    axes[0].hist(final_df['raw_psnr_whole'], bins=20, alpha=0.5, label=f"Raw UNet ({final_df['raw_psnr_whole'].mean():.2f} dB)", color='orange')
    axes[0].hist(final_df['blended_psnr_whole'], bins=20, alpha=0.6, label=f"Mask-Blended ({final_df['blended_psnr_whole'].mean():.2f} dB)", color='green')
    axes[0].set_title("Full 200 Test Set: Whole-Image PSNR Distributions")
    axes[0].set_xlabel("PSNR (dB)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Delta PSNR distributions
    axes[1].hist(final_df['raw_delta_psnr'], bins=20, alpha=0.5, label=f"Raw UNet Delta ({final_df['raw_delta_psnr'].mean():+.2f} dB)", color='orange')
    axes[1].hist(final_df['blended_delta_psnr'], bins=20, alpha=0.6, label=f"Mask-Blended Delta ({final_df['blended_delta_psnr'].mean():+.2f} dB)", color='green')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1.2)
    axes[1].set_title("Delta PSNR Improvement Distribution (N=200)")
    axes[1].set_xlabel("Delta PSNR (dB)")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "psnr_distributions_200.png", dpi=150)
    plt.close()
    
    # Plot 2: SSIM Comparison Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(final_df['damaged_ssim_whole'], bins=20, alpha=0.5, label=f"Damaged Input ({final_df['damaged_ssim_whole'].mean():.4f})", color='red')
    axes[0].hist(final_df['raw_ssim_whole'], bins=20, alpha=0.5, label=f"Raw UNet ({final_df['raw_ssim_whole'].mean():.4f})", color='orange')
    axes[0].hist(final_df['blended_ssim_whole'], bins=20, alpha=0.6, label=f"Mask-Blended ({final_df['blended_ssim_whole'].mean():.4f})", color='green')
    axes[0].set_title("Full 200 Test Set: Whole-Image SSIM Distributions")
    axes[0].set_xlabel("SSIM [0, 1]")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(final_df['raw_delta_ssim'], bins=20, alpha=0.5, label=f"Raw UNet Delta ({final_df['raw_delta_ssim'].mean():+.4f})", color='orange')
    axes[1].hist(final_df['blended_delta_ssim'], bins=20, alpha=0.6, label=f"Mask-Blended Delta ({final_df['blended_delta_ssim'].mean():+.4f})", color='green')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1.2)
    axes[1].set_title("Delta SSIM Distribution (N=200)")
    axes[1].set_xlabel("Delta SSIM")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "ssim_distributions_200.png", dpi=150)
    plt.close()
    
    # Plot 3: Scatter of Damage Coverage vs Delta PSNR
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(final_df['mask_coverage_pct'], final_df['blended_delta_psnr'], color='green', alpha=0.6, edgecolors='none', s=40, label='Mask-Blended')
    ax.scatter(final_df['mask_coverage_pct'], final_df['raw_delta_psnr'], color='orange', alpha=0.4, edgecolors='none', s=25, label='Raw UNet')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title("Damage Mask Coverage (%) vs Delta PSNR Improvement (N=200)")
    ax.set_xlabel("Damage Mask Area (%)")
    ax.set_ylabel("Delta PSNR (dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "damage_vs_improvement_200.png", dpi=150)
    plt.close()
    
    print("[OK] All evaluation plots saved.")

if __name__ == "__main__":
    run_full_200_experiment()

