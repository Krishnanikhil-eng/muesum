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

def load_eval_model(ckpt_path, device):
    print(f"Loading best checkpoint from: {ckpt_path}")
    model = ResNet34UNet(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"[OK] Model loaded successfully. Checkpoint metadata: epoch={ckpt.get('epoch')}, val_ssim={ckpt.get('val_ssim'):.4f}")
    return model

def run_evaluation():
    print("=" * 70)
    print("MUESUM — STAGE 5, PHASE E: BASELINE EVALUATION AUDIT")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    ckpt_path = root_dir / "checkpoints" / "baseline_run" / "best_model.pth"
    assert ckpt_path.exists(), f"Best model checkpoint not found at {ckpt_path}"
    model = load_eval_model(str(ckpt_path), device)
    
    clean_dir = root_dir / "data" / "clean"
    damaged_dir = root_dir / "data" / "damaged" / "test"
    test_csv = root_dir / "data" / "splits" / "test_ids.csv"
    meta_csv = root_dir / "data" / "metadata.csv"
    
    reports_dir = root_dir / "reports" / "baseline_evaluation"
    vis_dir = reports_dir / "visual_comparisons"
    plots_dir = reports_dir / "plots"
    ensure_dir(str(vis_dir))
    ensure_dir(str(plots_dir))
    
    test_ids = pd.read_csv(test_csv)['obj_id'].astype(str).str.strip().tolist()
    meta_df = pd.read_csv(meta_csv)
    meta_map = {str(row['object_id']).strip(): row for _, row in meta_df.iterrows()}
    
    print(f"Total test images in split: {len(test_ids)}")
    assert len(test_ids) == 200, f"Expected 200 test images, got {len(test_ids)}"
    
    # -------------------------------------------------------------
    # EVALUATION 1: 256x256 Center-Crop Benchmark (Identical to Validation)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STARTING EVALUATION 1: 256x256 Center-Crop Benchmark (N=200)")
    print("-" * 70)
    
    crop_results = []
    ps = 256
    
    t_start_e1 = time.time()
    for idx, obj_id in enumerate(test_ids):
        c_path = clean_dir / f"{obj_id}.jpg"
        d_path = damaged_dir / f"{obj_id}.jpg"
        m_path = damaged_dir / f"{obj_id}_mask.png"
        
        with Image.open(c_path) as c_im, Image.open(d_path) as d_im, Image.open(m_path) as m_im:
            clean = np.array(c_im.convert('RGB'))
            damaged = np.array(d_im.convert('RGB'))
            mask = np.array(m_im.convert('L'))
            
        h, w = clean.shape[:2]
        
        # Center crop coordinates
        top = (h - ps) // 2
        left = (w - ps) // 2
        clean_crop = clean[top:top+ps, left:left+ps]
        damaged_crop = damaged[top:top+ps, left:left+ps]
        mask_crop = mask[top:top+ps, left:left+ps]
        
        # Mask damage coverage
        dmg_px = np.sum(mask_crop > 127)
        mask_cov = (dmg_px / mask_crop.size) * 100.0
        
        # Model inference on crop
        inp_tensor = torch.from_numpy(damaged_crop.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        t0 = time.time()
        with torch.no_grad():
            out_tensor = model(inp_tensor.to(device))
        latency = (time.time() - t0) * 1000.0
        
        restored_crop = tensor_to_uint8(out_tensor[0].cpu())
        
        # Calculate metrics
        psnr_dmg = psnr(clean_crop, damaged_crop)
        ssim_dmg = safe_ssim(clean_crop, damaged_crop)
        psnr_res = psnr(clean_crop, restored_crop)
        ssim_res = safe_ssim(clean_crop, restored_crop)
        delta_psnr = psnr_res - psnr_dmg
        delta_ssim = ssim_res - ssim_dmg
        
        crop_results.append({
            "obj_id": obj_id,
            "eval_mode": "center_crop_256",
            "height": ps,
            "width": ps,
            "mask_coverage_pct": mask_cov,
            "damaged_psnr": psnr_dmg,
            "damaged_ssim": ssim_dmg,
            "restored_psnr": psnr_res,
            "restored_ssim": ssim_res,
            "delta_psnr": delta_psnr,
            "delta_ssim": delta_ssim,
            "latency_ms": latency
        })
        
        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_ids):
            print(f"  Processed {idx + 1:3d}/200 crops | Avg Delta PSNR: {np.mean([r['delta_psnr'] for r in crop_results]):+.2f} dB | Avg Delta SSIM: {np.mean([r['delta_ssim'] for r in crop_results]):+.4f}")
            
    dur_e1 = time.time() - t_start_e1
    print(f"[OK] Evaluation 1 complete in {dur_e1:.1f} s.")
    
    # Save representative 256 crop comparisons (Anchor 247010 + 2 others)
    for sample_id in ["247010", "437468", "324005"]:
        c_path = clean_dir / f"{sample_id}.jpg"
        d_path = damaged_dir / f"{sample_id}.jpg"
        with Image.open(c_path) as c_im, Image.open(d_path) as d_im:
            c = np.array(c_im.convert('RGB'))
            d = np.array(d_im.convert('RGB'))
        h, w = c.shape[:2]
        top, left = (h - ps) // 2, (w - ps) // 2
        c_c, d_c = c[top:top+ps, left:left+ps], d[top:top+ps, left:left+ps]
        inp_t = torch.from_numpy(d_c.transpose(2,0,1)).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            res_c = tensor_to_uint8(model(inp_t.to(device))[0].cpu())
        
        comp_img = Image.new("RGB", (ps * 3 + 20, ps + 40), color=(30, 30, 30))
        comp_img.paste(Image.fromarray(c_c), (0, 30))
        comp_img.paste(Image.fromarray(d_c), (ps + 10, 30))
        comp_img.paste(Image.fromarray(res_c), (ps * 2 + 20, 30))
        comp_img.save(vis_dir / f"crop256_{sample_id}_comparison.png")

    # -------------------------------------------------------------
    # EVALUATION 2: 1024-Pixel Resized Benchmark (Complete Composition, N=200)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STARTING EVALUATION 2: 1024-Pixel Resized Benchmark (N=200)")
    print("-" * 70)
    
    resized_results = []
    t_start_e2 = time.time()
    
    for idx, obj_id in enumerate(test_ids):
        c_path = clean_dir / f"{obj_id}.jpg"
        d_path = damaged_dir / f"{obj_id}.jpg"
        m_path = damaged_dir / f"{obj_id}_mask.png"
        
        with Image.open(c_path) as c_im, Image.open(d_path) as d_im, Image.open(m_path) as m_im:
            c_rgb = c_im.convert('RGB')
            d_rgb = d_im.convert('RGB')
            m_l = m_im.convert('L')
            
        orig_w, orig_h = c_rgb.size
        # Aspect-preserving scale to max 1024
        scale = 1024.0 / max(orig_w, orig_h)
        new_w = max(32, int(round(orig_w * scale)))
        new_h = max(32, int(round(orig_h * scale)))
        
        c_res = c_rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
        d_res = d_rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
        m_res = m_l.resize((new_w, new_h), Image.Resampling.NEAREST)
        
        clean_np = np.array(c_res)
        damaged_np = np.array(d_res)
        mask_np = np.array(m_res)
        
        dmg_px = np.sum(mask_np > 127)
        mask_cov = (dmg_px / mask_np.size) * 100.0
        
        # Pad to multiple of 32
        d_pad, pad_info = pad_to_multiple_of_32(damaged_np)
        inp_tensor = torch.from_numpy(d_pad.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        
        t0 = time.time()
        with torch.no_grad():
            out_tensor = model(inp_tensor.to(device))
        latency = (time.time() - t0) * 1000.0
        
        out_np = tensor_to_uint8(out_tensor[0].cpu())
        restored_np = unpad(out_np, pad_info)
        
        if restored_np.shape != clean_np.shape:
            restored_np = cv2.resize(restored_np, (clean_np.shape[1], clean_np.shape[0]))
            
        psnr_dmg = psnr(clean_np, damaged_np)
        ssim_dmg = safe_ssim(clean_np, damaged_np)
        psnr_res = psnr(clean_np, restored_np)
        ssim_res = safe_ssim(clean_np, restored_np)
        delta_psnr = psnr_res - psnr_dmg
        delta_ssim = ssim_res - ssim_dmg
        
        resized_results.append({
            "obj_id": obj_id,
            "eval_mode": "resized_1024",
            "height": new_h,
            "width": new_w,
            "mask_coverage_pct": mask_cov,
            "damaged_psnr": psnr_dmg,
            "damaged_ssim": ssim_dmg,
            "restored_psnr": psnr_res,
            "restored_ssim": ssim_res,
            "delta_psnr": delta_psnr,
            "delta_ssim": delta_ssim,
            "latency_ms": latency
        })
        
        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_ids):
            print(f"  Processed {idx + 1:3d}/200 resized | Avg Delta PSNR: {np.mean([r['delta_psnr'] for r in resized_results]):+.2f} dB | Avg Delta SSIM: {np.mean([r['delta_ssim'] for r in resized_results]):+.4f}")
            
    dur_e2 = time.time() - t_start_e2
    print(f"[OK] Evaluation 2 complete in {dur_e2:.1f} s.")
    
    # Save representative 1024 resized comparisons
    for sample_id in ["247010", "437468", "324005"]:
        c_path = clean_dir / f"{sample_id}.jpg"
        d_path = damaged_dir / f"{sample_id}.jpg"
        with Image.open(c_path) as c_im, Image.open(d_path) as d_im:
            ow, oh = c_im.size
            sc = 512.0 / max(ow, oh)
            nw, nh = int(ow * sc), int(oh * sc)
            c_s = c_im.resize((nw, nh), Image.Resampling.BILINEAR)
            d_s = d_im.resize((nw, nh), Image.Resampling.BILINEAR)
            
        d_pad, p_info = pad_to_multiple_of_32(np.array(d_s))
        inp_t = torch.from_numpy(d_pad.transpose(2,0,1)).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            res_p = tensor_to_uint8(model(inp_t.to(device))[0].cpu())
        res_s = unpad(res_p, p_info)
        
        comp_img = Image.new("RGB", (nw * 3 + 20, nh + 40), color=(30, 30, 30))
        comp_img.paste(c_s, (0, 30))
        comp_img.paste(d_s, (nw + 10, 30))
        comp_img.paste(Image.fromarray(res_s), (nw * 2 + 20, 30))
        comp_img.save(vis_dir / f"resized1024_{sample_id}_comparison.png")

    # Combine into test_metrics.csv
    all_metrics_df = pd.DataFrame(crop_results + resized_results)
    metrics_csv_path = reports_dir / "test_metrics.csv"
    all_metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[OK] Saved full metrics to {metrics_csv_path.name} ({len(all_metrics_df)} rows).")

    # -------------------------------------------------------------
    # EVALUATION 3: Original-Resolution Benchmark (10 Diverse Artworks)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STARTING EVALUATION 3: Original-Resolution Benchmark (10 Diverse Artworks)")
    print("-" * 70)
    
    diverse_selection = [
        ("Anchor (Wall Painting)", "247010"),
        ("Painting (Asian Art)", "48953"),
        ("Codex (Islamic Art)", "451301"),
        ("Textile (Woven Tapestry)", "451019"),
        ("Drawing (Allegorical)", "336663"),
        ("Sculpture (Stone)", "466379"),
        ("Print (Ancient Philosophers)", "403800"),
        ("Book (Works of Virgil)", "354833"),
        ("Photograph (Sculpture Study)", "283234"),
        ("Costume (Four-Cornered Hat)", "314620")
    ]
    
    orig_res_results = []
    
    for category_name, obj_id in diverse_selection:
        c_path = clean_dir / f"{obj_id}.jpg"
        d_path = damaged_dir / f"{obj_id}.jpg"
        m_path = damaged_dir / f"{obj_id}_mask.png"
        
        meta_row = meta_map.get(obj_id, {})
        title = meta_row.get('title', 'Unknown')
        classification = meta_row.get('classification', 'Unknown')
        department = meta_row.get('department', 'Unknown')
        
        print(f"\nProcessing {category_name}: ID {obj_id}...")
        gc.collect()
        mem_before = psutil.virtual_memory().available / (1024**3)
        
        with Image.open(c_path) as c_im, Image.open(d_path) as d_im, Image.open(m_path) as m_im:
            clean_orig = np.array(c_im.convert('RGB'))
            damaged_orig = np.array(d_im.convert('RGB'))
            mask_orig = np.array(m_im.convert('L'))
            
        orig_h, orig_w = clean_orig.shape[:2]
        mp = (orig_w * orig_h) / 1e6
        dmg_px = np.sum(mask_orig > 127)
        mask_cov = (dmg_px / mask_orig.size) * 100.0
        
        print(f"  Dimensions: {orig_w} x {orig_h} ({mp:.2f} Megapixels), Mask coverage: {mask_cov:.2f}%")
        
        # Pad to multiple of 32
        d_pad, pad_info = pad_to_multiple_of_32(damaged_orig)
        inp_tensor = torch.from_numpy(d_pad.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        
        t0 = time.time()
        with torch.no_grad():
            out_tensor = model(inp_tensor.to(device))
        latency_s = time.time() - t0
        
        out_np = tensor_to_uint8(out_tensor[0].cpu())
        restored_orig = unpad(out_np, pad_info)
        
        if restored_orig.shape != clean_orig.shape:
            restored_orig = cv2.resize(restored_orig, (clean_orig.shape[1], clean_orig.shape[0]))
            
        mem_after = psutil.virtual_memory().available / (1024**3)
        peak_ram_used = max(0.0, mem_before - mem_after)
        
        # Compute metrics
        psnr_dmg = psnr(clean_orig, damaged_orig)
        ssim_dmg = safe_ssim(clean_orig, damaged_orig)
        psnr_res = psnr(clean_orig, restored_orig)
        ssim_res = safe_ssim(clean_orig, restored_orig)
        delta_psnr = psnr_res - psnr_dmg
        delta_ssim = ssim_res - ssim_dmg
        
        print(f"  Latency: {latency_s:.2f} s | Damaged PSNR: {psnr_dmg:.2f} dB -> Restored PSNR: {psnr_res:.2f} dB (Delta: {delta_psnr:+.2f} dB)")
        print(f"  Damaged SSIM: {ssim_dmg:.4f} -> Restored SSIM: {ssim_res:.4f} (Delta: {delta_ssim:+.4f})")
        
        orig_res_results.append({
            "category": category_name,
            "obj_id": obj_id,
            "title": str(title)[:40],
            "classification": classification,
            "department": department,
            "width": orig_w,
            "height": orig_h,
            "megapixels": round(mp, 2),
            "mask_coverage_pct": round(mask_cov, 2),
            "inference_time_s": round(latency_s, 2),
            "ram_delta_gb": round(peak_ram_used, 2),
            "damaged_psnr": round(psnr_dmg, 2),
            "damaged_ssim": round(ssim_dmg, 4),
            "restored_psnr": round(psnr_res, 2),
            "restored_ssim": round(ssim_res, 4),
            "delta_psnr": round(delta_psnr, 2),
            "delta_ssim": round(delta_ssim, 4)
        })
        
        # Save side-by-side composite comparison
        # Scale to max height 600 for visual inspection composite
        scale_vis = 600.0 / orig_h
        vis_w = max(1, int(orig_w * scale_vis))
        vis_h = 600
        
        c_vis = Image.fromarray(clean_orig).resize((vis_w, vis_h), Image.Resampling.BILINEAR)
        d_vis = Image.fromarray(damaged_orig).resize((vis_w, vis_h), Image.Resampling.BILINEAR)
        r_vis = Image.fromarray(restored_orig).resize((vis_w, vis_h), Image.Resampling.BILINEAR)
        
        comp = Image.new("RGB", (vis_w * 3 + 20, vis_h + 40), color=(25, 25, 25))
        comp.paste(c_vis, (0, 30))
        comp.paste(d_vis, (vis_w + 10, 30))
        comp.paste(r_vis, (vis_w * 2 + 20, 30))
        
        safe_cat = category_name.split()[0].lower()
        comp_out = vis_dir / f"original_res_{obj_id}_{safe_cat}.png"
        comp.save(comp_out)
        print(f"  Saved visual comparison: {comp_out.name}")
        
    bench_df = pd.DataFrame(orig_res_results)
    bench_csv = reports_dir / "inference_benchmark.csv"
    bench_df.to_csv(bench_csv, index=False)
    print(f"\n[OK] Saved inference benchmark to {bench_csv.name}.")
    
    # -------------------------------------------------------------
    # GENERATE PLOTS
    # -------------------------------------------------------------
    print("\nGenerating evaluation plots in:", plots_dir)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Plot 1: PSNR & SSIM Distributions (256 crop vs 1024 resized)
    df_crop = all_metrics_df[all_metrics_df['eval_mode'] == 'center_crop_256']
    df_res1024 = all_metrics_df[all_metrics_df['eval_mode'] == 'resized_1024']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 256 crop PSNR
    axes[0, 0].hist(df_crop['damaged_psnr'], bins=20, alpha=0.5, label='Damaged Input', color='red')
    axes[0, 0].hist(df_crop['restored_psnr'], bins=20, alpha=0.6, label='Model Restored', color='green')
    axes[0, 0].set_title(f"256x256 Crop PSNR (dB)\nDamaged Mean: {df_crop['damaged_psnr'].mean():.2f} -> Restored: {df_crop['restored_psnr'].mean():.2f}")
    axes[0, 0].set_xlabel("PSNR (dB)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 256 crop SSIM
    axes[0, 1].hist(df_crop['damaged_ssim'], bins=20, alpha=0.5, label='Damaged Input', color='red')
    axes[0, 1].hist(df_crop['restored_ssim'], bins=20, alpha=0.6, label='Model Restored', color='green')
    axes[0, 1].set_title(f"256x256 Crop SSIM\nDamaged Mean: {df_crop['damaged_ssim'].mean():.4f} -> Restored: {df_crop['restored_ssim'].mean():.4f}")
    axes[0, 1].set_xlabel("SSIM [0, 1]")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 1024 resized PSNR
    axes[1, 0].hist(df_res1024['damaged_psnr'], bins=20, alpha=0.5, label='Damaged Input', color='orange')
    axes[1, 0].hist(df_res1024['restored_psnr'], bins=20, alpha=0.6, label='Model Restored', color='blue')
    axes[1, 0].set_title(f"1024-Resized PSNR (dB)\nDamaged Mean: {df_res1024['damaged_psnr'].mean():.2f} -> Restored: {df_res1024['restored_psnr'].mean():.2f}")
    axes[1, 0].set_xlabel("PSNR (dB)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 1024 resized SSIM
    axes[1, 1].hist(df_res1024['damaged_ssim'], bins=20, alpha=0.5, label='Damaged Input', color='orange')
    axes[1, 1].hist(df_res1024['restored_ssim'], bins=20, alpha=0.6, label='Model Restored', color='blue')
    axes[1, 1].set_title(f"1024-Resized SSIM\nDamaged Mean: {df_res1024['damaged_ssim'].mean():.4f} -> Restored: {df_res1024['restored_ssim'].mean():.4f}")
    axes[1, 1].set_xlabel("SSIM [0, 1]")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot1_path = plots_dir / "psnr_ssim_distributions.png"
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot1_path.name}")
    
    # Plot 2: Delta Improvements
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df_crop['delta_psnr'], bins=20, alpha=0.6, label='256 Crop', color='green')
    axes[0].hist(df_res1024['delta_psnr'], bins=20, alpha=0.6, label='1024 Resized', color='blue')
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_title(f"Delta PSNR Improvement Distribution\nCrop Mean: {df_crop['delta_psnr'].mean():+.2f} dB | Resized Mean: {df_res1024['delta_psnr'].mean():+.2f} dB")
    axes[0].set_xlabel("Delta PSNR (dB)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(df_crop['delta_ssim'], bins=20, alpha=0.6, label='256 Crop', color='purple')
    axes[1].hist(df_res1024['delta_ssim'], bins=20, alpha=0.6, label='1024 Resized', color='teal')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title(f"Delta SSIM Improvement Distribution\nCrop Mean: {df_crop['delta_ssim'].mean():+.4f} | Resized Mean: {df_res1024['delta_ssim'].mean():+.4f}")
    axes[1].set_xlabel("Delta SSIM")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot2_path = plots_dir / "delta_improvements.png"
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot2_path.name}")
    
    # Plot 3: Damage Coverage vs Delta PSNR
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(df_crop['mask_coverage_pct'], df_crop['delta_psnr'], color='green', alpha=0.6, edgecolors='none')
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_title("256 Crop: Mask Damage % vs Delta PSNR")
    axes[0].set_xlabel("Damage Mask Area (%)")
    axes[0].set_ylabel("Delta PSNR (dB)")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(df_res1024['mask_coverage_pct'], df_res1024['delta_psnr'], color='blue', alpha=0.6, edgecolors='none')
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title("1024 Resized: Mask Damage % vs Delta PSNR")
    axes[1].set_xlabel("Damage Mask Area (%)")
    axes[1].set_ylabel("Delta PSNR (dB)")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot3_path = plots_dir / "damage_vs_improvement.png"
    plt.savefig(plot3_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot3_path.name}")

    print("\n" + "=" * 70)
    print("ALL THREE EVALUATIONS AND PLOTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_evaluation()

