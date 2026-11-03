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

def run_mask_blending_experiment():
    print("=" * 75)
    print("MUESUM — MASK-GUIDED BLENDING INFERENCE EXPERIMENT (10 TEST ARTWORKS)")
    print("=" * 75)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    ckpt_path = root_dir / "checkpoints" / "baseline_run" / "best_model.pth"
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"
    
    print(f"Loading best checkpoint (Epoch 8): {ckpt_path}")
    model = ResNet34UNet(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    clean_dir = root_dir / "data" / "clean"
    damaged_dir = root_dir / "data" / "damaged" / "test"
    meta_csv = root_dir / "data" / "metadata.csv"
    
    out_dir = root_dir / "reports" / "baseline_evaluation" / "mask_blending_experiment"
    vis_dir = out_dir / "visual_comparisons"
    ensure_dir(str(vis_dir))
    
    meta_df = pd.read_csv(meta_csv)
    meta_map = {str(row['object_id']).strip(): row for _, row in meta_df.iterrows()}
    
    diverse_samples = [
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
    
    results = []
    
    for cat_name, obj_id in diverse_samples:
        print(f"\nEvaluating sample {obj_id} ({cat_name})...")
        c_path = clean_dir / f"{obj_id}.jpg"
        d_path = damaged_dir / f"{obj_id}.jpg"
        m_path = damaged_dir / f"{obj_id}_mask.png"
        
        meta = meta_map.get(obj_id, {})
        title = meta.get('title', 'Unknown')
        
        with Image.open(c_path) as c_im, Image.open(d_path) as d_im, Image.open(m_path) as m_im:
            clean = np.array(c_im.convert('RGB'))
            damaged = np.array(d_im.convert('RGB'))
            mask = np.array(m_im.convert('L'))
            
        h, w = clean.shape[:2]
        assert damaged.shape[:2] == (h, w), f"Damaged shape mismatch for {obj_id}"
        assert mask.shape[:2] == (h, w), f"Mask shape mismatch for {obj_id}"
        
        # 1. Mask normalization: 1 = damaged, 0 = undamaged
        mask_binary = (mask > 127).astype(np.float32)
        mask_coverage = (np.sum(mask_binary) / mask_binary.size) * 100.0
        mask_3c = np.repeat(mask_binary[:, :, np.newaxis], 3, axis=2)
        
        # 2. Raw UNet inference
        gc.collect()
        t0 = time.time()
        d_pad, pad_info = pad_to_multiple_of_32(damaged)
        inp_tensor = torch.from_numpy(d_pad.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        
        with torch.no_grad():
            out_tensor = model(inp_tensor.to(device))
        latency = time.time() - t0
        
        out_np = tensor_to_uint8(out_tensor[0].cpu())
        restored_raw = unpad(out_np, pad_info)
        if restored_raw.shape != clean.shape:
            restored_raw = cv2.resize(restored_raw, (w, h))
            
        # 3. Mask-guided blending: final = restored * mask + original * (1 - mask)
        # Using float32 arithmetic, rounded and clipped to uint8 [0, 255]
        blended_float = restored_raw.astype(np.float32) * mask_3c + damaged.astype(np.float32) * (1.0 - mask_3c)
        restored_blended = np.clip(np.round(blended_float), 0, 255).astype(np.uint8)
        
        # 4. Verification: Check that undamaged pixels are EXACTLY unchanged
        undamaged_mask = (mask_binary == 0.0)
        undamaged_diff = np.abs(restored_blended[undamaged_mask].astype(np.int32) - damaged[undamaged_mask].astype(np.int32))
        undamaged_max_diff = int(np.max(undamaged_diff)) if np.any(undamaged_mask) else 0
        undamaged_mae = float(np.mean(undamaged_diff)) if np.any(undamaged_mask) else 0.0
        undamaged_mse = float(np.mean(undamaged_diff ** 2)) if np.any(undamaged_mask) else 0.0
        
        print(f"  Dimensions: {w} x {h} | Mask coverage: {mask_coverage:.2f}% | Latency: {latency:.2f} s")
        print(f"  Undamaged preservation: Max diff = {undamaged_max_diff}, MAE = {undamaged_mae:.6f}, MSE = {undamaged_mse:.6f}")
        assert undamaged_max_diff == 0, f"Undamaged pixels were modified! Max diff={undamaged_max_diff}"
        
        # 5. Whole-image metrics
        psnr_dmg_whole = psnr(clean, damaged)
        ssim_dmg_whole = safe_ssim(clean, damaged)
        
        psnr_raw_whole = psnr(clean, restored_raw)
        ssim_raw_whole = safe_ssim(clean, restored_raw)
        
        psnr_blend_whole = psnr(clean, restored_blended)
        ssim_blend_whole = safe_ssim(clean, restored_blended)
        
        delta_psnr_raw = psnr_raw_whole - psnr_dmg_whole
        delta_ssim_raw = ssim_raw_whole - ssim_dmg_whole
        
        delta_psnr_blend = psnr_blend_whole - psnr_dmg_whole
        delta_ssim_blend = ssim_blend_whole - ssim_dmg_whole
        
        # 6. Damaged-region-only metrics (ROI)
        dmg_mask_bool = (mask_binary == 1.0)
        clean_roi = clean[dmg_mask_bool].astype(np.float64)
        dmg_roi = damaged[dmg_mask_bool].astype(np.float64)
        raw_roi = restored_raw[dmg_mask_bool].astype(np.float64)
        
        mse_dmg_roi = np.mean((clean_roi - dmg_roi) ** 2)
        mse_res_roi = np.mean((clean_roi - raw_roi) ** 2)
        
        psnr_dmg_roi = 20.0 * np.log10(255.0 / np.sqrt(mse_dmg_roi)) if mse_dmg_roi > 0 else 100.0
        psnr_res_roi = 20.0 * np.log10(255.0 / np.sqrt(mse_res_roi)) if mse_res_roi > 0 else 100.0
        delta_psnr_roi = psnr_res_roi - psnr_dmg_roi
        
        print(f"  Whole-Image PSNR: Damaged={psnr_dmg_whole:.2f} dB -> Raw={psnr_raw_whole:.2f} dB (Delta: {delta_psnr_raw:+.2f} dB) -> Blended={psnr_blend_whole:.2f} dB (Delta: {delta_psnr_blend:+.2f} dB)")
        print(f"  Whole-Image SSIM: Damaged={ssim_dmg_whole:.4f} -> Raw={ssim_raw_whole:.4f} (Delta: {delta_ssim_raw:+.4f}) -> Blended={ssim_blend_whole:.4f} (Delta: {delta_ssim_blend:+.4f})")
        print(f"  Damaged-ROI PSNR: Before={psnr_dmg_roi:.2f} dB -> After={psnr_res_roi:.2f} dB (Delta ROI: {delta_psnr_roi:+.2f} dB)")
        
        results.append({
            "category": cat_name,
            "obj_id": obj_id,
            "title": str(title)[:35],
            "width": w,
            "height": h,
            "mask_coverage_pct": round(mask_coverage, 2),
            "latency_s": round(latency, 2),
            "undamaged_max_diff": undamaged_max_diff,
            "undamaged_mae": undamaged_mae,
            "damaged_psnr_whole": round(psnr_dmg_whole, 2),
            "damaged_ssim_whole": round(ssim_dmg_whole, 4),
            "raw_psnr_whole": round(psnr_raw_whole, 2),
            "raw_ssim_whole": round(ssim_raw_whole, 4),
            "raw_delta_psnr": round(delta_psnr_raw, 2),
            "raw_delta_ssim": round(delta_ssim_raw, 4),
            "blended_psnr_whole": round(psnr_blend_whole, 2),
            "blended_ssim_whole": round(ssim_blend_whole, 4),
            "blended_delta_psnr": round(delta_psnr_blend, 2),
            "blended_delta_ssim": round(delta_ssim_blend, 4),
            "roi_damaged_psnr": round(psnr_dmg_roi, 2),
            "roi_restored_psnr": round(psnr_res_roi, 2),
            "roi_delta_psnr": round(delta_psnr_roi, 2)
        })
        
        # 7. Generate 4-panel visual comparison: [Clean | Damaged | Raw UNet | Mask-Blended]
        scale_vis = 500.0 / h
        vis_w = max(1, int(w * scale_vis))
        vis_h = 500
        
        c_vis = Image.fromarray(clean).resize((vis_w, vis_h), Image.Resampling.BILINEAR)
        d_vis = Image.fromarray(damaged).resize((vis_w, vis_h), Image.Resampling.BILINEAR)
        raw_vis = Image.fromarray(restored_raw).resize((vis_w, vis_h), Image.Resampling.BILINEAR)
        blend_vis = Image.fromarray(restored_blended).resize((vis_w, vis_h), Image.Resampling.BILINEAR)
        
        # 4-panel composite
        comp = Image.new("RGB", (vis_w * 4 + 30, vis_h + 40), color=(25, 25, 25))
        comp.paste(c_vis, (0, 30))
        comp.paste(d_vis, (vis_w + 10, 30))
        comp.paste(raw_vis, (vis_w * 2 + 20, 30))
        comp.paste(blend_vis, (vis_w * 3 + 30, 30))
        
        safe_cat = cat_name.split()[0].lower()
        comp_path = vis_dir / f"compare_4panel_{obj_id}_{safe_cat}.png"
        comp.save(comp_path)
        print(f"  Saved 4-panel visual: {comp_path.name}")
        
    df = pd.DataFrame(results)
    csv_out = out_dir / "mask_blending_metrics.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n[OK] Saved results table to {csv_out.name}")
    
    # Print summary statistics
    print("\n" + "=" * 75)
    print("EXPERIMENTAL SUMMARY: RAW UNET vs. MASK-BLENDED INFERENCE (N=10)")
    print("=" * 75)
    print(f"Damaged Input Whole PSNR:   Mean = {df['damaged_psnr_whole'].mean():.2f} dB | Median = {df['damaged_psnr_whole'].median():.2f} dB")
    print(f"Raw UNet Whole PSNR:        Mean = {df['raw_psnr_whole'].mean():.2f} dB | Delta = {df['raw_delta_psnr'].mean():+.2f} dB")
    print(f"Mask-Blended Whole PSNR:    Mean = {df['blended_psnr_whole'].mean():.2f} dB | Delta = {df['blended_delta_psnr'].mean():+.2f} dB")
    print("-" * 75)
    print(f"Damaged Input Whole SSIM:   Mean = {df['damaged_ssim_whole'].mean():.4f} | Median = {df['damaged_ssim_whole'].median():.4f}")
    print(f"Raw UNet Whole SSIM:        Mean = {df['raw_ssim_whole'].mean():.4f} | Delta = {df['raw_delta_ssim'].mean():+.4f}")
    print(f"Mask-Blended Whole SSIM:    Mean = {df['blended_ssim_whole'].mean():.4f} | Delta = {df['blended_delta_ssim'].mean():+.4f}")
    print("-" * 75)
    print(f"Damaged Region (ROI) PSNR:  Before = {df['roi_damaged_psnr'].mean():.2f} dB -> After = {df['roi_restored_psnr'].mean():.2f} dB (Delta = {df['roi_delta_psnr'].mean():+.2f} dB)")
    print(f"Undamaged Preservation Error: Max Pixel Diff = {df['undamaged_max_diff'].max()} (100% exact preservation)")
    print("=" * 75)

if __name__ == "__main__":
    run_mask_blending_experiment()

