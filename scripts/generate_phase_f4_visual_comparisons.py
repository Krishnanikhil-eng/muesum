import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

from inference import restore_artwork_pipeline, ModelManager
from utils import ensure_dir, psnr, safe_ssim

SELECTED_ARTWORKS = [
    # Paintings (2)
    (48953, "Paintings", "Landscape with Trees (Ni Zan)"),
    (437468, "Paintings", "The Portico of a Country Mansion"),
    # Drawings (2)
    (336663, "Drawings", "Allegorical Figure of Painting (Guercino)"),
    (427650, "Drawings", "Sculpture Designs for Saint Peter"),
    # Photographs (1)
    (283234, "Photographs", "Study of a Sculpture (Albumen)"),
    # Printed textiles (2)
    (157559, "Textiles-Printed", "Textile fragment (Calico Floral)"),
    (452913, "Textiles-Printed", "Textile fragment (Block Print)"),
    # Woven textiles (2)
    (451019, "Textiles-Woven", "Woven Tapestry Fragment"),
    (329977, "Textiles-Woven", "Fragment of a Woman (Coptic)"),
    # Sculptures (2)
    (466379, "Sculptures", "Sculpture of an Enthroned King"),
    (471850, "Sculptures", "Sculpture of Moses with Tablets"),
    # Bronze objects (2)
    (252965, "Bronzes", "Bronze handle with dogs flanking palmette"),
    (196488, "Bronzes", "Bronze statuette"),
    # Manuscripts and books (3)
    (451301, "Manuscripts", "Design for a Cup to Serve Wine (Codex)"),
    (354833, "Books", "The Works of Publius Virgilius Maro"),
    (680296, "Books", "Ein new getruckt model Büchli (Lace Book)")
]

def generate_comparisons():
    print("=" * 80)
    print("PHASE F.4: GENERATING 5-PANEL VISUAL RESTORATION ASSESSMENTS (16 ARTWORKS)")
    print("=" * 80)

    clean_dir = root_dir / "data" / "clean"
    damaged_dir = root_dir / "data" / "damaged" / "test"
    out_dir = root_dir / "reports" / "phase_f4_visual_comparisons"
    ensure_dir(str(out_dir))

    # Pre-warm ModelManager
    ModelManager.get_model()

    results = []

    for idx, (obj_id, category, title) in enumerate(SELECTED_ARTWORKS, 1):
        t0 = time.time()
        c_path = clean_dir / f"{obj_id}.jpg"
        d_path = damaged_dir / f"{obj_id}.jpg"
        m_path = damaged_dir / f"{obj_id}_mask.png"

        assert c_path.exists(), f"Missing clean image for {obj_id}"
        assert d_path.exists(), f"Missing damaged image for {obj_id}"
        assert m_path.exists(), f"Missing mask image for {obj_id}"

        # Run mask-guided restoration
        res = restore_artwork_pipeline(
            damaged_image=d_path,
            damage_mask=m_path,
            blend_mode="mask_guided",
            max_dimension=1024,
            return_diagnostics=True
        )

        clean_im = Image.open(c_path).convert("RGB")
        dmg_im = Image.open(d_path).convert("RGB")
        orig_w, orig_h = clean_im.size

        restored_blended = res["restored_image"]
        raw_unet = res["raw_unet_image"]
        mask_preview = res["mask_preview"]
        diag = res["diagnostics"]

        # Compute accurate metrics against clean ground truth
        clean_np = np.array(clean_im)
        damaged_np = np.array(dmg_im)

        p_dmg = psnr(clean_np, damaged_np)
        s_dmg = safe_ssim(clean_np, damaged_np)
        p_raw = psnr(clean_np, raw_unet)
        s_raw = safe_ssim(clean_np, raw_unet)
        p_blend = psnr(clean_np, restored_blended)
        s_blend = safe_ssim(clean_np, restored_blended)

        delta_psnr = p_blend - p_dmg
        delta_ssim = s_blend - s_dmg

        # Mask ROI PSNR
        mask_binary = (np.array(Image.open(m_path).convert("L")) > 127)
        if np.any(mask_binary):
            p_dmg_roi = psnr(clean_np[mask_binary], damaged_np[mask_binary])
            p_res_roi = psnr(clean_np[mask_binary], restored_blended[mask_binary])
            delta_roi_psnr = p_res_roi - p_dmg_roi
        else:
            p_dmg_roi, p_res_roi, delta_roi_psnr = 0, 0, 0

        # Create standardized 5-panel composite
        # Target height: 420 px
        panel_h = 420
        scale = panel_h / float(orig_h)
        panel_w = max(10, int(round(orig_w * scale)))

        p1 = clean_im.resize((panel_w, panel_h), Image.Resampling.BILINEAR)
        p2 = dmg_im.resize((panel_w, panel_h), Image.Resampling.BILINEAR)
        p3 = mask_preview.resize((panel_w, panel_h), Image.Resampling.BILINEAR)
        p4 = Image.fromarray(raw_unet).resize((panel_w, panel_h), Image.Resampling.BILINEAR)
        p5 = Image.fromarray(restored_blended).resize((panel_w, panel_h), Image.Resampling.BILINEAR)

        header_h = 45
        footer_h = 35
        gap = 10
        total_w = panel_w * 5 + gap * 6
        total_h = panel_h + header_h + footer_h

        comp = Image.new("RGB", (total_w, total_h), color=(24, 26, 27))
        draw = ImageDraw.Draw(comp)

        # Paste panels
        x_offsets = [gap + i * (panel_w + gap) for i in range(5)]
        comp.paste(p1, (x_offsets[0], header_h))
        comp.paste(p2, (x_offsets[1], header_h))
        comp.paste(p3, (x_offsets[2], header_h))
        comp.paste(p4, (x_offsets[3], header_h))
        comp.paste(p5, (x_offsets[4], header_h))

        # Panel Headers
        headers = [
            "1. Clean Reference",
            "2. Damaged Input",
            "3. Damage Mask Overlay",
            "4. Raw UNet Output",
            "5. Mask-Blended Output"
        ]
        for i, text in enumerate(headers):
            draw.text((x_offsets[i] + 4, 14), text, fill=(230, 230, 230))

        # Footer banner
        footer_text = (
            f"ID {obj_id} | {category} | {title} | Coverage: {diag['mask_coverage_pct']:.1f}% | "
            f"PSNR: {p_dmg:.2f} -> {p_blend:.2f} dB (Δ {delta_psnr:+.2f} dB) | "
            f"ROI Δ: {delta_roi_psnr:+.2f} dB | Bit-Exact: {diag['is_bit_exact']} (MAE: {diag['undamaged_mae']:.6f})"
        )
        draw.text((gap + 4, total_h - footer_h + 8), footer_text, fill=(180, 210, 240))

        comp_path = out_dir / f"compare_5panel_{obj_id}.png"
        comp.save(comp_path, format="PNG")

        elapsed = time.time() - t0
        print(f"[{idx:2d}/16] Saved 5-panel: {comp_path.name} | ΔPSNR: {delta_psnr:+.2f} dB | ROI Δ: {delta_roi_psnr:+.2f} dB | Time: {elapsed:.2f} s")

        results.append({
            "obj_id": obj_id,
            "category": category,
            "title": title,
            "coverage_pct": diag["mask_coverage_pct"],
            "damaged_psnr": round(p_dmg, 2),
            "blended_psnr": round(p_blend, 2),
            "delta_psnr": round(delta_psnr, 2),
            "damaged_ssim": round(s_dmg, 4),
            "blended_ssim": round(s_blend, 4),
            "delta_ssim": round(delta_ssim, 4),
            "roi_damaged_psnr": round(p_dmg_roi, 2),
            "roi_restored_psnr": round(p_res_roi, 2),
            "roi_delta_psnr": round(delta_roi_psnr, 2),
            "undamaged_mae": diag["undamaged_mae"],
            "is_bit_exact": diag["is_bit_exact"]
        })

    # Save summary CSV
    res_df = pd.DataFrame(results)
    csv_path = out_dir / "assessment_metrics.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved assessment metrics CSV to: {csv_path}")

if __name__ == "__main__":
    generate_comparisons()

