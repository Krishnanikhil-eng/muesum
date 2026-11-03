import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'app')
from demo import run_restoration, _extract_inputs
import gradio as gr

def test_stage2_workflows():
    print("=" * 70)
    print("STAGE 2: REAL GRADIO APPLICATION TESTING (WORKFLOWS A, B, C)")
    print("=" * 70)

    dmg_path = "data/damaged/test/247010.jpg"
    mask_path = "data/damaged/test/247010_mask.png"
    dmg_pil = Image.open(dmg_path).convert("RGB")
    mask_pil = Image.open(mask_path).convert("L")

    # ──────────────────────────────────────────────────────────────────
    # WORKFLOW A: Precomputed mask
    # ──────────────────────────────────────────────────────────────────
    print("\n--- WORKFLOW A: Precomputed Mask ---")
    slider_tuple, restored_pil, raw_pil, preview_pil, diag_md, download_btn = run_restoration(
        active_tab="upload",
        upload_img=dmg_pil,
        upload_mask=mask_pil,
        editor_data=None,
        blend_mode="mask_guided",
        max_dim=1024,
        ckpt_override=""
    )

    print("Slider tuple generated:", isinstance(slider_tuple, tuple), len(slider_tuple))
    print("Restored image dimensions:", restored_pil.size)
    print("Raw UNet image dimensions:", raw_pil.size)
    print("Preview image dimensions:", preview_pil.size)
    # In Gradio 6, DownloadButton value can be a dict {'path': '...', ...} or str
    download_path = download_btn.value['path'] if isinstance(download_btn.value, dict) else download_btn.value
    print("Download button file path:", download_path)
    assert os.path.exists(download_path), "Downloadable PNG does not exist!"
    downloaded_im = Image.open(download_path)
    print("Downloaded PNG verified valid, dimensions:", downloaded_im.size)
    print("Diagnostics snippet:\n", "\n".join(diag_md.strip().split("\n")[:8]))
    print("WORKFLOW A PASSED SUCCESSFULLY!")

    # ──────────────────────────────────────────────────────────────────
    # WORKFLOW B: Manually painted mask (Interactive Brush)
    # ──────────────────────────────────────────────────────────────────
    print("\n--- WORKFLOW B: Manually Painted Mask ---")
    # Simulate user drawing brush strokes on a canvas in Gradio ImageEditor
    w, h = dmg_pil.size
    # Create an RGBA layer with painted brush strokes (e.g. red rectangle in center)
    layer_arr = np.zeros((h, w, 4), dtype=np.uint8)
    layer_arr[h//3:2*h//3, w//3:2*w//3] = [255, 34, 34, 255] # Painted red with full alpha
    layer_pil = Image.fromarray(layer_arr, mode="RGBA")

    editor_data = {
        "background": dmg_pil,
        "layers": [layer_pil],
        "composite": dmg_pil
    }

    slider_tuple_b, restored_b, raw_b, preview_b, diag_b, download_btn_b = run_restoration(
        active_tab="brush",
        upload_img=None,
        upload_mask=None,
        editor_data=editor_data,
        blend_mode="mask_guided",
        max_dim=1024,
        ckpt_override=""
    )

    print("Workflow B Restored dimensions:", restored_b.size)
    download_path_b = download_btn_b.value['path'] if isinstance(download_btn_b.value, dict) else download_btn_b.value
    print("Workflow B Download button path:", download_path_b)
    assert os.path.exists(download_path_b)
    print("Workflow B Diagnostics snippet:\n", "\n".join(diag_b.strip().split("\n")[:8]))
    print("WORKFLOW B PASSED SUCCESSFULLY!")

    # ──────────────────────────────────────────────────────────────────
    # WORKFLOW C: Invalid inputs & Error handling
    # ──────────────────────────────────────────────────────────────────
    print("\n--- WORKFLOW C: Invalid Inputs & Error Handling ---")

    # C.1 Missing artwork input
    try:
        run_restoration(
            active_tab="upload",
            upload_img=None,
            upload_mask=mask_pil,
            editor_data=None,
            blend_mode="mask_guided",
            max_dim=1024,
            ckpt_override=""
        )
        assert False, "Failed to catch missing artwork input!"
    except gr.Error as e:
        print("C.1 Missing artwork caught as expected:", str(e))

    # C.2 Missing mask in mask_guided mode (Upload tab)
    try:
        run_restoration(
            active_tab="upload",
            upload_img=dmg_pil,
            upload_mask=None,
            editor_data=None,
            blend_mode="mask_guided",
            max_dim=1024,
            ckpt_override=""
        )
        assert False, "Failed to catch missing mask in mask_guided mode!"
    except gr.Error as e:
        print("C.2 Missing mask caught as expected:", str(e)[:70], "...")

    # C.3 Empty mask in interactive brush mode (user clicked restore without drawing)
    try:
        empty_editor = {
            "background": dmg_pil,
            "layers": [],
            "composite": dmg_pil
        }
        run_restoration(
            active_tab="brush",
            upload_img=None,
            upload_mask=None,
            editor_data=empty_editor,
            blend_mode="mask_guided",
            max_dim=1024,
            ckpt_override=""
        )
        assert False, "Failed to catch empty brush mask!"
    except gr.Error as e:
        print("C.3 Empty brush mask caught as expected:", str(e))

    # C.4 Dimension mismatch between image and mask
    try:
        bad_mask = mask_pil.resize((w // 2, h // 2))
        run_restoration(
            active_tab="upload",
            upload_img=dmg_pil,
            upload_mask=bad_mask,
            editor_data=None,
            blend_mode="mask_guided",
            max_dim=1024,
            ckpt_override=""
        )
        assert False, "Failed to catch dimension mismatch!"
    except gr.Error as e:
        print("C.4 Dimension mismatch caught as expected:", str(e))

    # C.5 Incorrect mask format (NaN mask passed into pipeline)
    from inference import restore_artwork_pipeline
    try:
        nan_mask = np.zeros((h, w), dtype=np.float32)
        nan_mask[10, 10] = np.nan
        restore_artwork_pipeline(
            damaged_image=dmg_pil,
            damage_mask=nan_mask,
            blend_mode="mask_guided"
        )
        assert False, "Failed to catch NaN mask in pipeline!"
    except ValueError as e:
        print("C.5 NaN mask caught as expected:", str(e))

    print("\nALL WORKFLOWS A, B, AND C VERIFIED CLEANLY!")

if __name__ == "__main__":
    test_stage2_workflows()
