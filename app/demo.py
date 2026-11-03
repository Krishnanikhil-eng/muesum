"""
app/demo.py
MUESUM Studio — AI-Based Historical Artwork Restoration
======================================================
Interactive Gradio Web Application featuring:
  - Dual Input Modes: File Upload (Image + Mask) and Interactive Brush Editor.
  - Mask-Guided Blending: 100% bit-exact mathematical preservation of undamaged artwork.
  - Raw UNet Inpainting: Unguided baseline comparison.
  - Interactive Before/After ImageSlider.
  - Real-time Curatorial Diagnostics: Coverage %, Latency, MAE, and Preservation Status.
  - Full-Resolution Restored Artwork Export.

Usage:
    python app/demo.py
    python app/demo.py --ckpt-path checkpoints/baseline_run/best_model.pth --port 7860
"""

import sys
import os
import argparse
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image

# ── Ensure src is in python path ──────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _APP_DIR.parent
_SRC_DIR = _ROOT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# ── Try importing Gradio ──────────────────────────────────────────────────────
try:
    import gradio as gr
except ImportError:
    print(
        "[error] Gradio is not installed.\n"
        "        Install it with: pip install gradio>=4.0.0\n"
    )
    sys.exit(1)

# ── Import inference engine ───────────────────────────────────────────────────
try:
    from inference import (
        restore_artwork_pipeline,
        find_default_checkpoint,
        ModelManager
    )
    INFERENCE_AVAILABLE = True
    INFERENCE_ERROR = ""
except ImportError as e:
    INFERENCE_AVAILABLE = False
    INFERENCE_ERROR = str(e)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Extract inputs from UI components
# ──────────────────────────────────────────────────────────────────────────────
def _extract_inputs(
    active_tab: str,
    upload_img: Optional[Image.Image],
    upload_mask: Optional[Image.Image],
    editor_data: Optional[Dict[str, Any]]
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """
    Standardize inputs depending on whether the user used Tab 1 (File Upload)
    or Tab 2 (Interactive Brush Editor).
    """
    if active_tab == "upload":
        return upload_img, upload_mask

    # Tab 2: Interactive Editor
    if editor_data is None or not isinstance(editor_data, dict):
        return None, None

    bg = editor_data.get("background")
    layers = editor_data.get("layers", [])

    if bg is None:
        return None, None

    # Convert bg to PIL RGB
    if isinstance(bg, np.ndarray):
        bg_pil = Image.fromarray(bg.astype(np.uint8)).convert("RGB")
    elif isinstance(bg, Image.Image):
        bg_pil = bg.convert("RGB")
    else:
        return None, None

    if not layers:
        return bg_pil, None

    # Merge painted brush strokes across all layers into a binary mask
    bg_arr = np.array(bg_pil)
    h, w = bg_arr.shape[:2]
    mask_arr = np.zeros((h, w), dtype=np.uint8)

    for layer in layers:
        if layer is None:
            continue
        l_arr = np.array(layer)
        if l_arr.ndim == 3 and l_arr.shape[2] == 4:  # RGBA
            # Alpha channel > 10 indicates painted damage
            mask_arr = np.maximum(mask_arr, (l_arr[:, :, 3] > 10).astype(np.uint8) * 255)
        elif l_arr.ndim == 2:
            mask_arr = np.maximum(mask_arr, (l_arr > 10).astype(np.uint8) * 255)

    if np.sum(mask_arr > 0) == 0:
        return bg_pil, None

    mask_pil = Image.fromarray(mask_arr, mode="L")
    return bg_pil, mask_pil


# ──────────────────────────────────────────────────────────────────────────────
# Restoration Handler
# ──────────────────────────────────────────────────────────────────────────────
def run_restoration(
    active_tab: str,
    upload_img: Optional[Image.Image],
    upload_mask: Optional[Image.Image],
    editor_data: Optional[Dict[str, Any]],
    blend_mode: str,
    max_dim: int,
    ckpt_override: str
):
    if not INFERENCE_AVAILABLE:
        raise gr.Error(f"Inference engine could not be loaded: {INFERENCE_ERROR}")

    img, mask = _extract_inputs(active_tab, upload_img, upload_mask, editor_data)

    if img is None:
        raise gr.Error("Please upload or provide a damaged artwork image.")

    if blend_mode == "mask_guided" and mask is None:
        if active_tab == "brush":
            raise gr.Error(
                "No painted damage detected!\n"
                "Please use the brush tool in the canvas to paint over the damaged regions before restoring."
            )
        else:
            raise gr.Error(
                "Mask-guided restoration requires a damage mask.\n\n"
                "Please upload a damage mask (white=damage, black=clean), "
                "switch to the 'Interactive Brush Canvas' tab to paint one interactively, "
                "or switch the mode to 'Raw UNet (Unguided Convolutions)'."
            )

    target_ckpt = ckpt_override.strip() if ckpt_override.strip() else None

    try:
        max_dimension = int(max_dim) if int(max_dim) > 0 else None
        res = restore_artwork_pipeline(
            damaged_image=img,
            damage_mask=mask,
            ckpt_path=target_ckpt,
            blend_mode=blend_mode,
            max_dimension=max_dimension,
            return_diagnostics=True
        )
    except Exception as e:
        raise gr.Error(f"Restoration failed: {str(e)}")

    restored_pil = Image.fromarray(res["restored_image"])
    raw_pil = Image.fromarray(res["raw_unet_image"])
    orig_pil = Image.fromarray(res["original_image"])
    preview_pil = res["mask_preview"] or orig_pil
    diag = res["diagnostics"]

    # Save restored image to a persistent temporary file for download
    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    restored_pil.save(tmp_file.name, format="PNG")
    tmp_file.close()

    # Format diagnostic status markdown
    is_exact = diag["is_bit_exact"]
    status_icon = "✅" if is_exact else "⚠️"
    status_title = "100% Bit-Exact Preservation" if is_exact else "Raw UNet (Unguided)"

    diag_md = f"""
### 📊 Restoration Diagnostics
| Diagnostic Metric | Measured Value | Curatorial Status |
| :--- | :---: | :--- |
| **Restoration Mode** | `{diag['blend_mode']}` | {'Mask-Guided Blending' if is_exact else 'Full-Image Inpainting'} |
| **Preservation Status** | **{status_icon} {status_title}** | {'Undamaged pixels are 100% unchanged' if is_exact else 'Global convolution shifts applied'} |
| **Damage Coverage** | `{diag['mask_coverage_pct']:.2f}%` | Area repaired by ResNet34-UNet |
| **Inference Latency** | `{diag['latency_ms']:.1f} ms` | Sequential CPU execution |
| **Undamaged MAE** | `{diag['undamaged_mae']:.6f}` | {'0.000000 (Mathematical bit-exact)' if is_exact else 'Non-zero pixel deviation'} |
| **Max Undamaged Diff** | `{diag['undamaged_max_diff']}` | {'0 (Exact match)' if is_exact else 'Pixel shift'} |
| **Original Resolution**| `{diag['original_size'][0]} × {diag['original_size'][1]} px` | Native artwork dimensions |
| **Neural Eval Size**   | `{diag['eval_size'][0]} × {diag['eval_size'][1]} px` | Aspect-preserving scale |
"""

    # ImageSlider takes (before, after) tuple
    slider_tuple = (orig_pil, restored_pil)

    return (
        slider_tuple,
        restored_pil,
        raw_pil,
        preview_pil,
        diag_md,
        gr.DownloadButton(value=tmp_file.name, visible=True, label="⬇️ Download Restored Artwork (PNG)")
    )


# ──────────────────────────────────────────────────────────────────────────────
# Build Gradio Interface
# ──────────────────────────────────────────────────────────────────────────────
def build_interface(default_ckpt: str = "") -> gr.Blocks:
    ckpt_name = Path(default_ckpt).name if default_ckpt else "None found"
    ckpt_status = (
        f"✅ **Active Checkpoint**: `{ckpt_name}` (`{default_ckpt}`)"
        if default_ckpt
        else "⚠️ **No Checkpoint Found**. Please specify a valid `.pth` checkpoint path below."
    )

    custom_css = """
    .gradio-container { max-width: 1200px !important; }
    .status-badge { font-weight: bold; padding: 4px 8px; border-radius: 4px; }
    """

    with gr.Blocks(title="🏛️ MUESUM Studio — AI Artwork Restoration") as demo:
        # Header
        gr.Markdown("""
# 🏛️ MUESUM Studio — AI Historical Artwork Restoration
### ResNet34-UNet Deep Learning Restoration with Bit-Exact Authentic Preservation
Restore museum artworks degraded by physical stains, mold, scratches, or tears while preserving **100% of authentic surviving artwork**.
        """)

        gr.Markdown(f"> {ckpt_status}")

        active_tab_state = gr.State(value="upload")

        with gr.Row():
            # ── LEFT COLUMN: INPUTS & SETTINGS ────────────────────────────────
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("📁 Upload Image & Mask", id="tab_upload") as tab_upload:
                        gr.Markdown("**Option A**: Upload a damaged artwork image alongside a precomputed damage mask.")
                        upload_img = gr.Image(
                            label="📷 Damaged Artwork (Upload)",
                            type="pil",
                            sources=["upload", "clipboard"]
                        )
                        upload_mask = gr.Image(
                            label="🎭 Damage Mask (White = Damaged, Black = Clean)",
                            type="pil",
                            sources=["upload", "clipboard"]
                        )
                        gr.Markdown("*Mask Polarity: White (255) represents damage to be repaired; Black (0) represents clean artwork.*")

                    with gr.Tab("🖌️ Interactive Brush Canvas", id="tab_brush") as tab_brush:
                        gr.Markdown(
                            "**Option B**: No mask file? Upload your artwork here and **paint directly over blemishes** using the brush tool."
                        )
                        image_editor = gr.ImageEditor(
                            label="Interactive Restoration Canvas",
                            type="pil",
                            brush=gr.Brush(colors=["#ff2222", "#ffffff"], default_size=24),
                            eraser=gr.Eraser()
                        )
                        gr.Markdown("*Draw red/white strokes over stains or cracks. The painted region will be extracted as the damage mask.*")

                # Controls & Settings
                with gr.Accordion("⚙️ Inference & Restoration Settings", open=True):
                    blend_mode = gr.Radio(
                        choices=[
                            ("Mask-Guided Blending (Recommended — 100% Bit-Exact Preservation)", "mask_guided"),
                            ("Raw UNet Inpainting (Unguided Convolutions across whole image)", "raw_unet")
                        ],
                        value="mask_guided",
                        label="Restoration Mode"
                    )
                    max_dim = gr.Dropdown(
                        choices=[
                            ("1024 px (Fast, Standard Evaluation)", 1024),
                            ("1536 px (High-Resolution Detail)", 1536),
                            ("Native Full Resolution (Slower on CPU)", 0)
                        ],
                        value=1024,
                        label="Inference Resolution Scale"
                    )
                    ckpt_override = gr.Textbox(
                        label="Checkpoint Path (.pth override, optional)",
                        placeholder="checkpoints/baseline_run/best_model.pth",
                        value=default_ckpt
                    )

                with gr.Row():
                    restore_btn = gr.Button("🎨 Restore Artwork", variant="primary", scale=2, size="lg")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)

            # ── RIGHT COLUMN: RESULTS & PREVIEWS ──────────────────────────────
            with gr.Column(scale=7):
                gr.Markdown("### 🔍 Restoration Results & Comparisons")

                # Interactive Before / After ImageSlider
                slider_view = gr.ImageSlider(
                    label="Interactive Before / After Comparison",
                    type="pil",
                    show_label=True
                )

                # Tabbed Views for Individual Inspection
                with gr.Tabs():
                    with gr.Tab("✨ Restored Artwork"):
                        restored_view = gr.Image(label="Final Restored Artwork (Mask-Blended)", type="pil", interactive=False)
                    with gr.Tab("🔬 Raw UNet Output"):
                        raw_view = gr.Image(label="Raw UNet (100% Pixels Replaced)", type="pil", interactive=False)
                    with gr.Tab("🎯 Damage Overlay Preview"):
                        overlay_view = gr.Image(label="Damage Mask Overlay (Red = Damaged)", type="pil", interactive=False)

                # Download Button
                download_btn = gr.DownloadButton(
                    label="⬇️ Download Restored Artwork (Full Resolution PNG)",
                    visible=False,
                    variant="primary"
                )

                # Diagnostics Panel
                diag_box = gr.Markdown(
                    """
### 📊 Restoration Diagnostics
*Upload an image and click **Restore Artwork** to see quantitative preservation metrics.*
                    """
                )

        # ── Curatorial Notice & Documentation ─────────────────────────────────
        gr.Markdown("""
---
### 🏛️ Curatorial & Scientific Notice
- **Mathematical Integrity**: In `mask_guided` mode, all undamaged pixels are mathematically locked to their original values ($\text{MAE} = 0.000000$). The neural network is permitted to generate pixels strictly within the area specified by the damage mask.
- **No Automatic Mask Fabrications**: MUESUM never guesses or invents damage masks automatically. Restoration fidelity depends directly on the accuracy of the provided or painted mask.
- **Architecture**: PyTorch ResNet34-UNet trained on 2,000 public-domain Met Museum artworks with simulated physical damage (stains, mold, cracks, scratches).
        """)

        # ── Tab Change Listeners ──────────────────────────────────────────────
        tab_upload.select(fn=lambda: "upload", outputs=[active_tab_state])
        tab_brush.select(fn=lambda: "brush", outputs=[active_tab_state])

        # ── Action Listeners ──────────────────────────────────────────────────
        restore_btn.click(
            fn=run_restoration,
            inputs=[
                active_tab_state,
                upload_img,
                upload_mask,
                image_editor,
                blend_mode,
                max_dim,
                ckpt_override
            ],
            outputs=[
                slider_view,
                restored_view,
                raw_view,
                overlay_view,
                diag_box,
                download_btn
            ]
        )

        def on_clear():
            return (
                None, None, None,
                None, None, None, None,
                "### 📊 Restoration Diagnostics\n*Cleared.*",
                gr.DownloadButton(visible=False)
            )

        clear_btn.click(
            fn=on_clear,
            outputs=[
                upload_img,
                upload_mask,
                image_editor,
                slider_view,
                restored_view,
                raw_view,
                overlay_view,
                diag_box,
                download_btn
            ]
        )

    return demo


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MUESUM Gradio Restoration Studio")
    parser.add_argument("--ckpt-path", type=str, default="", help="Path to model checkpoint (.pth)")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio share link")
    args = parser.parse_args()

    default_ckpt = args.ckpt_path or find_default_checkpoint()
    if default_ckpt:
        print(f"[MUESUM Studio] Active Checkpoint: {default_ckpt}")
    else:
        print("[MUESUM Studio] Warning: No checkpoint found automatically. Train model or specify --ckpt-path.")

    studio = build_interface(default_ckpt=default_ckpt)
    print(f"\n[MUESUM Studio] Launching interface on http://localhost:{args.port} ...")
    studio.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 1200px !important; }"
    )
