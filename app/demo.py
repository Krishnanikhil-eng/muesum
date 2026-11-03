"""
app/demo.py
MUESUM Studio — AI-Based Historical Artwork Restoration
======================================================
Interactive Gradio Web Application featuring:
  - Dual Architecture Support: Experimental Texture Model (Composite Loss) and Baseline (Epoch 8).
  - Dual Input Modes: File Upload (Image + Mask) and Interactive Brush Editor.
  - Interior Boundary Feathering: Smooth, natural transition strictly inside the damage mask.
  - 100% Bit-Exact Preservation: Surviving authentic pixels remain completely unaltered.
  - Interactive Before/After ImageSlider.
  - Real-time Curatorial Diagnostics: Coverage %, Latency, MAE, MaxDiff, and Preservation Status.
  - Full-Resolution Restored Artwork (PNG) & Provenance Diagnostics (JSON) Export.

Usage:
    python app/demo.py
    python app/demo.py --port 7860
"""

import sys
import os
import argparse
import tempfile
import json
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


CHECKPOINT_OPTIONS = {
    "Experimental Texture Model (Composite Loss — Recommended)": str(
        _ROOT_DIR / "checkpoints" / "experimental_texture_model" / "best_model.pth"
    ),
    "Baseline ResNet34-UNet (Epoch 8 Checkpoint)": str(
        _ROOT_DIR / "checkpoints" / "baseline_run" / "best_model.pth"
    ),
}


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
    composite_mask = np.zeros((h, w), dtype=bool)

    for layer in layers:
        if layer is None:
            continue
        if isinstance(layer, Image.Image):
            layer_arr = np.array(layer.convert("RGBA"))
        elif isinstance(layer, np.ndarray):
            layer_arr = layer
        else:
            continue

        # In Gradio ImageEditor, drawn strokes have positive alpha (> 10)
        if layer_arr.shape[-1] == 4:
            alpha = layer_arr[..., 3]
            composite_mask |= (alpha > 10)
        elif layer_arr.ndim == 2:
            composite_mask |= (layer_arr > 10)

    if not np.any(composite_mask):
        return bg_pil, None

    # Convert boolean mask to PIL Image (255 = damage, 0 = clean)
    mask_uint8 = (composite_mask.astype(np.uint8)) * 255
    mask_pil = Image.fromarray(mask_uint8, mode="L")
    return bg_pil, mask_pil


# ──────────────────────────────────────────────────────────────────────────────
# Restoration Handler
# ──────────────────────────────────────────────────────────────────────────────
def run_restoration(
    active_tab: str,
    upload_img: Optional[Image.Image],
    upload_mask: Optional[Image.Image],
    editor_data: Optional[Dict[str, Any]],
    model_choice: str,
    blend_mode: str,
    feather_mode: str,
    feather_radius: int,
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

    # Resolve target checkpoint
    if ckpt_override and ckpt_override.strip():
        target_ckpt = ckpt_override.strip()
    else:
        target_ckpt = CHECKPOINT_OPTIONS.get(model_choice, find_default_checkpoint())

    if not Path(target_ckpt).exists():
        # Fallback to default checkpoint
        target_ckpt = find_default_checkpoint()

    try:
        max_dimension = int(max_dim) if int(max_dim) > 0 else None
        res = restore_artwork_pipeline(
            damaged_image=img,
            damage_mask=mask,
            ckpt_path=target_ckpt,
            blend_mode=blend_mode,
            feather_mode=feather_mode,
            feather_radius=int(feather_radius),
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

    # Save restored image to a persistent temporary file for PNG download
    tmp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    restored_pil.save(tmp_png.name, format="PNG")
    tmp_png.close()

    # Save diagnostics JSON for metadata download
    tmp_json = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
    export_payload = {
        "muesum_version": "2.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_architecture": "ResNet34-UNet",
        "model_checkpoint": str(target_ckpt),
        "model_type": model_choice,
        "parameters": {
            "blend_mode": blend_mode,
            "feather_mode": feather_mode,
            "feather_radius": int(feather_radius),
            "max_dimension": max_dimension
        },
        "diagnostics": diag
    }
    json.dump(export_payload, tmp_json, indent=2)
    tmp_json.close()

    # Format diagnostic status markdown
    is_exact = diag["is_bit_exact"]
    status_icon = "✅" if is_exact else "⚠️"
    status_title = "100% Bit-Exact Preservation" if is_exact else "Raw UNet (Unguided)"

    diag_md = f"""
### 📊 Restoration Diagnostics & Curatorial Report
| Diagnostic Metric | Measured Value | Curatorial Status |
| :--- | :---: | :--- |
| **Active Model** | `{Path(target_ckpt).parent.name}` | {model_choice.split('(')[0].strip()} |
| **Restoration Mode** | `{diag['blend_mode']}` | {'Mask-Guided Blending' if is_exact else 'Full-Image Inpainting'} |
| **Seam Feathering** | `{diag.get('feather_mode', 'none')} (r={diag.get('feather_radius', 0)}px)` | {'Interior Smoothstep Seam' if diag.get('feather_mode') == 'feathered' else 'Hard Boundary'} |
| **Preservation Status** | **{status_icon} {status_title}** | {'Undamaged pixels are 100% untouched' if is_exact else 'Global convolution shifts applied'} |
| **Damage Coverage** | `{diag['mask_coverage_pct']:.2f}%` | Area repaired by neural network |
| **Inference Latency** | `{diag['latency_ms']:.1f} ms` | Sequential CPU execution |
| **Undamaged MAE** | `{diag['undamaged_mae']:.6f}` | {'0.000000 (Mathematical bit-exact)' if is_exact else 'Non-zero pixel deviation'} |
| **Max Undamaged Diff** | `{diag['undamaged_max_diff']}` | {'0 (Exact match)' if is_exact else 'Pixel shift'} |
| **Original Resolution**| `{diag['original_size'][0]} × {diag['original_size'][1]} px` | Native artwork dimensions |
| **Neural Scale**       | `{diag['eval_size'][0]} × {diag['eval_size'][1]} px` | Aspect-preserving evaluation |
"""

    slider_tuple = (orig_pil, restored_pil)

    return (
        slider_tuple,
        restored_pil,
        raw_pil,
        preview_pil,
        diag_md,
        gr.DownloadButton(value=tmp_png.name, visible=True, label="⬇️ Download Restored Artwork (PNG)"),
        gr.DownloadButton(value=tmp_json.name, visible=True, label="📄 Export Curatorial Diagnostics (JSON)")
    )


# ──────────────────────────────────────────────────────────────────────────────
# Build Gradio Interface
# ──────────────────────────────────────────────────────────────────────────────
def build_interface(default_ckpt: str = "") -> gr.Blocks:
    ckpt_name = Path(default_ckpt).name if default_ckpt else "None found"

    with gr.Blocks(title="🏛️ MUESUM Studio — AI Historical Artwork Restoration") as demo:
        # Header
        gr.Markdown("""
# 🏛️ MUESUM Studio — AI Historical Artwork Restoration
### ResNet34-UNet Deep Learning Restoration with 100% Bit-Exact Authentic Preservation
Restore museum-grade cultural heritage artworks degraded by physical stains, craquelure, mold, scratches, or tears.
        """)

        active_tab_state = gr.State(value="upload")

        with gr.Row():
            # ── LEFT COLUMN: INPUTS & CONTROLS ────────────────────────────────
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
                        gr.Markdown("*Mask Polarity: White (255) represents damage to repair; Black (0) represents authentic artwork.*")

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
                        gr.Markdown("*Draw red/white brush strokes over damaged areas. Painted strokes become the neural repair mask.*")

                # Controls & Settings
                with gr.Accordion("⚙️ Model Architecture & Inpainting Settings", open=True):
                    model_choice = gr.Radio(
                        choices=[
                            "Experimental Texture Model (Composite Loss — Recommended)",
                            "Baseline ResNet34-UNet (Epoch 8 Checkpoint)"
                        ],
                        value="Experimental Texture Model (Composite Loss — Recommended)",
                        label="🧠 Neural Model Selection"
                    )

                    blend_mode = gr.Radio(
                        choices=[
                            ("Mask-Guided Blending (Recommended — 100% Bit-Exact Preservation)", "mask_guided"),
                            ("Raw UNet Inpainting (Unguided Convolutions across whole image)", "raw_unet")
                        ],
                        value="mask_guided",
                        label="Restoration Mode"
                    )

                    with gr.Row():
                        feather_mode = gr.Radio(
                            choices=[
                                ("Feathered Interior (Seamless)", "feathered"),
                                ("Hard Cutoff (Bit-Exact Step)", "hard")
                            ],
                            value="feathered",
                            label="🎨 Boundary Seam Blending"
                        )
                        feather_radius = gr.Slider(
                            minimum=1,
                            maximum=15,
                            value=3,
                            step=1,
                            label="Feather Radius (px)"
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
                        value=""
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

                with gr.Row():
                    download_btn = gr.DownloadButton(
                        label="⬇️ Download Restored Artwork (Full Resolution PNG)",
                        visible=False,
                        variant="primary",
                        scale=1
                    )
                    download_json_btn = gr.DownloadButton(
                        label="📄 Export Curatorial Diagnostics (JSON)",
                        visible=False,
                        variant="secondary",
                        scale=1
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
### 🏛️ Curatorial & Scientific Standards
- **Mathematical Integrity**: In `mask_guided` mode, all authentic undamaged substrate is mathematically locked to original pixel values ($\text{MAE} = 0.000000, \text{MaxDiff} = 0$).
- **Interior Seam Feathering**: Softens transitions strictly inside the damage mask using Euclidean distance transform, completely eliminating border halos without contaminating authentic paint.
- **Museum Provenance**: Diagnostics export contains SHA metadata, inference scale, latency, and preservation metrics for conservation archives.
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
                model_choice,
                blend_mode,
                feather_mode,
                feather_radius,
                max_dim,
                ckpt_override
            ],
            outputs=[
                slider_view,
                restored_view,
                raw_view,
                overlay_view,
                diag_box,
                download_btn,
                download_json_btn
            ]
        )

        def on_clear():
            return (
                None, None, None,
                None, None, None, None,
                "### 📊 Restoration Diagnostics\n*Cleared.*",
                gr.DownloadButton(visible=False),
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
                download_btn,
                download_json_btn
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
