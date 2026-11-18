"""
app/demo.py
MUESUM Studio — AI-Based Historical Artwork Restoration
======================================================
Interactive Gradio Web Application featuring:
  - Dual Restoration Modes:
      1. Manual Mode (Interactive Brush Annotation & Curated Mask Upload) [Verified & Recommended]
      2. Automatic Detection Mode (Heuristic Multi-Scale CV Damage Detector) [Experimental]
  - Seamless Mask Correction: One-click transfer of auto-generated masks to Interactive Brush Canvas.
  - Interactive Before/After ImageSlider.
  - 100% Bit-Exact Authentic Preservation on Surviving Canvas (MAE = 0.000000, MaxDiff = 0).
  - Interior Boundary Feathering: Smooth, natural transition strictly inside the damage mask.
  - Real-time Curatorial Diagnostics: Coverage %, Latency, MAE, MaxDiff, and Preservation Status.
  - Full-Resolution Restored Artwork (PNG) & Provenance Diagnostics (JSON) Export.
  - Preloaded Demonstration Scenarios (Tests A, B, C, D, E).

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
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Ensure src is in python path ──────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _APP_DIR.parent
_SRC_DIR = _ROOT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

try:
    from dotenv import load_dotenv
    _env_file = _ROOT_DIR / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
    else:
        load_dotenv()
except ImportError:
    pass

# ── Try importing Gradio ──────────────────────────────────────────────────────
try:
    import gradio as gr
except ImportError:
    print(
        "[error] Gradio is not installed.\n"
        "        Install it with: pip install gradio>=4.0.0\n"
    )
    sys.exit(1)

# ── Import inference engine, config, and damage detection ─────────────────────
try:
    import config as muesum_cfg
except ImportError:
    muesum_cfg = None

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

try:
    from damage_detection import detect_damage_mask, create_mask_overlay
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False

try:
    from intelligent_pipeline import intelligent_restore
    INTELLIGENT_PIPELINE_AVAILABLE = True
    INTELLIGENT_PIPELINE_ERROR = ""
except ImportError as e:
    INTELLIGENT_PIPELINE_AVAILABLE = False
    INTELLIGENT_PIPELINE_ERROR = str(e)


if muesum_cfg is not None:
    CHECKPOINT_OPTIONS = {
        "Experimental Texture Model (Composite Loss — Recommended)": str(muesum_cfg.EXPERIMENTAL_CHECKPOINT_PATH),
        "Baseline ResNet34-UNet (Epoch 8 Checkpoint)": str(muesum_cfg.BASELINE_CHECKPOINT_PATH),
    }
else:
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
    or Tab 2 (Interactive Brush Editor). Kept for backward compatibility.
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

    # Convert bg to PIL RGB safely
    bg_pil = None
    if isinstance(bg, str):
        try:
            bg_pil = Image.open(bg).convert("RGB")
        except Exception:
            return None, None
    elif isinstance(bg, np.ndarray):
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
        layer_arr = None
        if isinstance(layer, str):
            try:
                layer_arr = np.array(Image.open(layer).convert("RGBA"))
            except Exception:
                continue
        elif isinstance(layer, Image.Image):
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


def _resolve_unified_inputs(
    mode_selection: str,
    active_tab: str,
    upload_img: Optional[Image.Image],
    upload_mask: Optional[Image.Image],
    editor_data: Optional[Dict[str, Any]],
    auto_img: Optional[Image.Image],
    auto_mask_arr: Optional[np.ndarray]
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """
    Resolves artwork image and damage mask across Manual and Automatic modes.
    Includes smart fallbacks to ensure an uploaded artwork is never lost across tab changes.
    """
    img, mask = None, None
    if "Manual" in mode_selection:
        img, mask = _extract_inputs(active_tab, upload_img, upload_mask, editor_data)
    else:
        # Automatic mode
        if auto_img is not None:
            img = auto_img
            if auto_mask_arr is not None:
                mask = Image.fromarray(auto_mask_arr.astype(np.uint8), mode="L")

    # Smart fallback: if primary mode returned no image, check other slots
    if img is None:
        if editor_data and isinstance(editor_data, dict) and editor_data.get("background"):
            img, mask = _extract_inputs("brush", None, None, editor_data)
        elif upload_img is not None:
            img = upload_img
            mask = upload_mask
        elif auto_img is not None:
            img = auto_img
            if auto_mask_arr is not None:
                mask = Image.fromarray(auto_mask_arr.astype(np.uint8), mode="L")

    return img, mask


# ──────────────────────────────────────────────────────────────────────────────
# Automatic Damage Detection & Correction Handlers
# ──────────────────────────────────────────────────────────────────────────────
def on_generate_auto_mask(
    image: Optional[Image.Image],
    profile: str,
    sensitivity: float
):
    """
    Runs computer-vision damage detection and updates preview.
    """
    if image is None:
        raise gr.Error("Please upload an artwork image first before running automatic damage detection.")

    if not DETECTION_AVAILABLE:
        raise gr.Error("Damage detection module not available.")

    mask_u8, overlay_pil, diag = detect_damage_mask(
        image=image,
        mode=profile,
        sensitivity=float(sensitivity)
    )

    coverage = diag["mask_coverage_pct"]
    total_px = diag["total_damaged_pixels"]
    thresh = diag["threshold_value"]

    info_md = f"""
### 🔍 Automatic Damage Detection Summary
- **Profile Selected**: `{profile}`
- **Sensitivity Level**: `{sensitivity}` (Threshold: `{thresh}`)
- **Detected Damage Area**: **`{coverage:.2f}%`** (`{total_px:,}` pixels)
- **Status**: ⚠️ **Heuristic Hypothesis Generated**
- **Curatorial Recommendation**: *Review detected regions in the overlay preview. If natural brushstrokes were mistakenly detected, click **'✏️ Load Mask into Brush Editor for Correction'** before proceeding to restoration.*
"""
    return mask_u8, overlay_pil, Image.fromarray(mask_u8), info_md


def on_transfer_to_editor(
    image: Optional[Image.Image],
    mask_u8: Optional[np.ndarray]
):
    """
    Transfers the automatic mask into the Interactive Brush Canvas as an editable layer.
    """
    if image is None:
        raise gr.Error("No artwork image to transfer.")
    if mask_u8 is None:
        raise gr.Error("No detected mask found. Please click 'Detect Damage Mask' first.")

    img_rgb = image.convert("RGB")
    w, h = img_rgb.size

    # Create RGBA layer with painted red strokes where mask == 255
    layer_arr = np.zeros((h, w, 4), dtype=np.uint8)
    damaged = (mask_u8 > 127)
    layer_arr[damaged] = [255, 30, 40, 220]  # Red brush strokes
    layer_pil = Image.fromarray(layer_arr, mode="RGBA")

    editor_payload = {
        "background": img_rgb,
        "layers": [layer_pil],
        "composite": None
    }

    status_msg = "✅ **Mask transferred to Brush Editor!** You can now switch to the Brush Canvas tab to paint or erase."
    return editor_payload, status_msg


def on_detect_canvas(editor_data):
    """
    1-Click Damage Detection directly on the interactive brush canvas.
    Extracts the uploaded background image, detects tears/scratches,
    and returns a red painted layer for immediate visual inspection and editing.
    """
    if not editor_data or not isinstance(editor_data, dict):
        raise gr.Error("Please upload or drop an artwork image into the canvas first.")

    bg = editor_data.get("background")
    if not bg:
        raise gr.Error("Please upload or drop an artwork image into the canvas first.")

    if isinstance(bg, str):
        try:
            bg_pil = Image.open(bg).convert("RGB")
            bg_val = bg
        except Exception as e:
            raise gr.Error(f"Could not open image file: {e}")
    elif isinstance(bg, Image.Image):
        bg_pil = bg.convert("RGB")
        bg_val = bg_pil
    elif isinstance(bg, np.ndarray):
        bg_pil = Image.fromarray(bg.astype(np.uint8)).convert("RGB")
        bg_val = bg_pil
    else:
        raise gr.Error("Could not read canvas image.")

    if not DETECTION_AVAILABLE:
        raise gr.Error("Damage detection module is currently unavailable.")

    mask_u8, _, diag = detect_damage_mask(image=bg_pil, mode="general", sensitivity=3.0)
    w, h = bg_pil.size

    # Create RGBA layer with red marks
    layer_arr = np.zeros((h, w, 4), dtype=np.uint8)
    layer_arr[mask_u8 > 127] = [255, 30, 40, 220]
    layer_pil = Image.fromarray(layer_arr, mode="RGBA")

    tmp_layer = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    layer_pil.save(tmp_layer.name, format="PNG")
    tmp_layer.close()

    coverage = diag.get("mask_coverage_pct", 0.0)
    msg = f"⚡ **Detected {coverage:.1f}% damage!** Tears and scratches are marked in red. You can paint more, erase, or click **'🎨 Restore Artwork'**."

    return {
        "background": bg_val,
        "layers": [tmp_layer.name],
        "composite": None
    }, msg


def on_clear_canvas_brush(editor_data):
    """Clears the painted brush layer while preserving the uploaded artwork."""
    if not editor_data or not isinstance(editor_data, dict):
        return None, "Brush cleared."
    bg = editor_data.get("background")
    return {"background": bg, "layers": [], "composite": None}, "Brush cleared. Artwork preserved."


def on_preview_mask_unified(
    mode_selection: str,
    active_tab: str,
    upload_img: Optional[Image.Image],
    upload_mask: Optional[Image.Image],
    editor_data: Optional[Dict[str, Any]],
    auto_img: Optional[Image.Image],
    auto_mask_arr: Optional[np.ndarray]
):
    """
    Generates damage mask overlay preview before inpainting.
    """
    img, mask = _resolve_unified_inputs(
        mode_selection, active_tab, upload_img, upload_mask, editor_data, auto_img, auto_mask_arr
    )
    if img is None:
        raise gr.Error("Please provide an artwork image first.")
    if mask is None:
        raise gr.Error("No damage mask detected. Please paint over damage or generate an automatic mask.")

    img_arr = np.array(img.convert("RGB"))
    mask_arr = np.array(mask.convert("L"))

    if img_arr.shape[:2] != mask_arr.shape[:2]:
        raise gr.Error(f"Dimension mismatch: Artwork is {img_arr.shape[:2]}, but mask is {mask_arr.shape[:2]}.")

    coverage = float((np.sum(mask_arr > 127) / mask_arr.size) * 100.0)
    overlay_arr = create_mask_overlay(img_arr, mask_arr)
    overlay_pil = Image.fromarray(overlay_arr)

    info_md = f"""
### 🎯 Damage Mask Verification Preview
- **Active Image Dimensions**: `{img.size[0]} × {img.size[1]} px`
- **Damage Mask Coverage**: **`{coverage:.2f}%`** (`{np.sum(mask_arr > 127):,}` pixels)
- **Authentic Canvas Preserved**: **`{100.0 - coverage:.2f}%`**
- **Mask Alignment**: Verified matching dimensions
"""
    return overlay_pil, Image.fromarray(mask_arr), info_md


# ──────────────────────────────────────────────────────────────────────────────
# Restoration Handler
# ──────────────────────────────────────────────────────────────────────────────
def run_restoration_unified(
    mode_selection: str,
    active_tab: str,
    upload_img: Optional[Image.Image],
    upload_mask: Optional[Image.Image],
    editor_data: Optional[Dict[str, Any]],
    auto_img: Optional[Image.Image],
    auto_mask_arr: Optional[np.ndarray],
    model_choice: str,
    blend_mode: str,
    feather_mode: str,
    feather_radius: int,
    max_dim: int,
    ckpt_override: str
):
    if not INFERENCE_AVAILABLE:
        raise gr.Error(f"Inference engine could not be loaded: {INFERENCE_ERROR}")

    img, mask = _resolve_unified_inputs(
        mode_selection, active_tab, upload_img, upload_mask, editor_data, auto_img, auto_mask_arr
    )

    if img is None:
        raise gr.Error("Please upload or drop an artwork image to restore.")

    # Smart Damage Handling: If mask-guided mode but no mask drawn/provided,
    # automatically detect tears and scratches so the user is never stuck in an error state.
    if blend_mode == "mask_guided" and mask is None:
        if DETECTION_AVAILABLE:
            try:
                auto_mask_u8, _, _ = detect_damage_mask(
                    image=img,
                    mode="general",
                    sensitivity=3.0
                )
                if np.any(auto_mask_u8 > 127):
                    mask = Image.fromarray(auto_mask_u8, mode="L")
            except Exception:
                pass

        # If no damage detected at all, fall back to global neural restoration
        if mask is None:
            blend_mode = "raw_unet"

    # Resolve target checkpoint
    if ckpt_override and ckpt_override.strip():
        target_ckpt = ckpt_override.strip()
    else:
        target_ckpt = CHECKPOINT_OPTIONS.get(model_choice, find_default_checkpoint())

    if not Path(target_ckpt).exists():
        target_ckpt = find_default_checkpoint()

    try:
        max_dimension = int(max_dim) if int(max_dim) > 0 else None

        if INTELLIGENT_PIPELINE_AVAILABLE:
            pipeline_result = intelligent_restore(
                image=img,
                mask=mask,
                ckpt_path=target_ckpt,
                blend_mode=blend_mode,
                feather_mode=feather_mode,
                feather_radius=int(feather_radius),
                max_dimension=max_dimension,
            )
            restored_np = pipeline_result["restored_image"]
            original_np = pipeline_result["original_image"]
            diag = pipeline_result["diagnostics"]
            status_message = pipeline_result["status_message"]
            final_engine_used = pipeline_result["final_engine_used"]

            restored_pil = Image.fromarray(restored_np)
            orig_pil = Image.fromarray(original_np)
            # raw_unet_image may not be present when cloud engine was used
            raw_np = diag.pop("_raw_unet_array", None)
            if raw_np is not None:
                raw_pil = Image.fromarray(raw_np)
            else:
                # Fall back: re-run via pipeline to get raw; or just use restored
                try:
                    _raw_res = restore_artwork_pipeline(
                        damaged_image=img,
                        damage_mask=mask,
                        ckpt_path=target_ckpt,
                        blend_mode="raw_unet",
                        feather_mode=feather_mode,
                        feather_radius=int(feather_radius),
                        max_dimension=max_dimension,
                        return_diagnostics=True,
                    )
                    raw_pil = Image.fromarray(_raw_res["raw_unet_image"])
                except Exception:
                    raw_pil = restored_pil
            preview_pil = orig_pil
        else:
            # Fallback: direct call to restore_artwork_pipeline
            status_message = "🏠 Local UNet Selected — Intelligent pipeline unavailable."
            final_engine_used = "local_unet"
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
            restored_pil = Image.fromarray(res["restored_image"])
            raw_pil = Image.fromarray(res["raw_unet_image"])
            orig_pil = Image.fromarray(res["original_image"])
            preview_pil = res["mask_preview"] or orig_pil
            diag = res["diagnostics"]

    except Exception as e:
        err_status = f"### ⚠️ Restoration Notice\nCould not complete restoration: {e}\n\nPlease check the image input and try again."
        err_diag = f"### ⚠️ Error Details\n```\n{e}\n```"
        return (
            (img, img),
            img,
            img,
            img,
            img,
            err_status,
            err_diag,
            gr.DownloadButton(visible=False),
            gr.DownloadButton(visible=False)
        )

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
            "mode_selection": mode_selection,
            "blend_mode": blend_mode,
            "feather_mode": feather_mode,
            "feather_radius": int(feather_radius),
            "max_dimension": max_dimension
        },
        "diagnostics": diag,
        "routing": {
            "engine": final_engine_used,
            "status": status_message,
        }
    }
    json.dump(export_payload, tmp_json, indent=2, default=str)
    tmp_json.close()

    # Format diagnostic status markdown
    is_exact = diag.get("is_bit_exact", False)
    status_icon = "✅" if is_exact else "⚠️"
    status_title = "100% Bit-Exact Preservation" if is_exact else "Raw UNet (Unguided)"

    _engine_label = {
        "local_unet": "Local ResNet34-UNet",
        "cloud_ai": "Cloud AI (Cloudflare SD 1.5)",
        "local_first_then_cloud": "Hybrid (Local → Cloud Standby)",
        "no_mask": "Local UNet (Raw / No Mask)",
    }.get(final_engine_used, final_engine_used)

    _lat = diag.get("latency_ms", 0.0)
    _lat_str = f"{_lat:.1f} ms" if isinstance(_lat, (int, float)) else str(_lat)
    _mae = diag.get("undamaged_mae", float("nan"))
    _mae_str = f"{_mae:.6f}" if isinstance(_mae, float) and not (_mae != _mae) else "N/A"
    _maxdiff = diag.get("undamaged_max_diff", "N/A")

    diag_md = f"""
### 📊 Restoration Diagnostics & Curatorial Report
| Diagnostic Metric | Measured Value | Curatorial Status |
| :--- | :---: | :--- |
| **Active Model** | `{Path(target_ckpt).parent.name}` | {model_choice.split('(')[0].strip()} |
| **Restoration Mode** | `{diag.get('blend_mode', blend_mode)}` | {'Mask-Guided Blending' if is_exact else 'Full-Image Inpainting'} |
| **Seam Feathering** | `{diag.get('feather_mode', 'none')} (r={diag.get('feather_radius', 0)}px)` | {'Interior Smoothstep Seam' if diag.get('feather_mode') == 'feathered' else 'Hard Boundary'} |
| **Preservation Status** | **{status_icon} {status_title}** | {'Undamaged pixels are 100% untouched' if is_exact else 'Global convolution shifts applied'} |
| **Damage Coverage** | `{diag.get('mask_coverage_pct', 0.0):.2f}%` | Area repaired by neural network |
| **Inference Latency** | `{_lat_str}` | Sequential CPU execution |
| **Undamaged MAE** | `{_mae_str}` | {'0.000000 (Mathematical bit-exact)' if is_exact else 'Non-zero pixel deviation'} |
| **Max Undamaged Diff** | `{_maxdiff}` | {'0 (Exact match)' if is_exact else 'Pixel shift'} |
| **Original Resolution**| `{diag.get('original_size', [0, 0])[0]} × {diag.get('original_size', [0, 0])[1]} px` | Native artwork dimensions |
| **Neural Scale**       | `{diag.get('eval_size', [0, 0])[0]} × {diag.get('eval_size', [0, 0])[1]} px` | Aspect-preserving evaluation |
| **Restoration Engine** | `{final_engine_used}` | {_engine_label} |
| **Routing Status** | {status_message} | Auto-selected |
"""

    slider_tuple = (orig_pil, restored_pil)
    engine_badge_md = f"### ⚙️ Engine Routing Status\n**{status_message}**"

    return (
        slider_tuple,
        restored_pil,
        orig_pil,
        preview_pil,
        raw_pil,
        engine_badge_md,
        diag_md,
        gr.DownloadButton(value=tmp_png.name, visible=True, label="⬇️ Download Restored Artwork (PNG)"),
        gr.DownloadButton(value=tmp_json.name, visible=True, label="📄 Export Curatorial Diagnostics (JSON)")
    )


# Backward compatibility wrapper for run_restoration
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
    slider_tuple, res_pil, orig_pil, prev_pil, raw_pil, status_msg, diag_md, dl_btn, dl_json = run_restoration_unified(
        mode_selection="Manual Mode",
        active_tab=active_tab,
        upload_img=upload_img,
        upload_mask=upload_mask,
        editor_data=editor_data,
        auto_img=None,
        auto_mask_arr=None,
        model_choice=model_choice,
        blend_mode=blend_mode,
        feather_mode=feather_mode,
        feather_radius=feather_radius,
        max_dim=max_dim,
        ckpt_override=ckpt_override
    )
    return (slider_tuple, res_pil, raw_pil, prev_pil, diag_md, dl_btn, dl_json)


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
Restore cultural heritage artifacts degraded by physical stains, craquelure, mold, scratches, or tears.
Supports **Manual Curatorial Annotation** and **Automatic Damage Detection** with seamless mask correction.
        """)

        # States
        active_tab_state = gr.State(value="brush")
        auto_mask_state = gr.State(value=None)

        with gr.Row():
            # ── LEFT COLUMN: INPUTS & CONTROLS ────────────────────────────────
            with gr.Column(scale=5):
                mode_selector = gr.Radio(
                    choices=[
                        "🖌️ Manual Mode (Brush Canvas / Upload Mask) — Verified & Recommended",
                        "🔍 Automatic Damage Detection Mode (Experimental CV Heuristic)"
                    ],
                    value="🖌️ Manual Mode (Brush Canvas / Upload Mask) — Verified & Recommended",
                    label="🛠️ Restoration & Mask Selection Mode"
                )

                # ── Group 1: Manual Mode ──────────────────────────────────────
                with gr.Group(visible=True) as manual_group:
                    with gr.Tabs():
                        with gr.Tab("🖌️ Interactive Brush & 1-Click Detect", id="tab_brush") as tab_brush:
                            gr.Markdown(
                                "**1. Upload artwork. 2. Paint over damage or click '⚡ Auto-Detect Damage'. 3. Click '🎨 Restore Artwork'!**"
                            )
                            image_editor = gr.ImageEditor(
                                label="Artwork Canvas & Brush",
                                type="filepath",
                                brush=gr.Brush(colors=["#ff2222", "#ffffff"], default_size=20),
                                eraser=gr.Eraser(),
                                transforms=[]
                            )
                            with gr.Row():
                                detect_canvas_btn = gr.Button("⚡ Auto-Detect Damage (1-Click)", variant="secondary", scale=2)
                                clear_brush_btn = gr.Button("🧹 Clear Brush", variant="secondary", scale=1)
                            canvas_status_box = gr.Markdown("💡 *Tip: Draw over scratches/tears with the red brush, or click **Auto-Detect Damage** to mark them automatically.*")

                        with gr.Tab("📁 Upload Image & Mask Files", id="tab_upload") as tab_upload:
                            gr.Markdown("**Direct File Upload**: Upload damaged artwork alongside a precomputed binary mask.")
                            upload_img = gr.Image(
                                label="📷 Damaged Artwork (Upload)",
                                type="pil",
                                sources=["upload", "clipboard"]
                            )
                            upload_mask = gr.Image(
                                label="🎭 Damage Mask File (White = Damaged, Black = Clean)",
                                type="pil",
                                sources=["upload", "clipboard"]
                            )
                            gr.Markdown("*Polarity: White (255) = Damaged region; Black (0) = Surviving authentic artwork.*")

                # ── Group 2: Automatic Mode ───────────────────────────────────
                with gr.Group(visible=False) as auto_group:
                    gr.Markdown("### 🔍 Advanced Automatic Damage Detection")
                    gr.Markdown(
                        "Upload an artwork below to run specific morphological filtering algorithms."
                    )
                    auto_img = gr.Image(
                        label="📷 Damaged Artwork for Automatic Analysis",
                        type="pil",
                        sources=["upload", "clipboard"]
                    )
                    with gr.Row():
                        detection_profile = gr.Dropdown(
                            choices=[
                                ("Fine Craquelure & Cracks (Multi-scale blackhat)", "craquelure"),
                                ("Bright Scratches & Surface Tears (Tophat)", "bright_scratches"),
                                ("Localized Paint Loss & Lacunae (Contrast)", "paint_loss"),
                                ("General Surface Blemishes (Combined filter)", "general")
                            ],
                            value="general",
                            label="🔬 Detection Profile"
                        )
                        sensitivity_slider = gr.Slider(
                            minimum=1.0,
                            maximum=5.0,
                            value=3.0,
                            step=0.5,
                            label="Sensitivity"
                        )

                    with gr.Row():
                        detect_btn = gr.Button("🔍 Detect Damage Mask", variant="secondary")
                        transfer_btn = gr.Button("✏️ Load Mask into Brush Editor for Correction", variant="secondary")

                    auto_info_box = gr.Markdown(
                        "⚠️ *Automatic detection provides an initial hypothesis. Review detected areas and use 'Load Mask into Brush Editor' to refine if needed.*"
                    )

                # Primary Action Row
                with gr.Row():
                    restore_btn = gr.Button("🎨 Restore Artwork", variant="primary", scale=2, size="lg")
                    preview_btn = gr.Button("👁️ Preview Mask", variant="secondary", scale=1)
                    clear_btn = gr.Button("🗑️ Reset", variant="secondary", scale=1)

                # Advanced Controls & Settings (Collapsed by default for clean UX)
                with gr.Accordion("⚙️ Advanced Model & Restoration Settings (Optional)", open=False):
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

            # ── RIGHT COLUMN: RESULTS & PREVIEWS ──────────────────────────────
            with gr.Column(scale=7):
                gr.Markdown("### 🔍 Restoration Results & Comparisons")

                # Engine Routing Status Badge
                engine_status_view = gr.Markdown(
                    "### ⚙️ Engine Routing Status\n🔄 *Ready — upload an artwork and click **'🎨 Restore Artwork'**.*",
                    elem_id="engine_status_view"
                )

                # Interactive Before / After ImageSlider
                slider_view = gr.ImageSlider(
                    label="↔️ Interactive Before / After Comparison (Drag slider to compare)",
                    type="pil",
                    show_label=True
                )

                with gr.Row():
                    download_btn = gr.DownloadButton(
                        label="⬇️ Download Restored Artwork (PNG)",
                        visible=False,
                        variant="primary",
                        scale=1
                    )
                    download_json_btn = gr.DownloadButton(
                        label="📄 Export Curatorial Report (JSON)",
                        visible=False,
                        variant="secondary",
                        scale=1
                    )

                # Tabbed Views for Individual Inspection
                with gr.Tabs():
                    with gr.Tab("✨ Restored Artwork"):
                        restored_view = gr.Image(label="Final Restored Artwork (Mask-Blended)", type="pil", interactive=False)
                    with gr.Tab("📷 Original Artwork"):
                        original_view = gr.Image(label="Original Uploaded Artwork", type="pil", interactive=False)
                    with gr.Tab("🎯 Damage Overlay Preview"):
                        overlay_view = gr.Image(label="Damage Mask Overlay (Red = Damaged)", type="pil", interactive=False)
                    with gr.Tab("🔬 Raw UNet Output"):
                        raw_view = gr.Image(label="Raw UNet (100% Pixels Replaced)", type="pil", interactive=False)
                    with gr.Tab("📊 Conservation Diagnostics"):
                        diag_box = gr.Markdown(
                            "### 📊 Restoration Diagnostics\n*Upload an image and click **Restore Artwork** to inspect preservation metrics.*"
                        )

        # ── Pre-Loaded Demonstration Samples (Click to test) ──────────────────
        gr.Markdown("---")
        gr.Markdown("### 🏛️ Pre-Loaded Submission Verification Examples (Click to test)")
        gr.Markdown(
            "Quickly test each required project submission scenario with verified sample artworks:"
        )
        sample_dir = _ROOT_DIR / "data" / "demo_samples"
        if sample_dir.exists():
            gr.Examples(
                examples=[
                    [str(sample_dir / "test_A_ordinary_artwork.jpg"), str(sample_dir / "test_A_ordinary_artwork_mask.png")],
                    [str(sample_dir / "test_B_renaissance_artwork.png"), str(sample_dir / "test_B_renaissance_artwork_mask.png")],
                    [str(sample_dir / "test_C_fine_scratches.png"), str(sample_dir / "test_C_fine_scratches_mask.png")],
                    [str(sample_dir / "test_D_localized_paint_loss.png"), str(sample_dir / "test_D_localized_paint_loss_mask.png")],
                    [str(sample_dir / "test_E_large_missing_region.png"), str(sample_dir / "test_E_large_missing_region_mask.png")],
                ],
                inputs=[upload_img, upload_mask],
                label="📁 Curatorial Test Samples (Test A: JPG, Test B: PNG, Test C: Scratches, Test D: Paint Loss, Test E: Large Region)"
            )

        # ── Curatorial Notice & Documentation ─────────────────────────────────
        gr.Markdown("""
---
### 🏛️ Curatorial & Scientific Standards
- **Mathematical Integrity**: In `mask_guided` mode, all authentic undamaged substrate is mathematically locked to original pixel values ($\text{MAE} = 0.000000, \text{MaxDiff} = 0$).
- **Interior Seam Feathering**: Softens transitions strictly inside the damage mask using Euclidean distance transform, completely eliminating border halos without contaminating authentic paint.
- **Museum Provenance**: Diagnostics export contains SHA metadata, inference scale, latency, and preservation metrics for conservation archives.
        """)

        # ── Event Listeners ───────────────────────────────────────────────────
        # Switch mode visibility
        def on_mode_change(choice: str):
            is_manual = "Manual" in choice
            return (
                gr.Group(visible=is_manual),
                gr.Group(visible=not is_manual)
            )

        mode_selector.change(
            fn=on_mode_change,
            inputs=[mode_selector],
            outputs=[manual_group, auto_group]
        )

        # Tab selections
        tab_brush.select(fn=lambda: "brush", outputs=[active_tab_state])
        tab_upload.select(fn=lambda: "upload", outputs=[active_tab_state])

        # 1-Click Detect & Clear on interactive canvas
        detect_canvas_btn.click(
            fn=on_detect_canvas,
            inputs=[image_editor],
            outputs=[image_editor, canvas_status_box]
        )

        clear_brush_btn.click(
            fn=on_clear_canvas_brush,
            inputs=[image_editor],
            outputs=[image_editor, canvas_status_box]
        )

        # Automatic detection
        detect_btn.click(
            fn=on_generate_auto_mask,
            inputs=[auto_img, detection_profile, sensitivity_slider],
            outputs=[auto_mask_state, overlay_view, overlay_view, auto_info_box]
        )

        # Transfer auto mask to brush editor for correction
        def on_transfer_click(image, mask_u8):
            ed_dict, msg = on_transfer_to_editor(image, mask_u8)
            # Make manual group visible and return editor content
            return (
                "🖌️ Manual Mode (Brush Canvas / Upload Mask) — Verified & Recommended",
                gr.Group(visible=True),
                gr.Group(visible=False),
                ed_dict,
                f"### ✏️ Mask Transferred for Correction\n{msg}"
            )

        transfer_btn.click(
            fn=on_transfer_click,
            inputs=[auto_img, auto_mask_state],
            outputs=[mode_selector, manual_group, auto_group, image_editor, auto_info_box]
        )

        # Unified Preview Mask
        preview_btn.click(
            fn=on_preview_mask_unified,
            inputs=[
                mode_selector,
                active_tab_state,
                upload_img,
                upload_mask,
                image_editor,
                auto_img,
                auto_mask_state
            ],
            outputs=[overlay_view, overlay_view, diag_box]
        )

        # Unified Restoration
        restore_btn.click(
            fn=run_restoration_unified,
            inputs=[
                mode_selector,
                active_tab_state,
                upload_img,
                upload_mask,
                image_editor,
                auto_img,
                auto_mask_state,
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
                original_view,
                overlay_view,
                raw_view,
                engine_status_view,
                diag_box,
                download_btn,
                download_json_btn
            ]
        )

        def on_clear():
            return (
                None, None, None, None, None,
                None, None, None, None, None,
                "### ⚙️ Engine Routing Status\n🔄 *Ready — upload an artwork and click **'🎨 Restore Artwork'**.*",
                "### 📊 Restoration Diagnostics\n*Cleared.*",
                gr.DownloadButton(visible=False),
                gr.DownloadButton(visible=False),
                "⚠️ *Automatic detection provides an initial hypothesis.*",
                "💡 *Tip: Draw over scratches/tears with the red brush, or click **Auto-Detect Damage** to mark them automatically.*"
            )

        clear_btn.click(
            fn=on_clear,
            outputs=[
                upload_img,
                upload_mask,
                image_editor,
                auto_img,
                auto_mask_state,
                slider_view,
                restored_view,
                original_view,
                overlay_view,
                raw_view,
                engine_status_view,
                diag_box,
                download_btn,
                download_json_btn,
                auto_info_box,
                canvas_status_box
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
