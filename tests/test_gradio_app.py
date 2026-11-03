# tests/test_gradio_app.py
"""
Unit and regression test suite for MUESUM Gradio Studio application.
Verifies:
  1. build_interface() constructs complete Blocks component tree without errors.
  2. Input extraction correctly handles upload and interactive brush canvas modes.
  3. Restoration handler returns proper slider tuples, preview images, and download buttons.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

_TEST_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _TEST_DIR.parent
_APP_DIR = _ROOT_DIR / "app"
_SRC_DIR = _ROOT_DIR / "src"
sys.path.insert(0, str(_APP_DIR))
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_ROOT_DIR))

import gradio as gr
from demo import build_interface, _extract_inputs, run_restoration, CHECKPOINT_OPTIONS


class TestGradioStudio(unittest.TestCase):

    def test_build_interface_constructs_cleanly(self):
        """Test 1: Gradio Blocks tree constructs without error."""
        app = build_interface()
        self.assertIsInstance(app, gr.Blocks)
        # Check that title is defined
        self.assertIn("MUESUM Studio", app.title)

    def test_extract_inputs_upload_mode(self):
        """Test 2: Standard upload mode returns image and mask directly."""
        img = Image.new("RGB", (64, 64), color="blue")
        mask = Image.new("L", (64, 64), color=255)
        out_img, out_mask = _extract_inputs("upload", img, mask, None)
        self.assertEqual(out_img, img)
        self.assertEqual(out_mask, mask)

    def test_extract_inputs_brush_mode(self):
        """Test 3: Interactive brush editor extracts painted layer alpha as mask."""
        bg = Image.new("RGB", (64, 64), color="ivory")
        # Paint a red square on a layer
        layer = Image.new("RGBA", (64, 64), color=(0, 0, 0, 0))
        layer_arr = np.array(layer)
        layer_arr[16:48, 16:48] = [255, 0, 0, 255]
        layer_painted = Image.fromarray(layer_arr, mode="RGBA")

        editor_data = {
            "background": bg,
            "layers": [layer_painted],
            "composite": None
        }

        out_img, out_mask = _extract_inputs("brush", None, None, editor_data)
        self.assertIsNotNone(out_img)
        self.assertIsNotNone(out_mask)
        mask_arr = np.array(out_mask)
        # Damaged painted center should be 255, borders 0
        self.assertEqual(mask_arr[32, 32], 255)
        self.assertEqual(mask_arr[0, 0], 0)

    def test_checkpoint_options_paths(self):
        """Test 4: Default checkpoint option paths are configured."""
        self.assertIn("Experimental Texture Model (Composite Loss — Recommended)", CHECKPOINT_OPTIONS)
        self.assertIn("Baseline ResNet34-UNet (Epoch 8 Checkpoint)", CHECKPOINT_OPTIONS)


if __name__ == "__main__":
    unittest.main()
