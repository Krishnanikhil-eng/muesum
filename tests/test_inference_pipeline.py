# tests/test_inference_pipeline.py
"""
Unit and regression test suite for MUESUM restoration inference pipeline.
Verifies:
  1. Undamaged pixels remain bit-exact after mask-guided restoration.
  2. Dimension mismatch between image and mask is strictly rejected.
  3. Missing mask is rejected in mask-guided mode (no silent guessing).
  4. Raw UNet mode executes properly without a mask.
  5. Invalid mask values (NaNs, out-of-range floats) are rejected appropriately.
  6. ModelManager caching avoids repeated checkpoint reloading.
  7. Full-resolution preservation when aspect-preserving resize is applied.
  8. Legacy unet_restore() backward compatibility wrapper.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# Add src to sys.path
_TEST_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _TEST_DIR.parent
_SRC_DIR = _ROOT_DIR / "src"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_ROOT_DIR))

from inference import (
    restore_artwork_pipeline,
    ModelManager,
    unet_restore,
    _standardize_image,
    _standardize_mask
)


class DummyRestorationModel(nn.Module):
    """
    Lightweight mock neural model for instant, deterministic testing.
    Inverts pixel values (1.0 - x) so inpainting produces distinct output.
    """
    def __init__(self):
        super().__init__()
        # 1x1 conv to simulate trainable weights without compute overhead
        self.conv = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.eye(3).view(3, 3, 1, 1))

    def forward(self, x):
        # Invert colors so output differs from input
        return 1.0 - self.conv(x)


class TestInferencePipeline(unittest.TestCase):

    def setUp(self):
        # Set reproducible seed
        np.random.seed(42)
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.dummy_model = DummyRestorationModel().to(self.device).eval()

        # Create standard test synthetic artwork: 128x128 RGB
        self.h, self.w = 128, 128
        self.clean_rgb = np.random.randint(50, 200, (self.h, self.w, 3), dtype=np.uint8)

        # Create binary damage mask: central box is damaged (1), borders are clean (0)
        self.mask_uint8 = np.zeros((self.h, self.w), dtype=np.uint8)
        self.mask_uint8[32:96, 32:96] = 255  # 25% area damaged

    def test_exact_undamaged_preservation(self):
        """Test 1: Undamaged pixels must remain 100% bit-exact under mask-guided restoration."""
        res = restore_artwork_pipeline(
            damaged_image=self.clean_rgb,
            damage_mask=self.mask_uint8,
            model=self.dummy_model,
            device=self.device,
            blend_mode="mask_guided",
            max_dimension=None
        )

        restored = res["restored_image"]
        diag = res["diagnostics"]

        # Undamaged boolean mask
        undamaged_mask = (self.mask_uint8 == 0)
        damaged_mask = (self.mask_uint8 == 255)

        # 1. Assert undamaged pixels are mathematically bit-exact
        self.assertTrue(diag["is_bit_exact"])
        self.assertEqual(diag["undamaged_max_diff"], 0)
        self.assertEqual(diag["undamaged_mae"], 0.0)
        self.assertTrue(np.array_equal(restored[undamaged_mask], self.clean_rgb[undamaged_mask]))

        # 2. Assert damaged pixels were actually modified by the model
        self.assertFalse(np.array_equal(restored[damaged_mask], self.clean_rgb[damaged_mask]))

    def test_dimension_mismatch_rejection(self):
        """Test 2: Dimension mismatch between image and mask must raise ValueError."""
        mismatched_mask = np.zeros((self.h + 20, self.w + 10), dtype=np.uint8)
        with self.assertRaises(ValueError) as ctx:
            restore_artwork_pipeline(
                damaged_image=self.clean_rgb,
                damage_mask=mismatched_mask,
                model=self.dummy_model,
                device=self.device,
                blend_mode="mask_guided"
            )
        self.assertIn("Dimension mismatch", str(ctx.exception))

    def test_missing_mask_rejection(self):
        """Test 3: Missing mask must be strictly rejected in mask-guided mode (no silent guessing)."""
        with self.assertRaises(ValueError) as ctx:
            restore_artwork_pipeline(
                damaged_image=self.clean_rgb,
                damage_mask=None,
                model=self.dummy_model,
                device=self.device,
                blend_mode="mask_guided"
            )
        self.assertIn("requires a damage mask", str(ctx.exception))

    def test_raw_unet_mode_without_mask(self):
        """Test 4: Raw UNet mode executes without mask, but does not claim bit-exact preservation."""
        res = restore_artwork_pipeline(
            damaged_image=self.clean_rgb,
            damage_mask=None,
            model=self.dummy_model,
            device=self.device,
            blend_mode="raw_unet",
            max_dimension=None
        )

        diag = res["diagnostics"]
        self.assertEqual(diag["blend_mode"], "raw_unet")
        self.assertFalse(diag["is_bit_exact"])
        self.assertGreater(diag["undamaged_max_diff"], 0)
        self.assertIsNone(res["mask_binary"])

    def test_invalid_mask_values_and_polarity(self):
        """Test 5: Masks with NaNs or out-of-range values are rejected."""
        # NaN mask
        nan_mask = np.zeros((self.h, self.w), dtype=np.float32)
        nan_mask[10, 10] = np.nan
        with self.assertRaises(ValueError) as ctx:
            restore_artwork_pipeline(
                damaged_image=self.clean_rgb,
                damage_mask=nan_mask,
                model=self.dummy_model,
                device=self.device,
                blend_mode="mask_guided"
            )
        self.assertIn("NaN or Infinite", str(ctx.exception))

        # Out-of-range float mask (e.g. max is 10.0 instead of 1.0)
        invalid_float_mask = np.ones((self.h, self.w), dtype=np.float32) * 5.0
        with self.assertRaises(ValueError) as ctx:
            restore_artwork_pipeline(
                damaged_image=self.clean_rgb,
                damage_mask=invalid_float_mask,
                model=self.dummy_model,
                device=self.device,
                blend_mode="mask_guided"
            )
        self.assertIn("range [0.0, 1.0]", str(ctx.exception))

        # Valid float mask [0.0, 1.0] vs uint8 mask [0, 255] should produce identical output
        float_mask = (self.mask_uint8 / 255.0).astype(np.float32)
        res_uint8 = restore_artwork_pipeline(
            self.clean_rgb, self.mask_uint8, model=self.dummy_model, blend_mode="mask_guided"
        )
        res_float = restore_artwork_pipeline(
            self.clean_rgb, float_mask, model=self.dummy_model, blend_mode="mask_guided"
        )
        self.assertTrue(np.array_equal(res_uint8["restored_image"], res_float["restored_image"]))

    def test_model_caching_behavior(self):
        """Test 6: ModelManager singleton caches model and does not reload checkpoint unnecessarily."""
        ModelManager.clear_cache()

        mock_state = {"model": self.dummy_model.state_dict()}
        dummy_ckpt_file = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
        dummy_ckpt_file.close()

        with patch("torch.load", return_value=mock_state) as mock_torch_load:
            with patch("inference.ResNet34UNet", return_value=self.dummy_model):
                # First call: should call torch.load
                m1, d1 = ModelManager.get_model(ckpt_path=dummy_ckpt_file.name, device=self.device)
                self.assertEqual(mock_torch_load.call_count, 1)

                # Second call with same path: should NOT call torch.load again
                m2, d2 = ModelManager.get_model(ckpt_path=dummy_ckpt_file.name, device=self.device)
                self.assertEqual(mock_torch_load.call_count, 1)
                self.assertIs(m1, m2)

        # Cleanup
        Path(dummy_ckpt_file.name).unlink(missing_ok=True)
        ModelManager.clear_cache()

    def test_resizing_and_orig_resolution_preservation(self):
        """Test 7: Aspect-preserving resize inference still produces full-resolution bit-exact blending."""
        large_h, large_w = 300, 400
        large_img = np.random.randint(0, 255, (large_h, large_w, 3), dtype=np.uint8)
        large_mask = np.zeros((large_h, large_w), dtype=np.uint8)
        large_mask[50:150, 80:200] = 255

        # Force neural inference down to max_dimension = 128
        res = restore_artwork_pipeline(
            damaged_image=large_img,
            damage_mask=large_mask,
            model=self.dummy_model,
            device=self.device,
            blend_mode="mask_guided",
            max_dimension=128
        )

        restored = res["restored_image"]
        diag = res["diagnostics"]

        # 1. Output must match original dimensions
        self.assertEqual(restored.shape, (large_h, large_w, 3))
        self.assertEqual(diag["original_size"], (large_w, large_h))
        self.assertLessEqual(max(diag["eval_size"]), 128)

        # 2. Undamaged pixels must still be 100% bit-exact
        self.assertTrue(diag["is_bit_exact"])
        self.assertEqual(diag["undamaged_max_diff"], 0)
        undamaged_idx = (large_mask == 0)
        self.assertTrue(np.array_equal(restored[undamaged_idx], large_img[undamaged_idx]))

    def test_legacy_unet_restore_wrapper(self):
        """Test 8: unet_restore() backward compatibility wrapper saves valid output image."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            inp_file = tmp_path / "test_input.jpg"
            out_file = tmp_path / "test_output.png"

            Image.fromarray(self.clean_rgb).save(inp_file)

            with patch("inference.ModelManager.get_model", return_value=(self.dummy_model, self.device)):
                unet_restore(inp_path=str(inp_file), out_path=str(out_file), ckpt_path="mock_ckpt.pth")

            self.assertTrue(out_file.exists())
            with Image.open(out_file) as out_im:
                out_arr = np.array(out_im)
                self.assertEqual(out_arr.shape, (self.h, self.w, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)

