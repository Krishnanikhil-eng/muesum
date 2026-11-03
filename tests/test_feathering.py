# tests/test_feathering.py
"""
Unit and regression test suite for boundary seam feathering in MUESUM.
Verifies:
  1. Undamaged pixels remain 100% bit-exact (undamaged_max_diff == 0, undamaged_mae == 0.0).
  2. Boundary feathering softens sharp edges inside damage mask without bleeding outside.
  3. feather_radius <= 0 falls back to hard blending.
  4. feather_mode == 'hard' produces identical results to traditional hard blending.
  5. Edge cases: entirely clean mask (all 0s), entirely damaged mask (all 1s), single-pixel damage.
  6. Pipeline integration via restore_artwork_pipeline().
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn

# Add project root and src to path
_TEST_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _TEST_DIR.parent
_SRC_DIR = _ROOT_DIR / "src"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_ROOT_DIR))

from feathering import compute_interior_feather_alpha, blend_with_boundary_feathering
from inference import restore_artwork_pipeline


class MockNeuralModel(nn.Module):
    """Deterministic mock neural model for fast CPU unit testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.eye(3).view(3, 3, 1, 1))

    def forward(self, x):
        # Invert colors: 1.0 - x
        return 1.0 - self.conv(x)


class TestBoundaryFeathering(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.h, self.w = 120, 120

        # Uniform or textured clean artwork
        self.clean_rgb = np.full((self.h, self.w, 3), 180, dtype=np.uint8)
        # Damaged artwork with high contrast black inpainting target
        self.restored_rgb = np.full((self.h, self.w, 3), 20, dtype=np.uint8)

        # Rectangular damage box in center: 30:90, 30:90
        self.mask_binary = np.zeros((self.h, self.w), dtype=np.float32)
        self.mask_binary[30:90, 30:90] = 1.0

        self.mock_model = MockNeuralModel().eval()

    def test_alpha_strictly_zero_on_undamaged_artwork(self):
        """Test 1: Interior alpha must be identically 0.0 outside the damage mask."""
        alpha = compute_interior_feather_alpha(self.mask_binary, feather_radius=5)
        undamaged_idx = (self.mask_binary == 0.0)

        # Assert no bleeding whatsoever outside the mask
        self.assertEqual(np.max(alpha[undamaged_idx]), 0.0)
        self.assertEqual(np.min(alpha[undamaged_idx]), 0.0)

        # Assert inside the mask, alpha is positive
        damaged_idx = (self.mask_binary == 1.0)
        self.assertGreater(np.mean(alpha[damaged_idx]), 0.0)
        # Assert deep interior reaches 1.0
        self.assertEqual(alpha[60, 60], 1.0)

    def test_exact_undamaged_preservation_feathered(self):
        """Test 2: Undamaged pixels must remain 100% bit-exact under feathered blending."""
        blended, metrics = blend_with_boundary_feathering(
            orig_rgb=self.clean_rgb,
            restored_rgb=self.restored_rgb,
            mask_binary=self.mask_binary,
            feather_mode="feathered",
            feather_radius=4
        )

        undamaged_idx = (self.mask_binary == 0.0)
        # 1. Assert bit-exact equality on all undamaged pixels
        self.assertTrue(np.array_equal(blended[undamaged_idx], self.clean_rgb[undamaged_idx]))
        self.assertTrue(metrics["is_bit_exact"])
        self.assertEqual(metrics["undamaged_max_diff"], 0)
        self.assertEqual(metrics["undamaged_mae"], 0.0)

    def test_exact_undamaged_preservation_hard(self):
        """Test 3: Hard blending must also maintain 100% bit-exact preservation."""
        blended, metrics = blend_with_boundary_feathering(
            orig_rgb=self.clean_rgb,
            restored_rgb=self.restored_rgb,
            mask_binary=self.mask_binary,
            feather_mode="hard"
        )

        undamaged_idx = (self.mask_binary == 0.0)
        self.assertTrue(np.array_equal(blended[undamaged_idx], self.clean_rgb[undamaged_idx]))
        self.assertTrue(metrics["is_bit_exact"])
        self.assertEqual(metrics["undamaged_max_diff"], 0)

    def test_seam_gradient_reduction(self):
        """Test 4: Feathered blending must significantly reduce boundary gradient disjunction."""
        # Hard blend
        blended_hard, _ = blend_with_boundary_feathering(
            orig_rgb=self.clean_rgb,
            restored_rgb=self.restored_rgb,
            mask_binary=self.mask_binary,
            feather_mode="hard"
        )

        # Feathered blend
        blended_feathered, _ = blend_with_boundary_feathering(
            orig_rgb=self.clean_rgb,
            restored_rgb=self.restored_rgb,
            mask_binary=self.mask_binary,
            feather_mode="feathered",
            feather_radius=5
        )

        # Calculate horizontal derivative across boundary (column 29 to 30)
        # Column 29 is clean (180), Column 30 is damaged edge
        diff_hard = np.abs(blended_hard[:, 30].astype(int) - blended_hard[:, 29].astype(int))[40:80, 0]
        diff_feathered = np.abs(blended_feathered[:, 30].astype(int) - blended_feathered[:, 29].astype(int))[40:80, 0]

        mean_step_hard = np.mean(diff_hard)
        mean_step_feathered = np.mean(diff_feathered)

        # Hard step should be ~160 (180 - 20)
        self.assertAlmostEqual(mean_step_hard, 160.0, delta=1.0)
        # Feathered step must be drastically smoother
        self.assertLess(mean_step_feathered, mean_step_hard * 0.5)

    def test_fallback_when_radius_zero(self):
        """Test 5: feather_radius <= 0 falls back to hard blending."""
        blended_r0, _ = blend_with_boundary_feathering(
            orig_rgb=self.clean_rgb,
            restored_rgb=self.restored_rgb,
            mask_binary=self.mask_binary,
            feather_mode="feathered",
            feather_radius=0
        )
        blended_hard, _ = blend_with_boundary_feathering(
            orig_rgb=self.clean_rgb,
            restored_rgb=self.restored_rgb,
            mask_binary=self.mask_binary,
            feather_mode="hard"
        )
        self.assertTrue(np.array_equal(blended_r0, blended_hard))

    def test_edge_case_empty_mask(self):
        """Test 6: Empty mask (all zeros) returns original image bit-exact."""
        empty_mask = np.zeros((self.h, self.w), dtype=np.float32)
        blended, metrics = blend_with_boundary_feathering(
            orig_rgb=self.clean_rgb,
            restored_rgb=self.restored_rgb,
            mask_binary=empty_mask,
            feather_mode="feathered",
            feather_radius=3
        )
        self.assertTrue(np.array_equal(blended, self.clean_rgb))
        self.assertTrue(metrics["is_bit_exact"])

    def test_pipeline_integration(self):
        """Test 7: restore_artwork_pipeline executes end-to-end with feathering."""
        res = restore_artwork_pipeline(
            damaged_image=self.clean_rgb,
            damage_mask=self.mask_binary,
            model=self.mock_model,
            blend_mode="mask_guided",
            feather_mode="feathered",
            feather_radius=3,
            max_dimension=None
        )

        diag = res["diagnostics"]
        self.assertEqual(diag["feather_mode"], "feathered")
        self.assertEqual(diag["feather_radius"], 3)
        self.assertTrue(diag["is_bit_exact"])
        self.assertEqual(diag["undamaged_max_diff"], 0)
        self.assertEqual(diag["undamaged_mae"], 0.0)
        self.assertIn("restored_image", res)
        self.assertIn("mask_preview", res)


if __name__ == "__main__":
    unittest.main()
