# tests/test_composite_loss.py
"""
Unit and regression test suite for composite restoration losses.
Verifies:
  1. MaskWeightedL1Loss correctly places higher weight on damaged pixels.
  2. SobelEdgeLoss computes edge gradient differences and backpropagates clean gradients.
  3. TotalVariationLoss penalizes spatial oscillations.
  4. CompositeRestorationLoss aggregates all terms with well-behaved gradients.
"""

import sys
import unittest
from pathlib import Path
import torch
import torch.nn as nn

_TEST_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _TEST_DIR.parent
_SRC_DIR = _ROOT_DIR / "src"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_ROOT_DIR))

from composite_loss import (
    MaskWeightedL1Loss,
    SobelEdgeLoss,
    TotalVariationLoss,
    CompositeRestorationLoss
)


class TestCompositeLoss(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        # Synthetic batch: B=2, C=3, H=64, W=64
        self.pred = torch.rand((2, 3, 64, 64), requires_grad=True)
        self.target = torch.rand((2, 3, 64, 64))
        # Mask: 1.0 for damaged central region, 0.0 for clean border
        self.mask = torch.zeros((2, 1, 64, 64))
        self.mask[:, :, 16:48, 16:48] = 1.0

    def test_mask_weighted_l1_damage_priority(self):
        """Test 1: Error inside the damage mask must produce higher loss when damage_weight > clean_weight."""
        loss_fn = MaskWeightedL1Loss(damage_weight=5.0, clean_weight=1.0)

        # Case A: Error entirely inside damage mask
        pred_a = self.target.clone()
        pred_a[:, :, 20:30, 20:30] += 0.5
        loss_dmg = loss_fn(pred_a, self.target, self.mask)

        # Case B: Identical magnitude error entirely in clean region
        pred_b = self.target.clone()
        pred_b[:, :, 0:10, 0:10] += 0.5
        loss_clean = loss_fn(pred_b, self.target, self.mask)

        self.assertGreater(loss_dmg.item(), loss_clean.item())

    def test_sobel_edge_loss_gradient_flow(self):
        """Test 2: SobelEdgeLoss computes finite positive loss and backpropagates gradients."""
        loss_fn = SobelEdgeLoss()
        loss = loss_fn(self.pred, self.target)

        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

        loss.backward()
        self.assertIsNotNone(self.pred.grad)
        self.assertFalse(torch.isnan(self.pred.grad).any())

    def test_tv_loss_smoothness_penalty(self):
        """Test 3: TotalVariationLoss correctly penalizes high-frequency noise over uniform image."""
        loss_fn = TotalVariationLoss()

        smooth_img = torch.full((1, 3, 32, 32), 0.5)
        noisy_img = smooth_img + 0.2 * torch.randn((1, 3, 32, 32))

        loss_smooth = loss_fn(smooth_img)
        loss_noisy = loss_fn(noisy_img)

        self.assertEqual(loss_smooth.item(), 0.0)
        self.assertGreater(loss_noisy.item(), 0.0)

    def test_composite_loss_aggregation(self):
        """Test 4: CompositeRestorationLoss returns total loss and dictionary breakdown."""
        # Use without perceptual in unit test for instant headless testing
        loss_fn = CompositeRestorationLoss(
            lam_l1=1.0,
            lam_edge=0.1,
            lam_tv=0.01,
            use_perceptual=False
        )

        total, metrics = loss_fn(self.pred, self.target, mask=self.mask)

        self.assertIn("total_loss", metrics)
        self.assertIn("l1_loss", metrics)
        self.assertIn("edge_loss", metrics)
        self.assertIn("tv_loss", metrics)
        self.assertGreater(total.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
