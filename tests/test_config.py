# tests/test_config.py
"""
Unit and regression test suite for MUESUM centralized configuration.
Verifies:
  1. Base and sub-directory paths resolve to valid Path instances.
  2. resolve_checkpoint() returns valid, existing paths for baseline and experimental models.
  3. get_device() correctly returns torch.device instance.
  4. Hyperparameter defaults and loss configuration match documented specifications.
"""

import sys
import unittest
from pathlib import Path
import torch

_TEST_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _TEST_DIR.parent
_SRC_DIR = _ROOT_DIR / "src"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_ROOT_DIR))

from config import (
    ROOT_DIR,
    SRC_DIR,
    DATA_DIR,
    CHECKPOINTS_DIR,
    BASELINE_CHECKPOINT_PATH,
    EXPERIMENTAL_CHECKPOINT_PATH,
    resolve_checkpoint,
    get_device,
    DEFAULT_BLEND_MODE,
    DEFAULT_FEATHER_MODE,
    DEFAULT_FEATHER_RADIUS,
    LOSS_CONFIG
)


class TestConfigSystem(unittest.TestCase):

    def test_paths_validity(self):
        """Test 1: Core filesystem paths are Path instances and point within workspace."""
        self.assertIsInstance(ROOT_DIR, Path)
        self.assertTrue(ROOT_DIR.exists())
        self.assertTrue(SRC_DIR.exists())
        self.assertTrue(DATA_DIR.exists())
        self.assertTrue(CHECKPOINTS_DIR.exists())

    def test_checkpoint_paths_exist(self):
        """Test 2: Both Baseline and Experimental checkpoints exist on disk."""
        self.assertTrue(BASELINE_CHECKPOINT_PATH.exists(), f"Baseline not found: {BASELINE_CHECKPOINT_PATH}")
        self.assertTrue(EXPERIMENTAL_CHECKPOINT_PATH.exists(), f"Experimental not found: {EXPERIMENTAL_CHECKPOINT_PATH}")

    def test_checkpoint_resolution_preference(self):
        """Test 3: resolve_checkpoint returns appropriate model paths based on preference."""
        exp_resolved = resolve_checkpoint("experimental")
        base_resolved = resolve_checkpoint("baseline")

        self.assertIn("experimental_texture_model", exp_resolved)
        self.assertIn("baseline_run", base_resolved)

    def test_device_detection(self):
        """Test 4: get_device returns a valid torch.device."""
        dev = get_device()
        self.assertIsInstance(dev, torch.device)
        self.assertIn(dev.type, ("cpu", "cuda"))

    def test_loss_hyperparameters_structure(self):
        """Test 5: LOSS_CONFIG contains all expected loss keys with positive values."""
        required_keys = ["lambda_l1", "lambda_edge", "lambda_tv", "lambda_perceptual", "damage_weight"]
        for k in required_keys:
            self.assertIn(k, LOSS_CONFIG)
            self.assertGreater(LOSS_CONFIG[k], 0.0)

    def test_restoration_defaults(self):
        """Test 6: Inpainting defaults match verified conservation protocols."""
        self.assertEqual(DEFAULT_BLEND_MODE, "mask_guided")
        self.assertEqual(DEFAULT_FEATHER_MODE, "feathered")
        self.assertEqual(DEFAULT_FEATHER_RADIUS, 3)


if __name__ == "__main__":
    unittest.main()
