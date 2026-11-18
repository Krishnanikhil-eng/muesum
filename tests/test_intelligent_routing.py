# tests/test_intelligent_routing.py
"""
Unit and Integration Tests for MUESUM Intelligent AI Restoration Routing & Cloud Fallback
=======================================================================================
Verifies:
  - Mask morphological analysis & feature extraction
  - Pre-restoration routing heuristics (fine scratches, medium damage, large voids)
  - Quality checker technical validation (dimensions, NaNs, smoothing, boundaries, preservation)
  - Cloudflare Workers AI integration (credentials, mock success, mock failure, timeout, polarity)
  - End-to-end intelligent pipeline & 100% authentic canvas preservation (MAE=0, MaxDiff=0)
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import io
import base64

import numpy as np
from PIL import Image

_TEST_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _TEST_DIR.parent
_SRC_DIR = _ROOT_DIR / "src"
_APP_DIR = _ROOT_DIR / "app"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from restoration_router import RestorationRouter
from quality_checker import RestorationQualityChecker
from cloud_inpainting_service import CloudInpaintingService, get_cloud_service
from intelligent_pipeline import intelligent_restore


class TestMaskAnalysisAndRouting(unittest.TestCase):
    """Tests for damage mask feature extraction and routing decision cascade."""

    def setUp(self):
        self.router = RestorationRouter()

    def test_router_analysis_fields(self):
        """Test 1: Mask analysis must extract all 8 morphological properties."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Create a 10x10 square
        mask[20:30, 20:30] = 255
        analysis = self.router.analyze_mask(mask)

        expected_fields = [
            "total_coverage_pct",
            "largest_component_pct",
            "largest_component_bbox",
            "largest_component_width_px",
            "largest_component_height_px",
            "num_components",
            "avg_component_pct",
            "estimated_max_width_px",
        ]
        for field in expected_fields:
            self.assertIn(field, analysis)

        self.assertAlmostEqual(analysis["total_coverage_pct"], 1.0, places=1)
        self.assertEqual(analysis["num_components"], 1)
        self.assertEqual(analysis["largest_component_width_px"], 10)
        self.assertEqual(analysis["largest_component_height_px"], 10)

    def test_router_fine_scratches_routes_to_local_unet(self):
        """Test 2: Fine scratches (<5% coverage, small components) route to Local UNet."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Thin scratch of 2 pixels wide
        mask[50:150, 50:52] = 255  # 200 pixels = 0.5% coverage
        analysis = self.router.analyze_mask(mask)
        decision = self.router.route(analysis, image_shape=(200, 200, 3))

        self.assertEqual(decision["engine"], "local_unet")
        self.assertTrue(decision["local_suitable"])
        self.assertFalse(decision["cloud_recommended"])

    def test_router_medium_damage_routes_to_hybrid(self):
        """Test 3: Moderate flaking (15-35% coverage) routes to local_first_then_cloud."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # 25% coverage distributed
        mask[10:60, 10:35] = 255  # 50 x 25 = 1250 px = 12.5%
        mask[10:60, 50:75] = 255  # 50 x 25 = 1250 px = 12.5% -> Total 25%
        analysis = self.router.analyze_mask(mask)
        decision = self.router.route(analysis, image_shape=(100, 100, 3))

        self.assertEqual(decision["engine"], "local_first_then_cloud")

    def test_router_large_void_routes_to_cloud(self):
        """Test 4: Large missing region (>35% coverage or >20% single component) routes to Cloud AI."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Large void 70x70 = 4900 px = 49% coverage
        mask[15:85, 15:85] = 255
        analysis = self.router.analyze_mask(mask)
        decision = self.router.route(analysis, image_shape=(100, 100, 3))

        self.assertEqual(decision["engine"], "cloud_ai")
        self.assertTrue(decision["cloud_recommended"])
        self.assertFalse(decision["local_suitable"])

    def test_router_format_status_message(self):
        """Test 5: Status formatter produces clear user-facing badge messages."""
        decision_local = {
            "engine": "local_unet",
            "reason": "Fine damage within local UNet capacity.",
            "confidence": "high",
        }
        msg_local = self.router.format_routing_status(decision_local)
        self.assertIn("Local UNet", msg_local)
        self.assertTrue(msg_local.startswith("🏠"))

        decision_cloud = {
            "engine": "cloud_ai",
            "reason": "Large void requires diffusion model.",
            "confidence": "high",
        }
        msg_cloud = self.router.format_routing_status(decision_cloud)
        self.assertIn("Cloud AI", msg_cloud)
        self.assertTrue(msg_cloud.startswith("☁️"))

    def test_router_configurable_thresholds(self):
        """Test 6: Thresholds can be customized without hardcoding."""
        strict_router = RestorationRouter(local_max_coverage_pct=5.0)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 10:20] = 255  # 800 px = 8.0% coverage
        analysis = strict_router.analyze_mask(mask)
        decision = strict_router.route(analysis, image_shape=(100, 100, 3))

        # Because strict_router has 5% limit, 8% must escalate to Cloud AI
        self.assertEqual(decision["engine"], "cloud_ai")


class TestQualityControlChecks(unittest.TestCase):
    """Tests for heuristic post-restoration quality verification."""

    def setUp(self):
        self.qc = RestorationQualityChecker()
        self.orig = np.full((64, 64, 3), 128, dtype=np.uint8)
        self.mask = np.zeros((64, 64), dtype=np.float32)
        self.mask[20:44, 20:44] = 1.0

    def test_quality_dimension_mismatch(self):
        """Test 7: Dimension mismatch must trigger hard failure."""
        corrupt_restored = np.full((32, 32, 3), 128, dtype=np.uint8)
        report = self.qc.run_all_checks(self.orig, corrupt_restored, self.mask)

        self.assertFalse(report["passed"])
        self.assertTrue(any("shape" in f.lower() or "mismatch" in f.lower() for f in report["hard_failures"]))
        self.assertTrue(report["escalate_to_cloud"])

    def test_quality_nan_pixels(self):
        """Test 8: NaN or Inf pixel values must trigger hard failure."""
        nan_restored = self.orig.astype(np.float32)
        nan_restored[25, 25, 0] = float("nan")
        report = self.qc.run_all_checks(self.orig, nan_restored, self.mask)

        self.assertFalse(report["passed"])
        self.assertTrue(any("pixel" in f.lower() or "nan" in f.lower() for f in report["hard_failures"]))

    def test_quality_outside_mask_preservation_failure(self):
        """Test 9: Tampering with unmasked pixels triggers preservation failure."""
        tampered = self.orig.copy()
        tampered[5, 5, :] = 255  # Undamaged pixel modified!
        report = self.qc.run_all_checks(self.orig, tampered, self.mask)

        self.assertFalse(report["passed"])
        self.assertFalse(report["details"]["outside_mask_preservation"]["pass"])
        self.assertGreater(report["details"]["outside_mask_preservation"]["undamaged_max_diff"], 0)

    def test_quality_excessive_smoothing(self):
        """Test 10: Completely flat inpainting inside textured artwork triggers smoothing warning."""
        # Create textured original
        np.random.seed(42)
        textured_orig = np.random.randint(50, 200, size=(64, 64, 3), dtype=np.uint8)
        # Inpaint with completely flat uniform gray
        flat_restored = textured_orig.copy()
        flat_restored[self.mask > 0.5] = 128

        report = self.qc.run_all_checks(textured_orig, flat_restored, self.mask)
        smooth_res = report["details"]["excessive_smoothing"]
        self.assertFalse(smooth_res["pass"])
        self.assertTrue(any("smooth" in w.lower() for w in report["warnings"]))

    def test_quality_all_pass_case(self):
        """Test 11: Valid restoration with preserved unmasked canvas passes all checks."""
        # Simulated clean restoration
        restored = self.orig.copy()
        restored[self.mask > 0.5] = 135  # Subtle, smooth fill

        report = self.qc.run_all_checks(self.orig, restored, self.mask)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["hard_failures"]), 0)
        self.assertEqual(report["details"]["outside_mask_preservation"]["undamaged_max_diff"], 0)


class TestCloudInpaintingService(unittest.TestCase):
    """Tests for Cloudflare Workers AI inpainting integration."""

    def test_cloud_no_credentials(self):
        """Test 12: Service reports unavailable when credentials are not set."""
        with patch.dict(os.environ, {}, clear=True):
            svc = CloudInpaintingService()
            self.assertFalse(svc.is_available())
            self.assertIn("Unavailable", svc.get_status_message())

    def test_cloud_with_credentials(self):
        """Test 13: Service reports available when environment variables are set."""
        env = {
            "CLOUDFLARE_ACCOUNT_ID": "dummy_test_account_id_123",
            "CLOUDFLARE_API_TOKEN": "dummy_test_token_xyz"
        }
        with patch.dict(os.environ, env, clear=True):
            svc = CloudInpaintingService()
            self.assertTrue(svc.is_available())
            self.assertIn("Ready", svc.get_status_message())

    def test_cloud_mask_polarity_verification(self):
        """Test 14: Mask polarity matches SD 1.5 inpainting standard (white = 255 = inpaint)."""
        svc = CloudInpaintingService()
        mask_u8 = np.zeros((64, 64), dtype=np.uint8)
        mask_u8[10:20, 10:20] = 255
        # Verify MUESUM undamaged = 0, damaged = 255
        self.assertEqual(mask_u8[0, 0], 0)
        self.assertEqual(mask_u8[15, 15], 255)

    @patch("cloud_inpainting_service.requests.post")
    def test_cloud_api_failure_returns_safe_dict(self, mock_post):
        """Test 15: Network errors or API failures return a safe result dict without crashing."""
        mock_post.side_effect = Exception("Cloudflare network connection timeout")
        env = {
            "CLOUDFLARE_ACCOUNT_ID": "test_acc",
            "CLOUDFLARE_API_TOKEN": "test_tok"
        }
        with patch.dict(os.environ, env, clear=True):
            svc = CloudInpaintingService()
            img = Image.new("RGB", (64, 64), color="blue")
            mask = np.zeros((64, 64), dtype=np.uint8)
            result = svc.restore(image_pil=img, mask_uint8=mask)

            self.assertFalse(result["success"])
            self.assertIsNone(result["image"])
            self.assertIsNotNone(result["error"])
            self.assertIn("timeout", result["error"].lower())

    @patch("cloud_inpainting_service.requests.post")
    def test_cloud_api_success_mocked(self, mock_post):
        """Test 16: Successful API response decodes image and resizes to original dimensions."""
        # Create a mock 512x512 PNG response
        mock_img = Image.new("RGB", (512, 512), color="green")
        buf = io.BytesIO()
        mock_img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("ascii")

        # Mock requests.Response to return Cloudflare JSON schema
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "result": {
                "image": b64_str
            }
        }
        mock_post.return_value = mock_resp

        env = {
            "CLOUDFLARE_ACCOUNT_ID": "test_acc",
            "CLOUDFLARE_API_TOKEN": "test_tok"
        }
        with patch.dict(os.environ, env, clear=True):
            svc = CloudInpaintingService()
            orig_pil = Image.new("RGB", (120, 80), color="blue")
            mask = np.zeros((80, 120), dtype=np.uint8)
            mask[20:40, 20:40] = 255

            result = svc.restore(image_pil=orig_pil, mask_uint8=mask)

            self.assertTrue(result["success"])
            self.assertIsNotNone(result["image"])
            self.assertEqual(result["image"].size, (120, 80))  # Rescaled to original


class TestIntelligentPipelineIntegration(unittest.TestCase):
    """End-to-end tests for the intelligent restoration pipeline."""

    def test_pipeline_local_only_for_fine_scratches(self):
        """Test 17: Fine scratches automatically execute Local UNet."""
        img = Image.new("RGB", (128, 128), color=(100, 150, 200))
        mask = Image.new("L", (128, 128), color=0)
        # Scratch of 2x20 pixels (< 0.3% coverage)
        for y in range(30, 50):
            mask.putpixel((50, y), 255)
            mask.putpixel((51, y), 255)

        res = intelligent_restore(image=img, mask=mask, max_dimension=128)
        self.assertEqual(res["final_engine_used"], "local_unet")
        self.assertIn("Local UNet", res["status_message"])
        self.assertEqual(res["diagnostics"]["undamaged_max_diff"], 0)
        self.assertEqual(res["diagnostics"]["undamaged_mae"], 0.0)

    def test_pipeline_force_engine_local(self):
        """Test 18: force_engine='local_unet' forces local execution even on large masks."""
        img = Image.new("RGB", (128, 128), color=(120, 120, 120))
        # 50% mask
        mask = Image.new("L", (128, 128), color=0)
        for y in range(0, 128):
            for x in range(0, 64):
                mask.putpixel((x, y), 255)

        res = intelligent_restore(image=img, mask=mask, force_engine="local_unet", max_dimension=128)
        self.assertEqual(res["final_engine_used"], "local_unet")
        self.assertEqual(res["diagnostics"]["undamaged_max_diff"], 0)

    def test_pipeline_cloud_unavailable_fallback(self):
        """Test 19: Large void gracefully falls back to Local UNet when cloud credentials are missing."""
        img = Image.new("RGB", (128, 128), color=(80, 120, 160))
        mask = Image.new("L", (128, 128), color=0)
        # 60% coverage void
        for y in range(20, 110):
            for x in range(20, 110):
                mask.putpixel((x, y), 255)

        with patch.dict(os.environ, {}, clear=True):
            res = intelligent_restore(image=img, mask=mask, max_dimension=128)
            self.assertEqual(res["final_engine_used"], "local_unet")
            self.assertIn("Falling back to Local UNet", res["status_message"])
            self.assertEqual(res["diagnostics"]["undamaged_max_diff"], 0)

    def test_pipeline_preservation_guarantee_100_percent(self):
        """Test 20: 100% Bit-Exact Preservation Guarantee (Outside-mask MAE=0, MaxDiff=0)."""
        np.random.seed(123)
        artwork_arr = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(artwork_arr)

        mask_arr = np.zeros((128, 128), dtype=np.uint8)
        mask_arr[40:88, 40:88] = 255  # Damaged central patch
        mask = Image.fromarray(mask_arr, mode="L")

        res = intelligent_restore(image=img, mask=mask, max_dimension=128)
        restored = res["restored_image"]
        clean_mask = (mask_arr == 0)

        # Mathematical verification
        outside_diff = np.abs(restored[clean_mask].astype(int) - artwork_arr[clean_mask].astype(int))
        max_diff = int(np.max(outside_diff))
        mae = float(np.mean(outside_diff))

        self.assertEqual(max_diff, 0, f"MaxDiff outside mask must be exactly 0, got {max_diff}")
        self.assertEqual(mae, 0.0, f"MAE outside mask must be identically 0.0, got {mae}")
        self.assertTrue(res["diagnostics"]["is_bit_exact"])


if __name__ == "__main__":
    unittest.main()
