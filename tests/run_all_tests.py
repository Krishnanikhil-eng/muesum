# tests/run_all_tests.py
"""
MUESUM — Comprehensive Automated Test Runner
============================================
Discovers and executes all unit and regression test modules across the project:
  - tests/test_inference_pipeline.py
  - tests/test_feathering.py
  - tests/test_composite_loss.py
  - tests/test_gradio_app.py
  - test_pipeline.py (legacy validation smoke tests)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import unittest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir / "app"))
sys.path.insert(0, str(root_dir))

def run_all_tests():
    print("=" * 80)
    print("MUESUM — COMPREHENSIVE AUTOMATED QA & REGRESSION TEST SUITE")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Discover tests in tests/
    discovered = loader.discover(str(root_dir / "tests"), pattern="test_*.py")
    suite.addTests(discovered)

    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - start_time

    # Run legacy test_pipeline.py routines
    print("\n" + "-" * 80)
    print("EXECUTING LEGACY PIPELINE SMOKE TESTS (test_pipeline.py)")
    print("-" * 80)
    try:
        from test_pipeline import (
            test_damage_simulator,
            test_utils_and_metrics,
            test_unet_model
        )
        test_damage_simulator()
        test_utils_and_metrics()
        test_unet_model()
        legacy_passed = True
        print("Legacy pipeline tests: ALL PASSED")
    except Exception as e:
        legacy_passed = False
        print(f"Legacy pipeline test failed: {e}")

    total_unit_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_unit_tests - failures - errors

    print("\n" + "=" * 80)
    print("TEST EXECUTION SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Unit Tests Run:      {total_unit_tests}")
    print(f"Tests Passed:              {passed} / {total_unit_tests} ({(passed/total_unit_tests)*100:.1f}%)")
    print(f"Tests Failed:              {failures}")
    print(f"Errors Encountered:        {errors}")
    print(f"Legacy Pipeline Smoke:     {'PASSED' if legacy_passed else 'FAILED'}")
    print(f"Total Test Time:           {elapsed:.2f} seconds")
    print("=" * 80)

    if failures > 0 or errors > 0 or not legacy_passed:
        print("\n❌ QA STATUS: FAILED")
        sys.exit(1)
    else:
        print("\n✅ QA STATUS: ALL SYSTEMS VERIFIED AND PASSING")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
