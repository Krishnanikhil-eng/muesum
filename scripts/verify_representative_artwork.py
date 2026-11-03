import time
import psutil
from PIL import Image
import numpy as np
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

from inference import restore_artwork_pipeline, ModelManager

def main():
    mem_before = psutil.virtual_memory()
    print(f"Initial RAM: {mem_before.used / (1024**3):.2f} GB used / {mem_before.available / (1024**3):.2f} GB avail")

    dmg_path = "data/damaged/test/247010.jpg"
    mask_path = "data/damaged/test/247010_mask.png"

    # 1. First run (loads model into ModelManager cache)
    t0 = time.time()
    res1 = restore_artwork_pipeline(
        damaged_image=dmg_path,
        damage_mask=mask_path,
        blend_mode="mask_guided",
        max_dimension=1024,
        return_diagnostics=True
    )
    t1 = time.time()
    d1 = res1["diagnostics"]
    print(f"\n--- RUN 1 (with model loading) ---")
    print(f"Total wall time: {t1 - t0:.2f} s | Pipeline latency: {d1['latency_ms']:.1f} ms")
    print(f"Mask coverage: {d1['mask_coverage_pct']:.2f}% | Mode: {d1['blend_mode']}")
    print(f"Preservation: {d1['preservation_status']} | MAE: {d1['undamaged_mae']:.6f} | MaxDiff: {d1['undamaged_max_diff']}")
    print(f"Orig size: {d1['original_size']} | Eval size: {d1['eval_size']}")
    assert d1["is_bit_exact"] is True
    assert d1["undamaged_max_diff"] == 0

    # 2. Second run (uses cached model in ModelManager)
    t2 = time.time()
    res2 = restore_artwork_pipeline(
        damaged_image=dmg_path,
        damage_mask=mask_path,
        blend_mode="mask_guided",
        max_dimension=1024,
        return_diagnostics=True
    )
    t3 = time.time()
    d2 = res2["diagnostics"]
    print(f"\n--- RUN 2 (cached model in memory) ---")
    print(f"Total wall time: {t3 - t2:.2f} s | Pipeline latency: {d2['latency_ms']:.1f} ms")
    print(f"Preservation: {d2['preservation_status']} | MAE: {d2['undamaged_mae']:.6f} | MaxDiff: {d2['undamaged_max_diff']}")
    assert d2["is_bit_exact"] is True
    assert d2["undamaged_max_diff"] == 0

    mem_after = psutil.virtual_memory()
    print(f"\nFinal RAM: {mem_after.used / (1024**3):.2f} GB used / {mem_after.available / (1024**3):.2f} GB avail")
    print("ALL VERIFICATIONS ON REPRESENTATIVE ARTWORK PASSED!")

if __name__ == "__main__":
    main()
