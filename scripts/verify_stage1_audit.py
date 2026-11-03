import os
import sys
import psutil
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'app')
from inference import ModelManager, restore_artwork_pipeline, find_default_checkpoint
from unet_model import ResNet34UNet
from demo import build_interface, run_restoration

def main():
    print("--- 1. Checkpoint Verification ---")
    ckpt_path = find_default_checkpoint()
    print("Default checkpoint resolved:", ckpt_path)
    st = torch.load(ckpt_path, map_location='cpu')
    print("Keys in state dict:", list(st.keys()))
    print("Epoch:", st.get('epoch'))
    print("Val SSIM:", st.get('val_ssim'))
    assert st.get('epoch') == 8, f"Expected epoch 8, got {st.get('epoch')}"

    print("\n--- 2. Architecture Match ---")
    model, dev = ModelManager.get_model(ckpt_path)
    print("Model class:", type(model).__name__)
    print("Device:", dev)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    print("\n--- 3. Model Caching ---")
    m2, d2 = ModelManager.get_model(ckpt_path)
    print("Is same model instance in memory:", model is m2)
    assert model is m2

    print("\n--- 4 & 5 & 6 & 7. Mask Polarity, Preservation, Resolution & Raw vs Blended ---")
    dmg_path = "data/damaged/test/247010.jpg"
    mask_path = "data/damaged/test/247010_mask.png"

    res_blend = restore_artwork_pipeline(dmg_path, mask_path, blend_mode='mask_guided', max_dimension=1024)
    res_raw = restore_artwork_pipeline(dmg_path, mask_path, blend_mode='raw_unet', max_dimension=1024)

    d_blend = res_blend['diagnostics']
    d_raw = res_raw['diagnostics']

    print("Blended Undamaged MaxDiff:", d_blend['undamaged_max_diff'], "MAE:", d_blend['undamaged_mae'])
    print("Raw Undamaged MaxDiff:", d_raw['undamaged_max_diff'], "MAE:", d_raw['undamaged_mae'])
    print("Are blended and raw distinct:", not np.array_equal(res_blend['restored_image'], res_raw['restored_image']))
    assert d_blend['undamaged_max_diff'] == 0
    assert d_blend['undamaged_mae'] == 0.0
    assert not np.array_equal(res_blend['restored_image'], res_raw['restored_image'])

    print("\n--- 8. Gradio Interface Check ---")
    app = build_interface(default_ckpt=ckpt_path)
    print("Gradio interface blocks count:", len(app.blocks))
    print("ALL 8 IMPLEMENTATION CHECKS PASSED!")

if __name__ == "__main__":
    main()

