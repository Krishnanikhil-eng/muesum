import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# Add src to path
src_path = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_path))

def test_damage_simulator():
    print("--- 1. Testing Damage Simulator (damage_sim_museum.py) ---")
    from damage_sim_museum import simulate_damage_strong
    
    # Create synthetic RGB image (256x256x3)
    img_np = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    damaged, mask = simulate_damage_strong(img_np, seed=42, strength='strong')
    
    assert damaged.shape == img_np.shape, f"Damaged shape mismatch: {damaged.shape}"
    assert damaged.dtype == np.uint8, f"Damaged dtype mismatch: {damaged.dtype}"
    assert mask.shape == (256, 256), f"Mask shape mismatch: {mask.shape}"
    assert mask.dtype == np.uint8, f"Mask dtype mismatch: {mask.dtype}"
    print(f"Damage simulator OK: damaged shape={damaged.shape}, mask damaged pixels={np.sum(mask > 0)}")

def test_utils_and_metrics():
    print("\n--- 2. Testing Image Utilities & Metrics (utils.py) ---")
    from utils import tensor_to_uint8, psnr, safe_ssim
    from dataset import pad_to_multiple_of_32, unpad

    # Test tensor_to_uint8
    dummy_tensor = torch.rand(3, 100, 100)
    arr = tensor_to_uint8(dummy_tensor)
    assert arr.shape == (100, 100, 3)
    assert arr.dtype == np.uint8

    # Test pad & unpad
    dummy_img = np.zeros((130, 210, 3), dtype=np.uint8)
    padded, pad_info = pad_to_multiple_of_32(dummy_img)
    assert padded.shape[0] % 32 == 0 and padded.shape[1] % 32 == 0, f"Padding failed: {padded.shape}"
    restored = unpad(padded, pad_info)
    assert restored.shape == dummy_img.shape, f"Unpad shape mismatch: {restored.shape} vs {dummy_img.shape}"

    # Test PSNR & SSIM
    a = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    b = a.copy()
    val_psnr_identical = psnr(a, b)
    assert val_psnr_identical == 100.0, f"Expected 100.0 for identical images, got {val_psnr_identical}"
    val_ssim_identical = safe_ssim(a, b)
    assert abs(val_ssim_identical - 1.0) < 1e-4, f"Expected 1.0 for identical images, got {val_ssim_identical}"
    print(f"Utils & Metrics OK: pad_shape={padded.shape}, PSNR={val_psnr_identical}, SSIM={val_ssim_identical:.4f}")

def test_unet_model():
    print("\n--- 3. Testing ResNet34UNet Architecture (unet_model.py) ---")
    from unet_model import ResNet34UNet
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet34UNet(pretrained=False).to(device)
    model.eval()

    # Pass dummy batch [2, 3, 128, 128]
    inp = torch.rand(2, 3, 128, 128, device=device)
    with torch.no_grad():
        out = model(inp)
    
    assert out.shape == inp.shape, f"UNet output shape mismatch: {out.shape} vs {inp.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0, f"UNet output range outside [0, 1]: min={out.min()}, max={out.max()}"
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet34UNet OK: Input shape={list(inp.shape)} -> Output shape={list(out.shape)}, Trainable params={num_params:,}")

def test_losses():
    print("\n--- 4. Testing Losses (losses.py) ---")
    from losses import SobelEdgeLoss, EdgeLoss, tv_loss
    
    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)
    
    edge_loss = EdgeLoss()
    l_edge = edge_loss(pred, target)
    assert torch.isfinite(l_edge)

    sobel_loss = SobelEdgeLoss()
    l_sobel = sobel_loss(pred, target)
    assert torch.isfinite(l_sobel)

    l_tv = tv_loss(pred)
    assert torch.isfinite(l_tv)
    print(f"Losses OK: EdgeLoss={l_edge.item():.4f}, SobelEdgeLoss={l_sobel.item():.4f}, TVLoss={l_tv.item():.4f}")

def test_end_to_end_inference():
    print("\n--- 5. Testing End-to-End Inference Flow (inference.py) ---")
    import tempfile
    from inference import unet_restore
    from unet_model import ResNet34UNet
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        inp_img_path = tmp_path / "damaged.jpg"
        out_img_path = tmp_path / "restored.png"
        ckpt_path = tmp_path / "model.pth"

        # Save dummy image
        Image.fromarray(np.uint8(np.random.rand(150, 190, 3) * 255)).save(inp_img_path)

        # Save dummy checkpoint
        model = ResNet34UNet(pretrained=False)
        torch.save({'model': model.state_dict()}, ckpt_path)

        # Run unet_restore
        unet_restore(str(inp_img_path), str(out_img_path), str(ckpt_path))

        assert out_img_path.exists(), "Restored image was not created!"
        with Image.open(out_img_path) as res_img:
            assert res_img.size == (190, 150), f"Restored image dimensions mismatch: {res_img.size} vs (190, 150)"
            print(f"Inference Flow OK: restored image saved and verified at {res_img.size}")

def test_training_pipeline_mini():
    print("\n--- 6. Testing Mini Training Loop / Data Pipeline ---")
    import tempfile
    import pandas as pd
    from dataset import ArtifactDataset
    from train_unet import collate_batch
    from torch.utils.data import DataLoader
    from unet_model import ResNet34UNet

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        clean_dir = tmp_path / "clean"
        clean_dir.mkdir()

        # Create 2 synthetic clean images
        for obj_id in ["1001", "1002"]:
            img = Image.fromarray(np.uint8(np.random.rand(128, 128, 3) * 255))
            img.save(clean_dir / f"{obj_id}.jpg")

        meta_csv = tmp_path / "meta.csv"
        pd.DataFrame({"obj_id": ["1001", "1002"]}).to_csv(meta_csv, index=False)

        # Initialize dataset with simulate=True
        ds = ArtifactDataset(clean_dir=str(clean_dir), metadata_csv=str(meta_csv), simulate=True, sim_seed=42)
        assert len(ds) == 2, f"Dataset length mismatch: {len(ds)}"

        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_batch)
        batch = next(iter(loader))
        assert batch is not None, "Failed to load batch"
        
        inps, tgts, ids, pad_infos, orig_sizes, masks = batch
        assert inps.shape[0] == 2
        assert tgts.shape[0] == 2

        # 1 forward + backward step
        model = ResNet34UNet(pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        preds = model(inps)
        loss = torch.nn.functional.l1_loss(preds, tgts)
        loss.backward()
        optimizer.step()
        print(f"Mini Training Step OK: Loss = {loss.item():.4f}")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING MUSEUM ARTIFACT RESTORATION DIAGNOSTIC TEST SUITE")
    print("=" * 60)
    test_damage_simulator()
    test_utils_and_metrics()
    test_unet_model()
    test_losses()
    test_end_to_end_inference()
    test_training_pipeline_mini()
    print("\n" + "=" * 60)
    print("ALL 6 TESTS PASSED SUCCESSFULLY! EVERYTHING IS IN WORKING ORDER.")
    print("=" * 60)
