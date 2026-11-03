import sys
import time
import psutil
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from dataset import ArtifactDataset, unpad
from unet_model import ResNet34UNet
from train_unet import collate_batch
from utils import tensor_to_uint8, psnr, safe_ssim

def run_smoke_test():
    print("=" * 65)
    print("MUESUM — PHASE C: HARDWARE-AWARE TRAINING SMOKE TEST")
    print("=" * 65)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    mem_start = psutil.virtual_memory()
    print(f"Initial RAM: Available={mem_start.available / (1024**3):.2f} GB | Total={mem_start.total / (1024**3):.2f} GB ({mem_start.percent}% used)")
    
    # 1. Initialize dataset with 256x256 patches and real precomputed damaged files
    clean_dir = root_dir / "data" / "clean"
    train_dmg = root_dir / "data" / "damaged" / "train"
    val_dmg = root_dir / "data" / "damaged" / "val"
    train_csv = root_dir / "data" / "splits" / "train_ids.csv"
    val_csv = root_dir / "data" / "splits" / "val_ids.csv"
    
    print("\n--- 1. Testing DataLoader with real museum artwork dataset ---")
    train_ds = ArtifactDataset(
        clean_dir=str(clean_dir),
        metadata_csv=str(train_csv),
        damaged_dir=str(train_dmg),
        patch_size=256,
        damage_aware_prob=0.7,
        is_train=True
    )
    val_ds = ArtifactDataset(
        clean_dir=str(clean_dir),
        metadata_csv=str(val_csv),
        damaged_dir=str(val_dmg),
        patch_size=256,
        is_train=False
    )
    print(f"Train dataset size: {len(train_ds)} samples")
    print(f"Val dataset size:   {len(val_ds)} samples")
    
    batch_size = 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)
    
    # Get first batch
    t_batch_start = time.time()
    batch = next(iter(train_loader))
    t_batch_load = time.time() - t_batch_start
    assert batch is not None, "Failed to load batch from train_loader"
    inps, tgts, ids, pad_infos, orig_sizes, masks = batch
    print(f"Batch loaded in {t_batch_load:.3f} s:")
    print(f"  Inputs shape: {list(inps.shape)} (dtype: {inps.dtype}, range: [{inps.min():.2f}, {inps.max():.2f}])")
    print(f"  Targets shape: {list(tgts.shape)} (dtype: {tgts.dtype}, range: [{tgts.min():.2f}, {tgts.max():.2f}])")
    print(f"  Masks shape: {list(masks.shape) if masks is not None else None}")
    print(f"  Sample IDs: {ids}")
    assert inps.shape == (2, 3, 256, 256), f"Unexpected input shape: {inps.shape}"
    assert tgts.shape == (2, 3, 256, 256), f"Unexpected target shape: {tgts.shape}"
    assert masks.shape == (2, 1, 256, 256), f"Unexpected mask shape: {masks.shape}"
    
    # 2. Forward Propagation
    print("\n--- 2. Testing Forward Propagation ---")
    model = ResNet34UNet(pretrained=False).to(device)
    model.train()
    
    t_fwd_start = time.time()
    preds = model(inps.to(device))
    t_fwd = time.time() - t_fwd_start
    print(f"Forward pass completed in {t_fwd:.3f} s:")
    print(f"  Predictions shape: {list(preds.shape)}")
    print(f"  Output range: [{preds.min():.4f}, {preds.max():.4f}]")
    assert preds.shape == tgts.shape, f"Prediction shape mismatch: {preds.shape} vs {tgts.shape}"
    assert torch.isfinite(preds).all(), "Predictions contain NaN or Inf!"
    
    # 3. Loss & Backward Propagation
    print("\n--- 3. Testing Loss Computation & Backward Propagation ---")
    opt = torch.optim.Adam(model.parameters(), lr=3e-5)
    opt.zero_grad()
    
    # Check masked L1 loss
    mask_c = masks.to(device).repeat(1, preds.shape[1], 1, 1)
    diff = (preds - tgts.to(device)).abs()
    masked_sum = (diff * mask_c).sum()
    denom = mask_c.sum()
    if denom > 0:
        loss = masked_sum / (denom + 1e-6)
    else:
        loss = diff.mean()
        
    print(f"Computed Masked L1 Loss = {loss.item():.6f}")
    assert torch.isfinite(loss), "Loss is not finite!"
    
    # Save initial weights of a layer to verify update
    initial_weight = model.final_conv[2].weight.clone()
    
    t_bwd_start = time.time()
    loss.backward()
    t_bwd = time.time() - t_bwd_start
    print(f"Backward pass completed in {t_bwd:.3f} s")
    
    # 4. Parameter Update Verification
    print("\n--- 4. Testing Optimizer Parameter Updates ---")
    opt.step()
    updated_weight = model.final_conv[2].weight
    weight_diff = (updated_weight - initial_weight).abs().sum().item()
    print(f"Sum of absolute weight update in final layer: {weight_diff:.8f}")
    assert weight_diff > 0.0, "Model weights were NOT updated by optimizer!"
    print("[OK] Optimizer step verified: model parameters updated successfully.")
    
    # 5. Checkpoint Saving & Loading
    print("\n--- 5. Testing Checkpoint Saving & Loading ---")
    ckpt_dir = root_dir / "checkpoints" / "smoke_test"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / "test_ckpt.pth"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": 0}, ckpt_file)
    assert ckpt_file.exists(), "Checkpoint file was not created!"
    print(f"Saved checkpoint: {ckpt_file.name} ({ckpt_file.stat().st_size / (1024**2):.2f} MB)")
    
    # Reload model
    new_model = ResNet34UNet(pretrained=False).to(device)
    state = torch.load(ckpt_file, map_location=device)
    new_model.load_state_dict(state["model"])
    print("[OK] Checkpoint successfully reloaded and verified.")
    
    # 6. Throughput Benchmarking (10 real batches)
    print("\n--- 6. Measuring Real Batch Throughput (10 batches) ---")
    batch_times = []
    mem_during = []
    
    for i, b in enumerate(train_loader):
        if i >= 10:
            break
        t0 = time.time()
        b_inp, b_tgt, _, _, _, b_mask = b
        opt.zero_grad()
        b_out = model(b_inp.to(device))
        b_diff = (b_out - b_tgt.to(device)).abs()
        b_mask_c = b_mask.to(device).repeat(1, b_out.shape[1], 1, 1)
        b_denom = b_mask_c.sum()
        if b_denom > 0:
            b_loss = (b_diff * b_mask_c).sum() / (b_denom + 1e-6)
        else:
            b_loss = b_diff.mean()
        b_loss.backward()
        opt.step()
        dt = time.time() - t0
        batch_times.append(dt)
        mem_during.append(psutil.virtual_memory().available / (1024**3))
        print(f"  Batch {i+1:02d}/10: {dt:.3f} s | Loss: {b_loss.item():.4f} | RAM Avail: {mem_during[-1]:.2f} GB")
        
    avg_batch_time = np.mean(batch_times)
    sec_per_sample = avg_batch_time / batch_size
    print(f"\nThroughput Statistics:")
    print(f"  Average batch processing time (batch_size={batch_size}): {avg_batch_time:.3f} s")
    print(f"  Average time per sample:                                  {sec_per_sample:.3f} s")
    print(f"  Samples per second:                                       {1.0 / sec_per_sample:.2f} samples/s")
    print(f"  Minimum available RAM during training:                   {min(mem_during):.2f} GB")
    
    # 7. Validation Loop Verification (5 batches)
    print("\n--- 7. Testing Validation Pipeline (5 batches) ---")
    model.eval()
    val_psnrs, val_ssims = [], []
    with torch.no_grad():
        for i, b in enumerate(val_loader):
            if i >= 5:
                break
            v_inp, v_tgt, v_ids, _, _, _ = b
            v_out = model(v_inp.to(device))
            for j in range(v_out.size(0)):
                out_np = tensor_to_uint8(v_out[j].cpu())
                tgt_np = tensor_to_uint8(v_tgt[j].cpu())
                p = psnr(tgt_np, out_np)
                s = safe_ssim(tgt_np, out_np)
                val_psnrs.append(p)
                val_ssims.append(s)
                print(f"  Val Sample {len(val_psnrs)} (ID: {v_ids[j]}): PSNR = {p:.2f} dB, SSIM = {s:.4f}")
                
    print(f"Validation inference verified: Avg PSNR = {np.mean(val_psnrs):.2f} dB, Avg SSIM = {np.mean(val_ssims):.4f}")
    
    # Clean up smoke test checkpoint
    if ckpt_file.exists():
        ckpt_file.unlink()
    if ckpt_dir.exists():
        ckpt_dir.rmdir()
        
    # 8. Epoch Duration Estimates
    total_train_samples = 1600
    total_val_samples = 200
    num_train_batches = (total_train_samples + batch_size - 1) // batch_size
    
    # For validation, assume center patch evaluation (~0.2s per val sample forward only)
    val_forward_time_per_sample = t_fwd / batch_size
    total_val_time_per_epoch = total_val_samples * val_forward_time_per_sample
    
    train_time_per_epoch = num_train_batches * avg_batch_time
    total_epoch_time = train_time_per_epoch + total_val_time_per_epoch
    
    print("\n" + "=" * 65)
    print("ESTIMATED TRAINING DURATION FOR BASELINE RESNET34-UNET")
    print(f"(1,600 Train Images + 200 Validation Images, Batch Size={batch_size}, Patch Size=256x256)")
    print("=" * 65)
    print(f"  Training time per epoch:   {train_time_per_epoch / 60.0:.2f} min ({train_time_per_epoch:.1f} s)")
    print(f"  Validation time per epoch: {total_val_time_per_epoch / 60.0:.2f} min ({total_val_time_per_epoch:.1f} s)")
    print(f"  Total time per epoch:      {total_epoch_time / 60.0:.2f} min ({total_epoch_time:.1f} s)")
    print("-" * 65)
    print(f"   5 Epochs: { (total_epoch_time * 5) / 60.0:.1f} min ({ (total_epoch_time * 5) / 3600.0:.2f} hours )")
    print(f"  10 Epochs: { (total_epoch_time * 10) / 60.0:.1f} min ({ (total_epoch_time * 10) / 3600.0:.2f} hours )")
    print(f"  20 Epochs: { (total_epoch_time * 20) / 60.0:.1f} min ({ (total_epoch_time * 20) / 3600.0:.2f} hours )")
    print("=" * 65)

if __name__ == "__main__":
    run_smoke_test()

