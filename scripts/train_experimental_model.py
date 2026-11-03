# scripts/train_experimental_model.py
"""
MUESUM — Experimental Texture Model Fine-Tuning
================================================
Fine-tunes the ResNet34-UNet initialized from the Epoch 8 baseline checkpoint
using the Multi-Term Composite Restoration Loss:
  L = L_L1_masked + 0.15 * L_SobelEdge + 0.01 * L_TV + 0.05 * L_VGGPerceptual

Key Safety & Reproducibility Invariants:
  - Base checkpoint 'checkpoints/baseline_run/best_model.pth' is read-only and NEVER modified.
  - Independent holdout set 'data/independent_holdout/' is NEVER used for training or validation.
  - Sequential CPU execution with batch size 2, 256x256 patches to prevent memory pressure.
  - Saves new weights exclusively to 'checkpoints/experimental_texture_model/'.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
import csv
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch import optim
from tqdm import tqdm
import torchvision.utils as vutils
from PIL import Image

# Add paths
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from dataset import ArtifactDataset, pad_to_multiple_of_32, unpad
from unet_model import ResNet34UNet
from composite_loss import CompositeRestorationLoss
from utils import ensure_dir, tensor_to_uint8, psnr, safe_ssim
from train_unet import collate_batch


def run_experimental_fine_tuning(
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 1.5e-5,
    max_train_samples: int = 60,
    max_val_samples: int = 20,
    seed: int = 42
):
    print("=" * 78)
    print("MUESUM — EXPERIMENTAL TEXTURE MODEL FINE-TUNING")
    print(f"Objective: Multi-Term Composite Loss (Masked L1 + Sobel Edge + TV + VGG Perceptual)")
    print("=" * 78)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    base_ckpt = root_dir / "checkpoints" / "baseline_run" / "best_model.pth"
    if not base_ckpt.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found at: {base_ckpt}")

    out_dir = root_dir / "checkpoints" / "experimental_texture_model"
    ensure_dir(str(out_dir))
    metrics_csv = out_dir / "training_metrics.csv"

    # 1. Load Baseline Checkpoint
    print(f"[Model] Initializing architecture with ResNet34-UNet...")
    model = ResNet34UNet(pretrained=False).to(device)
    print(f"[Model] Loading baseline weights from: {base_ckpt}")
    checkpoint = torch.load(str(base_ckpt), map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    baseline_val_loss = checkpoint.get("val_loss", 0.1057)
    baseline_val_ssim = checkpoint.get("val_ssim", 0.4908)
    print(f"[Model] Baseline starting reference: val_loss={baseline_val_loss:.4f}, val_ssim={baseline_val_ssim:.4f}")

    # 2. Setup Composite Loss & Optimizer
    print(f"[Loss] Initializing CompositeRestorationLoss with VGG-19 features...")
    loss_fn = CompositeRestorationLoss(
        lam_l1=1.0,
        lam_edge=0.15,
        lam_tv=0.01,
        lam_perc=0.05,
        damage_weight=4.0,
        use_perceptual=True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 3. Setup Datasets (Using 256x256 crops for efficiency and detail focus)
    clean_dir = root_dir / "data" / "clean"
    train_dmg_dir = root_dir / "data" / "damaged" / "train"
    val_dmg_dir = root_dir / "data" / "damaged" / "val"
    train_csv = root_dir / "data" / "splits" / "train_ids.csv"
    val_csv = root_dir / "data" / "splits" / "val_ids.csv"

    print(f"[Data] Loading training dataset (256x256 crops)...")
    train_ds = ArtifactDataset(
        clean_dir=str(clean_dir),
        metadata_csv=str(train_csv),
        damaged_dir=str(train_dmg_dir),
        patch_size=256,
        is_train=True
    )

    val_ds = ArtifactDataset(
        clean_dir=str(clean_dir),
        metadata_csv=str(val_csv),
        damaged_dir=str(val_dmg_dir),
        patch_size=256,
        is_train=False
    )

    if max_train_samples > 0 and len(train_ds) > max_train_samples:
        train_indices = np.random.choice(len(train_ds), size=max_train_samples, replace=False)
        train_subset = Subset(train_ds, train_indices)
    else:
        train_subset = train_ds

    if max_val_samples > 0 and len(val_ds) > max_val_samples:
        val_indices = np.random.choice(len(val_ds), size=max_val_samples, replace=False)
        val_subset = Subset(val_ds, val_indices)
    else:
        val_subset = val_ds

    print(f"[Data] Training samples: {len(train_subset)} | Validation samples: {len(val_subset)}")
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, collate_fn=collate_batch, num_workers=0)

    # Initialize metrics CSV
    fieldnames = [
        "epoch", "train_loss", "train_l1", "train_edge", "train_tv", "train_perc",
        "val_loss", "val_psnr", "val_ssim", "lr", "epoch_time_s"
    ]
    with open(metrics_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    best_val_ssim = -1.0
    start_epoch = 1
    best_ckpt_path = out_dir / "best_model.pth"
    latest_ckpt_path = out_dir / "latest_model.pth"

    if latest_ckpt_path.exists():
        print(f"[Model] Resuming fine-tuning from existing latest checkpoint: {latest_ckpt_path}")
        latest_ckpt = torch.load(str(latest_ckpt_path), map_location=device)
        model.load_state_dict(latest_ckpt["model"])
        if "optimizer" in latest_ckpt:
            optimizer.load_state_dict(latest_ckpt["optimizer"])
        start_epoch = latest_ckpt.get("epoch", 0) + 1
        best_val_ssim = latest_ckpt.get("val_ssim", -1.0)
        print(f"[Model] Resuming at Epoch {start_epoch}/{num_epochs} (Current Best SSIM: {best_val_ssim:.4f})")
    else:
        # Initialize metrics CSV header only if starting fresh
        with open(metrics_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # 4. Training Loop
    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_l1 = 0.0
        running_edge = 0.0
        running_tv = 0.0
        running_perc = 0.0
        batches = 0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            if batch is None:
                continue
            inp, tgt, obj_ids, pad_infos, orig_sizes, masks = batch
            inp = inp.to(device)
            tgt = tgt.to(device)
            masks = masks.to(device) if masks is not None else None

            optimizer.zero_grad()
            out = model(inp)

            loss, loss_dict = loss_fn(out, tgt, mask=masks)
            loss.backward()

            # Gradient clipping for stable training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss_dict["total_loss"]
            running_l1 += loss_dict["l1_loss"]
            running_edge += loss_dict["edge_loss"]
            running_tv += loss_dict["tv_loss"]
            running_perc += loss_dict["perc_loss"]
            batches += 1

            pbar.set_postfix({
                "loss": f"{loss_dict['total_loss']:.4f}",
                "edge": f"{loss_dict['edge_loss']:.4f}",
                "perc": f"{loss_dict['perc_loss']:.4f}"
            })

        scheduler.step()
        train_time = time.time() - t0
        avg_train_loss = running_loss / max(1, batches)
        avg_l1 = running_l1 / max(1, batches)
        avg_edge = running_edge / max(1, batches)
        avg_tv = running_tv / max(1, batches)
        avg_perc = running_perc / max(1, batches)

        # 5. Validation Loop
        model.eval()
        val_losses = []
        val_psnrs = []
        val_ssims = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                if batch is None:
                    continue
                inp_v, tgt_v, obj_v, pad_v, orig_v, mask_v = batch
                inp_v = inp_v.to(device)
                tgt_v = tgt_v.to(device)
                mask_v = mask_v.to(device) if mask_v is not None else None

                out_v = model(inp_v)
                v_loss, _ = loss_fn(out_v, tgt_v, mask=mask_v)
                val_losses.append(v_loss.item())

                # Compute quantitative metrics
                out_np = tensor_to_uint8(out_v[0].cpu())
                tgt_np = tensor_to_uint8(tgt_v[0].cpu())
                p = psnr(tgt_np, out_np)
                s = safe_ssim(tgt_np, out_np)
                val_psnrs.append(p)
                val_ssims.append(s if s >= 0 else 0.0)

        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        avg_val_psnr = float(np.mean(val_psnrs)) if val_psnrs else 0.0
        avg_val_ssim = float(np.mean(val_ssims)) if val_ssims else 0.0
        curr_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\n[Epoch {epoch:d}] Train Loss: {avg_train_loss:.4f} (L1: {avg_l1:.4f}, Edge: {avg_edge:.4f}, Perc: {avg_perc:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} | Val PSNR: {avg_val_psnr:.2f} dB | Val SSIM: {avg_val_ssim:.4f} | Time: {train_time:.1f}s"
        )

        # Save Metrics Row
        metric_data = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "train_l1": round(avg_l1, 6),
            "train_edge": round(avg_edge, 6),
            "train_tv": round(avg_tv, 6),
            "train_perc": round(avg_perc, 6),
            "val_loss": round(avg_val_loss, 6),
            "val_psnr": round(avg_val_psnr, 4),
            "val_ssim": round(avg_val_ssim, 6),
            "lr": round(curr_lr, 8),
            "epoch_time_s": int(round(train_time))
        }
        with open(metrics_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(metric_data)

        # Save Checkpoint
        ckpt_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "val_psnr": avg_val_psnr,
            "val_ssim": avg_val_ssim,
            "config": {
                "lam_l1": 1.0,
                "lam_edge": 0.15,
                "lam_tv": 0.01,
                "lam_perc": 0.05,
                "damage_weight": 4.0,
                "learning_rate": learning_rate,
                "base_checkpoint": "checkpoints/baseline_run/best_model.pth"
            }
        }
        torch.save(ckpt_payload, str(latest_ckpt_path))

        if avg_val_ssim > best_val_ssim:
            best_val_ssim = avg_val_ssim
            torch.save(ckpt_payload, str(best_ckpt_path))
            print(f"  [*] New best experimental checkpoint saved! (SSIM: {best_val_ssim:.4f}) -> {best_ckpt_path}")

        # Clean up memory after epoch
        gc.collect()

    print("\n" + "=" * 78)
    print("EXPERIMENTAL FINE-TUNING COMPLETED SUCCESSFULLY!")
    print(f"Best Experimental Checkpoint: {best_ckpt_path} (Best SSIM: {best_val_ssim:.4f})")
    print(f"Metrics Logged To:            {metrics_csv}")
    print("=" * 78)


if __name__ == "__main__":
    run_experimental_fine_tuning(
        num_epochs=3,
        batch_size=2,
        learning_rate=1.5e-5,
        max_train_samples=60,
        max_val_samples=20,
        seed=42
    )
