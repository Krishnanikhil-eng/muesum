"""
scripts/run_pipeline.py
Central Pipeline Manager — Museum Artifact Restoration
=======================================================
Coordinates: download → split → damage → train → evaluate.

Usage:
    python scripts/run_pipeline.py --stage download --target 2000
    python scripts/run_pipeline.py --stage prepare
    python scripts/run_pipeline.py --stage damage
    python scripts/run_pipeline.py --stage train
    python scripts/run_pipeline.py --stage evaluate
    python scripts/run_pipeline.py --stage all --target 2000

Use --yes to skip confirmation prompts (e.g., in CI/automated runs).
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
PYTHON = sys.executable  # same interpreter that is running this script
ROOT   = Path(__file__).resolve().parent.parent  # project root


def _run(cmd: list, cwd=None):
    """Run a subprocess and stream its output. Raises SystemExit on failure."""
    print(f"\n[pipeline] Running: {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        text=True,
    )
    if result.returncode != 0:
        print(f"[pipeline] ✗ Command failed with code {result.returncode}")
        raise SystemExit(result.returncode)


def _ask_confirm(message: str, auto_yes: bool = False) -> bool:
    """Ask user for confirmation. Returns True if confirmed."""
    if auto_yes:
        print(f"[pipeline] Auto-confirmed: {message}")
        return True
    print(f"\n{'─' * 60}")
    print(f"  CONFIRMATION REQUIRED")
    print(f"  {message}")
    print(f"{'─' * 60}")
    answer = input("  Proceed? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def _count_valid_images(clean_dir: Path) -> int:
    """Quick count of image files in clean_dir."""
    if not clean_dir.exists():
        return 0
    return sum(
        1 for f in clean_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


def _splits_exist(splits_dir: Path) -> bool:
    """Return True if all three split files exist and are non-empty."""
    for name in ("train_ids.csv", "val_ids.csv", "test_ids.csv"):
        p = splits_dir / name
        if not p.exists() or p.stat().st_size == 0:
            return False
    return True


def _damaged_exists(damaged_dir: Path) -> bool:
    """Return True if precomputed damaged train directory has images."""
    train_dam = damaged_dir / "train"
    if not train_dam.exists():
        return False
    imgs = [f for f in train_dam.iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
            and "_mask" not in f.stem]
    return len(imgs) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Stage implementations
# ──────────────────────────────────────────────────────────────────────────────
def stage_download(target: int, auto_yes: bool, out_dir: str, metadata: str):
    print(f"\n{'═' * 60}")
    print(f"  STAGE: DOWNLOAD  (target={target} images)")
    print(f"{'═' * 60}")

    clean_path = ROOT / out_dir
    current = _count_valid_images(clean_path)
    print(f"  Current valid images: {current}")
    print(f"  Target:               {target}")

    if current >= target:
        print(f"  ✓ Target already reached. Nothing to download.")
        return

    needed = target - current
    msg = (
        f"Download ~{needed} more images from the Metropolitan Museum API.\n"
        f"  Target: {target} total images\n"
        f"  Estimated time: {max(1, needed // 8)}-{max(2, needed // 4)} minutes depending on network speed."
    )
    if not _ask_confirm(msg, auto_yes):
        print("  [skip] Download cancelled by user.")
        return

    _run([
        PYTHON, str(ROOT / "api" / "api.py"),
        "--limit", str(target),
        "--out-dir", out_dir,
        "--metadata", metadata,
        "--resume",
    ])


def stage_prepare(target: int, auto_yes: bool, metadata: str, splits_dir: str, clean_dir: str):
    print(f"\n{'═' * 60}")
    print(f"  STAGE: PREPARE (splits)  (target={target})")
    print(f"{'═' * 60}")

    splits_path = ROOT / splits_dir
    clean_path = ROOT / clean_dir
    current = _count_valid_images(clean_path)

    print(f"  Valid images available: {current}")
    if current < target:
        print(f"  [warn] Only {current} images available but target is {target}.")
        if not _ask_confirm(f"Generate splits for {current} images instead of {target}?", auto_yes):
            print("  [skip] Prepare cancelled.")
            return
        actual_total = current
    else:
        actual_total = target

    _run([
        PYTHON, str(ROOT / "scripts" / "create_small_split.py"),
        "--metadata", metadata,
        "--out-dir", splits_dir,
        "--total", str(actual_total),
        "--clean-dir", clean_dir,
    ])

    # Validate splits
    print("\n  Running dataset manager for validation ...")
    _run([
        PYTHON, str(ROOT / "scripts" / "dataset_manager.py"),
        "--clean-dir", clean_dir,
        "--metadata", metadata,
        "--splits-dir", splits_dir,
        "--target", str(actual_total),
    ])


def stage_damage(splits_dir: str, clean_dir: str, damaged_root: str, auto_yes: bool):
    print(f"\n{'═' * 60}")
    print(f"  STAGE: DAMAGE SIMULATION")
    print(f"{'═' * 60}")

    splits_path = ROOT / splits_dir
    if not _splits_exist(splits_path):
        print(f"  ✗ Split files not found in {splits_path}. Run --stage prepare first.")
        raise SystemExit(1)

    damaged_path = ROOT / damaged_root
    if _damaged_exists(damaged_path) and not auto_yes:
        if not _ask_confirm("Damaged images already exist. Re-run precompute? (existing files are skipped)", auto_yes):
            print("  [skip] Damage precompute skipped.")
            return

    msg = (
        "Generate physically-simulated damaged images + masks.\n"
        "  This may take several minutes for large datasets."
    )
    if not _ask_confirm(msg, auto_yes):
        print("  [skip] Damage simulation cancelled.")
        return

    # precompute_damage.py expects sys.path to include src/ for the damage_sim import
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    train_csv = str(ROOT / splits_dir / "train_ids.csv")
    val_csv   = str(ROOT / splits_dir / "val_ids.csv")
    test_csv  = str(ROOT / splits_dir / "test_ids.csv")

    cmd = [
        PYTHON, str(ROOT / "src" / "precompute_damage.py"),
        "--clean-dir",  clean_dir,
        "--train-csv",  train_csv,
        "--val-csv",    val_csv,
        "--out-root",   damaged_root,
        "--strength",   "strong",
        "--max-types",  "4",
        "--write-mask",
    ]
    if Path(test_csv).exists():
        cmd += ["--test-csv", test_csv]

    print(f"\n[pipeline] Running: {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True)
    if result.returncode != 0:
        print(f"[pipeline] ✗ Precompute failed with code {result.returncode}")
        raise SystemExit(result.returncode)


def stage_train(
    exp_name: str,
    splits_dir: str,
    clean_dir: str,
    damaged_root: str,
    epochs: int,
    batch_size: int,
    lr: float,
    use_perceptual: bool,
    pretrained: bool,
    auto_yes: bool,
):
    print(f"\n{'═' * 60}")
    print(f"  STAGE: TRAIN  (experiment: {exp_name})")
    print(f"{'═' * 60}")

    out_dir = f"outputs/experiments/{exp_name}"
    out_path = ROOT / out_dir

    if (out_path / "best_model.pth").exists():
        if not _ask_confirm(
            f"A checkpoint already exists in {out_dir}. Start a new training run? (previous checkpoint is NOT deleted)",
            auto_yes
        ):
            print("  [skip] Training skipped.")
            return

    gpu_msg = ""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_msg = f" using GPU: {torch.cuda.get_device_name(0)}"
        else:
            gpu_msg = " using CPU (training will be very slow)"
    except Exception:
        pass

    msg = (
        f"Train ResNet34-UNet for {epochs} epochs (batch={batch_size}, lr={lr}).\n"
        f"  Perceptual loss: {'enabled' if use_perceptual else 'disabled'}\n"
        f"  Pretrained encoder: {'yes' if pretrained else 'no'}\n"
        f"  Output: {out_dir}\n"
        f"  Device: {gpu_msg or 'auto-detected at runtime'}"
    )
    if not _ask_confirm(msg, auto_yes):
        print("  [skip] Training cancelled.")
        return

    cmd = [
        PYTHON, str(ROOT / "src" / "train_unet.py"),
        "--clean-dir",     clean_dir,
        "--train-csv",     str(ROOT / splits_dir / "train_ids.csv"),
        "--val-csv",       str(ROOT / splits_dir / "val_ids.csv"),
        "--damaged-dir",   str(ROOT / damaged_root / "train"),
        "--damaged-dir-val", str(ROOT / damaged_root / "val"),
        "--out-dir",       out_dir,
        "--epochs",        str(epochs),
        "--batch-size",    str(batch_size),
        "--lr",            str(lr),
        "--num-workers",   "0",   # safe default for Windows
        "--val-interval",  "1",
    ]
    if use_perceptual:
        cmd += ["--use-perceptual", "--lam-l1", "1.0", "--lam-perc", "0.8", "--lam-edge", "1.0"]
    if pretrained:
        cmd.append("--pretrained")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\n[pipeline] Running: {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def stage_evaluate(
    exp_name: str,
    splits_dir: str,
    clean_dir: str,
    damaged_root: str,
    auto_yes: bool,
):
    print(f"\n{'═' * 60}")
    print(f"  STAGE: EVALUATE  (experiment: {exp_name})")
    print(f"{'═' * 60}")

    ckpt_path = ROOT / "outputs" / "experiments" / exp_name / "best_model.pth"
    if not ckpt_path.exists():
        print(f"  ✗ Checkpoint not found: {ckpt_path}")
        print(f"    Run --stage train first.")
        raise SystemExit(1)

    # If an eval script exists, call it; otherwise report the metrics.csv
    metrics_path = ckpt_path.parent / "metrics.csv"
    if metrics_path.exists():
        print(f"\n  Training metrics (from {metrics_path}):")
        with open(str(metrics_path)) as f:
            lines = f.readlines()
        for line in lines[:3]:  # header + first two rows
            print(f"    {line.rstrip()}")
        if len(lines) > 3:
            print(f"    [last epoch]: {lines[-1].rstrip()}")
    else:
        print("  No metrics.csv found yet.")

    print(f"\n  ✓ Best checkpoint: {ckpt_path}")
    print(f"  Run inference with:")
    print(f"    python src/inference.py --inp_path <damaged.jpg> --out_path <out.png> --ckpt_path {ckpt_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Central pipeline manager for Museum Artifact Restoration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stage options:
  download   Download images from Met API (respects existing downloads)
  prepare    Generate train/val/test splits
  damage     Precompute damaged images and masks
  train      Train the ResNet34-UNet model
  evaluate   Report metrics from a trained experiment
  all        Run all stages in sequence

Examples:
  python scripts/run_pipeline.py --stage download --target 2000
  python scripts/run_pipeline.py --stage prepare --target 2000
  python scripts/run_pipeline.py --stage damage
  python scripts/run_pipeline.py --stage train --exp-name exp_2000_baseline --epochs 20
  python scripts/run_pipeline.py --stage all --target 2000 --yes
        """,
    )

    # Stage
    parser.add_argument(
        "--stage",
        required=True,
        choices=["download", "prepare", "damage", "train", "evaluate", "all"],
        help="Pipeline stage to execute",
    )

    # Data
    parser.add_argument("--target",       type=int,   default=2000,            help="Target dataset size")
    parser.add_argument("--clean-dir",    default="data/clean",                help="Clean images directory")
    parser.add_argument("--metadata",     default="data/metadata.csv",         help="Metadata CSV path")
    parser.add_argument("--splits-dir",   default="data/splits",               help="Splits directory")
    parser.add_argument("--damaged-root", default="data/damaged",              help="Damaged images root")

    # Training
    parser.add_argument("--exp-name",     default="exp_2000_baseline",         help="Experiment name (output subfolder)")
    parser.add_argument("--epochs",       type=int,   default=20,              help="Training epochs")
    parser.add_argument("--batch-size",   type=int,   default=4,               help="Batch size")
    parser.add_argument("--lr",           type=float, default=3e-5,            help="Learning rate")
    parser.add_argument("--use-perceptual", action="store_true",               help="Enable perceptual loss")
    parser.add_argument("--pretrained",   action="store_true",                 help="Use pretrained ResNet34 encoder")

    # Control
    parser.add_argument("--yes",          action="store_true",                 help="Auto-confirm all prompts (non-interactive)")

    args = parser.parse_args()

    stages_to_run = (
        ["download", "prepare", "damage", "train", "evaluate"]
        if args.stage == "all"
        else [args.stage]
    )

    print(f"\n{'█' * 60}")
    print(f"  MUSEUM ARTIFACT RESTORATION — PIPELINE MANAGER")
    print(f"  Stage(s): {', '.join(stages_to_run)}")
    print(f"{'█' * 60}")

    for stage in stages_to_run:
        if stage == "download":
            stage_download(
                target=args.target,
                auto_yes=args.yes,
                out_dir=args.clean_dir,
                metadata=args.metadata,
            )
        elif stage == "prepare":
            stage_prepare(
                target=args.target,
                auto_yes=args.yes,
                metadata=args.metadata,
                splits_dir=args.splits_dir,
                clean_dir=args.clean_dir,
            )
        elif stage == "damage":
            stage_damage(
                splits_dir=args.splits_dir,
                clean_dir=args.clean_dir,
                damaged_root=args.damaged_root,
                auto_yes=args.yes,
            )
        elif stage == "train":
            stage_train(
                exp_name=args.exp_name,
                splits_dir=args.splits_dir,
                clean_dir=args.clean_dir,
                damaged_root=args.damaged_root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                use_perceptual=args.use_perceptual,
                pretrained=args.pretrained,
                auto_yes=args.yes,
            )
        elif stage == "evaluate":
            stage_evaluate(
                exp_name=args.exp_name,
                splits_dir=args.splits_dir,
                clean_dir=args.clean_dir,
                damaged_root=args.damaged_root,
                auto_yes=args.yes,
            )

    print(f"\n{'█' * 60}")
    print(f"  Pipeline stage(s) complete.")
    print(f"{'█' * 60}\n")


if __name__ == "__main__":
    main()

