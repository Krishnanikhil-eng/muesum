"""
scripts/dataset_manager.py
Dataset Health Monitor for Museum Artifact Restoration
======================================================
Validates images, metadata, splits, and reports dataset readiness.

Usage:
    python scripts/dataset_manager.py
    python scripts/dataset_manager.py --target 5000
    python scripts/dataset_manager.py --clean-dir data/clean --metadata data/metadata.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from collections import Counter

# PIL for image validation
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[warn] Pillow not installed - image validation will be skipped.")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _validate_image(path: Path) -> tuple:
    """
    Returns (is_valid: bool, width: int, height: int, error: str).
    """
    if not PIL_AVAILABLE:
        return True, 0, 0, ""
    try:
        with Image.open(str(path)) as img:
            img.verify()
        with Image.open(str(path)) as img:
            w, h = img.size
        return True, w, h, ""
    except Exception as e:
        return False, 0, 0, str(e)


def _load_csv_ids(csv_path: Path) -> list:
    """Load obj_id values from a CSV split file."""
    if not csv_path.exists():
        return []
    ids = []
    with open(str(csv_path), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            oid = row.get("obj_id", "").strip()
            if oid:
                ids.append(oid)
    return ids


def _load_metadata(meta_path: Path) -> list:
    """Load all metadata rows as list of dicts."""
    if not meta_path.exists():
        return []
    rows = []
    with open(str(meta_path), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────
def run_analysis(
    clean_dir: str = "data/clean",
    metadata_file: str = "data/metadata.csv",
    splits_dir: str = "data/splits",
    damaged_dir: str = "data/damaged",
    target: int = 0,
):
    clean_path = Path(clean_dir)
    meta_path = Path(metadata_file)
    splits_path = Path(splits_dir)
    damaged_path = Path(damaged_dir)

    print("\n" + "=" * 65)
    print("  MUSEUM ARTIFACT RESTORATION - DATASET MANAGER")
    print("=" * 65)

    # ── 1. Clean images on disk ──────────────────────────────────────────────
    print("\n[1/6] Clean Images on Disk")
    print(f"      Directory: {clean_path}")

    if not clean_path.exists():
        print("      [ERR] Directory does not exist")
        all_clean_files = []
    else:
        all_clean_files = [
            f for f in clean_path.iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]

    valid_imgs = []
    corrupt_imgs = []
    small_imgs = []
    dim_widths = []
    dim_heights = []

    for fpath in all_clean_files:
        ok, w, h, err = _validate_image(fpath)
        if not ok:
            corrupt_imgs.append((fpath, err))
        elif w < 100 or h < 100:
            small_imgs.append((fpath, w, h))
        else:
            valid_imgs.append(fpath)
            dim_widths.append(w)
            dim_heights.append(h)

    print(f"      Total files found:  {len(all_clean_files)}")
    print(f"      [OK]  Valid images:     {len(valid_imgs)}")
    print(f"      [ERR] Corrupt images:   {len(corrupt_imgs)}")
    print(f"      [WARN] Too small (<100): {len(small_imgs)}")
    if dim_widths:
        print(f"      Avg dimensions:    {int(sum(dim_widths)/len(dim_widths))} x {int(sum(dim_heights)/len(dim_heights))} px")
        print(f"      Size range:        [{min(dim_widths)}x{min(dim_heights)}] -> [{max(dim_widths)}x{max(dim_heights)}]")

    if corrupt_imgs:
        print(f"\n      Corrupt files (first 5):")
        for fpath, err in corrupt_imgs[:5]:
            print(f"        {fpath.name}: {err}")

    # ── 2. Metadata CSV ──────────────────────────────────────────────────────
    print("\n[2/6] Metadata CSV")
    print(f"      File: {meta_path}")

    meta_rows = _load_metadata(meta_path)
    if not meta_rows:
        print("      [ERR] No metadata found (file missing or empty)")
    else:
        meta_ids = [r.get("object_id", "").strip() for r in meta_rows]
        meta_id_set = set(meta_ids)
        dup_ids = [k for k, v in Counter(meta_ids).items() if v > 1 and k]
        print(f"      Total records:     {len(meta_rows)}")
        print(f"      Unique object IDs: {len(meta_id_set)}")
        if dup_ids:
            print(f"      [WARN] Duplicate IDs:   {len(dup_ids)}  (e.g. {dup_ids[:3]})")
        else:
            print(f"      [OK] No duplicate IDs")

        # Check for records whose image file is missing
        missing_imgs = []
        for row in meta_rows:
            img_path = Path(row.get("image_path", ""))
            if not img_path.exists():
                missing_imgs.append(row.get("object_id", "?"))
        if missing_imgs:
            print(f"      [WARN] Records with missing image file: {len(missing_imgs)}")
        else:
            print(f"      [OK] All metadata records have matching image files")

        # Category distribution
        classifications = [r.get("classification", "Unknown") or "Unknown" for r in meta_rows]
        top_cats = Counter(classifications).most_common(8)
        print(f"\n      Top classifications:")
        for cat, cnt in top_cats:
            bar = "#" * min(cnt // max(1, len(meta_rows) // 20), 20)
            print(f"        {cat[:35]:<35} {cnt:>5}  {bar}")

        # Public domain
        pub_domain = sum(1 for r in meta_rows if str(r.get("is_public_domain", "")).lower() in ("true", "1", "yes"))
        print(f"\n      Public domain images: {pub_domain} / {len(meta_rows)}")

    # ── 3. Dataset Splits ────────────────────────────────────────────────────
    print("\n[3/6] Dataset Splits")
    print(f"      Directory: {splits_path}")

    split_files = {
        "train": splits_path / "train_ids.csv",
        "val":   splits_path / "val_ids.csv",
        "test":  splits_path / "test_ids.csv",
    }

    all_split_ids = set()
    split_counts = {}
    for split_name, split_file in split_files.items():
        ids = _load_csv_ids(split_file)
        split_counts[split_name] = len(ids)
        all_split_ids.update(ids)
        status = "[OK]" if ids else "[ERR] (missing)"
        print(f"      {split_name:>5}_ids.csv: {len(ids):>6} IDs  {status}")

    # Check for overlap
    train_ids = set(_load_csv_ids(split_files["train"]))
    val_ids   = set(_load_csv_ids(split_files["val"]))
    test_ids  = set(_load_csv_ids(split_files["test"]))
    tv_overlap = train_ids & val_ids
    tt_overlap = train_ids & test_ids
    vt_overlap = val_ids & test_ids
    if tv_overlap or tt_overlap or vt_overlap:
        print(f"      [WARN] Data leakage detected!")
        if tv_overlap: print(f"         train ∩ val:  {len(tv_overlap)} IDs")
        if tt_overlap: print(f"         train ∩ test: {len(tt_overlap)} IDs")
        if vt_overlap: print(f"         val ∩ test:   {len(vt_overlap)} IDs")
    else:
        if all_split_ids:
            print(f"      [OK] No data leakage between splits")

    # ── 4. Damaged images ────────────────────────────────────────────────────
    print("\n[4/6] Precomputed Damaged Images")
    for split_name in ("train", "val", "test"):
        dam_split = damaged_path / split_name
        if dam_split.exists():
            imgs = [f for f in dam_split.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png") and "_mask" not in f.stem]
            masks = [f for f in dam_split.iterdir() if "_mask" in f.stem]
            print(f"      data/damaged/{split_name}/: {len(imgs)} images, {len(masks)} masks")
        else:
            print(f"      data/damaged/{split_name}/: [ERR] (not created yet)")

    # ── 5. Preprocessing status ──────────────────────────────────────────────
    print("\n[5/6] Training Readiness")
    if train_ids:
        train_damaged = damaged_path / "train"
        already_processed = set()
        if train_damaged.exists():
            already_processed = {f.stem for f in train_damaged.iterdir() if "_mask" not in f.stem}
        needs_processing = train_ids - already_processed
        print(f"      Train split size:         {len(train_ids)}")
        print(f"      Already preprocessed:     {len(already_processed)}")
        if needs_processing:
            print(f"      Still need damage sim:    {len(needs_processing)} images")
        else:
            print(f"      [OK] All training images preprocessed")
    else:
        print("      [ERR] No train split found")

    # ── 6. Summary & Target progress ────────────────────────────────────────
    print("\n[6/6] Summary")
    print(f"      Valid clean images:   {len(valid_imgs)}")
    print(f"      Metadata records:     {len(meta_rows)}")
    total_split = sum(split_counts.values())
    print(f"      Total split IDs:      {total_split}")

    if target > 0:
        pct = min(100, int((len(valid_imgs) / target) * 100))
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        print(f"\n      Target: {target} images")
        print(f"      Progress: [{bar}] {pct}%  ({len(valid_imgs)}/{target})")
        if len(valid_imgs) < target:
            delta = target - len(valid_imgs)
            print(f"      Still need: {delta} more images")
            print(f"      Run: python api/api.py --limit {target} --resume")
        else:
            print(f"      [OK] Target reached!")

    print("\n" + "=" * 65 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate and report dataset health for Museum Artifact Restoration."
    )
    parser.add_argument("--clean-dir",    default="data/clean",       help="Clean images directory")
    parser.add_argument("--metadata",     default="data/metadata.csv", help="Metadata CSV path")
    parser.add_argument("--splits-dir",   default="data/splits",       help="Splits directory")
    parser.add_argument("--damaged-dir",  default="data/damaged",      help="Damaged images root directory")
    parser.add_argument("--target",       type=int, default=0,         help="Target dataset size (0 = no target)")
    args = parser.parse_args()

    run_analysis(
        clean_dir=args.clean_dir,
        metadata_file=args.metadata,
        splits_dir=args.splits_dir,
        damaged_dir=args.damaged_dir,
        target=args.target,
    )
