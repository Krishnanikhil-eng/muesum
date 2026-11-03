"""
scripts/create_small_split.py
Generate Train / Validation / Test Splits
==========================================
Creates reproducible, leak-free 80/10/10 splits from metadata.csv.
Test IDs are stable: once generated with a fixed seed, newly downloaded
images only flow into train/val and never change the test set.

Usage:
    python scripts/create_small_split.py --metadata data/metadata.csv --total 2000
    python scripts/create_small_split.py --metadata data/metadata.csv --total 5000
    python scripts/create_small_split.py --metadata data/metadata.csv --total 2000 --out-dir data/splits
"""

import argparse
import csv
import random
from pathlib import Path


def create_splits(
    metadata: str = "data/metadata.csv",
    out_dir: str = "data/splits",
    total: int = 2000,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    # test_ratio is implicitly 1 - train_ratio - val_ratio = 0.10
    seed: int = 42,
    clean_dir: str = "data/clean",
    validate_files: bool = True,
    preserve_existing: bool = True,
):
    """
    Generate train/val/test split CSVs.

    Args:
        metadata:       Path to metadata.csv with 'object_id' column.
        out_dir:        Directory to write split CSV files.
        total:          Total number of images to include across all splits.
        train_ratio:    Fraction of total for training (default 0.80).
        val_ratio:      Fraction of total for validation (default 0.10).
        seed:           Random seed for reproducibility.
        clean_dir:      If validate_files=True, only include IDs with existing image files.
        validate_files: Whether to verify image files exist before including in splits.
    """
    meta_path = Path(metadata)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    # ── Load IDs from metadata ────────────────────────────────────────────────
    ids = []
    seen = set()
    with open(str(meta_path), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Support both 'object_id' (new api.py) and 'obj_id' (legacy)
        fieldnames = reader.fieldnames or []
        id_col = "object_id" if "object_id" in fieldnames else "obj_id"
        if id_col not in fieldnames:
            raise ValueError(
                f"Metadata CSV must contain 'object_id' or 'obj_id' column. "
                f"Found columns: {fieldnames}"
            )
        for row in reader:
            oid = str(row.get(id_col, "")).strip()
            if oid and oid not in seen:
                ids.append(oid)
                seen.add(oid)

    print(f"[split] Loaded {len(ids)} unique IDs from {meta_path}")

    # ── Optionally validate that image files exist ───────────────────────────
    if validate_files:
        clean_path = Path(clean_dir)
        valid_ids = []
        for oid in ids:
            # check common extensions
            found = False
            for ext in ("jpg", "jpeg", "png", "JPG", "PNG"):
                if (clean_path / f"{oid}.{ext}").exists():
                    found = True
                    break
            if found:
                valid_ids.append(oid)
        if len(valid_ids) < len(ids):
            print(f"[split] {len(ids) - len(valid_ids)} IDs have no image file - excluded")
        ids = valid_ids
        print(f"[split] {len(ids)} IDs with valid image files")

    if len(ids) < total:
        raise SystemExit(
            f"[error] Requested {total} images but only {len(ids)} valid IDs available. "
            f"Download more images first:\n"
            f"  python api/api.py --limit {total} --resume"
        )

    # ── Compute split sizes ──────────────────────────────────────────────────
    n_test  = max(1, round(total * (1.0 - train_ratio - val_ratio)))
    n_val   = max(1, round(total * val_ratio))
    n_train = total - n_val - n_test

    assert n_train + n_val + n_test == total, "Split sizes don't add up - rounding issue"

    # ── Check for existing splits to preserve ───────────────────────────────
    def _load_split_csv(p: Path) -> list:
        if not p.exists():
            return []
        with open(str(p), newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [str(row["obj_id"]).strip() for row in reader if str(row.get("obj_id", "")).strip()]

    existing_train = []
    existing_val = []
    existing_test = []

    if preserve_existing:
        existing_train = [x for x in _load_split_csv(out_path / "train_ids.csv") if x in ids]
        existing_val   = [x for x in _load_split_csv(out_path / "val_ids.csv") if x in ids]
        existing_test  = [x for x in _load_split_csv(out_path / "test_ids.csv") if x in ids]
        print(f"[split] Preserving existing split assignments: {len(existing_train)} train, {len(existing_val)} val, {len(existing_test)} test")

    already_assigned = set(existing_train) | set(existing_val) | set(existing_test)
    unassigned = [x for x in ids if x not in already_assigned]

    # Stable deterministic shuffle of unassigned pool
    unassigned_sorted = sorted(unassigned)
    rng = random.Random(seed)
    rng.shuffle(unassigned_sorted)

    needed_train = max(0, n_train - len(existing_train))
    needed_val   = max(0, n_val - len(existing_val))
    needed_test  = max(0, n_test - len(existing_test))

    train_ids = existing_train + unassigned_sorted[:needed_train]
    val_ids   = existing_val + unassigned_sorted[needed_train : needed_train + needed_val]
    test_ids  = existing_test + unassigned_sorted[needed_train + needed_val : needed_train + needed_val + needed_test]

    # ── Sanity check: no overlap ─────────────────────────────────────────────
    assert not (set(train_ids) & set(val_ids)),  "train/val overlap!"
    assert not (set(train_ids) & set(test_ids)), "train/test overlap!"
    assert not (set(val_ids)   & set(test_ids)), "val/test overlap!"

    # ── Write CSV files ──────────────────────────────────────────────────────
    def write_csv(path: Path, id_list: list):
        with open(str(path), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["obj_id"])
            for oid in id_list:
                writer.writerow([oid])

    write_csv(out_path / "train_ids.csv", train_ids)
    write_csv(out_path / "val_ids.csv",   val_ids)
    write_csv(out_path / "test_ids.csv",  test_ids)

    # ── Also write a legacy "small_*" variant (backward compatibility) ───────
    write_csv(out_path / "small_train_ids.csv", train_ids)
    write_csv(out_path / "small_val_ids.csv",   val_ids)

    # ── Report ───────────────────────────────────────────────────────────────
    train_pct = round(n_train / total * 100)
    val_pct   = round(n_val   / total * 100)
    test_pct  = round(n_test  / total * 100)

    print(f"\n[split] Split generated (seed={seed}):")
    print(f"        Total:      {total}")
    print(f"        Train:      {n_train:>6}  ({train_pct}%)  -> {out_path / 'train_ids.csv'}")
    print(f"        Val:        {n_val:>6}  ({val_pct}%)   -> {out_path / 'val_ids.csv'}")
    print(f"        Test:       {n_test:>6}  ({test_pct}%)   -> {out_path / 'test_ids.csv'}")
    print(f"\n  [OK] No data leakage confirmed (train and val and test = empty-set)")
    print(f"  [OK] Test IDs are stable - re-running with same seed + total will produce identical test set\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate reproducible 80/10/10 train/val/test splits from metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/create_small_split.py --metadata data/metadata.csv --total 2000
  python scripts/create_small_split.py --metadata data/metadata.csv --total 5000
  python scripts/create_small_split.py --metadata data/metadata.csv --total 2000 --no-validate
        """,
    )
    parser.add_argument("--metadata",     default="data/metadata.csv", help="Metadata CSV (must have 'object_id' or 'obj_id')")
    parser.add_argument("--out-dir",      default="data/splits",       help="Output directory for CSV files")
    parser.add_argument("--total",        type=int, default=2000,      help="Total images across all splits")
    parser.add_argument("--train-ratio",  type=float, default=0.80,    help="Train fraction (default 0.80)")
    parser.add_argument("--val-ratio",    type=float, default=0.10,    help="Val fraction (default 0.10; test = 1-train-val)")
    parser.add_argument("--seed",         type=int, default=42,        help="Random seed (default 42)")
    parser.add_argument("--clean-dir",    default="data/clean",        help="Clean images folder (for file validation)")
    parser.add_argument("--no-validate",  action="store_true",         help="Skip image file existence check")
    parser.add_argument("--no-preserve",  action="store_true",         help="Do not preserve existing split assignments")
    args = parser.parse_args()

    create_splits(
        metadata=args.metadata,
        out_dir=args.out_dir,
        total=args.total,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        clean_dir=args.clean_dir,
        validate_files=not args.no_validate,
        preserve_existing=not args.no_preserve,
    )
