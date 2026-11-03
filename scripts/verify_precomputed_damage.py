import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

def verify_dataset(data_dir: Path, reports_dir: Path):
    clean_dir = data_dir / "clean"
    damaged_dir = data_dir / "damaged"
    splits_dir = data_dir / "splits"
    
    print("=" * 60)
    print("STAGE 4: FULL DATASET PREPROCESSING INTEGRITY VERIFICATION")
    print("=" * 60)
    
    splits = {
        "train": (splits_dir / "train_ids.csv", 1600),
        "val": (splits_dir / "val_ids.csv", 200),
        "test": (splits_dir / "test_ids.csv", 200)
    }
    
    all_split_ids = {}
    total_damaged = 0
    total_masks = 0
    total_bytes_damaged = 0
    total_bytes_masks = 0
    
    # 1. Check split file contents and disjointness
    for split_name, (split_csv, expected_count) in splits.items():
        if not split_csv.exists():
            print(f"[FAIL] Split file {split_csv} does not exist!")
            sys.exit(1)
        df = pd.read_csv(split_csv, dtype=str)
        col = "objectID" if "objectID" in df.columns else df.columns[0]
        ids = list(df[col].astype(str).str.strip())
        print(f"\nVerifying {split_name} split ({len(ids)} IDs, expected {expected_count})...")
        assert len(ids) == expected_count, f"{split_name} count mismatch: {len(ids)} != {expected_count}"
        assert len(set(ids)) == len(ids), f"Duplicate IDs inside {split_name} split!"
        all_split_ids[split_name] = set(ids)
        
    # Check disjointness (no split leakage)
    train_ids = all_split_ids["train"]
    val_ids = all_split_ids["val"]
    test_ids = all_split_ids["test"]
    assert len(train_ids.intersection(val_ids)) == 0, "[FAIL] Split leakage between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "[FAIL] Split leakage between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "[FAIL] Split leakage between val and test!"
    print("[OK] Split disjointness verified: 0 ID overlap across train, val, and test splits.")
    
    # Check anchor IDs
    assert "247010" in test_ids, "[FAIL] Anchor test ID 247010 missing from test split!"
    assert "246991" in val_ids, "[FAIL] Anchor val ID 246991 missing from val split!"
    print("[OK] Anchor IDs verified: 247010 in test, 246991 in val.")

    # 2. Check files on disk
    corrupt_count = 0
    missing_count = 0
    dim_mismatch_count = 0
    empty_mask_count = 0
    mask_coverages = []
    
    for split_name, ids in all_split_ids.items():
        split_dir = damaged_dir / split_name
        split_damaged_count = 0
        split_mask_count = 0
        
        for obj_id in ids:
            clean_file = clean_dir / f"{obj_id}.jpg"
            damaged_file = split_dir / f"{obj_id}.jpg"
            mask_file = split_dir / f"{obj_id}_mask.png"
            
            if not clean_file.exists():
                print(f"[FAIL] Missing clean image: {clean_file}")
                missing_count += 1
                continue
            if not damaged_file.exists():
                print(f"[FAIL] Missing damaged image: {damaged_file}")
                missing_count += 1
                continue
            if not mask_file.exists():
                print(f"[FAIL] Missing mask image: {mask_file}")
                missing_count += 1
                continue
                
            split_damaged_count += 1
            split_mask_count += 1
            total_bytes_damaged += damaged_file.stat().st_size
            total_bytes_masks += mask_file.stat().st_size
            
            # Verify image opening and dimensions
            try:
                with Image.open(clean_file) as c_img:
                    c_size = c_img.size
                with Image.open(damaged_file) as d_img:
                    d_size = d_img.size
                    d_mode = d_img.mode
                with Image.open(mask_file) as m_img:
                    m_size = m_img.size
                    m_mode = m_img.mode
                    mask_np = np.array(m_img)
            except Exception as e:
                print(f"[FAIL] Corrupt image for ID {obj_id}: {e}")
                corrupt_count += 1
                continue
                
            if c_size != d_size or c_size != m_size:
                print(f"[FAIL] Dimension mismatch for {obj_id}: Clean={c_size}, Damaged={d_size}, Mask={m_size}")
                dim_mismatch_count += 1
                
            # Mask verification
            damaged_px = np.sum(mask_np > 0)
            total_px = mask_np.size
            coverage = (damaged_px / total_px) * 100.0
            mask_coverages.append(coverage)
            
            if damaged_px == 0:
                print(f"[WARN] Empty mask for {obj_id}")
                empty_mask_count += 1
                
        print(f"[{split_name.upper()}] Verified {split_damaged_count} damaged images and {split_mask_count} masks.")
        total_damaged += split_damaged_count
        total_masks += split_mask_count
        
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total damaged images: {total_damaged} (Expected: 2,000)")
    print(f"Total masks:          {total_masks} (Expected: 2,000)")
    print(f"Total missing files:  {missing_count}")
    print(f"Total corrupt files:  {corrupt_count}")
    print(f"Dimension mismatches: {dim_mismatch_count}")
    print(f"Empty masks:          {empty_mask_count}")
    print(f"Damaged images size:  {total_bytes_damaged / (1024**3):.2f} GB")
    print(f"Damage masks size:    {total_bytes_masks / (1024**3):.2f} GB")
    print(f"Total disk usage:     {(total_bytes_damaged + total_bytes_masks) / (1024**3):.2f} GB")
    print(f"Average mask coverage:{np.mean(mask_coverages):.2f}% (min: {np.min(mask_coverages):.2f}%, max: {np.max(mask_coverages):.2f}%)")
    
    assert total_damaged == 2000, f"Expected 2000 damaged images, found {total_damaged}"
    assert total_masks == 2000, f"Expected 2000 masks, found {total_masks}"
    assert missing_count == 0, f"Missing files found: {missing_count}"
    assert corrupt_count == 0, f"Corrupt files found: {corrupt_count}"
    assert dim_mismatch_count == 0, f"Dimension mismatches found: {dim_mismatch_count}"
    assert empty_mask_count == 0, f"Empty masks found: {empty_mask_count}"
    print("\n[SUCCESS] ALL INTEGRITY CHECKS PASSED PERFECTLY!")

    # 3. Generate Visual Quality Comparison Samples (Section 4)
    reports_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = [
        ("train", "436535"),
        ("train", "436528"),
        ("train", "436121"),
        ("train", "436533"),
        ("train", "437056"),
        ("train", "436947"),
        ("val", "246991"),   # Anchor val
        ("val", "435882"),
        ("test", "247010"),  # Anchor test
        ("test", "437998"),
        ("test", "438012"),
        ("test", "438015")
    ]
    
    print("\nGenerating visual comparison composite samples in:", reports_dir)
    for split_name, obj_id in sample_ids:
        c_path = clean_dir / f"{obj_id}.jpg"
        d_path = damaged_dir / split_name / f"{obj_id}.jpg"
        m_path = damaged_dir / split_name / f"{obj_id}_mask.png"
        
        if not (c_path.exists() and d_path.exists() and m_path.exists()):
            continue
            
        with Image.open(c_path) as c_im, Image.open(d_path) as d_im, Image.open(m_path) as m_im:
            c_rgb = c_im.convert("RGB")
            d_rgb = d_im.convert("RGB")
            m_rgb = m_im.convert("RGB") # grayscale mask in RGB
            
            # Resize for composite visualization (height 512, preserving aspect ratio)
            target_h = 512
            scale = target_h / c_rgb.height
            target_w = max(1, int(c_rgb.width * scale))
            
            c_resized = c_rgb.resize((target_w, target_h), Image.Resampling.BILINEAR)
            d_resized = d_rgb.resize((target_w, target_h), Image.Resampling.BILINEAR)
            m_resized = m_rgb.resize((target_w, target_h), Image.Resampling.NEAREST)
            
            # Combine side by side: [Clean | Damaged | Mask]
            composite = Image.new("RGB", (target_w * 3 + 20, target_h + 40), color=(30, 30, 30))
            composite.paste(c_resized, (0, 35))
            composite.paste(d_resized, (target_w + 10, 35))
            composite.paste(m_resized, (target_w * 2 + 20, 35))
            
            comp_path = reports_dir / f"{split_name}_{obj_id}_comparison.png"
            composite.save(comp_path)
            print(f"Saved sample comparison: {comp_path.name}")
            
    print("[SUCCESS] Visual comparison samples generated successfully.")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    verify_dataset(base_dir / "data", base_dir / "reports" / "damage_preprocessing_samples")

