# src/precompute_damage.py
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from damage_sim_museum import simulate_damage_strong
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_one_image(item):
    i, oid, clean_dir, out_dir, strength, max_types, base_seed, write_mask = item
    clean_path = clean_dir / f"{oid}.jpg"
    if not clean_path.exists():
        found = list(clean_dir.glob(f"{oid}.*"))
        if found:
            clean_path = found[0]
        else:
            return False, f"Missing clean file for {oid}"
    out_path = out_dir / f"{oid}.jpg"
    mask_path = out_dir / f"{oid}_mask.png"
    if out_path.exists() and (not write_mask or mask_path.exists()):
        return True, "already_exists"
    try:
        with Image.open(str(clean_path)) as img_obj:
            img = np.array(img_obj.convert('RGB'))
        seed = base_seed + i
        damaged, mask = simulate_damage_strong(img, seed=seed, strength=strength, max_types=max_types)
        Image.fromarray(damaged).save(str(out_path), "JPEG", quality=95)
        if write_mask:
            Image.fromarray((mask * 255).astype('uint8')).save(str(mask_path))
        return True, "ok"
    except Exception as e:
        return False, str(e)

def precompute_for_split(clean_dir, ids_csv, out_dir, strength='strong', max_types=4, base_seed=0, write_mask=True, num_workers=4):
    clean_dir = Path(clean_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ids_csv)
    ids = df['obj_id'].astype(str).tolist()

    items = [
        (i, oid, clean_dir, out_dir, strength, max_types, base_seed, write_mask)
        for i, oid in enumerate(ids)
    ]

    desc = f"Precompute {out_dir.name}"
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(process_one_image, items), total=len(items), desc=desc))
    else:
        results = [process_one_image(item) for item in tqdm(items, desc=desc)]

    failed = [items[idx][1] for idx, (ok, _) in enumerate(results) if not ok]
    if failed:
        print(f"[warn] {len(failed)} images failed in {out_dir.name}: {failed[:5]}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean-dir', default='data/clean')
    parser.add_argument('--train-csv', default='data/splits/train_ids.csv')
    parser.add_argument('--val-csv', default='data/splits/val_ids.csv')
    parser.add_argument('--test-csv', default='data/splits/test_ids.csv')
    parser.add_argument('--out-root', default='data/damaged')
    parser.add_argument('--strength', default='strong', choices=['strong','moderate'])
    parser.add_argument('--max-types', type=int, default=4)
    parser.add_argument('--base-seed', type=int, default=0)
    parser.add_argument('--write-mask', action='store_true')
    parser.add_argument('--num-workers', type=int, default=6, help='Parallel worker threads (default: 6)')
    args = parser.parse_args()

    # train
    precompute_for_split(args.clean_dir, args.train_csv, os.path.join(args.out_root, 'train'),
                         strength=args.strength, max_types=args.max_types, base_seed=args.base_seed,
                         write_mask=args.write_mask, num_workers=args.num_workers)
    # val
    precompute_for_split(args.clean_dir, args.val_csv, os.path.join(args.out_root, 'val'),
                         strength=args.strength, max_types=args.max_types, base_seed=args.base_seed+100000,
                         write_mask=args.write_mask, num_workers=args.num_workers)
    # test
    if Path(args.test_csv).exists():
        precompute_for_split(args.clean_dir, args.test_csv, os.path.join(args.out_root, 'test'),
                             strength=args.strength, max_types=args.max_types, base_seed=args.base_seed+200000,
                             write_mask=args.write_mask, num_workers=args.num_workers)
