"""
scripts/acquire_independent_holdout.py
======================================
Acquire 40 new, independent public-domain artworks from The Metropolitan Museum of Art API.
Guarantees zero overlap with existing 2,000 project IDs.
Generates paired clean images, synthetic damaged images, and binary masks.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
import json
import csv
from pathlib import Path
import requests
from PIL import Image
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from damage_sim_museum import simulate_damage_strong
from utils import ensure_dir

BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"
SEARCH_URL = "https://collectionapi.metmuseum.org/public/collection/v1.1/search"
OBJECT_URL = f"{BASE_URL}/objects"

HEADERS = {
    "User-Agent": "MUESUM-Restoration-Research/2.0 (Academic Research; github.com/museum-restoration)"
}

CATEGORIES_QUERIES = [
    ("Paintings", "oil painting"),
    ("Drawings", "ink drawing"),
    ("Photographs", "albumen photograph"),
    ("Sculptures", "marble sculpture"),
    ("Bronzes", "bronze statue"),
    ("Textiles", "woven textile"),
    ("Books", "printed book woodcut"),
    ("Manuscripts", "illuminated manuscript"),
    ("Ceramics", "terracotta ceramic"),
    ("Metalwork", "silver ornament")
]

METADATA_COLUMNS = [
    "object_id",
    "title",
    "artist",
    "artist_display",
    "object_date",
    "medium",
    "classification",
    "department",
    "culture",
    "country",
    "object_url",
    "image_url",
    "image_path",
    "image_width",
    "image_height",
    "is_public_domain",
    "rights_and_reproduction",
]

def load_existing_ids():
    existing_ids = set()
    meta_csv = root_dir / "data" / "metadata.csv"
    if meta_csv.exists():
        with open(meta_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj_id = str(row.get("object_id", "")).strip()
                if obj_id:
                    existing_ids.add(obj_id)

    # Check splits
    splits_dir = root_dir / "data" / "splits"
    if splits_dir.exists():
        for csv_f in splits_dir.glob("*.csv"):
            with open(csv_f, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    obj_id = str(row.get("obj_id", "")).strip()
                    if obj_id:
                        existing_ids.add(obj_id)

    print(f"[Holdout] Loaded {len(existing_ids)} existing project object IDs for exclusion.")
    return existing_ids

def acquire_holdout(target_count=40):
    print("=" * 75)
    print("MUESUM — ACQUIRING INDEPENDENT HOLDOUT DATASET (40 NEW ARTWORKS)")
    print("=" * 75)

    existing_ids = load_existing_ids()

    holdout_dir = root_dir / "data" / "independent_holdout"
    clean_dir = holdout_dir / "clean"
    damaged_dir = holdout_dir / "damaged"
    masks_dir = holdout_dir / "masks"
    ensure_dir(str(clean_dir))
    ensure_dir(str(damaged_dir))
    ensure_dir(str(masks_dir))

    acquired_metadata = []
    acquired_ids = set()

    # Target per category: 4 artworks across 10 categories = 40 total
    target_per_cat = max(1, target_count // len(CATEGORIES_QUERIES))

    for cat_name, query in CATEGORIES_QUERIES:
        if len(acquired_ids) >= target_count:
            break

        print(f"\n[Acquisition] Querying category: {cat_name} (query='{query}')...")
        try:
            params = {"hasImages": "true", "isPublicDomain": "true", "q": query}
            resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"  Warning: search failed for {cat_name} (status {resp.status_code})")
                continue
            search_data = resp.json()
            candidate_ids = search_data.get("objectIDs", [])
            print(f"  Found {len(candidate_ids)} candidates from Met Search API.")
        except Exception as e:
            print(f"  Search error for {cat_name}: {e}")
            continue

        cat_acquired = 0
        for obj_id_num in candidate_ids:
            if cat_acquired >= target_per_cat or len(acquired_ids) >= target_count:
                break

            obj_id = str(obj_id_num).strip()
            if obj_id in existing_ids or obj_id in acquired_ids:
                continue

            # Fetch object metadata
            time.sleep(0.1)  # Respectful rate limiting
            try:
                obj_resp = requests.get(f"{OBJECT_URL}/{obj_id}", headers=HEADERS, timeout=15)
                if obj_resp.status_code != 200:
                    continue
                obj_data = obj_resp.json()
            except Exception:
                continue

            if not obj_data.get("isPublicDomain", False):
                continue

            img_url = obj_data.get("primaryImage", "") or obj_data.get("primaryImageSmall", "")
            if not img_url:
                continue

            # Download image
            try:
                img_resp = requests.get(img_url, headers=HEADERS, timeout=20)
                if img_resp.status_code != 200:
                    continue
                # Verify image with PIL
                img_bytes = img_resp.content
                with Image.open(io.BytesIO(img_bytes)) as pil_test:
                    pil_test.verify()
                
                with Image.open(io.BytesIO(img_bytes)) as pil_img:
                    rgb_img = pil_img.convert("RGB")
                    w, h = rgb_img.size

                if w < 200 or h < 200:
                    continue

                # Save clean image
                clean_path = clean_dir / f"{obj_id}.jpg"
                rgb_img.save(str(clean_path), quality=95)

                meta_row = {
                    "object_id": obj_id,
                    "title": obj_data.get("title", ""),
                    "artist": obj_data.get("artistDisplayName", ""),
                    "artist_display": obj_data.get("artistDisplayBio", ""),
                    "object_date": obj_data.get("objectDate", ""),
                    "medium": obj_data.get("medium", ""),
                    "classification": cat_name,
                    "department": obj_data.get("department", ""),
                    "culture": obj_data.get("culture", ""),
                    "country": obj_data.get("country", ""),
                    "object_url": obj_data.get("objectURL", ""),
                    "image_url": img_url,
                    "image_path": str(clean_path),
                    "image_width": w,
                    "image_height": h,
                    "is_public_domain": True,
                    "rights_and_reproduction": obj_data.get("rightsAndReproduction", ""),
                }
                acquired_metadata.append(meta_row)
                acquired_ids.add(obj_id)
                cat_acquired += 1
                print(f"  [{len(acquired_ids):2d}/{target_count}] Acquired ID {obj_id} ({w}x{h}) - {obj_data.get('title', '')[:35]}")
            except Exception as e:
                continue

    # If we still need more images to reach target_count, make a general pass
    if len(acquired_ids) < target_count:
        print(f"\n[Acquisition] General pass to acquire remaining {target_count - len(acquired_ids)} artworks...")
        try:
            resp = requests.get(SEARCH_URL, params={"hasImages": "true", "isPublicDomain": "true", "q": "masterpiece"}, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                gen_ids = resp.json().get("objectIDs", [])
                for obj_id_num in gen_ids:
                    if len(acquired_ids) >= target_count:
                        break
                    obj_id = str(obj_id_num).strip()
                    if obj_id in existing_ids or obj_id in acquired_ids:
                        continue
                    time.sleep(0.1)
                    try:
                        obj_resp = requests.get(f"{OBJECT_URL}/{obj_id}", headers=HEADERS, timeout=15)
                        if obj_resp.status_code != 200:
                            continue
                        obj_data = obj_resp.json()
                        if not obj_data.get("isPublicDomain", False):
                            continue
                        img_url = obj_data.get("primaryImage", "") or obj_data.get("primaryImageSmall", "")
                        if not img_url:
                            continue
                        img_resp = requests.get(img_url, headers=HEADERS, timeout=20)
                        if img_resp.status_code != 200:
                            continue
                        with Image.open(io.BytesIO(img_resp.content)) as pil_img:
                            rgb_img = pil_img.convert("RGB")
                            w, h = rgb_img.size
                        if w < 200 or h < 200:
                            continue
                        clean_path = clean_dir / f"{obj_id}.jpg"
                        rgb_img.save(str(clean_path), quality=95)
                        cat = obj_data.get("classification", "") or "General Art"
                        meta_row = {
                            "object_id": obj_id,
                            "title": obj_data.get("title", ""),
                            "artist": obj_data.get("artistDisplayName", ""),
                            "artist_display": obj_data.get("artistDisplayBio", ""),
                            "object_date": obj_data.get("objectDate", ""),
                            "medium": obj_data.get("medium", ""),
                            "classification": cat,
                            "department": obj_data.get("department", ""),
                            "culture": obj_data.get("culture", ""),
                            "country": obj_data.get("country", ""),
                            "object_url": obj_data.get("objectURL", ""),
                            "image_url": img_url,
                            "image_path": str(clean_path),
                            "image_width": w,
                            "image_height": h,
                            "is_public_domain": True,
                            "rights_and_reproduction": obj_data.get("rightsAndReproduction", ""),
                        }
                        acquired_metadata.append(meta_row)
                        acquired_ids.add(obj_id)
                        print(f"  [{len(acquired_ids):2d}/{target_count}] Acquired ID {obj_id} ({w}x{h}) - {obj_data.get('title', '')[:35]}")
                    except Exception:
                        continue
        except Exception as e:
            print("General pass error:", e)

    print(f"\n[Acquisition Complete] Successfully acquired {len(acquired_ids)} new artworks.")
    assert len(acquired_ids & existing_ids) == 0, "FATAL: Overlap detected with existing project IDs!"

    # Save metadata CSV
    meta_path = holdout_dir / "metadata.csv"
    with open(meta_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(acquired_metadata)
    print(f"[Holdout] Metadata saved to: {meta_path}")

    # Save object_ids.csv
    ids_path = holdout_dir / "object_ids.csv"
    with open(ids_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["object_id"])
        for obj_id in sorted(acquired_ids):
            writer.writerow([obj_id])
    print(f"[Holdout] Object IDs saved to: {ids_path}")

    # ──────────────────────────────────────────────────────────────────
    # PRECOMPUTE DAMAGE & MASKS
    # ──────────────────────────────────────────────────────────────────
    print("\n[Holdout] Precomputing synthetic damage and binary masks...")
    for idx, obj_id in enumerate(sorted(acquired_ids), 1):
        clean_file = clean_dir / f"{obj_id}.jpg"
        dmg_file = damaged_dir / f"{obj_id}.jpg"
        mask_file = masks_dir / f"{obj_id}_mask.png"

        with Image.open(clean_file) as c_im:
            clean_np = np.array(c_im.convert("RGB"))

        # Deterministic simulation with reproducible seed
        seed = int(obj_id) % 1000000
        dmg_np, mask_np = simulate_damage_strong(clean_np, max_damage_types=4, seed=seed)

        Image.fromarray(dmg_np).save(str(dmg_file), quality=95)
        # Ensure mask is uint8: 255 = damage, 0 = clean
        mask_uint8 = ((mask_np > 0) * 255).astype(np.uint8)
        Image.fromarray(mask_uint8, mode="L").save(str(mask_file))

        cov = (np.sum(mask_uint8 > 127) / mask_uint8.size) * 100.0
        print(f"  [{idx:2d}/{len(acquired_ids)}] Preprocessed ID {obj_id} | Coverage: {cov:.1f}%")

    print("\n[Holdout] All damage simulation and binary masks generated successfully!")

if __name__ == "__main__":
    import io
    acquire_holdout(target_count=40)
