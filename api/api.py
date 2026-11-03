"""
api/api.py
Metropolitan Museum of Art Collection API — Robust Image Downloader
=====================================================================
Usage:
    python api/api.py --limit 2000 --resume
    python api/api.py --limit 5000 --resume
    python api/api.py --limit 10000 --resume --out-dir data/clean

The requested --limit is the TOTAL desired dataset size.
If valid images already exist, only the remaining delta is downloaded.
Duplicate object IDs are never downloaded twice.
"""

import os
import time
import csv
import json
import argparse
import io
import sys
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import threading

import requests
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"
SEARCH_URL = "https://collectionapi.metmuseum.org/public/collection/v1.1/search"
OBJECT_URL = f"{BASE_URL}/objects/{{obj_id}}"

HEADERS = {
    "User-Agent": "MuseumArtifactRestoration/2.0 (Educational/Research; github.com/museum-restoration)"
}

# Queries to cover multiple artwork categories
SEARCH_QUERIES = [
    "ancient art",
    "painting",
    "textiles",
    "manuscripts",
    "sculpture",
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

MIN_DIM = 100          # reject images smaller than this in either dimension
REQUEST_TIMEOUT = 20   # seconds
MAX_RETRIES = 3        # attempts per request
RATE_SLEEP = 0.08      # polite delay between individual requests


# ──────────────────────────────────────────────────────────────────────────────
# Network helpers
# ──────────────────────────────────────────────────────────────────────────────
def _get_with_retry(url: str, params=None, stream=False) -> Optional[requests.Response]:
    """GET with exponential-backoff retry. Returns Response or None on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
                stream=stream,
            )
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  [rate-limit] 429 - sleeping {wait}s before retry {attempt}/{MAX_RETRIES}", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code in (404, 403):
                return None  # not worth retrying
            # other HTTP errors - retry with backoff
            wait = 2 ** attempt
            print(f"  [http {resp.status_code}] retry {attempt}/{MAX_RETRIES} in {wait}s for {url}", flush=True)
            time.sleep(wait)
        except requests.exceptions.Timeout:
            wait = 2 ** attempt
            time.sleep(wait)
        except requests.exceptions.ConnectionError:
            wait = 2 ** attempt
            time.sleep(wait)
        except Exception:
            return None
    return None


def _is_valid_image(data: bytes) -> Optional[tuple]:
    """
    Check whether raw bytes represent a valid, sufficiently large image.
    Returns (width, height) on success, or None if invalid/too small.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        if w < MIN_DIM or h < MIN_DIM:
            return None
        return (w, h)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Metadata helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_existing_metadata(metadata_path: Path) -> dict:
    """
    Load existing metadata CSV and return a dict keyed by object_id (str).
    """
    records = {}
    if not metadata_path.exists():
        return records
    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            oid = str(row.get("object_id", "")).strip()
            if oid:
                records[oid] = row
    return records


def _count_valid_images(out_path: Path, existing_meta: dict) -> int:
    """
    Count images that exist on disk AND have a metadata record.
    Validates each file with PIL to skip corrupted ones.
    """
    valid = 0
    for oid, meta in existing_meta.items():
        img_path = Path(meta.get("image_path", ""))
        if img_path.exists():
            try:
                with Image.open(str(img_path)) as im:
                    im.verify()
                valid += 1
            except Exception:
                pass
    return valid


# ──────────────────────────────────────────────────────────────────────────────
# Core downloader
# ──────────────────────────────────────────────────────────────────────────────
def fetch_met_images(
    limit: int = 2000,
    out_dir: str = "data/clean",
    metadata_file: str = "data/metadata.csv",
    resume: bool = True,
    workers: int = 8,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(metadata_file)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load existing state ──────────────────────────────────────────────────
    existing_meta = _load_existing_metadata(metadata_path)
    existing_ids: set = set(existing_meta.keys())

    if resume or existing_meta:
        valid_count = _count_valid_images(out_path, existing_meta)
        print(f"\n[resume] Found {valid_count} valid existing images (of {len(existing_ids)} metadata records).", flush=True)
    else:
        valid_count = 0

    if valid_count >= limit:
        print(f"\n[OK] Target of {limit} images already reached ({valid_count} valid). Nothing to download.", flush=True)
        return

    remaining = limit - valid_count
    print(f"\n[info] Need {remaining} more images to reach target of {limit}.\n", flush=True)

    # ── Collect object IDs from all search queries ───────────────────────────
    query_candidate_lists = []
    seen_in_candidates: set = set()
    target_candidates_per_query = max(30, int((remaining * 2.5) / len(SEARCH_QUERIES)) + 50)

    for query in SEARCH_QUERIES:
        print(f'[search] Querying: "{query}" (hasImages=true) ...', flush=True)
        offset = 0
        page_size = 500
        this_query_candidates = []

        while len(this_query_candidates) < target_candidates_per_query:
            resp = _get_with_retry(SEARCH_URL, params={
                "hasImages": "true",
                "q": query,
                "limit": page_size,
                "offset": offset,
            })
            if resp is None:
                print(f"  [warn] Search failed for query: {query} at offset {offset}", flush=True)
                break
            try:
                data = resp.json()
            except Exception:
                break

            ids = data.get("objectIDs", []) or []
            if not ids:
                break

            new_ids = [str(i) for i in ids if str(i) not in seen_in_candidates and str(i) not in existing_ids]
            seen_in_candidates.update(new_ids)
            this_query_candidates.extend(new_ids)
            offset += len(ids)

            print(f"  -> page (offset={offset-len(ids)}, limit={page_size}): {len(ids)} objects, {len(new_ids)} new candidates for '{query}'", flush=True)
            time.sleep(RATE_SLEEP)

            if len(ids) < page_size:
                break

        query_candidate_lists.append(this_query_candidates)

    # Interleave candidates across categories for balanced representation
    all_candidate_ids = []
    max_candidates = max((len(l) for l in query_candidate_lists), default=0)
    for i in range(max_candidates):
        for c_list in query_candidate_lists:
            if i < len(c_list):
                all_candidate_ids.append(c_list[i])

    print(f"\n[search] Total interleaved candidates ready: {len(all_candidate_ids)}", flush=True)

    if not all_candidate_ids:
        print("\n[warn] No new candidate objects found. All IDs are already in the dataset.", flush=True)
        return

    need_header = not metadata_path.exists() or metadata_path.stat().st_size == 0

    downloaded = 0
    skipped = 0
    failed = 0
    lock = threading.Lock()

    print(f"\n[download] Starting parallel download (workers={workers}) of up to {remaining} images...\n", flush=True)

    def process_artwork(obj_id: str):
        nonlocal downloaded, skipped, failed, need_header

        with lock:
            if downloaded >= remaining:
                return

        detail_url = OBJECT_URL.format(obj_id=obj_id)
        resp = _get_with_retry(detail_url)
        time.sleep(RATE_SLEEP)

        if resp is None:
            with lock:
                failed += 1
            return

        try:
            details = resp.json()
        except Exception:
            with lock:
                failed += 1
            return

        # Require public domain
        if not details.get("isPublicDomain", False):
            with lock:
                skipped += 1
            return

        # Prefer primaryImage (full-res), fall back to primaryImageSmall
        img_url = details.get("primaryImage") or details.get("primaryImageSmall")
        if not img_url:
            with lock:
                skipped += 1
            return

        # Download image
        img_resp = _get_with_retry(img_url, stream=True)
        time.sleep(RATE_SLEEP)
        if img_resp is None:
            with lock:
                failed += 1
            return

        try:
            img_data = img_resp.content
        except Exception:
            with lock:
                failed += 1
            return

        # Validate image
        dims = _is_valid_image(img_data)
        if dims is None:
            with lock:
                skipped += 1
            return

        width, height = dims
        img_path = out_path / f"{obj_id}.jpg"

        try:
            pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
            pil_img.save(str(img_path), "JPEG", quality=95)
        except Exception:
            with lock:
                failed += 1
            return

        with lock:
            if downloaded >= remaining:
                return

            downloaded += 1
            total_so_far = valid_count + downloaded

            row = {
                "object_id":              obj_id,
                "title":                  details.get("title", ""),
                "artist":                 details.get("artistDisplayName", ""),
                "artist_display":         details.get("artistDisplayBio", ""),
                "object_date":            details.get("objectDate", ""),
                "medium":                 details.get("medium", ""),
                "classification":         details.get("classification", ""),
                "department":             details.get("department", ""),
                "culture":                details.get("culture", ""),
                "country":                details.get("country", ""),
                "object_url":             details.get("objectURL", ""),
                "image_url":              img_url,
                "image_path":             str(img_path),
                "image_width":            width,
                "image_height":           height,
                "is_public_domain":       details.get("isPublicDomain", False),
                "rights_and_reproduction":details.get("rightsAndReproduction", ""),
            }

            # Append to metadata CSV atomically
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            write_head = need_header and downloaded == 1
            mode = "w" if write_head else "a"
            with open(metadata_path, mode, newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
                if write_head:
                    writer.writeheader()
                    need_header = False
                writer.writerow(row)

            pct = int((total_so_far / limit) * 100)
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            title_short = str(details.get("title", "")).encode("ascii", "replace").decode("ascii")[:42]
            print(
                f"  [{total_so_far:>5}/{limit}] [{bar}] {pct:>3}%  "
                f"ID={obj_id:<8} {width}x{height}  {title_short}",
                flush=True
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in executor.map(process_artwork, all_candidate_ids):
            if downloaded >= remaining:
                break

    total = valid_count + downloaded
    print("\n" + "=" * 60, flush=True)
    print(f"  Download complete.", flush=True)
    print(f"  Previously valid:   {valid_count}", flush=True)
    print(f"  Newly downloaded:   {downloaded}", flush=True)
    print(f"  Skipped (non-PD/none): {skipped}", flush=True)
    print(f"  Failed (net/parse): {failed}", flush=True)
    print(f"  Total valid images: {total} / {limit}", flush=True)
    print(f"  Images saved to:    {out_path}/", flush=True)
    print(f"  Metadata saved to:  {metadata_path}", flush=True)
    if total < limit:
        print(f"\n  [warn] Only {total} images obtained (target: {limit}).", flush=True)
        print(f"         Run again with --resume to fetch from additional offsets.", flush=True)
    print("=" * 60, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download artwork images from The Metropolitan Museum of Art Collection API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python api/api.py --limit 2000 --resume
  python api/api.py --limit 5000 --resume
  python api/api.py --limit 10000 --resume --out-dir data/clean
        """,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Total desired dataset size (not additional images). Default: 2000",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/clean",
        help="Folder to save downloaded images. Default: data/clean",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/metadata.csv",
        help="Path to metadata CSV. Default: data/metadata.csv",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume an interrupted download session (skips already-downloaded IDs).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent download threads (default: 8, safe and respectful).",
    )
    args = parser.parse_args()

    fetch_met_images(
        limit=args.limit,
        out_dir=args.out_dir,
        metadata_file=args.metadata,
        resume=args.resume,
        workers=args.workers,
    )
