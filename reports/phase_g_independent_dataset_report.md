# MUESUM — Stage 1: Independent Holdout Dataset Acquisition Report
**Project**: MUESUM (Historical Artwork Inpainting & Neural Restoration)  
**Date**: 2026-11-04  
**Author**: Lead CV Engineer / Release Architect  
**Status**: APPROVED & ACQUIRED  

---

## 1. Executive Summary

To provide rigorous, unbiased validation of the MUESUM artwork restoration system, an entirely new independent holdout dataset of **40 public-domain artworks** has been acquired from The Metropolitan Museum of Art Collection API.

### Key Dataset Milestones
- **Acquired Count**: 40 clean artworks, 40 synthetically damaged artworks, and 40 binary damage masks.
- **Project Overlap**: **Strictly 0 IDs** ($0 / 2,000$ overlap with existing training, validation, or test sets).
- **Diversity**: 10 distinct cultural artifact categories (4 items per category).
- **Damage Coverage**: Mean $18.0\%$, median $18.7\%$, ranging from $4.3\%$ to $36.8\%$.
- **Strict Holdout Policy**: This dataset is completely quarantined from model training, loss tuning, or checkpoint selection. It is reserved solely for post-training independent evaluation.

---

## 2. API Acquisition Architecture & Protocol

### 2.1 Endpoint Specification
- **Search API**: `https://collectionapi.metmuseum.org/public/collection/v1.1/search`  
  *(Upgraded from deprecated v1 endpoint which returns HTTP 410)*
- **Object Details API**: `https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}`
- **Authentication / Headers**: Polite academic crawler identification with custom User-Agent.
- **Pacing**: Sequential acquisition with 100 ms rate-limiting sleep between requests to respect museum servers.

### 2.2 Category & Query Distribution
Artworks were curated evenly across 10 departments and medium classifications:

| # | Category | Search Query | Acquired IDs | Primary Media |
|---|---|---|---|---|
| 1 | **Paintings** | `oil painting` | 659725, 656610, 435581, 646741 | Oil on canvas, wood panel |
| 2 | **Drawings** | `ink drawing` | 420455, 12464, 74669, 74662 | Ink on paper, graphite |
| 3 | **Photographs** | `albumen photograph` | 646973, 263532, 422287, 422552 | Albumen silver prints |
| 4 | **Sculptures** | `marble sculpture` | 329970, 247464, 257433, 253423 | Carved marble, stone |
| 5 | **Bronzes** | `bronze statue` | 254502, 257580, 247117, 246701 | Cast bronze, patina |
| 6 | **Textiles** | `woven textile` | 44579, 39727, 74365, 463308 | Woven silk, linen, wool |
| 7 | **Books** | `printed book woodcut` | 388471, 625017, 466568, 661829 | Woodcut prints, incunabula |
| 8 | **Manuscripts** | `illuminated manuscript`| 739852, 456964, 74877, 74882 | Tempera & gold on parchment |
| 9 | **Ceramics** | `terracotta ceramic` | 450206, 456206, 450249, 450193 | Glazed terracotta, earthenware |
| 10| **Metalwork** | `silver ornament` | 50256, 309006, 309087, 309088 | Chased silver, hammered plaques |

---

## 3. Damage Simulation & Mask Generation

Damage synthesis for the holdout set replicates the multi-modal degradation patterns observed in archival heritage collections:
1. **Translucent Stains**: Irregular liquid staining and foxing with Gaussian-warped boundaries.
2. **Surface Cracks & Craquelure**: Random walks simulating pigment and glaze fissure networks.
3. **Biological & Mold Spots**: Variable-diameter clustered spotting replicating mildew colonization.
4. **Abrasion & Scratches**: Linear mechanical distress marks.

### 3.1 Determinism & Reproducibility
- Seed determination: $\text{seed} = \text{int}(\text{object\_id}) \pmod{10^6}$.
- Re-running `scripts/acquire_independent_holdout.py` guarantees bit-identical damage patterns and binary masks.

### 3.2 Mask Convention
- Binary mask format: 8-bit single-channel PNG (`uint8`).
- Polarity: `255` = damaged pixels (loss of pigment/substrate); `0` = pristine authentic artwork.
- Coverage distribution:
  - Minimum coverage: **4.3%** (ID 422552)
  - Maximum coverage: **36.8%** (ID 456964)
  - Mean coverage: **18.0%**
  - Median coverage: **18.7%**

---

## 4. Integrity Verification

The dataset has been verified against the repository database:
```
Clean count: 40
Damaged count: 40
Masks count: 40
Overlap with original 2,000 project IDs: 0
```
All files are saved with UTF-8 metadata in:
- `data/independent_holdout/clean/`
- `data/independent_holdout/damaged/`
- `data/independent_holdout/masks/`
- `data/independent_holdout/metadata.csv`
- `data/independent_holdout/object_ids.csv`
