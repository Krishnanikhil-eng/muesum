# MUESUM — Dataset Card
**Dataset Name**: MUESUM Cultural Heritage Restoration Corpus  
**Version**: 2.0  
**Source**: The Metropolitan Museum of Art Open Access Collection  
**Standard**: Follows Gebru et al. (2021), *Datasheets for Datasets*  

---

## 1. Motivation & Purpose

Digital restoration research frequently relies on synthetic geometric masks (e.g., random rectangular occlusions or artificial random brushstrokes) applied to generic consumer photo datasets (such as Places2 or ImageNet). Generic consumer datasets fail to capture the unique spectral, textural, and historical characteristics of archival cultural heritage artifacts: craquelure networks, fragile parchment fibers, woven textile warps, ceramic glaze reflections, and patinated bronze surfaces.

The MUESUM dataset was created to provide:
1. An authentic public-domain corpus of diverse historical artworks from a world-renowned museum collection.
2. A physically motivated synthetic degradation simulation taxonomy replicating real-world aging mechanisms.
3. Strict separation of training, validation, test, and independent holdout sets with zero identity leakage.

---

## 2. Dataset Composition & Taxonomy

### 2.1 Primary Baseline Corpus (2,000 Artworks)
- **Total Samples**: 2,000 paired clean and synthetically damaged artworks with binary masks.
- **Data Splits**:
  - **Training Split**: 1,600 artworks (`data/splits/train_ids.csv`)
  - **Validation Split**: 200 artworks (`data/splits/val_ids.csv`)
  - **Test Split**: 200 artworks (`data/splits/test_ids.csv`)
- **Metadata**: Comprehensive curatorial metadata recorded in `data/metadata.csv` (artist, date, medium, culture, department, dimensions).

### 2.2 Independent Holdout Benchmark (40 Artworks)
- **Total Samples**: 40 newly acquired public-domain artworks (`data/independent_holdout/`).
- **Quarantine Guarantee**: Strictly **0 ID overlap** with the 2,000 baseline items ($0 / 2,000$).
- **Category Stratification** (4 items per category):
  1. **Paintings**: Oil on canvas, wood panels (IDs: 659725, 656610, 435581, 646741)
  2. **Drawings**: Ink on paper, graphite studies (IDs: 420455, 12464, 74669, 74662)
  3. **Photographs**: Albumen silver prints (IDs: 646973, 263532, 422287, 422552)
  4. **Sculptures**: Carved marble, limestone (IDs: 329970, 247464, 257433, 253423)
  5. **Bronzes**: Cast bronze statues, patinas (IDs: 254502, 257580, 247117, 246701)
  6. **Textiles**: Woven silks, linens, tapestries (IDs: 44579, 39727, 74365, 463308)
  7. **Books**: Woodcut prints, incunabula (IDs: 388471, 625017, 466568, 661829)
  8. **Manuscripts**: Illuminated vellum, tempera & gold (IDs: 739852, 456964, 74877, 74882)
  9. **Ceramics**: Terracotta, glazed earthenware (IDs: 450206, 456206, 450249, 450193)
  10. **Metalwork**: Chased silver, hammered ornaments (IDs: 50256, 309006, 309087, 309088)

---

## 3. Physical Degradation Simulation Taxonomy

The synthetic degradation engine (`src/damage_sim_museum.py`) simulates 4 authentic decay modalities:
1. **Translucent Liquid Stains & Foxing**:
   - Irregular elliptical moisture boundaries with Gaussian-warped margins and tea/iron-oxide chromatic shifts.
2. **Craquelure & Substrate Fissures**:
   - Branched stochastic random-walk pathways simulating brittle glaze and oil drying cracks.
3. **Biological & Mildew Colonies**:
   - Clustered variable-radius dark mold spots with fuzzy biological perimeter dispersion.
4. **Mechanical Scratches & Abrasions**:
   - High-contrast directional linear scratches simulating surface handling distress.

### Damage Coverage Distribution
- Mean surface coverage: **$18.0\%$**
- Median coverage: **$18.7\%$**
- Minimum coverage: **$4.3\%$**
- Maximum coverage: **$36.8\%$**

---

## 4. Archival Standards & Licensing

- **Clean Artworks**: Sourced from The Metropolitan Museum of Art Open Access Collection under **Creative Commons Zero v1.0 Universal (CC0-1.0)**.
- **Synthetic Annotations & Masks**: Released under CC0-1.0.
- **Mask Convention**: 8-bit grayscale PNG where pixel value `255` = damaged substrate, and `0` = pristine authentic artwork.
