# MUESUM — Phase F.4: Visual Quality Assessment

**Document ID**: `reports/phase_f4_visual_quality_assessment.md`  
**Date**: November 3, 2026  
**Auditor**: Antigravity Technical Pair Programmer  
**Dataset Split**: 16 Representative Artworks across 8 Primary Curatorial Categories  
**Visual Artifacts**: 5-Panel Composites in `reports/phase_f4_visual_comparisons/compare_5panel_*.png`  
**Checkpoint**: ResNet34-UNet Epoch 8 (`checkpoints/baseline_run/best_model.pth`)  
**Scope**: Perceptual evaluation, artifact classification, and curatorial viability analysis.

---

## 1. Executive Summary & Methodology

Restoration quality cannot be measured by PSNR alone. In museum conservation, mathematical proximity in pixel space does not guarantee historical fidelity. This assessment evaluates 16 diverse artworks spanning 8 distinct curatorial media:
- **Paintings** (Oil on canvas, tempera)
- **Drawings** (Pen, bistre ink, wash)
- **Photographs** (19th-century albumen/silver gelatin)
- **Printed Textiles** (Calico, block prints)
- **Woven Textiles** (Tapestry, Coptic weave)
- **Sculptures** (Carved limestone, marble)
- **Bronze Objects** (Cast bronze reliefs, statuettes)
- **Manuscripts & Books** (Illuminated codices, woodcuts)

For each artwork, a standardized **5-panel composite** was constructed:
`[1. Clean Reference | 2. Damaged Input | 3. Damage Mask Overlay | 4. Raw UNet Output | 5. Mask-Blended Output]`  
Saved in: `reports/phase_f4_visual_comparisons/`

### Curatorial Performance Distribution ($N = 16$)
- **Excellent**: **3 artworks (18.8%)** — Exceptional, natural reconstruction of volumetric form and mineral patinas.
- **Good**: **5 artworks (31.2%)** — Strong restoration; minor textural smoothing visible only under magnification.
- **Acceptable**: **4 artworks (25.0%)** — Effective damage removal, but slight boundary seams or tonal flattening present.
- **Poor**: **2 artworks (12.5%)** — Structural loss; severed lines or coarse weaves replaced with blurry patches.
- **Failed**: **2 artworks (12.5%)** — Severe perceptual and metric degradation; destruction of periodic pattern or photographic grain.

---

## 2. Systematic Evaluation Across 6 Quality Dimensions

### Dimension 1: Damage Removal Efficacy
- **Performance**: **High across all categories**.
- **Observation**: The baseline ResNet34-UNet successfully eliminates dark mold stains, opaque occlusions, surface grime, and deep synthetic scratches. In $100\%$ of cases, the physical damage signature is removed from the visible canvas.

### Dimension 2: Texture Preservation
- **Performance**: **Strong for 3D continuous surfaces; Poor for high-frequency textures**.
- **Observation**:
  - *Smooth / Mineral Media*: Stone carvings, cast bronzes, and smooth plaster retain convincing surface relief.
  - *Textured Media*: Authentic canvas grain, gelatin silver photographic grain, and woven yarn fibers are completely flattened within the inpainting mask due to the conditional-mean smoothing of L1 loss.

### Dimension 3: Colour Consistency & Chromatic Drift
- **Performance**: **High with Mask Blending; Catastrophic with Raw UNet**.
- **Observation**:
  - *Raw UNet*: Severe global color drift ($-3.03\text{ dB}$ across test set). Authentic white paper shifts toward cool gray, and warm varnishes lighten.
  - *Mask-Guided Output*: $100\%$ of authentic pixels retain exact original RGB values ($\text{MAE} = 0.000000$). Inside the damaged mask, the infilled color accurately matches surrounding local chromaticity in $\approx 85\%$ of cases, with minor hue drift occurring only across high-contrast color boundaries.

### Dimension 4: Fine Line & Pattern Reconstruction
- **Performance**: **Low**.
- **Observation**: The model struggles with thin, directional structures ($< 2$ pixels wide). In pen-and-ink drawings (e.g. Guercino `336663`), severed cross-hatching shade lines are not interpolated across gaps; the model treats the space as flat background. Similarly, periodic floral textile prints (`157559`) lose geometric symmetry.

### Dimension 5: Mask Boundary Quality & Seam Artifacts
- **Performance**: **Moderate-to-Good**.
- **Observation**:
  - In images with diffuse lighting or low background grain, the hard binary mask transition is imperceptible.
  - In images with heavy photographic film grain (`283234`) or distinct paper texture, the contrast between authentic high-frequency noise and smoothed model inpainting creates a visible 1-pixel contour boundary ("halo/seam artifact").

### Dimension 6: Structural & Semantic Consistency
- **Performance**: **Good for macro-structures; Weak for micro-semantics**.
- **Observation**: Large architectural elements, sculpture contours, and manuscript margins align cleanly with surrounding context. However, complex semantic features (e.g., facial features, fingers, typographic letterforms) can blur into indistinct forms if damage directly covers them.

---

## 3. Individual Artwork Quality Scorecards (16 Samples)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       16 REPRESENTATIVE ARTWORK ASSESSMENTS                                      │
├────────┬──────────────────┬─────────────────────────────────────┬───────────┬───────────┬─────────────┬──────────┤
│ Obj ID │ Category         │ Title                               │ Mask Cov. │ ΔPSNR     │ ROI ΔPSNR   │ Verdict  │
├────────┼──────────────────┼─────────────────────────────────────┼───────────┼───────────┼─────────────┼──────────┤
│ 252965 │ Bronzes          │ Bronze handle with dogs flanking... │ 8.68%     │ +6.77 dB  │ +7.99 dB    │ EXCELLENT│
│ 196488 │ Bronzes          │ Bronze statuette                    │ 9.79%     │ +4.90 dB  │ +4.93 dB    │ EXCELLENT│
│ 471850 │ Sculptures       │ Sculpture of Moses with Tablets     │ 8.30%     │ +4.69 dB  │ +4.69 dB    │ EXCELLENT│
│ 427650 │ Drawings         │ Sculpture Designs for Saint Peter   │ 16.07%    │ +4.83 dB  │ +5.50 dB    │ GOOD     │
│ 451301 │ Manuscripts      │ Design for a Cup to Serve Wine      │ 17.75%    │ +4.67 dB  │ +5.71 dB    │ GOOD     │
│ 680296 │ Books            │ Ein new getruckt model Büchli       │ 25.63%    │ +4.53 dB  │ +4.79 dB    │ GOOD     │
│ 48953  │ Paintings        │ Landscape with Trees (Ni Zan)       │ 7.77%     │ +3.91 dB  │ +3.92 dB    │ GOOD     │
│ 466379 │ Sculptures       │ Sculpture of an Enthroned King      │ 21.88%    │ +2.73 dB  │ +2.84 dB    │ GOOD     │
│ 451019 │ Textiles-Woven   │ Woven Tapestry Fragment             │ 11.12%    │ +2.31 dB  │ +2.41 dB    │ACCEPTABLE│
│ 354833 │ Books            │ The Works of Publius Virgilius Maro │ 28.71%    │ +1.74 dB  │ +1.74 dB    │ACCEPTABLE│
│ 452913 │ Textiles-Printed │ Textile fragment (Block Print)      │ 27.82%    │ +0.27 dB  │ +0.29 dB    │ACCEPTABLE│
│ 437468 │ Paintings        │ The Portico of a Country Mansion    │ 30.91%    │ -0.17 dB  │ -0.19 dB    │ACCEPTABLE│
│ 329977 │ Textiles-Woven   │ Fragment of a Woman (Coptic)        │ 10.51%    │ -0.38 dB  │ -0.38 dB    │ POOR     │
│ 336663 │ Drawings         │ Allegorical Figure of Painting      │ 17.08%    │ -1.15 dB  │ -1.16 dB    │ POOR     │
│ 157559 │ Textiles-Printed │ Textile fragment (Calico Floral)    │ 31.34%    │ -2.48 dB  │ -2.49 dB    │ FAILED   │
│ 283234 │ Photographs      │ Study of a Sculpture (Albumen)      │ 15.21%    │ -4.10 dB  │ -4.13 dB    │ FAILED   │
└────────┴──────────────────┴─────────────────────────────────────┴───────────┴───────────┴─────────────┴──────────┘
```

---

### Detailed Profiles: Excellent Artworks (3 Samples)

#### 1. Object ID `252965` — Bronze handle with dogs flanking palmette (Bronzes)
- **Metrics**: Damaged $21.64\text{ dB} \rightarrow \mathbf{28.41\text{ dB}}$ ($\mathbf{+6.77\text{ dB}}$ Whole, $\mathbf{+7.99\text{ dB}}$ ROI, $\Delta\text{SSIM} = +0.0440$).
- **Visual Analysis**: Heavy green mold and dark occlusions obscured the classical Roman bronze relief. The UNet reconstructed the diffuse greenish-brown bronze patina, volumetric curved surfaces, and metallic highlights seamlessly. No visible boundaries or color smearing.
- **Classification**: **EXCELLENT**. Production-grade restoration.

#### 2. Object ID `196488` — Bronze statuette (Bronzes)
- **Metrics**: Damaged $25.47\text{ dB} \rightarrow \mathbf{30.37\text{ dB}}$ ($\mathbf{+4.90\text{ dB}}$ Whole, $\mathbf{+4.93\text{ dB}}$ ROI, $\Delta\text{SSIM} = +0.0209$).
- **Visual Analysis**: Scratch marks across the torso and limbs of a cast bronze figure were cleanly inpainted with exact bronze oxidation tone. Surrounding dark studio background remained 100% bit-exact.
- **Classification**: **EXCELLENT**.

#### 3. Object ID `471850` — Sculpture of Moses with Tablets (Sculptures-Stone)
- **Metrics**: Damaged $21.19\text{ dB} \rightarrow \mathbf{25.88\text{ dB}}$ ($\mathbf{+4.69\text{ dB}}$ Whole, $\mathbf{+4.69\text{ dB}}$ ROI, $\Delta\text{SSIM} = +0.0043$).
- **Visual Analysis**: Weathered carved marble relief. The model matched the soft limestone grain and diffuse shadow transitions across carved folds of drapery.
- **Classification**: **EXCELLENT**.

---

### Detailed Profiles: Good Artworks (5 Samples)

#### 4. Object ID `427650` — Sculpture Designs for Saint Peter (Drawings)
- **Metrics**: Damaged $20.26\text{ dB} \rightarrow \mathbf{25.09\text{ dB}}$ ($\mathbf{+4.83\text{ dB}}$ Whole, $\mathbf{+5.50\text{ dB}}$ ROI, $\Delta\text{SSIM} = +0.0113$).
- **Visual Analysis**: Architectural drawing on sepia paper. Large dark stains on open paper areas were completely removed and replaced with harmonious sepia wash tone.
- **Classification**: **GOOD**. Minor line softening on fine border outlines.

#### 5. Object ID `451301` — Design for a Cup to Serve Wine (Codices / Manuscripts)
- **Metrics**: Damaged $22.04\text{ dB} \rightarrow \mathbf{26.71\text{ dB}}$ ($\mathbf{+4.67\text{ dB}}$ Whole, $\mathbf{+5.71\text{ dB}}$ ROI, $\Delta\text{SSIM} = -0.0374$).
- **Visual Analysis**: Islamic illuminated manuscript page. The model accurately matched gold leaf margins and deep cinnabar red borders, removing stains convincingly.
- **Classification**: **GOOD**. Subtle boundary seam detectable under $400\%$ zoom against aged paper texture.

#### 6. Object ID `680296` — Ein new getruckt model Büchli (Books / Woodcut)
- **Metrics**: Damaged $20.34\text{ dB} \rightarrow \mathbf{24.87\text{ dB}}$ ($\mathbf{+4.53\text{ dB}}$ Whole, $\mathbf{+4.79\text{ dB}}$ ROI, $\Delta\text{SSIM} = -0.0489$).
- **Visual Analysis**: High damage coverage ($25.63\%$). Dark blotches over cream paper were eliminated without smudging thick black woodcut border lines.
- **Classification**: **GOOD**.

#### 7. Object ID `48953` — Landscape with Trees in Manner of Ni Zan (Paintings)
- **Metrics**: Damaged $24.27\text{ dB} \rightarrow \mathbf{28.18\text{ dB}}$ ($\mathbf{+3.91\text{ dB}}$ Whole, $\mathbf{+3.92\text{ dB}}$ ROI, $\Delta\text{SSIM} = +0.0021$).
- **Visual Analysis**: Chinese ink and color on paper. Inpainted scratches across soft ink wash match surrounding tone; brushwork outlines remain crisp.
- **Classification**: **GOOD**.

#### 8. Object ID `466379` — Sculpture of an Enthroned King (Sculptures-Stone)
- **Metrics**: Damaged $20.98\text{ dB} \rightarrow \mathbf{23.72\text{ dB}}$ ($\mathbf{+2.73\text{ dB}}$ Whole, $\mathbf{+2.84\text{ dB}}$ ROI, $\Delta\text{SSIM} = -0.0235$).
- **Visual Analysis**: Limestone carving with heavy surface stains. Repaired naturally with soft stone texture.
- **Classification**: **GOOD**.

---

### Detailed Profiles: Acceptable Artworks (4 Samples)

#### 9. Object ID `451019` — Woven Tapestry Fragment (Textiles-Woven)
- **Metrics**: Damaged $23.83\text{ dB} \rightarrow \mathbf{26.15\text{ dB}}$ ($\mathbf{+2.31\text{ dB}}$ Whole, $\mathbf{+2.41\text{ dB}}$ ROI, $\Delta\text{SSIM} = -0.0359$).
- **Visual Analysis**: Restores broad color fields (indigo, terracotta, ochre), but yarn texture is smoothed inside inpainting patches.
- **Classification**: **ACCEPTABLE**. Structurally coherent at normal view, but synthetic smooth patches are obvious upon zoom.

#### 10. Object ID `354833` — The Works of Publius Virgilius Maro (Books)
- **Metrics**: Damaged $16.90\text{ dB} \rightarrow \mathbf{18.65\text{ dB}}$ ($\mathbf{+1.74\text{ dB}}$ Whole, $\mathbf{+1.74\text{ dB}}$ ROI, $\Delta\text{SSIM} = -0.0986$).
- **Visual Analysis**: Heavy damage ($28.71\%$). Removes blotches from paper margins; minor text stroke blurring where damage overlapped blackletter type.
- **Classification**: **ACCEPTABLE**.

#### 11. Object ID `452913` — Textile fragment (Textiles-Printed / Block Print)
- **Metrics**: Damaged $24.27\text{ dB} \rightarrow \mathbf{24.54\text{ dB}}$ ($\mathbf{+0.27\text{ dB}}$ Whole, $\mathbf{+0.29\text{ dB}}$ ROI, $\Delta\text{SSIM} = -0.0486$).
- **Visual Analysis**: High damage ($27.82\%$). Stains are removed, but block-printed dye irregularities are slightly flattened into uniform red-brown tone.
- **Classification**: **ACCEPTABLE**.

#### 12. Object ID `437468` — The Portico of a Country Mansion (Paintings)
- **Metrics**: Damaged $24.75\text{ dB} \rightarrow \mathbf{24.58\text{ dB}}$ ($\mathbf{-0.17\text{ dB}}$ Whole, $\mathbf{-0.19\text{ dB}}$ ROI, $\Delta\text{SSIM} = -0.0789$).
- **Visual Analysis**: Oil painting with clouds and sky. Damage covering sky is inpainted with close tone, but subtle tonal variance between blue sky and white clouds causes minor metric loss. Visually pleasant.
- **Classification**: **ACCEPTABLE**.

---

### Detailed Profiles: Poor & Failed Artworks (4 Samples)

#### 13. Object ID `329977` — Fragment of a Woman (Textiles-Woven / Coptic)
- **Metrics**: Damaged $24.09\text{ dB} \rightarrow \mathbf{23.71\text{ dB}}$ ($\mathbf{-0.38\text{ dB}}$ Whole, $\mathbf{-0.38\text{ dB}}$ ROI, $\Delta\text{SSIM} = -0.0038$).
- **Visual Analysis**: Coarse orthogonal warp-and-weft linen weave. The inpainting replaced damaged threads with a flat, blurry textile patch lacking any orthogonal yarn ridges.
- **Classification**: **POOR**. Loss of textile structural continuity.

#### 14. Object ID `336663` — Allegorical Figure of Painting (Drawings / Guercino)
- **Metrics**: Damaged $26.77\text{ dB} \rightarrow \mathbf{25.62\text{ dB}}$ ($\mathbf{-1.15\text{ dB}}$ Whole, $\mathbf{-1.16\text{ dB}}$ ROI, $\Delta\text{SSIM} = -0.0459$).
- **Visual Analysis**: Parallel cross-hatching shade lines across the figure's shoulder severed by scratches. The UNet failed to infer directional line continuity, filling the scratch with flat paper wash.
- **Classification**: **POOR**. Severed hatching lines break the artist's structural drawing technique.

#### 15. Object ID `157559` — Textile fragment (Textiles-Printed / Calico Floral)
- **Metrics**: Damaged $20.92\text{ dB} \rightarrow \mathbf{18.43\text{ dB}}$ ($\mathbf{-2.48\text{ dB}}$ Whole, $\mathbf{-2.49\text{ dB}}$ ROI, $\Delta\text{SSIM} = -0.1566$).
- **Visual Analysis**: High damage ($31.34\%$) across periodic micro-patterned printed calico floral motif. The UNet averaged the floral petals into indistinct gray-pink smudges rather than reconstructing periodic floral symmetry.
- **Classification**: **FAILED**. Severe visual and metric breakdown.

#### 16. Object ID `283234` — Study of a Sculpture (Photographs / Albumen Print)
- **Metrics**: Damaged $32.84\text{ dB} \rightarrow \mathbf{28.74\text{ dB}}$ ($\mathbf{-4.10\text{ dB}}$ Whole, $\mathbf{-4.13\text{ dB}}$ ROI, $\Delta\text{SSIM} = -0.0111$).
- **Visual Analysis**: 19th-century albumen photograph. High initial quality ($32.84\text{ dB}$) with thin scratches. The original print exhibits micro-grain and subtle silver gelatin tone. The UNet infilled completely flat, grainless pixels. The lack of grain variance produced higher L2 distance than the thin synthetic scratches.
- **Classification**: **FAILED**. Complete destruction of authentic photographic grain structure.

---

## 4. Curatorial Caution: Visual Plausibility vs. Historical Authenticity

> [!WARNING]
> **CRITICAL CURATORIAL DISTINCTION**:  
> In several "Good" and "Acceptable" cases (e.g. `451019` Woven Tapestry or `452913` Block Print), the infilled pixels look smooth, clean, and visually pleasing to a casual observer. However, **this is perceptual hallucination, not historical conservation**.
> 
> A museum conservator requires micro-textural veracity (thread counts, paper fibers, pigment crackle). An AI model that infills flat color across a severed weave or missing ink stroke may look "clean" on screen, but it violates conservation ethics by replacing historical evidence with synthetic blur. 
> 
> Therefore, MUESUM must strictly be positioned as a **digital visualization and conservation preview tool**, not an autonomous replacement for professional physical inpainting.

