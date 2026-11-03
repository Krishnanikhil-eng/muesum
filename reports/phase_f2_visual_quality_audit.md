# MUESUM — Phase F.2: Visual Quality Audit

**Date**: November 3, 2026  
**Artifacts Audited**:
- 12 4-Panel Visual Comparisons: `reports/baseline_evaluation/mask_blending_experiment/full_200/visual_comparisons/compare_4panel_*.png`
- 27 Negative-Delta Samples from: `reports/baseline_evaluation/mask_blending_experiment/full_200/full_200_metrics.csv`  
**Checkpoint**: Epoch 8 Best Checkpoint (`checkpoints/baseline_run/best_model.pth`)  
**Scope**: Qualitative inspection, failure mode categorization, and benchmark artwork profiling.

---

## 1. Executive Summary

While Phase F.1 established mathematical integrity ($+1.94\text{ dB}$ whole-image gain across 200 images, $86.5\%$ win rate), Phase F.2 investigates the **perceptual quality** of the restored artworks. 

The audit reveals that mask-guided blending resolves the primary failure mode of raw UNet (global luminance drift and loss of untouched artwork fidelity). However, inside the damaged regions themselves, the L1-trained baseline model exhibits distinct perceptual limitations:
1. **Low-Pass Filtering (Texture Smoothing)**: High-frequency details (film grain, fine cross-hatching, textile thread weaves) are replaced with smooth, continuous color tones.
2. **Boundary Discontinuities ("Seam Artifacts")**: Binary transition between authentic high-frequency noise and smoothed model inpainting can produce a visible boundary under magnification.
3. **Curatorial Sensitivity**: Continuous-tone three-dimensional artifacts (sculptures, bronzes, stone carvings, broad parchment manuscripts) achieve exceptional visual restorations ($+3\text{ dB}$ to $+7.6\text{ dB}$ ROI gain), whereas high-frequency graphic media (fine line drawings, silver gelatin photographs, delicate lace textiles) experience metric penalties due to oversmoothing.

---

## 2. Detailed Inspection of the 12 Representative Benchmark Artworks

The 12 four-panel comparison composites (`[Original Clean | Damaged Input | Raw UNet Output | Mask-Blended Output]`) located in `reports/baseline_evaluation/mask_blending_experiment/full_200/visual_comparisons/` were analyzed:

| Object ID | Classification | Title | Mask Cov. | Blended $\Delta\text{PSNR}$ | ROI $\Delta\text{PSNR}$ | SSIM | Visual Assessment & Key Observations |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `247010` | Miscellaneous-Paintings | Wall painting from Villa of P. Fannius Synistor | 27.00% | **$+3.60\text{ dB}$** | $+3.62\text{ dB}$ | 0.8678 | **High Quality**. Dark stains and mold on the vibrant cinnabar-red fresco and black architectural borders are removed without discoloration. |
| `48953` | Paintings | Landscape with Trees (Ni Zan style) | 7.77% | **$+3.93\text{ dB}$** | $+3.94\text{ dB}$ | 0.9652 | **High Quality**. Fine scratches across open paper wash are filled with matching warm cream tone. Ink tree contours remain crisp. |
| `451301` | Codices | Design for a Cup to Serve Wine | 17.75% | **$+4.76\text{ dB}$** | $+6.00\text{ dB}$ | 0.9309 | **Outstanding**. Mold and scratches on illuminated gold leaf and red borders are inpainted cleanly with correct local hue. |
| `451019` | Textiles-Woven | Woven Tapestry Fragment | 11.12% | **$+3.58\text{ dB}$** | $+3.85\text{ dB}$ | 0.9015 | **Good**. Large occlusions and blotches are removed; macro geometric patterns are reconstructed, though micro-weave texture is smoothed. |
| `336663` | Drawings | Allegorical Figure of Painting (Guercino) | 17.09% | **$-2.85\text{ dB}$** | $-2.87\text{ dB}$ | 0.9431 | **Problematic**. Fine pen-and-ink cross-hatching severed by synthetic scratches is filled with flat paper color rather than continuous line strokes. |
| `466379` | Sculpture-Stone | Sculpture of an Enthroned King | 21.89% | **$+2.68\text{ dB}$** | $+2.83\text{ dB}$ | 0.8030 | **Strong**. Weathered limestone carvings and shadows are restored naturally; stains and surface marks blend seamlessly into stone texture. |
| `403800` | Prints | Portraits of Ancient Philosophers | 28.29% | **$+4.03\text{ dB}$** | $+4.10\text{ dB}$ | 0.8634 | **Strong**. Severe damage (28.29% coverage) covering wide margin paper and engraving borders is inpainted without hue distortion. |
| `354833` | Books | The Works of Publius Virgilius Maro | 28.71% | **$+2.85\text{ dB}$** | $+2.85\text{ dB}$ | 0.8420 | **Good**. Heavy black woodcut print and aged paper stains are removed cleanly; typographic block contours are well preserved. |
| `283234` | Photographs | [Study of a Sculpture] | 15.20% | **$-4.32\text{ dB}$** | $-4.36\text{ dB}$ | 0.9733 | **Most Problematic**. Ultra-smooth photographic gradient with silver gelatin grain. UNet infill is unnaturally smooth, causing pixel variance loss against authentic film grain. |
| `314620` | Textiles-Costumes | Four-Cornered Hat (Wari Culture) | 16.39% | **$+1.11\text{ dB}$** | $+1.19\text{ dB}$ | 0.8683 | **Moderate**. Polychrome geometric wool patterns are restored to correct colors, but fibrous knot texture is smoothed out. |
| `437468` | Paintings | The Portico of a Country Mansion | 30.90% | **$-0.11\text{ dB}$** | $-0.12\text{ dB}$ | 0.8680 | **Neutral/Borderline**. Oil painting with cloud gradients; inpainting tone closely matches sky, but slight tonal discrepancy yields nominal $-0.11\text{ dB}$ shift. |
| `324005` | Unclassified (Vase) | Vase with a handle in the form of a dog | 17.52% | **$-0.30\text{ dB}$** | $-0.32\text{ dB}$ | 0.8748 | **Neutral/Borderline**. Smooth ceramic clay finish is visually well-repaired, but specular highlight reconstruction differs slightly from ground truth. |

---

## 3. Systematic Investigation of Visual Quality Dimensions

### 3.1 Colour Shifts: Global Elimination vs. Local Infilling Hue
- **Raw UNet Baseline**: Suffers from pervasive global color shifts. Because raw UNet passes the entire image through encoder-decoder convolutions, authentic white paper shifts toward pale gray-blue, and dark varnishes lighten, resulting in $-3.03\text{ dB}$ mean degradation.
- **Mask-Guided Blending**: Completely eliminates global shifts. $100\%$ of undamaged pixels maintain their exact authentic RGB values ($\text{MAE} = 0.000000$).
- **Local Infilling Hue**: Within the damaged regions ($M=1$), the model generally captures the dominant local chromaticity. However, where damage crosses high-contrast color boundaries (e.g., from deep blue drapery to golden skin tones), the infilled pixels display a 2-to-3 pixel transitional blur rather than a sharp color transition.

### 3.2 Blurring and Smoothing (The L1 Expectation Phenomenon)
- The baseline model was trained under masked L1 pixel loss: $\mathcal{L}_{\text{L1}} = \|M \odot (\hat{Y} - Y)\|_1$.
- Mathematically, minimizing L1/L2 distance under multimodal uncertainty forces the network toward the conditional mean $\mathbb{E}[Y|X]$.
- Consequently, high-frequency spatial variation is penalized if uncertain, causing the model to generate smooth, averaged color fills. This is perceptually pleasing on flat surfaces (plaster, stone, skies) but creates noticeable smoothing on textured canvas, craquelure, or paper pulp.

### 3.3 Loss of Fine Details
- **Intricate Line Work**: In drawings and etchings (e.g. `336663`), scratches that cross $1$-pixel-wide pen strokes are not continued across the gap. The model treats the line interruption as flat background.
- **Micro-Textures**: Film grain (`283234`), woven thread ribs (`329977`), and woodblock hatching (`157559`) are lost inside the inpainting zone.
- **Semantic Edges**: Major compositional boundaries (e.g. architectural columns, horizon lines) are reconstructed accurately, but secondary textures within those regions are low-pass filtered.

### 3.4 Boundary Transitions & Seam Artifacts
- The blending equation is hard binary: $\text{Final} = \text{Restored} \odot M + \text{Original} \odot (1 - M)$.
- Where the authentic background has high-frequency noise/grain (e.g. standard deviation $\sigma > 8.0$), the transition to a smooth model inpainting ($\sigma < 2.0$) creates a visible perceptual seam along the contour of $M$.
- **Recommendation**: In Phase F.3, introduce an optional **Poisson image editing** or **narrow-band Gaussian feathering** ($\sigma = 1.0$, $3 \times 3$ kernel) at mask boundaries to eliminate abrupt seam transitions.

### 3.5 Differences Across Curatorial Categories

```
Higher Visual Fidelity ────────────────────────────────────────────────────────► Lower Visual Fidelity
[Stone / Sculptures]    [Codices / Books]    [Oil Paintings]    [Woven Textiles]    [Photographs / Drawings]
(Smooth gradients,      (Broad margins,      (Rich impasto,     (Geometric weave,   (High-frequency grain,
 continuous tone)        clear contrasts)     moderate grain)    fiber texture lost) fine cross-hatching)
```

1. **Stone, Sculptures, and Bronzes**: Highest fidelity ($\Delta\text{PSNR} > +3.5\text{ dB}$, win rate $100\%$). Inherent mineral grain and diffuse lighting mask any minor smoothing.
2. **Codices and Manuscripts**: High fidelity ($\Delta\text{PSNR} > +2.2\text{ dB}$). Wide manuscript margins and large decorative illuminations recover cleanly.
3. **Paintings**: Moderate-to-high fidelity ($\Delta\text{PSNR} \approx +1.8\text{ dB}$). Canvas weave is smoothed, but overall compositional harmony is preserved.
4. **Textiles (Woven/Embroidered)**: Mixed fidelity ($\Delta\text{PSNR} \approx +1.3\text{ dB}$). Macro colors are restored, but loss of weave pattern is discernible upon zooming.
5. **Photographs & Fine Drawings**: Vulnerable ($\Delta\text{PSNR} < 0\text{ dB}$). Unforgiving to smoothing due to strict reliance on microscopic paper grain and line continuity.

---

## 4. Benchmark Profiling: The 5 Most Problematic Artworks

These 5 artworks exhibited the largest metric degradation under mask blending. In all 5 cases, the undamaged regions are bit-exact ($0.0$ error), meaning the negative whole-image score is driven entirely by the model's inability to match high-frequency textures inside the damage mask:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               5 MOST PROBLEMATIC ARTWORKS                               │
├────────┬─────────────────┬─────────────────────────────────────┬───────────┬───────────┤
│ Obj ID │ Category        │ Title                               │ Mask Cov. │ Blended Δ │
├────────┼─────────────────┼─────────────────────────────────────┼───────────┼───────────┤
│ 283234 │ Photographs     │ [Study of a Sculpture]              │ 15.20%    │ -4.32 dB  │
│ 336663 │ Drawings        │ Allegorical Figure of Painting      │ 17.09%    │ -2.85 dB  │
│ 157559 │ Textiles-Print. │ Textile fragment                    │ 31.35%    │ -2.71 dB  │
│ 329977 │ Textiles-Woven  │ Fragment of a Woman                 │ 10.53%    │ -2.19 dB  │
│ 54421  │ Textiles-Embr.  │ Eyeglasses Case Bag with Bamboo     │ 32.52%    │ -1.92 dB  │
└────────┴─────────────────┴─────────────────────────────────────┴───────────┴───────────┘
```

### Detailed Problem Profiles:

1. **Object ID `283234` — `[Study of a Sculpture]` (Photographs)**
   - **Baseline Damaged PSNR**: $33.13\text{ dB}$ (extremely high initial quality; damage consisted of thin, localized hairline scratches).
   - **Blended PSNR**: $28.80\text{ dB}$ ($\Delta\text{PSNR} = -4.32\text{ dB}$, $\Delta\text{PSNR}_{\text{ROI}} = -4.36\text{ dB}$).
   - **Root Cause**: The original albumen print exhibits fine photographic grain and subtle gelatin tone gradients. While the synthetic scratches were thin, the UNet's infilled pixels were completely uniform and grainless. The lack of microscopic noise variance yielded a higher L2 distance than the thin synthetic scratches themselves.

2. **Object ID `336663` — `Allegorical Figure of Painting` (Drawings)**
   - **Baseline Damaged PSNR**: $28.55\text{ dB}$ | **Blended PSNR**: $25.70\text{ dB}$ ($\Delta\text{PSNR} = -2.85\text{ dB}$).
   - **Root Cause**: Guercino pen-and-brown-ink master drawing. Scratches bisected parallel cross-hatching shade lines on the figure's shoulder. The UNet failed to infer directional line continuity and inpainted flat paper wash, breaking the hatching rhythm.

3. **Object ID `157559` — `Textile fragment` (Textiles-Printed)**
   - **Baseline Damaged PSNR**: $22.14\text{ dB}$ | **Blended PSNR**: $19.43\text{ dB}$ ($\Delta\text{PSNR} = -2.71\text{ dB}$).
   - **Root Cause**: High damage coverage ($31.35\%$) across a repetitive micro-patterned printed calico floral motif. The UNet averaged the floral petals into indistinct gray-pink smudges rather than reconstructing periodic floral symmetry.

4. **Object ID `329977` — `Fragment of a Woman` (Textiles-Woven)**
   - **Baseline Damaged PSNR**: $27.71\text{ dB}$ | **Blended PSNR**: $25.51\text{ dB}$ ($\Delta\text{PSNR} = -2.19\text{ dB}$).
   - **Root Cause**: Coptic woven tapestry with distinct coarse warp-and-weft threads. The inpainting replaced damaged threads with a flat, blurry textile patch lacking any orthogonal weave structure.

5. **Object ID `54421` — `Eyeglasses Case Bag with Bamboo on Waves` (Textiles-Embroidered)**
   - **Baseline Damaged PSNR**: $20.73\text{ dB}$ | **Blended PSNR**: $18.81\text{ dB}$ ($\Delta\text{PSNR} = -1.92\text{ dB}$).
   - **Root Cause**: High damage coverage ($32.52\%$) over metallic thread embroidery. Specular light reflections from silk embroidery threads were replaced with matte, dull fills.

---

## 5. Benchmark Profiling: The 5 Strongest Artworks

These 5 artworks achieved the highest restoration improvements in the entire 200-image test set:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                5 STRONGEST ARTWORKS                                    │
├────────┬─────────────────┬─────────────────────────────────────┬───────────┬───────────┤
│ Obj ID │ Category        │ Title                               │ Mask Cov. │ Blended Δ │
├────────┼─────────────────┼─────────────────────────────────────┼───────────┼───────────┤
│ 252965 │ Bronzes         │ Bronze handle with dogs flanking... │ 8.68%     │ +6.33 dB  │
│ 572111 │ Unclassified    │ Amulet of a crown of Lower Egypt    │ 17.37%    │ +5.46 dB  │
│ 891645 │ Unclassified    │ Brooch with Assyrian winged bull    │ 8.58%     │ +5.05 dB  │
│ 317301 │ Stone-Ornaments │ Celt-Form Pendant                   │ 5.32%     │ +4.93 dB  │
│ 680296 │ Books           │ Ein new getruckt model Büchli...    │ 25.65%    │ +4.93 dB  │
└────────┴─────────────────┴─────────────────────────────────────┴───────────┴───────────┘
```

### Detailed Success Profiles:

1. **Object ID `252965` — `Bronze handle with dogs flanking a palmette` (Bronzes)**
   - **Baseline Damaged PSNR**: $22.19\text{ dB}$ | **Blended PSNR**: **$28.52\text{ dB}$** ($\mathbf{+6.33\text{ dB}}$ whole, $\mathbf{+7.60\text{ dB}}$ ROI, SSIM $0.9509$).
   - **Success Factors**: Heavy mold stains and opaque occlusions obscured the green bronze patina and cast bronze relief. The UNet accurately reconstructed the diffuse bronze oxidation hue and volumetric shading, yielding the highest ROI gain in the entire project.

2. **Object ID `572111` — `Amulet of a crown of Lower Egypt` (Egyptian Art)**
   - **Baseline Damaged PSNR**: $20.38\text{ dB}$ | **Blended PSNR**: **$25.84\text{ dB}$** ($\mathbf{+5.46\text{ dB}}$ whole, $\mathbf{+5.90\text{ dB}}$ ROI, SSIM $0.9807$).
   - **Success Factors**: Bright red faience glass glaze with high damage coverage ($17.37\%$). The UNet smoothly inpainted the red saturated surface and isolated the object cleanly against its neutral museum background.

3. **Object ID `891645` — `Brooch with Assyrian human-headed winged bull` (Ancient Near East)**
   - **Baseline Damaged PSNR**: $20.75\text{ dB}$ | **Blended PSNR**: **$25.79\text{ dB}$** ($\mathbf{+5.05\text{ dB}}$ whole, $\mathbf{+5.22\text{ dB}}$ ROI, SSIM $0.9635$).
   - **Success Factors**: Gold repoussé brooch. The model accurately matched the golden metallic sheen and surrounding shadow contours, effectively eliminating dark surface blemishes.

4. **Object ID `317301` — `Celt-Form Pendant` (Pre-Columbian Art, Stone-Ornaments)**
   - **Baseline Damaged PSNR**: $28.04\text{ dB}$ | **Blended PSNR**: **$32.96\text{ dB}$** ($\mathbf{+4.93\text{ dB}}$ whole, $\mathbf{+4.99\text{ dB}}$ ROI, SSIM $0.9576$).
   - **Success Factors**: Polished green jadeite carving. The model perfectly matched the translucent green stone color and soft surface reflection across multiple small scratch marks.

5. **Object ID `680296` — `Ein new getruckt model Büchli...` (Illustrated Lace Books)**
   - **Baseline Damaged PSNR**: $20.97\text{ dB}$ | **Blended PSNR**: **$25.90\text{ dB}$** ($\mathbf{+4.93\text{ dB}}$ whole, $\mathbf{+5.32\text{ dB}}$ ROI, SSIM $0.8733$).
   - **Success Factors**: High damage coverage ($25.65\%$) across aged cream paper with geometric woodcut borders. The UNet filled paper stains with the authentic aged paper tone without smudging black ink contours.

---

## 6. Actionable Conclusions for System Architecture

1. **Retain Mask-Guided Blending as Default**:
   Mask blending is indispensable. It protects $100\%$ of undamaged pixels and turns an overall failure mode (raw UNet $-3.03\text{ dB}$) into a substantial net gain ($+1.94\text{ dB}$).
2. **Explicit User Mask Requirement**:
   Because the inpainting quality depends on restricting the model to damaged pixels, the inference interface must never "guess" or fabricate a mask. The user must provide a mask, or use an interactive painting tool.
3. **Dual Comparison in UI**:
   The user interface must provide side-by-side visualization of (a) Damaged Input, (b) Mask-Blended Restoration, and (c) Raw UNet Output, allowing conservators to verify exactly what was altered.
4. **Diagnostic Feedback**:
   The application must report damage coverage percentage, inference latency, and confirm $100\%$ undamaged preservation to provide transparency and build trust with conservators.

