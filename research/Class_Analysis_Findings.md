# Class-Characterization — Findings & Deep Analysis

*Findings companion to `Class_Analysis_Documentation.md` (which records the methodology). This document
records **what the class-characterization analysis (`src/class_analysis.py`) actually found** on the
SIPaKMeD cropped single cells, with every number traced to a persisted artefact. Primary results are
**[measured]** from `results/metrics/class_analysis/`. Follow-on cross-checks (scale, redundancy,
morphometric-vs-CNN agreement) are **[derived]** — computed on the same 53-feature matrix and the saved
classifier metrics, and reproducible with the script snippets noted. No value is fabricated; where a fact
cannot be established locally it is tagged **[unverified]**.*

**Substrate.** N = 4,049 SIPaKMeD cropped cells · 53 handcrafted nucleus+cytoplasm morphometric features
(`explain.features.load_features`, raw/unstandardized) · 5-class morphology and the binary Normal/Abnormal
collapse.

---

## Executive summary

1. **The classes overlap heavily, yet every feature is statistically "significant."** Global silhouette
   is **0.089** (5-class) / **0.095** (binary) — barely above 0 — while **53/53** features pass q<0.05.
   The class signal is *real but distributed and weak per feature*; significance ≠ separability.
2. **Cytoplasm SIZE is the master axis,** and the elevated N/C ratio of abnormal cells is driven by
   **cytoplasm atrophy, not nuclear enlargement** (abnormal nuclei are not larger on average).
3. **Koilocytotic is morphometrically inseparable** (per-class silhouette **−0.004**, the only negative);
   its signature is **localized cytoplasmic texture** (i2 skew/kurtosis — the koilocyte halo), which
   global descriptors barely capture.
4. **The single strongest discriminator is scale-confounded.** `cytoplasm_area` correlates **+0.989** with
   raw crop pixel area, whereas `nucleus_area` correlates **−0.045** (scale-robust) and `nc_ratio`
   is the scale-stable diagnostic. This is the morphometric fingerprint of the cross-dataset collapse.
5. **The 53 features are highly redundant** — **11 pairs** at |r|>0.95 — so the effective dimensionality
   is far below 53.
6. **Morphometric difficulty predicts the CNN's confusion structure:** the least-separable classes
   (Koilocytotic, Metaplastic) are exactly where the strong CNN's **precision** is lowest, and the most-
   separable class (Parabasal) is classified perfectly — an independent, model-free validation.

---

## Finding 1 — Real but weak, distributed signal (overlap ≫ separation)

**[measured]** Global silhouette = **0.0888** (5-class) and **0.0949** (binary). A silhouette near 0 means
the average cell sits almost equidistant from its own class and the nearest other class — the classes
**overlap heavily** in the 53-D morphometric space.

Yet **all 53 features are significant** at q<0.05 (5-class; 50/53 binary). The reconciliation: with
N=4,049, even a negligible effect is "significant." The clearest example is **`nucleus_orientation`**:
q = 0.029 (significant) but ε² = **0.0017** (essentially zero effect). **[measured]**

> **Implication.** No handful of handcrafted features cleanly separates the classes; the signal is spread
> thinly across many weakly-discriminative features. This is *why* a high-capacity CNN (test accuracy
> **0.970**) vastly outperforms any morphometric rule — it composes the distributed weak cues that no
> single feature carries. Handcrafted features are best used as a *complementary* fusion stream, not a
> standalone classifier.

## Finding 2 — Cytoplasm size is the master axis; abnormal N/C is cytoplasm atrophy

The top univariate discriminators are **cytoplasm size** features **[measured]** (5-class ε²):
`cytoplasm_area` 0.68 · `cytoplasm_equivdiameter` 0.68 · `cytoplasm_minoraxis` 0.665 ·
`cytoplasm_majoraxis` 0.633, then `nc_ratio` 0.569.

Raw per-class means **[derived]** make the structure concrete:

| class | cytoplasm_area (px) | nucleus_area (px) | nc_ratio |
|---|---:|---:|---:|
| Superficial-Intermediate | **71,312** | 1,696 | **0.027** |
| Metaplastic | 20,381 | 2,236 | 0.151 |
| Parabasal | 17,494 | **3,141** | 0.257 |
| Koilocytotic | 13,342 | 2,710 | 0.262 |
| Dyskeratotic | **4,771** | **1,229** | **0.340** |

Normal vs Abnormal **[derived]**: `cytoplasm_area` **36,993 → 9,088** (4.1× smaller) while `nc_ratio`
**0.143 → 0.301** (2.1× higher).

> **Implication.** The N/C ratio rises in abnormal cells **mainly because the cytoplasm shrinks**, not
> because the nucleus enlarges — abnormal nuclei are *not* larger on average (Dyskeratotic has the
> **smallest** nucleus, 1,229 px). For this dataset and these contour-derived features, the discriminative
> axis is dominated by overall cell/cytoplasm size, with Superficial-Intermediate as a huge-cytoplasm
> outlier. **Caveat:** this is the morphometric picture from expert contours; it does not contradict
> nuclear-enlargement cytology in general, but it is what *these features on this data* show.

## Finding 3 — Koilocytotic is the inseparable class; its signal is localized texture

**[measured]** Per-class silhouette: Parabasal **0.223** (best) · Dyskeratotic 0.107 · Superficial-
Intermediate 0.074 · Metaplastic 0.049 · **Koilocytotic −0.004** (the only negative — its cells are, on
average, *closer to other classes* than to their own in global morphometric space).

Its defining one-vs-rest features are **cytoplasmic intensity-distribution shape** **[measured]**:
`cytoplasm_i2_skew (+)`, `cytoplasm_i2_kurt (+)`, `cytoplasm_ratio_b (+)`, `cytoplasm_i2_mean (+)`,
`cytoplasm_i1_var (+)` — i.e. perinuclear-halo texture, not a global size/shape change.

> **Implication.** The koilocyte halo is a **localized** texture feature; whole-cell summary statistics
> dilute it, so Koilocytotic does not form a compact morphometric cluster. This is the morphometric root
> of the classifier's hardest decision boundary (see Finding 6) and motivates **localized perinuclear /
> radial-texture features** as a targeted rescue.

## Finding 4 — The strongest discriminator is scale-confounded (the cross-dataset fingerprint)

**[derived]** Correlation of each feature with the **raw crop pixel area** (W×H of the cell image,
n=800 sampled):

| feature | corr with crop pixel area |
|---|---:|
| `cytoplasm_area` | **+0.989** |
| `cytoplasm_equivdiameter` | +0.978 |
| `cytoplasm_minoraxis` | +0.947 |
| `cytoplasm_solidity` | +0.674 |
| `nc_ratio` | −0.617 |
| **`nucleus_area`** | **−0.045** |

The #1 discriminator (`cytoplasm_area`) is **almost entirely a function of crop scale**. Critically, the
confound is **specific to cytoplasm/cell-extent features**: `nucleus_area` is essentially uncorrelated
with crop size (−0.045, scale-robust), and `nc_ratio` is scale-invariant by construction (a ratio of two
areas). Within SIPaKMeD (≈constant µm/px) apparent size is a valid cue, but across datasets imaged at
different scales it is not.

> **Implication.** A model leaning on apparent cytoplasm size has learned a cue that **does not transfer**.
> This is the morphometric mechanism behind the cross-dataset collapse (the data-level scale span —
> SIPaKMeD median ~196 px vs CRIC 1,376 px, ~11× — is documented in `Data_Characterization_Documentation.md`).
> The transferable features are the **scale-robust** ones: `nucleus_area`, `nc_ratio`, and shape ratios.
> **Caveat [unverified]:** physical resolution (µm/px) per dataset is unknown, so the confound can be
> *exposed and mitigated* (N/C-centric features, scale augmentation) but not directly *corrected*.

## Finding 5 — Heavy feature redundancy (effective dimensionality ≪ 53)

**[derived]** **11 feature pairs** have |Pearson r| > 0.95. The strongest:

`cytoplasm_area ↔ cytoplasm_equivdiameter` (+0.975, a geometric identity) · `cytoplasm_i2_skew ↔
cytoplasm_i2_kurt` (+0.975) · `cytoplasm_minoraxis ↔ cytoplasm_equivdiameter` (+0.974) · `cytoplasm_i0_mean
↔ cytoplasm_i0_var` (+0.970) · `cytoplasm_i1_var ↔ cytoplasm_i1_skew` (+0.968) · `nucleus_i2_skew ↔
nucleus_i2_kurt` (+0.967) · `nucleus_i1_var ↔ nucleus_i1_skew` (+0.963) · … (11 total).

Two redundancy families dominate: **size descriptors** (area/equiv-diameter/axes are near-identical) and
**high-order intensity moments** (variance/skew/kurtosis co-move within a channel).

> **Implication.** The "53 features all significant" headline overstates the independent information. A
> redundancy-aware reduction (correlation clustering or an L1/sparse model) would collapse these to a far
> smaller independent set without losing discriminative power, and would de-emphasise the scale-bound size
> cluster of Finding 4.

## Finding 6 — Morphometric separability predicts the CNN's confusion structure (cross-validation)

**[derived]** Per-class morphometric silhouette vs the trained student's per-class metrics
(`indist_student.json`, test acc 0.970):

| class | silhouette | CNN precision | CNN recall |
|---|---:|---:|---:|
| Parabasal | **0.223** | **1.000** | **1.000** |
| Dyskeratotic | 0.107 | 0.987 | 0.926 |
| Superficial-Intermediate | 0.074 | 1.000 | 0.976 |
| Metaplastic | 0.049 | 0.963 | 0.975 |
| Koilocytotic | **−0.004** | **0.909** | 0.976 |

The ordering agrees where it matters: **Parabasal** — most separable morphometrically — is classified
**perfectly**; **Koilocytotic** and **Metaplastic** — the two least separable (silhouette −0.004 and 0.049)
— are exactly the two classes with the **lowest CNN precision** (0.909, 0.963), i.e. the classifier's
false positives concentrate in the morphometrically fuzziest classes. (The CNN is strong enough to keep
*recall* high across the board, so the agreement appears in precision / the confusion structure: the
dominant off-diagonal is Dyskeratotic→Koilocytotic.)

> **Implication.** An entirely model-free, handcrafted-feature analysis **predicts in advance where the
> deep model will struggle.** This independent corroboration strengthens confidence in the benchmark and
> tells us which class boundary (Koilocytotic) most needs targeted features, abstention, or extra data.

## Finding 7 — Orientation and eccentricity carry no class information (augmentation is safe)

**[measured]** The least class-dependent features (smallest ε², 5-class): `nucleus_orientation` 0.0017 ·
`cytoplasm_orientation` 0.0037 · `nucleus_eccentricity` 0.017 · `cytoplasm_eccentricity` 0.033.

> **Implication.** Cell orientation is non-informative and shape is near-isotropic across classes, so
> **rotation/flip augmentation cannot destroy class signal** — it is a safe, well-justified invariance to
> impose. (Contrast with scale augmentation, which *does* touch the diagnostic-but-fragile size axis —
> Finding 4 — and must be reasoned about, not applied blindly.)

---

## Per-class morphometric fingerprint (one-vs-rest defining features) **[measured]**

| class | top defining features (direction) |
|---|---|
| Superficial-Intermediate | cytoplasm_area (+), equivdiameter (+), minoraxis (+), majoraxis (+), solidity (+) → **huge, regular cytoplasm** |
| Parabasal | nucleus_i1_mean (+), nucleus_i2_var (+), cytoplasm_i1_var (−), cytoplasm_i2_mean (−), nucleus_equivdiameter (+) → **large, bright nucleus** |
| Metaplastic | nucleus_i0_kurt (+), nucleus_i0_entropy (−), cytoplasm_perimeter (−), nucleus_i0_mean (−), nucleus_perimeter (−) → **compact, dark-nucleus** |
| Koilocytotic | cytoplasm_i2_skew (+), i2_kurt (+), ratio_b (+), i2_mean (+), i1_var (+) → **perinuclear-halo texture** |
| Dyskeratotic | cytoplasm_i1_mean (−), nucleus_i1_mean (−), nucleus_ratio_a (+), cytoplasm_i2_var (−), nucleus_ratio_b (−) → **small, dark, keratinised** |

## Implications for the framework

- **Fusion, not replacement (RQ7).** Morphometrics separate weakly (silhouette 0.089) where the CNN
  reaches 0.970 accuracy → the value of handcrafted features is complementary; keep them as a fusion
  stream and report the per-class lift, especially on Koilocytotic.
- **Scale is the transfer risk (RQ2/RQ4).** Prefer scale-invariant features (N/C, nuclear, shape ratios);
  treat scale/zoom augmentation as a candidate *fix* for domain shift (and ablate it), not a hazard.
- **Target Koilocytotic (RQ5/RQ7).** Add localized perinuclear/radial-texture features; it is both the
  least-separable class and the CNN's precision sink.
- **Report effective dimensionality, not 53.** Use a redundancy-aware feature set in the tabular/fusion
  branch.

## Reproducibility & provenance

- **[measured]** primary outputs: `results/metrics/class_analysis/class_analysis_{summary,5class,binary}.json`,
  `discriminative_{5class,binary}.csv`, `univariate_relevance_{5class,binary}.csv`; figures under
  `results/figures/class_analysis/`. Regenerate with `python -m class_analysis`.
- **[derived]** Findings 2/4/5/6 recomputed on `explain.features.load_features(standardize=False)` and
  `results/metrics/indist_student.json`; the crop-scale correlation uses raw image W×H (n=800, seeded).
  Deterministic given `C.SEED`.
- **[unverified]** physical resolution (µm/px) per dataset is unknown — the scale confound (Finding 4) is
  therefore exposed and mitigable but not directly correctable from local data.

*Cross-references: `Class_Analysis_Documentation.md` (methodology), `Data_Characterization_Documentation.md`
(dataset-level scale span and exclusion), Volume II Ch. 10–11.*
