# Class-Characterization Analysis — Methodology, Results, and Documentation

*Companion technical documentation for `src/class_analysis.py`. This document records the methodology,
the measured results, their interpretation, the figure/artefact catalogue, and reproducibility details
for the morphometric class-characterization analysis used in Volume II (Chapters 10–11). All numerical
results reported here are **[measured]** — produced by running the analysis on the on-disk data — and are
reproducible from the persisted artefacts. No value has been fabricated.*

---

## 1. Purpose

The analysis answers a question that underpins the whole framework: **what, morphometrically, does each
class (5-class morphology and binary Normal/Abnormal) have in common, and what distinguishes it?** It
characterises the *data*, independently of any classifier, using ground-truth handcrafted features, and
thereby (i) establishes the intrinsic separability of the classes, (ii) identifies the features that
define each class, and (iii) provides an independent, model-free check on the deep classifier's behaviour
(does the morphometric difficulty of a class predict the CNN's difficulty on it?). It informs RQ1 (honest
characterisation of the benchmark), RQ4 (the basis for clinically meaningful class grouping), and RQ7
(which morphometric features are diagnostic — the substrate for the explanation suite).

## 2. Data substrate

- **Source:** SIPaKMeD cropped single cells (the training source). **[measured]** N = 4,049 cells.
- **Features:** 53 handcrafted morphometric features per cell, extracted from the expert nucleus and
  cytoplasm contours (`.dat`), via `explain.features.load_features`. The feature families are:
  *nuclear* (26) and *cytoplasmic* (26) descriptors — geometry (area, major/minor axis, eccentricity,
  orientation, equivalent diameter, solidity, extent, perimeter), three-channel intensity statistics
  (mean, variance, skewness, kurtosis, entropy), and two ratio descriptors — plus the diagnostic
  **nuclear-to-cytoplasmic area ratio** (`nc_ratio`). **[measured: 53 features]**
- **Groupings analysed:** the native **5-class** morphology taxonomy and the **binary** Normal/Abnormal
  collapse (Normal = {Superficial-Intermediate, Parabasal, Metaplastic}; Abnormal = {Koilocytotic,
  Dyskeratotic}).
- **Standardization:** features are z-scored (per-feature mean/SD over all cells) for centroid and
  separability analyses; raw features are used for effect-size and distribution analyses.

## 3. Methodology

The analysis follows standard practice in morphometric/omics class characterization. Each method is
non-parametric or distributional-assumption-light, because morphometric features are typically non-normal.

### 3.1 Class prototype profiles (shared signature)
The z-scored per-class centroid matrix (class × feature) is computed and rendered as a diverging heatmap.
A cell's value is the mean z-score of that feature within the class: positive = the class is
characteristically high on the feature, negative = characteristically low. This answers "what each class
has in common." *Artefacts:* `profiles_5class`, `profiles_binary` (+ CSV).

### 3.2 Discriminative features (one-vs-rest Cohen's d)
For each class k and feature j, Cohen's d (standardized mean difference, pooled SD) is computed between
class k and all other classes. |d| is rendered as a heatmap, and the top features per class are tabulated.
This answers "what makes each class different." *Artefacts:* `discriminative_absD_*` heatmaps;
`discriminative_*.csv` (per-class ranked features with d and direction). Effect-size convention (Cohen,
1988): |d| ≈ 0.2 small, 0.5 medium, 0.8 large.

### 3.3 Univariate relevance with multiple-comparison control
Per feature, a class-dependence test is run: **Kruskal–Wallis H** (non-parametric one-way ANOVA) for the
5-class grouping and **Mann–Whitney U** for the binary grouping. p-values across the 53 features are
adjusted by the **Benjamini–Hochberg FDR** procedure (q-values). Effect size is reported as **epsilon²**
(5-class) or |rank-biserial| (binary). *Artefacts:* `univariate_relevance_*.csv` (feature, statistic,
p, effect size, q).

### 3.4 Feature-space separability (projection + silhouette)
Three 2-D projections of the standardized feature space, coloured by class: **PCA** (unsupervised linear),
**LDA** (supervised, maximises class separation), and **t-SNE** (non-linear neighbourhood embedding,
seed-fixed). Cluster separability is quantified by the **silhouette coefficient** (Rousseeuw, 1987),
global and per-class; silhouette ∈ [−1, 1], higher = better-separated, ≈0 = overlapping, <0 = the average
cell is closer to another class than its own. *Artefacts:* `pca_*`, `lda_*`, `tsne_*` (+ CSV of coords);
silhouettes in the JSON.

### 3.5 Visual prototypes (average cell)
The mean cell image per class (and per binary group), over up to 400 randomly sampled cells per class, is
rendered as a montage — the "average appearance." *Artefacts:* `avg_cell_5class`, `avg_cell_binary`.

### 3.6 Top-feature distributions
Violin plots of the most globally discriminative features, split by class. *Artefacts:* `dist_5class_*`,
`dist_binary_*` (+ CSV of per-group summaries).

*References:* Cohen (1988) effect size; Kruskal & Wallis (1952); Benjamini & Hochberg (1995) FDR;
Fisher (1936) LDA; van der Maaten & Hinton (2008) t-SNE; Rousseeuw (1987) silhouette.

## 4. Measured results — 5-class morphology

**[measured; `class_analysis_5class.json`]**

### 4.1 Separability
- **Global silhouette: 0.089** — a *low* value indicating that the five morphology classes **overlap
  substantially** in the 53-D morphometric space; they are not cleanly separable clusters.
- **Per-class silhouette:**

  | Class | Silhouette | Interpretation |
  |---|---|---|
  | Parabasal | **0.223** | best-separated class |
  | Dyskeratotic | 0.107 | moderately separated |
  | Superficial-Intermediate | 0.074 | weakly separated |
  | Metaplastic | 0.049 | poorly separated |
  | **Koilocytotic** | **−0.004** | **inseparable** (avg. cell nearer another class) |

### 4.2 Feature relevance
- **All 53/53 features are significantly class-dependent** at q < 0.05 (Kruskal–Wallis, BH-corrected) —
  every morphometric feature carries some class information.
- **Most relevant features (epsilon² effect size):** cytoplasm_area (0.68), cytoplasm_equivdiameter
  (0.68), cytoplasm_minoraxis (0.665), cytoplasm_majoraxis (0.633), **nc_ratio (0.569)**, nucleus_i1_mean
  (0.55), cytoplasm_i1_mean (0.529), cytoplasm_solidity (0.51). **Cytoplasm size features dominate, with
  the nuclear-to-cytoplasmic ratio close behind at rank 5.**

### 4.3 Defining features per class (top one-vs-rest |d|)
- **Superficial-Intermediate:** large cytoplasm (area +, equiv-diameter +, axes +, solidity +) — the
  abundant-cytoplasm normal cell.
- **Parabasal:** prominent nuclear intensity/texture (nucleus_i1_mean +, nucleus_i2_var +,
  nucleus_equiv-diameter +) — small cell, conspicuous nucleus.
- **Metaplastic:** nuclear chromatin texture (nucleus_i0_kurtosis +, nucleus_i0_entropy −) and reduced
  perimeter.
- **Koilocytotic:** cytoplasmic intensity asymmetry (cytoplasm_i2_skew +, i2_kurtosis +, i2_mean +) —
  consistent with the perinuclear-halo cytopathic effect.
- **Dyskeratotic:** reduced intensity (cytoplasm_i1_mean −, nucleus_i1_mean −) and altered nuclear ratios
  — consistent with abnormal keratinisation/darkening.

## 5. Measured results — binary Normal/Abnormal

**[measured; `class_analysis_binary.json`]**
- **Global silhouette: 0.095** (Normal 0.109, Abnormal 0.074) — again low; the binary classes overlap.
- **50/53 features significant** at q < 0.05 (Mann–Whitney, BH-corrected).
- **Most relevant features:** cytoplasm_minoraxis (0.723), cytoplasm_equivdiameter (0.718),
  cytoplasm_area (0.718), cytoplasm_majoraxis (0.672), nucleus_i1_mean (0.637), cytoplasm_i1_mean (0.628).
- **Direction:** Abnormal cells have **smaller cytoplasm** (negative on all cytoplasm-size features),
  i.e., a **higher N/C ratio** — the textbook diagnostic direction. `nc_ratio` ranks ~8th by univariate
  effect; cytoplasm *size* is the proximate driver, and N/C ratio (which size partly determines) is
  strongly and correctly oriented.

## 6. Interpretation and key findings

1. **The morphology classes genuinely overlap (silhouette ≈ 0.09).** This is the quantitative,
   model-free confirmation of the "continuous morphological spectrum" premise (Volume I, §2.2): the
   classes are not separable clusters in morphometric space, which bounds the achievable separability of
   *any* classifier and reframes high reported accuracy as warranting scrutiny.
2. **Cross-validation of class difficulty between two independent methods.** The morphometric per-class
   silhouette ranks Koilocytotic *last* (−0.004, inseparable) and Parabasal *first* (0.223); independently,
   the deep classifier's per-class recall is *lowest* for Koilocytotic (~0.87, confused with Metaplastic)
   and *highest* for Parabasal (~1.00). Two methods that share no machinery — handcrafted morphometrics
   and learned deep features — **agree on which classes are hard**, providing strong evidence that the
   difficulty is *intrinsic to the cell morphology* rather than an artefact of either method. This is, in
   the author's interpretation, the most important finding of the analysis.
3. **Cytoplasm size leads; N/C ratio is strong and correctly oriented.** The a-priori clinical expectation
   that N/C ratio is the dominant discriminator is *partially* revised by the data: cytoplasm *size*
   features are the strongest univariate discriminators, with N/C ratio close behind (rank 5 of 53,
   5-class) and oriented exactly as cytology predicts (abnormal → smaller cytoplasm → higher N/C). This is
   an honest, data-driven refinement, not a contradiction, since cytoplasm area is a direct component of
   the N/C ratio.
4. **Every feature is class-informative (53/53 significant).** No morphometric feature is irrelevant,
   supporting the use of the full handcrafted feature vector for the multimodal/explanation branch (RQ7).

## 7. Artefact catalogue

**Metrics (`results/metrics/class_analysis/`):**
- `class_analysis_5class.json`, `class_analysis_binary.json`, `class_analysis_summary.json` — silhouettes,
  significant-feature counts, top relevant features, defining features per class.
- `discriminative_{5class,binary}.csv` — per-class ranked one-vs-rest Cohen's d.
- `univariate_relevance_{5class,binary}.csv` — per-feature statistic, p, effect size, BH q-value.

**Figures (`results/figures/class_analysis/`; each data-plot has a companion `.csv`):**
- `profiles_{5class,binary}` — class prototype-profile heatmaps (z-scored centroids).
- `discriminative_absD_{5class,binary}` — |Cohen's d| one-vs-rest heatmaps.
- `pca_{5class,binary}`, `lda_{5class,binary}`, `tsne_{5class,binary}` — 2-D separability projections.
- `dist_{5class,binary}_<feature>` — violins of the top discriminative features by class.
- `avg_cell_{5class,binary}` — mean-cell visual prototypes (image montages; no companion CSV by design).

## 8. Reproducibility

- **Run:** `PYTHONPATH=src python -m class_analysis` (or `class_analysis.run()`); seed = 42 (config).
- **Determinism:** PCA/LDA/t-SNE are seeded; t-SNE uses `init="pca"`, perplexity 30. Average-cell sampling
  is seeded. Re-running reproduces the metrics; t-SNE coordinates are seed-stable but, as a non-convex
  embedding, should be read qualitatively.
- **Inputs:** depends only on the SIPaKMeD contours via `explain.features.load_features`; no model
  checkpoint is required (this is a data analysis, not a model analysis).
- **Wiring:** invoked automatically after the all-dataset EDA in `run.eda()`, and available as a standalone
  module and notebook cell.

## 9. Limitations

- The analysis characterises **SIPaKMeD only** (the dataset with ground-truth contours); the external
  datasets lack the per-cell nucleus/cytoplasm contours required for identical morphometric features, so
  cross-dataset morphometric comparison is out of scope here.
- Handcrafted features are a *proxy* for cell morphology; they do not capture all information the deep
  model may use, so the morphometric separability is a lower bound on achievable separability.
- t-SNE is a visualisation, not a metric; quantitative separability claims rest on the silhouette
  coefficient and the effect sizes, not on the embedding.
- The binary collapse inherits the morphological overlap of its constituent classes, so its low silhouette
  is expected and should not be read as a defect of the binary task.

---

*This documentation accompanies the measured artefacts; all figures and tables are regenerable from the
persisted CSV/JSON files and the seeded analysis.*
