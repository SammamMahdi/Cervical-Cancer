# Data Plan — SIPaKMeD-Trained Cervical Cytology Classifier

Committed design: **train a 5-class SIPaKMeD model; evaluate cross-dataset by projecting each other dataset's labels into the SIPaKMeD space.** All dataset facts verified against primary sources (June 2026). Companion docs: [LABEL_ALIGNMENT_METHOD.md](LABEL_ALIGNMENT_METHOD.md) (methodology), [Ablation and Evaluation plan.md](Ablation%20and%20Evaluation%20plan.md) (experiments), [PIPELINE_DOCUMENTATION.md](../PIPELINE_DOCUMENTATION.md) (current model).

## 0. Decision summary
- **Train label space = SIPaKMeD 5 classes** (Superficial-Intermediate, Parabasal, Metaplastic, Koilocytotic, Dyskeratotic) — unchanged.
- **Train data = SIPaKMeD only** (the sole dataset native to these 5 classes), with a **group-aware 80/10/10 train/val/test split**.
- **Cross-dataset evaluation** = run the trained model on Herlev / CRIC / APCData / RIVA / APACC (and optionally image-level BMT / Mendeley LBC), then score by collapsing **both** the model's 5-class prediction **and** the target dataset's native labels into a **shared, deterministically-mappable level** (binary, and a 3-class grade with a stated structural blind spot).
- **Why not train multi-dataset in 5-class:** projecting other labels *into* the 5 classes is one-to-many / no-map (see [LABEL_ALIGNMENT_METHOD.md](LABEL_ALIGNMENT_METHOD.md)); fine for **evaluation collapse**, too lossy for **training supervision**. Other datasets may still be used **label-free (SSL pretraining)** and as **external test sets**.

---

## 1. Dataset landscape (in-scope)

| Dataset | Image type | Size | Native labels | Per-cell? | Role here |
|---|---|---|---|---|---|
| **SIPaKMeD** | single-cell crops (+966 cluster FOVs) | 4,049 cells | 5 morphology | ✅ | **Train 80% + val 10% + in-dist test 10%** |
| **Herlev** | single-cell crops | 917 cells | 7 WHO dysplasia | ✅ | External eval |
| **CRIC** | conventional FOV, point-labeled | 11,534 cells / 400 FOV | 6 Bethesda | ✅ (points) | External eval |
| **APCData** | LBC FOV, box+point | 3,619 cells / 425 FOV | 6 Bethesda | ✅ (points) | External eval |
| **RIVA** | FOV mini-patches, point, multi-annotator | 15,949 cells | 8 Bethesda | ✅ (points) | External eval |
| **APACC** | auto-seg single-cell crops | 103,675 cells | 4 coarse | ✅ | External eval (binary) + SSL |
| **HiCervix** | single-cell crops | 40,229 cells | 29-leaf Bethesda | ✅ | SSL + optional Bethesda eval |
| **Mendeley LBC** | FOV (image-level) | 963 img | 4 Bethesda | ❌ image-level | Eval (image-level only) |
| **BMT** | multicellular ThinPrep FOV | 600 img / 180 pts | 3 Bethesda | ❌ image-level | Eval (image-level only) |

*Out of scope:* segmentation/detection sets (Cx22/CCEDD/ISBI/CCAgT), histology WSI (GigaScience, TCGA-CESC), private WSI (CSD, UniCAS).

---

## 1b. Cross-dataset compatibility (species · stain · preparation · data type)
*Verified per dataset, June 2026. Answers: "will these be compatible if we train on SIPaKMeD?"*

**Species — all human, no animal.** Every dataset is **human** cervical cells (Greece, Denmark, Brazil, India, Uruguay, Argentina, Hungary, China, USA). **No animal-derived data exists** in this space (cervical screening is inherently human). Only **ISBI-2014** is *mostly synthetic* (computer-generated) imagery; everything else is real human microscopy.

**"Textual"/non-image — only one true case.** The **only** purely non-image dataset is **UCI Risk Factors** (tabular questionnaire — out of scope for an image model). The **SIPaKMeD (26/cell) and Herlev (20/cell) handcrafted-feature tables** are *numeric companions* to image datasets — feed the model the image crops, not the CSVs (see [MULTIMODAL_PLAN.md](MULTIMODAL_PLAN.md)). Every other dataset flagged "textual" is really an **image** dataset whose text is just annotation/label metadata (point coordinates, YOLO boxes, class CSVs).

**The axes that decide compatibility:**
- **Stain (most load-bearing):** Papanicolaou matches SIPaKMeD → compatible. **H&E (Mendeley-LBC, histology WSI) and AgNOR (CCAgT) are different stains → incompatible** without stain normalization/retraining.
- **Preparation:** conventional smear (= SIPaKMeD) vs **LBC/ThinPrep** → a real but tractable domain shift even when the stain matches.
- **Granularity:** single-cell crop (= SIPaKMeD) vs multi-cell FOV vs WSI vs segmentation → FOV/WSI must be **cropped to cells** first.
- **Label taxonomy:** SIPaKMeD-5 vs Herlev-7 vs Bethesda-6/8 vs HiCervix-29 → **remap** (collapse to binary normal/abnormal is the safest common ground; see §3).

> ⚠️ **Corrections from this audit:** (1) **SIPaKMeD is Papanicolaou-stained** (the cyanophilic/orangeophilic palette H&E can't produce) — a secondary-source "H&E" claim is wrong. (2) **Mendeley-LBC is H&E, not Pap** → reclassify it as *incompatible-without-stain-normalization* (it had been listed as an image-level eval set).

### Compatibility matrix
| Dataset | Species | Stain | Preparation | Data type | Verdict |
|---|---|---|---|---|---|
| **SIPaKMeD** (anchor) | Human | Papanicolaou | Conventional smear | single-cell crops | training distribution |
| **Herlev** | Human | Papanicolaou | Conventional smear | single-cell crops | ✅ compatible (domain shift; map 7→5/binary) |
| **APACC** | Human | Papanicolaou | Conventional smear | single-cell crops (+FOV, 34 WSIs) | ✅ compatible (use crop subset; map 4→binary) |
| **HiCervix** | Human | Papanicolaou | LBC (likely) | single-cell crops | ✅ compatible (29-class → crosswalk) |
| **CRIC** | Human | Papanicolaou | Conventional smear | multi-cell FOV + points | 🔧 needs cropping (6 Bethesda → remap) |
| **APCData** | Human | Papanicolaou | LBC | multi-cell FOV + boxes | 🔧 needs cropping (LBC shift) |
| **RIVA** | Human | Papanicolaou | Conventional smear | FOV mini-patches + points | 🔧 needs cropping (8-class → remap) |
| **BMT** | Human | Papanicolaou | ThinPrep/LBC | multi-cell FOV (image-level) | 🔧 needs cropping (3 grades; ThinPrep shift) |
| **Cervix93** | Human | Papanicolaou | ThinPrep | EDF/z-stack FOV | 🔧 needs cropping (grade labels) |
| **Cx22 / CCEDD / ISBI** | Human (ISBI-14 synthetic) | Papanicolaou | LBC/mixed | segmentation masks | 🔧 crop + relabel (no class labels) |
| **Mendeley-LBC** | Human | **H&E** | LBC/SurePath | multi-cell FOV | ❌ incompatible-stain |
| **CCAgT** | Human | **AgNOR** | — | segmentation masks | ❌ incompatible-stain |
| **GigaScience / TCGA-CESC** | Human | **H&E** | FFPE tissue | gigapixel WSI | ❌ incompatible-tissue/modality (histology) |
| **UCI Risk Factors** | Human | none | — | tabular | ❌ non-image |
| **SIPaKMeD / Herlev feature tables** | Human | Papanicolaou | — | numeric-tabular | ❌ non-image (use the matching crops) |

### Usage tiers — every dataset is kept, each with a defined role
- **Tier 1 — direct external test sets (single-cell Pap crops; only label-map + ordinary domain shift):** **Herlev, APACC, HiCervix.**
- **Tier 2 — external test after a crop+relabel pipeline (Pap stain, but FOV/WSI/seg):** CRIC, APCData, RIVA, BMT, Cervix93, Cx22/CCEDD/ISBI.
- **Tier 3 — kept, with a dedicated usage path** (not a direct 5-class Pap cross-dataset test, but each has a concrete role in the project):
  - **SIPaKMeD / Herlev numeric feature tables** → **multimodal fusion** with the image model (late/early fusion → regularization + SHAP "classified X because N/C ratio…" explainability). See [MULTIMODAL_PLAN.md](MULTIMODAL_PLAN.md).
  - **UCI Risk Factors (tabular)** → **separate Tier-1 risk model + decision-funnel cascade** (risk triage → cytology diagnosis); never per-cell fused (patient-level, no shared cohort).
  - **Mendeley-LBC (H&E)** → external probe **only after H&E→Pap stain normalization / color transfer**; report as a higher-risk *secondary* generalization test, with the stain-normalized vs raw gap.
  - **CCAgT (AgNOR)** → **stain-robustness / open-set OOD stress test** (AgNOR's silver-dot palette can't be color-normalized into Pap, and its labels are NOR segmentation) — or a standalone AgNOR task; not a labeled Pap eval.
  - **Histology WSI (GigaScience S-BIAD1168, TCGA-CESC, H&E tissue)** → a **separate histopathology track** or **OOD-rejection negatives** for the cytology model; not a cytology-cell cross-dataset test (different tissue + modality).

**Three honesty caveats for the paper:** (1) almost no dataset shares SIPaKMeD's 5-morphology taxonomy, so report cross-dataset numbers under a **pre-registered label map (binary normal/abnormal recommended)**, not spurious 5-class accuracy; (2) **preparation (conventional vs LBC/ThinPrep) is a second-order domain shift** even when the stain is Papanicolaou — don't attribute the whole accuracy drop to scanner/lab differences; (3) **keeping the Tier-3 sets gives each a defined role but does not make AgNOR/H&E/histology valid as *direct* Pap-cytology cross-dataset tests** — their results must be reported under their stated role (fusion / cascade / stain-normalized probe / OOD), not as standard external accuracy.

---

## 2. SIPaKMeD split (80% train / 10% validation / 10% in-dist test, leakage-safe) + external test
- **Group unit = source cluster image** (966 clusters, ~4.2 cells each). The 4,049 crops trace to these; **rebuild the cell→cluster map from the 5 per-class `.dat`/`.mat` feature tables** (`image number` + `cell number` fields) — the cropped filenames don't encode it. **Mandatory before splitting.**
- **Split:** a **group-aware 80 / 10 / 10 train / validation / in-distribution-test** split, **stratified by class, grouped by cluster id** (`StratifiedGroupKFold`) — **no cluster shared across the three partitions**.
- **Roles:** *train* = fit weights; *validation* = **hyperparameter tuning, early stopping, decision-threshold + calibration-temperature fitting**; *test* = **sealed** until the very end → the unbiased in-distribution number.
- **External / cross-dataset** (Herlev / APACC / HiCervix; §3) is the *additional, stronger* test; the **headline metric is internal−external Δ**.
- **Protocol:** grouped k-fold CV on **train+val** for development / CIs (report mean ± std), then **one frozen retrain** → evaluate **once** on the sealed test + the external sets.
- ⚠️ This replaces the notebook's random `validation_split=0.2` (which leaks same-cluster cells and inflates the 97.11%).
- 🔒 **Never touch the 10% test during tuning** (see Hyperparameter Optimization in the upgrade plan).

---

## 3. Dataset mappings — SIPaKMeD ⇄ other datasets (for evaluation)

The model emits 1 of 5 SIPaKMeD classes. To score on a dataset that uses a *different* scheme, collapse **both sides** to a shared level. Two scorable tiers:

### 3a. Model prediction → evaluation tiers
| SIPaKMeD prediction | → **E0 Binary** | → **E1 Grade** |
|---|---|---|
| Superficial-Intermediate | Normal | Normal |
| Parabasal | Normal | Normal |
| Metaplastic | Normal | Normal |
| Koilocytotic | Abnormal | Low-grade |
| Dyskeratotic | Abnormal | Low-grade* |

\* The model has **no High-grade output** (SIPaKMeD has no HSIL class). Dyskeratotic defaults to Low-grade. → at E1 the model can only ever predict {Normal, Low-grade}; **High-grade recall is 0 by construction** — reported as a *structural blind spot*, not a tuning failure.

### 3b. Target native label → evaluation tiers (ground truth)
| Dataset | Native class | → E0 | → E1 | 
|---|---|---|---|
| **Herlev** | Superficial / Intermediate squamous | Normal | Normal |
| | Columnar (glandular) | **EXCLUDE** | EXCLUDE |
| | Mild dysplasia (CIN1) | Abnormal | Low-grade |
| | Moderate / Severe / Carcinoma-in-situ | Abnormal | High-grade |
| **CRIC / APCData / RIVA** | NILM | Normal | Normal |
| | ASC-US | Abnormal | EXCLUDE (indeterminate) |
| | LSIL | Abnormal | Low-grade |
| | ASC-H / HSIL / SCC | Abnormal | High-grade |
| **RIVA only** | INFL (inflammatory) | Normal | Normal |
| | ENDO (glandular) | **EXCLUDE** | EXCLUDE |
| **APACC** | Healthy | Normal | Normal |
| | Unhealthy | Abnormal | EXCLUDE (no grade) |
| | Rubbish / Bothcells | **EXCLUDE** | EXCLUDE |
| **Mendeley LBC / BMT** *(image-level)* | NILM | Normal | Normal |
| | LSIL | Abnormal | Low-grade |
| | HSIL / SCC | Abnormal | High-grade |

### 3c. How accuracy is computed
For each target dataset:
1. **Filter out** `EXCLUDE`/no-map cells (glandular, indeterminate, debris); **report the excluded fraction** (an honesty metric).
2. Collapse predictions and labels to the chosen tier; compute:
   - **E0 (primary):** accuracy, **balanced accuracy**, macro-F1, **sensitivity (abnormal recall)**, **specificity (normal recall)**, and **AUROC** using `P(abnormal)=softmax(Koilocytotic)+softmax(Dyskeratotic)`, at the pre-specified abnormal threshold.
   - **E1 (diagnostic):** confusion matrix; **High-grade recall reported and expected ≈ 0** (quantifies the blind spot); meaningful performance read on the {Normal, Low-grade} subset.
   - **Optional E-morph (candidate-set hit):** prediction counts correct if it lies in the native label's candidate set (NILM correct if pred ∈ {SI, Parabasal, Metaplastic}; LSIL correct if pred ∈ {Koilocytotic, Dyskeratotic}).
3. Report the **internal-vs-external gap** (SIPaKMeD test E0 vs each external E0) as the headline generalization number.
4. Image-level datasets (BMT, Mendeley) are scored at **image level** by aggregating cell predictions (e.g., "abnormal if any cell ≥ threshold"), or excluded from per-cell scoring.

> **Direction note:** evaluation uses safe **many-to-one** collapses (Superficial+Intermediate→Normal, Moderate+Severe+CIS→High-grade). The dangerous **one-to-many** cases (NILM→3 normal classes, LSIL→{Koilo,Dysk}) never bite at evaluation because we score at the collapsed level, not the 5-class level — that's exactly why cross-dataset scoring is done at E0/E1, not E5.

---

## 4. Data preprocessing (all datasets)
1. **Decode → RGB** (handle BMP/PNG/JPG; SIPaKMeD is BMP).
2. **Obtain cell crops:**
   - Already-cropped (SIPaKMeD, Herlev, HiCervix): use as-is.
   - Point/box-labeled FOV (CRIC, RIVA, APCData): **extract a fixed physical-size patch around each nucleus point** (or use the provided box).
   - Image-level FOV (Mendeley, BMT, APACC tiles): run a class-agnostic nucleus detector/segmenter to crop cells (for eval) or score whole-image.
3. **Resolution harmonization (critical for cross-dataset):** crops differ in µm/pixel (Herlev 0.201, CRIC 0.228, RIVA 0.25, APCData ~). Extract a **constant physical field** (e.g. ~40–50 µm around the nucleus) where µm/px is known, so a cell is the same *real* size everywhere; then resize.
4. **Resize to model input** — **224×224** (EfficientNet-B0 native; **fix the current 380×380 mismatch**), **aspect-preserving pad** (no distortion of nuclear shape).
5. **Stain/color normalization:** Macenko or Reinhard to a fixed reference, **fit on the SIPaKMeD train split only** (never on val/test/external). This is the single biggest lever for cross-dataset transfer. (Alternative: leave raw and rely on stain augmentation — compare both in the ablation.)
6. **Channel normalization:** EfficientNet `preprocess_input` (or ImageNet mean/std), applied inside the graph.
7. **Leakage discipline:** split on **original source IDs first**, then preprocess/augment **train only**; exclude any pre-augmented/precomputed-feature copies shipped with datasets.

---

## 5. Data augmentation plan (cytology-appropriate, train-only)
**Geometric (safe — cells have no canonical orientation):**
- Random horizontal **and** vertical flip.
- Random rotation **0–360°** (full, not just ±20°).
- Mild random zoom/scale (±10–15 %) and small translation.

**Photometric / stain (simulate scanner & staining variation — the real domain shift):**
- Mild color jitter (brightness/contrast/saturation; small hue).
- **Stain augmentation** — HED-space jitter or **MixStyle** (strong domain-generalization choice; doubles as the "no explicit stain-norm" arm of the ablation).
- Gaussian blur (focus variation) and light Gaussian noise.

**Imbalance handling (Parabasal & Metaplastic are minority):**
- Class-weighted loss or **focal loss**; minority oversampling.
- MixUp/CutMix **with caution** (can blur cell boundaries / nuclear detail — treat as an ablation arm, not a default).

**Do NOT:** heavy shear/elastic warps or extreme hue shifts — they distort nuclear morphology and break Pap-stain semantics (clinically meaningful signal).

**Eval-time:** no augmentation, except **optional TTA** (average over flips/rotations). Never augment val/test/external.

---

## 6. EDA plan (run before training)
1. **Rebuild & verify SIPaKMeD cell→cluster map** from `.dat`/`.mat`; confirm all 4,049 cells join one of 966 clusters; tabulate cells/cluster (governs split coarseness).
2. **Re-count every per-class folder** vs cited figures (Herlev per-class sums to 928 vs 917; SIPaKMeD Superficial 831-on-disk vs 813-in-paper) — resolve against the actual download.
3. **Class imbalance** per dataset and per eval tier (SCC/ASC-H/Parabasal/Metaplastic are rare) → plan weights/focal/resampling.
4. **Mapping-coverage audit:** for each external dataset, count cells per native class and the **EXCLUDE fraction** (how much is unscorable: glandular, ASC-US, debris). Fail loudly if a class is missing.
5. **Grouping-key audit:** confirm cluster_id (SIPaKMeD), image_id (CRIC), sample-id (RIVA), study-id (APCData), patient-id (BMT) exist before splitting.
6. **Resolution / domain inventory:** record µm/px, scanner, prep (LBC vs conventional), country, stain per dataset → drives stain-norm + harmonization.
7. **Duplicate / near-duplicate detection** (perceptual hashing) within & across datasets (Mendeley 10-FOV-per-slide, SIPaKMeD mirrors).
8. **Label-noise spot check** (APACC auto-segmented crops are noisy; sample-review minority classes).
9. **Inter-rater agreement on RIVA's 4-annotator subset** → realistic human ceiling.

---

## 7. Leakage risks & caveats
- **SIPaKMeD same-cluster leakage** → use cluster-grouped splits (re-derive the 97.11% honestly).
- **Patient-disjoint eval is only possible** on APACC/BMT/RIVA; SIPaKMeD is cluster-level, Herlev/Mendeley/HiCervix have no usable IDs — state every "leakage-safe" claim at its true granularity.
- **Structural blind spot:** the SIPaKMeD model **cannot detect HSIL/glandular** — High-grade recall ≈ 0 at E1, and glandular cells are excluded from scoring. State prominently (screening-safety).
- **Excluded fraction matters:** dropping ASC-US/HSIL/glandular *inflates* the binary score — always report the excluded fraction alongside accuracy.
- **Fit normalization/class-weights on train only.**
- **Licenses:** Mendeley-LBC (NC-ND) and RIVA (ND) restrict redistribution of derived crops.
