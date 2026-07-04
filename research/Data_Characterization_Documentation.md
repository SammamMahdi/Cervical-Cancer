# Data-Characterization Analysis — Methodology, Results, and Documentation

*Companion technical documentation for `src/data/eda.py`, `src/data/clustering.py`,
`src/data/splits.py`, and `src/data/labelmap.py`. This document records the methodology, the measured
results, their interpretation, the artefact catalogue, and reproducibility details for the all-dataset
data characterization that underpins the new pipeline. All numerical results reported here are
**[measured]** — produced by running the EDA on the on-disk data (`results/metrics/eda/`,
`results/metrics/eda_summary.json`) — and are reproducible from the persisted artefacts. No value has
been fabricated. Where a fact cannot be measured from the local data (e.g. physical resolution in µm/px),
it is explicitly tagged **[unverified]** and never invented.*

---

## 1. Purpose

Before any model is trained, the framework characterizes **what data it is learning from and what data it
is evaluated on**. This serves four goals: (i) quantify the class balance and size/scanner heterogeneity
of every dataset; (ii) establish the **leakage-safe grouping** that makes the in-distribution benchmark
honest (RQ1); (iii) make the cross-taxonomy evaluation **safe and auditable** through a deterministic
many-to-one crosswalk with a reported *excluded fraction* (RQ4); and (iv) surface the structural data
properties — imbalance, scale heterogeneity, and the high-grade blind spot — that later explain the
model's behaviour. It is the data-level companion to the morphometric class characterization
(`Class_Analysis_Documentation.md`).

## 2. Data substrate

Five datasets are profiled by a **single, eval-consistent enumeration** that reuses the evaluation-path
label maps (`data/labelmap.py`), so the EDA counts match exactly what the model is later evaluated on.

| Dataset | Role | Native taxonomy | N (images) | Format |
|---|---|---|---|---|
| **SIPaKMeD** | train + sealed test source | 5-class morphology | **4,049** cropped cells | `.bmp` |
| **Herlev** | external (E0/E1) | 7-class dysplasia | **917** | `.bmp` |
| **APACC** | external (E0) | 2-class healthy/unhealthy | **1,125** | `.png` |
| **HiCervix** | external (E0/E1) | hierarchical Bethesda (level-1) | **8,051** | `.jpg` |
| **CRIC** | external (E0) | 6-class Bethesda, whole-FOV | **11,534** | `.png` |

**[measured]** Total = 25,676 images profiled. SIPaKMeD is the *only* training/test source; the other four
are held entirely out-of-distribution and used only for cross-dataset evaluation.

## 3. Methodology

### 3.1 Eval-consistent enumeration (`data/eda.py`)
Each dataset has a dedicated enumerator (`ENUMERATORS`) returning `(path, native_class, e0_label)`
triples. SIPaKMeD is enumerated through the same `clustering.load_cells()` used by the trainer; the
external sets are enumerated through the same folder/CSV walkers and `labelmap.py` mappers used at
evaluation time. This guarantees EDA counts are not a separate, drifting bookkeeping path.

### 3.2 Integrity guard (`_safe_open`)
Every sampled image is opened with an explicit guard against the legacy failure modes — **0-byte files,
truncated/unreadable images** — returning `None` on failure and counting it as `corrupt_or_zero`. This is
the data-level descendant of the legacy pipeline's silent-corruption bug.

### 3.3 Size statistics (`_size_stats`, sampled)
For each dataset, up to **400 images are sampled deterministically** (seeded `random.Random(C.SEED)`) and
their width/height min/max/mean/median and file-format counts recorded. *Sampling*, not exhaustive
measurement, is used for speed — see Limitations §8.

### 3.4 Leakage-safe clustering (`data/clustering.py`)
SIPaKMeD ships, per class, field-of-view (FOV) images and a `CROPPED/` folder of single cells whose
filename encodes the source FOV (`<FOV>_<cell>.bmp`). Each of the 4,049 cells is therefore mapped to its
**source FOV cluster** (`<Class>/<FOV>`) with no boundary-table parsing. The FOV cluster is the
**grouping unit**: cells from one FOV are highly correlated (same patient/field/illumination), so a
naïve per-cell split would leak. **[measured]** 4,049 cells resolve to **966 FOV clusters**.

### 3.5 Stratified group split (`data/splits.py`)
An 80/10/10 train/val/test split is built **group-aware** (StratifiedGroupKFold over FOV clusters), so a
cluster is wholly within one fold. The split is asserted to have **zero cluster overlap** between folds,
and class balance is preserved per fold. The 10% test split is **sealed** — never touched during training
or early stopping.

### 3.6 Deterministic E0/E1/E-morph crosswalk (`data/labelmap.py`)
Both the model's 5-class prediction and each external dataset's native labels collapse to a shared,
**many-to-one** (never one-to-many) level:

- **E0 (binary):** `normal` / `abnormal` — the primary cross-dataset metric.
- **E1 (grade):** `normal` / `low` / `high` — diagnostic only; **the model has no `high` output by
  design**, so high-grade recall is structurally 0 (an honestly reported blind spot, not a silent gap).
- **E-morph:** lenient candidate-set hit — a 5-class prediction is correct if it lies in the native
  label's morphology candidate set.

Native classes with **no safe home** in the SIPaKMeD morphology space (glandular/columnar, ASC-US
indeterminate, microbe/debris) map to **EXCLUDE** and are reported as the **excluded fraction** — an
honesty metric — never silently dropped. The crosswalk is keyword-based and **versioned**
(`LABELMAP_VERSION = "v1-2026-06"`).

## 4. Measured results

### 4.1 Native class counts (per dataset) **[measured]**

- **SIPaKMeD (4,049):** Superficial-Intermediate 831 · Koilocytotic 825 · Dyskeratotic 813 ·
  Metaplastic 793 · Parabasal 787. **Near-perfectly balanced** (max/min ratio 1.06).
- **Herlev (917):** severe_dysplastic 197 · light_dysplastic 182 · carcinoma_in_situ 150 ·
  moderate_dysplastic 146 · normal_columnar 98 · normal_superficiel 74 · normal_intermediate 70.
- **APACC (1,125):** healthy 576 · unhealthy 549.
- **HiCervix (8,051, level-1):** negative 2,996 · ASC 1,827 · AGC 1,639 · microbe 1,589.
- **CRIC (11,534):** Negative-for-intraepithelial-lesion 6,779 · HSIL 1,703 · LSIL 1,360 · ASC-H 925 ·
  ASC-US 606 · SCC 161.

### 4.2 E0 coverage + excluded fraction **[measured]**

| Dataset | E0 normal | E0 abnormal | excluded | **excluded fraction** | what is excluded |
|---|---:|---:|---:|---:|---|
| SIPaKMeD | 2,411 | 1,638 | 0 | **0.000** | — |
| Herlev | 144 | 675 | 98 | **0.107** | columnar (glandular) |
| APACC | 576 | 549 | 0 | **0.000** | — |
| HiCervix | 2,996 | 1,827 | 3,228 | **0.401** | AGC (1,639) + microbe (1,589) |
| CRIC | 6,779 | 4,755 | 0 | **0.000** | — (ASC-US→abnormal at E0) |

**HiCervix loses 40.1% of cells** to the E0 crosswalk (glandular + microbe have no squamous-morphology
home); this is the single largest exclusion and a headline honesty figure. Herlev loses 10.7% (glandular
columnar cells).

### 4.3 Image size / scanner heterogeneity (sampled n=400 each; 0 corrupt) **[measured]**

| Dataset | width mean / median | height mean / median | width range | format |
|---|---|---|---|---|
| SIPaKMeD | 217.6 / **196** | 188.9 / **163** | 71–513 | bmp |
| Herlev | 152.7 / **121** | 139.7 / **109** | 42–768 | bmp |
| APACC | 326.3 / **327** | 339.1 / **337** | 130–645 | png |
| HiCervix | 730.9 / **585** | 678.6 / **538** | 97–3,220 | jpg |
| CRIC | 1376.0 / **1376** | 1020.0 / **1020** | fixed 1376×1020 | png |

**Corrupt/zero-byte = 0** across all five datasets (within the 400-image sample). CRIC is a single fixed
**whole-field** resolution (1376×1020) with annotated nucleus points — *not* pre-cropped single cells,
unlike the other four.

### 4.4 SIPaKMeD cluster structure **[measured]**

- 4,049 cells → **966 FOV clusters**; cells per cluster: min **1**, max **26**, mean **4.19**.
- Clusters per class: Metaplastic 271 · Koilocytotic 238 · Dyskeratotic 223 · Superficial-Intermediate
  126 · Parabasal 108.
- **Derived cells-per-cluster by class:** Parabasal **7.3**, Superficial-Intermediate **6.6**,
  Dyskeratotic 3.6, Koilocytotic 3.5, Metaplastic 2.9. The *normal* classes pack far more cells into each
  FOV → they carry the highest within-cluster correlation and would be the worst leakage offenders under a
  naïve per-cell split. This is exactly why FOV grouping is necessary.

### 4.5 The sealed leakage-safe split **[measured]** (`eda_summary.json`)

| Fold | cells | clusters | SI | Para | Meta | Koilo | Dysk |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 3,238 | 773 | 665 | 629 | 634 | 660 | 650 |
| val | 405 | 96 | 82 | 79 | 79 | 83 | 82 |
| test | 406 | 97 | 84 | 79 | 80 | 82 | 81 |

**Cluster overlap = 0** for all fold pairs (train∩val, train∩test, val∩test). Class balance is preserved
across folds (every class is ~16–21% of every fold). The split is reproducible from the seed and the
persisted cluster map.

## 5. Interpretation — key data-level findings

1. **SIPaKMeD is balanced; the real world is not.** The training source is near-uniform (1.06 imbalance),
   but every external set is skewed and, more importantly, drawn from a *different label universe*. Models
   that overfit the balanced 5-class prior will be miscalibrated on the skewed external priors — this is
   the data-level seed of the cross-dataset calibration failures.

2. **A ~11× linear scale span across datasets.** Median cell/field linear size ranges from Herlev ~121 px
   to CRIC 1,376 px (SIPaKMeD ~196 px). Because area features scale with the *square* of linear size, the
   apparent-size span is ~100×. This is the data-level substrate of the **scale confound** documented in
   the class analysis (`corr(cytoplasm_area, crop_pixel_area)=+0.988`): a model that keys on apparent cell
   size has learned a cue that does not transfer across these scales. **Caveat [unverified]:** physical
   resolution (µm/px) is unknown per dataset, so apparent pixel size is *not* a calibrated physical size —
   which is precisely why the confound cannot be corrected, only mitigated. See
   `Class_Analysis_Documentation.md` and the scale-confound investigation.

3. **The high-grade blind spot is structural and disclosed.** SIPaKMeD has no HSIL/high-grade morphology
   class, so E1 can never output `high` and high-grade recall is 0 *by construction*. Rather than hide
   this, the crosswalk maps high-grade native labels (Herlev moderate/severe/CIS; HiCervix HSIL/ASC-H/SCC)
   to `high` at E1 and reports them; at E0 they collapse to `abnormal` (detectable, not gradable).

4. **The excluded fraction is a first-class honesty metric.** HiCervix's 40.1% and Herlev's 10.7%
   exclusions are reported, not dropped. They quantify how much of each external set is simply
   *unrepresentable* in the SIPaKMeD morphology space (glandular disease, microbes, debris) — a fairer
   framing than reporting accuracy over a silently filtered subset.

5. **CRIC is categorically different input.** It is whole-FOV (1376×1020) with nucleus points, not
   pre-cropped cells. Any single-cell model must crop CRIC, making the crop geometry a confound — the
   data-level origin of CRIC's degenerate E0 behaviour (see the CRIC crop-geometry / crop-sweep analyses).

6. **Format/scanner heterogeneity is real.** Four file formats (bmp/png/jpg) and five acquisition
   pipelines; JPEG compression (HiCervix) and whole-slide scanning (CRIC) differ from the bitmap
   single-cell crops (SIPaKMeD/Herlev). This motivates stain normalization and scale/photometric
   augmentation as domain-generalization tools (RQ2).

## 6. Artefact catalogue

| Artefact | Path | Content |
|---|---|---|
| Combined EDA | `results/metrics/eda/eda_all.json` | per-dataset counts, E0 coverage, size stats, cluster summary |
| Overview table | `results/metrics/eda/eda_overview.csv` | one row per dataset (n, classes, E0, excluded frac, sizes) |
| Per-dataset | `results/metrics/eda/{sipakmed,herlev,apacc,hicervix,cric}.json` (+ `_classes.csv`) | full record per dataset |
| Split + clusters + labelmap | `results/metrics/eda_summary.json` | cluster summary, sealed split, label-map manifest |
| Class-balance figures | `results/figures/eda/<ds>_classes.{pdf,png,csv}` | native class bars |
| E0-coverage figures | `results/figures/eda/<ds>_e0coverage.{pdf,png,csv}` | normal/abnormal/excluded bars |
| Sample montages | `results/figures/eda/<ds>_montage.{pdf,png}` | a few cells per native class |
| Top-level balance | `results/figures/eda_class_balance.*`, `eda_cluster_balance.*` | SIPaKMeD class / cluster balance |

## 7. Reproducibility

```bash
# regenerate every number and figure in this document
python -m data.eda            # or: run.eda()  (also writes research framing + split)
```

The EDA is deterministic given `C.SEED` (size-stat sampling and montage selection are seeded). Counts are
exhaustive (every file enumerated); only the width/height statistics are over a fixed 400-image sample.
All figures carry the house-style provenance footer (generated-at timestamp). The label crosswalk is
versioned (`v1-2026-06`) and persisted in `eda_summary.json["labelmap"]`.

## 8. Limitations & provenance

- **[unverified] Physical resolution (µm/px) is unknown** for every dataset, so cross-dataset size
  comparisons are in *pixels*, not microns. The apparent-scale span (§5.2) conflates physical cell size
  with imaging magnification and cannot be disentangled from local data alone.
- **[measured, sampled] Size statistics use a 400-image sample** per dataset, not the full set; means/medians
  are estimates (the integrity check is likewise over the sample, so a corrupt file outside the sample
  would be missed at EDA time — training-time loaders still guard every file).
- **[primary] The E0/E1 crosswalk is keyword-based.** It is deterministic and versioned, but a native
  label whose folder/string wording is unanticipated would fall through to EXCLUDE. The excluded fraction
  makes such fall-through visible rather than silent.
- **High-grade recall is 0 by construction** (no HSIL morphology class); this is a disclosed design
  limitation of the SIPaKMeD label space, not a measurement error.
- **HiCervix counts are from `test.csv` level-1 labels** (the evaluation split shipped with the dataset);
  the 8,051 figure is the enumerated test partition, of which 4,823 survive the E0 crosswalk.

---

*Cross-references: `Class_Analysis_Documentation.md` (morphometric class characterization on the same
SIPaKMeD substrate), Volume II "Dataset Engineering" §§ on the split and crosswalk, and the scale-confound
investigation. Every number in §4 is regenerable from `results/metrics/eda/`.*
