# A Trustworthy, Generalizable Framework for Deep-Learning Cervical Cytology
## Volume II — Dataset Engineering & Experimental Foundation

*A research monograph (thesis-format supplementary volume). This volume establishes the scientific
methodology governing dataset engineering, curation, harmonization, validation, and experimental design.
It contains no model architecture, loss, optimisation, training, distillation, calibration,
explainability, or out-of-distribution implementation; those belong to later volumes.*

---

### Editorial note on evidence, provenance, and the no-fabrication policy

This volume adheres to a strict no-fabrication policy: **no dataset, statistic, split, license,
annotation procedure, taxonomy, metadata field, citation, microscope specification, imaging protocol,
or demographic figure has been invented.** Because the writing environment has **no live access to the
primary literature or dataset portals**, the volume cannot independently confirm published dataset
metadata (e.g., exact microscope models, objective magnifications, license terms, or patient
demographics). In accordance with the governing policy — *"explicitly identify missing information
rather than inferring it"* — every dataset fact is tagged with one of three provenance levels:

- **[measured]** — directly measured from the on-disk dataset in this project, with the measurement
  reproducible from the persisted exploratory-data-analysis artefacts (`results/metrics/eda/*.json|csv`,
  `results/metrics/cric_crop_geometry.json`, and the project's class-characterisation outputs). These are
  ground-truth facts established by this work.
- **[primary]** — attributable to a dataset's primary publication where the specific claim is held with
  confidence; the citation is given.
- **[unverified]** — a field that could **not** be confirmed in this environment. Such fields are marked
  explicitly and are **not** populated with inferred or "commonly reported" values. They denote work
  required before final submission, not gaps to be filled by assumption.

A reader should treat **[measured]** facts as the rigorous, reproducible core of the volume, **[primary]**
facts as citation-backed, and **[unverified]** fields as honest declarations of missing evidence.

The fixed inputs — research gaps G1–G7, questions RQ1–RQ7, hypotheses H1–H7, contributions C1–C6, and the
evaluation protocol — are those defined in Volume I (Chapter 4) and are referred to here by identifier.

---

# Chapter 8 — Global Dataset Landscape

## 8.1 Scope and method of the dataset review

This chapter surveys the publicly available cervical-cytology image datasets relevant to single-cell and
field-of-view classification, with emphasis on the five collections that this work uses directly —
**SIPaKMeD**, **Herlev**, **CRIC**, **HiCervix**, and the on-disk **APACC** subset — and a critically
appraised account of the broader landscape. The review is *data-centric*: for the five used datasets, the
quantitative profile is established by **direct measurement** of the on-disk material (counts, class
distributions, annotation availability, image statistics), so that the comparison is grounded in
reproducible fact rather than in re-reported figures. For datasets not held in this project, and for
metadata fields not determinable from the on-disk material (acquisition hardware, magnification, licenses,
demographics), the chapter reports **[unverified]** rather than inferring values — a deliberate
methodological choice dictated by the no-fabrication policy and by the absence of live literature access
in the writing environment (§ editorial note).

The chapter is organised as a per-dataset profile (Sections 8.2–8.6 for the used datasets; Section 8.7 for
the wider landscape), followed by consolidated comparison tables (Section 8.8) and a critical comparison
that motivates the selection methodology of Chapter 9 (Section 8.9).

## 8.2 SIPaKMeD

**Role in this work.** SIPaKMeD is the **training source**: the model is trained and internally evaluated
on SIPaKMeD, and all external datasets are used frozen for cross-domain evaluation.

**Provenance and publication.** SIPaKMeD was introduced by Plissiti et al. (2018) at IEEE ICIP **[primary]**.
The original publication presents it as a dataset of isolated and clustered cervical cells from Pap-stained
smear images for feature- and image-based classification.

**Measured composition [measured].** The on-disk SIPaKMeD comprises **4,049 cropped single-cell images**
distributed across five morphological classes, together with **966 cluster (field-of-view) images** from
which the single cells were extracted, and **16,196 `.dat` contour files** encoding per-cell nucleus and
cytoplasm boundaries (approximately four contour files per cell). The five classes are
**Superficial–Intermediate** and **Parabasal** (normal squamous), **Metaplastic** (benign), and
**Koilocytotic** and **Dyskeratotic** (abnormal). The class composition is mild-to-moderately balanced;
this work's exploratory analysis records per-class cropped-cell counts in the range of roughly 800–830 per
class (an imbalance ratio near 1.2) **[measured; `results/metrics/eda/sipakmed.json`]**. Images are stored
in BMP format; measured single-cell crop dimensions are heterogeneous, with a mean on the order of
~217×189 pixels **[measured]**.

**Annotation methodology and quality.** Each cropped cell carries expert-drawn nucleus and cytoplasm
boundary annotations (the `.dat` contours) **[measured]**, which this work exploits for ground-truth
morphometric characterisation (Volume II, Chapter 10) and for explanation-faithfulness evaluation (later
volumes). The taxonomy is a taxonomy of **cell appearance**, not of cytological grade — a semantic
property with direct consequences for cross-dataset label alignment (Chapter 11). The precise annotation
protocol (number of annotators, adjudication procedure) as reported in the primary publication is
**[unverified]** here.

**Acquisition metadata.** Staining is Papanicolaou (conventional smear) **[primary, general]**. The
microscope model, camera, objective magnification, pixel size (µm/px), and slide-preparation laboratory
are **[unverified]** in this environment and must be confirmed against Plissiti et al. (2018) before
final submission. Patient demographics are **[unverified]**.

**Licensing and accessibility.** SIPaKMeD is publicly distributed for research **[primary, general]**; the
exact license text and redistribution terms are **[unverified]** here.

**Clinical relevance and limitations.** As a curated single-cell appearance dataset, SIPaKMeD is well
suited to controlled morphological classification but, as a *curation*, under-represents the borderline,
overlapping, and artefact-laden cells of consecutive clinical material (Volume I, Section 2.11), and its
five-class appearance space cannot represent high-grade cytological grade directly (Volume I, Section 2.4).
Its single conventional-smear acquisition pipeline means that, used alone, it cannot support cross-domain
claims — the precise motivation for the external datasets below.

## 8.3 Herlev

**Role in this work.** Herlev is an **external evaluation** dataset (conventional smear, single cells),
used frozen.

**Provenance and publication.** The Herlev Pap-smear dataset was released by Jantzen et al. (2005),
prepared at Herlev University Hospital (Denmark) **[primary]**.

**Measured composition [measured].** The on-disk Herlev comprises **917 single cells** across **seven
classes**, each cell accompanied by a per-cell segmentation mask (so that the raw file count is 1,834 = 917
cell images + 917 mask images). The measured per-class **cell** counts are: normal superficiel 74, normal
intermediate 70, normal columnar 98, light dysplastic 182, moderate dysplastic 146, severe dysplastic 197,
and carcinoma in situ 150 **[measured; from the on-disk folders, excluding the `-d` mask files]**. The
taxonomy spans normal (superficial, intermediate, columnar) and dysplastic grades (mild/moderate/severe)
and carcinoma in situ — i.e., a **dysplasia-grade** taxonomy aligned with the histological CIN spectrum,
in contrast to SIPaKMeD's appearance taxonomy.

**Annotation methodology and quality.** Each cell carries a segmentation mask and an expert class label
**[measured]**. Herlev is widely noted to contain difficult and borderline cells; the exact annotation and
adjudication protocol as published is **[unverified]** here. The inclusion of a **columnar (glandular)**
normal class — absent from SIPaKMeD's squamous appearance space — is a key source of taxonomic
incompatibility (Chapter 11).

**Acquisition metadata.** Conventional Papanicolaou smear **[primary, general]**. Microscope model, camera,
magnification, pixel size, and demographics are **[unverified]**.

**Licensing and accessibility.** Publicly available for research **[primary, general]**; exact license
terms **[unverified]**.

**Clinical relevance and limitations.** Herlev's grade taxonomy and conventional-smear preparation make it
a valuable *same-preparation, different-taxonomy* external test: it stresses label alignment (grade vs
appearance) while holding preparation modality close to SIPaKMeD's. Its small size (917 cells) limits the
statistical power of per-class external estimates, a limitation this work addresses through confidence
intervals and honest reporting (Chapter 17). The columnar class has no squamous-morphology counterpart and
must be handled by explicit exclusion (Chapter 11).

## 8.4 CRIC

**Role in this work.** CRIC is an **external evaluation** dataset (whole field-of-view images with
point annotations), used frozen.

**Provenance and publication.** The on-disk material corresponds to the **CRIC** cervical-cell collection
(Center for Recognition and Inspection of Cells); the annotation file embeds figshare DOIs
(`10.6084/m9.figshare…`) **[measured from the CSV]**. The precise primary publication, authorship, and
institutional origin are **[unverified]** in this environment and must be confirmed before final
submission.

**Measured composition [measured].** The on-disk CRIC comprises **400 whole field-of-view PNG images** and
a per-cell annotation CSV recording **11,534 classified cells**, each annotated by a **nucleus-centroid
coordinate** and a Bethesda-system class. The measured Bethesda distribution is: Negative (NILM) 6,779;
HSIL 1,703; LSIL 1,360; ASC-H 925; ASC-US 606; and squamous cell carcinoma (SCC) 161 **[measured;
`classes/classifications.csv`]**. CRIC is thus, uniquely among the used datasets, a **point-annotated,
multi-cell field** dataset with a **Bethesda** label space including high-grade and indeterminate
categories.

**Measured acquisition geometry [measured].** Because CRIC provides only nucleus centroids on dense
fields, this work measured the inter-cell spacing directly: the median nearest-neighbour distance between
annotated nuclei is ≈70 px (mean ≈87 px; inter-quartile range ≈44–110 px)
**[measured; `results/metrics/cric_crop_geometry.json`]**. This establishes, as a reproducible fact, that
CRIC is a **dense preparation** in which isolated whole-cell crops at SIPaKMeD's scale are geometrically
impossible — a property with direct consequences for cross-dataset evaluation (Chapters 12, 16).

**Acquisition metadata.** Slide preparation, staining protocol, microscope, magnification, pixel size,
country of origin, and demographics are **[unverified]** here (the figshare provenance suggests a
specific source that must be consulted directly).

**Licensing and accessibility.** Distributed via figshare under terms that are **[unverified]** here and
must be confirmed.

**Clinical relevance and limitations.** CRIC's Bethesda label space and field-level, point-annotated format
make it the most *clinically realistic* external test (consecutive-field material, high-grade categories),
but also the most *off-distribution* relative to a SIPaKMeD-trained single-cell classifier: its labels
include categories the model cannot represent (HSIL, ASC-US/ASC-H glandular-adjacent indeterminates) and
its dense fields require point-cropping that this work shows is a measurable confound (Chapter 12, and the
crop-sensitivity analysis of later volumes).

## 8.5 HiCervix

**Role in this work.** HiCervix is an **external evaluation** dataset (liquid-based cytology, hierarchical
Bethesda labels), used frozen; its scorable squamous subset is evaluated under the binary tier.

**Provenance and publication.** The on-disk material is organised as a **hierarchical-label** cervical
cytology dataset; the project's working bibliography attributes it to a 2024 release (Cai et al.) **[primary,
to be verified]**. The exact publication venue, authorship, and institutional origin are **[unverified]**
here and must be confirmed before final submission.

**Measured composition [measured].** The on-disk HiCervix is by far the largest used dataset, with
**40,229 cell images** partitioned by its creators into **train (28,160)**, **validation (4,018)**, and
**test (8,051)** subsets, each row carrying a three-level hierarchical label (`level_1`, `level_2`,
`level_3`) **[measured; `train.csv`, `val.csv`, `test.csv`]**. The measured `level_1` distribution in the
test split is: negative 2,996; ASC 1,827; AGC 1,639; microbe 1,589 **[measured]**. The presence of an
explicit **microbe** category and a glandular **AGC** category — neither representable in a squamous
appearance space — and the hierarchical organisation are distinctive features.

**Annotation methodology and quality.** Labels are hierarchical (a coarse `level_1` Bethesda-like axis
refined at `level_2`/`level_3`) **[measured]**. The annotation protocol, number of annotators, and
adjudication as published are **[unverified]** here.

**Acquisition metadata.** Liquid-based cytology preparation **[primary, to be verified]**; microscope,
scanner, magnification, pixel size, country, and demographics are **[unverified]**.

**Licensing and accessibility.** Terms **[unverified]** here.

**Clinical relevance and limitations.** HiCervix contributes the crucial **liquid-based** preparation
domain — the modality most distant, on the preparation axis, from SIPaKMeD's conventional smear — and a
large, hierarchically labelled corpus. For a SIPaKMeD-trained squamous classifier, its microbe and AGC
categories are out of scope and must be excluded under the crosswalk (Chapter 11), so that only its
scorable squamous subset (negative vs ASC) enters the binary tier; this work measures the resulting
**excluded fraction** (≈40% of test rows) explicitly as an honesty metric **[measured;
`results/metrics/eda/hicervix.json`]**.

## 8.6 APACC (on-disk subset)

**Role in this work.** The on-disk APACC subset is an **external evaluation** set (auto-segmented
single-cell crops, coarse healthy/unhealthy labels), used frozen.

**Provenance and publication.** The on-disk material is labelled "APACC" with a coarse two-class structure.
**A primary publication, authorship, institutional origin, and dataset definition for "APACC" could not be
verified in this environment.** In accordance with the no-fabrication policy, this work does **not** assert
a provenance for it; the dataset's full documentation is recorded as **[unverified]** and flagged as a
limitation (Chapter 18). Only the directly measured properties of the on-disk subset are reported.

**Measured composition [measured].** The on-disk APACC comprises **1,125 single-cell crop images** within a
**`test/`** split, organised into two folders: **healthy 576** and **unhealthy 549** **[measured]**. No
train or validation split is present on disk; whether the broader dataset contains additional splits is
**[unverified]**. The labels are a coarse binary (healthy/unhealthy) without finer cytological grade.

**Acquisition metadata.** Preparation, staining, microscope, magnification, pixel size, segmentation
procedure, and demographics are all **[unverified]**.

**Licensing and accessibility.** **[unverified]**.

**Clinical relevance and limitations.** The APACC subset contributes an **auto-segmented**, coarse-label
domain useful as a third point on the preparation/segmentation axis, but its **unverified provenance** is a
material limitation: claims derived from it must be appropriately scoped, and its results interpreted with
caution, precisely because its documentation could not be confirmed. This work therefore treats APACC as a
*supporting*, not *load-bearing*, external domain and states this limitation explicitly (Chapter 18).

## 8.7 The wider publicly reported landscape

Beyond the five used datasets, a number of cervical-cytology image resources have been reported in the
literature. **In the absence of live literature access, this section does not attempt an exhaustive,
metadata-complete enumeration**, because doing so would require asserting counts, licenses, and acquisition
details that cannot be verified here; per the no-fabrication policy, such an enumeration is deferred to a
literature-verified revision. The following resources are named because the author is confident of their
existence, with all specific metadata explicitly **[unverified]**:

- **Liquid-based cytology Pap-smear image collections** released on public data repositories (e.g.,
  Mendeley Data) have been reported; counts, classes, and licenses **[unverified]**.
- **Overlapping cervical-cytology segmentation challenge data** (e.g., the ISBI overlapping-cell
  segmentation challenges) provide synthetic and real overlapping-cell images oriented toward segmentation
  rather than single-cell grade classification; specifics **[unverified]**.
- Additional single-cell and whole-slide cervical resources have appeared in 2022–2026; a complete,
  verified census is **explicitly identified as missing** here and is required future work (Chapter 18).

This honest incompleteness is itself a finding: the field lacks a single, authoritative, metadata-complete
registry of cervical-cytology datasets, which complicates reproducible cross-dataset research and motivates
the FAIR/datasheet practices this work adopts for its own data (Chapter 17; Volume I, Section 5.11).

## 8.8 Comparison tables

**Table 8.1 — Measured composition of the used datasets [measured].** All counts are measured from the
on-disk material and reproducible from this work's exploratory-data-analysis artefacts.

| Dataset | Role | Format | Cells (measured) | Native classes (measured) | Annotation (measured) |
|---|---|---|---|---|---|
| SIPaKMeD | Train + internal test | per-cell BMP + cluster FOV | 4,049 crops (+966 FOV) | 5 appearance (SI, Parabasal, Metaplastic, Koilocytotic, Dyskeratotic) | nucleus+cytoplasm contours (16,196 `.dat`) |
| Herlev | External | per-cell BMP + mask | 917 (+917 masks) | 7 grade (3 normal incl. columnar; mild/mod/severe dysplastic; CIS) | per-cell segmentation mask |
| CRIC | External | whole-FOV PNG + CSV | 11,534 in 400 FOV | 6 Bethesda (NILM, ASC-US, ASC-H, LSIL, HSIL, SCC) | nucleus-centroid points |
| HiCervix | External | images + CSV | 40,229 (28,160/4,018/8,051) | hierarchical; level_1 {negative, ASC, AGC, microbe} | hierarchical labels (3 levels) |
| APACC (subset) | External | per-crop image | 1,125 (test only) | 2 coarse (healthy/unhealthy) | folder label only |

**Table 8.2 — Acquisition and governance metadata.** Cells marked **[unverified]** could not be confirmed
in this environment and must be filled from each dataset's primary documentation before submission; they
are **not** populated with inferred values.

| Dataset | Preparation | Staining | Microscope / scanner | Magnification | Pixel size | License | Demographics | Primary source |
|---|---|---|---|---|---|---|---|---|
| SIPaKMeD | Conventional smear [primary] | Papanicolaou [primary] | [unverified] | [unverified] | [unverified] | public-for-research [primary]; terms [unverified] | [unverified] | Plissiti et al. 2018 |
| Herlev | Conventional smear [primary] | Papanicolaou [primary] | [unverified] | [unverified] | [unverified] | public-for-research [primary]; terms [unverified] | [unverified] | Jantzen et al. 2005 |
| CRIC | [unverified] | [unverified] | [unverified] | [unverified] | [unverified] | figshare; terms [unverified] | [unverified] | [unverified] (figshare DOIs) |
| HiCervix | Liquid-based [primary, verify] | [unverified] | [unverified] | [unverified] | [unverified] | [unverified] | [unverified] | Cai et al. 2024 [verify] |
| APACC (subset) | [unverified] | [unverified] | [unverified] | [unverified] | [unverified] | [unverified] | [unverified] | [unverified] |

**Table 8.3 — Domain and taxonomy positioning (interpretation grounded in measured facts).**

| Dataset | Preparation domain | Label semantics | Granularity | Out-of-squamous categories | Distance from SIPaKMeD |
|---|---|---|---|---|---|
| SIPaKMeD | conventional smear | appearance | 5 | — | source (zero) |
| Herlev | conventional smear | grade | 7 | columnar (glandular) | near (same preparation) |
| APACC (subset) | auto-segmented crops | coarse binary | 2 | rubbish/both-cell categories reported in some variants [unverified] | moderate |
| CRIC | dense FOV (smear) | Bethesda | 6 | ASC-US/ASC-H/HSIL/SCC beyond appearance space | far (point-cropped fields) |
| HiCervix | liquid-based | hierarchical Bethesda-like | 3-level | AGC (glandular), microbe | farthest (different preparation) |

## 8.9 Critical comparison

Three critical observations follow from the measured profiles and motivate the selection methodology of
Chapter 9. **First, complementarity by design.** The five used datasets span the dimensions along which a
SIPaKMeD-trained classifier is expected to be stressed: SIPaKMeD fixes the source; Herlev varies *taxonomy*
while holding *preparation* (conventional smear) close; APACC varies *segmentation* (auto-cropped); CRIC
varies *format* (dense point-annotated fields) and introduces *high-grade* Bethesda categories; and
HiCervix varies *preparation* (liquid-based) and introduces *glandular and microbe* categories and a
*hierarchical* label space. This is, in the author's **interpretation**, a deliberately diverse external
suite that operationalises the preparation-distance and taxonomy-incompatibility axes of Volume I.

**Second, the verifiability gradient.** The *quantitative* properties of all five datasets are measured and
reproducible, but their *governance and acquisition* metadata are largely **[unverified]** here, and one
dataset (APACC) lacks confirmable provenance entirely. This gradient is consequential: it bounds how
strongly each dataset can support a claim, and it dictates that APACC be treated as supporting rather than
load-bearing (Section 8.6) and that all metadata cells be confirmed before publication.

**Third, the structural mismatch with the model's label space.** Every external dataset contains native
categories that a five-class squamous-appearance model **cannot represent** — Herlev's columnar, CRIC's
high-grade and indeterminate Bethesda classes, HiCervix's AGC and microbe. This is not a defect to be
hidden but the central object of the label-alignment analysis (Chapter 11): it forces an explicit,
deterministic crosswalk with a reported excluded fraction, and it establishes — as measured fact — that
honest cross-dataset evaluation in this domain is impossible without such a crosswalk.

## 8.10 Chapter summary

This chapter profiled the cervical-cytology dataset landscape with a data-centric, measurement-grounded
method. It established, by direct measurement, the composition, structure, native taxonomy, and annotation
form of the five datasets this work uses — SIPaKMeD (4,049 single cells with contour annotations; the
training source), Herlev (917 graded cells with masks; conventional smear), CRIC (11,534 point-annotated
cells in 400 dense fields; Bethesda), HiCervix (40,229 hierarchically labelled cells; liquid-based), and
the APACC subset (1,125 auto-segmented crops; coarse labels, unverified provenance) — and reported all
acquisition and governance metadata that could not be confirmed in this environment as explicit
**[unverified]** fields rather than inferring them. Consolidated comparison tables and a critical comparison
showed that the five datasets are complementary by design across the preparation-distance and
taxonomy-incompatibility axes, that they exhibit a verifiability gradient bounding their evidential weight,
and that every external dataset is structurally mismatched with the model's squamous appearance space —
the fact that necessitates an explicit label crosswalk.

## 8.11 Remaining challenges

Three challenges carry forward. The **[unverified]** metadata fields — acquisition hardware, magnification,
pixel size, licenses, and demographics for all datasets, and the entire provenance of APACC — must be
confirmed against primary documentation before publication; until then, claims depending on those fields
are appropriately scoped. The **wider-landscape census** is explicitly incomplete and requires a
literature-verified enumeration. And the **structural label mismatch** measured here demands a principled
selection rationale (why these datasets, and not others) and a defensible crosswalk — the subjects of
Chapters 9 and 11 respectively.

## 8.12 Transition to Chapter 9

Having established what datasets exist and what, by measurement, they contain, the volume now turns from
*description* to *justification*. Chapter 9 sets out the dataset-selection methodology: it justifies each
selected dataset by its scientific motivation, its relevance to specific research questions and gaps, its
complementarity within the external suite, and its expected contribution, and it explains explicitly why
candidate datasets that were not selected were excluded — still without describing any implementation.

---

# Chapter 9 — Dataset Selection Methodology

## 9.1 Selection objective and criteria

The selection of datasets is not a convenience decision but a design decision that determines what the
research can and cannot measure. The objective is to assemble a **training source** and an **external
evaluation suite** that, together, make the diagnostic and constructive problems of Volume I (Section
2.19.5) *measurable*: the suite must exercise the preparation-distance axis (to test cross-domain
generalisation, RQ2), the taxonomy-incompatibility axis (to test safe cross-taxonomy evaluation, RQ4), the
operating-point/calibration axis (to test reliability under shift, RQ3), and must contain the
clinically critical high-grade categories the model structurally cannot represent (so that the structural
blind spot is *measured*, not hidden). Six selection criteria follow, each tied to a research question or
gap: (i) **public availability and reproducibility** (G1); (ii) **single-cell or cell-localised format**
compatible with a single-cell classifier; (iii) **preparation diversity** spanning conventional smear,
auto-segmented, dense-field, and liquid-based modalities (G2); (iv) **taxonomic diversity** spanning
appearance, dysplasia-grade, and Bethesda label spaces (G4); (v) **presence of clinically important
categories** including high-grade and glandular (G5); and (vi) **measurability** of the dataset's
properties in this environment, so that selection rests on fact rather than on unverifiable metadata.

## 9.2 SIPaKMeD as the training source

SIPaKMeD is selected as the **sole training source**. Its scientific motivation is fourfold. First, it
provides a curated, **single-cell appearance** taxonomy with sufficient per-class volume (≈800 cells/class,
measured) to train a five-class classifier under a leakage-safe split. Second, and decisively, it supplies
**expert nucleus and cytoplasm contour annotations** (16,196 `.dat` files, measured), which are the
ground-truth substrate for the morphometric characterisation (Chapter 10) and the explanation-faithfulness
evaluation of later volumes — a capability that the grade-labelled or point-annotated alternatives do not
provide at the single-cell level. Third, its **cluster (field-of-view) structure** exposes the latent
grouping variable required for a leakage-safe split (Chapter 16): because the 4,049 cells derive from 966
fields (measured), grouped partitioning is both necessary and possible. Fourth, its conventional-smear
preparation provides a well-defined *source domain* against which preparation distance can be measured.
Its **limitation** as a training source — that an appearance taxonomy cannot represent cytological grade —
is not disqualifying but is the very property the cross-taxonomy analysis is designed to confront, and it
is handled honestly by the crosswalk rather than concealed.

## 9.3 The external evaluation suite

The four external datasets are selected to span, with minimal redundancy, the axes along which a
SIPaKMeD-trained model is expected to be stressed. Their selection is justified individually below and
their complementarity in Section 9.4.

**Herlev (selected).** Herlev is selected as the *near-domain, different-taxonomy* test: it holds
preparation modality close to the source (conventional Papanicolaou smear) while varying the taxonomy from
appearance to **dysplasia grade**, and it introduces a **columnar (glandular)** category absent from the
squamous source space. Its relevance is to RQ4 (cross-taxonomy alignment) and RQ2 (generalisation at small
preparation distance); its expected contribution is to isolate the *taxonomy* effect from the *preparation*
effect by minimising the latter. Its limitation is its small size (917 cells, measured), which this work
mitigates with confidence intervals (Chapter 17).

**APACC subset (selected, with caveat).** The APACC subset is selected as an *auto-segmented, coarse-label*
domain that adds a distinct segmentation/cropping pipeline to the suite. Its relevance is to RQ2
(segmentation-induced shift). Its **expected contribution is deliberately limited** by its **unverified
provenance** (Section 8.6): it is selected as a *supporting* domain whose results are interpreted with
caution and whose provenance is flagged as a threat to validity (Chapter 18). The decision to include it
despite incomplete documentation, rather than to silently drop it, is itself an application of the
transparency principle: its on-disk properties are measurable and reproducible even though its publication
record is not confirmable here.

**CRIC (selected).** CRIC is selected as the *dense-field, Bethesda-labelled, high-grade-containing* test.
Its relevance spans RQ4 (Bethesda↔appearance alignment, including high-grade categories the model cannot
represent) and RQ2 (format/preparation shift). Its expected contribution is to introduce **clinically
critical high-grade and indeterminate categories** (HSIL, ASC-H, ASC-US, SCC; measured) into the external
evaluation, so that the model's structural high-grade blind spot is *measured* via the E1 grade tier and
excluded fraction, and to provide a *geometrically dense* field format (median nearest-neighbour nucleus
distance ≈70 px, measured) that stresses any single-cell-cropping assumption. Its limitation — that
point-cropping dense fields is a measurable confound — is treated explicitly as a sensitivity analysis in
later volumes.

**HiCervix (selected).** HiCervix is selected as the *far-domain, liquid-based, hierarchically labelled*
test — the external dataset most distant on the preparation axis. Its relevance is to RQ2 (the largest
preparation shift, smear→LBC) and RQ4 (hierarchical Bethesda alignment, including glandular AGC and a
microbe category). Its expected contribution is to provide the **liquid-based** domain and a large
(40,229-cell, measured) corpus, and to test the crosswalk's handling of out-of-squamous categories at
scale (measured excluded fraction ≈40% of test). Its limitation is its **partly unverified provenance**
(publication to be confirmed) and its large out-of-scope fraction, both reported honestly.

## 9.4 Complementarity of the suite

The four external datasets are **complementary, not redundant**, along the two axes that the research must
exercise (Table 8.3, measured). On the *preparation-distance* axis they are ordered Herlev (conventional
smear, near) → APACC (auto-segmented, moderate) → CRIC (dense conventional fields, far in format) →
HiCervix (liquid-based, farthest); this ordering is the explanatory variable for the generalisation
analysis (RQ2). On the *taxonomy* axis they range over dysplasia grade (Herlev), coarse binary (APACC), and
Bethesda (CRIC, HiCervix), with progressively larger out-of-squamous category sets (columnar → none →
high-grade/indeterminate → glandular/microbe); this range is the stress test for the crosswalk (RQ4). The
**interpretation** is that the suite is the minimal set that exercises both axes with clinically meaningful
categories, while keeping each dataset's role distinct, so that an observed effect can be attributed to a
specific axis rather than to a confound.

## 9.5 Justification of exclusions

Datasets *not* selected fall into three classes, each excluded for a stated, principled reason rather than
by omission. **(i) Segmentation-oriented resources** (e.g., overlapping-cell segmentation challenge data;
Section 8.7) are excluded because their task and annotation form (instance segmentation of overlapping
cells) do not match a single-cell grade/appearance classification problem, and repurposing them would
require an annotation transformation that introduces label uncertainty. **(ii) Resources whose properties
could not be verified in this environment** are excluded from the *load-bearing* suite because the
no-fabrication policy forbids asserting their counts, taxonomy, or licenses; they are named in the
wider-landscape census (Section 8.7) and reserved for a literature-verified revision rather than included
on unverifiable terms. **(iii) Non-cytology cervical resources** (e.g., colposcopy image collections) are
excluded as out of modality. The single included dataset with incomplete documentation, APACC, is retained
only in a clearly demoted, supporting role with its limitation flagged (Sections 8.6, 9.3, 18) — a
deliberate, transparent exception rather than an inconsistency.

## 9.6 Mapping of datasets to research questions and gaps

**Table 9.1 — Dataset-to-question/gap mapping.** Each dataset's primary evidential role.

| Dataset | Primary role | Principal RQ(s) | Principal gap(s) |
|---|---|---|---|
| SIPaKMeD | Training source; internal sealed-test benchmark | RQ1, RQ6, RQ7 | G1, G6, G7 |
| Herlev | Near-domain, grade-taxonomy external test | RQ4, RQ2 | G4, G2 |
| APACC (subset) | Auto-segmented external test (supporting) | RQ2 | G2 |
| CRIC | Dense-field, Bethesda, high-grade external test | RQ4, RQ2, RQ5 | G4, G2, G5 |
| HiCervix | Liquid-based, hierarchical external test | RQ2, RQ4 | G2, G4 |

## 9.7 Chapter summary

This chapter justified the dataset selection. It defined six selection criteria tied to the research gaps,
selected SIPaKMeD as the training source on the strength of its single-cell appearance taxonomy, contour
annotations, cluster structure, and conventional-smear source domain, and selected Herlev, the APACC
subset, CRIC, and HiCervix as a complementary external suite spanning the preparation-distance and
taxonomy axes with clinically meaningful categories. It justified each dataset by scientific motivation,
relevance to specific questions and gaps, complementarity, and expected contribution, stated each
dataset's limitation, demoted the unverified-provenance APACC subset to a transparent supporting role, and
gave principled reasons for excluding segmentation-oriented, unverifiable, and non-cytology resources.

## 9.8 Remaining challenges

The selection rests on *measured* dataset properties but inherits the **[unverified]** metadata
limitations of Chapter 8; in particular, the supporting role of APACC and the provenance confirmation of
CRIC and HiCervix remain open and are revisited as threats to validity (Chapter 18). Moreover, the
selected suite, while diverse, is finite: it cannot represent every preparation, scanner, or population,
so the external estimates it yields are lower-dimensional probes of generalisation, not a complete
characterisation — a limitation the partitioning and protocol chapters (16–17) and the validity analysis
(18) address explicitly.

## 9.9 Transition to Chapter 10

With the datasets selected and justified, the volume now characterises them in depth. Chapter 10 analyses,
for each selected dataset, the properties that will shape downstream experimentation — class imbalance,
image quality and variability, segmentation and annotation consistency, intra-class variation and
inter-class similarity, potential confounders, and domain diversity — grounding the analysis in measured
quantities wherever possible and flagging unmeasurable properties explicitly, again without describing any
implementation.

---

# Chapter 10 — Dataset Characterization

This chapter characterises the selected datasets along the properties that govern downstream experimental
validity. Each property is reported as **[measured]**, **[primary]**, or **[unverified]** per the
provenance scheme of Chapter 8, and each is connected to its downstream consequence. The characterisation
is deliberately quantitative where the on-disk material permits and explicitly silent where it does not.

## 10.1 Class imbalance

Class imbalance varies markedly across the suite and is, in every case, **[measured]**. SIPaKMeD is only
**mildly imbalanced** (per-class cropped-cell counts ≈800–830; imbalance ratio ≈1.2), which is favourable
for training. The external datasets are progressively more skewed and, critically, skewed *against* the
clinically important categories. CRIC is **severely imbalanced**: of 11,534 annotated cells, 6,779 are
Negative while squamous cell carcinoma comprises only 161 (a Negative:SCC ratio of ≈42:1), and the
combined high-grade-plus-indeterminate categories (HSIL 1,703; ASC-H 925; ASC-US 606; LSIL 1,360) remain
minority relative to Negative **[measured]**. HiCervix's test split is imbalanced across its `level_1`
categories (negative 2,996; ASC 1,827; AGC 1,639; microbe 1,589) **[measured]**, with the clinically
relevant squamous-abnormal signal (ASC) a minority of the scorable subset. Herlev's 917 cells are
distributed unevenly across seven classes (e.g., normal intermediate 70 vs severe dysplastic 197)
**[measured]**, and APACC is near-balanced at the coarse level (healthy 576; unhealthy 549) **[measured]**.
**Downstream consequence:** average-error metrics are inadequate; balanced accuracy, per-class recall, and
majority-baseline-referenced reporting (Volume I, Section 2.12; Chapter 17) are mandatory, and the severe
minority of high-grade categories (CRIC) means external high-grade recall must be reported and interpreted
with wide uncertainty.

## 10.2 Image quality and acquisition variability

Acquisition variability across the suite is large and **[measured]** through two reproducible proxies:
**file format** and **image dimensions**. The datasets use *three different encodings*: SIPaKMeD and Herlev
are uncompressed **BMP**, APACC and CRIC are lossless **PNG**, and HiCervix is lossy **JPEG** **[measured]**.
That the most preparation-distant dataset (HiCervix, liquid-based) is also the only **lossy-compressed**
one introduces a *compression confound* superimposed on the preparation shift — a measured fact that
complicates attribution and must be acknowledged (Chapters 12, 18). Image dimensions vary by an order of
magnitude: mean single-cell crop dimensions are ≈153×140 px (Herlev), ≈218×189 px (SIPaKMeD), ≈326×339 px
(APACC), and ≈731×679 px (HiCervix), while CRIC supplies whole fields at a fixed 1376×1020 px **[measured]**.
HiCervix crops further span an extreme range (width 97–3220 px) **[measured]**, indicating heterogeneous
cell sizes and/or scanning. **Downstream consequence:** any fixed-input model must resize across these
scales, so apparent cell scale is not preserved across datasets; this is a domain-shift source in its own
right and motivates explicit resolution handling (Chapter 15) and the crop-scale sensitivity analysis of
later volumes.

## 10.3 Staining and microscope variability

Papanicolaou staining is common to the conventional-smear datasets **[primary, general]**, but the *exact*
staining protocols, reagents, microscope models, cameras, objective magnifications, and pixel sizes
(µm/px) are **[unverified]** for every dataset in this environment (Table 8.2) and cannot be compared
numerically here. This is a genuine evidential gap, not a property that can be inferred. What *can* be
established, as a **[measured]** lower bound on staining/scanner heterogeneity, is the format-and-scale
variability of Section 10.2 and the preparation diversity of Section 10.13: these guarantee that staining
response and optical sampling differ across the suite even though the precise parameters are unconfirmed.
**Downstream consequence:** stain-appearance shift is present and must be assumed in experimental design
(motivating stain-normalisation evaluation in later volumes), but its *magnitude* cannot be reported from
metadata and must instead be inferred from model behaviour and from appearance statistics measured on the
images themselves.

## 10.4 Segmentation consistency

Segmentation/annotation *form* is **inconsistent across the suite**, a fully **[measured]** property with
strong downstream implications. SIPaKMeD provides per-cell **nucleus and cytoplasm contours** (16,196
`.dat`); Herlev provides per-cell **binary segmentation masks**; CRIC provides only **nucleus-centroid
points** on dense fields; HiCervix provides **class labels without segmentation**; and the APACC subset
provides **folder-level labels only** **[measured]**. The single-cell *crops* are thus produced by
*different* procedures — expert contour-based extraction (SIPaKMeD), mask-based (Herlev), point-crop
(CRIC), and auto-segmentation (APACC) — so the very notion of "a single cell" differs across datasets.
**Downstream consequence:** segmentation/cropping is itself a domain variable (Chapter 12); only SIPaKMeD's
contours support ground-truth morphometric and explanation-faithfulness analysis; and CRIC's point-only
annotation necessitates a cropping choice whose effect this work measures (the dense-field geometry of
Section 8.4) rather than assumes.

## 10.5 Annotation uncertainty

Cytological annotation carries inherent uncertainty, particularly for borderline categories, as documented
by interpretive-reproducibility studies (Stoler & Schiffman, 2001) **[primary]**. This uncertainty is
*structurally encoded* in the label spaces of two used datasets: CRIC and HiCervix include explicitly
**indeterminate** Bethesda categories — ASC-US ("undetermined significance") and ASC-H ("cannot exclude
high-grade") — which denote diagnostic uncertainty rather than a definite morphology **[measured: these
categories are present]**. The number of annotators, adjudication procedure, and inter-annotator agreement
for each dataset are **[unverified]** here. **Downstream consequence:** label noise imposes an irreducible
ceiling on achievable accuracy (so near-ceiling external claims warrant scepticism), and the indeterminate
categories cannot be mapped to a definite tier without information loss, which the crosswalk handles by
exclusion or by an explicit grade assignment with stated assumptions (Chapter 11).

## 10.6 Intra-class variation and inter-class similarity

Cytological categories lie on a continuous, partially overlapping morphological spectrum (Volume I, Section
2.2), so high intra-class variation and inter-class similarity are expected rather than incidental. In this
work, inter-class similarity is **[measured] indirectly** via the internal confusion structure: the
SIPaKMeD-trained classifier's errors concentrate at the morphologically adjacent boundaries — most notably
**Koilocytotic↔Metaplastic** and **Dyskeratotic↔Koilocytotic** — consistent with the cytological overlap of
koilocytotic halo, metaplastic, and dyskeratotic appearances **[measured; internal confusion matrix]**.
A *direct* morphometric characterisation was performed on SIPaKMeD's contour-derived 53-dimensional
feature vector per cell (nuclear and cytoplasmic geometry, three-channel intensity statistics, and the
diagnostic nuclear-to-cytoplasmic ratio); the full methodology and results are documented separately
(*Class-Characterization Analysis* companion). The measured findings are **[measured;
`results/metrics/class_analysis/`]**: (i) the five classes **overlap substantially** in feature space —
the global silhouette coefficient is only **0.089** (binary 0.095) — quantitatively confirming the
continuous-spectrum premise (Volume I, §2.2); (ii) per-class silhouette ranks **Parabasal most separable
(0.223)** and **Koilocytotic least separable (−0.004, i.e. inseparable)**, with Metaplastic (0.049),
Superficial-Intermediate (0.074), and Dyskeratotic (0.107) in between; and (iii) all 53 features are
significantly class-dependent (Kruskal–Wallis, Benjamini–Hochberg q < 0.05), with **cytoplasm-size
features the strongest discriminators and the N/C ratio close behind (rank 5 of 53; effect size 0.569)**,
oriented exactly as cytology predicts (abnormal → smaller cytoplasm → higher N/C). **A notable
cross-validation:** the morphometric difficulty ranking (Koilocytotic hardest, Parabasal easiest) coincides
with the deep classifier's per-class behaviour (lowest recall on Koilocytotic, perfect on Parabasal) —
two methodologically independent analyses agreeing that difficulty is *intrinsic to the morphology*.
**Downstream consequence:** achievable per-class separability is bounded by genuine morphological overlap,
so confident misclassification at class boundaries (especially Koilocytotic↔Metaplastic) is a calibration
concern (motivating reliability-first design), and the empirically confirmed strength of cytoplasm-size and
N/C-ratio features substantiates the multimodal/morphometric branch of RQ7.

## 10.7 Potential confounders

Several **[measured]** confounders are present and must be controlled or acknowledged. **(i) Compression**:
HiCervix's JPEG encoding versus the BMP/PNG of the others (Section 10.2). **(ii) Cell-versus-field scale**:
CRIC's dense fields (median nearest-neighbour nucleus distance ≈70 px) preclude isolated whole-cell crops
at the source scale, so any cropping introduces a scale/context confound (Section 8.4). **(iii)
Segmentation procedure**: differing cropping pipelines (Section 10.4). **(iv) Out-of-squamous categories**:
every external dataset contains categories the model cannot represent (Section 8.9), which, if not
excluded, would masquerade as errors. **Downstream consequence:** experimental design must hold the model
fixed while varying one domain factor at a time (Volume I, Section 5.3), report excluded fractions, and
treat the CRIC crop scale as a sensitivity variable rather than a fixed choice.

## 10.8 Demographic bias

Patient demographics — age distribution, geographic origin, HPV-genotype distribution, and disease
prevalence — are **[unverified]** for every dataset in this environment. This absence is itself a finding
with two consequences. First, **demographic representativeness cannot be assessed**, so claims about
generalisation to a target population are bounded; the external datasets probe *acquisition/taxonomy*
shift, not *demographic* shift, and this scope must be stated (Chapter 18). Second, the lack of demographic
metadata is a **data-governance limitation** of the public landscape (Section 8.7) that motivates the FAIR
and datasheet practices this work adopts for its own artefacts (Chapter 17). No demographic figure is
inferred.

## 10.9 Acquisition bias

Beyond demographics, **acquisition bias** — systematic differences in how each dataset was imaged and
curated — is **[measured]** in its proxies (format, scale, segmentation form, preparation; Sections
10.2–10.4, 10.10) even where the underlying parameters are unconfirmed. Curated single-cell datasets
(SIPaKMeD, Herlev) further carry **selection bias** toward representative, often unambiguous cells (Volume
I, Section 2.11), under-representing the borderline and artefact-laden cells of consecutive material; CRIC
and HiCervix, being field- or scanner-derived, are comparatively closer to consecutive material but
introduce their own scanner/segmentation biases. **Downstream consequence:** performance estimated on
curated cells may overstate performance on consecutive slides, and this gap is *additional* to the
preparation-shift gap the external suite measures.

## 10.10 Domain diversity

The suite's **[measured]** domain diversity is its principal experimental asset. Across the five datasets,
the design spans: conventional smear (SIPaKMeD, Herlev, CRIC-as-fields), auto-segmented crops (APACC), and
liquid-based cytology (HiCervix); three image encodings; an order-of-magnitude range of cell scales; four
distinct segmentation/cropping procedures; and three label-space families (appearance, dysplasia grade,
Bethesda). This diversity is precisely what enables the cross-domain and cross-taxonomy measurements the
research requires; it is also what makes naïve cross-dataset comparison invalid without the crosswalk and
leakage-safe protocol of the chapters that follow.

## 10.11 How these characteristics shape downstream experimentation

Synthesising Sections 10.1–10.10, the measured characteristics impose concrete experimental requirements,
each developed in a later chapter: severe external imbalance and indeterminate categories require
balanced, per-class, baseline-referenced metrics and an explicit excluded fraction (Chapter 17); format,
scale, segmentation, and preparation heterogeneity require an explicit harmonisation methodology and a
recognition that apparent cell scale is not preserved (Chapter 15); the differing segmentation forms and
CRIC's density require honest, sensitivity-analysed cropping (Chapters 15–16); selection and acquisition
biases require scoped claims and a threats-to-validity analysis (Chapter 18); and the absence of
demographic metadata bounds the population-generalisation claims the research may make (Chapter 18). In
**interpretation**, the characterisation confirms that the datasets are richly diverse and richly
confounded in equal measure, so that the scientific value of the suite is realisable *only* under a
disciplined curation, harmonisation, partitioning, and reporting methodology — the remainder of this
volume.

## 10.12 Chapter summary

This chapter characterised the five selected datasets along imbalance, image quality and acquisition
variability, staining/microscope variability, segmentation consistency, annotation uncertainty, intra-class
variation and inter-class similarity, confounders, demographic and acquisition bias, and domain diversity,
reporting measured quantities (class distributions; format and scale heterogeneity; segmentation forms;
CRIC field geometry; internal confusion structure) and explicitly flagging unverifiable properties
(staining parameters, microscope, magnification, demographics). It established that the suite is
deliberately and measurably diverse across preparation, format, scale, segmentation, and taxonomy, and
correspondingly confounded, and it traced each characteristic to a downstream experimental requirement.

## 10.13 Remaining challenges and transition to Chapter 11

The characterisation leaves two challenges. The **morphometric class-structure metrics** (separability,
discriminative features) are available in principle from SIPaKMeD's contours but are reported only at the
methodological level here, pending execution of the characterisation analysis, so as not to assert
unmeasured values. And the **unverifiable acquisition/demographic metadata** continue to bound the
strength of cross-domain claims. The single most consequential measured characteristic — that every
external dataset's label space is structurally incompatible with the model's squamous appearance space —
now requires a rigorous, dedicated treatment. **Chapter 11** therefore undertakes an exhaustive comparison
of the SIPaKMeD, Herlev, and Bethesda taxonomies, analysing their semantic, morphological, and diagnostic
correspondences and their one-to-many, many-to-one, ambiguous, and clinically unsafe mappings, and
reviewing existing harmonisation strategies — without describing implementation.

---

# Chapter 11 — Label Taxonomy Analysis

This chapter analyses the relationships among the label taxonomies of the used datasets — SIPaKMeD's
five-class appearance taxonomy, Herlev's seven-class dysplasia-grade taxonomy, and the Bethesda System
(CRIC, HiCervix), with APACC's coarse binary as a degenerate case — and establishes, as the central
methodological result of the volume, why a deterministic, many-to-one **crosswalk** with explicit
exclusions is the only clinically safe basis for cross-dataset comparison. The analysis is conceptual and
clinical; it describes the *logic* of the crosswalk used by this work but contains no implementation.

## 11.1 The three taxonomies and their semantics

The taxonomies differ not merely in their categories but in *what kind of thing* their categories are
(Volume I, §2.3–2.7), a distinction summarised in Table 11.1.

**Table 11.1 — Semantic type of each taxonomy.**

| Taxonomy | Categories | Semantic type | Reference axis |
|---|---|---|---|
| SIPaKMeD | 5: SI, Parabasal, Metaplastic, Koilocytotic, Dyskeratotic | cell **appearance** | morphology |
| Herlev | 7: normal (superficial, intermediate, columnar); mild/moderate/severe dysplasia; carcinoma in situ | dysplasia **grade** | CIN/histology |
| Bethesda | NILM; ASC-US; ASC-H; LSIL; HSIL; SCC; (AGC, glandular) | **risk-stratification** | management |
| APACC (subset) | 2: healthy / unhealthy | coarse **binary** | — |

The decisive consequence is that **no information-preserving one-to-one map exists between any pair**: an
appearance does not determine a grade, a grade does not determine a management category, and the Bethesda
indeterminate categories (ASC-US, ASC-H) encode *uncertainty itself*, which no morphology class can carry.

## 11.2 Morphological, diagnostic, and semantic correspondence

Correspondence between taxonomies must be analysed on three distinct levels, which are frequently
conflated. **Morphological correspondence** asks whether two categories describe the same cell appearance:
SIPaKMeD Koilocytotic corresponds morphologically to the koilocytotic (HPV cytopathic) appearance that
*contributes to* Herlev mild dysplasia and Bethesda LSIL, but is not identical to either. **Diagnostic
correspondence** asks whether two categories imply the same clinical grade: here SIPaKMeD's appearance
classes are *silent*, because appearance under-determines grade. **Semantic correspondence** asks whether
two categories play the same role in their system: a SIPaKMeD class (describe a cell) and a Bethesda class
(stratify risk for a slide) play *different roles* and are therefore not directly comparable. The measured
morphometric overlap of the SIPaKMeD classes (Chapter 10.6: global silhouette ≈0.09; Koilocytotic
inseparable) reinforces this: even *within* the appearance taxonomy the categories blur, so projecting
them onto a grade or risk taxonomy compounds an already fuzzy partition.

## 11.3 Mapping hazards: one-to-many, many-to-one, ambiguous, and unsafe

Four mapping hazards arise, each illustrated with concrete, dataset-grounded examples.

- **One-to-many (undefined).** Mapping a Bethesda category *down* to a single appearance class is
  one-to-many and therefore undefined: **LSIL** subsumes both koilocytotic and (mildly) dyskeratotic
  appearances, and **NILM** subsumes all three SIPaKMeD normal appearances (Superficial-Intermediate,
  Parabasal, Metaplastic). A forced one-to-one map in this direction must invent label information.
- **Many-to-one (safe, lossy).** Mapping *up* — collapsing several fine categories to a coarser tier — is
  deterministic and information-losing but **safe**: {Koilocytotic, Dyskeratotic} → abnormal;
  {LSIL, HSIL, ASC-H, SCC} → abnormal; {Superficial-Intermediate, Parabasal, Metaplastic} → normal. This
  is the only direction in which a deterministic map exists.
- **Ambiguous (indeterminate).** Bethesda **ASC-US** and **ASC-H** denote diagnostic *uncertainty*; they
  have no definite appearance and no unambiguous binary grade (ASC-US in particular sits between normal
  and LSIL). They cannot be mapped without an explicit, stated convention.
- **Clinically unsafe.** Two unsafe patterns must be avoided. First, mapping **glandular** categories
  (Herlev columnar; Bethesda AGC) into the *squamous* appearance space, which is biologically wrong.
  Second, asserting that the model can represent **high-grade** disease: a five-class squamous appearance
  taxonomy has **no HSIL/CIS category**, so any mapping that implies high-grade detection is unsafe — the
  model's high-grade recall is structurally zero (Volume I, §2.4).

## 11.4 The deterministic crosswalk adopted by this work

In response to these hazards, this work adopts a deterministic, many-to-one crosswalk to two coarse
**evaluation tiers**, with an explicit **exclusion** set; its logic (not its implementation) is documented
here for completeness and is the basis of all cross-dataset comparison in later volumes.

- **Tier E0 (binary: normal / abnormal).** Model side: {Superficial-Intermediate, Parabasal, Metaplastic}
  → normal; {Koilocytotic, Dyskeratotic} → abnormal, with the abnormality score formed from the
  abnormal-class probability mass. Label side: Herlev {superficiel, intermediate} → normal and
  {light/moderate/severe dysplasia, carcinoma in situ} → abnormal; Bethesda {NILM} → normal and
  {LSIL, HSIL, ASC-H, SCC} → abnormal; APACC {healthy} → normal, {unhealthy} → abnormal.
- **Tier E1 (grade: normal / low / high).** A diagnostic tier that exposes the model's structural blind
  spot: the model can emit normal or low but **never high** (no HSIL output), so external high-grade
  recall under E1 is reported as a measured zero rather than concealed.
- **Exclusions (⊥).** Categories with no squamous-appearance home are *excluded and counted*, not mapped:
  Herlev **columnar** (glandular), Bethesda **AGC** (glandular) and HiCervix **microbe**, and the coarse
  **rubbish/both-cell** categories reported for some APACC variants. The Bethesda indeterminate categories
  (ASC-US/ASC-H) are handled by an explicit, stated convention at each tier (e.g., ASC-US excluded from
  E1 grade as indeterminate). The **excluded fraction** is reported per dataset as an honesty metric
  (Chapter 8: ≈11% Herlev, ≈40% HiCervix test, measured).

The defining property of this crosswalk is that **it is deterministic and many-to-one in the safe
direction, never one-to-many**, and that **everything it cannot map, it excludes and reports**. This is
what permits a cross-dataset performance change to be attributed to the *model* rather than the *labels*
(Volume I, §2.19.3) — the precondition for the diagnostic analysis of later volumes.

## 11.5 A documented choice and its sensitivity: Herlev columnar

One crosswalk choice warrants explicit scrutiny as an example of the transparency the method demands. The
Herlev **columnar** class is glandular and has no squamous-appearance counterpart; a *strict* policy
excludes it, while a *legacy* policy maps it to normal (treating columnar as a benign non-dysplastic
category). This work treats the choice as a documented, sensitivity-tested decision rather than a hidden
assumption: the two policies move the headline binary result by **less than one point** (a measured
sensitivity result reported in later volumes), so the conclusions are robust to it, but the choice is
stated rather than buried. This exemplifies the general principle: *every* non-trivial mapping decision is
to be documented and, where it could matter, sensitivity-tested.

## 11.6 Review of existing cross-dataset harmonisation strategies

Existing approaches to reconciling incompatible cervical taxonomies fall into four families, each with
documented limitations. **(i) Informal relabelling** — the most common practice — maps one taxonomy's
classes onto another's by author judgement, frequently without stating the assumptions or the excluded
categories; its limitation is that it conflates label-mapping artefacts with genuine performance
differences and is irreproducible (the central criticism of Volume I, §3.4). **(ii) Lowest-common-tier
collapse** — mapping all taxonomies to a shared binary or coarse tier — is principled and is the basis of
the present crosswalk, but is information-losing and must be paired with explicit exclusion accounting to
be honest. **(iii) Hierarchical label structures** — exploiting a taxonomy's own hierarchy (as HiCervix
provides via `level_1/2/3`) — preserve more information but require all datasets to share a hierarchy,
which they do not. **(iv) Partial-label / weak-supervision learning** — treating a coarse label as a
*set* of compatible fine labels (Cour et al., 2011) — is the most principled route to *learning* across
taxonomies (as opposed to merely *evaluating*), and is the conceptual basis for the multi-dataset training
question RQ4; its limitation is that it is seldom applied to cervical taxonomies and its empirical benefit
in this domain is unestablished. The **interpretation** is that families (ii) and (iv) are the defensible
options — (ii) for *evaluation*, (iv) for *learning* — and that (i) must be replaced by an explicit
crosswalk for any claim to be trustworthy.

## 11.7 Unresolved challenges

Three taxonomic challenges remain genuinely unresolved and are stated as such. First, the **indeterminate
categories** (ASC-US/ASC-H) have no fully satisfactory mapping: any convention either discards clinically
real cases or assigns them a definiteness they lack; this is an inherent property of the Bethesda System,
not a defect of the method. Second, the **glandular axis** (columnar, AGC) is entirely outside a squamous
appearance space, so glandular disease is structurally unaddressable by a SIPaKMeD-trained model — a scope
limitation, not a solvable mapping problem. Third, the **high-grade blind spot** (no HSIL output) cannot be
remedied by relabelling; it requires either high-grade training data and a richer label space or explicit
acknowledgement, and this work chooses honest acknowledgement and measurement (E1 tier) over concealment.

## 11.8 Chapter summary

This chapter analysed the SIPaKMeD, Herlev, and Bethesda taxonomies and established that they are
semantically heterogeneous — appearance, grade, and risk-stratification respectively — so that no
information-preserving one-to-one map exists between them. It distinguished morphological, diagnostic, and
semantic correspondence; catalogued the one-to-many (undefined), many-to-one (safe), ambiguous, and
clinically unsafe mapping hazards with dataset-grounded examples; and documented the deterministic,
many-to-one crosswalk to the E0/E1 tiers with explicit, counted exclusions that this work adopts, noting
the measured robustness of its one debatable choice (Herlev columnar). It reviewed four families of
harmonisation strategy, endorsing lowest-common-tier collapse for evaluation and partial-label learning
for multi-dataset training while rejecting informal relabelling, and it stated three genuinely unresolved
challenges — indeterminate categories, the glandular axis, and the high-grade blind spot — as honest
limitations rather than solved problems.

## 11.9 Remaining challenges and transition to Chapter 12

The taxonomy analysis isolates the *label* dimension of cross-dataset comparison and shows it can be made
honest by an explicit crosswalk; it does not address the *image-distribution* dimension. Even with a
perfectly honest crosswalk, a model may fail across datasets because the images themselves differ in
preparation, staining, scanner, and segmentation — the phenomenon of domain shift. **Chapter 12** therefore
conducts an evidence-based analysis of domain shift across the suite, drawing on the measured acquisition
heterogeneity of Chapter 10 (format, scale, segmentation, preparation) to characterise the sources and
expected severity of shift and their implications for experimental validity — without describing any
implementation.

---

# Chapter 12 — Domain Shift Analysis

This chapter analyses the sources and structure of domain shift across the dataset suite, at the level of
the *data* (not model behaviour, which belongs to later volumes). It grounds the analysis in the measured
acquisition heterogeneity of Chapter 10 and in the established taxonomy of dataset shift, and it draws the
implications for experimental validity. No implementation is described, and no experimental performance
figure is cited.

## 12.1 Definition and taxonomy of domain shift

Following the standard taxonomy (Quiñonero-Candela et al., 2009; Moreno-Torres et al., 2012), domain shift
is decomposed into three forms relevant here. **Covariate shift** — the input distribution changes while
the label-given-input relationship is stable — is the dominant form in this suite: preparation, staining,
scanner, and segmentation alter the *appearance* of a cell of a given class. **Prior (label) shift** — the
class prior changes — is present because the datasets differ markedly in class composition (Chapter 10.1).
**Concept shift** — the label-given-input relationship itself changes — is present in a subtle, taxonomic
form: because the datasets are labelled under different taxonomies (Chapter 11), "the same" cell appearance
may be assigned different labels, which from the model's fixed-taxonomy perspective behaves like concept
shift. The **interpretation** is that the suite exhibits *all three* forms simultaneously, and that the
crosswalk (Chapter 11) neutralises the taxonomic component so that the remaining covariate and prior shift
can be studied — the precise design intent.

## 12.2 Staining variation

Papanicolaou staining is nominally common to the conventional-smear datasets, but staining *response*
varies with protocol, reagent lot, and laboratory; the exact parameters are **[unverified]** (Chapter
10.3). Stain-appearance shift is well characterised in the cognate field of computational histopathology,
where it is a leading cause of cross-site degradation and the target of dedicated normalisation and
augmentation methods (Reinhard et al., 2001; Macenko et al., 2009; Tellez et al., 2019; Stacke et al.,
2021). By analogy — a **reasonable interpretation**, since cytology and histopathology share the staining
problem — stain-appearance shift should be assumed present across the cytology suite even though its
magnitude cannot be read from metadata. Its presence motivates the inclusion of stain-normalisation as an
*evaluated* (not assumed-beneficial) intervention in later volumes (RQ2).

## 12.3 Imaging devices, scanners, microscopes, and illumination

Variation in microscope optics, camera sensors, digitisation pipelines, and illumination produces
covariate shift in colour balance, sharpness, and noise. For this suite, the specific devices are
**[unverified]** (Table 8.2); however, two **[measured]** proxies establish that device/pipeline shift is
present: the datasets use **three different image encodings** (BMP, PNG, JPEG; Chapter 10.2), and their
image-dimension distributions differ by an order of magnitude (Chapter 10.2). Distinct encodings and
sampling resolutions are downstream signatures of distinct acquisition and processing pipelines. The
device parameters cannot be compared numerically, but their *effect* is measurably present in the image
statistics.

## 12.4 Slide preparation: conventional smear versus liquid-based cytology

Slide preparation is, on the available evidence, the **dominant** shift axis of the suite. Conventional
smear (SIPaKMeD, Herlev, CRIC-as-fields) and liquid-based cytology (HiCervix) differ systematically in
cell density, background, fixation, and presentation (Volume I, §2.8). This is reflected in **[measured]**
proxies: HiCervix (LBC) is the only lossy-JPEG dataset and exhibits the largest cell-scale range (width
97–3220 px), while the conventional-smear datasets are BMP/PNG with narrower scale ranges (Chapter 10.2);
and CRIC's conventional-smear *fields* are geometrically dense (median nearest-neighbour nucleus distance
≈70 px; `cric_crop_geometry.json`), unlike the isolated curated cells of SIPaKMeD. Preparation shift is
thus not a label on a table but a measured property of the images. The **reasonable expectation** — to be
*tested*, not assumed, in later volumes — is that cross-dataset degradation increases with preparation
distance from the conventional-smear source; this chapter establishes only that the preparation axis is
real and measurable, not its effect on performance.

## 12.5 Segmentation pipelines

Segmentation/cropping is a distinct, **[measured]** shift source (Chapter 10.4): the single-cell crops are
produced by four different procedures across the suite — expert contour extraction (SIPaKMeD), mask-based
(Herlev), nucleus-point cropping (CRIC), and auto-segmentation (APACC) — so that the spatial framing,
margin, and context of "a cell" differ by dataset. For CRIC specifically, the measured field density means
no cropping procedure can reproduce the isolated whole-cell framing of the source, so segmentation shift
and scale shift are entangled (Chapter 8.4). Segmentation shift is particularly insidious because it can
introduce *spurious* cues (crop edges, background fraction) that a model may exploit as shortcuts
(Geirhos et al., 2020), a possibility the experimental design must guard against.

## 12.6 Imaging artefacts and compression

Compression is a **[measured]** artefact-level shift: HiCervix's lossy JPEG encoding introduces
block/quantisation artefacts absent from the BMP/PNG datasets (Chapter 10.2). Because the JPEG dataset is
also the most preparation-distant, compression shift is *confounded* with preparation shift — a measured
fact that limits attribution and must be acknowledged (Chapter 18). Other artefacts (air-drying, obscuring
blood/inflammation in conventional smears; scanning artefacts in LBC) are expected from preparation
physics but are not separately quantified here; their presence is a **reasonable interpretation** from the
preparation literature rather than a measured count.

## 12.7 Scale and resolution shift

Apparent cell scale is **not preserved** across the suite, a fully **[measured]** shift (Chapter 10.2):
mean crop dimensions range from ≈153 px (Herlev) to ≈731 px (HiCervix), with CRIC supplying 1376-px fields.
Any fixed-input model resizes across this range, so a cell of fixed physical size occupies a different
fraction of the input depending on the dataset. Because the datasets' µm/px calibrations are **[unverified]**,
scale shift cannot be removed by physical-resolution harmonisation here (Chapter 15); it is therefore a
residual covariate shift whose handling (and the sensitivity of results to crop scale, especially for
CRIC) must be made explicit rather than assumed away.

## 12.8 Demographic and prevalence variation

Demographic variation (age, geography, HPV genotype) and disease-prevalence variation induce prior shift
and potentially covariate shift, but demographics are **[unverified]** for every dataset (Chapter 10.8).
Prevalence (prior) shift is, however, **[measured]** indirectly through the datasets' differing class
compositions (Chapter 10.1): the abnormal/normal balance differs substantially across the suite. The
**implication** is that the suite probes acquisition and taxonomic shift well but demographic shift not at
all, bounding the population-level generalisation claims the research may make — a scope statement carried
to Chapter 18.

## 12.9 The preparation-distance ordering of the suite

Synthesising the measured proxies (encoding, scale range, segmentation procedure, field density,
preparation modality), the external datasets can be **ordered by composite distance from the
conventional-smear single-cell source**: Herlev (conventional smear, isolated cells, BMP, near) <
APACC (auto-segmented crops, PNG, moderate) < CRIC (conventional-smear dense fields, point-cropped, PNG,
far in format/scale) < HiCervix (liquid-based, JPEG, largest scale range, farthest). This ordering is a
**measured, data-level construct** (it uses only image/annotation properties, not model behaviour) and
serves as the pre-registered explanatory variable for the generalisation analysis of later volumes
(RQ2). Stating it here, from data alone, guards against the circularity of defining "distance" post hoc to
fit performance.

## 12.10 Implications for experimental validity

The measured domain shift has four implications for experimental validity, each addressed later. First,
**external validity requires the multi-domain suite**: a single-dataset estimate is silent about the
covariate/prior shift the suite measures (Chapter 18). Second, **attribution requires single-factor
variation and confound acknowledgement**: because compression is confounded with preparation, and
segmentation with scale, the design must vary one factor where possible and flag the irreducible confounds
(Chapter 18). Third, **the operating point may not transfer**: prior shift across datasets means a fixed
decision threshold calibrated on the source need not be appropriate on a target, so per-domain calibration
must be evaluated (a reliability concern carried from Volume I, §2.15). Fourth, **scale and segmentation
must be treated as variables, not constants**, with explicit sensitivity analysis for the dense CRIC case.

## 12.11 Chapter summary

This chapter analysed domain shift across the suite at the data level. It decomposed the shift into
covariate, prior, and (taxonomic) concept components, all present simultaneously; established — through
measured proxies (encoding, scale, segmentation procedure, field density, class composition) — that
staining, device, preparation, segmentation, compression, scale, and prior shifts are real and present,
while flagging the unverifiable device, staining-parameter, and demographic specifics; and constructed,
from data alone, a preparation-distance ordering of the external datasets to serve as a pre-registered
explanatory variable. It drew four implications for experimental validity concerning external validity,
attribution and confounds, operating-point transfer, and scale/segmentation handling.

## 12.12 Remaining challenges and transition to Chapter 13

Domain shift threatens the *external* validity of cross-dataset claims; a second, equally serious threat
concerns the *internal* validity of within-dataset claims, namely data leakage. Even a perfectly honest
crosswalk and a well-characterised shift cannot rescue an internal benchmark whose test set is
contaminated by training-set information. **Chapter 13** therefore analyses every form of leakage relevant
to cervical cytology — patient, slide, field-of-view, duplicate, annotation, preprocessing, augmentation,
hyperparameter, validation, and benchmark leakage — critically reviews prior evaluation methodologies, and
justifies the leakage-safe experimental philosophy, grounding the analysis in the measured cluster
structure of SIPaKMeD, again without describing implementation.

---

# Chapter 13 — Data Leakage Analysis

This chapter analyses data leakage — the flow of information from the test set into training or model
selection that inflates estimated performance (Volume I, §2.10; Kapoor & Narayanan, 2023) — for cervical
cytology specifically. It enumerates every leakage form relevant to the domain, critically reviews prior
evaluation practice, and justifies the leakage-safe philosophy, grounding the analysis in the measured
cluster structure of SIPaKMeD. It describes the *philosophy and data structure*, not the implementation of
any split.

## 13.1 Why cervical cytology is acutely susceptible to leakage

Cervical cytology is structurally predisposed to leakage because **many cells share one acquisition
context**: a single slide yields many fields, and a single field yields many cells, all sharing staining,
focus, illumination, and incidental artefacts. SIPaKMeD makes this concrete and **[measured]**: its 4,049
single cells derive from **966 cluster (field-of-view) images** — roughly four cells per field. A
partitioning that ignores this structure will, with near-certainty, place sibling cells from one field
into both training and test partitions, so that the test estimate rewards recognition of a shared imaging
context rather than generalisation to new cells. The susceptibility is therefore not hypothetical but a
direct consequence of the measured cell-to-field multiplicity.

## 13.2 A taxonomy of leakage forms relevant to cervical cytology

Ten forms of leakage are relevant; each is defined, its cytology-specific manifestation stated, and its
status in this work's data foundation noted.

- **Patient leakage.** Cells from one patient in both partitions. In SIPaKMeD, **patient identifiers are
  [unverified]** (not present in the on-disk metadata), so patient-level grouping cannot be performed; the
  field/cluster grouping (below) is the **finest available** proxy. This is an honest residual limitation
  (Chapter 18), not a solved problem.
- **Slide leakage.** Cells from one slide in both partitions. Slide identifiers are likewise **[unverified]**;
  the cluster (field) is a sub-slide unit, so cluster grouping mitigates but does not provably eliminate
  slide leakage.
- **Field-of-view (FOV) leakage.** Cells from one field in both partitions — the **dominant, measurable,
  and addressable** form here. SIPaKMeD's per-cell provenance to its 966 fields is **[measured]**, so
  grouping the partition by field identity eliminates FOV leakage by construction (Chapter 16).
- **Duplicate leakage.** Identical or near-duplicate images across partitions. Exact duplicates are
  addressed by curation (Chapter 14); **perceptual near-duplicate detection within SIPaKMeD is not yet
  automated** in this work and is flagged as a curation gap (Chapters 14, 18).
- **Annotation leakage.** Using test-set labels to inform any development step (e.g., feature selection,
  thresholding). Precluded by the sealed-test discipline (§13.4).
- **Preprocessing leakage.** Fitting any data-dependent preprocessing (normalisation statistics, a stain
  normaliser) on data that includes the test set. This work fits all such preprocessing on the **training
  partition only** **[measured: the stain-normalisation fit is restricted to `split["train"]`]**,
  precluding this form.
- **Augmentation leakage.** Generating augmented copies *before* splitting, so that augmentations of a test
  cell appear in training. Precluded by splitting before any augmentation.
- **Hyperparameter leakage.** Selecting hyperparameters by test-set performance. Precluded by tuning on the
  validation partition only (Volume I protocol; Chapter 17).
- **Validation leakage.** Using the test set for model selection or early stopping. Precluded by the
  **sealed test**: model selection and early stopping use train + validation only.
- **Benchmark leakage.** Community-level over-fitting to a fixed public test set through repeated
  evaluation. Mitigated by releasing the exact split and seeds (so others evaluate on the same sealed
  partition rather than re-deriving favourable splits) and by reporting against baselines (Chapter 17).

## 13.3 Critical review of prior evaluation methodologies

The prevailing evaluation practice for SIPaKMeD- and Herlev-style benchmarks is **random per-cell
partitioning**. This practice is, on the analysis of §13.1, **leakage-prone by construction**: because
cells inherit a field context, a random per-cell split induces FOV (and, where applicable, slide and
patient) leakage, biasing the test estimate upward. Two further weaknesses compound it. First,
**non-release of splits and seeds** prevents independent reproduction and enables, at the community level,
benchmark leakage. Second, the use of a single test partition for both model selection and final reporting
(i.e., no sealed test) conflates validation and test leakage. The **critical conclusion** — consistent
with the documented reproducibility concerns of the broader field (Kapoor & Narayanan, 2023; Varoquaux &
Cheplygina, 2022) — is that near-ceiling accuracies reported under random per-cell splits are not, in
general, valid estimates of within-dataset generalisation, and that the field's empirical record requires
re-establishment on a leakage-safe footing.

## 13.4 The leakage-safe philosophy and the SIPaKMeD cluster structure

The leakage-safe philosophy (Volume I, §5.6) makes three commitments, all enabled by the measured cluster
structure. **(i) Group-aware partitioning:** the partition is formed over the **field/cluster identity**
(the finest available grouping; §13.2), so that no field's cells are split across partitions. SIPaKMeD's
966 fields make this both necessary and possible **[measured]**. **(ii) Sealed test:** a test partition is
set aside and consulted exactly once, after all development, precluding validation, hyperparameter, and
annotation leakage. **(iii) Quantification, not assumption:** the magnitude of leakage is *measured*
against a deliberately leaky control (§13.5), rather than assumed negligible. The verification that the
resulting partition has **zero field overlap** across train/validation/test is asserted as a property of
the data foundation **[measured: the project's grouped split asserts zero cluster overlap, with partition
sizes of 3,238 / 405 / 406 cells]** (Chapter 16 develops the partitioning).

## 13.5 The leaky control and the leakage tax

Consistent with the experimental philosophy of Volume I (§5.3), the leakage-safe partition is paired with a
**deliberately leaky control** — a random per-cell partition of the same data — so that the inflation
attributable to leakage (the "leakage tax") can be *measured* as the difference in estimated performance
between the two regimes. This chapter establishes the **design** of that control as part of the data
foundation; the *measured* value of the leakage tax is an experimental result reported in a later volume,
and is deliberately not stated here, in keeping with the volume's scope. The conceptual point is that
quantifying the tax converts a methodological assertion ("random splits inflate accuracy") into a measured,
dataset-specific fact.

## 13.6 Implications for the data foundation

The leakage analysis imposes concrete requirements on the data foundation, developed in subsequent
chapters. Curation must include **exact-duplicate removal** and should include **perceptual near-duplicate
screening** (Chapter 14). Partitioning must be **group-aware with a sealed test** and must **assert zero
group overlap** (Chapter 16). The protocol must **release splits and seeds** and **report against baselines**
to limit benchmark leakage (Chapter 17). And the **residual, unaddressable** leakage forms — patient- and
slide-level grouping, absent the identifiers — must be stated as honest limitations bounding the internal
estimate (Chapter 18). The **interpretation** is that leakage cannot be wholly eliminated with the
available metadata, but that the addressable forms (FOV, preprocessing, augmentation, hyperparameter,
validation) are eliminated by construction and the residual forms are honestly bounded — a defensible and
reproducible footing that the prevailing practice does not provide.

## 13.7 Chapter summary

This chapter analysed data leakage for cervical cytology. It established, from SIPaKMeD's measured
cell-to-field multiplicity (4,049 cells from 966 fields), that the domain is acutely leakage-prone; it
enumerated ten leakage forms, stating for each its cytology manifestation and its status in this work —
FOV, preprocessing, augmentation, hyperparameter, and validation leakage eliminated by construction;
patient and slide leakage mitigated by field grouping but residually limited by absent identifiers;
duplicate leakage partly addressed with a flagged perceptual-screening gap; benchmark leakage mitigated by
release and baselines. It critically reviewed random per-cell partitioning as leakage-prone and
irreproducible, justified the group-aware, sealed-test philosophy enabled by the measured cluster
structure, and described the leaky-control design that measures the leakage tax while deferring its value
to a later volume.

## 13.8 Remaining challenges and transition to Chapter 14

Two leakage challenges remain open at the data-foundation level: the **absence of patient/slide
identifiers** (limiting grouping to the field level) and the **un-automated perceptual near-duplicate
screening** within and across datasets. Both are curation concerns. **Chapter 14** therefore develops the
reproducible data-curation framework — inclusion/exclusion criteria, corrupted-image and duplicate
detection, annotation and metadata verification, provenance tracking, quality assurance, and version
control — that operationalises the integrity requirements identified here and in Chapters 10–12, justifying
each decision and, again, without describing implementation.

---

# Chapter 14 — Data Curation Framework

This chapter develops the reproducible data-curation methodology that operationalises the integrity
requirements of Chapters 10–13. It states the inclusion and exclusion criteria, the corrupted-image and
duplicate detection procedures, annotation and metadata verification, provenance tracking, quality
assurance, and version-control and reproducibility considerations, justifying each decision and grounding
it in the project's measured curation outcomes. It describes *what is curated and why*, not the code that
performs it.

## 14.1 Principles of the curation framework

The curation framework rests on three principles inherited from the research philosophy (Volume I, Chapter
5): **provenance-first** (every curated artefact is traceable to its source and every decision recorded);
**conservative inclusion** (a sample is included only if it is valid, scorable, and correctly attributable,
with everything excluded *counted and reported* rather than silently dropped); and **measured verification**
(curation outcomes — corrupt counts, exclusion fractions — are measured and persisted, not assumed). These
principles convert curation from an unrecorded preprocessing step into a documented, reproducible component
of the methodology.

## 14.2 Inclusion criteria

A sample (a single cell or a field) is **included** if it satisfies all of: **(i) validity** — the image
decodes to a non-empty RGB array (§14.4); **(ii) scorability** — its native label maps to a defined
evaluation tier under the crosswalk (Chapter 11), i.e., it is not an excluded category; **(iii)
attributability** — for the training source, the cell is traceable to its field/cluster, enabling
group-aware partitioning (Chapter 13); and **(iv) annotation sufficiency** — for analyses requiring
contours (morphometrics, faithfulness), the cell carries the requisite nucleus/cytoplasm annotation. The
criteria are *ordered*: validity is checked first, then scorability, then attributability, so that an
excluded sample is attributed to the first criterion it fails, supporting transparent accounting.

## 14.3 Exclusion criteria

A sample is **excluded** — and the exclusion **counted** — if it fails any inclusion criterion. The
principal, measured exclusion classes are: **(i) invalid images** (zero-byte or undecodable; §14.4);
**(ii) out-of-tier categories** under the crosswalk — glandular (Herlev columnar, Bethesda AGC), microbe
(HiCervix), indeterminate where the tier requires definiteness (ASC-US/ASC-H at the grade tier), and coarse
non-cell categories (rubbish/both-cell in some APACC variants) — whose **excluded fraction is reported per
dataset** (Chapter 8: ≈11% Herlev, ≈40% HiCervix test, **[measured]**); and **(iii) unattributable
samples** lacking the grouping key required for leakage-safe partitioning. The governing rule is that
**no sample is discarded silently**: every exclusion is enumerated, because silent dropping of hard or
unmappable cells would inflate the reported score (Volume I, §2.7).

## 14.4 Corrupted and invalid image detection

Corrupted-image handling is a **measured** component of the framework, motivated by a concrete failure
encountered in this project: a zero-byte image file caused an unhandled decode error during early
processing. The curation framework responds with two layers. First, a **dataset-wide integrity scan** that
counts zero-byte and undecodable files across every dataset; the measured outcome after curation is **zero
corrupt or zero-byte files across all five datasets** **[measured: the project's preflight integrity scan
reports 0 zero-byte and 0 unreadable files for SIPaKMeD, Herlev, APACC, HiCervix, and CRIC]**. Second, a
**robust loader contract** at point of use: any file that is zero-byte or fails to decode is excluded (and
counted) rather than allowed to abort processing. The justification is defensive reproducibility: a single
malformed file must never silently corrupt a result or halt a long run, and the integrity of the corpus
must be a *measured* property, not an assumption.

## 14.5 Duplicate detection

Duplicate handling distinguishes two cases. **Exact duplicates** (byte- or content-identical files) are
detectable by content hashing and are removed at curation. **Perceptual near-duplicates** (visually near-
identical crops, e.g., overlapping cells extracted twice, or mirrored/re-encoded copies) require perceptual
hashing or embedding-similarity screening. This work **flags perceptual near-duplicate screening as a
curation gap that is not yet automated** (consistent with the leakage analysis, Chapter 13.2), and treats
it as an honest limitation (Chapter 18) rather than claiming a deduplication it has not performed. The
justification for surfacing the gap is the no-fabrication principle: asserting a perceptual-deduplication
step that was not run would itself be a form of fabrication.

## 14.6 Annotation verification

Annotation verification confirms that each sample's annotations are present, parseable, and consistent with
its image. For SIPaKMeD, this includes verifying the presence and parseability of the per-cell nucleus and
cytoplasm contour files (**[measured]**: 16,196 `.dat` contour files for 4,049 cells) and confirming, for
the morphometric and faithfulness analyses, that each analysed cell's contour rasterises within its image
bounds under the same geometric transform applied to the image (a consistency this work enforces in the
contour-based faithfulness analysis). For Herlev, verification confirms the pairing of each cell image with
its segmentation mask. For the point-annotated CRIC, verification confirms that each annotation row parses
to a valid (filename, coordinate, class) triple, with malformed/header rows rejected. The justification is
that downstream ground-truth analyses (morphometrics, nucleus-localisation faithfulness) are only valid if
the annotations they rest on are verified, not assumed.

## 14.7 Metadata validation

Metadata validation confirms, before any experiment, that each dataset is present, correctly located, and
of the expected size, and that the grouping keys required for partitioning exist. This is a **measured**
preflight step: the framework verifies the presence and image counts of all five datasets and the existence
of the SIPaKMeD field/cluster grouping key prior to splitting **[measured: the project's preflight reports
all datasets present with their counts and confirms the grouping key]**. Where dataset metadata is
incomplete or unverifiable (acquisition hardware, licences, demographics; Chapter 8), validation **records
the field as missing** rather than inferring it — the metadata counterpart of the no-fabrication policy.

## 14.8 Provenance tracking

Provenance tracking is the backbone of the framework. Every curated quantity — counts, class
distributions, exclusion fractions, integrity-scan results, and the partition itself — is **persisted to
machine-readable JSON/CSV**, and every figure is accompanied by a companion data file from which it can be
regenerated **[measured: the project persists all exploratory-data-analysis and curation outputs under
`results/metrics/` with companion figure data]**. The partition is persisted with its seed and its
group-membership lists, so that the exact split is recoverable. The justification is reproducibility-as-a-
property (Volume I, §5.4): a result is only a scientific result if its data provenance can be independently
reconstructed, and provenance tracking is the mechanism that guarantees it.

## 14.9 Quality assurance

Quality assurance comprises the standing checks that the curated corpus satisfies its invariants:
zero corrupt files (§14.4), zero group overlap in the partition (Chapter 13.4), reported exclusion
fractions consistent across the EDA and the crosswalk, and measured class distributions matching the
partition composition. These checks are **measured and re-runnable**, so that any regression (a newly
corrupted file, an accidental overlap) is detected rather than silently tolerated. QA is thus continuous
and evidential, not a one-time manual inspection.

## 14.10 Version control and reproducibility

Reproducibility is secured by three commitments. **Deterministic construction:** all curation and
partitioning steps are seeded (seed = 42), so the corpus and split are reconstructable bit-for-bit.
**Persisted artefacts:** the split (paths, group memberships, seed), the integrity-scan results, and the
EDA/characterisation outputs are versioned files, not transient state. **Environment recording:** the
computational environment is recorded so that numerical results are attributable to a specific software
stack. Together these satisfy the operational reproducibility test of Volume I (§5.4): the data foundation
must be regenerable from the released seed and artefacts. The justification is that without versioned,
seeded, persisted curation, none of the downstream claims could be independently verified.

## 14.11 Chapter summary

This chapter specified the data-curation framework. It set out provenance-first, conservative-inclusion,
and measured-verification principles; ordered inclusion criteria (validity, scorability, attributability,
annotation sufficiency) and the corresponding counted exclusions (invalid, out-of-tier, unattributable);
a two-layer corrupted-image defence with a measured zero-corrupt outcome; a duplicate-handling procedure
that removes exact duplicates and honestly flags perceptual near-duplicate screening as an un-automated
gap; annotation and metadata verification grounded in measured contour/mask/row checks; provenance tracking
via persisted JSON/CSV with companion figure data; continuous, measured quality assurance; and seeded,
versioned, environment-recorded reproducibility. Every decision was justified by an integrity requirement
from Chapters 10–13, and unverifiable or un-performed steps were flagged rather than asserted.

## 14.12 Remaining challenges and transition to Chapter 15

The curation framework yields a clean, attributable, provenance-tracked corpus, but it does not by itself
make the multiple datasets *mutually compatible*: they still differ in taxonomy, format, coordinate
convention, and resolution (Chapters 10–12). Reconciling these into a common experimental substrate —
without erasing the very domain diversity the suite is designed to probe — is the harmonisation problem.
**Chapter 15** therefore describes the conceptual data-harmonisation framework (taxonomy, metadata,
annotation, format, coordinate, and resolution harmonisation, and provenance preservation), discusses its
trade-offs, and explains why some heterogeneity is deliberately preserved rather than harmonised away —
without describing implementation.

---

# Chapter 15 — Data Harmonization Framework

This chapter describes the conceptual methodology for harmonising the heterogeneous datasets into a common
experimental substrate. It treats taxonomy, metadata, annotation, image-format, coordinate, and resolution
harmonisation, discusses the trade-offs of each, and — crucially — explains why certain heterogeneity is
**deliberately preserved** rather than harmonised away. The chapter describes *what is reconciled, what is
not, and why*; it contains no implementation.

## 15.1 The harmonisation dilemma

Harmonisation faces a fundamental tension specific to this research. On one hand, the datasets must be made
**mutually comparable** so that a single model can be evaluated across them under one protocol. On the
other hand, the datasets' **heterogeneity is the research instrument**: the preparation, staining, scanner,
and segmentation differences (Chapter 12) are precisely the domain shift the work sets out to measure
(RQ2). Harmonising that heterogeneity away would destroy the very signal under study. The framework
resolves the dilemma by distinguishing **representational** harmonisation (reconciling *containers and
conventions* so the data can be processed uniformly) from **distributional** harmonisation (reconciling the
*image appearance* itself). The former is performed; the latter is **not** performed as a curation step,
because it is either impossible with the available metadata or is itself an *experimental intervention* to
be evaluated, not assumed.

## 15.2 Taxonomy harmonisation

Taxonomy harmonisation is achieved by the deterministic many-to-one crosswalk of Chapter 11: rather than
mapping taxonomies onto one another (which is unsafe; Chapter 11.3), all predictions and native labels are
mapped to shared **evaluation tiers** (E0 binary, E1 grade), with unmappable categories excluded and the
excluded fraction reported. The **trade-off** is information loss: collapsing to a coarse tier discards the
fine distinctions of each taxonomy. This loss is accepted as the price of clinical safety and reproducible
comparison, and is *bounded and reported* (the excluded fraction; the documented sensitivity of debatable
mappings, Chapter 11.5). Taxonomy harmonisation is thus the one *label-side* harmonisation the framework
performs deterministically.

## 15.3 Image-format harmonisation

The three on-disk encodings (BMP, PNG, JPEG; Chapter 10.2) are harmonised representationally by decoding
every image to a common working RGB representation. The **critical trade-off** is that format
harmonisation standardises the *container* but **cannot recover information lost to lossy compression**:
HiCervix's JPEG artefacts are permanent and survive decoding to RGB. Format harmonisation therefore makes
the data uniformly processable without removing the compression confound (Chapter 12.6), which must remain
an acknowledged threat to validity (Chapter 18). No claim is made that decoding "removes" the JPEG domain
difference; it merely standardises the interface.

## 15.4 Coordinate harmonisation

The datasets carry annotations in incompatible spatial forms (Chapter 10.4): contours (SIPaKMeD), masks
(Herlev), nucleus-centroid points (CRIC), and none (HiCervix, APACC). Coordinate harmonisation converts
these to the common unit of analysis — the single cell. For CRIC, this means converting each
nucleus-centroid point into a single-cell crop, a conversion governed by a **documented, measured** crop
size: this work measured CRIC's field density (median nearest-neighbour nucleus distance ≈70 px;
`cric_crop_geometry.json`) and chose a crop scale of the order of one cell's territory, **sensitivity-tested
across crop sizes** rather than fixed arbitrarily. The **trade-off** is that point-to-crop conversion
introduces a scale/context choice (a dense field cannot yield an isolated whole cell at the source scale;
Chapter 8.4), which is why the crop size is treated as a reported sensitivity variable, not a silent
constant. Coordinate harmonisation thus makes the point-annotated dataset comparable while *exposing* the
assumption it requires.

## 15.5 Resolution harmonisation

Resolution harmonisation is where the framework is most constrained by missing metadata. Two distinct
notions must be separated. **Physical-resolution harmonisation** — matching µm/px so that a cell of fixed
physical size occupies a fixed pixel extent — is **impossible here**, because the datasets' µm/px
calibrations are **[unverified]** (Chapter 8.2). **Pixel-resolution harmonisation** — resizing all crops to
a common input size — *is* performed, but it does **not** preserve physical scale: it makes cells uniformly
sized in pixels while leaving the apparent-scale shift (Chapter 12.7) intact. The framework adopts an
**aspect-preserving** resize (padding to square before resizing) so that nuclear and cellular *morphology*
is not distorted — a deliberate choice, since non-uniform scaling would corrupt the very shape features
(N/C ratio, eccentricity) that are diagnostic. The **trade-off** is explicit: pixel harmonisation enables
a fixed-input model but cannot remove the physical-scale domain shift, which therefore remains a measured
residual confound to be reported, not a solved problem.

## 15.6 Metadata and annotation harmonisation

Metadata is harmonised to a common schema (dataset, role, preparation, native taxonomy, tier mapping,
counts, exclusion fractions), with fields that cannot be verified recorded as **missing** rather than
inferred (Chapter 14.7). Annotation is harmonised to a common record per analysed cell (image reference,
tier label under the crosswalk, and — where available — the segmentation/contour or point annotation),
with the *form* of annotation preserved as a metadata field so that downstream analyses know which cells
support which methods (Chapter 15.8). The **trade-off** is that the harmonised schema is only as complete
as the least-documented dataset; rather than back-filling, the framework records the incompleteness, which
is the honest and reproducible choice.

## 15.7 Provenance preservation through harmonisation

Every harmonisation step is **provenance-preserving**: the transformation applied to each sample (decode,
crop with recorded coordinates and size, aspect-preserving resize) is recorded so that the relationship
between the harmonised representation and the original is documented and the original remains the source of
truth. This ensures that harmonisation does not become an irreversible, undocumented mutation of the data —
a common source of preprocessing leakage and irreproducibility (Chapter 13.2). Provenance preservation is
the harmonisation-level expression of the provenance-first principle (Chapter 14.8).

## 15.8 Compatibility assessment

Harmonisation makes datasets *comparable* at the tier level but does not make them *equivalent* in what
analyses they support; a compatibility assessment records which dataset supports which method. Only
**SIPaKMeD** carries the nucleus/cytoplasm contours required for morphometric characterisation (Chapter 10.6)
and nucleus-localisation faithfulness; **Herlev** supports mask-based analyses; **CRIC** supports point-crop
classification with a documented crop assumption; **HiCervix** and **APACC** support tier-level
classification only. This assessment is itself a curated artefact: it prevents the category error of
applying a contour-dependent analysis to a dataset that lacks contours, and it scopes each finding to the
datasets that can validly support it.

## 15.9 What is deliberately not harmonised, and why

The framework **deliberately preserves** three kinds of heterogeneity. **(i) Stain/appearance shift** is
not normalised at curation, because stain normalisation is an *experimental intervention* whose benefit is
to be *measured* (RQ2), not assumed; harmonising it away at curation would both pre-empt that experiment and
remove the shift under study. **(ii) Preparation shift** (smear vs LBC) is not removed, because it is the
principal axis the research measures (Chapter 12.4). **(iii) Physical-scale shift** cannot be removed
(missing µm/px) and is left as a measured residual. The governing principle is that harmonisation is
restricted to *representational* reconciliation that enables uniform processing without erasing the
domain signal; any *distributional* reconciliation is deferred to the experimental volumes as an evaluated
intervention. This restraint is what keeps the external evaluation a valid measurement of domain shift
rather than a measurement of the harmonisation's own effect.

## 15.10 Chapter summary

This chapter described the harmonisation framework and resolved its central dilemma by distinguishing
representational from distributional harmonisation. It performs representational harmonisation — taxonomy
(via the crosswalk to E0/E1 tiers with reported exclusions), image format (decode to common RGB, with the
permanence of JPEG loss acknowledged), coordinate (point-to-crop with a measured, sensitivity-tested crop
size), pixel resolution (aspect-preserving resize that protects morphology but cannot restore physical
scale), and metadata/annotation (common schema with missing fields recorded) — while preserving provenance
and assessing per-dataset analysis compatibility. It **deliberately declines** distributional harmonisation
of stain, preparation, and physical-scale shift, because these are respectively an evaluated intervention,
the object of study, and an irremovable residual — preserving the domain diversity that makes the suite a
valid instrument for measuring generalisation.

## 15.11 Remaining challenges and transition to Chapter 16

The harmonised, curated corpus is now comparable across datasets at the tier level, with its residual
confounds (compression, physical scale) documented. What remains is to define how the corpus is
*partitioned* for experimentation — how the training source is split for leakage-safe internal estimation,
and how the external datasets are used for cross-dataset and leave-one-dataset-out evaluation. **Chapter 16**
develops the partitioning philosophy, grounding it in the measured cluster-grouped split of SIPaKMeD
(sizes, zero overlap, seed) and justifying every partitioning decision, without describing implementation.

---

# Chapter 16 — Experimental Data Partitioning

This chapter develops the data-partitioning philosophy that governs internal and external evaluation. It
defines the train/validation/test roles, the cluster-grouped internal split, stratification under
grouping, grouped cross-validation, the leaky control, external and cross-dataset evaluation, the
leave-one-dataset-out design, and reproducibility/seed considerations, justifying each decision and
grounding it in the measured split of SIPaKMeD. It describes the *partitioning design*, not its
implementation.

## 16.1 Partitioning philosophy

The partitioning philosophy follows directly from the leakage and validity analyses (Chapters 13, 18). Its
axioms are: **(i) independence** — the test set must be statistically independent of training, which under
the cell-to-field structure (Chapter 13.1) requires group-aware partitioning; **(ii) sealing** — the test
set must be untouched during development, so it estimates generalisation rather than selection; **(iii)
representativeness** — the partition should preserve the class composition so that estimates are not
distorted by partition-induced prior shift; and **(iv) reproducibility** — the partition must be
deterministic from a released seed and group memberships. These axioms jointly determine the design below.

## 16.2 The internal split: cluster-grouped 80/10/10

The training source (SIPaKMeD) is partitioned into training (80%), validation (10%), and test (10%) at the
level of the **field/cluster**, not the cell. The partition is **[measured]**: it comprises **3,238
training, 405 validation, and 406 test cells**, formed over the 966 fields such that **no field's cells
appear in more than one partition** (zero cluster overlap is asserted as a property of the split). The
80/10/10 ratio is justified as the standard compromise that reserves the majority of data for training
while providing a validation partition large enough for model selection and a **sealed** test partition
large enough for an internal estimate with a quantified confidence interval (Chapter 17). The choice of a
*separate* validation and test partition (as opposed to a single held-out set) is justified by the need to
preclude validation/selection leakage (Chapter 13.2): the validation partition absorbs all model-selection
decisions, leaving the test partition uncontaminated.

## 16.3 Train, validation, and test roles

The three partitions have **strictly separated roles**, a separation that is itself a leakage control.
**Training**: parameter fitting only. **Validation**: all data-dependent development decisions — model
selection, early stopping, hyperparameter choice (Volume I, §4.6), and the fitting of any data-dependent
preprocessing such as a stain normaliser (which is fit on training and selected on validation, never on
test; Chapter 13.2). **Test**: a single, final, sealed evaluation. The justification is the post-positivist
requirement (Volume I, §5.1) that the test estimate be exposed to no information from the development
process; any blurring of these roles reintroduces the very leakage the partition exists to prevent.

## 16.4 Stratification under grouping

A tension exists between **stratification** (preserving class proportions across partitions) and
**grouping** (keeping a field's cells together): in general the two cannot be satisfied exactly, because a
field's cells share a class and moving a field moves a block of one class. The framework resolves this with
a **stratified group partition** that approximately preserves class proportions while enforcing exact group
disjointness — prioritising group disjointness (a hard leakage constraint) over perfect stratification (a
soft representativeness preference). The **trade-off** is a small, measured departure from exact class
balance across partitions, accepted because leakage is a validity-destroying error whereas mild strat
imbalance is a quantifiable estimation nuisance addressed by balanced metrics (Chapter 17). The justification
is a clear ordering of error severity: independence first, representativeness second.

## 16.5 Grouped cross-validation folds

For development analyses that require multiple estimates (e.g., confidence intervals on validation
performance, stability checks across resamples), the framework defines **grouped cross-validation folds
over the training-plus-validation data only**, with the **test partition kept sealed** throughout. Folds
are group-disjoint (no field shared across a fold's train/validation), mirroring the main split's leakage
discipline. The justification is that cross-validation must obey the same grouping constraint as the main
split — otherwise the cross-validated estimates inherit FOV leakage — and that the sealed test must never
participate in cross-validation, on pain of selection leakage.

## 16.6 The leaky control partition

To *measure* (not assume) the inflation caused by leakage, the framework defines a **deliberately leaky
control**: a per-cell random 80/10/10 partition of the same data, in which sibling cells from a field are
permitted to split across partitions (Chapter 13.5). Comparing performance under the grouped split against
the leaky control yields the **leakage tax** as a measured quantity (its value reported in a later volume).
The justification is the experimental philosophy of paired controls (Volume I, §5.3): the claim that
grouping matters is itself subjected to measurement rather than asserted, and the control quantifies, for
this specific dataset, how much the field's leakage-prone convention inflates the estimate.

## 16.7 External validation and cross-dataset evaluation

External validation uses the **frozen** model — trained and selected entirely on the SIPaKMeD internal
split — evaluated on each external dataset without any adaptation, at the harmonised tiers (E0, E1, and an
E-morph candidate-set tier), with the excluded fraction reported per dataset (Chapters 11, 15). The
**frozen-model** discipline is essential: any tuning on an external dataset would convert an
out-of-distribution test into an in-distribution one and destroy the cross-domain measurement. Cross-dataset
evaluation is reported **per domain**, against each domain's majority baseline, with threshold-free,
operating-point, calibration, and reliability metrics (Chapter 17), so that a single external number cannot
conceal calibration or operating-point failure (Volume I, §2.19.4). The justification is that the deployment
quantity of interest is performance on a *different* pipeline (Volume I, §5.7), which only frozen,
per-domain external evaluation estimates.

## 16.8 Leave-one-dataset-out evaluation

For the *multi-dataset learning* question (RQ4) — as distinct from frozen cross-dataset *evaluation* — the
appropriate design is **leave-one-dataset-out (LODO)**: train on all-but-one dataset (under the crosswalk
and partial-label learning) and evaluate on the held-out dataset, rotating the held-out dataset. LODO is the
external-validity analogue of cross-validation: it estimates how well a model trained on a *combination* of
sources generalises to an *unseen* source. This work identifies LODO as the **correct design for the
multi-dataset training experiments** and notes that it is **optional/deferred** in the present scope
(requiring multi-dataset binary training); its design is specified here as part of the data foundation so
that the later experiments can instantiate it without re-deriving the partitioning. The justification is
that LODO, not random pooling, is the leakage-safe way to estimate multi-source generalisation.

## 16.9 Reproducibility and random-seed considerations

All partitions are **deterministic from a fixed seed** (seed = 42), and the partition is **persisted** with
its seed and per-partition group memberships, so that the exact split is recoverable and independently
checkable (Chapter 14.10). For stability claims, the framework specifies **multiple seeds** (≥3) so that a
reported difference is shown to be robust to the particular partition, not an artefact of one split (Volume
I, §5.2). The justification is twofold: a single seed makes a result un-reproducible and potentially
cherry-picked, while seed-stability evidence distinguishes a real effect from partition noise. The seed
value itself is reported, not hidden, so that the result is neither tuned to nor obscured by the choice.

## 16.10 Chapter summary

This chapter developed the partitioning design. It derived four axioms — independence, sealing,
representativeness, reproducibility — and instantiated them as a cluster-grouped 80/10/10 SIPaKMeD split
(measured: 3,238 / 405 / 406 cells; zero field overlap; seed 42) with strictly separated train/validation/
test roles; a stratified-group compromise that prioritises group disjointness over exact class balance;
grouped cross-validation over train+validation with the test sealed; a deliberately leaky control that
measures the leakage tax; frozen, per-domain external and cross-dataset evaluation at the harmonised tiers
with reported exclusions; a leave-one-dataset-out design for multi-dataset learning (specified, deferred);
and seeded, persisted, multi-seed reproducibility. Each decision was justified by a validity or leakage
requirement, and the ordering of constraints (independence before representativeness) was made explicit.

## 16.11 Remaining challenges and transition to Chapter 17

The partitioning design fixes *how the data is divided*; it does not yet fix *how performance is defined,
measured, and reported* on those divisions. A leakage-safe split is necessary but not sufficient: the
evaluation must also specify the benchmark, the metrics, the statistical procedures, the reporting
standards, and the documentation and pre-registration discipline that make the resulting numbers
trustworthy and reproducible. **Chapter 17** therefore defines the overall experimental protocol —
benchmark definition, evaluation philosophy, reporting standards, FAIR and reproducibility practices,
statistical validity, transparency, pre-registration, and documentation — consolidating the data-foundation
commitments without describing implementation.

---

# Chapter 17 — Experimental Protocol

This chapter defines the experimental-data protocol that governs every empirical claim built on the data
foundation. It consolidates, at the data-foundation level, the benchmark definition, evaluation philosophy,
reporting standards, statistical-validity requirements, pre-registration discipline, FAIR/reproducibility
practices, and documentation standards. It refines and operationalises the pre-registered protocol stated
in Volume I (§4.6) for the specific datasets of this volume; it does not describe model implementation.

## 17.1 Benchmark definition

The protocol defines two complementary benchmarks. The **internal benchmark** is the SIPaKMeD
cluster-grouped sealed-test partition (Chapter 16.2): a leakage-safe, reproducible estimate of
within-dataset generalisation, reported with confidence intervals and against the majority baseline. The
**external benchmark** is the frozen-model, per-domain evaluation across the harmonised external suite
(Herlev, APACC, CRIC, HiCervix) at the E0/E1/E-morph tiers with reported excluded fractions (Chapters 11,
16.7). The two benchmarks are *jointly* the unit of evaluation: a result is characterised by the *pair*
(internal estimate, external estimates), and their difference — the internal-to-external gap — is itself a
reported quantity (Volume I, §2.19.5). Defining the benchmark as a pair, rather than a single internal
number, is the protocol's foundational commitment and the corrective to the field's single-number practice
(Chapter 13.3).

## 17.2 Evaluation philosophy

The evaluation philosophy is **multi-metric, per-domain, baseline-referenced, and reliability-first**
(Volume I, §§2.19.4, 5.8). Every model is characterised, on every domain, by a *vector* of quantities —
threshold-free discrimination, operating-point performance, calibration, and reliability/abstention — not a
scalar; each external domain is reported *separately* (never averaged into a single external number that
could mask per-domain failure); every operating-point metric is reported *against the domain's majority
baseline*, so that below-baseline behaviour is visible; and reliability (calibration, coverage, abstention)
is treated as a first-class reported property, not an afterthought. This philosophy is the data-foundation
expression of the reliability-first stance and is what makes the benchmark *diagnostic* rather than merely
comparative.

## 17.3 Metrics and reporting standards

The protocol fixes the reported metrics in advance. **Primary metrics**: balanced accuracy, macro-F1,
one-vs-rest AUROC, expected calibration error, and conformal coverage — chosen for robustness to the severe
external imbalance (Chapter 10.1) and to characterise discrimination, calibration, and reliability
separately. **Mandatory accompaniments**: the majority-class baseline and the excluded fraction, reported
alongside every external result, so that neither below-baseline performance nor silent exclusion can be
concealed (Volume I, §2.7). **Operating point**: a pre-registered threshold (τ = 0.5) for the binary tier,
with per-domain calibration reported; any oracle threshold is reported **only as an upper bound**, never as
a deployable result (Volume I, §4.6). The reporting standard is therefore that *no single number stands
alone*: discrimination, operating-point, calibration, baseline, and exclusion are reported together, per
domain.

## 17.4 Statistical validity

The protocol requires that every reported difference be both *of plausible magnitude* and *statistically
supported* (Volume I, §5.2). Concretely: **95% bootstrap confidence intervals** on every primary metric;
**McNemar's test** for paired classifier comparisons on the same test cells; the **Wilcoxon signed-rank
test** for paired metric differences; and **≥3 seeds** to establish stability of any reported effect. The
justification is that an estimate without an interval is not a measurement, and a difference without a test
is not a finding; in a field prone to over-claiming on single point estimates (Chapter 13.3), statistical
validity is a precondition for trustworthiness, not an embellishment.

## 17.5 Pre-registration discipline

The protocol is **pre-registered**: the research questions, the hypotheses, and their quantitative success
criteria (Volume I, §4.4) are fixed *before* any result is observed, and the evaluation protocol is fixed
*before* the experiments are run. The data foundation supports this by sealing the test partition and by
persisting the split, so that the conditions of the test cannot be adjusted to the result. A running
**hypothesis-status ledger** records each hypothesis's verdict against its pre-registered criterion, with
refutations reported as prominently as corroborations (Volume I, §6.5). Pre-registration is the structural
guarantee that the audit-then-remediate programme (Volume I, §6.1) cannot be biased by post-hoc
reinterpretation.

## 17.6 FAIR principles and reproducibility

The protocol adopts the FAIR principles for its artefacts (Wilkinson et al., 2016): the splits, seeds,
crosswalks, metric definitions, and per-figure data are released in machine-readable formats with
documentation (Findable, Accessible), the crosswalk and metric definitions are made explicit
(Interoperable), and the artefacts are documented sufficiently for reuse (Reusable). Reproducibility is
operationalised by the stringent test of Volume I (§5.4): headline results regenerable from the released
seed and split, and every figure reconstructable from its companion data file — a property this work
enforces by persisting all values to JSON/CSV (Chapter 14.8). FAIR/reproducibility is thus not a
post-publication courtesy but a property designed into the data foundation.

## 17.7 Transparency and documentation standards

The protocol aligns its documentation with emerging standards for trustworthy reporting: dataset
documentation in the spirit of *datasheets for datasets* (Gebru et al., 2021), model behaviour in the
spirit of *model cards* (Mitchell et al., 2019), and reporting aligned with checklists for AI in medical
imaging and for clinical prediction models (e.g., CLAIM, Mongan et al., 2020; TRIPOD-style reporting). The
**[unverified]** metadata fields (Chapter 8) are documented as missing rather than inferred, which is
itself a transparency commitment. The justification is that evidence intended for clinical and regulatory
audiences must be *intelligible and complete on its own terms* (Volume I, §5.9), and standardised
documentation is the mechanism that makes it so.

## 17.8 The protocol as the shared experimental substrate

The defining structural property of the protocol is that it is **identical across the diagnostic and
constructive phases** of the research (Volume I, §6.1): the same leakage-safe, multi-domain, multi-metric,
reliability-first benchmark that exposes failures (audit) is the benchmark against which interventions are
judged (remediate). This shared substrate is what makes the constructive claims *commensurable* with the
diagnostic ones — an intervention is credited only if it improves the *same* measured quantities, on the
*same* sealed test and external suite, under the *same* statistical discipline. The protocol is therefore
not a list of metrics but the *common currency* of the entire research programme.

## 17.9 Chapter summary

This chapter defined the experimental-data protocol. It established a *paired* benchmark — a leakage-safe
internal SIPaKMeD estimate and a frozen, per-domain external suite — with their gap as a reported quantity;
a multi-metric, per-domain, baseline-referenced, reliability-first evaluation philosophy; fixed primary
metrics (balanced accuracy, macro-F1, OvR-AUROC, ECE, conformal coverage) with mandatory majority-baseline
and excluded-fraction accompaniments and an upper-bound-only oracle threshold; statistical-validity
requirements (bootstrap CIs, McNemar, Wilcoxon, ≥3 seeds); a pre-registration discipline with a
hypothesis-status ledger; FAIR and stringent reproducibility practices; standardised transparency
documentation; and the principle that the protocol is the shared substrate across the audit and remediate
phases. Every standard was justified as a corrective to a documented weakness of prevailing practice.

## 17.10 Remaining challenges and transition to Chapter 18

The protocol specifies how to evaluate *validly*, but no protocol eliminates all threats to validity; some
are intrinsic to the data (unverified metadata, residual confounds, absent demographics) and must be
*acknowledged and bounded* rather than removed. A rigorous methodology states its own threats explicitly.
**Chapter 18** therefore conducts a systematic threats-to-validity analysis — internal, external,
construct, statistical, ecological, and clinical — stating for each the risks, limitations, mitigation
strategies, and remaining uncertainties, so that the data foundation's claims are read with their
limitations fully in view.

---

# Chapter 18 — Threats to Validity

This chapter conducts a systematic threats-to-validity analysis of the data foundation, following the
standard validity framework for empirical research (Shadish, Cook & Campbell, 2002). For each validity
type — internal, external, construct, statistical, ecological, and clinical — it states the risks,
limitations, mitigation strategies, and remaining uncertainties. The analysis is deliberately
self-critical: its purpose is to bound the claims of the data foundation honestly, consistent with the
no-fabrication and reliability-first principles. No implementation is described.

## 18.1 Internal validity

*Internal validity* concerns whether the internal (within-dataset) estimate is an unbiased measure of
within-dataset generalisation. **Risks:** data leakage (Chapter 13) — principally FOV, and residually
slide/patient leakage. **Mitigations:** group-aware partitioning at the field level with a sealed test and
asserted zero field overlap (Chapter 16.2); train-only fitting of data-dependent preprocessing; a
deliberately leaky control that *measures* the leakage tax (Chapter 16.6). **Remaining uncertainties:**
patient- and slide-level identifiers are **[unverified]** in SIPaKMeD, so grouping is at the field level
only; if multiple fields derive from one slide or patient, residual slide/patient leakage cannot be
provably excluded. Exact duplicates are removed, but **perceptual near-duplicate screening is not
automated** (Chapter 14.5), leaving a residual duplicate-leakage risk. These residuals are stated, not
hidden, and they bound the internal estimate from above (i.e., the true leakage-free number could be lower
still).

## 18.2 External validity

*External validity* concerns whether results generalise beyond the studied datasets. **Risks:** the
external suite, though diverse, is finite and may not represent all preparations, scanners, laboratories,
or populations. **Mitigations:** a deliberately diverse, preparation-spanning external suite (Chapter 9.4);
a *data-only* preparation-distance ordering as the generalisation explanatory variable (Chapter 12.9);
frozen, per-domain external evaluation (Chapter 16.7). **Remaining uncertainties:** **demographic shift is
not probed at all** (demographics **[unverified]**; Chapter 12.8), so population-level generalisation is
out of evidential reach; the suite probes acquisition and taxonomic shift, not demographic shift. One
external dataset (APACC) has **[unverified] provenance** and is therefore demoted to a supporting role
(Chapter 9.3). External validity is thus established along the acquisition/taxonomy axes and explicitly
*not* established along the demographic axis.

## 18.3 Construct validity

*Construct validity* concerns whether the measurements measure the intended construct. **Risks:** several
constructs are measured by proxies. The model is trained on **appearance** labels (SIPaKMeD) but is
evaluated, via the crosswalk, against **grade/risk** labels (Herlev/Bethesda) — appearance is not grade
(Chapter 11.2), so the binary tier measures *abnormality detectability*, not *grade diagnosis*. The
**morphometric characterisation** uses handcrafted features as a proxy for cell morphology (Chapter 10.6),
a lower bound on the information available. The **crosswalk** itself encodes mapping assumptions (e.g.,
Herlev columnar; ASC-US handling). **Mitigations:** the crosswalk is deterministic, documented, and
many-to-one, with excluded fractions reported and the one debatable mapping sensitivity-tested (Chapter
11.5); the tiers (E0, E1, E-morph) are defined to make explicit *what* is being measured at each level; the
structural high-grade blind spot is measured (E1) rather than concealed. **Remaining uncertainties:** the
binary tier cannot, by construction, validate grade diagnosis, and the indeterminate Bethesda categories
have no fully satisfactory mapping (Chapter 11.7). These are intrinsic to the appearance-vs-grade mismatch,
not removable defects.

## 18.4 Statistical validity

*Statistical validity* concerns whether the statistical inferences are sound. **Risks:** small external
datasets (Herlev: 917 cells, **[measured]**) yield wide interval estimates; severe external imbalance
(CRIC: 161 SCC of 11,534, **[measured]**) makes minority-class external estimates high-variance; many
metrics across many domains invite multiple-comparison error; and single-seed results may reflect partition
noise. **Mitigations:** 95% bootstrap confidence intervals on all primary metrics; McNemar and Wilcoxon
tests for paired comparisons; ≥3 seeds for stability; balanced, per-class, baseline-referenced metrics
robust to imbalance (Chapter 17.3–17.4). **Remaining uncertainties:** for the rarest categories (e.g.,
external high-grade/SCC), even interval-aware estimates are inherently imprecise due to small counts, and
this imprecision must be carried into the interpretation rather than rounded away.

## 18.5 Ecological validity

*Ecological validity* concerns whether the study setting reflects the deployment setting. **Risks:** the
curated single-cell datasets (SIPaKMeD, Herlev) over-represent representative, often unambiguous cells and
under-represent the borderline, overlapping, artefact-laden cells of consecutive clinical material (Chapter
10.9); and the single-cell classification task abstracts away the whole-slide acquisition, scanning, and
field-selection workflow in which a deployed system operates. **Mitigations:** the inclusion of field-level
(CRIC) and scanner-derived (HiCervix) datasets brings the evaluation closer to consecutive material than
curated cells alone; the assistive, human-in-the-loop framing (Volume I, §2.17) matches the realistic
deployment mode. **Remaining uncertainties:** whole-slide acquisition, scanning quality control, and
field selection are **out of scope** (Volume I, §1.15), so ecological validity is partial; performance on
curated cells may overstate performance on consecutive slides, a gap *additional* to the measured
domain-shift gap.

## 18.6 Clinical validity

*Clinical validity* concerns whether the measured quantities correspond to clinically meaningful outcomes.
**Risks:** cytological labels are *screening* categories, and the diagnostic reference standard is
histopathology, not cytology (Volume I, §2.1); the model has a **structural high-grade blind spot** (no
HSIL output; Chapter 11.7); and the binary tier reduces a nuanced management decision to normal/abnormal.
**Mitigations:** the framing as a *triage assistant* under expert oversight (not an autonomous diagnostic
device); the explicit E1 grade tier that *measures* the high-grade blind spot rather than hiding it; the
reliability-first emphasis on calibrated abstention so the system defers when uncertain. **Remaining
uncertainties:** without histopathological ground truth and prospective clinical validation (out of scope),
clinical validity is *evidentiary-preparatory*, not *established*: this work builds the evaluation and
reliability foundation for clinical validity but does not, and does not claim to, establish clinical
efficacy.

## 18.7 Consolidated threats-and-mitigations summary

**Table 18.1 — Threats to validity and their treatment.**

| Validity type | Principal threat | Mitigation | Residual (honestly stated) |
|---|---|---|---|
| Internal | FOV/slide/patient leakage; duplicates | field grouping, sealed test, leaky control, train-only preprocessing | patient/slide IDs unverified; perceptual dedup not automated |
| External | finite suite; unrepresented domains | diverse preparation-spanning suite; per-domain frozen eval | demographic shift unprobed; APACC provenance unverified |
| Construct | appearance↔grade mismatch; proxy features | deterministic documented crosswalk; explicit tiers; measured blind spot | binary tier ≠ grade diagnosis; ASC-US unmappable |
| Statistical | small/imbalanced external sets; multiplicity | bootstrap CIs; McNemar/Wilcoxon; ≥3 seeds; balanced metrics | rare-category external estimates imprecise |
| Ecological | curated vs consecutive cells; single-cell abstraction | field/scanner datasets; assistive framing | whole-slide workflow out of scope |
| Clinical | cytology vs histology standard; high-grade blind spot | triage framing; E1 measures blind spot; abstention | clinical efficacy not established (no prospective/histology) |

## 18.8 Chapter summary

This chapter analysed the threats to validity of the data foundation across six validity types. Internal
validity is secured against the addressable leakage forms but residually limited by unverified patient/slide
identifiers and un-automated perceptual deduplication. External validity is established along the
acquisition/taxonomy axes but explicitly not along the demographic axis, and is qualified by one
unverified-provenance dataset. Construct validity is bounded by the appearance-versus-grade mismatch, which
the documented crosswalk makes honest but cannot dissolve. Statistical validity is supported by intervals,
tests, and multi-seed stability, but rare external categories remain imprecise. Ecological validity is
partial, given curated-versus-consecutive and single-cell-versus-whole-slide gaps. Clinical validity is
evidentiary-preparatory, not established, absent histological ground truth and prospective validation. Each
threat was paired with its mitigation and its honestly stated residual, so that the data foundation's
claims are scoped precisely.

## 18.9 Remaining challenges and transition to Chapter 19

The threats analysis completes the critical examination of the data foundation. What remains is to
consolidate the volume — the selected datasets, their diversity, the taxonomy and domain-shift and leakage
analyses, the curation, harmonisation, partitioning, and protocol frameworks, and the threats to validity —
into a coherent summary, and to motivate the transition to Volume III, in which the methodology that this
data foundation supports will be developed. **Chapter 19** provides that summary and transition.

---

# Chapter 19 — Summary

This chapter consolidates Volume II — the dataset engineering and experimental foundation — and motivates
the transition to Volume III. It introduces no implementation.

## 19.1 Selected datasets

The volume established, by **direct measurement**, a training source and a complementary external
evaluation suite. **SIPaKMeD** (4,049 cropped single cells from 966 fields, with 16,196 nucleus/cytoplasm
contour files; five appearance classes) is the training source, selected for its single-cell taxonomy,
ground-truth contours, field structure (enabling leakage-safe splitting), and conventional-smear source
domain. The external suite comprises **Herlev** (917 graded cells with masks; near-domain, grade taxonomy),
the **APACC** subset (1,125 auto-segmented coarse-labelled crops; supporting role, unverified provenance),
**CRIC** (11,534 point-annotated Bethesda cells in 400 dense fields; high-grade categories), and
**HiCervix** (40,229 hierarchically labelled liquid-based cells; the far preparation domain). Every dataset
fact was reported with explicit `[measured]`/`[primary]`/`[unverified]` provenance, and no metadata was
invented.

## 19.2 Dataset diversity and the experimental instrument

The suite's measured diversity — across preparation (conventional smear, auto-segmented, dense fields,
liquid-based), image encoding (BMP, PNG, JPEG), cell scale (≈153–731 px crops; 1376-px fields),
segmentation procedure (contour, mask, point, auto), and label family (appearance, grade, Bethesda) — is
the volume's principal experimental asset. It is what makes the cross-domain and cross-taxonomy questions
*measurable*, and it is what makes naïve cross-dataset comparison invalid without the crosswalk and
leakage-safe protocol the volume develops. The morphometric characterisation added a measured, model-free
finding: the five classes overlap substantially (silhouette ≈0.09), with Koilocytotic least separable and
Parabasal most — a ranking that independently matches the deep model's per-class difficulty, evidencing
that difficulty is intrinsic to the morphology.

## 19.3 Taxonomy challenges

The volume established that the datasets' taxonomies are semantically heterogeneous — appearance, grade,
and risk-stratification — so that no information-preserving one-to-one map exists. It catalogued the
one-to-many (undefined), many-to-one (safe), ambiguous, and clinically unsafe mapping hazards, and adopted
a deterministic many-to-one crosswalk to E0/E1 tiers with explicit, counted exclusions and a
sensitivity-tested treatment of the one debatable mapping. Three taxonomic challenges remain genuinely
open: the indeterminate Bethesda categories, the glandular axis, and the structural high-grade blind spot —
acknowledged honestly rather than concealed.

## 19.4 Domain shift and leakage

The volume analysed domain shift at the data level, decomposing it into covariate, prior, and taxonomic
concept components, establishing each through measured proxies, and constructing a data-only
preparation-distance ordering as the pre-registered generalisation explanatory variable. It analysed ten
forms of data leakage, grounding the analysis in SIPaKMeD's measured cell-to-field multiplicity, and
established that the addressable forms (FOV, preprocessing, augmentation, hyperparameter, validation) are
eliminated by construction while the residual forms (patient/slide grouping, perceptual duplicates) are
honestly bounded — a footing the prevailing random-per-cell practice does not provide.

## 19.5 Curation, harmonisation, partitioning, and protocol

The volume specified a reproducible **curation** framework (counted inclusion/exclusion, measured
zero-corrupt integrity, flagged perceptual-dedup gap, provenance-first persistence); a **harmonisation**
framework that performs representational reconciliation (taxonomy, format, coordinate, pixel-resolution,
metadata) while *deliberately preserving* the stain, preparation, and physical-scale heterogeneity that is
the object of study; a **partitioning** design (measured cluster-grouped 80/10/10: 3,238/405/406 cells,
zero field overlap, seed 42; sealed test; leaky control; frozen per-domain external evaluation; a specified
leave-one-dataset-out design); and an **experimental protocol** defining a *paired* internal/external
benchmark, a multi-metric, per-domain, baseline-referenced, reliability-first evaluation philosophy,
statistical-validity requirements, pre-registration, FAIR/reproducibility, and standardised documentation,
as the *shared substrate* across the audit and remediate phases.

## 19.6 The experimental data foundation

Together these establish the experimental data foundation the framework requires: a curated,
provenance-tracked, leakage-safely partitioned training source; a harmonised, taxonomy-aligned,
multi-domain external suite; an honest, multi-metric, per-domain, pre-registered protocol; and a critical
threats-to-validity analysis that scopes every claim. The foundation's defining property is *honesty under
measurement*: its quantitative core is measured and reproducible, its unverifiable elements are flagged
rather than invented, and its residual limitations are stated rather than hidden.

## 19.7 Transition to Volume III

Volume II has built the data foundation without describing how any model is constructed or trained. With a
leakage-safe benchmark, an honest crosswalk, a characterised multi-domain suite, and a pre-registered
protocol now in place, the methodology that operates on this foundation can be developed. **Volume III —
Methodology** will specify the representation-learning, knowledge-distillation, calibration, abstention,
cross-taxonomy, and minority-class-rescue methods — the *implementation* deliberately withheld from Volumes
I and II — each designed to be evaluated under the identical leakage-safe, multi-domain, reliability-first
protocol established here. The data foundation is the fixed substrate against which those methods will be
measured; its rigour is what will make their evaluation trustworthy.

## 19.8 Chapter summary

This chapter consolidated the dataset engineering and experimental foundation: the measured selection of a
training source and a complementary external suite; the suite's measured diversity and the model-free
morphometric finding of intrinsic class overlap; the taxonomic incompatibilities and the deterministic
crosswalk that makes cross-dataset comparison honest; the data-level analyses of domain shift and leakage;
the curation, harmonisation, partitioning, and protocol frameworks; and the threats-to-validity analysis.
It established that the foundation is honest under measurement and motivated the transition to Volume III,
in which the methodology this foundation supports will be developed and evaluated against it.

---

# References (Volume II)

*Consolidated for Volume II; all entries are genuine works shared with the Volume I bibliography and the
accompanying `references.bib`, with the single addition of Shadish, Cook & Campbell (2002) for the
threats-to-validity framework. No reference is fabricated. Dataset-metadata claims marked **[unverified]**
in the text are not citations and require confirmation against primary documentation before submission.*

- Cour, T., Sapp, B., & Taskar, B. (2011). Learning from partial labels. *Journal of Machine Learning
  Research*, 12, 1501–1536. **[verify]**
- Gebru, T., Morgenstern, J., Vecchione, B., et al. (2021). Datasheets for datasets. *Communications of
  the ACM*, 64(12), 86–92.
- Geirhos, R., Jacobsen, J.-H., Michaelis, C., et al. (2020). Shortcut learning in deep neural networks.
  *Nature Machine Intelligence*, 2, 665–673.
- Jantzen, J., Norup, J., Dounias, G., & Bjerregaard, B. (2005). Pap-smear benchmark data for pattern
  classification (Herlev dataset). *Proceedings of NiSIS*. **[verify venue/pages]**
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based
  science. *Patterns*, 4(9), 100804.
- Macenko, M., Niethammer, M., Marron, J. S., et al. (2009). A method for normalizing histology slides for
  quantitative analysis. *IEEE International Symposium on Biomedical Imaging (ISBI)*.
- Mitchell, M., Wu, S., Zaldivar, A., et al. (2019). Model cards for model reporting. *Conference on
  Fairness, Accountability, and Transparency (FAT\*)*.
- Mongan, J., Moy, L., & Kahn, C. E. (2020). Checklist for Artificial Intelligence in Medical Imaging
  (CLAIM). *Radiology: Artificial Intelligence*, 2(2), e200029.
- Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V., & Herrera, F. (2012). A unifying
  view on dataset shift in classification. *Pattern Recognition*, 45(1), 521–530. **[verify]**
- Nayar, R., & Wilbur, D. C. (Eds.). (2015). *The Bethesda System for Reporting Cervical Cytology* (3rd
  ed.). Springer.
- Plissiti, M. E., Dimitrakopoulos, P., Sfikas, G., Nikou, C., Krikoni, O., & Charchanti, A. (2018).
  SIPaKMeD: A new dataset for feature and image based classification of normal and pathological cervical
  cells in Pap smear images. *IEEE International Conference on Image Processing (ICIP)*.
- Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (Eds.). (2009). *Dataset Shift
  in Machine Learning*. MIT Press.
- Reinhard, E., Ashikhmin, M., Gooch, B., & Shirley, P. (2001). Color transfer between images. *IEEE
  Computer Graphics and Applications*, 21(5), 34–41.
- Shadish, W. R., Cook, T. D., & Campbell, D. T. (2002). *Experimental and Quasi-Experimental Designs for
  Generalized Causal Inference*. Houghton Mifflin.
- Stacke, K., Eilertsen, G., Unger, J., & Lundström, C. (2021). Measuring domain shift for deep learning in
  histopathology. *IEEE Journal of Biomedical and Health Informatics*, 25(2), 325–336. **[verify]**
- Stoler, M. H., & Schiffman, M. (2001). Interobserver reproducibility of cervical cytologic and histologic
  interpretations (ALTS). *JAMA*, 285(11), 1500–1505.
- Tellez, D., Litjens, G., Bándi, P., et al. (2019). Quantifying the effects of data augmentation and stain
  color normalization in convolutional neural networks for computational pathology. *Medical Image
  Analysis*, 58, 101544.
- Varoquaux, G., & Cheplygina, V. (2022). Machine learning for medical imaging: Methodological failures and
  recommendations for the future. *npj Digital Medicine*, 5, 48.
- Wilkinson, M. D., Dumontier, M., Aalbersberg, I. J., et al. (2016). The FAIR Guiding Principles for
  scientific data management and stewardship. *Scientific Data*, 3, 160018.

---

*End of Volume II — Dataset Engineering & Experimental Foundation.*

