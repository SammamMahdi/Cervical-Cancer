# Phase 5 — Taxonomy of AI Methods for Cervical Screening & Evidence-Based Landscape Review

**Evidence base:** 155 verified papers (`Literature/CSV/papers.csv`). ⟨V⟩ = verified from source; ⟨DB⟩ = counted from our database; ⟨OWN⟩ = measured in this repository.

---

## 1. Taxonomy

```
AI for cervical cancer screening
│
├── A. IMAGE-ONLY
│   ├── A1 Single-cell cytology classification ....... SIPaKMeD/Herlev; CNN → ViT/Swin era
│   ├── A2 Whole-slide (WSI) / MIL aggregation ....... slide-level rare-event detection
│   ├── A3 Segmentation / de-overlapping ............. ISBI14/15, Cx22; DoNet
│   ├── A4 Colposcopy & cervigram .................... incl. NCI AVE
│   └── A5 Histopathology ............................ H&E WSI subtyping
│
├── B. CLINICAL-RISK (TABULAR)
│   └── B1 UCI 383 risk-factor models ................ RF/XGB/stacking (+SMOTE)  ⚠ leakage-dominated
│
├── C. MULTIMODAL
│   ├── C1 TRUE cross-modal (image + non-image clinical/EHR)
│   ├── C2 Multi-VIEW relabelled "multimodal" ........ ⚠ same organ, different captures/derivatives
│   └── C3 Fusion strategy: early | joint | late/decision | attention/transformer
│
├── D. MISSING-MODALITY LEARNING  (no cervical instance exists)
│   ├── D1 Generative/imputation .................... requires a cohort to impute from
│   ├── D2 Shared-specific decomposition ............ ShaSpec
│   ├── D3 Prior/prototype reconstruction ........... SMIL
│   ├── D4 Prompt/token conditioning ................ Lee et al.; Gu et al.
│   ├── D5 Marginalisation / vacuous evidence ....... ProbEn; Dempster identity
│   └── D6 Routing / MoE ............................ Flex-MoE
│
├── E. UNCERTAINTY & CALIBRATION
│   ├── E1 Post-hoc calibration ..................... temperature/Platt/isotonic; BCTS+MLLS
│   ├── E2 Evidential / subjective logic ............ EDL, TMC, RCML  (u = relative, not absolute)
│   ├── E3 Conformal / distribution-free ............ Mondrian CP  (marginal only)
│   └── E4 Selective prediction / learn-to-defer .... Chow; Mozannar & Sontag
│
├── F. FOUNDATION MODELS
│   ├── F1 Cervical-specific ........................ UniCAS; Smart-CCS   (image-only)
│   └── F2 General pathology ........................ UNI, CONCH, Virchow, Prov-GigaPath, TITAN
│                                                      (2nd modality = TEXT, never structured variables)
│
├── G. DEPLOYMENT / TRUST
│   ├── G1 Domain shift & external validation
│   ├── G2 Explainability (Grad-CAM/SHAP/LIME)
│   ├── G3 Human-AI collaboration / reader studies
│   └── G4 Federated learning / privacy
│
└── H. COMPOSITION THEORY (imported, not native to cervical AI)
    ├── H1 Classifier combination ................... Kittler 1998; Genest & Zidek 1986
    ├── H2 Prior/prevalence shift ................... Saerens 2002; Godau 2023; Alexandari 2020
    ├── H3 Statistical matching / data fusion ....... ~50 yrs; non-identified w/o cond. independence
    └── H4 No-gold-standard / latent class .......... Rutjes 2007
```

---

## 2. Landscape review by category

### A — Image-only screening
**Evolution.** Hand-crafted features (Herlev era, 2005) → transfer-learned CNNs (2018–2022) → ViT/Swin hybrids and ensembles (2023–2025) → **cervical foundation models** (2025–2026).
**Advantages.** Mature; strong reported numbers; edge-deployable after distillation.
**Limitations (evidence-backed).** The headline numbers **do not transfer**: CerviPro C-index 0.806 internal → **AUC ≈ 0.63 external**; Multi-Modal GNN **−9%** on real-world Apollo Hospitals data ⟨V⟩. Our own model: **0.998 in-distribution → 0.73 / 0.59 / 0.53** on Herlev / HiCervix / APACC ⟨OWN⟩. Liu et al. (eClinicalMedicine 2024, 77 studies) pool AUC **0.99** while stating prospective evidence is lacking ⟨V⟩. Only **one** SIPaKMeD↔Herlev transfer matrix exists (Coskun 2026) ⟨V⟩. SIPaKMeD/Herlev publish **no patient IDs** ⇒ patient-wise splitting impossible **in principle**.
**Clinical readiness.** Moderate for triage-assist (Xue et al. 2023 deploy a cytologist-in-the-loop threshold); low for autonomy.
**Open problems.** Calibration under shift (**never measured**); slide-level rare-event detection; prospective validation.

### B — Clinical-risk (tabular)
**Evolution.** UCI 383 (Fernandes et al. 2017) became the field's near-universal substrate; methods drifted to stacking + SMOTE.
**Limitations.** **Leakage-dominated.** Published 0.98–1.00 AUC (e.g. Roy et al. 2025: 0.98 acc / **0.995 AUC**, SMOTE, no external validation ⟨V⟩) versus the **honest anchor from the dataset's own creators** (Fernandes, Chicco et al., PeerJ CS 2018 — stratified 10-fold with **nested** inner CV, **no oversampling**) ⟨V⟩. Our leakage-safe replication: **Biopsy 0.656 [0.584, 0.720]**; `Dx:Cancer` from other `Dx` columns **0.998** vs **0.592** from legitimate features; 8-family tuned ceiling **≈0.65**; best single feature AUROC **0.572** ⟨OWN⟩.
**Clinical readiness.** **Low.** Single Venezuelan cohort (n=858, 2012–13), **zero external validation anywhere in this literature** ⟨V⟩.
**Open problems.** A leakage audit (none exists); calibration (none exists on UCI at all); any external cohort.

### C — Multimodal cervical screening
**Evolution.** 2024–2026: transformer fusion (TopotiCerv), GNN fusion (Multi-Modal GNN), CNN-transformer-GRU (CerviFusionNet), radiomics+DL (ultrasound).
**Critical structural finding ⟨V⟩.** Two senses of "multimodal" are conflated: **true cross-modal** (image + non-image clinical: Xu 2016, Li 2024, Madathil 2025, TopotiCerv 2026, CerviPro 2025) vs **multi-view relabelled multimodal** (CerviFusionNet's four colposcopy captures; Multi-Modal GNN's image + its own mask + its own graph; Niu 2025's radiomics + deep features from the *identical* region). **In the latter a missing modality cannot arise by construction** — which partly explains why the field never confronted the problem.
**Limitations.** **0/10 support missing-modality inference**; fusion depends on complete, matched, co-registered modalities; calibration essentially absent (TopotiCerv claims "calibrated" **with no method named**; Madathil claims "better calibrated" **with no metric**) ⟨V⟩. **Fusion can hurt**: Kelebet et al. (2025) image+tabular late fusion **92%** vs tabular alone **96%** ⟨V⟩.
**Clinical readiness.** Low. **No reader study exists on multimodal cervical AI** — no clinician has ever been shown an output combining an image verdict with a questionnaire verdict ⟨V⟩.

### D — Missing-modality learning
**Evolution.** HeMIS (2016) → SMIL (2021) → mmFormer (2022) → ShaSpec/MMANet/prompts (2023) → Flex-MoE (2024) → surveys (Wu et al., TMLR).
**Advantages.** Mature, diverse mechanisms; strong results on BraTS/MM-IMDb/ADNI/MIMIC.
**Limitations.** **Every canonical method needs a paired anchor** (teacher, shared encoder, priors, router, or pretraining triplets) — the dependency is *architectural* ⟨V⟩. **Calibration reported: 0/10** ⟨DB⟩. **No cervical instance exists at all.** The field's own survey lists six open issues and **neither calibration-under-missing-modality nor the disjoint-cohort setting is among them** ⟨V⟩.
**Clinical readiness.** N/A for cervical (nonexistent).
**Open problems.** Disjoint cohorts; whether confidence *should* degrade when a modality is absent (**nobody asks**).

### E — Uncertainty & calibration
**Advantages.** EDL/TMC give an explicit uncertainty mass; conformal gives distribution-free guarantees; deferral is clinically proven.
**Limitations (sharp).** Evidential *u* is **"relative (rather than absolute)"** (Jürgens, ICML 2024); independently trained branches' uncertainties are **not on a common scale** (TMUR 2026, ECE 26.50%→18.23%) ⟨V⟩. Conformal guarantees are **marginal** and **do not compose**; Podkopaev & Ramdas's label-shift fix needs **unlabeled target data** ⟨V⟩. Saerens's prior correction presupposes calibration, "**not true of modern neural networks**" (Alexandari, ICML 2020) ⟨V⟩.
**Counter-evidence to note.** **AECF** already enforces *and measures* monotone confidence across all 2^M−1 subsets (ECE 0.011→0.009 full; 0.020 at 50% dropout) — on **paired, non-medical** data ⟨V⟩.
**Clinical readiness.** Deferral: proven (Xue 2023) but **threshold-based, uncalibrated, modality-blind**.

### F — Foundation models
**Finding ⟨V⟩.** **No cytology/pathology FM ingests a structured risk vector.** Where FMs are "multimodal", modality #2 is **always free text**. The two cervical FMs are image-only: **UniCAS** (48,532 WSIs, slide AUC 92.60%, **CC BY 4.0**) and **Smart-CCS** (127,471 WSIs / 48 centres, internal AUC 0.965, external 0.950, **Apache-2.0**). **ALTER** is the sole any-to-any framework ("attention is computed over the available subset… without requiring imputation") — but pretrains on **6,850 paired** TCGA triplets.
**Licensing inverts the ranking:** UNI/CONCH CC-BY-NC-ND gated; Virchow non-commercial; **Prov-GigaPath: "Any deployed use case — commercial or otherwise — is out of scope."** The **cervical** models are the most permissive.

### G — Deployment / trust
**Explainability.** **Jabbour et al. (JAMA 2023, n=457 clinicians):** image-based AI explanations **did not mitigate** the harm of a systematically biased model ⟨V⟩. Nothing attributes a decision **across** modalities.
**Human-AI.** Every verified human-AI cervical record is **cytology-image-only**.
**Federated learning.** The credible route to large, diverse, privacy-compliant data, but current results are weak (68.72% ResNet50 via Flower; 94.36% IID vs **78.4% non-IID**) ⟨V⟩. **Note FL is structurally the inverse of our problem** (shared feature space, distributed samples).

### H — Composition theory (imported)
Cervical AI has never used it. It splits cleanly: **H2 retargets one classifier to one population; H1 combines several under a common prior.** Nobody crosses. That crossing is our Phase-6 gap **G3** and the sole novelty of the proposal.

---

## 3. Cross-cutting landscape findings

| Finding | Evidence |
|---|---|
| **Benchmarks are saturated and non-transferable** | pooled AUC 0.99 (77 studies) vs 10–45 pt external collapse; only 1 transfer matrix ⟨V,OWN⟩ |
| **Calibration is a non-dimension** | 0/10 × 3 clusters; 2/45 in-house; Yin et al.'s 2026 benchmark **omits** it ⟨DB,V⟩ |
| **Uncertainty is absent from cervical AI** | 0/45 in-house corpus report an uncertainty method ⟨DB⟩ |
| **"Multimodal" is frequently multi-view** | ⇒ missing modality cannot arise by construction ⟨V⟩ |
| **No cervical system degrades gracefully** | 0/10 missing-modality support ⟨V⟩ |
| **The tabular arm is a leakage artefact** | 0.98–1.00 published vs 0.656 honest ⟨V,OWN⟩ |
| **The paired cohort does not exist** | verified dataset-by-dataset; the UCI 383/384 near-miss is decisive ⟨V⟩ |

---

## 4. Readiness summary

| Category | Maturity | Clinical readiness | Biggest open problem |
|---|---|---|---|
| A Image-only | high | moderate (triage-assist) | calibration under shift; prospective validation |
| B Tabular risk | low (inflated) | **very low** | leakage audit; external cohort |
| C Multimodal cervical | emerging | low | needs complete paired modalities; no reader study |
| D Missing-modality | high (general) / **nil (cervical)** | n/a | paired-anchor dependency; disjoint cohorts |
| E Uncertainty/calibration | high (general) / **nil (cervical)** | low | non-comparable scales; marginal-only guarantees |
| F Foundation models | high | moderate | **no structured-variable input**; licences |
| G Deployment/trust | low | low | explanation ≠ benefit; no multimodal reader study |
| H Composition theory | mature elsewhere | n/a | **never crossed into fusion (our G3)** |
