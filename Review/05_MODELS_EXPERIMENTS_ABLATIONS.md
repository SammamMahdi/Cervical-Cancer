# Phases 12–14 — Model Selection, Experimental Design, Ablations, Statistical Plan

---

## Phase 12 — Model selection

**Hardware assumption throughout:** a single **RTX 5060 Ti (16 GB VRAM)**, Windows, Python 3.12 / PyTorch cu128 (the machine this repository runs on). All recommendations must fit that budget; anything that does not is marked ✗.

### 12.1 Image experts

| Model | Licence | VRAM (infer / train) | Clinical suitability | Advantages | Limitations |
|---|---|---|---|---|---|
| **EfficientNet-B0/B4** *(recommended baseline)* | Apache-2.0 (torchvision) | ~1 / ~8 GB | ✔ edge-deployable | proven on SIPaKMeD; distillable; tiny | not a foundation model; no FM priors |
| **Smart-CCS** (arXiv:2502.09662) *(recommended FM)* | **Apache-2.0, weights released** | ~4–8 GB | ✔ cervical-specific | **127,471 WSIs / 48 centres**; internal AUC **0.965**, external **0.950** — the only cervical FM with strong external numbers *and* a permissive licence | WSI/slide-level; image-only; **no missing-modality mechanism, no calibration** |
| **UniCAS** (Cell Reports Medicine 2026) | **CC BY 4.0** | ~4–8 GB | ✔ cervical-specific | 48,532 TCT WSIs / 80.3M patches; slide AUC **92.60%** | image-only; no calibration; no risk-vector input |
| UNI / CONCH | **CC-BY-NC-ND, gated** | ~6–10 GB | ⚠ non-commercial | strong pathology FMs | licence blocks deployment; **second modality is text, never structured variables** |
| Virchow | non-commercial | ~10–14 GB | ⚠ | ViT-H/14, 632M params; pan-cancer AUC 0.950 | non-commercial; large |
| Prov-GigaPath | ✗ **"Any deployed use case — commercial or otherwise — is out of scope"** | ~12+ GB | ✗ | strong WSI FM | **licence forbids deployment** |

> **Licensing inverts the obvious ranking:** the two *cervical* models (Smart-CCS Apache-2.0; UniCAS CC BY 4.0) are the **most permissive**, while the famous pathology FMs are the most restricted. ⟨V⟩

### 12.2 Tabular experts

| Model | Licence | Hardware | Advantages | Limitations |
|---|---|---|---|---|
| **HistGradientBoosting / XGBoost** *(recommended)* | BSD / Apache-2.0 | CPU, seconds | native NaN; best-in-class on small tabular (Grinsztajn et al., NeurIPS 2022); trivially frozen; stable calibration under imbalance | needs explicit imbalance handling |
| Logistic regression (L2) | BSD | CPU | interpretable; well-behaved probabilities | measured **AUROC 0.580** on Biopsy ⟨OWN⟩ — worse than trees |
| TabPFN | Apache-2.0 | ~2–4 GB | no training loop | "well-calibrated out of the box" claim **fails at our base rates** (4–12%); still needs imbalance correction + recalibration ⟨V⟩ |

**Measured bake-off (nested CV, tuned, 8 families)** ⟨OWN⟩ — best per target: **Biopsy kNN 0.654**, Schiller kNN 0.639, any-of-4 kNN 0.611. All families land in **0.54–0.65** with heavily overlapping CIs; kNN's edge over HistGB (**0.626**) is **within noise**. → **Recommendation: HistGradientBoosting** for robustness (native NaN, stable calibration), not kNN.

### 12.3 Composition / calibration / UQ stack (all CPU, closed-form)

| Component | Choice | Why |
|---|---|---|
| Calibration | **BCTS + MLLS** (Alexandari, ICML 2020) | Saerens EM presupposes calibration, "which is not true of modern neural networks" |
| Composition | prior-stripped LLR pool (M4→M6) + **Kittler sum rule** as robust comparator | product rule is high-variance at 35–74 positives |
| UQ | Mondrian split conformal (per expert) | distribution-free, **marginal**, fits per-cohort |
| Evidential | TMC-style *u* — **ranking only** | Jürgens (ICML 2024): epistemic *u* is "relative (rather than absolute)" |

### 12.4 Explicitly **not** recommended
✗ Joint co-trained multimodal transformer — **no paired data exists** (absence #1).
✗ Any FM as the tabular path — **no FM ingests a structured risk vector** (absence #7).
✗ Pseudo-pairing / chimeric users — see risk R2.

---

## Phase 13 — Experimental protocol

### 13.1 Datasets

| Role | Dataset | Size | Split rule |
|---|---|---|---|
| Image train/val/test | **SIPaKMeD** | 4,049 cells / 966 cluster images, 5 classes | **group-aware by source cluster image**, 70/15/15 |
| Image external | **Herlev** (917 cells, 7 cls), **HiCervix** (40,229 cells / 4,496 WSIs), **APACC** (103,675 cells / 107 smears), **CRIC** (400 img / 118 pts) | — | **never trained on**; external only |
| Tabular | **UCI 383** (858 pts, 36 attrs, 4 targets, CC BY 4.0) | — | **stratified nested CV** (outer = estimate, inner = tune + calibrate) |
| Paired | **none exists** | — | **absence #1 — this is a result, not an omission** |

> ⚠ **SIPaKMeD and Herlev release no patient IDs**, so patient-wise splitting is **impossible in principle**; group-aware-by-cluster-image is the strongest available approximation and this limitation must be stated in any results table. ⟨G8⟩

### 13.2 Protocol per RQ

| RQ | Experiment | Primary metric | Guard |
|---|---|---|---|
| **RQ1** | Leakage-safe UCI re-benchmark; 2-factor leakage ablation (`Dx:*` in/out × resampling before/inside folds) | AUROC + BCa CI; ECE; PR-AUC | resampling **inside folds only**; report base rate |
| **RQ2** ★ | Prior-stripping vs (i) raw pooling, (ii) single post-hoc correction of the pooled output, over τ ∈ {0.02, 0.06, 0.10, 0.20} | **ΔECE**, Δlog-loss | **H2c is the load-bearing test — per-expert, single-cohort, needs no paired data** |
| **RQ3** | Dependence-sensitivity envelope over ρ ∈ [−1,1] at the **real** marginals; signed calibration error | ΔAUROC vs max(single); signed ECE | envelope is **model-based** → label as a bound, not a measurement |
| **RQ4** | Confidence monotonicity across availability patterns; per-path ECE at shifted τ | monotonicity violations; ECE | **cite AECF**; claim only the medical/disjoint instance |
| **RQ5** | Enumerate verifiable / boundable / unfalsifiable properties; cost a minimal prospective paired pilot | — | the deliverable is the taxonomy itself |

### 13.3 Baselines (mandatory comparators)

1. **Image-only** (external AUROC — *not* in-distribution).
2. **Tabular-only** (leakage-safe).
3. **`max(image-only, tabular-only)`** ← **the falsifier baseline**.
4. Raw posterior pooling (no prior-stripping) — the ablation that isolates M4.
5. Naive-Bayes/ProbEn with a **common assumed prior** — the published rule.
6. Kittler **sum rule**.
7. **Leaky reproduction** (SMOTE-before-split + `Dx:*`) — to reproduce the literature's 0.98–1.00 and show it is an artefact.

### 13.4 Evaluation metrics

- **Discrimination:** AUROC, **PR-AUC** (mandatory at 4–12% prevalence), sensitivity @ fixed specificity.
- **Calibration:** **ECE, Brier, reliability diagrams, signed calibration error** (direction matters for H3a).
- **Selective:** risk–coverage curve, conformal coverage per path, deferral rate.
- **Clinical:** sensitivity at a screening-relevant operating point; **abstention rate vs plausible cytologist workload**.
- **All with BCa bootstrap 95% CIs (10,000 resamples).** No point estimates without intervals.

### 13.5 OOD / robustness / shift
- Cross-dataset transfer matrix (SIPaKMeD ↔ Herlev ↔ HiCervix ↔ APACC) — note **only one such matrix exists in the literature** (Coskun 2026) ⟨G9⟩.
- **Calibration under shift** — never measured in cervical AI ⟨G9⟩; we measure ECE on every external set.
- Input corruption (noise/blur/JPEG; skipped questionnaire items).
- Missing-modality: every availability pattern (`{img}`, `{tab}`, `{both}`), verifying `{img}` ≡ image-only path exactly.

---

## Phase 14 — Ablation study

| # | Ablation | Isolates | Expected/notes |
|---|---|---|---|
| A1 | Image branch: B0 vs B4 vs Smart-CCS; distilled vs not | image capacity | external AUROC is the only meaningful axis |
| A2 | Tabular branch: 8 families × tuned | tabular ceiling | measured: all 0.54–0.65 ⟨OWN⟩ |
| A3 | **`Dx:*` in vs out** | target leakage | measured: 0.998 → 0.592 ⟨OWN⟩ |
| A4 | **Resampling before split vs inside folds** | resampling leakage | expected to reproduce ≈0.98–1.00 |
| A5 | Target choice: Biopsy / Schiller / Citology / Hinselmann / any-of-4 | estimand | Citology 0.488 & Hinselmann 0.450 are **≤ chance** ⟨OWN⟩ → excluded |
| **A6** ★ | **Prior-stripping ON vs OFF** | **M4 — the novel component** | **the core ablation; if ΔECE ≈ 0 the contribution dies (H2b)** |
| A7 | π_k mis-specification: sweep declared π_img, π_tab ±50% | sensitivity to the prior estimate | π for a curated cohort is a construction artefact |
| A8 | τ sweep ∈ {0.02, 0.06, 0.10, 0.20} | deployment prior | the fused number is meaningless without τ |
| A9 | Composition rule: product/ProbEn vs sum rule vs weighted | rule choice | sum expected more robust (Kittler) |
| A10 | Reliability weights `w_k`: equal vs AUROC-derived vs off | weighting | |
| A11 | Calibration: none / temperature / **BCTS+MLLS** / isotonic | M3 | Alexandari: naive prior-correction on uncalibrated p is invalid |
| A12 | Conformal α ∈ {0.05, 0.1, 0.2}; per-path vs pooled | M8 | pooled = the error we refuse to make |
| A13 | Availability pattern × dependence ρ | **interaction** | does composition help *more* or *less* when one modality is absent? |
| A14 | Evidential *u* as weight vs as ranking-only | G6 | expected: as-weight degrades (TMUR) |

**Interaction effects explicitly studied:** A6×A7 (does prior-stripping help *because* π is right, or is it robust to mis-specification?), A6×A9 (does stripping matter more for product than sum?), A13 (availability × dependence).

---

## Deliverable 18 — Statistical analysis plan

- **Pre-registration.** RQs, hypotheses, primary metric per hypothesis, and the **global falsifier** are fixed before fitting (Phase 9). Any post-hoc analysis is labelled exploratory.
- **Primary comparisons.**
  - AUROC differences (fused vs `max(single)`): **DeLong test** (DeLong et al., Biometrics 1988) for correlated ROCs on the same cases; BCa bootstrap where DeLong's assumptions fail.
  - ECE differences: **paired bootstrap** (10,000 resamples) with BCa intervals.
  - Leakage attribution: two-factor ablation with ΔAUROC per factor + CI.
- **Multiplicity.** Primary family = {H2a, H3b} → **Holm–Bonferroni**. All other hypotheses are secondary and reported with unadjusted CIs, labelled as such.
- **Variance.** Repeated stratified CV (≥5 repeats × 5 folds) with seeds recorded; report mean ± SD across repeats **and** the bootstrap CI on pooled OOF (they answer different questions).
- **Power / precision.** With 55 Biopsy positives, the AUROC CI half-width is ≈0.07 (observed: [0.584, 0.720]). **We therefore pre-declare that differences < 0.05 AUROC on the tabular arm are not resolvable** and will not be claimed.
- **Effect sizes over p-values.** Every claim reports an effect size with CI; p-values only where a formal test is specified above.
- **Negative results are reported.** The falsifier makes a null outcome a publishable finding (precedent: Kelebet et al. 2025).
