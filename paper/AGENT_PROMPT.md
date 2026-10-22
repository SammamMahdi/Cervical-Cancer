# AI-Agent Prompt — write / extend the CytoBridge paper

Copy everything below the line into an AI coding agent that has the
`research-proposal-latex` skill (VEGA house style) available and access to this
repository. It is a complete, self-contained brief: the honest framing, the three
methodology parts (image, tabular, fusion), every real number, the figure/table
inventory with file paths, the placeholders for missing results, and the verified
citation list. A first compile-clean draft already exists at `paper/main.tex`
(10 pages, 0 errors / 0 undefined / 0 overfull hbox, Tectonic); use this brief to
extend, deepen, or regenerate it.

---

## ROLE & FORMAT
You are typesetting a completed-work B.Sc. thesis paper (not a proposal) in the
**VEGA house style** via the `research-proposal-latex` skill. Load
`references/{voice-and-wording,preamble,structure,section-by-section-guide,style-and-colors,diagrams}.md`.
Use `\documentclass[sigplan,nonacm,screen]{acmart}`, the exact preamble (callout
boxes `defbox/exbox/keybox/gapbox`; black-on-white TikZ), BibTeX with
`ACM-Reference-Format`. Compile with `tectonic -X compile main.tex` in `paper/`
(binary at `~/tectonic_bin/tectonic.exe`); iterate to **0 errors, 0 undefined,
0 overfull hbox**. `\setlength{\emergencystretch}{3em}` is already in the preamble and
clears the small paragraph overfulls; keep the title two lines.

## NON-NEGOTIABLES
1. **Never fabricate** a citation, number, or result. Every number below is measured
   in this repo (`Tabular results/NHANES/metrics/*.json`, `Legacy_src/results/metrics/*.json`).
2. **Honest novelty.** Only **CO-PRIOR** and the **leakage-audited re-benchmark** are new;
   distillation, prior-shift, sum/product combiners, and conformal prediction are
   **borrowed** and must be named as such.
3. **Pre-registered falsifier, reported honestly:** symmetric fusion does NOT beat the
   best single expert in the realistic (positive-dependence) regime.
4. **Explain everything:** every equation gets a "Reading Eq. term by term" note; every
   hard idea an `exbox` analogy; define each term on first use.
5. **Plain language.** A CS grad with basic ML must follow the fusion section.

## PAPER STRUCTURE (sections)
Abstract → 1 Introduction (1.1 defbox+exbox background, 1.2 problems, 1.3 gap+idea,
1.4 contributions) → 2 Related Work (cytology CNNs / questionnaire-leakage / fusion,
each ending in a `gapbox`) → 3 Methodology (3.1 overview+honest-positioning keybox +
borrowed/new table + **architecture figure**; 3.2 Image Expert; 3.3 Tabular Expert;
3.4 Fusion) → 4 Results (4.1 image, 4.2 tabular, 4.3 fusion, 4.4 explainability) →
5 Threats to Validity & Limitations → 6 Conclusion → References.

---

## CONTENT BRIEF (all numbers are real)

### The one-paragraph thesis
Two frozen experts fused at the decision level. The honest contributions are (1) a
leakage-audited questionnaire HPV-risk re-benchmark with an **empirical ceiling
≈0.71 AUROC** (not the field's 0.95), with the inflation **reproduced** (0.71→0.90);
(2) **CO-PRIOR**, prior-stripping before pooling (restores calibration, ECE 0.20→0.002);
(3) a **falsifier**: symmetric fusion does not beat the best single expert when the
experts are positively dependent (they share HPV as a cause). Value is operational
(any-modality degradation, conformal abstention, safety-asymmetric cascade,
calibration), not a joint-accuracy gain.

### METHODOLOGY 3.2 — Image expert (how it was trained + results)
- **Data:** SIPaKMeD~\cite{plissiti2018}, 4,049 single-cell images, 5 classes
  (Dyskeratotic, Koilocytotic = abnormal; Metaplastic, Parabasal,
  Superficial–Intermediate = normal). 380×380, ImageNet-normalised.
- **Teacher–student KD~\cite{hinton2015}:** teacher EfficientNet-B4~\cite{tan2019efficientnet}
  two-stage (30 ep frozen head, lr 1e-3 → 50 ep fine-tune, lr 1e-5, ReduceLROnPlateau
  0.3); student EfficientNet-B0, 40 ep, lr 5e-5, KD loss with **T=4.0, α=0.1**
  (0.1 hard CE + 0.9 soft KL). BF16 on RTX 5060 Ti, seed 42, batch 32. Include the KD
  equation + "Reading the equation". **Flag the leaky random 80/20 split honestly.**
- **Deployment output:** p_img = P(Koilocytotic)+P(Dyskeratotic).
- **Results (Table):** Teacher in-dist bal-acc 0.946, AUROC 0.995, ECE 0.013. Student
  in-dist bal-acc 0.968, AUROC 0.998, ECE 0.016 (student edges teacher — expected under
  KD). **External binary-abnormal transfer (the honest number): Herlev AUROC 0.728**
  (bal-acc 0.651, sens 0.625, spec 0.678, **ECE 0.299** = OOD miscalibration);
  APACC bal-acc 0.535; HiCervix bal-acc 0.562.
- **Grad-CAM~\cite{selvaraju2017}** (`figures/image_gradcam.png`): abnormal evidence
  localises to the nucleus/perinuclear halo (HPV cytopathic signature); coarse (B0).

### METHODOLOGY 3.3 — Tabular expert
- **Data:** NHANES 2003–2016, 7 cycles, **n=12,634** women 18–59, label = high-risk
  HPV DNA (any of 14 IARC genotypes, Roche Linear Array), **prevalence 24.6%**.
  30-item female questionnaire + 3 engineered exposure ratios; items resolved by stable
  question text across cycles.
- **Leakage guard:** EXCLUDE `SXQ753` ("ever told you have HPV") = self-report of the
  target (the variable the 0.95 papers used). Nothing lab-derived enters.
- **Model:** monotonic-constrained LightGBM~\cite{ke2017lightgbm} + isotonic calibration,
  chosen after Optuna~\cite{akiba2019optuna} search + leakage-safe bake-off. Evaluate
  with repeated 5-fold OOF + bootstrap CI + **leave-one-cycle-out (LOCO)**.
- **Results:** AUROC **0.712–0.713** [±0.012], **LOCO 0.708** (no temporal gap),
  ECE 0.008, cal-slope ≈1.0, cal-intercept ≈0, PR-AUC ≈0.44, sens 0.80 @ spec 0.48.
  Ablation: demographics 0.667 → +questionnaire 0.713 (**+0.046**); +HSV-2 biomarker ≈0;
  any-HPV endpoint 0.734. Univariate: partners-per-active-year 0.669, lifetime partners
  0.640, marital 0.630, age 0.606.
- **Attack matrix (Table + `figures/expv2.png`, `figures/diagnostics.png`):** HPO +0.005;
  ensembling ≤+0.002 (trees ρ=0.98); SMOTE-correct 0; preprocessing ≈0; temporal/
  interaction/age/missingness features ≈0; GAM/EBM 0.706; age-stratified −0.031;
  genotype 16/18 & other-HR both 0.709; native-categorical CatBoost 0.712; candidate CIs
  overlap. Frame as "empirical predictive ceiling ≈0.71–0.72 **under the evaluated
  design**" — NOT a theoretical bound.
- **Leakage reproduction (`figures/leakage_demo.png`):** identical RF scores 0.707
  honestly, **0.905 with SMOTE-before-split**; 67% of balanced positives are synthetic —
  the mechanism behind Hu et al.~\cite{hu2024nhanes} (0.9485).

### METHODOLOGY 3.4 — Fusion (plain language)
- **Step 1 CO-PRIOR:** logit(p̂) = logit(p) − logit(π) + logit(τ) — strip each expert's
  training prior π, install the deployment prior τ, *per expert before pooling*
  (modifies Saerens et al.~\cite{saerens2002}). Give the equation + reading + the
  "two witnesses / common ruler" analogy.
- **Step 2 combiner:** Kittler sum rule~\cite{kittler1998} over re-anchored posteriors
  (robust to correlation).
- **Step 3 safety asymmetry (cascade):** p_fused = max(p_sum, p̂_img) when p̂_img≥θ, else
  p_sum — the questionnaire can raise but never lower a positive image. Equation + reading.
- **Step 4 abstention:** split-conformal~\cite{angelopoulos2023} per expert →
  RELEASE / DEFER / FLAG.
- Architecture figure: two frozen experts → CO-PRIOR → combiner → conformal → gate, with
  a missing-modality bypass (thick boxes = new).

### RESULTS 4.3 — Fusion (all real, `figures/fusion_experiments.png`)
- **Marginals:** image Herlev 0.728 (ECE 0.299), tabular 0.712 (ECE 0.011) — comparably
  strong. In-dist image 0.998 is leaky.
- **Dependence envelope + FALSIFIER (Gaussian-copula over real per-class marginals):**
  fused − max(single) = **+0.012 at ρ=−0.3; −0.003 at ρ=0; −0.025 at ρ=+0.6; −0.033 at
  ρ=+1.0**. Fusion fails in the realistic ρ≥0 region — the honest null.
- **CO-PRIOR:** ECE no-shift→shift: 0.196→0.002 (τ=0.02), 0.123→0.005 (τ=0.10); robust to
  ±25% π-misspecification.
- **Combiner ablation:** sum ECE 0.05–0.07 (robust); naive-Bayes AUROC 0.73–0.75 but ECE
  0.12–0.15 (overconfident, worsens with ρ).
- **Conformal coverage** holds per expert (0.95/0.90/0.80 targets met); AURC tabular
  0.134, image-indist 0.001, Herlev 0.296. Structural invariants: missing≡single, safety
  asymmetry, lone-tabular DEFER all hold.

### RESULTS 4.4 — Explainability
SHAP (`figures/shap_beeswarm.png`) and EBM shapes (`figures/ebm_shapes.png`) agree on top
drivers (marital, lifetime partners, age, acquisition rate); EBM shapes recover textbook
epidemiology (risk ↑ with exposure, saturating; ↓ with age and income). Grad-CAM as above.

---

## FIGURE / TABLE INVENTORY (files in `paper/figures/`)
- `image_gradcam.png` — image Grad-CAM (Fig, results 4.1).
- `char_signal.png` — univariate signal (Fig, 4.2).
- `eval_all.png` — all-approach ROC + metrics + confusion matrices (full-width Fig, 4.2).
- `leakage_demo.png` — honest vs SMOTE-before-split (Fig, 4.2).
- `clinical_eval.png` — ablation + calibration + decision curve (full-width Fig, 4.2).
- `hpo_history.png`, `expv2.png`, `diagnostics.png` — optional supporting figs (attack matrix).
- `fusion_experiments.png` — six-panel fusion figure (full-width Fig, 4.3).
- `shap_beeswarm.png`, `ebm_shapes.png` — explainability (Figs, 4.4).
- Tables: novelty (borrowed/modified/new), image results, attack matrix.

## PLACEHOLDERS (missing results — keep the `\placeholder{}` boxes)
1. **Group-aware (slide/patient) split** honest in-distribution image AUROC/PR-AUC/ECE +
   cross-dataset transfer matrix with CIs — replaces the leaky 0.998.
2. **DeLong paired tests** for the attack-matrix head-to-heads (complement bootstrap CIs).
3. **Prospective paired pilot** (n≈50–100, image+questionnaire+biopsy) with a DeLong test
   of fused vs best-single — the only route to a measured joint-accuracy claim.

## VERIFIED CITATIONS (already in `refs.bib`; do not alter details)
walboomers1999, sung2021, plissiti2018, jantzen2005, tan2019efficientnet, hinton2015,
kittler1998, saerens2002, angelopoulos2023, hu2024nhanes (Hu, Peng, Zhang, Wang, Ren,
*BMC Med Inform Decis Mak* 2024;24:24 — the audited 0.9485 paper), ke2017lightgbm,
akiba2019optuna, selvaraju2017, lundberg2017shap, nori2019interpretml. Add new citations
only after verifying author/title/venue against the primary source.

## STYLE REMINDERS
Voice: precise, warm, blunt about limits; intuition → mechanism → grounded number →
so-what. Numbers written out ($0.71\!\to\!0.90$). Callout boxes for definition/analogy/
key-insight/gap. One architecture TikZ diagram (mandatory, black-on-white). Tables
overfull-safe (`\footnotesize`, fixed `p{}` columns, `\setlength{\tabcolsep}{4pt}`).
