# NHANES HPV-Risk Tabular Expert — Data + Model Pipeline

The public, large, construct-aligned alternative to UCI 383. Pipeline (all under `legacy_src/src/`):
[`nhanes_data.py`](../../legacy_src/src/nhanes_data.py) (download + table) →
[`nhanes_characterize.py`](../../legacy_src/src/nhanes_characterize.py) (dataset characterization) →
[`nhanes_pipeline.py`](../../legacy_src/src/nhanes_pipeline.py) (model bake-off) →
[`nhanes_hpo.py`](../../legacy_src/src/nhanes_hpo.py) (Optuna hyperparameter search).
Raw files cached in `data/nhanes/`; numbers in `metrics/`; figures in `figures/` (`char_*` = characterization,
`nhanes_*` = bake-off, `hpo_*` = tuning). Leakage-safe throughout (impute/encode/calibrate/resample fit
**inside** folds).

---

## 1. The dataset we assembled

Pooled **5 NHANES cycles (2007–2016)**, downloaded from the CDC public site (SAS-transport `.XPT`):

| | value |
|---|---|
| **n (women, 18–59, with an HPV swab result)** | **9,348** |
| **Label** | **high-risk HPV DNA positive** — any of 14 IARC oncogenic types (16/18/31/33/35/39/45/51/52/56/58/59/66/68) positive on the Roche Linear Array |
| **Prevalence** | **23.3 %** (2,177 positive) — a workable base rate, unlike UCI's 6.4 % |
| **Features (12)** | age, race, education, poverty ratio, marital status, age at first sex, lifetime # sex partners, # pregnancies, age at menarche, ever-pregnant, smoked-100, smoking status |
| **Missingness** | ≤ 16.6 % per feature (handled natively by GBDTs / imputed inside folds) |

**This is the honest contrast with UCI:** the target is **upstream HPV infection** (a lab test), not a biopsy
outcome — which is exactly why it is *construct-aligned with the image expert* (koilocytosis = the cytopathic
effect of HPV; see the project HPV Q&A). Engineering notes that mattered:

- NHANES **renames variables across cycles** — lifetime partners is `SXQ101` in 2007-08 but `SXD101`
  afterwards; resolved with a per-cycle candidate list.
- `SXD171` (past-year partners) is a **male-only** item — zero overlap with the female swab cohort — so it
  was dropped rather than silently imputed.
- `smoking_status` (never/former/current) is **derived** from `SMQ020`+`SMQ040` to remove the structural
  missingness of "current smoker" (only asked of ever-smokers). `n_pregnancies` set to 0 for never-pregnant
  women (a true zero, not a missing).

---

## 2. Dataset characterization (`figures/char_*`, `metrics/nhanes_characterization.json`)

The counterpart to the UCI characterization — but where UCI showed dead signal, NHANES shows real structure:

- **`char_01_missingness`** — all features ≤ 16.6 % missing (no STD-block cliff like UCI).
- **`char_02_prevalence`** — HPV+ rate is stable across cycles (20–26 %) and **declines steeply with age**
  (highest in 18–24, lowest in 50–59) — the textbook HPV epidemiology, and a real predictive gradient.
- **`char_03_dist_by_class`** — HPV+ vs HPV− distributions **visibly separate** on lifetime partners, age,
  and age-at-first-sex (they sat on top of each other in UCI).
- **`char_04_correlation`** — coherent risk-factor structure (age ↔ pregnancies ↔ age-at-first-sex).
- **`char_05_signal`** — **single-feature AUROC and mutual information**:

  | feature | univariate AUROC |
  |---|---|
  | **lifetime # sex partners** | **0.640** |
  | marital status | 0.630 |
  | age | 0.606 |
  | poverty ratio | 0.587 |
  | age at first sex | 0.578 |

  **One NHANES feature (0.640) beats UCI's entire multivariate ceiling (0.66)**, and mutual information is an
  order of magnitude above UCI's < 0.01 — genuine, additive signal.
- **`char_06_pca` / `char_07_tsne`** — positives form a **gradient/enriched region**, not the total overlap
  UCI's t-SNE showed.
- **`char_08_separability`** — the RF out-of-fold risk-score distributions for HPV+ vs HPV− **genuinely
  differ** (AUROC 0.707).
- **`char_09_leakage_demo`** — **the honest-vs-inflated demonstration**: the same RandomForest scores
  **0.707 honestly** but **0.905 when SMOTE is applied *before* the split** (+0.20 pure artifact). This
  reproduces the exact mechanism behind the [published 0.9485 claim](https://pmc.ncbi.nlm.nih.gov/articles/PMC10809625/)
  (ADASYN, unspecified split timing, AUC 1.000 for gonorrhea) — see §8.

---

## 3. Model bake-off — which pipeline works best

10 families, each a leakage-safe pipeline; repeated stratified 5-fold OOF, bootstrap 95% CI
(`figures/nhanes_02_bakeoff`):

| Model | AUROC | 95% CI | PR-AUC | ECE (raw) |
|---|---|---|---|---|
| **CatBoost** | **0.712** | [0.700, 0.724] | 0.434 | 0.197 |
| **XGBoost** | **0.712** | [0.699, 0.723] | 0.431 | 0.202 |
| RandomForest | 0.710 | [0.698, 0.722] | 0.418 | 0.149 |
| LightGBM | 0.705 | [0.693, 0.717] | 0.423 | 0.169 |
| HistGradientBoosting | 0.697 | [0.685, 0.709] | 0.415 | 0.149 |
| LogisticRegression | 0.684 | [0.672, 0.696] | 0.370 | 0.239 |
| ExtraTrees | 0.683 | [0.670, 0.695] | 0.386 | 0.209 |
| kNN (k=51) | 0.672 | [0.658, 0.684] | 0.375 | 0.011 |
| *TabPFN (local)* | *skipped* | — | — | — | *(one-time license/login required; add via token — §6)* |

**Read this the right way:** with 2,177 positives the CI half-width is **~0.012** — so the GBDT cluster
(0.70–0.71) is *significantly* above linear/kNN, and the ranking is trustworthy (the opposite of UCI, where
every model overlapped inside ±0.08). **Gradient-boosted trees win**; the signal is non-linear and
interaction-driven, so LogReg trails by ~0.03.

## 4. It generalises across time (the deployment test)

Leave-one-cycle-out — train on four cycles, test on the held-out one (`figures/nhanes_03_loco`):

| held-out cycle | AUROC |
|---|---|
| 2007-2008 | 0.699 |
| 2009-2010 | 0.709 |
| 2011-2012 | 0.730 |
| 2013-2014 | 0.707 |
| 2015-2016 | 0.696 |
| **mean LOCO** | **0.708** (vs in-distribution 0.712) |

**Essentially no generalisation gap (0.712 → 0.708).** The model trained on past cycles holds up on a future,
unseen survey wave — strong evidence the signal is stable and real, not cycle-specific overfitting.

## 5. Calibration & operating point (what the fusion needs)

Class-weighting inflates the raw probabilities (ECE 0.197). One post-hoc calibrator fixes it
(`figures/nhanes_04_reliability`, `nhanes_05_pr_operating`):

| calibration | AUROC | ECE | Brier |
|---|---|---|---|
| raw (class-weighted) | 0.707 | 0.197 | 0.205 |
| **+ isotonic** | 0.706 | **0.008** | **0.161** |
| + sigmoid | 0.708 | 0.008 | 0.161 |

**Isotonic drops ECE from 0.197 → 0.008 with no AUROC cost** — giving a trustworthy `P_tab = P(HPV+)` for the
cascade. Screening operating points:

- **sensitivity 0.80 → specificity 0.484** (threshold 0.375) — catches 80 % of HPV+ while clearing ~half the
  HPV−. *(UCI at the same sensitivity gave only 0.34 specificity.)*
- **sensitivity 0.90 → specificity 0.310** (threshold 0.286).

---

## 6. Recommended pipeline

> **CatBoost (or XGBoost) on the raw 12-feature matrix → isotonic calibration → prior-shift to the deployment
> prevalence → emit `P_tab = P(high-risk HPV+)`.**
> AUROC **0.712** [0.700, 0.724], LOCO **0.708**, ECE **0.008**, sens 0.80 @ spec 0.48.

- **Native-missing GBDTs are preferred** — no imputation step, they beat imputed linear/kNN by ~0.03, and
  CatBoost/XGBoost/RF are statistically tied at the top (pick CatBoost for native categoricals + native NaN, or
  RandomForest if you want a dependency-light default).
- **Always wrap in isotonic calibration** (folds respected) — the fusion consumes probabilities, not labels.
- **Evaluate with LOCO**, not just random CV — it is the honest "next survey cycle" deployment estimate.
- **TabPFN** (n < 10k, native missing, well-calibrated) is a natural fit and worth one run; local inference
  needs a one-time Prior Labs license acceptance, or set `TABPFN_TOKEN` for the cloud client. Given the GBDT
  cluster is already tight and the signal is modest, expect it to land in the same 0.70–0.71 band.

## 7. Clinical-prediction evaluation (TRIPOD+AI additions)

Full questionnaire (n=12,634, 2003–2016), tuned LightGBM + isotonic calibration, OOF.
Figure: `nhanes_clinical_eval`; numbers in `metrics/nhanes_clinical_eval.json`.

**Feature-set ablation (does the questionnaire add over demographics?):**

| Model | high-risk HPV AUROC | any-HPV AUROC | Brier | ECE | cal-slope | cal-intercept |
|---|---|---|---|---|---|---|
| **A** demographics only | 0.667 | 0.675 | 0.173 | 0.006 | 1.00 | −0.00 |
| **B** + questionnaire (behaviour, no lab) | **0.713** | **0.734** | 0.165 | 0.013 | 1.05 | +0.04 |
| **C** + HSV-2 biomarker | 0.713 | 0.735 | 0.165 | 0.012 | 1.05 | +0.04 |

- **The behavioural questionnaire adds a real +0.046 over demographics** (0.667 → 0.713) — a clean
  A < B ablation. **The HSV-2 lab biomarker adds nothing** (B → C ≈ 0.000), so the pure **questionnaire-only
  Model B is the primary** (and needs no blood draw).
- **Secondary endpoint (any HPV, 45 % prevalence) is more learnable**: **0.734** vs 0.713 for high-risk —
  as expected (more common → less stochastic), though high-risk is the clinically meaningful target.
- **Calibration is near-ideal**: slope ≈ 1.0 (ideal 1), intercept ≈ 0 (ideal 0), ECE ≤ 0.013, Brier 0.165 —
  the probabilities are trustworthy, which is what the fusion consumes.
- **Decision-curve analysis**: Model B's net benefit is **above both "treat all" and "treat none"** across
  the plausible threshold range — i.e., it has genuine clinical utility as a triage flag, not just a good AUC.

## 8. NHANES vs UCI — the bottom line for the fusion

| | UCI 383 | **NHANES (this work)** |
|---|---|---|
| n | 858 (383 complete) | **9,348** |
| positives | 55 (6.4 %) | **2,177 (23.3 %)** |
| label | biopsy (downstream) | **high-risk HPV (upstream, HPV-axis)** |
| honest AUROC | 0.66 [±0.08] | **0.712 [±0.012]** |
| CI width | wide (unreliable) | **tight (trustworthy)** |
| temporal test | n/a | **LOCO 0.708 (no gap)** |
| construct-alignment w/ image | biopsy ≠ cytology | **HPV ↔ koilocytosis (aligned)** |

NHANES gives a **stronger, tighter, temporally-validated, construct-aligned** tabular arm. Its cost is the
one we already flagged: because both arms key on HPV, they are more positively dependent given the truth, so
the *marginal* benefit of fusing them is smaller — but each expert is individually far more defensible.
