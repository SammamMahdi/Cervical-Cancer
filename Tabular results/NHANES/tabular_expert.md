# Tabular Expert — Full Methodology

**A leakage-free, calibrated, questionnaire-only machine-learning model that estimates a woman's probability
of carrying high-risk HPV, built on nationally-representative NHANES data, to serve as the tabular arm of the
any-modality cervical-screening fusion.**

This document is the complete, reproducible methodology — data collection → label definition → feature
engineering → preprocessing → imbalance handling → models → hyperparameter optimization → evaluation →
calibration → clinical utility → integration. Every number is measured in this repository; every design
choice is justified and, where it matters, verified empirically.

Code: [`legacy_src/src/nhanes_*.py`](../../legacy_src/src/). Data cache: `data/nhanes/*.XPT`.
Figures: [`figures/`](figures/). Metrics: [`metrics/`](metrics/). Item-level Q&A: [`QUESTIONNAIRE.md`](QUESTIONNAIRE.md).
Headline results: [`NHANES_RESULTS.md`](NHANES_RESULTS.md).

---

## 0. Objective and prediction task

The system fuses a cervical **cytology-image** expert with a **questionnaire** expert. This document covers
the questionnaire (tabular) expert. Its job is **not** to diagnose disease; it is to emit a **calibrated
probability** `P_tab = P(high-risk HPV+)` that can raise suspicion in the fusion cascade (never overrule a
positive image). Formally:

> Given information obtainable from a participant questionnaire **before** any HPV test,
> estimate `P(Y = 1 | X_questionnaire)`, where `Y` is the laboratory HPV result.

The lab HPV result is **only ever the target, never an input.** This single rule is the backbone of the
leakage discipline throughout.

---

## 1. Data collection

### 1.1 Source and cycles
**NHANES** (US National Health and Nutrition Examination Survey), a nationally-representative cross-sectional
survey. We pool **seven two-year cycles, 2003–2016**, the cycles for which NHANES publicly released the
self-collected vaginal-swab HPV DNA assay for women aged 18–59.

| Cycle | Year code | HPV swab file | Feature files |
|---|---|---|---|
| 2003–2004 | C | `L37SWR_C` | `DEMO_C, SXQ_C, RHQ_C, SMQ_C, IMQ_C*, HSV_C*` |
| 2005–2006 | D | `HPVSWR_D` | `*_D` |
| 2007–2008 | E | `HPVSWR_E` | `*_E` |
| 2009–2010 | F | `HPVSWR_F` | `*_F` |
| 2011–2012 | G | `HPVSWR_G` | `*_G` |
| 2013–2014 | H | `HPVSWR_H` | `*_H` |
| 2015–2016 | I | `HPVSWR_I` | `*_I` |

`* ` = file/variable not present in every cycle (e.g. HPV-vaccine question begins 2007; HSV-2 serology begins
2005). Absent files/variables resolve to `NaN` and are handled downstream.

### 1.2 Download and caching
Files are SAS-transport (`.XPT`). The working URL pattern (verified) is:

```
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{startYear}/DataFiles/{FILE}.XPT
```

`nhanes_data._download()` fetches each file once, validates the XPORT magic header (`b"HE…"`, guarding
against HTML error pages), caches to `data/nhanes/`, and reads it with `pandas.read_sas(format="xport")`.
Variable **question text** (labels embedded in the XPT header) is read with `pyreadstat.read_xport(...,
metadataonly=True)` — this is what lets us resolve items by their stable question wording across cycles
(§3.2). All acquisition is from the CDC public site; no restricted/youth files are used.

### 1.3 Cohort
Female respondents with a **valid HPV swab result** (adequate assay, at least one genotype called). Merging
is by respondent id `SEQN`. Pooled cohort:

> **n = 12,634 women · 3,103 high-risk-HPV positive (24.6 %) · 30 questionnaire items (+3 engineered).**

Per-cycle positivity ranges 20.4 %–29.1 %, stable and epidemiologically plausible (HPV declines with age).

---

## 2. Outcome (label) definition

### 2.1 Primary endpoint — high-risk HPV
`high_risk_hpv = 1` if **any** of the 14 IARC oncogenic genotypes tests positive on the Roche Linear Array:

```
{16, 18, 31, 33, 35, 39, 45, 51, 52, 56, 58, 59, 66, 68}
```

Each genotype is an `LBDR##` variable coded `1 = positive, 2 = negative, 3 = inadequate`. A woman is positive
if any high-risk `LBDR## == 1`. Using the per-genotype Linear-Array calls (identical panel across all seven
cycles) **harmonizes the label** — we deliberately do *not* use the cycle-specific Cobas summary variable,
which differs between cycles.

### 2.2 Secondary endpoint — any HPV
`target="any"` redefines positivity as **any** detectable HPV type (all `LBDR##` genotypes, low- and
high-risk). Prevalence rises to 45.0 %. Reported alongside the primary because it is more common and more
learnable, though clinically less specific.

### 2.3 Inadequate / missing samples
An inadequate assay (`==3`) or an entirely un-typed sample is set to `NaN` and **dropped** — never coerced to
negative. Silently turning inadequate/missing into "negative" is a common inflation source we avoid.

### 2.4 The critical exclusion (leakage)
`SXQ753` — *"Has a doctor ever told you that you had HPV?"* (present 2011+, 84 % answered) — is a **self-report
of the outcome itself**. It is **excluded and never used as a feature**. This is the exact variable the
published 0.90–0.95-AUC NHANES papers used as their *label*; used as a *feature* it is textbook target
leakage. The exclusion is enforced in code (`EXCLUDED` set in `nhanes_data.py`).

---

## 3. Feature set — the questionnaire

The feature set is the **actual female patient questionnaire**, not a hand-picked shortlist. Full item-level
Q&A (question text + answer coding + coverage) is in [`QUESTIONNAIRE.md`](QUESTIONNAIRE.md).

### 3.1 Items by section (30 questionnaire + 3 engineered)

| Section | Items |
|---|---|
| **Demographics** | age, race/ethnicity, education, marital status, income-to-poverty ratio |
| **Sexual behaviour** | ever had sex; age at first sex; age at first oral sex; ever oral/anal sex with a man; lifetime male partners; past-year male partners; lifetime vaginal & oral partners; times sex without condom (past yr); new partner (past yr); sexual orientation |
| **STI history** (doctor-diagnosed) | genital herpes; gonorrhea; chlamydia; genital warts *(HPV-adjacent — §3.3)* |
| **Reproductive** | age at menarche; ever pregnant; number of pregnancies |
| **Smoking** | smoked ≥100 cigarettes; smoking status (never/former/current) |
| **Vaccination** | received HPV vaccine |
| **Biomarker** (one blood test) | HSV-2 antibody serology |
| **Engineered** (§3.4) | years sexually active; partners per active year; pregnancies per active year |

### 3.2 Cross-cycle resolution by question text
NHANES renames variables across cycles (e.g. lifetime partners is `SXQ101` in 2007–08 but `SXD101` from 2009).
Rather than brittle name lists, each item is resolved **by its stable question label** (read via pyreadstat)
with a name-fallback list. `_resolve()` first tries known variable names, then matches the label text. This
robustly reconstructs the same conceptual questionnaire from seven differently-named file schemas.

### 3.3 Feature admissibility (leakage audit)
Every candidate variable is classified by whether the information would exist **at questionnaire time**:

| Category | Admitted? |
|---|---|
| Demographics / socioeconomic | ✅ |
| Sexual & reproductive history | ✅ |
| Smoking history | ✅ |
| HPV vaccination history | ✅ (temporally appropriate; set "No" pre-2007) |
| STI history (herpes / gonorrhea / chlamydia — *different pathogens*) | ✅ |
| HSV-2 serology (different virus; shared-transmission proxy) | ✅ (as one optional biomarker) |
| **HPV lab result / genotype / cytology** | ❌ (the target) |
| **Self-reported HPV diagnosis (`SXQ753`)** | ❌ (self-report of the target) |
| **Genital warts** | ⚠️ *HPV-adjacent* — warts are low-risk HPV 6/11 while the target is high-risk types; **included but reported with/without** (`ADJACENT` set). Empirically contributes ≈ 0 (univariate AUROC 0.507). |

### 3.4 Engineered features
Ratio features that make weak exposure signals denser (the one honest lever that added signal on UCI too):
`years_active = age − age_first_sex`; `partners_per_active_year = n_partners_life / (years_active + 1)`;
`pregnancies_per_active_year`. `partners_per_active_year` becomes the single strongest feature (univariate
AUROC 0.669).

---

## 4. Preprocessing

### 4.1 Cleaning (kind-aware sentinel handling)
NHANES encodes "refused / don't know / not applicable" as sentinels (7, 9, 77, 99, …). Blanket-masking these
would delete real small counts (7 or 9 partners are real). `_clean()` is therefore **kind-aware**:

| Kind | Rule |
|---|---|
| categorical / binary | mask {7, 9, 77, 99} |
| serology (HSV-2) | mask {3 = indeterminate, 7, 9} |
| age | mask {77, 99, 777, 999}; clip to [8, 60] |
| count | mask {7777, 9999, 77777, 99999}; clip > 500 (kills 99999-style codes, keeps real counts) |

### 4.2 Derived variables (removing structural missingness)
- **`smoking_status`** (never / former / current) derived from `SMQ020` + `SMQ040`, so "current smoker" is not
  structurally missing for never-smokers.
- **`n_pregnancies = 0`** for women who report never pregnant (a true zero, not a missing).
- **`hpv_vaccine = No`** for the pre-2007 cycles (the vaccine did not exist → definitionally unvaccinated).

### 4.3 Missing data — handled inside folds, natively where possible
No global imputation. Two strategies, both leakage-safe:
- **Native-NaN GBDTs** (LightGBM/XGBoost/CatBoost/HistGB) receive the raw matrix and learn a default
  direction for missing values.
- **Imputation-requiring models** (RF/LogReg/kNN) get a `SimpleImputer` **fit inside each training fold**.

The preprocessing ablation ([`nhanes_preproc_test.py`](../../legacy_src/src/nhanes_preproc_test.py),
`figures/nhanes_preproc_ablation`) showed **native NaN is the best option (+0.0014 over median impute)**;
KNN/MICE imputation are equal or worse.

### 4.4 Encoding, scaling — and why they don't matter for the chosen models
Categoricals are one-hot encoded for linear/kNN models; native GBDTs take integer codes. **Trees split on
rank, so scaling / log / Yeo-Johnson / quantile transforms cannot change their AUROC** — the ablation
confirms this empirically (those variants land within ±0.0002 of baseline). Preprocessing changes the
*representation*, not the *information*; it is not a lever for accuracy here. The only representation choice
that matters (native-NaN vs imputation) is tiny.

---

## 5. Class imbalance

### 5.1 Class weights, not resampling
At 24.6 % positive the imbalance is mild. We use **class weighting** (`class_weight="balanced"` /
`scale_pos_weight`), which reweights the loss without fabricating data. AUROC is threshold-free, so imbalance
handling mainly affects the operating point and calibration, both handled explicitly (§8.4, §8.6).

### 5.2 SMOTE-before-split is leakage — proven, not asserted
([`nhanes_smote_test.py`](../../legacy_src/src/nhanes_smote_test.py),
`figures/nhanes_smote_comparison`, `char_09_leakage_demo`.)

| Method | AUROC |
|---|---|
| Honest (class-weight, no resampling) | 0.715 |
| **Split-then-SMOTE** (SMOTE inside training fold only) | **0.712** — *zero gain* |
| **SMOTE-before-split** (SMOTE whole set, then split) | **0.902–0.906** — *leak* |

The inflation was decomposed by tracking real vs synthetic test points: scoring the same
SMOTE-before-split model on **real patients only** returns **0.705** — the entire +0.20 comes from grading the
model on the **67 % of "positive" test rows that are synthetic**. Two independent failures: (i) synthetic test
points interpolate training points (leakage), and (ii) you are measuring accuracy on data that will never
occur in a clinic (construct invalidity). **We therefore never use SMOTE**; it is included only as the demo
that explains the literature's inflated numbers (§10).

---

## 6. Model families

A leakage-safe bake-off ([`nhanes_pipeline.py`](../../legacy_src/src/nhanes_pipeline.py)) over 10 families,
each in a `Pipeline` whose preprocessing is fit inside folds:

- **Gradient-boosted trees:** LightGBM, XGBoost, CatBoost, HistGradientBoosting
- **Bagged trees:** RandomForest, ExtraTrees
- **Linear / distance:** L2-Logistic Regression, kNN
- **Foundation model:** TabPFN v2 *(n < 10 k, native missing — attempted; skipped when the one-time Prior Labs
  license/login is unavailable, or run via `TABPFN_TOKEN`)*

**Finding:** the GBDT cluster (LightGBM ≈ CatBoost ≈ XGBoost ≈ RandomForest) leads at ~0.71 AUROC and is
statistically tied; linear/kNN trail by ~0.03 (the signal is non-linear and interaction-driven).

---

## 7. Hyperparameter optimization

([`nhanes_hpo.py`](../../legacy_src/src/nhanes_hpo.py), `figures/hpo_history`, `hpo_tuned_comparison`,
`metrics/nhanes_hpo_results.json`.)

### 7.1 Optuna, TPE
Each of LightGBM / XGBoost / CatBoost / RandomForest is tuned with an Optuna **TPE sampler (multivariate)**,
**80 trials/model**, a TQDM progress bar, and a per-model wall-clock safety cap. Broad search spaces:

| Model | Tuned hyperparameters (ranges) |
|---|---|
| LightGBM | n_estimators 100–3000 · learning_rate 2e-3–0.3 (log) · num_leaves 7–511 (log) · max_depth 2–16 · min_child_samples 2–250 · min_split_gain · subsample · colsample_bytree · reg_alpha/lambda 1e-8–30 (log) |
| XGBoost | n_estimators 100–3000 · learning_rate 2e-3–0.3 · max_depth 2–14 · min_child_weight · subsample · colsample_bytree/bylevel · gamma · reg_alpha/lambda · scale_pos_weight |
| CatBoost | iterations 100–2000 · learning_rate · depth 3–11 · l2_leaf_reg · bagging_temperature · random_strength · border_count 32–255 |
| RandomForest | n_estimators 200–1500 · max_depth 3–40 · min_samples_leaf/split · max_features · max_samples |

### 7.2 Compute — CPU, not GPU
The environment has a CUDA GPU (RTX 5060 Ti), but **GPU is 4–18× slower for GBDTs at this data size**
(benchmarked: XGBoost 0.15 s CPU vs 0.65 s GPU; CatBoost 1.15 s CPU vs 20.4 s GPU) because kernel-launch and
host↔device transfer overhead dominate for ~10 k-row tables. HPO therefore runs multi-threaded on CPU.

### 7.3 Objective and result
Objective = **5-fold stratified OOF AUROC** (imputation for RF fit inside the fold). The optimization history
shows all four models **plateau within ~15 trials and stay flat** for the remaining ~65 — the wide ranges were
explored and confirmed the ceiling rather than breaking it. The best configuration uses **small, heavily
regularized trees** (LightGBM `num_leaves ≈ 8`), itself evidence that the signal is low-complexity. **Tuning
lifts AUROC by only ~+0.005** over sensible defaults (0.712 → 0.717 on the 5-cycle table), with all four tuned
models tied at 0.714–0.717.

---

## 8. Evaluation protocol

### 8.1 Cross-validation and confidence intervals
**Repeated (×3) stratified 5-fold** out-of-fold predictions, pooled, with **bootstrap 95 % CIs** (1000
resamples). With 3,103 positives the CI half-width is ~±0.012 — differences below ~0.02 AUROC are noise.

### 8.2 Temporal validation (the deployment test)
**Leave-one-cycle-out (LOCO):** train on six cycles, test on the held-out survey wave; repeat for all seven.
This answers the real question — *does a model trained on past cycles generalize to a future one?* — and is
stronger than a single 2013→2015 split. Result: **LOCO ≈ in-distribution (0.708 vs 0.712) — essentially no
generalization gap.**

### 8.3 Metrics (and why not accuracy)
Reported: **AUROC, PR-AUC, Brier, ECE, calibration slope/intercept, sensitivity/specificity at operating
points, F1, and net benefit.** **Accuracy is *not* a selection metric** — at 24.6 % prevalence a do-nothing
"all-negative" model scores 75.4 %, so accuracy rewards ignoring the positives. It is reported only for
completeness and explicitly flagged.

### 8.4 Calibration
The fusion consumes **probabilities**, so calibration is first-class. Class-weighted GBDTs are miscalibrated
(ECE ~0.15–0.20); an **isotonic** wrapper (fit inside folds) fixes it with no AUROC cost:

> ECE **0.197 → 0.008**, Brier 0.205 → 0.161, **calibration slope ≈ 1.0, intercept ≈ 0.0**.

Isotonic ≈ sigmoid here; both far better than raw. (`figures/nhanes_04_reliability`, `nhanes_clinical_eval`.)

### 8.5 Decision-curve analysis (clinical usefulness)
Net benefit `NB(p_t) = TP/n − FP/n · p_t/(1−p_t)` is computed across threshold probabilities and compared to
**treat-all** and **treat-none**. The questionnaire model's net benefit is **above both** across the plausible
range — genuine clinical utility as a triage flag, not merely a good AUC. (`figures/nhanes_clinical_eval`.)

### 8.6 Operating points
For a high-sensitivity screen: **sensitivity 0.80 → specificity ≈ 0.48–0.49**; **sensitivity 0.90 →
specificity ≈ 0.31**. (UCI at the same 0.80 sensitivity gave only 0.34 specificity.)

---

## 9. Results summary

### 9.1 Discrimination
| | AUROC | Notes |
|---|---|---|
| GBDT cluster (tuned) | **0.71–0.72** | LightGBM/CatBoost/XGBoost tied; LOCO ≈ same; ECE 0.008 |
| Linear / kNN | 0.67–0.69 | signal is non-linear |
| Empirical predictive ceiling | ~0.71–0.72 | **under the evaluated questionnaire-only feature space + leakage-free model/validation configs**; HPV carriage is partly stochastic given behaviour |

### 9.2 Feature-set ablation (TRIPOD+AI, [`nhanes_clinical_eval.py`](../../legacy_src/src/nhanes_clinical_eval.py))
| Model | high-risk AUROC | any-HPV AUROC | Brier | ECE | cal-slope | cal-intercept |
|---|---|---|---|---|---|---|
| A demographics only | 0.667 | 0.675 | 0.173 | 0.006 | 1.00 | −0.00 |
| **B + questionnaire** | **0.713** | **0.734** | 0.165 | 0.013 | 1.05 | +0.04 |
| C + HSV-2 biomarker | 0.713 | 0.735 | 0.165 | 0.012 | 1.05 | +0.04 |

The **behavioural questionnaire adds +0.046 over demographics**; the **HSV-2 biomarker adds ≈ 0** → the
**questionnaire-only Model B is the primary** (no blood draw needed). The any-HPV endpoint is more learnable
(0.734).

### 9.3 Ensembling (does a hybrid help?)
([`nhanes_ensemble_test.py`](../../legacy_src/src/nhanes_ensemble_test.py).) The four trees are **0.979
correlated** → soft-vote / rank-average / stacking beat the best single model by **≤ +0.002** (noise); adding
diverse-but-weak learners (LogReg, kNN) *hurts*. **Recommendation: a single calibrated GBDT**, not an
ensemble. (Genuine fusion gain lives at the **modality** boundary — tabular × image — not within the tabular
arm.)

### 9.4 Univariate signal
`partners_per_active_year` 0.669 · lifetime partners 0.640 · marital 0.630 · age 0.606 · age-first-sex 0.578.
Behavioural exposure carries the signal; **HSV-2 / STI-history / HPV-vaccine were all near-chance (~0.51–0.53)**
— the biomarker lever under-delivered because shared-risk is already captured by partner counts.

### 9.5 Attacks on the ceiling — the experiment matrix
Every leakage-free attempt to exceed ~0.72, and what each rules out
([`nhanes_experiment_v2.py`](../../legacy_src/src/nhanes_experiment_v2.py),
[`nhanes_diagnostics.py`](../../legacy_src/src/nhanes_diagnostics.py); figures `nhanes_expv2_summary`,
`nhanes_diagnostics`):

| Attack | Result | What it rules out |
|---|---|---|
| Hyperparameter optimization (Optuna, 80 trials/model) | +0.005 | hyperparameters aren't the bottleneck |
| GBDT family (LightGBM/CatBoost/XGBoost/RF) | ~0.71–0.72, tied | model family isn't the bottleneck |
| **Native-categorical CatBoost** | **0.712** (= LightGBM) | categorical encoding isn't the bottleneck |
| Ensembling (vote/stack; trees ρ = 0.979) | ≤ +0.002 | trees carry near-identical information |
| SMOTE (done correctly) | 0 gain | imbalance isn't the bottleneck |
| Preprocessing (scale/transform/impute) | ~0 | representation (of existing features) isn't the bottleneck |
| Recent-vs-cumulative exposure features | −0.001 | temporal exposure adds nothing measurable |
| Nonlinear / age-interaction / missingness / warts-interaction blocks | ~0 | no incremental structure |
| GAM / EBM | 0.706 [0.696, 0.716] | an additive model uncovers no hidden signal |
| Age-stratified models | −0.031 | splitting destroys statistical power |
| **Monotonic constraints** | AUROC ≈ 0; LOCO + cal-slope slightly better | structure aids *defensibility*, not discrimination |
| **Genotype subgroups (16/18 vs other-HR)** | **both ≈ 0.709** | the ceiling is **genotype-agnostic** — not a biology-mixing artifact |
| CI overlap (LightGBM 0.714 / monotonic 0.713 / EBM 0.706) | all CIs overlap | **no meaningful model ordering** |

> **Conclusion — empirical predictive ceiling ≈ 0.71–0.72 *under the evaluated design*.** Not a theoretical
> bound; the statement that two independent lines of attack (algorithm/tuning **and** representation/
> heterogeneity) each failed to reproducibly beat ~0.72 on this questionnaire-only feature space. The
> genotype-subgroup diagnostic adds that the limit is a *general* property of predicting HPV carriage from
> behaviour, not specific to any oncogenic type. **The tabular expert is therefore frozen** as a calibrated
> weak signal; the open question moves to the fusion (§11).

**Frozen configuration:** GBDT (LightGBM or native-categorical CatBoost) + **monotonic constraints** on
exposure features (free defensibility) + **isotonic calibration** → `P_tab`. **EBM is retained as the
interpretable companion** (glass-box shape functions, ECE 0.008) — the GBDT is the predictive model, the EBM
shows *what the model believes*.

---

## 10. Leakage audit of the literature

We verified that every comparable published number reaching 0.90+ does so through a mechanism we reproduced:

- **NHANES CatBoost paper (HPV AUC 0.9485):** self-reported HPV *diagnosis* label (not the swab) + ADASYN with
  unspecified split timing; the same pipeline reports **AUC 1.000 for gonorrhea** (a leak signature). We
  reproduced the mechanism (0.705 → 0.902).
- **UCI risk-factor papers (98–99.99 %):** accuracy at 6 % prevalence (93.6 % free) + SMOTE-before-split +
  `Dx:*` target leakage.
- **NHANES 2017–2020 lab study (AUC 0.57 → 0.83):** the entire jump is SMOTE-Tomek resampling — impossible to
  gain AUROC from resampling done correctly.
- **Image models (0.83–0.89):** legitimate but a *different modality* (cytology images), not comparable to a
  questionnaire.

**Conclusion:** no method legitimately beats ~0.71 on the questionnaire→HPV task. Two tells identify the fakes:
a headline **accuracy** on imbalanced data, or **resampling credited with "improving AUC."**

---

## 11. Integration with the fusion

The expert exposes a single scalar to the fusion:

1. Train the calibrated questionnaire model → OOF-calibrated `P_raw = P(HPV+)` at the cohort prevalence
   `π_tab ≈ 0.246`.
2. **Prior-shift** (Saerens–Latinne–Decaestecker 2002) to the deployment prevalence `τ`:
   `logit(P_tab) = logit(P_raw) − logit(π_tab) + logit(τ)`.
3. Emit `P_tab`. The fusion (`fusion.py`) consumes only this scalar + its prior; swapping UCI→NHANES is a
   drop-in change of `(P_tab, π_tab)` — the cascade code is unchanged.

The tabular arm's designed role is a **calibrated weak triage flag** (raise suspicion, never lower a positive
image). Its construct-alignment with the image expert is biological: the image "abnormal" class is
koilocytosis — the cytopathic effect of HPV — so both arms are HPV-axis markers.

---

## 12. Reproducibility

### 12.1 Scripts and run order
```
nhanes_data.py          # download + build the questionnaire table (+ QUESTIONNAIRE.md)
nhanes_characterize.py  # dataset characterization (char_* figures, incl. leakage demo)
nhanes_pipeline.py      # leakage-safe bake-off: AUROC/PR/Brier/ECE + bootstrap CI + LOCO + operating points
nhanes_hpo.py           # Optuna TPE hyperparameter search (--trials 80), tuned best per model
nhanes_smote_test.py    # split-then-SMOTE vs SMOTE-before-split (leakage proof)
nhanes_preproc_test.py  # preprocessing ablation (native-NaN vs impute vs transforms)
nhanes_ensemble_test.py # voting / stacking vs best single (+ correlation)
nhanes_expanded_test.py # feature-set expansion (OLD12 vs CLEAN vs FULL), with/without genital warts
nhanes_evaluation.py    # ONE figure: ROC + metrics matrix + confusion matrices for every approach
nhanes_clinical_eval.py # ablation + calibration slope/intercept + decision curve + any-HPV endpoint
```
All are deterministic (`RNG = 42`). `ND.load(target=..., engineered=..., cycles=...)` is the single entry
point; every downstream script imports the feature lists (`NUM`, `CAT`) from it.

### 12.2 Dependencies
Python 3.12 · pandas · numpy · scipy · scikit-learn 1.9 · lightgbm · catboost · xgboost · imbalanced-learn ·
optuna 4.9 · pyreadstat · matplotlib · requests · (optional: tabpfn, shap). GPU not required (§7.2).

### 12.3 Outputs
Figures → [`figures/`](figures/) (`char_*`, `nhanes_*`, `hpo_*`); metrics JSON → [`metrics/`](metrics/);
raw data cached in `data/nhanes/`. No secrets are committed (any `TABPFN_TOKEN` is read from the environment
at runtime only).

---

## 13. Limitations and the honest ceiling

- **Empirical predictive ceiling ≈ 0.71–0.72 *under the evaluated design*.** This is **not** a theoretical
  upper bound — no such bound can be established from these experiments. It is the statement that a broad,
  leakage-free set of interventions (HPO, GBDT family, ensembling, resampling, preprocessing, temporal/
  nonlinear/age-interaction/missingness feature engineering, GAM/EBM, age-stratification) **each failed to
  reproducibly exceed ~0.72** on this questionnaire-only feature space. HPV carriage is partly stochastic
  given a static questionnaire (recent exposure, partner status, immune clearance are unobserved), which is a
  plausible mechanism for the ceiling. Every attack is documented (`nhanes_*_test.py`, `nhanes_experiment_v2.py`,
  `nhanes_diagnostics.py`).
- **No paired image+questionnaire cohort exists**, so the both-modality fusion path is validated per-expert,
  not jointly (see the research plan).
- **Construct, not diagnosis.** The target is upstream HPV infection, not cervical disease; the expert is a
  triage flag, deliberately weak by design.
- **Survey weights** (NHANES `WTMEC`) are not applied — discrimination is weight-invariant; population-level
  calibration to the US would use them and is future work.
- The honest ~0.71, calibrated and clinically useful by net benefit, **is the contribution** — together with
  the audit showing why the literature's 0.90–0.95 is leakage.

---

## 14. Figure and artifact index

**Characterization:** `char_01_missingness` · `char_02_prevalence` · `char_03_dist_by_class` ·
`char_04_correlation` · `char_05_signal` · `char_06_pca` · `char_07_tsne` · `char_08_separability` ·
`char_09_leakage_demo`
**Bake-off / calibration:** `nhanes_02_bakeoff` · `nhanes_03_loco` · `nhanes_04_reliability` ·
`nhanes_05_pr_operating`
**HPO:** `hpo_history` · `hpo_tuned_comparison`
**Experiments:** `nhanes_smote_comparison` · `nhanes_preproc_ablation` · `nhanes_ensemble_corr` ·
`nhanes_ensemble_comparison` · `nhanes_expanded_comparison` · `nhanes_expanded_new_signal`
**Consolidated evaluation:** `nhanes_evaluation_all` (ROC + metrics + confusion matrices) ·
`nhanes_clinical_eval` (ablation + calibration + decision curve)
**Metrics:** `nhanes_meta` · `nhanes_characterization` · `nhanes_pipeline_results` · `nhanes_hpo_results` ·
`nhanes_smote_test` · `nhanes_preproc_test` · `nhanes_ensemble_test` · `nhanes_expanded_test` ·
`nhanes_evaluation` · `nhanes_clinical_eval` `.json`
**Docs:** [`QUESTIONNAIRE.md`](QUESTIONNAIRE.md) · [`NHANES_RESULTS.md`](NHANES_RESULTS.md)
