# Tabular Dataset — Work Plan (UCI Cervical Cancer Risk Factors)

*Focused plan for the questionnaire/tabular arm only, built from the characterization results in
[`TABULAR_CHARACTERIZATION.md`](TABULAR_CHARACTERIZATION.md). Every expected number is grounded in what we
already measured. Code lives in [`src/tabular.py`](src/tabular.py) and
[`src/tabular_characterize.py`](src/tabular_characterize.py).*

---

## 0. The honest premise (read first)

The characterization is unambiguous: **the ceiling is ≈ 0.66 AUROC for Biopsy and it is imposed by the
data, not the model.** Evidence: the best single feature reaches univariate AUROC **0.57**, mutual
information is **< 0.01** for every feature, and the t-SNE/PCA show positives sitting *inside* the negative
cloud. A 9-model bake-off lands in **0.52–0.66 with overlapping CIs**.

> **So "improve accuracy" has two honest meanings, and we pursue both:**
> **(a)** squeeze the last ~0.02–0.03 of *real* AUROC with legitimate levers (Track B/C/D below), and
> **(b)** make the model *clinically better* — calibrated, imbalance-correct, tuned to a screening
> operating point — which matters far more than a third decimal of AUROC at 6.4% prevalence.
> **What we will NOT do:** oversample-before-split (fake 0.994), keep `Dx:*` (fake 0.998), or report
> accuracy (93.6% is free at this prevalence).

---

## 1. Tracks, in priority order (highest expected value first)

### Track A — Lock the ceiling with a stronger model arsenal *(quick, do first)*
The bake-off used sklearn defaults + XGBoost. Three strong families are missing and are the most likely to
move the number.

| # | Task | How | Expected gain | Effort |
|---|---|---|---|---|
| A1 | **TabPFN** — transformer foundation model for small tables (n < 10k), best-out-of-box on tiny data | `pip install tabpfn`; add to `models()` in `tabular_characterize.py` | **+0.00 → +0.03** (best single bet) | S |
| A2 | **LightGBM + CatBoost** with `scale_pos_weight` / native categoricals | `pip install lightgbm catboost` | +0.00 → +0.02 | S |
| A3 | **Nested CV hyperparameter tuning** for the top-4 (already did once → kNN 0.654) | `RandomizedSearchCV` inner loop, repeated outer | stabilises, ~+0.00 | M |
| A4 | **Calibrated stacking** of the top-3 (RF + HGB + TabPFN) → LR meta-learner | `StackingClassifier` + `CalibratedClassifierCV`, folds respected | ≈ +0.01 | M |

**Exit:** a definitive bake-off table with 95% CIs. Realistic outcome: **best ≈ 0.66–0.68**.

### Track B — Legitimate feature engineering *(the only real chance to add signal)*
With max univariate AUROC 0.57, engineering can only *recombine* weak signals — but it is the one honest
lever that can add, not just redistribute, signal.

| # | Task | Rationale | Expected gain |
|---|---|---|---|
| B1 | **Interaction / ratio features** — `Age × #partners`, `Smokes(years) × packs`, `#partners / age-at-first-sex`, `#pregnancies / Age` | trees split on these implicitly, but explicit ratios can help linear/kNN and small trees | +0.00 → +0.02 |
| B2 | **Aggregate risk scores** — total STDs (exists), STD-diversity count, a smoking pack-year, a contraceptive-exposure total | compress sparse binary flags into denser signals | +0.00 → +0.02 |
| B3 | **Missingness indicators** — a binary "STD block answered?" flag (the STD columns are missing *together*) | missingness may be informative (who skips the STD section) | ≈ 0 (tested weakly; retry combined with B1/B2) |
| B4 | **Feature selection** — drop noise via mutual-info / model-based RFE, keep ~10–12 | fewer noise dims → better generalisation on n=858 | +0.00 → +0.02 |

**Exit:** an engineered feature set with a measured ΔAUROC (report honestly, expect small).

### Track C — Target strategy *(borrow strength across the 4 correlated tests)*
The four targets are correlated (Schiller↔Biopsy φ ≈ 0.6). This is a real, untried lever.

| # | Task | Rationale | Expected gain |
|---|---|---|---|
| C1 | Keep **Biopsy** as the primary (histology gold standard) | cleanest clinical meaning | baseline 0.656 |
| C2 | **`any-of-4`** composite (102 positives, 11.9%) | more positives → tighter CI | not higher (≈0.61) but more *stable* |
| C3 | **Multi-task / auxiliary learning** — predict all 4 tests jointly, or use Schiller as an auxiliary label for Biopsy | shares the little signal across correlated targets | +0.00 → +0.02 (worth trying) |

### Track D — Imbalance & calibration *(this is where "clinically better" lives)*
More important than the AUROC third-decimal for a screening tool.

| # | Task | How | Deliverable |
|---|---|---|---|
| D1 | **Imbalance done right** — class weights / `scale_pos_weight` (in place); compare **SMOTE INSIDE folds** | `pip install imbalanced-learn`; `Pipeline` with SMOTE only on train folds | fair imbalance comparison |
| D2 | **Calibration** — sigmoid/isotonic (in place); add temperature/BCTS; report **ECE + reliability diagram** | `CalibratedClassifierCV`; reliability plot | calibrated `P_tab`, ECE table |
| D3 | **Operating point** — choose the threshold for a **sensitivity target** (screening favours recall) | tune on OOF for e.g. sensitivity ≥ 0.80, report the resulting specificity | operating-point table + cost curve |

### Track E — Evaluation protocol *(make every number trustworthy)*
| # | Task | Deliverable |
|---|---|---|
| E1 | **Nested stratified CV** (outer = estimate, inner = tune + calibrate), ≥ 5 repeats, bootstrap 95% CIs | locked harness |
| E2 | Report **PR-AUC, sensitivity@fixed-specificity, balanced accuracy, ECE, Brier** — never raw accuracy | metrics table |
| E3 | **Leakage guards enforced in code** — no oversample-before-split, `Dx:*` dropped, resampling inside folds only | assertions in the harness |

### Track F — Interpretability & the honest conclusion
| # | Task | Deliverable |
|---|---|---|
| F1 | **SHAP** (done) + permutation importance + partial-dependence for the top features | interpretability figures |
| F2 | Document the ceiling + the tabular expert's role: a **calibrated weak triage flag** feeding the fusion (raise suspicion, never overrule the image) | conclusion section |

---

## 2. Recommended sequence (what to actually run)

1. **Track A first (1–2 days).** Install `tabpfn lightgbm catboost imbalanced-learn`, add them to the
   bake-off, re-run `tabular_characterize.py`. This tells you the true ceiling with the best models.
2. **Track D2/D3 in parallel.** Calibration + operating point are needed regardless of the AUROC outcome.
3. **Track B + C3 next (2–3 days).** Feature engineering + multi-task — the only levers that can add signal.
4. **Track E throughout.** Every comparison goes through the nested-CV harness with CIs.
5. **Track F to close.** Interpretability + the honest write-up.

---

## 3. Realistic outcome & the pre-registered stop rule

- **Most likely:** best honest AUROC lands at **0.66–0.68** (Track A/B/C combined). A well-calibrated model
  with a sensitivity-tuned operating point is the real win.
- **Stop rule (state it now):** if Tracks A–C together do not push OOF AUROC above **0.70** (upper CI), we
  conclude the questionnaire is a **weak triage signal by nature** and finalise it as such — that is the
  honest finding, and it is exactly the role the fusion assigns it. We do **not** chase the number into
  leakage.

---

## 4. Concrete first step

```
# from legacy_src/
../.venv/Scripts/python.exe -m pip install tabpfn lightgbm catboost imbalanced-learn
# then add the three models to models() in src/tabular_characterize.py and re-run:
cd src && ../../.venv/Scripts/python.exe tabular_characterize.py
```

This regenerates `11_model_bakeoff` with 12 model families and gives the definitive ceiling. I can do this
for you on request.
