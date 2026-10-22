# Tabular Results — index

Two tabular arms for the cervical screening fusion, each self-contained in its own subfolder.

```
Tabular results/
├── UCI/       cervical-cancer risk-factors questionnaire (UCI 383)  → biopsy target
│   ├── UCI_RESULTS.md          headline: honest ceiling ≈ 0.66 AUROC (data-imposed)
│   ├── figures/                01–12 characterization + A–F work-plan
│   └── metrics/                tabular_plan_results.json
└── NHANES/    NHANES 2007–2016 questionnaire  → high-risk HPV DNA target
    ├── NHANES_RESULTS.md        headline: honest ≈ 0.71 AUROC, tight CI, temporally validated
    ├── figures/                 char_* (characterization) · nhanes_* (bake-off) · hpo_* (tuning)
    └── metrics/                 nhanes_meta / _characterization / _pipeline / _hpo .json
```

| | **UCI 383** | **NHANES** |
|---|---|---|
| n | 858 (383 complete) | **9,348** |
| positives | 55 (6.4 %) | **2,177 (23.3 %)** |
| label | biopsy (downstream) | high-risk HPV DNA (upstream, HPV-axis) |
| honest AUROC | 0.66 [±0.08] | **0.712 [±0.012]** |
| verdict | weak triage, ceiling is the *data* | **genuine signal; the recommended tabular arm** |

**Code** (all in `legacy_src/src/`): UCI → `tabular.py`, `tabular_characterize.py`, `tabular_plan.py` ·
NHANES → `nhanes_data.py`, `nhanes_characterize.py`, `nhanes_pipeline.py`, `nhanes_hpo.py`.

Both arms are **leakage-safe** (no oversample-before-split, no target-derived features, resampling/impute/
calibration fit inside folds). The NHANES folder additionally documents *why* the published 0.90–0.95 HPV
numbers are leakage (`figures/char_09_leakage_demo`).
