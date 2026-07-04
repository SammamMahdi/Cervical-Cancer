# Pipeline Upgrade Plan

How to upgrade the current cervical-cytology pipeline, what each upgrade is, **how it will be implemented**, and the **established method + reference** behind it. Pairs with [DATA_PLAN.md](DATA_PLAN.md), [Ablation and Evaluation plan.md](Ablation%20and%20Evaluation%20plan.md), [LABEL_ALIGNMENT_METHOD.md](LABEL_ALIGNMENT_METHOD.md), [MULTIMODAL_PLAN.md](MULTIMODAL_PLAN.md), [../PIPELINE_DOCUMENTATION.md](../PIPELINE_DOCUMENTATION.md). Compute: single **RTX 5060 Ti (16 GB)**.

> **Anti-hallucination contract.** Every technique below is an **established, citable method** (see the References table). **No number in this plan is a claimed result** — figures are either (a) *targets/acceptance thresholds* we will measure, or (b) *literature evidence* explicitly attributed to a cited paper. Real numbers only appear after the experiments are run. Cervical-specific claims reuse facts already source-verified earlier in this project (literature review + grounding workflows).

---

## 0. Starting point (honest baseline)
Current pipeline ([../PIPELINE_DOCUMENTATION.md](../PIPELINE_DOCUMENTATION.md)): EfficientNet-B4 teacher (two-stage frozen→fine-tune) **knowledge-distilled** into an EfficientNet-B0 student on **SIPaKMeD** single-cell crops; Grad-CAM for explainability. Known issues to fix: (1) **leaky random split** (val doubles as test; the 97.11% is inflated); (2) **KD bug** — temperature applied to already-softmaxed probabilities and the `T²` term omitted, so the KL term ≈0.005 and distillation barely contributes; (3) student runs at 380 px not its native 224; (4) **no external validation, no calibration, no measured efficiency**; (5) overfitting (≈92% train / ≈65% val — first confirm which dataset/classes this is on).

**Which field-gaps this pipeline currently sits on:** it *aims at* edge-efficiency (Gap 7) but doesn't validate it, and currently *embodies* the internal-only-evaluation (Gap 1) and leakage/rigor (Gap 6) failure modes. The upgrades below convert those from weaknesses into contributions.

---

## 1. Upgrade scope (what's in / out, and why)
| # | Upgrade | Field gap | Feasible on 1× 5060 Ti + public cell data? | Priority |
|---|---|---|---|---|
| U1 | Leakage-safe evaluation foundation | 6 | Yes (pure code) | **P0** |
| U2 | Fix knowledge distillation + ablation | 7 | Yes | **P0** |
| U2b | **Hyperparameter optimization (HPO)** | 6 | Yes | **P1** |
| U3 | External / cross-dataset validation | 1 | Yes (datasets in hand) | **P1** |
| U4 | Stain / domain-shift robustness | 2 | Yes | **P1** |
| U5 | Calibration + clinical operating points | 5 | Yes (cheap) | **P1** |
| U6 | Validated edge efficiency | 7 | Yes | **P2** |
| U7 | Multimodal fusion (numeric features) | 11 | Yes (optional) | **P3** |
| U8 | Faithful explainability (SHAP + Grad-CAM) | 5 | Yes (optional) | **P3** |
| U9 | Release-side conformal risk audit | 5 | Yes (stretch) | **P3** |
| — | WSI/MIL, federated, prospective/regulatory | 3,4,8,12 | **No** — no public cytology WSI; no multi-site infra; no clinical deployment | **out of scope** |

---

## 2. The upgrades — what, why, and how to implement

### U1 — Leakage-safe evaluation foundation (P0)
- **Why:** the current random split leaks same-source cells across train/val, inflating accuracy (Gap 6).
- **How:** rebuild the SIPaKMeD **cell→cluster** map from the shipped `.dat`/`.mat` feature tables (`image#`+`cell#`); make a **group-aware 80/10/10 train / validation / in-dist-test** split with **`sklearn.model_selection.StratifiedGroupKFold`** (groups = cluster id, stratified by class, no cluster shared across the three) — plus **external** test (Herlev/APACC/HiCervix). Validation = tuning / early-stopping / calibration; the **test stays sealed** until the end. Fix seeds (`numpy`, `torch`, `PYTHONHASHSEED`), enable deterministic ops, report **mean ± 95% CI** (bootstrap, grouped k-fold on train+val) and **McNemar / Wilcoxon** for comparisons.
- **Validated method:** StratifiedGroupKFold (scikit-learn); group-aware CV is the standard leakage-safe protocol; bootstrap CIs (Efron); McNemar's test for paired classifiers.
- **Acceptance:** zero cluster overlap across splits (assert); the re-derived baseline replaces 97.11%.

### U2 — Fix knowledge distillation + ablation (P0)
- **Why:** the KD term is near-zero, so the "distillation" is barely doing anything (Gap 7 credibility).
- **How:** distill from **logits** with the canonical Hinton loss: `L = α·CE(y, student) + (1−α)·T²·KL(softmax(t_logits/T) ‖ softmax(s_logits/T))`; expose the teacher's pre-softmax logits (not the softmax output); re-tune `T` and `α`. Run the **ablation:** B0 from scratch vs B0 distilled (fixed) vs B0 distilled (old buggy), and report **accuracy-retained-vs-teacher** (student/teacher fidelity).
- **Validated method:** Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network", **NIPS 2014 Deep Learning Workshop** (arXiv:1503.02531) — the `T²` scaling and logit-temperature are from this paper (Section 2.1).
- **Acceptance:** KL term is non-trivial (≫0.005); ablation shows whether KD beats from-scratch B0.

### U3 — External / cross-dataset validation (P1) — *the headline*
- **Why:** the field's #1 gap; internal-only numbers aren't deployable (Gap 1).
- **How:** evaluate the SIPaKMeD-trained model on **Herlev, APACC, HiCervix** (Tier-1) via the **binary normal/abnormal** label collapse from [DATA_PLAN.md §3](DATA_PLAN.md); report **internal−external Δ** as the headline metric, with per-dataset balanced-accuracy / sensitivity / specificity / AUROC and the **excluded-fraction**. Optional leave-one-dataset-out at the binary level.
- **Validated method:** external / leave-one-dataset-out evaluation is standard generalization practice; the binary collapse is documented in our label-alignment method (avoids the one-to-many 5-class trap).
- **Acceptance:** Δ reported with CIs; per-dataset confusion + excluded fraction published.

### U4 — Stain / domain-shift robustness (P1)
- **Why:** documented 10–30-point collapse under stain/scanner shift (Gap 2; literature evidence, not our number).
- **How:** add **stain normalization** (Macenko and/or Reinhard, fit on the train split only) and **MixStyle** stain augmentation; optionally lightweight **test-time adaptation** (BN-stat recalibration). Report the **accuracy-vs-shift** curve: external accuracy **with vs without** normalization. Implement via `torchstain`/`staintools` (Macenko/Reinhard) and a MixStyle module (`timm`-compatible).
- **Validated method:** Macenko et al., ISBI 2009 (stain normalization); Reinhard et al., IEEE CG&A 2001 (color transfer); Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.
- **Acceptance:** the external Δ shrinks with normalization (or is honestly reported if not).

### U5 — Calibration + clinical operating points (P1)
- **Why:** calibration is essentially absent in the field yet screening needs trustworthy thresholds (Gap 5).
- **How:** fit **temperature scaling** on a held-out validation slice; report **ECE + reliability diagram** before/after; pre-specify operating thresholds (binary "abnormal" = the ASC-US+ screening positive) and report sensitivity/specificity at them. Also report on external sets.
- **Validated method:** Guo et al., "On Calibration of Modern Neural Networks", ICML 2017 (temperature scaling, ECE, reliability diagrams). Cervical precedent that calibration is feasible on SIPaKMeD: recent 2025–2026 cervical-cytology studies report ECE/Brier (evidential-DL and fuzzy-ensemble evaluations). *(SiCal-CytoNet 2026 was confirmed via the publisher galley PDF earlier in this project but is poorly indexed and could not be re-located by a second search — treat it as a soft, secondary citation.)*
- **Acceptance:** ECE reported pre/post; thresholds pre-registered, not tuned on test.

### U6 — Validated edge efficiency (P2)
- **Why:** "lightweight/edge" is claimed but never measured (Gap 7).
- **How:** measure **parameter count, FLOPs (fvcore / ptflops / thop), peak inference VRAM, latency/throughput** for teacher vs student on the RTX 5060 Ti (and, if available, a CPU/edge proxy); plot the **accuracy-cost frontier** and **accuracy-retained-vs-teacher**.
- **Validated method:** standard profiling libraries (fvcore `FlopCountAnalysis`, ptflops, thop); `torch.cuda.max_memory_allocated` for VRAM; warm-up + averaged timing for latency.
- **Acceptance:** a table of params/FLOPs/latency/VRAM for both models on real hardware.

### U7 — Multimodal fusion with the numeric feature tables (P3, optional)
- **Why:** the shipped per-cell handcrafted features (SIPaKMeD ~52, Herlev 20) can regularize the overfitting model and add interpretability (Gap 11). Cervical literature evidence: handcrafted+deep fusion adds a few points (e.g. radiomics + EfficientNet-B4, Frontiers in Oncology 2025: fused ~94.6% vs image-only baselines of ~85% end-to-end to ~90% feature-based; Kupas & Harangi, EMBC 2022: +3.2% F1).
- **How:** **late fusion** first (image CNN probabilities + a GBDT/**TabPFN** on the features → small logistic-regression meta-learner); only if it helps, try early concat with dimensionality balancing. Same-cell pairing only (never Herlev-features with SIPaKMeD-images). See [MULTIMODAL_PLAN.md](MULTIMODAL_PLAN.md).
- **Validated method:** Huang et al., npj Digital Medicine 2020 (fusion taxonomy / small-data guidance); GBDT (XGBoost) / TabPFN (Hollmann et al., Nature 2025) for the tabular branch.
- **Acceptance:** clean ablation (image-only vs tabular-only vs fused) on the leakage-safe split.

### U8 — Faithful explainability (P3, optional)
- **Why:** move from "where it looked" (Grad-CAM) to "classified X because y1,y2,y3" (Gap 5 trust).
- **How:** run **SHAP** on the numeric-feature branch from U7 → templated reasons ("N/C ratio high (+), nuclear area large (+)…"); keep Grad-CAM for spatial attribution. Frame as **attribution**, not claimed model reasoning (faithfulness caveat).
- **Validated method:** Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (SHAP), NeurIPS 2017; Selvaraju et al., Grad-CAM, ICCV 2017.
- **Acceptance:** per-prediction reasons that map to real cytology criteria; faithfulness limits stated.

### U9 — Release-side conformal risk audit under prevalence shift (P3, stretch)
- **Why:** the genuinely under-explored angle — safe automated "release" under prevalence shift (Gap 5 / opportunity).
- **How:** cell/tile-level **split / Mondrian conformal prediction**; report **release-side abnormal-class coverage (FNrel/Prel)** under simulated prevalence shift, not just marginal coverage. Libraries: `MAPIE` or `crepes`.
- **Validated method:** Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction…", 2023; Vovk et al. (Mondrian/class-conditional CP); the prevalence-shift critique (arXiv 2605.20956, verified earlier).
- **Acceptance:** release-side risk curves under shift; no "guaranteed safety" overclaim.

---

### U2b — Hyperparameter optimization (HPO) — cross-cutting (P1)
- **Yes, HPO is an explicit step.** Tune on the **validation split only** (or grouped CV on train+val); the **10% in-dist test stays sealed** ([DATA_PLAN §2](DATA_PLAN.md)). Tuning against the test set = leakage.
- **What to tune:** learning rate, weight decay, **optimizer** (Adam / **AdamW** / SGD-momentum), LR schedule (cosine / ReduceLROnPlateau), batch size, dropout, label smoothing, augmentation strength, stain-norm choice; **KD-specific:** temperature `T` and `α`; classifier-head width; **tabular branch (U7):** XGBoost depth / lr / n_estimators (TabPFN needs no tuning).
- **Frameworks for a single RTX 5060 Ti (ranked):**
  - **Optuna** *(recommended)* — lightweight, define-by-run, **TPE** Bayesian sampler + **median / Hyperband pruning** (kills weak trials early → many more configs per GPU-hour); plugs into Keras/PyTorch/XGBoost/sklearn. *(Akiba et al., KDD 2019.)*
  - **Keras Tuner** — **native to your `tf_keras` notebook**; RandomSearch / **Hyperband** / BayesianOptimization; lowest-friction for the image model.
  - **Ray Tune** ("raytube") — the **scalable** option; **ASHA** async early-stopping, **PBT**, and can drive Optuna/HyperOpt samplers; ideal if you later add GPUs/cluster, slightly more setup on one GPU. *(Liaw et al., 2018.)*
  - **Weights & Biases Sweeps** — if you also want experiment tracking + a dashboard (Bayesian / grid / random).
- **Method:** prefer **early-stopping search (Hyperband / ASHA)** over grid search — on one 16 GB GPU it covers far more configs by pruning bad trials. Budget ~**20–50 trials**; cache teacher logits so KD trials don't reload the B4.
- **Committed approach: Optuna + Hyperband pruning** (single RTX 5060 Ti). Implementation:
  - `study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42), pruner=HyperbandPruner(min_resource=3, max_resource=<epochs>, reduction_factor=3))`
  - **Objective** = group-aware **validation macro-F1** (or balanced accuracy) on the 10% val; report the per-epoch metric with `trial.report(val_f1, epoch)` and `if trial.should_prune(): raise optuna.TrialPruned()` so Hyperband kills weak trials early.
  - **Search space:** lr (log 1e-5–1e-3), weight decay (log 1e-6–1e-3), optimizer {Adam, **AdamW**, SGD-momentum}, dropout (0.2–0.6), label smoothing (0–0.1), augmentation strength, **KD** `T` (2–6) and `α` (0.05–0.5).
  - **Budget** ~30–50 trials, `n_jobs=1` (one GPU), fixed seed; cache teacher logits so KD trials don't reload the B4.
  - **Anti-leakage:** the objective uses the **10% val only**; the **10% test stays sealed** → re-fit the single best config and evaluate **once** on test + external. Log the search space + budget + final config for reproducibility.
  - *(On `tf_keras` you can get the same with **Keras Tuner `Hyperband`** natively; use **Ray Tune `ASHA`** only if you later scale to multiple GPUs.)*

## 3. Sequencing (phased)
- **Phase 1 — Foundation (P0):** U1 + U2 → one trustworthy baseline + a real distillation result. *(Unblocks everything; also resolves the 92/65 confusion.)*
- **Phase 2 — Generalization & trust (P1):** U3 + U4 + U5 → the external-validation headline, shift-robustness, calibration.
- **Phase 3 — Efficiency & rigor (P2):** U6 → measured accuracy-cost frontier.
- **Phase 4 — Optional contributions (P3):** U7, U8, U9.

A defensible paper is complete after Phase 1–3; Phase 4 strengthens novelty.

---

## 4. How it gets built (engineering)
- **Refactor** the notebook into modules: `data/` (cluster-grouped loader, label map, stain norm), `models/` (teacher, student, KD loss, fusion head), `eval/` (metrics, calibration, efficiency, conformal), `configs/`, `run.py`. Keep the notebook as a thin driver.
- **Reproducibility:** fixed seeds + deterministic flags; persist split files, configs, and per-run metrics (JSON); version the label-mapping config.
- **Venv libraries to add** (install **into the project `.venv` only**, never `C:\Python312`): `scikit-learn`, `torch`/`torchvision` (or keep TF + `tf_keras`), `timm`, `torchstain` (or `staintools`), `fvcore`/`ptflops`, `shap`, `xgboost`, `mapie` (or `crepes`), **`optuna`** / `keras-tuner` (or `ray[tune]` for scale), `matplotlib`. (Use the `publication-figures` house style for plots.)
- **Compute discipline:** BF16 mixed precision, gradient accumulation, 224 px student; cache teacher logits offline so KD doesn't hold the B4 in VRAM.

---

## 5. References (all established / real)
| Method | Reference |
|---|---|
| Group-aware CV | `StratifiedGroupKFold`, scikit-learn |
| Knowledge distillation (T², logit temperature) | Hinton, Vinyals, Dean, NIPS 2014 Deep Learning Workshop (arXiv:1503.02531) |
| Stain normalization | Macenko et al., IEEE ISBI 2009; Reinhard et al., IEEE CG&A 2001 |
| Stain/domain augmentation | Zhou et al., MixStyle, ICLR 2021 |
| Calibration | Guo et al., On Calibration of Modern Neural Networks, ICML 2017 |
| Efficiency profiling | fvcore `FlopCountAnalysis`; ptflops; thop |
| Hyperparameter optimization | Akiba et al., **Optuna**, KDD 2019; Liaw et al., **Ray Tune**, arXiv:1807.05118 (2018); Li et al., **Hyperband**, JMLR 2018; Li et al., **ASHA**, MLSys 2020; **Keras Tuner** (O'Malley et al., 2019) |
| Tabular branch / fusion | Huang et al., npj Digital Medicine 2020; Hollmann et al., TabPFN, Nature 2025; XGBoost (Chen & Guestrin, KDD 2016) |
| Explainability | Lundberg & Lee, SHAP, NeurIPS 2017; Selvaraju et al., Grad-CAM, ICCV 2017 |
| Conformal prediction | Angelopoulos & Bates 2023; Vovk et al. (Mondrian CP); deployment-audit arXiv 2605.20956 |
| Cervical fusion evidence | Frontiers in Oncology 2025 (radiomics+EfficientNet-B4: fused ~94.6% vs image-only ~85–90%); Kupas & Harangi, IEEE EMBC 2022 (+3.2% F1) |
| Cervical calibration precedent | 2025–2026 SIPaKMeD calibration studies (evidential-DL / fuzzy-ensemble, report ECE/Brier); SiCal-CytoNet 2026 *(publisher-verified, poorly indexed)* |

---

## 6. How we keep this honest
- No figure here is presented as an achieved result — only **targets** and **cited literature**.
- Every method has a real, established reference (table above); the cervical-specific facts were source-verified earlier in this project.
- All results will be reported under the leakage-safe protocol (U1), with CIs and the **internal−external Δ** and **excluded fraction**, and with the standing caveat that the SIPaKMeD model **cannot detect HSIL/glandular** (structural blind spot).
