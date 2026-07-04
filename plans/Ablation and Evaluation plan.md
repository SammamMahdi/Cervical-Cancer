# Ablation & Evaluation Plan

Experimental protocol for the SIPaKMeD-trained cervical cytology classifier. Pairs with [DATA_PLAN.md](DATA_PLAN.md) (datasets, splits, mappings, preprocessing/augmentation) and [PIPELINE_DOCUMENTATION.md](../PIPELINE_DOCUMENTATION.md) (current B4→B0 model). Compute target: single **RTX 5060 Ti (16 GB)**.

---

## 1. Evaluation plan

### 1.1 In-distribution (SIPaKMeD sealed 10% test)
Evaluated **once** on the **group-aware (cluster-grouped) 10% in-distribution test split** (SIPaKMeD = 80% train / 10% val / 10% test; the test is **sealed** until the end — tuning, early stopping and calibration use the 10% **val** only). Report mean ± 95 % CI (grouped k-fold on train+val for development).
- Overall **accuracy**, **balanced accuracy**, **macro-F1**, **Cohen's κ**.
- **Per-class** precision / recall / F1 (5 classes) + 5×5 **confusion matrix**.
- **ROC-AUC** (one-vs-rest, macro).
- **Calibration:** ECE + reliability diagram, before/after temperature scaling.
- **Efficiency:** params, FLOPs, peak VRAM, latency/img on the 5060 Ti.

### 1.2 Cross-dataset (external) — via the §3 mappings in DATA_PLAN
For each of Herlev, CRIC, APCData, RIVA, APACC (per-cell) and BMT/Mendeley (image-level):
- **E0 (binary Normal/Abnormal) — primary:** accuracy, balanced accuracy, macro-F1, **sensitivity (abnormal recall)**, **specificity**, **AUROC** (`P(abnormal)=Koilo+Dysk`), at the pre-specified threshold.
- **E1 (3-class grade) — diagnostic:** confusion matrix; **High-grade recall reported (expected ≈ 0 = structural blind spot)**; performance read on the {Normal, Low-grade} subset.
- **E-morph (candidate-set hit) — optional:** lenient morphology-consistent accuracy.
- **Excluded fraction** reported per dataset (glandular / ASC-US / debris).
- **Headline number: internal-vs-external Δ** (SIPaKMeD-test E0 − external E0) — the generalization claim.
- **External calibration:** ECE on each external set (uncalibrated + post temperature-scaling fit on a small target dev slice).

### 1.3 Statistical rigor
- ≥ 3 seeds (or 5-fold); report **mean ± 95 % CI** (bootstrap).
- **McNemar's test** (paired, p<0.01) for model-vs-model on binary classifications.
- **Wilcoxon signed-rank** across folds/images for continuous metrics.
- Always report metrics **with and without** excluded classes (excluding hard cells inflates accuracy).

---

## 2. Ablation study (isolate each design choice)

Each row toggles **one** factor; all else fixed; evaluated under §1 (in-dist + cross-dataset E0).

| # | Factor | Arms | Question it answers |
|---|---|---|---|
| A1 | **Split protocol** | random per-cell **vs** cluster-grouped | **Quantifies the leakage inflation** in the current 97.11% (rigor headline) |
| A2 | **Knowledge distillation** | B0 from scratch · B0 distilled (current, buggy) · B0 distilled (**T² + single-softmax fixed**) | Does KD actually help? Is the current KD bug masking it? |
| A3 | **Input resolution** | 224 (native) **vs** 380 (current) | Accuracy vs efficiency trade-off; justify the edge choice |
| A4 | **Stain normalization** | none · Macenko · Reinhard · MixStyle (aug) | Which closes the cross-dataset gap most |
| A5 | **Augmentation** | none · geometric · +photometric · +stain-aug | Marginal value of each augmentation tier |
| A6 | **Imbalance handling** | none · class weights · focal · oversampling | Effect on minority classes (Parabasal/Metaplastic) |
| A7 | **Calibration** | raw · temperature-scaled | ECE reduction; decision-threshold reliability |
| A8 | **Backbone** | EfficientNet-B0 · B2 · MobileNetV3 · ResNet50 | Capacity vs efficiency on the same protocol |
| A9 *(opt)* | **SSL pretraining** | ImageNet init **vs** SSL on pooled cervical crops | Does label-free pretraining improve external transfer? |

**Reporting:** one table, Δ vs the full pipeline per arm, on in-dist + the mean external E0. A1 and A2 are the two most important (rigor + distillation validity).

---

## 2b. Hyperparameter optimization protocol (Optuna + Hyperband)
- **Method:** an **Optuna** study with **TPE sampler + `HyperbandPruner`** (reduction factor 3), maximizing **validation macro-F1** (or balanced accuracy) on the group-aware **10% validation** split; ~30–50 early-pruned trials, fixed seed, single GPU.
- **Searched:** learning rate, weight decay, optimizer (Adam / AdamW / SGD-momentum), dropout, label smoothing, augmentation strength, and the **KD `T` / `α`**. (Tabular branch, U7: a separate Optuna study over XGBoost params; TabPFN needs none.)
- **Anti-leakage:** tuning touches **only** the val split (or grouped CV on train+val); the **10% test stays sealed** and is evaluated **once**, after the single best config is fixed. Report the search space, trial budget, and final config for reproducibility.
- Full recipe + code sketch in [PIPELINE_UPGRADE_PLAN.md → U2b](PIPELINE_UPGRADE_PLAN.md).

## 3. Model comparison (apples-to-apples)
All models trained on the **same cluster-grouped SIPaKMeD split** and evaluated under the **same cross-dataset protocol** (§1.2) — so comparison is fair, not cherry-picked from papers.

**Families:**
- **Lightweight / edge:** EfficientNet-B0 (yours), MobileNetV3-Large, ShuffleNetV2, EfficientNet-B2.
- **Heavier CNN:** ResNet50, EfficientNet-B4 (your teacher), DenseNet121.
- **Transformer / hybrid:** Swin-Tiny / MaxViT-Tiny (small enough for 16 GB).
- **Published SIPaKMeD SOTA** (re-run under this protocol, or cited with the caveat their numbers are random-split / internal): CCanNet (98.53 %, 1.27 M params), MaxCerVixT, CerviXpert.

**Reported per model:** in-dist 5-class macro-F1; **mean external E0 + internal-vs-external Δ**; calibration (ECE); **efficiency (params / FLOPs / latency / VRAM)**. The thesis claim is *best accuracy-retained-under-shift per compute*, **not** highest internal accuracy.

---

## 4. Cross-dataset evaluation (generalization protocol)
- **Primary:** train on SIPaKMeD → evaluate on every external dataset at E0/E1 (§1.2). The **internal-vs-external Δ per dataset** is the core generalization result.
- **Preparation-stratified read:** group externals by preparation — conventional (Herlev, CRIC, RIVA, APACC) vs LBC (APCData, Mendeley, BMT) — and report SIPaKMeD(conventional-ish)→LBC vs →conventional separately (preparation is the dominant shift).
- **Stain-norm interaction:** report external E0 with vs without stain normalization (ties to A4) — shows how much of the gap is color vs morphology.
- **Optional symmetric LODO** (if you later relax to binary multi-dataset training): leave-one-dataset-out across {SIPaKMeD, Herlev, CRIC, RIVA, APACC} at E0; never split a lineage across train/test.
- **Honesty controls:** always publish the **excluded fraction** and the **E1 High-grade recall (≈0)** so the binary score isn't read as full screening capability.

---

## 5. Reporting standards
- Lead with **balanced accuracy / macro-F1 / per-class recall**, not raw accuracy (heavy imbalance).
- Every external number paired with its **excluded fraction** and **internal-vs-external Δ**.
- State the **structural blind spot** (no HSIL/glandular) wherever screening performance is claimed.
- Release **code, seeds, split files, the cluster-grouping join, and the label-mapping config** for reproducibility.
- Don't compare against the literature's random-split >97–99 % numbers without flagging their likely leakage (A1 demonstrates why).
