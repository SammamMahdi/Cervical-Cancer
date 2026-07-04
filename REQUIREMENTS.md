# Project Requirements — Cervical Cancer Screening AI

A living log of all stated needs for this project. Status key: ✅ done · 🔄 in progress · ⬜ not started.

_Last updated: 2026-06-21_

> **Repository layout:** active docs at root (`README.md`, `REQUIREMENTS.md`, `PIPELINE_DOCUMENTATION.md`, the notebook); `research/` = investigations done; `plans/` = future-implementation specs; `legacy/` = original pre-handover inputs. See [README.md](README.md).

---

## 1. Project goal & framing
- Build a **Two-Tier Intelligent Screening Framework** for cervical cancer:
  - **Tier 1** — tabular/demographic risk engine (Random Forest + Logistic Regression) → risk probability.
  - **Tier 2** — edge-deployable cytology image classifier (distilled EfficientNet) → malignancy probability.
  - Optional **fusion** of both into a weighted clinical decision.
- Target a **Q1/Q2 journal** submission; framing must be defensible to reviewers.
- Core selling point: an **ultra-lightweight, edge-deployable** model for low-resource clinics.
- Novelty must be **grounded in real, documented gaps** of recent SOTA — not accuracy on saturated benchmarks.

## 2. Environment & compute constraints
- Use Python at **`C:\Python312\python.exe`** (the `python`/`py` launchers on PATH are broken stubs).
- **Virtual environment must live inside the project folder** (`f:\Cervaical Cancer\.venv`).
- **No module installation into the global `C:\Python312`** site-packages — project venv only.
- New work targets a **single NVIDIA RTX 5060 Ti (16 GB)** — one consumer GPU, not a cluster.
  - Must use mixed precision (BF16/FP16), gradient accumulation, patch/cell-based processing.
  - (Original baseline was trained on Kaggle 2× Tesla T4.)
- All facts/results must be **grounded and real** (verified against live sources), never fabricated; mark "not reported" where unknown.

## 3. Deliverables requested
- ✅ **Comprehensive end-to-end pipeline documentation** with a diagram → [PIPELINE_DOCUMENTATION.md](PIPELINE_DOCUMENTATION.md)
  - Cover datasets, preprocessing, full pipeline, and **state whether evaluation methods were done**.
- ✅ **Comprehensive literature review as an Excel file** → [research/Cervical Cancer Literature Review.xlsx](research/Cervical%20Cancer%20Literature%20Review.xlsx) + concise [research/LITERATURE_REVIEW.md](research/LITERATURE_REVIEW.md)
  - Columns: **methodology, gaps, novelty, training datasets, testing/evaluation datasets, evaluation results**; grounded & real (June 2026); ~30–40 papers; **find gaps in the field**.
- ✅ **Research prompt** for an AI agent to propose a novel, gap-grounded pipeline → [research/RESEARCH_PROMPT.md](research/RESEARCH_PROMPT.md) (tuned for RTX 5060 Ti).
- ✅ **Verify external AI research outputs** (ChatGPT / Gemini) against live sources → [research/VERIFICATION_REPORTS.md](research/VERIFICATION_REPORTS.md).
- ✅ **Grounding check of the two proposed pipelines** (CerviWISE vs FICCD): which is novel, which will work, + my recommendation → [research/PIPELINE_NOVELTY_ASSESSMENT.md](research/PIPELINE_NOVELTY_ASSESSMENT.md).
- ✅ **Strategic positioning** advice — answered in-chat.
- ✅ **Data plan** → [plans/DATA_PLAN.md](plans/DATA_PLAN.md):
  - ✅ Find **other datasets** (verified landscape); train set (SIPaKMeD); external eval sets; **EDA plan**; **group-aware train/val/test split**; **cross-dataset evaluation** + label mappings; **preprocessing + augmentation** for all datasets.
  - ✅ **Cross-dataset compatibility audit** (species human/animal, stain, preparation variant, data type, compatibility verdict) — folded into the data plan.
- ✅ **Label-alignment method** (keep SIPaKMeD's 5 classes fixed, map other datasets in) → [plans/LABEL_ALIGNMENT_METHOD.md](plans/LABEL_ALIGNMENT_METHOD.md).
- ✅ **Multimodal (image + numerical features) feasibility** → [plans/MULTIMODAL_PLAN.md](plans/MULTIMODAL_PLAN.md).
- ✅ **Ablation + evaluation plan** (in-pipeline ablations, metrics, other-model comparison, cross-dataset eval) → [plans/Ablation and Evaluation plan.md](plans/Ablation%20and%20Evaluation%20plan.md).
- ✅ **Pipeline upgrade plan** (U1–U9: what to upgrade + how to implement, with validated methods/refs; citation-verified) → [plans/PIPELINE_UPGRADE_PLAN.md](plans/PIPELINE_UPGRADE_PLAN.md).
- ✅ **README index** → [README.md](README.md). · **This requirements log** → [REQUIREMENTS.md](REQUIREMENTS.md).

## 4. Pipeline / methodology requirements
- Build on the **existing EfficientNet-B4 (teacher) → B0 (student) knowledge-distillation** baseline ([efficientnetb4-sipakmed.ipynb](efficientnetb4-sipakmed.ipynb)).
- Stay **cell/FOV-level** (no public cervical-cytology WSI data exists for a WSI/MIL pipeline).
- Differentiate on rigor, not accuracy (benchmarks are saturated). Required rigor moves:
  - **External / leave-one-dataset-out validation** (the field's #1 gap; report internal−external Δ).
  - **Probability calibration** (temperature scaling, ECE, reliability diagrams) — framed as good practice, citing **SiCal-CytoNet (2026)**.
  - **Release-side conformal risk auditing under prevalence shift** (the genuinely novel, un-closed angle).
  - **Measured edge efficiency** — params / FLOPs / latency / peak VRAM on the RTX 5060 Ti.
  - **Fix the knowledge-distillation bug** (double-softmax / missing T²; KL term ≈0.005) and run a **no-teacher ablation**.
  - Use a **group-aware 80/10/10 train/val/test split** (don't let the validation set double as the test set).
  - **Hyperparameter tuning via Optuna + Hyperband pruning** on the validation split only (test sealed); ~30–50 early-pruned trials.
- Avoid known pitfalls: don't label the task "ASC-US+" (SIPaKMeD has no ASC-US class); don't claim calibration/stain-invariance/evidential heads as novel; don't pursue WSI/MIL without compatible data.

## 5. Reporting & reproducibility requirements
- Report metrics that are scarce in the field: internal-vs-external ΔAUC, ECE/reliability, sensitivity/specificity at pre-specified thresholds, release-side risk, and measured efficiency.
- Statistical rigor: multiple seeds, confidence intervals, significance tests.
- Leakage-safe protocols (group-aware splits by patient/source slide); release code, seeds, and splits.

## 6. Open items / decisions pending
- ⬜ Confirm RTX 5060 Ti VRAM variant (16 GB assumed; 8 GB changes batch/resolution).
- ⬜ Decide whether to implement **Tier 1 (tabular risk) + fusion**, or keep the paper Tier-2-only.
- ⬜ Turn the rigor moves into a concrete, runnable implementation wired to the notebook (offered).
- ⬜ Diagnose the **92%/65%** number: confirm exactly which dataset/classes/split it was measured on (the "light dysplasia" wording hints at Herlev grading, where ~65% is near the real ceiling).

---
_Add new needs as bullets under the relevant section; flip status markers as work completes._
