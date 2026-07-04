# Cervical Cancer Screening AI — Project Index

A two-tier cervical-cancer screening project. The implemented work is **Tier 2**: a lightweight, edge-deployable cervical **cytology** classifier — an EfficientNet-B4 teacher knowledge-distilled into an EfficientNet-B0 student on **SIPaKMeD** — repositioned around **external generalization, calibration, and validated edge efficiency** rather than (saturated) internal accuracy.

_Last updated: 2026-06-21_

## Repository layout
```
/  (root)      active, current docs + the notebook
  research/    investigations already completed (reference)
  plans/       specs for future implementation
  legacy/      original pre-handover inputs (superseded)
```

## Root — active / current
| File | What it is |
|---|---|
| [PIPELINE_DOCUMENTATION.md](PIPELINE_DOCUMENTATION.md) | the current B4→B0 model, end-to-end, with diagram + what evaluation was/wasn't done |
| [REQUIREMENTS.md](REQUIREMENTS.md) | living log of all project requirements + open decisions |
| `efficientnetb4-sipakmed.ipynb` | the training notebook (Tier 2) |
| `.venv/` | project-local Python env (`C:\Python312` interpreter; no global installs) |

## `plans/` — future implementation
| File | What it is |
|---|---|
| [PIPELINE_UPGRADE_PLAN.md](plans/PIPELINE_UPGRADE_PLAN.md) | **the upgrade roadmap** — what to upgrade (U1–U9), how to implement each, with validated method + references |
| [DATA_PLAN.md](plans/DATA_PLAN.md) | datasets, group-aware train/val/test split, SIPaKMeD⇄other label mappings, **cross-dataset compatibility matrix**, EDA, preprocessing, augmentation |
| [Ablation and Evaluation plan.md](plans/Ablation%20and%20Evaluation%20plan.md) | evaluation metrics, 9-arm ablation, model comparison, cross-dataset protocol |
| [LABEL_ALIGNMENT_METHOD.md](plans/LABEL_ALIGNMENT_METHOD.md) | how to map other datasets into SIPaKMeD's 5 classes (crosswalk + partial-label + pseudo-label/open-set) |
| [MULTIMODAL_PLAN.md](plans/MULTIMODAL_PLAN.md) | fusing image + numerical (handcrafted-feature) data; what's possible and what isn't |

## `research/` — investigations done
| File | What it is |
|---|---|
| [LITERATURE_REVIEW.md](research/LITERATURE_REVIEW.md) · [Cervical Cancer Literature Review.xlsx](research/Cervical%20Cancer%20Literature%20Review.xlsx) | 45 source-verified papers (2023–2026) + 12 field gaps + ranked opportunities |
| [PIPELINE_NOVELTY_ASSESSMENT.md](research/PIPELINE_NOVELTY_ASSESSMENT.md) | novelty/feasibility verdict on the CerviWISE & FICCD proposals + a buildable alternative |
| [VERIFICATION_REPORTS.md](research/VERIFICATION_REPORTS.md) | web-grounded fact-check of the external (ChatGPT) research report |
| [RESEARCH_PROMPT.md](research/RESEARCH_PROMPT.md) | reusable prompt to design a gap-grounded, RTX-5060-Ti-feasible pipeline |
| `deep-research-report.md` | raw external proposals (ChatGPT CerviWISE + Gemini FICCD) |
| `build_lit_review_xlsx.py`, `lit_review_data.json` | regenerate the literature workbook |

## `legacy/` — original inputs (superseded)
| File | What it is |
|---|---|
| `Cervical Cancer Research.xlsx` | original prior-work literature workbook (predates this work) |
| `Cervical Cancer Paper Pipeline.docx` | original project roadmap / upgrade log |

## Key facts to keep in view
- **Train space = SIPaKMeD 5 classes**, trained on **SIPaKMeD only**; other datasets are for **external evaluation** (via label mapping) and optional **label-free SSL pretraining**.
- **All datasets are human** cervical cells — no animal data. The only purely **non-image** data is UCI risk-factors (tabular) and the SIPaKMeD/Herlev numeric feature tables.
- **All datasets are kept, each with a defined role.** Direct Pap-cytology eval: Tier-1 (Herlev/APACC/HiCervix) and Tier-2 (crop+relabel). The stain/modality-incompatible ones are retained with dedicated paths: **Mendeley-LBC (H&E)** → stain-normalized secondary probe; **CCAgT (AgNOR)** → OOD/robustness test; **histology WSI** → separate track / OOD; **UCI** → Tier-1 risk cascade; **feature tables** → multimodal fusion. Full matrix + roles in [DATA_PLAN.md §1b](plans/DATA_PLAN.md).
- **Cross-dataset scoring is at binary / 3-class level** (the 5-class→other mapping is one-to-many/no-map at the fine level).
- **Structural blind spot:** the SIPaKMeD model **cannot detect HSIL or glandular lesions** — state wherever screening performance is reported.
- ⚠️ The notebook's **97.11 %** used a leaky random split; re-derive under the **cluster-grouped** split. The newer **92%/65%** needs its dataset/classes confirmed (possibly Herlev grading).
- **Compute:** single **RTX 5060 Ti (16 GB)**; **Python at `C:\Python312`**, project venv at `.venv`, no global installs.

## Environment
```
"C:\Python312\python.exe" -m venv ".venv"
.venv\Scripts\python.exe -m pip install -U <pkgs>   # never install into C:\Python312
```
