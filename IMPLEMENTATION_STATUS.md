# New-Pipeline Implementation Status (vs. `plans/`)

Definitive map of every plan item to its implementation in `src/`. Status legend:
**✅ done & verified** (imports + CPU/real-data smoke) · **⚠️ partial / needs note** · **❌ deferred** (with reason).
Compute: single RTX 5060 Ti. All modules import cleanly with `PYTHONPATH=src`. Nothing here claims a
*result* that hasn't been run — the GPU training/eval is driven by `new_pipeline.ipynb`.

## Research framing & provenance (NEW — drives the whole pipeline)
| Item | Status | Where |
|---|---|---|
| Pre-registered RQs / hypotheses / contributions (positive, anchored to **literature gaps** G1–G7; no "set-to-fail" experiments) | ✅ | `research.py` → `results/metrics/research_framing.json` |
| All-dataset EDA (SIPaKMeD + Herlev + APACC + HiCervix + CRIC): native-class counts, E0 mapping coverage + excluded fraction, image-size stats, corrupt/0-byte scan, montages | ✅ | `data/eda.py` → `results/metrics/eda/*.json|csv`, `results/figures/eda/*` |
| **Full provenance:** every figure persists a companion `.csv` (+`.json` for arrays) so each diagram is regenerable | ✅ | `plotting.py` (`write_csv`/`write_json` in every fig fn) |
| Preflight: verify every dataset downloaded (auto-fetch via legacy DL) + guard every legacy failure (0-byte/corrupt, `ssl.py` shadow, drive-letter case, GPU/VRAM/BF16, num_workers) | ✅ | `preflight.py` → `results/metrics/preflight.json` |
| Split protocol: cluster-grouped 80/10/10, stratified, seeded, **sealed test**; train/val only for fitting; internal test only at the end | ✅ | `data/splits.py`, `train.py`, `run.evaluate` |

## PIPELINE_UPGRADE_PLAN.md (U1–U9 + U2b)
| Upgrade | Status | Where |
|---|---|---|
| **U1** leakage-safe grouped 80/10/10 split + seeds + bootstrap CI + McNemar + Wilcoxon | ✅ | `data/splits.py`, `eval/metrics.py` |
| **U2** fixed Hinton KD (T²+logits) + buggy-KD control + ablation | ✅ | `models/kd.py`, `ablation.py` (A2) |
| **U2b** Optuna+Hyperband HPO, val-only, refit-best+re-eval; tabular XGBoost study | ✅ | `hpo.py` |
| **U3** cross-dataset E0/E1/E-morph, excluded-fraction, internal−external Δ | ✅ | `eval/metrics.py`, `run.py` |
| **U4** stain norm (Macenko/Reinhard) wired into transform/train/eval (train-only fit) + MixStyle module | ✅ (MixStyle-in-model arm ⚠️) | `data/stain.py`, `data/dataset.py` |
| **U5** temperature calibration + ECE + reliability (in-dist & external) | ✅ | `eval/calibration.py`, `run.py`, `eval/metrics.py` |
| **U6** efficiency (params/FLOPs/latency/VRAM) | ✅ | `eval/efficiency.py` |
| **U7** multimodal **late fusion** (image+tabular meta-learner) + tabular XGBoost (val-MF1≈0.89) | ✅ | `explain/features.py` |
| **U7** early/joint fusion (DAFT/FiLM) | ⚠️ built but **dropped from scope** (standalone, optional; nothing depends on it) | `models/fusion.py` |
| **U8** Grad-CAM family + IG/LIME/SHAP-pixel + feature-SHAP + exemplar/counterfactual + ProtoPNet | ✅ | `explain/{cam,attribution,features,exemplar}.py`, `models/protopnet.py` |
| **U8** TCAV | ⚠️ scaffold (synthetic-concept smoke; needs curated concept sets) | `explain/concept.py` |
| **U9** conformal prediction (split + Mondrian, LAC/APS) | ✅ | `eval/conformal.py` |

## DATA_PLAN.md
| Item | Status | Where |
|---|---|---|
| cell→cluster map (4,049→966), group-aware split | ✅ | `data/clustering.py`, `data/splits.py` |
| 224px aspect-preserving pad; augmentation tiers | ✅ | `data/dataset.py` |
| Route-A E0/E1 deterministic crosswalk + per-class candidate sets | ✅ | `data/labelmap.py` |
| Tier-1 datasets (Herlev, APACC, HiCervix) downloaded + wired | ✅ | `data/` + `data/raw/` |
| Tier-2 **CRIC** downloaded + point-crop loader wired (11,534 cells, E0/E1) | ✅ | `data/external_tier2.py`, `run.py` |
| Tier-2 APCData / RIVA / BMT | ❌ not fetchable here (not on Kaggle; Zenodo firewalled) — **loader supports them when provided** |
| Tier-3 Mendeley-LBC / CCAgT / histology / UCI | ❌ special roles, not downloaded |
| EDA (cluster/split report, class balance, mapping-coverage/excluded fraction) | ✅ core; ⚠️ duplicate-hash / inter-rater / label-noise not automated |
| Resolution harmonization (µm/px) | ⚠️ fixed-px point crop (not µm-calibrated) |

## LABEL_ALIGNMENT_METHOD.md
| Route | Status | Where |
|---|---|---|
| **A** deterministic NO-MAP crosswalk (candidate sets) | ✅ | `data/labelmap.py` |
| **C** partial-label marginal+exclusion loss + multi-head training | ✅ (built+smoke; needs empirical validation) | `route_c.py` |
| **B** OOD/open-set gate (Mahalanobis+energy) + constrained pseudo-labeling | ✅ (built+smoke; needs empirical validation) | `route_b.py` |
| SSL pretraining (SimCLR, 46,721 pooled label-free crops) | ✅ (built+smoke; GPU run pending) | `selfsup.py` |
| Orphan-class rescue (CellGAN synthesis) | ✅ (built+smoke; class-conditional DCGAN, trains on TRAIN split only; RQ5 augment arm) | `cellgan.py` |
| Cytopathologist crosswalk validation (Cohen/Fleiss κ) | ❌ requires human experts — not code |

## MULTIMODAL_PLAN.md
| Option | Status |
|---|---|
| Late/decision fusion (recommended) + tabular XGBoost + SHAP reasons | ✅ `explain/features.py` |
| Early concat / joint DAFT-FiLM | ⚠️ `models/fusion.py` built, dropped from scope (optional) |
| UCI two-tier cascade | ❌ deferred (UCI not downloaded; patient-level) |

## Ablation and Evaluation plan.md
| Item | Status |
|---|---|
| §1.1 in-dist 5-class (acc/bal-acc/MF1/κ/per-class/CM/OvR-AUROC) + **95% bootstrap CI** + calibration + efficiency | ✅ |
| §1.2 cross-dataset E0/E1/E-morph + excluded-fraction + internal−external Δ + **external ECE** | ✅ |
| §1.3 McNemar + Wilcoxon + bootstrap CI; ≥3-seed/CV | ✅ (CV via `cv_folds`; full multi-seed retrain loop ⚠️ available not auto-run) |
| §2 ablations A1,A2,A3,A4,A5,A6(weights/focal/oversample),A7,A8 | ✅ (18 arms) |
| §2 A5 "+stain-aug" (MixStyle-in-model) | ⚠️ MixStyle module exists; in-backbone insertion not wired |
| §2 A9 SSL-init arm | ✅ (`ssl_init=True`; needs `selfsup.pretrain` first) |
| §2b HPO (image + tabular) | ✅ |
| §3 model comparison: B0/B2/MNv3/ShuffleNetV2/ResNet50/DenseNet121/Swin-T/MaxViT-T + runner | ✅ `ablation.model_comparison()` |
| §3 published-SOTA (CCanNet/MaxCerVixT/CerviXpert) | ✅ reference reimpls under identical protocol | `models/sota.py`, `ablation.sota_comparison()` |
| §4 prep-stratified read (conventional vs LBC) + stain-norm interaction | ✅ |
| §4 LODO | ⚠️ deferred (optional; needs multi-dataset binary training) |
| §5 reporting (lead-with-balanced, excluded fraction, blind-spot, release seeds/splits/maps) | ✅ |

## Bottom line
The new pipeline is framed as a **solution to cervical-cytology DL literature gaps** (G1–G7), not an audit:
each RQ asks whether a specific upgrade *works* (leakage-safe benchmark, domain generalization, calibrated
abstention, cross-taxonomy learning, minority rescue, deployable+SOTA, explainable+multimodal). Every
plan-committed item reachable in code/data is implemented and import/runtime-verified, including the two
previously-deferred items: **CellGAN orphan-class rescue** (`cellgan.py`) and **published-SOTA reimpls**
(`models/sota.py`). All datasets are present (preflight green: SIPaKMeD/Herlev/APACC/HiCervix/CRIC, 0 corrupt,
RTX 5060 Ti + BF16). Every value and figure is persisted as JSON/CSV.

Remaining honest gaps: (1) datasets this environment cannot fetch (APCData/RIVA/BMT/Mendeley — loaders ready);
(2) items needing human experts (κ validation); (3) small optional arms (MixStyle-in-model, LODO) and the
early/joint fusion dropped from scope. Research-grade modules (Routes B/C, SSL, ProtoPNet, CellGAN, SOTA) are
built and smoke-tested; the **GPU training runs that produce the headline numbers are pending** (driven by
`new_pipeline.ipynb` / `run.py all`) — nothing here claims a result that hasn't been run.
