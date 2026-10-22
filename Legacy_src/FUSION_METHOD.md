# Any-modality cervical screening — method & grounding (v2)

**Goal.** One deployable system that combines two experts trained on **two distinct, unpaired datasets** (no shared patients, different label spaces): the frozen **SIPaKMeD cytology-image expert** in `legacy_src` and a new **UCI risk-factor tabular expert**. It must accept **either modality or both**, run whichever expert's modality is present, combine when both are present, and **degrade gracefully** — *"if one uses any form of data to classify, the model runs the model specified for that form of data."* Distinct from `legacy/another paper.pdf` (Alif et al., *Array* 2026), which co-trains one CNN-GRU with epoch-wise stochastic modality masking on randomly pseudo-paired rows.

> **v2 note.** An adversarial review (statistical / clinical / ML-methodology) rated the first design **FLAWED** and it was corrected. What changed and why is in §6. Read §5 before quoting any number.

## 1. The two experts and the shared axis
| | Image expert (`legacy_src`, frozen) | Tabular expert (new) |
|---|---|---|
| Input | SIPaKMeD Pap-smear single cell | UCI risk-factor questionnaire (858 patients) |
| Native label | 5-class cell morphology | 4 binary screening targets + any-of-4 |
| **Shared output** | `P_img(abnormal)=P(Koilo)+P(Dysk)`, calibrated | `P_tab(positive)`, calibrated per target |
| **Honest AUROC** | 0.998 in-distribution *(leaky split)*; **0.53–0.73 external** (Herlev 0.73, HiCervix 0.59, APACC 0.53) | **Biopsy 0.63 [0.55, 0.70]**, Schiller 0.62; Hinselmann 0.45 & Citology 0.49 are **≤ chance** |

The two meet only in a latent binary **"abnormal / SIL-risk"** construct — and even that bridge is weak (see §5, koilocytosis caveat).

## 2. Why not the reference paper's approach
It fabricates a joint distribution (random label-pairing injects a genuinely-zero cross-modal correlation, which its own limitations concede), and it discards the frozen CNN to co-train end-to-end. We keep both experts frozen and combine **posteriors** on the shared axis — the datasets are never joined.

## 3. Method (v2) — prevalence-anchored, asymmetric, calibrated combiner
1. **Per-expert calibration** on each expert's own held-out data; report ECE (reuse `evaluate.expected_calibration_error`).
2. **Prior-shift re-anchoring** — each posterior is calibrated to a *different* prior (SIPaKMeD ~40% vs UCI ~6%), so before any pooling both are re-anchored to one **declared deployment prevalence** via the per-sample label-shift correction (**Saerens, Latinne & Decaestecker, Neural Comp. 2002**). `fusion.prior_shift`.
3. **Combiner = Kittler robust SUM rule** on the anchored posteriors (**Kittler, Hatef, Duin & Matas, "On Combining Classifiers", IEEE TPAMI 1998**) — the operating default because the tiny UCI positive class makes the product/log-LR rule high-variance and overconfident. The naive-Bayes/log-LR rule is available (`mode='nb'`) but flagged overconfident under the experts' shared-HPV dependence.
4. **Asymmetric cascade + safety floor** (`mode='cascade'`, default) — tabular is an *upstream pre-test risk*, image a *downstream finding*, so risk may **raise** suspicion or trigger abstention but may **never lower a positive cytology** (`p_fused = max(sum_rule, P_img)` when the image call is positive). This is how real HPV→cytology→colposcopy screening operates and it avoids the sign-inversion trap.
5. **Abstention = split-conformal risk control** (**Angelopoulos & Bates, 2023**), calibrated per-expert on each expert's own labelled data (no paired cohort needed) — replaces hand-set thresholds. Dempster–Shafer is retained **only** as a conflict *flag* (report `K` when the experts strongly disagree), never as the point estimate.
6. **Routing / graceful degradation** — one modality present → that (anchored) expert alone; missing modality = a dropped term. Structural, not learned.

## 4. The tabular expert (`tabular.py`) — leakage-safe
Drop `Dx:*` (prior-diagnosis leakage) and the two ~92%-missing columns; `HistGradientBoosting` ingests remaining NaNs natively; class-balanced sample weights; **out-of-fold AUROC with a bootstrap 95% CI**; targets flagged **≤ chance** are excluded from combination. Verified: the literature's 98% is **SMOTE-before-split leakage** — and including the `Dx:*` columns changes Biopsy AUROC by **0.000**, so they were never the signal.

## 5. What we may and may not claim (the honest core)
- **Tabular has a hard ceiling ≈ 0.65.** Across HistGB/XGBoost/LogReg × {±Dx, ±missingness flags} × {Biopsy, Schiller, any-of-4}, nothing exceeds ~0.65 without leakage. UCI risk factors are intrinsically weak predictors of biopsy outcome.
- **Symmetric equal-weight fusion HURTS at the real marginals.** The dependence-sensitivity envelope (`eval_fusion.py`, real marginals, Gaussian-score model) shows: in-distribution (image 0.998) fused = 0.99→0.95 as dependence rises — always **below** image alone; external (image 0.73) fused = **+0.016 under independence, −0.030 under realistic positive dependence**. The shared HPV cause biases the true dependence **positive**, exactly where fusion fails. Hence the **asymmetric cascade with an image floor** — which recovers image-only performance and lets tabular only *raise* — is the sole defensible combiner.
- **No joint patient-level accuracy is measurable** — no dataset pairs a cell with a patient; the multimodal claim is **unfalsifiable** here, not merely unvalidated. Honest validation = per-expert marginal metrics (with CIs) + the dependence envelope (no fabricated favorable point) + structural properties. A real joint claim needs a small **prospective paired pilot** (n≈50–100, DeLong test).
- **Construct caveat.** The image "abnormal" class is koilocytosis-dominated; koilocytosis marks productive/often-transient HPV and is a **negative** predictor of high-grade CIN2+ — so `P_img(abnormal)` and the Biopsy target can be weakly/oppositely aligned. Another reason symmetric fusion is unsafe and the cascade only *raises* suspicion.
- **Deliverable, stated honestly:** *a unified, calibrated, prevalence-aware, any-modality interface with principled missing-modality routing and guaranteed abstention* — **not** "fusion raises accuracy."

## 6. What changed from v1 (adversarial corrections)
| v1 (flawed) | v2 (corrected) |
|---|---|
| `simulate_joint()` "fused ≥ best single" | **Deleted** — it assumed independence + a fictional tabular AUROC 0.98 (circular). Replaced by the real-marginal dependence envelope. |
| Symmetric Dempster–Shafer as headline; `u=1−r` constant "TMC" | DS demoted to a **conflict flag**; `u=1−r` carried no per-sample info (mis-attributed to TMC). |
| `base_rate=0.5`; `poe_combine` asserts one shared prior | **Prior-shift re-anchoring** to a declared deployment prevalence (Saerens 2002). |
| "GMU-style gate" (learned) | Plain routing + **conformal** abstention (Angelopoulos–Bates); GMU citation dropped. |
| All 4 targets fused | **Drop ≤-chance targets** (Hinselmann, Citology); Biopsy primary. |
| Symmetric "fusion improves accuracy" | **Asymmetric cascade + safety floor**; fusion claim withdrawn. |

## 7. Implementation map
- [`src/tabular.py`](src/tabular.py) — UCI load, leakage-safe features, 4+composite calibrated heads, OOF AUROC + bootstrap CI, ≤-chance flagging.
- [`src/fusion.py`](src/fusion.py) — prior-shift, Kittler sum/nb rules, cascade + safety asymmetry, conformal abstention, DS conflict flag, any-subset `predict()`.
- [`src/eval_fusion.py`](src/eval_fusion.py) — real per-expert AUROCs, dependence-sensitivity envelope (in-dist + external), routing demo. **No** joint-accuracy claim.
- *Future (not yet built):* attention-MIL cell→patient aggregation (Ilse et al., ICML 2018) to make the estimands commensurate; a prospective paired pilot for a real joint claim.

## References (verified, correctly attributed)
- Kittler, Hatef, Duin, Matas. *On Combining Classifiers.* IEEE TPAMI 20(3):226–239, 1998.
- Saerens, Latinne, Decaestecker. *Adjusting the outputs of a classifier to new a priori probabilities.* Neural Computation 14(1):21–41, 2002.
- Angelopoulos, Bates. *Conformal Prediction: A Gentle Introduction.* FnT ML, 2023. https://arxiv.org/abs/2107.07511
- Kittler-companion / prior art still cited: Sensoy et al., *Evidential Deep Learning*, NeurIPS 2018 (https://papers.nips.cc/paper/7580) and Han et al., *Trusted Multi-View Classification*, ICLR 2021 → TPAMI 2023 (https://arxiv.org/abs/2204.11423) — used **only** for the DS conflict flag, not the point estimate.
- Ilse, Tomczak, Welling. *Attention-based Deep MIL.* ICML 2018. https://arxiv.org/abs/1802.04712 (future cell→patient aggregation).
- Huang et al. *Fusion of medical imaging and EHR using deep learning.* npj Digital Medicine 3:136, 2020. https://www.nature.com/articles/s41746-020-00341-z (paired-fusion context).
- Fernandes, Cardoso, Fernandes. *Cervical Cancer (Risk Factors)*, UCI, 2017. https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors
- Reference to differ from: Alif et al. *Multimodal fusion of visual and auditory biomarkers…* Array 30:100787, 2026 (`legacy/another paper.pdf`).
