# Phase 15 — Risk Analysis · Deliverable 20 — Reproducibility Checklist

---

## Phase 15 — Risk analysis

Severity × likelihood are judgements; each risk names its evidence and a **concrete** mitigation.

### 15.1 Clinical risks

| ID | Risk | Sev | Evidence | Mitigation |
|---|---|:--:|---|---|
| **C1** | **A weak questionnaire lowers a positive cytology call → missed lesion.** Sensitivity/NPV dominate in screening. | **High** | Kelebet et al. (2025): fused **92%** < tabular-alone **96%** — fusion demonstrably degrades performance | **Safety asymmetry**: the questionnaire may only *raise* suspicion or trigger deferral, never reduce a positive image call. Pre-registered falsifier H3b. |
| **C2** | **Over-confidence in the fused score.** | **High** | Grill et al. (2017): forced conditional independence over-estimates risk **up to 56%** (≈27% real) | Report **signed** calibration error (H3a); widen abstention thresholds; never ship an uncalibrated fused probability. |
| **C3** | **Construct error**: `p_img` is koilocytosis-dominated (productive, often transient HPV) and is *not* a histology-risk proxy; treating it as one may invert the intended signal. | **High** | Critic's defect #6; UCI ships 4 separate correlated targets *because* they are different tests | Latent-class framing (Rutjes et al. 2007); report composition **conditional on** the assumption; never emit an unconditional "P(cancer)". |
| **C4** | **Unit error**: P(abnormal *cell*) fused with P(biopsy-positive *patient*). | **High** | absence #1: SIPaKMeD/Herlev have **no patient IDs** → aggregation is uncalibratable | Scope the system to *per-cell triage + per-patient risk*, reported separately; do **not** claim one patient-level probability. |
| **C5** | Abstention rate exceeds cytologist capacity → the tool is unusable. | Med | Xue et al. (2023) triage at a fixed 0.35 threshold; no work validates abstention vs staffing | Report risk–coverage against an explicit workload budget. |
| **C6** | Automation bias: clinicians defer to a wrong-but-explained model. | Med | **Jabbour et al. (JAMA 2023, n=457)**: explanations **did not mitigate** biased-model harm | Do **not** claim XAI confers safety; require a reader study before any such claim. |

### 15.2 Engineering / methodological risks

| ID | Risk | Sev | Evidence | Mitigation |
|---|---|:--:|---|---|
| **E1** | **Leakage silently re-enters** (resampling before split; `Dx:*` back in). | **High** | measured: 0.592 → **0.998** with `Dx:*` ⟨OWN⟩ | Leakage is enforced **in code**, not prose (`Literature/build_database.py` pattern); A3/A4 ablations are mandatory; publish the audit. |
| **E2** | `π_k` is unknowable for curated cohorts (Herlev selected cells "not to collect a natural distribution"). | **High** | ⟨V⟩ | Treat π_k as a **declared, audited** input; ablation A7 sweeps ±50%; report sensitivity. |
| **E3** | Prior-stripping applied to an **uncalibrated** posterior ⇒ λ is not an LLR ⇒ the whole operator is invalid. | **High** | Alexandari (ICML 2020): Saerens presupposes calibration, "not true of modern neural networks" | M3 (BCTS+MLLS) is a **precondition**, with per-expert ECE reported. |
| **E4** | Claiming a **joint** conformal guarantee. | **High** | G5; Podkopaev & Ramdas (UAI 2021) — fix needs unlabeled target data, **unavailable here** | Claim **per-path marginal coverage only**; state the limitation explicitly. |
| **E5** | Using evidential *u* as a probability or a weight. | Med | Jürgens (ICML 2024): *u* is "relative (rather than absolute)"; TMUR: scales not comparable | *u* is **ranking-only** (A14 tests the degradation). |
| **E6** | The envelope (Eq. 6) is mistaken for a measurement. | Med | it is a Gaussian-score model | Label every envelope figure "**bound, not measurement**". |
| **E7** | Novelty collision at review (ProbEn / AECF / Genest & Zidek / Gu et al.). | **High** | verified collisions | Declare all borrowed parts **in the abstract**; centre novelty solely on M4. |

### 15.3 Dataset bias & fairness

| ID | Risk | Sev | Evidence | Mitigation |
|---|---|:--:|---|---|
| **F1** | **Single-population tabular arm**: all 858 patients from one Venezuelan hospital (2012–13). Any risk model inherits that population's structure; deploying elsewhere is unvalidated. | **High** | absence #10: zero external validation in the entire cervical-tabular literature | State the population explicitly; **do not** claim generalisation; τ must be re-declared per deployment. |
| **F2** | Risk-factor variables (partners, STDs, contraception) encode **behaviour and stigma** → potential for discriminatory triage. | **High** | — | Report subgroup performance where recorded; prefer the *image* arm for decisions; questionnaire raises-only (C1). |
| **F3** | Image datasets are lab-curated, balanced, pre-segmented single cells — not whole-slide reality. | **High** | in-house corpus theme; 27–47 pt external collapse ⟨OWN⟩ | External validation is mandatory in every table; report in-dist **and** external. |
| **F4** | No demographic metadata exists in the image sets (consent/de-identification) → **fairness cannot be audited**. | **High** | CRIC: "not possible to provide demographic information… including age" | State the impossibility as a finding; require prospective collection for any fairness claim. |

### 15.4 Ethical / regulatory

| ID | Risk | Sev | Mitigation |
|---|---|:--:|---|
| **R1** | Screening tool → a false negative is a missed cancer. | **High** | Safety asymmetry (C1); deferral; never auto-release a negative on a lone weak modality. |
| **R2** | **Pseudo-pairing ("chimeric users")** fabricates virtual patients from two cohorts. | **High** | **Poh & Bengio (2005/2006) dedicated two papers to interrogating exactly this** — and we could **not read their conclusions** (publisher-elided). **Retrieve before any use**; currently we do **not** pseudo-pair. |
| **R3** | Regulatory: an uncalibrated, unvalidated joint path would not survive review (FDA/CE evidence expectations). | **High** | We claim no joint performance; the deliverable is an operator + audit, not a device. |
| **R4** | Licence violation via FM reuse. | Med | Prov-GigaPath: "**Any deployed use case — commercial or otherwise — is out of scope**"; UNI/CONCH CC-BY-NC-ND. Prefer **Smart-CCS (Apache-2.0)** / **UniCAS (CC BY 4.0)**. |
| **R5** | Data licence for redistribution. | Low | We redistribute **no** PDFs; only OA-licensed copies were downloaded, closed items are abstract-only with reasons logged. |

### 15.5 Top failure modes (ranked)

1. **Definitional collapse** (C3+C4) — fusing two constructs on two units and calling it P(abnormal). *Mitigation: make it the object of study, not a step.*
2. **Unvalidatable headline** — the both-modality path has no test set. *Mitigation: RQ5 reframing; falsifier.*
3. **Novelty collision** (E7) — everything except M4 is published. *Mitigation: declare borrowings up front.*
4. **Leakage relapse** (E1).
5. **Over-confidence** (C2) — predicted and directional, so at least testable.

---

## Deliverable 20 — Reproducibility checklist

| Item | Status |
|---|---|
| **Literature corpus** | `Literature/CSV/papers.csv` — 155 papers × 39 cols; every row carries URL, DOI, verification status, **verbatim evidence quote**, extraction confidence |
| **Provenance of every cell** | `source` (author-stated) vs `derived` (conservative keyword match) vs literal `"not reported"`; **no value invented** |
| **Anti-fabrication enforcement** | verifier verdicts applied **in code** (`build_database.py`): 2 fabricated quotes removed, 3 unsupported result-sets blanked, 3 metadata errors flagged |
| **Acquisition log** | `Literature/Metadata/acquisition_log.csv` — status **and reason** for all 155 (107 full text: 64 OA PDFs + 43 Europe PMC OA XML; 28 abstract-only; 20 metadata-only) |
| **Legal policy** | PDFs only from Unpaywall-confirmed OA / arXiv / PMC OA subset; **no paywall bypassed**; reasons recorded |
| **Rebuild command** | `python Literature/build_database.py` then `python Literature/acquire_sources.py` |
| **Unverified leads quarantined** | MC2, Uddin 2024, the "88.9–92.6%" review, UAD-FM, Poh & Bengio conclusions — listed, **not cited** |
| **Empirical results** ⟨OWN⟩ | UCI: nested stratified CV, `Dx:*` dropped, no pre-split resampling; seeds fixed (42, 7); BCa bootstrap 10,000; scripts in `legacy_src/src/tabular.py` |
| **Reported with CIs** | yes — no point estimate without an interval |
| **Pre-registration** | RQs + hypotheses + **global falsifier** fixed before fitting (`03_OPPORTUNITIES_RQ_HYPOTHESES.md`) |
| **Negative-result commitment** | explicit: if fusion loses to `max(single)` on discrimination **and** calibration, the null is the reported result |
| **Environment** | Python 3.12, PyTorch 2.11+cu128, scikit-learn 1.9, XGBoost 3.3; single RTX 5060 Ti (16 GB) |
| **Known non-reproducible** | the **joint/paired** path — no dataset exists (absence #1). Stated, not worked around. |
| **Data licences** | UCI 383 CC BY 4.0; SIPaKMeD/Herlev per source terms; Smart-CCS Apache-2.0; UniCAS CC BY 4.0 |
