# Deliverable 21 — Explicit Novelty Statement

> **This document opens by rejecting our own first proposal**, because the stated quality standard requires it. What survives is smaller, harder, and defensible.

---

## 1. Self-rejection of the original candidate (mandatory, per the quality standard)

**Candidate as originally framed:** *"a flexible, uncertainty-aware cervical screening framework that trains a cytology-image expert and a risk-questionnaire expert on disjoint public cohorts, maps both to a shared binary abnormality axis, calibrates and prior-shift re-anchors each, and combines them with an asymmetric, evidence-weighted rule supporting image-only / questionnaire-only / both, with conformal abstention and conflict flagging."*

**Verdict: REJECTED as written.** It fails three of the seven kill criteria.

### 1.1 "Novelty is only combining existing models" → **FAIL**
Every clause is off-the-shelf, and adversarial search located the owner of each:

| Clause | Prior art that owns it | Verified evidence |
|---|---|---|
| "combines… with an asymmetric evidence-weighted rule supporting image-only/questionnaire-only/both" | **ProbEn** (Chen et al., **ECCV 2022 oral**, arXiv:2104.02904) | "derived from Bayes' rule and first principles"; assumes "conditional independence across modalities"; "through probabilistic marginalization, ProbEn elegantly handles missing modalities" |
| …and one level up | **Genest & Zidek**, *Combining Probability Distributions* (Statistical Science 1986) | a 40-year characterised literature on linear vs logarithmic opinion pools — "aggregating a number of expert opinions… expressed in some numerical form" |
| "calibrated confidence across modality subsets" | **AECF** (arXiv:2505.15417, 2025) | "A novel contrastive loss enforces monotone confidences across *all* 2^M−1 modality subsets"; reports per-dropout ECE (full 0.011→**0.009**; 50% dropout **0.020**) |
| "modality dropout / any-subset image+tabular" | **Gu, Saito & Ma** (MICCAI 2025, arXiv:2509.18284) | image+tabular with learnable modality tokens + improved modality dropout; SOTA "particularly… where only a single modality is available" (AUROC 0.836→0.840; 0.712→0.722) |
| "calibrates and prior-shift re-anchors each" | **Saerens et al. (2002)**; superseded by **Alexandari et al. (ICML 2020)** MLLS+BCTS | Saerens's precondition — p(y\|x) calibrated — is "not true of modern neural networks" |
| "conformal abstention" | Mondrian CP (Olsson et al., Nat Commun 2022); Angelopoulos & Bates | standard |
| "conflict flagging" | ECML/RCML (AAAI 2024); Dempster–Shafer (Shafer 1976) | standard |

**A five-part assembly of published parts is not a contribution.**

### 1.2 "Cannot be experimentally validated" → **FAIL on the headline path**
**No public dataset pairs a cervical image with a risk questionnaire for the same patient** — verified affirmatively, dataset by dataset (absence #1). The decisive test: **UCI 384** ("Quality Assessment of Digital Colposcopies") shares **the same authors, hospital and year** as UCI 383, yet has **no questionnaire attributes, no patient ID, and mismatched cardinality** (287 images vs 858 patients). The single most likely candidate pair in existence is a confirmed miss. The barrier is **consent/de-identification**, so it will not be fixed by re-annotation (CRIC: *"it is not possible to provide demographic information on the participants… including age"*).

Therefore the **both-modalities path — the system's entire reason to exist — has no held-out data on which to measure discrimination or calibration.** Both escapes fail: pseudo-pairing is the "chimeric users" practice that **Poh & Bengio dedicated two papers to interrogating** (2005/2006), and simulation must assume the very dependence structure that **Grill et al. (2017)** show is what breaks calibration. *This cannot be fixed with effort or compute.*

### 1.3 The defect the gap analysis missed — **construct + unit mismatch** (arguably fatal)
The innocuous verb **"maps both to a shared binary abnormality axis"** smuggles in two unvalidated assumptions:

- **Construct.** SIPaKMeD/Herlev labels are **cytomorphological** (koilocytotic, dyskeratotic, parabasal, metaplastic…). UCI's target is a **biopsy/histology** outcome. These are **not two noisy measurements of one latent variable** — they are different diagnostic tests with different reference standards. *This is exactly why UCI ships Hinselmann/Schiller/Cytology/Biopsy as four separate correlated columns rather than one label.*
- **Unit.** SIPaKMeD/Herlev are **cell-level**; UCI is **patient-level**. Fusing P(abnormal cell) with P(biopsy-positive patient) needs a cell→patient aggregation, and **no data exists to calibrate it** (no patient IDs).

**Fusing two constructs measured on two units and calling the output "calibrated P(abnormal)" is a *definitional* problem, not a calibration problem. No amount of prior correction repairs it.**

---

## 2. What we claim instead

> **CO-PRIOR — "Composing disjoint-cohort experts: prior-stripped opinion pooling, and the price of the conditional-independence assumption you cannot avoid paying."**

### 2.1 The single novel claim (everything else is declared borrowed)

**The two-prior composition operator (M4).**

> Every expert emits a posterior contaminated by **its own training-cohort prior**. Composing experts from *different* cohorts therefore requires **stripping each expert's prior out before pooling** and applying a single declared deployment prior **after**:
> `λ_k = logit(p̃_k) − logit(π_k)` , then `logit P = logit(τ) + Σ_k w_k λ_k`.

**Why this is genuinely uncollided.** The literature splits cleanly either side of this operation and **never crosses**:

| Side | Papers | What they do | What they don't |
|---|---|---|---|
| **Prior correction** | Saerens et al. (2002); Godau et al. (MICCAI 2023); Alexandari et al. (ICML 2020) | retarget **ONE** classifier to **ONE** population | never combine several |
| **Classifier combination** | Kittler et al. (1998); Genest & Zidek (1986); ProbEn (2022) | combine **SEVERAL** classifiers | always under a **COMMON** prior |

**Nobody performs prior-stripping as a *precondition of pooling*.** This was the only axis on which the adversarial critic could not find a collision, corroborated by an independent OpenAlex search for prevalence correction when combining classifiers across populations, which returned **nothing on point**.

### 2.2 Why it matters (not merely unclaimed)
1. **It is a hole in the theory**, not an unfilled application slot — which is what survives review.
2. **It is unavoidable.** Two cohorts ⇒ two priors. You either strip them or you silently pool two different priors and count them twice. Herlev's prior is a *construction artefact* (its creators selected cells "not to collect a natural distribution"); UCI's is 6.4%. Neither is a deployment prevalence.
3. **★ It is validatable without any paired data.** Prior-stripping and re-anchoring are evaluated **per expert, on that expert's own cohort** (H2c). This is the property that rescues the programme from the §1.2 kill.

### 2.3 The two supporting contributions
- **The leakage-audited UCI re-benchmark (RQ1).** No paper satisfies all three of {`Dx:*` dropped, resampling confined inside folds, calibration reported}. Our measurement: Biopsy **AUROC 0.656 [0.584, 0.720]** under nested CV; `Dx:Cancer` from the other `Dx` columns gives **0.998** vs **0.592** from legitimate features; an 8-family tuned bake-off cannot exceed **≈0.65**; no single feature exceeds univariate AUROC **0.572**. Against the honest anchor of the dataset's own creators (Fernandes, Chicco et al., PeerJ CS 2018 — nested CV, no oversampling), the published 0.98–1.00 is **fully accounted for by leakage**, with essentially **no residual signal**.
- **The price of the forced assumption (RQ3).** With zero overlap the dependence ρ is **non-identified**, so independence is *forced, not chosen*. We transport **Grill et al.'s (2017)** quantified, **directional** penalty (over-estimation of risk **up to 56%**, ≈27% on real data) into an ML fusion system as a **pre-registered, falsifiable prediction about our own output**, and report an **envelope over ρ** instead of a single optimistic number.

### 2.4 Explicitly **not** claimed
- ✗ Novelty in the fusion rule → **ProbEn / opinion pooling**.
- ✗ Novelty in per-subset calibration → **AECF** (which measures it; we will cite it and will **not** quote TMUR's 26.50% ECE as "the field's best").
- ✗ Novelty in modality dropout / any-subset inference → **Gu et al. (MICCAI 2025)**; the vacuous-opinion mechanism is **ProbEn's marginalisation** and Dempster's identity element.
- ✗ Novelty in conformal or evidential machinery.
- ✗ **Any joint conformal guarantee** — per-expert marginal guarantees do not compose (Podkopaev & Ramdas's fix needs unlabeled target data we do not have).
- ✗ **Any joint patient-level performance number** — unvalidatable (§1.2).
- ✗ That "pseudo-pairing is uncritiqued" — **refuted** by Poh & Bengio (2005/2006), whose conclusions we have **not** been able to read and therefore do not characterise.

---

## 3. Does the reframed proposal pass the quality standard?

| Kill criterion | Verdict | Why |
|---|---|---|
| Novelty is only combining existing models | **PASS** | novelty re-centred on **M4 alone**; all other parts declared borrowed **in the abstract** |
| Lacks literature support | **PASS** | 155 verified papers; every gap ≥2 independent sources; contrary evidence honoured |
| Cannot be clinically justified | **PASS (scoped)** | reframed as *triage-support + an operator*, not a diagnostic device; safety asymmetry; deferral |
| Lacks uncertainty estimation | **PASS** | calibrated probability (ECE/Brier) **separated** from evidential *u* (ranking-only, per Jürgens/TMUR) |
| Cannot handle missing modalities | **PASS** | absent expert ⇒ `λ_k = 0`, the additive identity; no imputation (impossible at zero overlap) |
| Overlaps heavily with prior work | **PASS** | the *system* would have; the **operator** does not |
| Cannot be experimentally validated | **PASS (by reframing)** | the claim is now **per-expert** (H2c) — single-cohort, public data. The unvalidatable joint path is **reported as the finding (RQ5)**, not asserted as a result |

**Conclusion.** The original proposal was a system whose novelty was an assembly and whose headline was untestable. The reframed proposal is a **theory-level operator** with a **validatable per-expert claim**, a **standalone leakage audit**, and a **pre-registered falsifier** — and it treats the impossibility of joint validation as its most interesting finding rather than its embarrassment.

---

## 4. One-sentence statement

> **We contribute the two-prior composition operator — prior-stripping as a precondition of pooling experts trained on disjoint cohorts, the one operation the prior-correction and classifier-combination literatures each leave to the other — together with a leakage-audited honest baseline for the UCI cervical questionnaire (AUROC 0.656 vs a published 0.98–1.00 that is fully explained by leakage) and a pre-registered, directional prediction of the over-confidence that the forced conditional-independence assumption must produce; every other component of the system is explicitly borrowed, and the joint image+questionnaire path is reported as unvalidatable because no public cohort pairs the two modalities for the same patient.**
