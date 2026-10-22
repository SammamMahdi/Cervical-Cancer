# Phases 7–9 — Opportunity Ranking, Research Questions, Hypotheses

---

## Phase 7 — Opportunity ranking

**Scoring:** 1–5 per criterion (5 = best). *Computational requirements* is scored as **low-cost = 5** (i.e. cheaper is better). Scores are judgements grounded in the Phase-6 evidence; the justification column names the deciding evidence.

| # | Candidate direction | Novelty | Clin. impact | Feasibility | Data avail. | Publ. potential | Compute (5=cheap) | Regulatory | P(measurable improvement) | **Σ** |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **O1** | **Two-prior composition operator** — prior-*stripping* as a precondition of pooling disjoint-cohort experts (G3) | **5** | 3 | **5** | **5** | **5** | **5** | 3 | 4 | **35** |
| **O2** | **Leakage-audited UCI re-benchmark** + honest calibrated tabular baseline (G8) | 3 | 3 | **5** | **5** | 4 | **5** | 4 | **5** | **34** |
| **O3** | **"Does fusion hurt?"** — dependence-sensitivity envelope + pre-registered falsifier (G4, G10) | 4 | 3 | **5** | **5** | 4 | **5** | 3 | **5** | **34** |
| **O4** | Calibration-under-missing-modality **for the medical/cervical setting** (G2, narrowed by AECF) | 3 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 31 |
| **O5** | Modality-aware **calibrated deferral** to a cytologist (G12) | 3 | **5** | 3 | 3 | 4 | 4 | **5** | 3 | 30 |
| **O6** | Cross-modal **explainability** ("which modality drove this") + reader study (G11) | 3 | 4 | 2 | 2 | 4 | 4 | 4 | 2 | 25 |
| **O7** | Cervical **foundation model that ingests a risk vector** (absence #7) | 4 | 4 | 1 | 1 | 4 | 1 | 2 | 2 | 19 |
| **O8** | Prospective **paired cohort** collection (absence #1) | 2 | **5** | 1 | 1 | 3 | 3 | **5** | **5** | 25 |
| **O9** | Build the flexible any-modality system **as originally framed** | **1** | 3 | 3 | 2 | 2 | 4 | 2 | 2 | **19** |

### Selected opportunity — **O1 + O2 + O3 as a single programme**

**O1 is the only axis on which adversarial search found no collision.** Every other clause of the original framing collided (see §Rejected gaps, Phase 6): the fusion rule is ProbEn / a weighted opinion pool; per-subset calibration is AECF; modality dropout is Gu et al. (MICCAI 2025); prior correction for *one* model is Saerens/Alexandari; conformal is Mondrian CP.

**Why O1 is the right hill:**
1. It is a hole in the **theory**, not an unfilled application slot — the literature splits cleanly either side of it and never crosses (G3).
2. It is **unavoidable** — with two cohorts at two prevalences, you either strip the priors or you silently pool contaminated posteriors. It cannot be dismissed as contrived.
3. It is **fully validatable per-expert on public data**, because prior-stripping and re-anchoring are evaluated **on each cohort independently** and never require a paired set. *This is the decisive property:* it side-steps the hard kill in Phase 6 (the both-modality path has no held-out data).
4. **O2 de-risks the whole programme** and is a standalone result: nobody satisfies all three of {`Dx:*` dropped, resampling+imputation confined inside folds, calibration reported}. Our own measurement already provides it (Biopsy AUROC 0.656 [0.584, 0.720]).
5. **O3 turns the project's biggest weakness into its finding.** Grill et al. (2017) predict the forced independence assumption over-estimates risk by **up to 56%**; that is a *pre-registered, directional, testable* property of our own fused output.

**Selected programme title:**
> **"Composing disjoint-cohort experts: prior-stripped opinion pooling, and the price of the conditional-independence assumption you cannot avoid paying."**

**O9 is explicitly rejected** — it scores 1 on novelty and fails the stated quality standard ("reject if the novelty is only combining existing models").

---

## Phase 8 — Research questions

Each RQ names the gap it closes and the evidence that makes it open.

> ### RQ1 — The honest baseline *(closes G8)*
> **What is the leakage-free predictive ceiling of the UCI cervical risk-factor questionnaire for a clinically meaningful target (biopsy), and what fraction of the published 0.98–1.00 AUC is attributable to (a) target leakage from `Dx:*` columns and (b) resampling-before-split?**
> *Open because:* no paper satisfies all three of {no target leakage, fold-confined resampling, calibration reported}; Demircioğlu's leakage protocol has never been run on UCI.

> ### RQ2 — The two-prior composition operator *(closes G3)* ★ **primary**
> **When two experts are trained on disjoint cohorts with different, non-clinical training prevalences, does explicitly *stripping* each expert's own prior before pooling and re-applying a single declared deployment prior produce better-calibrated composed posteriors than (i) pooling raw posteriors and (ii) applying a single post-hoc prior correction to the pooled output?**
> *Open because:* prior correction exists only for one classifier → one population (Saerens 2002; Godau 2023; Alexandari 2020); pooling exists only under a common prior (Kittler 1998; Genest & Zidek 1986; ProbEn 2022). Nobody strips priors as a *precondition* of pooling.

> ### RQ3 — The price of the forced assumption *(closes G4, G10)*
> **Given that conditional independence is *non-identified* at zero cohort overlap and therefore forced rather than chosen, what is the magnitude and direction of the resulting miscalibration of the composed posterior, and over what range of plausible cross-modal dependence does composition beat the better single expert?**
> *Open because:* Grill et al. (2017) quantify the penalty in biostatistics (≤56% over-estimation) but it has never been transported to an ML fusion system; "does fusion ever hurt?" is never posed as a first-class question, despite Kelebet et al. (2025) reporting fused 92% < tabular-alone 96%.

> ### RQ4 — Confidence under absence *(closes G2, narrowed by AECF)*
> **In the medical/cervical setting, does a composed any-subset system become *appropriately less confident* when a modality is absent — i.e. is confidence monotone in the available evidence set — and does per-path calibration survive prevalence shift?**
> *Open because:* calibration is reported in 0/10 missing-modality-general, 0/10 missing-modality-medical, 0/10 domain-shift and 2/45 in-house cervical papers; no paper asks whether confidence *should* degrade. **Constrained:** AECF already claims/measures monotone confidence across subsets on non-medical paired data — so this RQ is explicitly the *medical, disjoint-cohort, unpaired* instance, not the general claim.

> ### RQ5 — What can be validated without a paired cohort *(closes the hard kill)*
> **Absent any dataset pairing a cervical image with a questionnaire for the same patient, which properties of a composed system are empirically verifiable, which are only boundable, and which are unfalsifiable — and what is the minimum prospective data collection that would convert each class?**
> *Open because:* absence #1 is verified affirmatively (incl. the decisive UCI 383 vs 384 miss); the field has never stated the limit.

---

## Phase 9 — Hypotheses (statistically testable)

Each hypothesis specifies **estimand, test, and decision rule**. All are pre-registered *before* fitting. α = 0.05 throughout; CIs are BCa bootstrap (10,000 resamples) unless stated.

| ID | Hypothesis | Estimand | Test / decision rule |
|---|---|---|---|
| **H1a** | Under a leakage-safe protocol (`Dx:*` dropped, resampling confined inside folds), the UCI **Biopsy** AUROC is **≤ 0.75**. | OOF AUROC, nested stratified CV | Reject H1a if the **lower** bound of the 95% CI > 0.75. *(Current evidence: 0.656 [0.584, 0.720] — consistent.)* |
| **H1b** | The published ≈0.98–1.00 AUC is **fully accounted for** by leakage: honest AUROC + leakage-attributable increment ≈ published AUROC, with **no residual signal**. | ΔAUROC when re-introducing (i) `Dx:*`, (ii) pre-split resampling, separately and jointly | Two-factor ablation; report ΔAUROC per factor with CI. H1b holds if honest + Δ(i) + Δ(ii) ≥ published − 0.02. *(Falsifier: a residual > 0.05 unexplained by either factor.)* |
| **H1c** | `Dx:*` columns are **target leakage, not informative history**: predicting `Dx:Cancer` from the remaining `Dx` columns yields AUROC ≥ 0.95. | AUROC | *(Current evidence: 0.998 leaky vs 0.592 leakage-safe — consistent.)* |
| **H2a** ★ | **Prior-stripping before pooling** yields lower ECE of the composed posterior than pooling raw posteriors, at a declared deployment prevalence τ. | ΔECE (stripped − raw), per τ ∈ {0.02, 0.06, 0.10, 0.20} | Paired bootstrap over the simulated joint; reject H2a if the 95% CI of ΔECE includes 0 or is positive. |
| **H2b** | Prior-stripping is **not** equivalent to a single post-hoc correction of the pooled output; the two differ measurably when the two experts' priors differ (π_img ≠ π_tab). | ΔECE and Δlog-loss between the two operators, as a function of \|logit π_img − logit π_tab\| | Regression of Δ on prior-gap; H2b holds if slope ≠ 0 (95% CI excludes 0). **Falsifier:** if they coincide, the operator is a re-parameterisation and the novelty claim dies. |
| **H2c** | Each expert's prior-stripping step is **individually validatable**: re-anchoring a single expert from its training prevalence to a known target prevalence improves its ECE on a held-out cohort resampled to that prevalence. | ECE before vs after re-anchoring, **per expert, per cohort** | Paired bootstrap; one-sided. *This is the test that needs no paired data.* |
| **H3a** | The composed posterior is **over-confident in a predictable direction**, consistent with Grill et al. (2017). | signed calibration error (mean predicted − observed) at the fused operating point | H3a holds if the signed error is **> 0** with 95% CI excluding 0, across ρ ≥ 0. |
| **H3b** | **Composition does not beat the better single expert** at realistic dependence: with image AUROC ≈ 0.73 (external) and tabular ≈ 0.65, equal-weight pooling gains ≤ +0.02 AUROC at ρ = 0 and is **negative** for ρ ≥ 0.3. | ΔAUROC(fused − max(single)) as a function of ρ | Dependence-sensitivity envelope over the *real* marginals. *(Current evidence: +0.016 at ρ=0; −0.030 at ρ=0.6 — consistent.)* |
| **H4a** | Confidence is **monotone in the evidence set**: mean confidence(both) ≥ mean confidence(single) ≥ mean confidence(none), and no subset is *more* confident than a superset containing it. | mean max-probability per availability pattern | Monotonicity violated if any superset has lower mean confidence (95% CI). |
| **H4b** | **Per-path marginal conformal coverage** is achieved at 1−α on each expert's own cohort, and **degrades under prevalence shift** absent unlabeled target data. | empirical coverage per path, per τ | H4b holds if coverage ∈ [1−α−ε, 1−α+ε] at τ = training prevalence and falls outside at shifted τ (per Podkopaev & Ramdas). **No joint guarantee is claimed or tested.** |
| **H5** | The image expert's **in-distribution** AUROC over-states its deployable performance by ≥ 20 points. | AUROC(in-dist) − AUROC(external) | *(Current evidence: 0.998 vs 0.73/0.59/0.53 — consistent.)* |

### Pre-registered global falsifier (the project's kill switch)

> **If composition does not beat `max(image-only, tabular-only)` on *both* discrimination (AUROC) *and* calibration (ECE) at realistic dependence, we report the negative result and the paper's contribution becomes the operator + the impossibility analysis — not a screening system.**
>
> Precedent that this outcome is publishable and real: **Kelebet et al. (BMC MIDM 2025)** — image+tabular late fusion scored **92%** vs the tabular expert alone at **96%**.

### Hypotheses deliberately **not** made (because they are unfalsifiable here)

- ❌ "The fused system achieves AUROC X on patients with both modalities." — **No paired cohort exists**; there is no held-out data on which to measure joint discrimination or calibration. Stating such a number would require pseudo-pairing (the "chimeric users" practice, Poh & Bengio 2005/2006 — whose conclusions we have **not** been able to read) or simulation that assumes the very dependence structure Grill (2017) shows is what breaks calibration.
- ❌ "The system provides a joint conformal guarantee." — Per-expert marginal guarantees **do not compose** (G5); claiming otherwise is a technical error.
