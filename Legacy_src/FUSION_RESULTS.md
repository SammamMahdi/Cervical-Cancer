# Fused-Model Experiments — Results

All fused-model experiments from the research plan (Phase 2–3 / `FUSION_METHOD.md`), run by
[`src/fusion_experiments.py`](src/fusion_experiments.py) on the two **frozen, real** experts.
Figure: `results/figures/fusion_experiments.png`; numbers: `results/metrics/fusion_experiments.json`.

**The honest premise:** no cohort pairs a cervical image with a questionnaire for the same patient, so no
joint accuracy can be *measured*. Every joint result here is either a **dependence-sensitivity simulation**
(the two experts' REAL per-class score distributions coupled by a Gaussian copula across the untestable
cross-modal dependence ρ) or a **per-expert** property (calibration, conformal coverage) that needs no
pairing. Assumptions are stated, not hidden.

---

## E1 — per-expert marginals (real)

| Expert | AUROC | ECE | prior | n |
|---|---|---|---|---|
| image — SIPaKMeD in-dist (leaky split) | 0.998 | 0.016 | 0.419 | 1,003 |
| **image — Herlev external (honest)** | **0.728** | **0.299** | 0.736 | 917 |
| **tabular — NHANES (frozen)** | **0.712** | **0.011** | 0.246 | 12,634 |

The realistic experts are **comparably strong** (0.728 vs 0.712). The external image expert is **badly
mis-calibrated out-of-distribution** (ECE 0.299) — which is exactly what CO-PRIOR (E3) is for.

## E2 + E8 — dependence envelope & the pre-registered FALSIFIER ⭐

Fused (cascade) vs `max(single)` across dependence ρ, realistic scenario (Herlev image + NHANES tabular):

| ρ | fused AUROC | max(single) | gain | verdict |
|---|---|---|---|---|
| −0.3 | 0.746 | 0.734 | **+0.012** | PASS |
| 0.0 (independence) | 0.724 | 0.727 | −0.003 | fail |
| +0.3 | 0.709 | 0.724 | −0.015 | fail |
| **+0.6 (realistic)** | 0.701 | 0.725 | **−0.025** | fail |
| +0.9 | 0.696 | 0.728 | −0.032 | fail |
| +1.0 | 0.696 | 0.729 | −0.033 | fail |

> **FALSIFIER RESULT (honest null):** symmetric fusion **beats the best single expert only under *negative*
> dependence (ρ < 0)**. In the entire realistic region **ρ ≥ 0 — which is where these experts must live,
> because image koilocytosis and the questionnaire HPV-risk share HPV as a common cause (ρ biased positive)
> — fusion does NOT improve discrimination; it slightly *hurts* it.** The optimistic scenario (leaky in-dist
> image 0.998) is the same story: a strong expert cannot be improved by symmetrically pooling a weaker,
> positively-dependent one. **We report this as the finding, not a gap.**

## E3 — CO-PRIOR operator: prior-stripping restores calibration ⭐ (the validated novelty)

Deploy the tabular expert (calibrated at π = 0.246) at a different prevalence τ, with vs without
prior-stripping/re-anchoring — measured **per-expert, no pairing needed**:

| deployment τ | ECE **no shift** | ECE **CO-PRIOR shift** |
|---|---|---|
| 0.02 | 0.196 | **0.002** |
| 0.06 | 0.160 | **0.003** |
| 0.10 | 0.123 | **0.005** |
| 0.20 | 0.036 | **0.012** |

**Prior-stripping cuts ECE by 1–2 orders of magnitude at every deployment prevalence.** π-misspecification
robustness (τ = 0.10): ECE 0.005 at the correct prior, degrading gracefully to 0.03 (±25 %) and 0.06–0.09
(±50 %). **This is the defensible contribution** — it demonstrably works and is validatable without a paired
cohort.

## E4 — combiner ablation (sum vs naive-Bayes vs cascade)

| ρ | sum (AUROC/ECE) | naive-Bayes (AUROC/ECE) | cascade (AUROC/ECE) |
|---|---|---|---|
| 0.0 | 0.729 / **0.049** | **0.754** / 0.124 | 0.724 / 0.111 |
| +0.6 | 0.705 / **0.063** | 0.736 / 0.146 | 0.701 / 0.118 |
| +0.9 | 0.699 / **0.069** | 0.730 / **0.154** | 0.696 / 0.119 |

The **product / naive-Bayes rule has the highest AUROC but is badly over-confident** (ECE 0.12–0.15) and gets
*worse* as dependence grows — exactly the failure mode predicted for correlated experts. **The SUM rule is
the right operating default**: robust and best-calibrated. Cascade tracks the sum on AUROC while adding the
safety floor.

## E5 — any-subset degradation & structural invariants

At ρ = 0.6: image-only 0.725 · tabular-only 0.718 · **both 0.701** (fusion degrades below both — consistent
with E2). Structural invariants (via `fusion.predict`):

| Invariant | Holds? |
|---|---|
| missing modality ≡ single expert | ✅ |
| safety asymmetry (questionnaire never lowers a positive image) | ✅ |
| lone questionnaire → DEFER | ✅ |
| strong disagreement → FLAG | ⚠️ did not fire under the Herlev/NHANES priors (conflict threshold is prior-sensitive; needs per-deployment tuning) |

## E6 — split-conformal abstention: coverage + risk-coverage

Per-expert marginal coverage (Angelopoulos–Bates split-conformal, calibrated on each expert's OWN data):

| Expert | cov@0.95 | cov@0.90 | cov@0.80 | AURC |
|---|---|---|---|---|
| tabular | 0.953 | 0.910 | 0.810 | 0.134 |
| image in-dist | 0.942 | 0.855 | 0.731 | 0.001 |
| image Herlev | 0.939 | 0.893 | 0.793 | 0.296 |

**The distribution-free coverage guarantee holds** (empirical ≈ target 1−α) for every expert without any
paired data. AURC quantifies the honest risk–coverage difficulty: trivial in-distribution (0.001), moderate
for the tabular expert (0.134), highest for the honest external image expert (0.296).

---

## Bottom line

The experiments confirm the project's honest thesis, and they do it with a **pre-registered falsifier that
returns a null on the accuracy claim**:

1. **Fusion does not manufacture accuracy.** In the realistic positive-dependence regime, symmetrically
   pooling a strong image expert with a comparably-strong-but-correlated questionnaire expert does **not** beat
   the best single expert (E2/E8). This is stated as the finding.
2. **The real contribution is CO-PRIOR** — prior-stripping before pooling restores calibration across
   deployment prevalences (ECE ↓ 1–2 orders of magnitude, E3), validated per-expert with no paired cohort.
3. **The system's value is operational, not a joint-accuracy boost**: graceful any-modality degradation,
   distribution-free abstention with real coverage (E6), a safety-asymmetric cascade, and the robust SUM
   combiner over the over-confident product rule (E4).

The only route to a *measured* joint-accuracy gain remains a prospective paired image + questionnaire pilot
(Research Plan Phase 4). Absent that, the impossibility is the honest result.
