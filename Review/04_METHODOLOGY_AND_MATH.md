# Phases 10–11 — Methodology & Mathematical Formulation

> **Read `10_NOVELTY_STATEMENT.md` first.** The candidate framing ("train two experts, map to a shared axis, fuse") was **rejected under the stated quality standard** — it is an assembly of published parts and its headline path is unvalidatable. What follows is the **reframed** design, in which exactly **one** component is claimed novel and every other component is explicitly declared borrowed with its owner named.

---

## 0. The reframing, in one paragraph

We do **not** propose "a flexible multimodal cervical screening system". We propose **CO-PRIOR**: an operator and an evaluation regime for **composing experts trained on disjoint cohorts**, in which each expert's own training prevalence is **stripped out** before pooling and a single declared deployment prior is applied **after** — plus an honest accounting of what such a composition can and cannot be validated to do. The cervical image + questionnaire setting is the **motivating instance**, not the contribution.

---

## 1. Architecture overview

```
                    ┌───────────── COHORT A (SIPaKMeD/Herlev) ─────────────┐
  Pap-smear cell ──▶│ M1 Image expert (frozen)  ──▶ p_img , π_img          │
                    └──────────────────────────────────────────────────────┘
                    ┌───────────── COHORT B (UCI 383) ─────────────────────┐
  Questionnaire ───▶│ M2 Tabular expert (frozen) ──▶ p_tab , π_tab         │
                    └──────────────────────────────────────────────────────┘
                                    │
            M3 Per-expert calibration (BCTS)  ── ECE/Brier per expert ──▶ audit
                                    │
            M4 PRIOR-STRIPPING  λ_k = logit(p̃_k) − logit(π_k)      ★ NOVEL
                                    │
            M5 Availability router (vacuous = identity)  ── any subset
                                    │
            M6 Composition at declared prevalence τ:  logit(P) = logit(τ) + Σ_k w_k λ_k
                                    │
            M7 Dependence-sensitivity envelope (ρ sweep) + over-confidence audit
                                    │
            M8 Per-path conformal abstention (MARGINAL only) ──▶ RELEASE / DEFER
                                    │
            M9 Modality-attribution report ("image drove this; questionnaire dissented")
```

**Nothing after M2 is trained on joint data, because no joint data exists.** M4–M6 are closed-form. M3 and M8 are fitted **per expert on that expert's own cohort** — the property that makes the programme validatable at all.

---

## 2. Modules

For each: *Purpose · Inputs · Outputs · Algorithm · Expected contribution · Failure modes · Supporting literature · Novelty justification.*

---

### M1 — Image expert (**borrowed**)
- **Purpose.** Emit a cytomorphology posterior from one Pap-smear cell image.
- **Inputs.** Cell image `x_img`. **Outputs.** `p_img ∈ [0,1]` = P(abnormal morphology), plus training prevalence `π_img`.
- **Algorithm.** EfficientNet backbone (Tan & Le, ICML 2019), optionally distilled (Hinton et al. 2015). Binary collapse of SIPaKMeD's 5 classes: abnormal = {koilocytotic, dyskeratotic}.
- **Expected contribution.** The strong arm. **External** AUROC ≈ 0.73 (Herlev) — *not* the 0.998 in-distribution figure.
- **Failure modes.** (i) 27–47 pt collapse under scanner/stain/site shift ⟨OWN, G9⟩; (ii) SIPaKMeD/Herlev release **no patient IDs**, so patient-wise splitting is impossible in principle ⟨G8⟩; (iii) koilocytosis marks *productive, often transient* HPV and is a weak/negative predictor of high-grade disease — so `p_img` is **not** a histology-risk proxy (see §3).
- **Literature.** Plissiti et al. (ICIP 2018); Jantzen et al. (2005); Tan & Le (2019); Coskun et al. (2026).
- **Novelty.** **None claimed.** Off-the-shelf.

### M2 — Tabular expert (**borrowed**, but re-benchmarked)
- **Purpose.** Emit a pre-test risk posterior from the questionnaire.
- **Inputs.** Risk vector `x_tab` (age, parity, smoking, contraceptives, STD/HPV history). **Outputs.** `p_tab`, `π_tab ≈ 0.064`.
- **Algorithm.** Gradient-boosted trees (Grinsztajn et al., NeurIPS 2022) under a **leakage-safe protocol**: `Dx:*` dropped; resampling/imputation confined inside folds; nested stratified CV; bootstrap CIs.
- **Expected contribution.** The **honest anchor** (RQ1). Measured: Biopsy **AUROC 0.656 [0.584, 0.720]** ⟨OWN⟩ — a *weak triage signal*, not a diagnostician.
- **Failure modes.** Severe imbalance (14.6:1); no feature exceeds univariate AUROC 0.572; single Venezuelan cohort (n=858, 2012–13) with **zero external validation** anywhere in this literature ⟨G8, absence #10⟩.
- **Literature.** Fernandes, Chicco et al. (PeerJ CS 2018) — the honest anchor; Roy et al. (2025) — the leaky comparator; Chawla et al. (2002); Demircioğlu (leakage protocol).
- **Novelty.** **None in the model.** The *audit* (RQ1/H1a–c) is a contribution because nobody satisfies all three of {no `Dx:*`, fold-confined resampling, calibration reported}.

### M3 — Per-expert calibration (**borrowed**)
- **Purpose.** Make each `p_k` a genuine frequency **before** any prior arithmetic.
- **Algorithm.** **BCTS (bias-corrected temperature scaling) + MLLS**, per Alexandari et al. (ICML 2020) — **not** naive Saerens EM.
- **Why this specific choice.** Alexandari's warning is load-bearing: Saerens's EM "does not require model retraining", **but** "a limiting assumption of this algorithm is that p(y|x) is calibrated, which is not true of modern neural networks." Prior correction on an *uncalibrated* posterior is invalid. ⟨G3 constraint⟩
- **Outputs.** Calibrated `p̃_k`; reported ECE/Brier + reliability diagram **per expert, per cohort**.
- **Failure modes.** Calibration set too small on the minority class (UCI has 35–74 positives per target) → high-variance temperature. Mitigation: cross-fitted calibration, report CI.
- **Literature.** Guo et al. (ICML 2017); Alexandari et al. (ICML 2020); Saerens et al. (2002).
- **Novelty.** **None claimed.**

### M4 — Prior-stripping ★ **THE NOVEL COMPONENT**
- **Purpose.** Remove each expert's **own cohort prevalence** from its posterior, so that what is pooled is *evidence*, not *evidence ⊗ an accident of dataset construction*.
- **Inputs.** `p̃_k`, `π_k` (expert *k*'s training prevalence). **Outputs.** A prior-free log-likelihood-ratio `λ_k`.
- **Algorithm.** `λ_k = logit(p̃_k) − logit(π_k)`. Under a calibrated `p̃_k`, `λ_k = log[ P(x_k | Y=1) / P(x_k | Y=0) ]` — a quantity that **does not depend on the training prevalence**.
- **Expected contribution.** Makes two posteriors from two differently-constructed cohorts commensurable *at all*. Herlev's prior is curated by construction (its creators selected cells "not to collect a natural distribution"); UCI's is 6.4%. Pooling them raw silently pools two different priors twice.
- **Failure modes.** (i) `π_k` must be *known* — for a curated set it is a construction artefact, not a population rate, and for SIPaKMeD it is a labelling choice (≈40% abnormal cells). **Mitigation: treat π_k as a declared, audited quantity and report sensitivity to it.** (ii) If `p̃_k` is not calibrated, λ_k is not an LLR (hence M3 is mandatory, not optional).
- **Literature.** *This is the gap.* One side: Saerens (2002), Godau (MICCAI 2023), Alexandari (ICML 2020) retarget **one** classifier to **one** population. Other side: Kittler (1998), Genest & Zidek (1986), ProbEn (ECCV 2022) combine **several** classifiers under a **common** prior. Neither strips priors as a *precondition of pooling*. An independent OpenAlex search for prevalence correction when combining classifiers across populations returned **nothing on point**.
- **Novelty justification.** The **only** axis on which adversarial search found no collision. It is a hole in the theory (not an unfilled application slot), it is **unavoidable** (two cohorts ⇒ two priors), and — decisively — **it is validatable per-expert on public data without any paired set** (H2c).

### M5 — Availability router (**borrowed**)
- **Purpose.** Operate on any subset {image}, {questionnaire}, {both}.
- **Algorithm.** An absent expert contributes `λ_k = 0` — the **identity element** of the additive pool (equivalently a vacuous opinion `b=0, u=1`, the identity of Dempster's rule). **No imputation**, which is structurally impossible at zero cohort overlap ⟨G7⟩.
- **Failure modes.** A lone weak expert must not auto-decide → routed to M8 (defer).
- **Literature.** ProbEn (ECCV 2022) — "through probabilistic marginalization, ProbEn elegantly handles missing modalities"; TMC (ICLR 2021); Shafer (1976).
- **Novelty.** **None claimed.** Explicitly ProbEn's marginalisation. Modality dropout is **also not novel** — Gu, Saito & Ma (MICCAI 2025) already do image+tabular with modality tokens + dropout.

### M6 — Composition at a declared prevalence (**borrowed rule, novel input**)
- **Purpose.** Produce the composed posterior at a **declared deployment prevalence τ**.
- **Algorithm.** `logit P(Y=1 | available) = logit(τ) + Σ_{k ∈ A} w_k λ_k`, with `w_k ∈ [0,1]` a reliability discount. The prior is now counted **exactly once**, by construction.
- **Expected contribution.** τ is a *declared, auditable* clinical input — not an artefact inherited from whichever dataset happened to be used.
- **Failure modes.** The additive form **is** naive-Bayes and therefore assumes conditional independence — see M7. Also: an equal-weight pool can be **worse than the better single expert** (Kelebet 2025: fused 92% < tabular 96%).
- **Literature.** Kittler et al. (1998) (sum vs product rules); Genest & Zidek (1986) (linear vs logarithmic opinion pools); ProbEn (2022).
- **Novelty.** **The rule is borrowed and declared so.** What is new is only that its inputs are prior-*stripped* LLRs rather than raw posteriors.

### M7 — Dependence-sensitivity envelope & over-confidence audit (**novel framing, standard tools**)
- **Purpose.** Quantify the **price of the assumption you cannot avoid paying**.
- **Algorithm.** With zero overlap the cross-modal dependence ρ is **non-identified**. Rather than assume ρ=0 and report one number, sweep ρ ∈ [−1, 1] over the *real* marginals and report the **envelope** of composed AUROC/ECE.
- **Expected contribution.** Converts the project's central weakness into a *reported, bounded* result. Predicted (Grill et al. 2017): over-estimation of risk **up to 56%** (≈27% on real data), **directional** ⇒ testable (H3a).
- **Failure modes.** The envelope is model-based (Gaussian-score approximation); it bounds, it does not measure. **Must be labelled as such.**
- **Literature.** Grill et al. (Stat Med 2017); Kelebet et al. (2025); Fréchet–Makarov bounds; statistical matching (non-identified without conditional independence).
- **Novelty.** Transporting Grill's biostatistical result into an ML fusion system as a **pre-registered, falsifiable prediction about our own output** is, to our knowledge, unclaimed. The envelope itself is standard.

### M8 — Per-path conformal abstention (**borrowed, deliberately weakened**)
- **Purpose.** A reject option with a distribution-free guarantee.
- **Algorithm.** **Mondrian split conformal**, calibrated **per expert on that expert's own cohort**. Non-singleton prediction set ⇒ **DEFER** to a cytologist.
- **Critical honesty.** We claim **per-path MARGINAL coverage only**. We do **not** claim a joint guarantee: per-expert marginal guarantees **do not compose** (G5), and Podkopaev & Ramdas (UAI 2021) show label shift degrades coverage with a fix requiring "unlabeled data from the target distribution" — **a precondition unmet here**.
- **Failure modes.** Coverage is marginal, not conditional on the availability pattern; small minority classes make the quantile unstable.
- **Literature.** Angelopoulos & Bates (2021/2023); Olsson et al. (Nat Commun 2022); Zaffran et al. (ICML 2023); Podkopaev & Ramdas (UAI 2021); Xue et al. (Modern Pathology 2023) — the clinical precedent (fixed 0.35 threshold, uncalibrated).
- **Novelty.** **None in the mechanism.** Modality-*conditional* deferral would be novel (G12) but is **not claimed** without a paired cohort to validate it.

### M9 — Modality attribution (**borrowed idea, honest scope**)
- **Purpose.** Report *which* expert drove the composition — `λ_img` vs `λ_tab` are on the same LLR scale, so their contributions are directly comparable and additive.
- **Expected contribution.** Addresses G11 ("no system explains which modality drove a decision"). **Free**: M4 already produces the decomposition.
- **Failure modes.** **Explanation ≠ benefit.** Jabbour et al. (JAMA 2023, n=457 clinicians) found image-based explanations **did not mitigate** the harm of a biased model. We therefore do **not** claim clinical benefit without a reader study.
- **Novelty.** Modest; a natural read-out of M4, not a separate contribution.

---

## 3. The defect the design must confront head-on (construct + unit mismatch)

This is **not** a literature gap; it is a defect in the naive task framing, and it is **fatal if hidden**:

- **Construct mismatch.** SIPaKMeD/Herlev labels are **cytomorphological** categories; UCI's target is a **biopsy/histology** outcome. These are **not two noisy measurements of one latent variable** — they are different diagnostic tests with different reference standards. *That is precisely why UCI ships Hinselmann/Schiller/Cytology/Biopsy as four separate correlated columns rather than one label.*
- **Unit-of-analysis mismatch.** SIPaKMeD/Herlev are **cell-level**; UCI is **patient-level**. Fusing P(abnormal cell) with P(biopsy-positive patient) needs a cell→patient aggregation, and **no data exists anywhere to calibrate it** (absence #1: no patient IDs).

**Fusing two different constructs measured on two different units and calling the output "calibrated P(abnormal)" is a *definitional* problem, not a calibration problem. No amount of prior correction repairs it.**

**Our resolution (the honest one).** We make the mismatch a **first-class object of study** rather than a step: *what is the weakest assumption under which composing a cytomorphology posterior with a histology-risk posterior is coherent?* We model the two targets as **imperfect indicators of one latent state** using the no-gold-standard / latent-class machinery (Rutjes et al., HTA 2007), and we report the composition **conditional on that model** — never as an unconditional P(abnormal). Where the assumption fails, we say so.

---

## Phase 11 — Mathematical formulation

### 11.1 Notation

| Symbol | Meaning |
|---|---|
| `Y ∈ {0,1}` | latent target state (defined explicitly per §3; **not** assumed identical across cohorts) |
| `k ∈ {img, tab}` | expert index |
| `D_k` | cohort on which expert *k* was trained; **`D_img ∩ D_tab = ∅`** |
| `x_k` | expert *k*'s input |
| `π_k = P_{D_k}(Y=1)` | expert *k*'s **training prevalence** (a property of the cohort, not the world) |
| `τ` | **declared deployment prevalence** |
| `A ⊆ {img, tab}` | the available-modality set for a given case |
| `σ(z) = 1/(1+e^{−z})`; `logit(p) = log[p/(1−p)]` | |
| `ρ` | cross-modal dependence of the experts' errors given `Y` — **non-identified here** |

### 11.2 Problem definition

Given frozen experts `f_k : x_k ↦ p_k = P̂_{D_k}(Y=1 | x_k)`, trained on **disjoint** cohorts with **different** priors `π_img ≠ π_tab`, and **no sample with both modalities**, construct

```
g : ({p_k}_{k∈A}, {π_k}_{k∈A}, τ)  ↦  (P(Y=1 | {x_k}_{k∈A}), decision ∈ {RELEASE, DEFER})
```

such that (i) `g` is defined for **every** `A ≠ ∅`; (ii) each component of `g` is **fitted and validated on a single cohort at a time**; (iii) `g` never claims a guarantee that requires joint data.

### 11.3 Calibration objective (M3, per expert)

Fit temperature `T_k` and bias `b_k` (BCTS) on expert *k*'s **own** held-out fold:

```
(T_k*, b_k*) = argmin_{T,b}  −(1/n) Σ_i [ y_i log σ(z_i/T + b) + (1−y_i) log(1 − σ(z_i/T + b)) ]
p̃_k = σ(z_k/T_k* + b_k*)
```
Report `ECE_k`, `Brier_k`, reliability diagram. **Precondition for §11.4** (Alexandari et al., ICML 2020).

### 11.4 Prior-stripping (M4) ★

```
λ_k  =  logit(p̃_k) − logit(π_k)                                   (1)
```

**Proposition (why this is the right object).** If `p̃_k` is calibrated on `D_k`, then by Bayes' rule
`logit P_{D_k}(Y=1|x_k) = logit π_k + log[ f(x_k|Y=1)/f(x_k|Y=0) ]`, hence

```
λ_k = log[ f(x_k | Y=1) / f(x_k | Y=0) ]                          (2)
```
which is a **likelihood ratio, free of the training prevalence**. This is exactly the quantity that is transportable across cohorts; `p̃_k` is not.

### 11.5 Composition (M6)

```
logit P(Y=1 | {x_k}_{k∈A})  =  logit(τ)  +  Σ_{k∈A} w_k λ_k        (3)
```
with `w_k ∈ [0,1]`. `A = ∅ ⇒ P = τ` (the prior itself). An absent expert contributes `λ_k = 0` — the additive identity (M5).

**Relation to prior art (declared).** Equation (3) with `w_k = 1` **is** the naive-Bayes / log-opinion-pool / ProbEn rule. The **only** difference from ProbEn is that its inputs are prior-stripped LLRs (1) rather than raw posteriors, and that the prior enters exactly once, at τ. *We claim no novelty in (3) itself.*

**Robust alternative.** Kittler et al. (1998) show the **sum rule** is less sensitive to individual estimation error than the product rule — relevant because `p_tab` is high-variance (35–74 positives). We evaluate both:
```
P_sum = (1/|A|) Σ_{k∈A} σ( logit(τ) + λ_k )                        (4)
```

### 11.6 Uncertainty formulation

Two **separate** quantities, deliberately not conflated (per G6):

1. **Calibrated probability** — `P` from (3)/(4), evaluated by ECE/Brier. *This is the only quantity allowed to be called a probability.*
2. **Evidential/epistemic score** — an ordinal reliability signal only. Justification: Jürgens et al. (ICML 2024) show EDL epistemic uncertainty is **"relative (rather than absolute)"**; TMUR (2026) shows independently trained branches' uncertainties are **not on a common scale**. Hence:
```
u_k used ONLY for ranking / routing;   NEVER as P(Y=1) or as a weight without recalibration.
```
Conflict between experts (for M9 reporting):
```
K = |σ(logit τ + λ_img) − σ(logit τ + λ_tab)| · 1[ λ_img · λ_tab < 0 ]     (5)
```

### 11.7 The non-identified dependence, and its bound (M7)

Conditional independence gives (3). It is **forced**: with `D_img ∩ D_tab = ∅`, the joint `f(x_img, x_tab | Y)` is **non-identified** — no estimator of ρ exists from the available data. We therefore report an **envelope** rather than a point. Under an equal-variance Gaussian-score model with per-expert separations `d_k = √2 · Φ^{-1}(AUROC_k)`:

```
AUROC_fused(ρ) = Φ( (d_img + d_tab) / ( √2 · √(2 + 2ρ) ) )                 (6)
```
`ρ = 0` (independence) is the optimistic end; `ρ = 1` gives `AUROC_fused = AUROC_single` (composition adds nothing). Because both signals are HPV-driven, **ρ > 0 is the realistic region** — where composition gains little or nothing.

**Predicted over-confidence (H3a).** Grill et al. (2017) give the directional penalty (≤56% over-estimation of risk) which we pre-register as a property of our own output.

### 11.8 Conformal abstention (M8)

Per expert *k*, Mondrian split-conformal on `D_k`'s calibration fold; non-conformity `s = 1 − p̃_{k,y}`; class-wise threshold `q_{k,c} = Quantile_{1−α}({s_i : y_i = c})`; prediction set `C_k = { c : 1 − p̃_{k,c} ≤ q_{k,c} }`.

```
Guarantee claimed:   P( Y ∈ C_k ) ≥ 1 − α    — MARGINAL, on cohort D_k only.     (7)
Guarantee NOT claimed: any joint statement over (x_img, x_tab).
```
DEFER iff `|C| ≠ 1`.

### 11.9 Complexity

| Stage | Train | Inference |
|---|---|---|
| M1 image expert | one CNN fit (hours, 1 GPU) | `O(1)` forward pass |
| M2 tabular expert | nested CV over 858 × 26 (seconds–minutes, CPU) | `O(1)` |
| M3 calibration | 2 scalars per expert; `O(n log n)` | `O(1)` |
| **M4–M6 composition** | **no training at all** | **`O(|A|)` — 2 logs, 2 adds, 1 sigmoid** |
| M7 envelope | `O(\|ρ-grid\|)` closed form | n/a (analysis only) |
| M8 conformal | one quantile per class per expert | `O(1)` |

**The entire contribution after the two experts is closed-form and costs microseconds.** No joint training, no gradients across cohorts, no paired data — which is precisely why it is feasible *and* why its claims must be modest.
