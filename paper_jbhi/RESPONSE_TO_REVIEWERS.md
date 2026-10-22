# Response to the reviewer

Written for: the JBHI reviewer and associate editor.

We thank the reviewer for a detailed and unusually constructive report. The central criticism — that the fusion experiment largely instantiated the assumptions used to derive the fusion rule — was correct, and acting on it changed the paper's conclusions rather than merely its wording. The manuscript has been rewritten. Its title and framing now foreground the region of validity rather than the rule, the epidemiological link is estimated from data rather than assumed, the image expert is validated externally, and we report cases in which the proposed rule is **not** the best choice.

The most consequential changes are these:

- The paper no longer claims the rule "never harms". It claims the rule is exact under three stated assumptions and shows where each fails.
- Under real domain shift (the external image expert), **naive averaging outperforms the proposed rule** on ranking, and a paired-data stacker beats both on calibration. This is now reported in the abstract, Results and Discussion.
- The rule loses its advantage once residual dependence between the modalities exceeds ρ ≈ 0.2.

Section and float numbers below refer to the revised manuscript.

---

## Major comments

### 1. The fusion experiment validated assumptions built into the generator

**Agreed, and this was the most important change.** Section IV-F ("Breaking the assumptions") now defines six controlled violations, all applied to the generator with the fusion rule unchanged and the oracle refitted so it can exploit whatever structure appears:

| | Violation | Mechanism |
|---|---|---|
| V1 | residual dependence given the latent state | Gaussian copula of correlation ρ over the ranks from which the two real records are drawn, preserving marginals |
| V2 | direct questionnaire → cytology path | strength γ, violating A ⊥ x_t \| H |
| V3 | infection → image at fixed cytology | shift δ, violating x_i ⊥ H \| A |
| V4 | subgroup-varying link | q₁, q₀ differ between halves of the population while the rule uses pooled values |
| V5 | unknown miscalibration | temperature on either expert's log-odds, independently |
| V6 | domain shift, alone and with prevalence shift | bias on the image log-odds plus a prior change |

V1 induces exactly the correlated expert errors the reviewer asked for (point 4 of the list), because the copula acts on the draw of the real records, so the two experts' errors become dependent given the latent state.

Results are in Table V and Fig. 11, and all four comparators the reviewer named (image-only, naive, proposed, paired oracle) are reported for every violation. The headline findings:

- **ρ ≈ 0.2 is the break-even point.** At ρ = 0.6 the proposed rule reaches 0.885 while an oracle that can learn the dependence reaches 0.922.
- **Under V2 the naive mean beats the proposed rule** (0.9196 vs 0.9082 at γ = 1), because the rule is bound to a structure that is no longer true.
- V3, V4 and V5 are tolerated: differences of ≤ 0.002 AUROC.
- **V6 is outside what prior re-anchoring can repair**: fused ECE rises from 0.010 to 0.038, and to 0.058 with a simultaneous prevalence change.

We have also added a practical note: both damaging violations are diagnosable in deployment — dependence from a small paired sample, domain shift from score distributions without labels.

### 2. Unit-of-analysis mismatch between cell-level images and woman-level prevalence

**Agreed.** This is now treated in Methods, Results and Discussion, not only as a limitation.

- Methods, "Unit of analysis" (Section IV-B): the expert scores a cell; a deployment prevalence refers to a woman; these are not the same random variable.
- We added slide-level aggregation (mean, max, top-k, noisy-OR) over the held-out cells of each source slide: 97 slides, slide-level AUROC 0.999, abnormal-slide prior 0.474 vs cell-level 0.401. We state plainly that because SIPaKMeD slides carry a single class this changes the unit but not the difficulty, so it is a step towards the clinical unit, not an arrival at it.
- Discussion now says that until an image expert is trained and calibrated on whole smears, the calibration results demonstrate **the operator**, not patient-level risk.
- The headline deployment setting has moved to a co-test cohort (below), which removes the ATHENA-specific part of the mismatch but not the cell-vs-woman part.

We did not build a multiple-instance slide model, because SIPaKMeD provides no mixed-cell slides on which such a model could be trained or evaluated honestly. We prefer to state the limit than to simulate the data that would hide it.

### 3. The epidemiological bridge was not estimated

**Fixed.** Section III-D now derives the link from the full cross-tabulation of **812,598 screening co-tests** at Kaiser Permanente Northern California (Castle et al., *Obstet Gynecol* 2009): 6.27% HPV-positive, 3.99% HPV-positive with normal cytology, 5.18% ASC-US or worse, 2.87% abnormal while HPV-negative. Two independent paths to the joint cell agree to 0.03 percentage points, which we report as a consistency check. This gives

> q₁ = P(A=1 | H=1) = 0.364, q₀ = P(A=1 | H=0) = 0.031, OR = 17.9 (95% CI 17.5–18.3).

Uncertainty is propagated by resampling the reported counts. The odds-ratio sweep is retained as a sensitivity analysis for services facing a different link, with the estimated value marked on Fig. 10; the supplement shows that misspecifying the link by a factor of ten in either direction barely affects discrimination, so the width of this interval is not a practical source of error.

### 4. The two conditional-independence assumptions need justification

**Added as a graphical model.** Fig. 2 is a DAG stating the assumed structure (H → A, H → x_t, A → x_i) with the three violations drawn as dashed edges, so the reader can see exactly which arrow each experiment adds. Section IV-A states A1–A3 explicitly and says that none can be checked without paired data — which is the stated motivation for Section IV-F. We agree with the reviewer's biological argument that smoking, age and immune factors may act on persistence and progression, not only acquisition; that is precisely violation V2, and we now report that the rule loses to naive averaging under it.

### 5. NHANES complex survey design

**Added.** Methods IV-C and Results VI-B now report a survey-weighted sensitivity analysis using WTMEC2YR, SDMVSTRA and SDMVPSU. Weighted prevalence is 22.1% against an unweighted 24.6%, and weighted AUROC is **0.713** against 0.694, so the unweighted analysis is conservative. Both are reported in Table III. We retain the unweighted model as primary because it is the model whose out-of-fold predictions feed the fusion, and say so.

### 6. Integer encoding of nominal variables

**Fixed, and it mattered in the direction the reviewer predicted.** Race/ethnicity, marital status and sexual orientation are now one-hot encoded, yes/no items recoded to 0/1, genuinely ordinal items left ordered, giving 71 features. AUROC rose from 0.687 to **0.697**, and the coefficient on marital status is now interpretable as a contrast rather than a spurious linear trend. All coefficients, importances and calibration statistics have been recomputed.

### 7. Structural missingness up to 59%

**Addressed.** Indicators were added for the 27 items with more than 5% missingness, so the model can use the fact of non-response. We added the temporal experiment the reviewer suggested: training on 2003–2010 and testing on 2011–2016 costs little discrimination (0.684) but **triples the calibration error** (0.022 → 0.064) as prevalence drifts from 26.7% to 21.9%. We report per-cycle performance in the subgroup analysis. This experiment also makes the paper's own point: what drifts is calibration, which is what the re-anchoring step exists to absorb.

### 8. The slide-grouped inference was too strong

**Agreed; the claim has been weakened and the protocol promoted.** Protocol B (slide-grouped) is now the **primary** estimate and Protocol A is reported because it is the split used in the literature. We now state explicitly that the protocols differ in composition and test size as well as grouping, so this is a comparison of two training runs and not a controlled equivalence test, and we no longer claim equivalence from overlapping intervals. Repeated independent fits are reported (Section VI-A).

### 9. Nonstandard knowledge-distillation formulation

**Acknowledged in the equation and ablated.** Section IV-B now states that the temperature is applied to probabilities rather than logits, and that this is not the standard formulation. We compare four students under the slide-grouped protocol: no distillation, the probability-temperature formulation used here, standard logit distillation, and logit distillation with T² scaling (Section VI-A).

### 10. A single training run

**Addressed.** Independent seeds are now reported for the grouped protocol for both teacher and student, with mean ± SD across fits, and we distinguish these from the bootstrap intervals over test observations, which quantify a different thing.

### 11. External validation

**Added, and it is the most informative new result in the paper.** The same slide-grouped student was applied unchanged to the Herlev collection (917 cells, 73.6% abnormal): AUROC falls from 0.995 to **0.641** (95% CI 0.595–0.684), ECE 0.247. The failure is specific rather than diffuse — **normal columnar cells, a type absent from SIPaKMeD, receive a median abnormal score of 0.979**, while other normal cells receive 0.038 and abnormal cells 0.926. Excluding columnar cells raises external AUROC to **0.805**. Fig. 5.

This changed the fusion analysis too: the externally shifted expert is now used as a *real* weak expert in the fusion experiments, and it is under that expert that naive averaging beats the proposed rule.

### 12. "Calibration at screening prevalence"

**Rephrased throughout** to "calibration under prior-probability shift", with an explicit statement that this corrects the prior only and not scanner, staining, demographic or vaccination-era differences. Fig. 9's caption makes the same point, and V6 quantifies what happens when the covariate distribution moves too.

### 13. Conformal coverage under shift

**Added, and the guarantee does not survive.** Coverage was measured under deliberate shift (Section VI-C): matched data 0.901 at a 0.90 target with 21% deferral; **0.880 under dependence; 0.755 under domain shift plus prevalence change**; and under a miscalibrated image expert coverage is held only by deferring 48% of women. We now state that coverage is conditional on exchangeability and that domain shift breaks it.

### 14. Subgroup and fairness analyses

**Added.** Fig. 7 reports AUROC and ECE by age, race/ethnicity and survey cycle: AUROC 0.637–0.725, ECE ≤ 0.050, with the largest miscalibration in women aged 18–24. The Discussion now addresses the reviewer's point about socially sensitive proxies directly: marital status, the strongest single item, is a social proxy rather than a biological one, and whether such proxies belong in a deployed screening tool is a policy question rather than a modelling one.

## Novelty positioning

We accept the reviewer's framing and have adopted it. Related Work now situates the rule among products of experts, mixtures of experts and the pre-test/post-test calculation of diagnostic medicine, and states plainly: *"The novelty is therefore the composition and its evaluation, not the individual probability operations."* The contributions list now leads with the honest evaluation of the two experts and the region of validity, and explicitly includes negative results.

## Minor comments

1. **Model size vs parameter ratio.** Clarified in Table I: both checkpoints are float32 in the same format; the teacher file also stores Adam optimiser state, which accounts for the 11.4× file ratio against a 4.0× parameter ratio.
2. **Model-selection optimism.** A nested outer loop was added; logistic regression was selected in all five outer folds, and the unbiased calibrated estimate is 0.694 (against 0.697 from the optimistic procedure).
3. **Calibration reporting.** Calibration intercept and slope are now reported alongside ECE (slope 1.16).
4. **Decision-curve analysis.** Explicitly described as simulation-based, not evidence of clinical utility; the curve itself has moved to the supplement.
5. **Grad-CAM.** Cautious language retained, maps moved to the supplement, and the coarse 12 × 12 resolution is stated in the caption.
6. **Terminology.** "Screening support" throughout; no diagnostic claims.
7. **Paired confidence intervals.** All fusion comparisons are now paired: every rule sees the same simulated women and the same expert outputs within a replicate, and gains are reported as the mean and 95% interval of the per-replicate difference (for example +0.0028, 95% CI +0.0016 to +0.0034, positive in 20 of 20 replicates).
8. **OR = 10 as headline.** Removed; the link is now estimated (major comment 3).
9. **Bethesda mapping.** Moved to the first paragraph of Materials.
10. **Code availability.** The release now includes the cohort simulator with all random seeds and configurations, the fixed test-set file lists, and per-case held-out predictions.

## Two things the reviewer may wish to weigh

First, we have not built a slide- or patient-level image model, for the reason given under comment 2: the data to train and evaluate one honestly do not exist in the public collections we use. We think stating the limit is better than simulating past it.

Second, the revision is less flattering to the proposed method than the original submission was. The rule is worth roughly two hundred paired patients on matched data and is never beaten there, because it is already optimal in that regime; but under a direct questionnaire-to-cytology path, and under real domain shift, a naive mean ranks better than it does, and paired data win outright once residual dependence is present. We believe a mapped region of validity is a more useful contribution to this field than another small AUROC gain, but we recognise this changes what the paper is claiming, and we would rather the reviewer judge the revision on that basis.
