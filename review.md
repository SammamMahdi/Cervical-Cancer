I read the manuscript carefully as an IEEE JBHI reviewer would, including the methodology, equations, experimental design, figures/tables, discussion, and limitations. I would classify it as **promising and within JBHI scope, but not ready for acceptance in its current form**.

IEEE reviewers are asked to consider scope, novelty, validity, correctness of data/analysis, clarity, compliance, and significance. :chatgpt-content-reference{index="0"} This manuscript clearly fits biomedical/health informatics because it combines heterogeneous clinical information sources for screening decision support. :chatgpt-content-reference{index="1"}

## Reviewer recommendation: **Major Revision**

The paper has a genuinely interesting central idea, and I would **not recommend rejection based on topic or lack of contribution**. However, several assumptions used to justify the central fusion result are also embedded directly into the semi-synthetic evaluation. That substantially weakens the empirical evidence for the strongest claims.

The manuscript proposes CO-PRIOR, a decision-level framework for combining an image expert predicting cytological abnormality with a questionnaire expert predicting high-risk HPV infection, despite there being no paired patients across the two datasets. The approach strips the experts' source priors, converts questionnaire-estimated HPV probability into a prior probability of abnormal cytology, and treats the image contribution as a likelihood ratio. :chatgpt-content-reference{index="2"} The paper additionally incorporates missing-modality routing and class-conditional conformal abstention. :chatgpt-content-reference{index="3"}

This is a clever and useful framing of an important practical problem. The issue is not the basic idea; it is **how strongly the current experiments validate it**.

---

# Overall assessment

| Criterion | Assessment |
|---|---|
| Relevance to JBHI | **Strong** |
| Clinical motivation | **Strong** |
| Technical presentation | **Strong** |
| Novelty | **Moderate–strong, but positioning needs strengthening** |
| Experimental rigor of individual experts | **Generally good** |
| Experimental rigor of fusion claim | **Currently insufficient for strongest claims** |
| Reproducibility | **Good, assuming promised code/data outputs are released** |
| Clinical validation | **Weak / simulation based** |
| Writing and organization | **Very good** |
| Recommendation | **Major Revision** |

The abstract is refreshingly specific about the actual result: the primary effect is calibration rather than a large discrimination improvement. It reports that gains in discrimination are small and occur mainly when image evidence is uncertain. :chatgpt-content-reference{index="4"} That restraint is a strength.

---

# Major comments

## 1. The principal fusion experiment partially validates assumptions that were built into the data generator

This is my biggest concern.

The fusion derivation assumes

\[
x_i \perp x_t \mid A,
\]

leading to

\[
\operatorname{logit}P(A=1\mid x_i,x_t)
=
[\operatorname{logit}p_{\rm img}-\operatorname{logit}\pi_{\rm img}]
+
\operatorname{logit}r(x_t).
\]

The manuscript explicitly states this conditional-independence assumption. :chatgpt-content-reference{index="5"}

However, the semi-synthetic cohort is then generated so that questionnaire and image records are independently sampled **conditional on the latent HPV/cytology states**. :chatgpt-content-reference{index="6"}

Consequently, the simulator essentially constructs a world in which the main structural assumptions underlying Eq. (4) are true.

Under those circumstances, the finding that CO-PRIOR approximately matches a paired-data logistic stacker is reassuring as a **verification of the implementation**, but it is less persuasive as an empirical demonstration that the method will recover a paired-data fusion model in real populations.

The paper itself even notes that, under its conditional-independence assumption, the proposed combination has the Bayes-optimal form. That makes the oracle equivalence partly expected rather than surprising.

### What I would require

Add simulations explicitly violating the model assumptions.

At minimum:

1. Introduce residual dependence between questionnaire variables and image evidence after conditioning on \(A\).
2. Allow questionnaire covariates to affect cytological abnormality directly rather than only through HPV.
3. Allow \(H\) to affect image score distributions even after conditioning on \(A\).
4. Create correlated expert errors.
5. Introduce subgroup-dependent \(q_0\) and \(q_1\).
6. Vary calibration error independently for the two experts.
7. Introduce simultaneous covariate shift + prevalence shift.
8. Compare degradation of CO-PRIOR, stacking, naive fusion, and image-only prediction.

A useful experiment would define a dependence parameter \(\rho\), where \(\rho=0\) corresponds to the paper's current assumptions, and progressively increase \(\rho\).

Then the important question becomes:

> **How badly does CO-PRIOR fail when its assumptions are only approximately true?**

That experiment would make the manuscript substantially stronger.

---

## 2. There is a fundamental unit-of-analysis mismatch between the cytology model and the deployment endpoint

This concern may be even more important clinically.

SIPaKMeD consists of cell-level/cropped images and full-field cell clusters, while the deployment prevalence \(\tau_A=0.071\) is taken from the ATHENA screening trial's **woman-level abnormal cytology prevalence**. The paper then treats the cytology classifier's output as \(P(A=1\mid x_i)\), strips its source prevalence, and re-anchors it to the 7.1% screening prevalence.

The manuscript itself states that SIPaKMeD contains individual/cropped cells and has an abnormal-cell proportion of 41.8%. :chatgpt-content-reference{index="7"} ATHENA, in contrast, provides a screening-population abnormal-cytology prevalence of 7.1%. :chatgpt-content-reference{index="8"}

These are not obviously the same random variable.

The following distinction is crucial:

\[
P(\text{abnormal cell})
\neq
P(\text{woman has abnormal cytology}).
\]

A woman-level smear can contain thousands of cells, and abnormal-smear status depends on slide-level evidence and diagnostic criteria rather than on the state of one randomly selected cell.

Therefore, prior correction from approximately \(41.8\%\) cell abnormality to \(7.1\%\) patient abnormality is not merely a prevalence adjustment unless the authors establish that both probabilities correspond to the same endpoint.

### Why this matters

The paper's strongest numerical calibration result—e.g. ECE dropping dramatically at screening prevalence—depends on interpreting these probabilities on the same scale.

That interpretation requires much more justification.

### What I recommend

The authors should either:

- reformulate the present study explicitly as a **proof-of-concept cell-level simulation**, avoiding patient-level calibration claims;

or preferably,

- build a slide/patient-level image model, perhaps using multiple-instance learning or slide aggregation, so that \(A\) means the same thing in the image dataset, epidemiological bridge, and deployment population.

This point needs to be addressed in the main Methods and Discussion, not only as a limitation.

---

## 3. The epidemiological bridge is central to the method, but its clinically deployed value is not actually estimated

The bridge

\[
r(x_t)=q_1\tilde p_H+q_0(1-\tilde p_H)
\]

is conceptually clean. The manuscript determines \(q_1\) and \(q_0\) from the two marginal prevalences plus an infection–cytology odds ratio. :chatgpt-content-reference{index="9"}

However, the reference experiments appear to use an assumed OR such as 10, while sweeping the OR from 1 to 100. :chatgpt-content-reference{index="10"}

A sensitivity analysis is useful, but it is not equivalent to estimating the actual bridge required for deployment.

This matters because the claim of calibrated screening probabilities depends on having approximately correct \(q_0\) and \(q_1\).

### Required revision

The authors should ideally derive \(q_0\), \(q_1\), or the HPV–cytology OR from an independent epidemiological cohort/trial where both variables were measured.

If this cannot be obtained, the paper should clearly distinguish:

> **demonstration of the mathematical framework**

from

> **empirical demonstration of calibrated cervical-screening probabilities.**

I would also propagate uncertainty in the epidemiological link rather than treat the OR as fixed. For example:

\[
\mathrm{OR}\sim p(\mathrm{OR})
\]

could generate intervals on fused probabilities.

That would be much more compelling for a biomedical informatics audience.

---

## 4. Two conditional-independence assumptions need much stronger biological justification

The method appears to rely on two related assumptions.

First:

\[
x_i\perp x_t\mid A.
\]

Second, Eq. (3) effectively assumes that questionnaire information affects abnormality through the HPV state represented by \(H\), i.e. approximately

\[
A\perp x_t\mid H.
\]

This may not hold biologically.

Age, smoking, immunological factors, sexual exposure history and other questionnaire variables can potentially be associated not only with HPV acquisition but also persistence/progression and therefore abnormal cytology conditional on HPV status.

Similarly, cytological appearance may vary with factors correlated with infection subtype, patient age, or other biological variables after conditioning on the binary abnormal/normal label.

The paper therefore needs either:

- clinical/epidemiological evidence supporting these assumptions;
- a more general graphical model;
- or explicit robustness experiments showing how violations affect predictions.

A causal/probabilistic graphical diagram would help considerably.

---

## 5. The NHANES analysis appears to ignore the complex survey design

NHANES is not a simple random sample. It uses survey weighting, stratification, clustering and cycle-specific sampling procedures.

The manuscript pools seven cycles and reports calibrated risk estimation on 12,634 women. :chatgpt-content-reference{index="11"} However, I do not see discussion of NHANES examination/sample weights, strata or PSU structure.

This is important because the paper is explicitly concerned with:

- prevalence,
- probability calibration,
- population transfer,
- and epidemiological interpretation.

Those are precisely settings where ignoring the survey design can matter.

### The authors should

Explain whether NHANES survey weights were used.

If not, provide:

1. a justification;
2. weighted sensitivity analyses;
3. ideally weighted estimates of prevalence/calibration;
4. possibly cycle-aware validation.

This is more important here than it would be in a purely discriminative ML benchmark.

---

## 6. Integer encoding of nominal questionnaire variables is inappropriate for logistic regression unless carefully justified

The manuscript states that the questionnaire model includes categorical features such as race/ethnicity, marital status, sexual orientation, smoking status, etc. :chatgpt-content-reference{index="12"}

The limitations section later indicates that nominal variables were retained using released integer coding.

For logistic regression this can impose an artificial ordering and equal spacing:

\[
\beta x
\]

implicitly interprets category 4 as "twice" category 2 in a numerical sense unless contrast encoding is applied.

For variables such as marital status or race/ethnicity, that is generally not defensible.

This is especially concerning because marital status is reported as one of the strongest variables.

I do not think this should be described merely as something that "might slightly improve" performance if one-hot encoding were used.

### Required

Re-run logistic regression using proper categorical encoding.

Options include:

- one-hot treatment coding,
- effect coding,
- carefully justified ordinal encoding only for truly ordinal features.

Then re-evaluate AUROC, calibration, coefficients and feature importance.

---

## 7. Missingness as high as 59% requires more careful treatment

The paper notes that some questionnaire variables were introduced only in later NHANES cycles, leading to missingness as high as 59%. :chatgpt-content-reference{index="13"}

Median imputation inside folds correctly prevents leakage, which is commendable. :chatgpt-content-reference{index="14"}

But this missingness is partly **structural by survey cycle**, not necessarily ordinary missing-at-random observation noise.

Median imputation alone can therefore obscure temporal differences.

I suggest at least:

- missingness indicators;
- cycle identifier or cycle-aware modeling;
- performance by NHANES cycle;
- a temporal experiment: train on earlier cycles, test on later cycles.

That last experiment would be particularly useful because the authors themselves acknowledge temporal changes in HPV epidemiology/vaccination.

---

## 8. The slide-grouped experiment is valuable, but the inference drawn from it is too strong

I appreciate that the authors explicitly identified a major weakness in the standard SIPaKMeD random split.

Under Protocol A, 404/414 cropped test cells come from slides represented in training. :chatgpt-content-reference{index="15"}

The grouped experiment is therefore an important addition.

The slide-grouped student still performs strongly, around 95.8% accuracy and 0.995 abnormal-vs-normal AUROC. :chatgpt-content-reference{index="16"}

However, concluding that slide overlap "did not materially inflate" performance based largely on overlapping confidence intervals is too strong.

Protocol A and B differ in more than grouping:

- Protocol A uses full-field + cropped images.
- Protocol B uses cropped cells only.
- training sample composition changes;
- test set size changes;
- the model is retrained only once per protocol.

Overlapping bootstrap confidence intervals do not establish equivalence.

### Better experiment

Use repeated slide-grouped train/validation/test splits or GroupKFold with multiple seeds.

Report something like:

\[
\Delta \mathrm{Accuracy}
=
\mathrm{Accuracy}_{random}
-
\mathrm{Accuracy}_{grouped}
\]

with uncertainty across independent model fits.

I would make Protocol B the **primary result**, not Protocol A.

---

## 9. The knowledge-distillation formulation is nonstandard and needs correction or ablation

Equation (1) is described as

\[
\mathcal L=
\alpha CE(y,s)
+
(1-\alpha)KL\left(\sigma(t/T)\parallel\sigma(s/T)\right),
\]

but the text states that \(t\) and \(s\) are already the **softmax outputs**, so temperature is being applied to probabilities rather than logits. :chatgpt-content-reference{index="17"}

Standard knowledge distillation applies temperature to logits:

\[
p_T=\operatorname{softmax}(z/T),
\]

often with a \(T^2\) scaling of the KL term.

Applying another softmax to already normalized probabilities substantially changes the intended temperature behavior and compresses the teacher distribution.

This does not invalidate the student classifier performance, but it makes the claim specifically about **knowledge distillation** less clear.

### I would ask for an ablation

Compare:

1. student trained normally without KD;
2. current probability-temperature formulation;
3. standard logit-temperature KD;
4. perhaps \(T^2\)-scaled KD.

Then report accuracy, calibration and convergence.

It may turn out that the B0 architecture itself explains most of the performance.

---

## 10. A single training run is insufficient for the image-model comparisons

The manuscript says one training run per cytology protocol was used, while bootstrap intervals are calculated from held-out predictions. :chatgpt-content-reference{index="18"}

These confidence intervals quantify uncertainty from the **test observations**, not model-training instability.

For deep learning on a dataset of this size, random initialization, augmentation order and split variation could materially affect results.

Please report at least 3–5 independent seeds for:

- B4 teacher;
- B0 student;
- grouped protocol.

Mean ± SD or bootstrap across both cases and runs would be preferable.

---

## 11. External validation of the image model is needed for the strongest clinical claims

SIPaKMeD is a relatively small, single-source dataset.

The manuscript itself recognizes heterogeneity across cervical cytology collections and acknowledges that external validation is absent. :chatgpt-content-reference{index="19"}

For a JBHI paper making claims about calibrated screening support, this is important.

At least one external dataset such as Herlev, CRIC, APACC, or HiCervix would significantly strengthen the work.

Even if exact five-class labels cannot be aligned, the authors' own Bethesda collapse provides a natural mechanism to test:

\[
\text{normal} \quad \text{vs} \quad \text{abnormal}.
\]

This is probably the **single most valuable additional real-data experiment** after the fusion robustness tests.

---

## 12. "Calibration at screening prevalence" should be phrased more carefully

The paper demonstrates an elegant consequence of prior probability correction under label shift:

\[
\operatorname{logit}\tilde p
=
\operatorname{logit}p-\operatorname{logit}\pi+\operatorname{logit}\tau.
\]

:chatgpt-content-reference{index="20"}

That is theoretically appropriate if

\[
P_{\rm train}(x\mid y)
=
P_{\rm deploy}(x\mid y).
\]

But changing prevalence alone does not account for scanner shift, staining differences, demographic change, vaccination-era differences, or altered relationships between covariates and outcomes.

The manuscript should consistently call this something like:

> calibration under simulated/prior-probability shift

rather than implying general deployment calibration.

This is particularly important because the paper is targeted at a clinical informatics journal.

---

## 13. The conformal prediction section is mathematically reasonable but clinically undervalidated

Using Mondrian/class-conditional split conformal prediction is sensible, and the paper correctly gives class-conditional coverage under exchangeability. :chatgpt-content-reference{index="21"}

However, the guarantee applies to the distribution represented by the calibration set.

The fused conformal experiment uses synthetic patients generated from the same basic data-generating process.

Therefore it does not demonstrate coverage under:

- scanner/domain shift;
- temporal shift;
- incorrect epidemiological bridge;
- subgroup shift;
- miscalibrated experts.

A highly useful experiment would report conformal coverage under deliberate shift.

That could become a substantial additional contribution.

---

## 14. Sensitive questionnaire variables require subgroup and fairness analyses

The questionnaire uses highly sensitive variables including race/ethnicity, sexual behavior, sexual orientation, marital status, income, STI history and smoking. :chatgpt-content-reference{index="22"}

For a system proposed for screening support, overall AUROC/ECE are not sufficient.

At minimum report performance/calibration stratified by:

- age;
- race/ethnicity;
- possibly vaccination status/era;
- important sex-behavior strata if sample size permits.

The manuscript should also discuss the potential consequences of using socially sensitive proxies such as marital status as risk predictors.

This is not necessarily an argument to remove those variables, but their behavior should be transparent.

---

# Novelty assessment

I see the novelty as **real but narrower than the paper occasionally suggests**.

The mathematical ingredients themselves are established:

- prior-shift correction;
- likelihood-ratio interpretation;
- Bayesian updating;
- endpoint mapping;
- decision-level classifier combination;
- conformal prediction.

The contribution is their composition into a useful framework for the unusual setting:

> **two separately trained experts, on completely unpaired datasets, predicting different but epidemiologically linked endpoints.**

That is a meaningful systems/methodological contribution.

The related-work section already correctly notes that conventional classifier combination generally assumes experts for the same endpoint/prior and that most missing-modality methods still require some paired observations. :chatgpt-content-reference{index="23"}

However, the paper should sharpen the novelty claim to:

> a principled composition and evaluation framework for heterogeneous endpoint experts without paired multimodal training data

rather than presenting the individual probability operations themselves as fundamentally new.

I would also expand the literature discussion around:

- Bayesian evidence fusion;
- product-of-experts;
- likelihood-ratio fusion;
- diagnostic test pre-test/post-test probability models;
- modular Bayesian models;
- evidence synthesis from independently trained predictors.

---

# Minor comments

A few additional issues are relatively easy to fix:

1. **Model size comparison:** B4 has about 4× the parameters of B0, yet the files differ by 11.4×. :chatgpt-content-reference{index="24"} Clarify whether both files use identical serialization formats and numerical precision.

2. **Model-selection optimism:** six questionnaire families are compared using the same OOF predictions from which the winning family is selected. A nested outer model-selection loop would remove even minor selection optimism.

3. **Calibration reporting:** in addition to ECE, report calibration intercept and slope and preferably confidence intervals. ECE is bin-dependent.

4. **Decision-curve analysis:** because the fusion cohort is semi-synthetic, describe DCA as simulation-based rather than evidence of clinical utility.

5. **Grad-CAM:** the paper appropriately treats the maps only as plausibility checks; retain that cautious language. The manuscript itself notes coarse 12×12 localization and background attention. :chatgpt-content-reference{index="25"}

6. **Terminology:** "screening support" is preferable to language suggesting diagnosis or clinically validated decision support.

7. **Confidence intervals for fusion comparisons:** since each fusion method sees exactly the same simulated subjects, consider paired CIs for metric differences, not merely separate summaries.

8. **Reference scenario:** provide a clinical justification for selecting OR=10 as the headline scenario rather than simply using it as a convenient midpoint.

9. **Abnormality definition:** explicitly give the mapping from the five SIPaKMeD classes to Bethesda-oriented binary status early in Methods. The current mapping is present but deserves stronger emphasis. :chatgpt-content-reference{index="26"}

10. **Code availability:** releasing the fixed splits and per-case predictions is excellent. I strongly encourage releasing the entire simulator and all random seeds/configurations.

---

# What the paper does particularly well

There is quite a lot I like here.

The authors are unusually transparent about leakage. Instead of hiding the SIPaKMeD crop/full-field overlap, they quantify it and run an unseen-slide experiment. :chatgpt-content-reference{index="27"}

The tabular preprocessing is generally careful: imputation/scaling are performed inside folds, oversampling is avoided, and calibrated predictions are generated out of fold. :chatgpt-content-reference{index="28"}

The manuscript also avoids pretending that multimodal fusion gives a dramatic AUROC improvement. The abstract explicitly states that discrimination gains are small and mainly occur when image evidence is uncertain. :chatgpt-content-reference{index="29"} That is scientifically much more credible than overselling a tiny gain.

The missing-modality behavior is elegant:

\[
\text{image only}\rightarrow\sigma(\lambda_i+\operatorname{logit}\tau_A),
\]

\[
\text{questionnaire only}\rightarrow r(x_t),
\]

\[
\text{neither}\rightarrow\tau_A.
\]

:chatgpt-content-reference{index="30"}

And the clinical interpretation—questionnaire as pre-test risk, image evidence as a likelihood ratio—is intuitive and potentially useful.

---

# The three revisions that would change my opinion most

If the authors can only do a few large additions before resubmission, I would prioritize:

**First:** perform **assumption-violation experiments** rather than generating only data conforming to the CO-PRIOR graphical model.

**Second:** perform **external/slide-level image validation**, ideally making the image endpoint patient/slide compatible with the 7.1% patient-level prevalence.

**Third:** redo the NHANES logistic model with **proper categorical encoding, survey-design sensitivity analysis, and subgroup calibration**.

If those three are convincingly addressed, the manuscript would become substantially stronger for JBHI.

---

# Suggested reviewer report

### Summary

This manuscript proposes CO-PRIOR, a decision-level method for combining independently trained cytology and questionnaire models when no subjects are observed in both modalities. The method corrects expert training priors, maps predicted HPV infection risk to a pre-test probability of cytological abnormality through an epidemiological link, and uses the image expert as likelihood-ratio evidence. The work additionally considers missing modalities, knowledge distillation and conformal abstention. The problem is relevant to biomedical informatics and the manuscript is clearly written. The authors also make commendable efforts to control preprocessing leakage and investigate slide overlap.

The main concern is that the principal fusion evaluation is semi-synthetic and its data-generating mechanism enforces much of the conditional structure assumed in deriving the proposed fusion rule. Thus, the result that CO-PRIOR matches a paired-data oracle is partly expected by construction. In addition, the work uses a single-cell abnormality classifier while re-anchoring probabilities to a patient-level abnormal-cytology prevalence, creating a potentially important endpoint/unit mismatch. Several aspects of the NHANES analysis—including complex survey design, nominal-variable coding, structural missingness and subgroup calibration—also need stronger treatment. These issues currently prevent the paper's strongest clinical/calibration claims from being fully supported.

### Recommendation

**Major Revision.**

I would encourage resubmission because the central concept is interesting, the paper is unusually transparent, and the proposed framework has potential. However, stronger validation of the assumptions underlying CO-PRIOR is needed before interpreting the fused output as calibrated patient-level screening risk.

---

## Likely JBHI outcome in the present form

If I were one of the reviewers, I would recommend **Major Revision rather than acceptance**. A particularly strict reviewer could argue for reject-and-resubmit because the patient/cell endpoint mismatch and by-construction semi-synthetic validation affect the central claim, not merely peripheral experiments.

The encouraging part is that I do **not** see the core paper as fundamentally unsalvageable. The strongest idea is the unpaired, endpoint-aware Bayesian fusion framework itself. The manuscript already has a good technical spine; the main task is to make the evidence test the framework rather than mainly instantiate its assumptions.