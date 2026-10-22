# Phase 6 — Gap Analysis

**Evidence base:** 155 verified papers (`Literature/CSV/papers.csv`); 107 with full text. Sources: an in-house corpus of 45 cervical-AI papers (2023–2026, each re-verified against its live source) + 110 papers from an 11-cluster systematic search, each independently checked by an anti-fabrication verifier (**110/110 CONFIRMED to exist; 0 invented**; 3 metadata errors and 3 unsupported number-sets were corrected or blanked in the database).

**Retention rule (as instructed):** a gap is retained **only** if ≥2 *independent* papers support it. Speculative gaps are rejected and listed in §3.

**Provenance convention:** ⟨V⟩ = verified from the source page by a cluster agent and re-checked by the verifier. ⟨DB⟩ = counted directly from our database. ⟨OWN⟩ = our own measurement in this repository (reproducible; scripts in `legacy_src/src/`). ⟨UNVERIFIED⟩ = must be retrieved before citing (see §4).

---

## 1. Retained gaps

### G1 — No multimodal method learns fusion across **disjoint cohorts** (no shared patients, different label spaces); every canonical method requires a paired anchor
**Severity:** high · **Maturity:** unaddressed · **Confidence:** high

**Description.** The entire missing-modality literature assumes the modalities are co-observed on the same subject *somewhere* in the pipeline, and the dependency is **architectural, not incidental**. The anchor takes a different form in each method but is always present:

| Method | The paired anchor it cannot do without |
|---|---|
| MMANet (CVPR 2023) | a teacher pre-trained on **complete** modalities |
| ShaSpec (CVPR 2023) | shared encoder + distribution-alignment loss learned from **co-observed** modalities |
| SMIL (AAAI 2021) | modality priors learned from examples where the modality **is** observed alongside the other |
| mmFormer (MICCAI 2022), HeMIS (MICCAI 2016) | a shared/aligned latent space learned from paired scans |
| Flex-MoE (NeurIPS 2024) | full-modality samples for router generalisation |
| ALTER | 6,850 **paired** same-patient TCGA triplets |
| Reza et al. (TPAMI 2024), Lee et al. (CVPR 2023) | a full-modality pretrained backbone (ViLT) |

**Nearest misses, and why they still fail here** ⟨V⟩:
- **Kim & Kim (ECCV 2024, arXiv:2407.12616)** explicitly rejects the jointly-pretrained-encoder assumption ("we propose utilizing independently pretrained unimodal encoders for each modality") — but still "requires modality-complete data during training to optimize feature prediction."
- **Tran et al. (arXiv:2606.10504)** handles unpaired distillation but presupposes a **shared label space**.
- **Eklund (arXiv:2601.17986)** names the exact setting ("strictly private and share no common samples") but needs a **public anchor set** that does not exist here.
- **Vertical federated learning** is structurally the *inverse* (different features over *shared* samples) and degenerates at zero overlap.

**Supporting papers (independent):** MMANet (Wei et al., CVPR 2023, arXiv:2304.08028); Flex-MoE (NeurIPS 2024); Reza et al. (TPAMI 2024); HeMIS (Havaei et al., MICCAI 2016); ShaSpec (Wang et al., CVPR 2023); SMIL (Ma et al., AAAI 2021); mmFormer (MICCAI 2022); Kim & Kim (ECCV 2024). Confirmed independently by 4 clusters.

---

### G2 — Calibration is **not reported** anywhere in the missing-modality, evidential, cervical-multimodal or domain-shift literatures; and **no paper asks whether confidence *should* degrade when a modality is absent**
**Severity:** high · **Maturity:** unaddressed · **Confidence:** high

**Description.** The most uniform absence in the corpus, holding across four independent clusters ⟨DB⟩:

| Cluster | Papers reporting a calibration method |
|---|---|
| missing-modality-general | **0 / 10** |
| missing-modality-medical | **0 / 10** |
| domain-shift / external validation | **0 / 10** |
| in-house cervical corpus | **2 / 45** |

No cervical multimodal paper reports ECE, Brier or a reliability diagram. TopotiCerv (2026) claims a "calibrated" output **with no method named**; Madathil (2025) claims "better calibrated" **without naming technique or metric**; only Wang (2025, PLNM) names a method (Platt scaling) and it is a *prognostic ultrasound* task ⟨V⟩. Yin et al.'s 2026 healthcare-fusion benchmark selects accuracy, robustness and fairness as its axes and **omits calibration entirely** — confirming it is not yet treated as a benchmark dimension ⟨V⟩.

**⚠ Constraint on this gap (contrary evidence — must be honoured).** **AECF** (Chlon et al., arXiv:2505.15417) *does* claim and measure exactly this property: "A novel contrastive loss enforces monotone confidences across *all* 2^M−1 modality subsets", with per-dropout ECE (full 0.011→**0.009**; 30% dropout **0.014**; 50% dropout **0.020**) ⟨V⟩. Therefore:
- the claim "per-subset calibration is unreported in missing-modality work" is **false as stated** and must be narrowed to *the medical/cervical literatures*;
- quoting TMUR's 26.50%→18.23% ECE as "the field's best" would be **cherry-picked** (AECF is ~0.01–0.02, orders of magnitude better) and a reviewer will catch it.
AECF does **not** pre-empt the disjoint-cohort setting: it trains on complete paired data (AV-MNIST, MS-COCO) with *simulated* masking.

**Supporting papers:** Wu et al. (TMLR survey, arXiv:2409.07825); HeMIS; mmFormer; ShaSpec; MMANet; Flex-MoE; DrFuse; DRIM; Yin et al. (2026). Contrary: AECF; TMUR; TITAN (measures ECE, applies no recalibration).

---

### G3 — No method composes two experts that each carry a **different baked-in training-cohort prior**; prior-shift correction exists only for *one* classifier retargeted to *one* population ★
**Severity:** high · **Maturity:** unaddressed · **Confidence:** high

**Description.** This is the sharpest unclaimed methodological gap in the corpus and a prerequisite the target system cannot skip. The literature splits cleanly either side of it and **never crosses**:

- **Prior correction, one classifier → one population:** Saerens, Latinne & Decaestecker (Neural Computation 2002, 10.1162/089976602753284446) — the canonical EM procedure; Godau et al. (MICCAI 2023, arXiv:2303.12540) — deployment under prevalence shift; Alexandari et al. (ICML 2020, arXiv:1901.06852) — MLLS+BCTS.
- **Combining several classifiers, under a *common* prior:** Kittler et al. (TPAMI 1998); Genest & Zidek (Statistical Science 1986); ProbEn (ECCV 2022).

**Nobody performs prior-*stripping* as a precondition of pooling.** Each expert's posterior is contaminated by its own cohort's prevalence — Herlev's is curated by construction (its creators selected cells "not to collect a natural distribution"); UCI sits at ~6.4% biopsy-positive (14.6:1 imbalance) ⟨V⟩. Composing them requires dividing each expert's own prior *out* before pooling and applying the deployment prior *after*. An independent OpenAlex search for prevalence correction when *combining* classifiers across populations returned nothing on point ⟨V⟩.

**⚠ Constraint.** Alexandari et al. (ICML 2020) is a live methodological warning, not just a citation: Saerens's EM "does not require model retraining", but "a limiting assumption of this algorithm is that p(y|x) is calibrated, **which is not true of modern neural networks**" ⟨V⟩. Any prior-correction step must use BCTS+MLLS or justify why not.

**Supporting papers:** Saerens et al. (2002); Godau et al. (2023); Alexandari et al. (2020); Kittler et al. (1998); Genest & Zidek (1986); ProbEn (2022).

---

### G4 — Conditional-independence (naive-Bayes) fusion is the **only** route available without paired data, and it is the route that miscalibrates most — in a known direction, by a quantified amount
**Severity:** high · **Maturity:** partially-solved · **Confidence:** high

**Description.** **Grill et al. (Statistics in Medicine 2017, 10.1002/sim.7190, PMC8182952)** converts "we assume conditional independence" from a footnote into a **pre-registered, quantified risk**: naive-Bayes-style independent combination **overestimates risk by up to 56%** (≈27% on real data) ⟨V⟩. This is directional (over-confidence) and therefore *predictable and testable* — a fused system built this way should be expected to be over-confident, and that expectation can be pre-registered as a falsifiable property of its own output.

Crucially, this is not avoidable by better engineering: with zero cohort overlap the joint dependence structure is **non-identified**, so the independence assumption is forced, not chosen.

**Supporting papers:** Grill et al. (Stat Med 2017); Kittler et al. (TPAMI 1998); ProbEn (ECCV 2022); Genest & Zidek (1986); statistical-matching literature (non-identified without conditional independence).

---

### G5 — Per-expert conformal guarantees **do not compose** into a joint guarantee across disjoint cohorts; conditional coverage per modality-availability pattern is unestablished
**Severity:** high · **Maturity:** emerging · **Confidence:** medium

**Description.** Conformal prediction's guarantee is defined over a *single exchangeable calibration population* and is **marginal** — averaged over the population, not conditional on a subgroup (Angelopoulos & Bates, arXiv:2107.07511) ⟨V⟩. Two experts calibrated on two different cohorts therefore yield two marginal guarantees that **do not multiply into a joint one**.

**⚠ Constraint.** **Podkopaev & Ramdas (UAI 2021, arXiv:2103.03323)**: "It is known that label shift hurts prediction, and we first argue that it also hurts UQ, by showing degradation in coverage and calibration"; their fix reweights conformal/calibration procedures "whenever **some unlabeled data from the target distribution is available**" ⟨V⟩ — a precondition that is **unmet here** (no joint-distribution data exists). Any conformal claim must therefore be demoted to **per-path marginal coverage, separately calibrated per expert**; a joint guarantee would be a genuine technical error.

**Supporting papers:** Angelopoulos & Bates (arXiv:2107.07511); Zaffran et al. (ICML 2023, conformal with missing values); Podkopaev & Ramdas (UAI 2021); Olsson et al. (Nat Commun 2022, Mondrian CP).

---

### G6 — Independently estimated **per-expert uncertainties are not numerically comparable** — the most likely silent failure of any disjoint-cohort fusion
**Severity:** high · **Maturity:** emerging · **Confidence:** high

**Description.** Every evidential and quality-aware fusion method silently assumes per-view uncertainties share a scale. **TMUR (arXiv:2604.09288, 2026)** attacks the assumption directly and shows it is **false** for independently trained branches, reporting ECE 26.50% → 18.23% after unified routing ⟨V⟩. Independently, **Jürgens et al. (ICML 2024)** establish that EDL epistemic uncertainty is **"relative (rather than absolute)"** ⟨V⟩.

**Consequence for any design:** an evidential *u* is defensible as a **ranking signal only**. If a paper says "uncertainty-aware" and means "evidential *u* is a probability", it fails. Calibrated P(abnormal) must come from an explicit, separately-evaluated post-hoc step with reported ECE/Brier.

**Supporting papers:** TMUR (2026); QMF (ICML 2023); Jürgens et al. (ICML 2024); TMC (ICLR 2021) / TMC-DEF (TPAMI 2023).

---

### G7 — No evidential fusion paper evaluates a **truly absent** modality as a vacuous opinion; missingness is universally handled by **imputation**, which is structurally impossible across disjoint cohorts
**Severity:** high · **Maturity:** emerging · **Confidence:** high

**Description.** A vacuous opinion (b = 0, u = 1) is the **identity element of Dempster's rule** and is the obvious principled mechanism for any-subset operation — a view that knows nothing contributes nothing, automatically. Yet no paper in the evidential cluster evaluates it. Missingness is instead handled by imputation (EDP-MVC's K-means class-centre imputation; APLN's coarse imputation) ⟨V⟩ — which **structurally requires a shared cohort to impute from** and is therefore unavailable at zero overlap.

**Supporting papers:** TMC (ICLR 2021); TMC-DEF (TPAMI 2023); EDP-MVC; APLN; ALTER (the sole any-to-any framework: "attention is computed over the available subset of modalities… without requiring imputation" — but pretrains on 6,850 *paired* triplets).

---

### G8 — Both arms sit on **leakage-inflated** benchmarks: UCI tabular reports 0.98–1.00 AUC fully accounted for by target + resampling leakage; cervical image datasets make patient-wise splitting **impossible in principle**
**Severity:** high · **Maturity:** unaddressed · **Confidence:** high

**Description — tabular arm.** The honest anchor comes from the dataset creators themselves: **Fernandes, Chicco, Cardoso & Fernandes (PeerJ CS 2018, 10.7717/peerj-cs.154)** use stratified 10-fold CV with **nested** inner CV and apply **no oversampling** ⟨V⟩. Against that, published UCI results cluster at 0.98–1.00 AUC via SMOTE-before-split and/or `Dx:*` target leakage — e.g. Roy et al. (DIGITAL HEALTH 2025) report **0.98 accuracy / 0.995 AUC** with SMOTE and author-confirmed **no external validation** ⟨V⟩.

**Our independent corroboration** ⟨OWN⟩ (reproducible, this repository): under stratified nested CV, leakage-safe features (`Dx:*` dropped, no oversampling before split):

| Target | AUROC | 95% CI (bootstrap) | Verdict |
|---|---|---|---|
| **Biopsy** | **0.656** | [0.584, 0.720] | best; significantly > 0.5 but weak |
| Schiller | 0.619 | [0.549, 0.689] | weak |
| Citology | 0.488 | [0.407, 0.575] | **≤ chance** |
| Hinselmann | 0.450 | [0.353, 0.542] | **≤ chance** |

A hyperparameter-tuned bake-off over **8 model families** (LogReg, RF, ExtraTrees, HistGB, XGBoost, SVM-RBF, GaussianNB, kNN) with nested CV does not exceed **AUROC ≈ 0.65** for Biopsy. Decomposition of the leak: predicting `Dx:Cancer` from the other `Dx` columns + test outcomes gives **AUROC 0.998**; from leakage-safe features only, **0.592**. The single most predictive feature reaches only AUROC **0.572**; mutual information is **< 0.01** for every feature — i.e. there is little signal to model.

**Description — image arm.** SIPaKMeD and Herlev release **no patient IDs at all**, so patient-wise splitting is impossible *in principle*; cells cropped from the same slide can fall on both sides of a random split ⟨V⟩.

**Supporting papers:** Fernandes, Chicco et al. (PeerJ CS 2018); Sun et al. (Front Oncol 2022); Ijaz et al. (Sensors 2020); Roy et al. (DIGITAL HEALTH 2025); Demircioğlu (leakage protocol, radiomics).

**⚠ Constraint.** Demircioğlu's +0.34 AUC leakage-bias bound was established on **15 radiomic datasets, never on UCI cervical** — it transfers **by analogy only** and must be labelled as such.

---

### G9 — Cervical AI benchmark numbers **do not transfer** (10–45 point collapse documented), yet essentially **one** cross-dataset transfer matrix exists and calibration under shift has **never** been measured
**Severity:** high · **Maturity:** partially-solved · **Confidence:** high

**Description.** The contradiction is quantified from both sides. Liu et al. (eClinicalMedicine 2024, 77 studies, QUADAS-2) pools AI-assisted cytology at **AUC 0.99 (95% CI 0.97–0.99)** while simultaneously stating prospective evidence is lacking ⟨V⟩. Where external validation *is* performed: **CerviPro** C-index 0.806 internal → **AUC ≈ 0.63 external**, with Kaplan–Meier separation losing significance (p=0.085, p=0.211); the **Multi-Modal GNN dropped ~9%** on 1,000 real-world Apollo Hospitals images ⟨V⟩. **Coskun et al. (Bioengineering 2026)** is the **only** verified paper reporting a SIPaKMeD↔Herlev transfer matrix, despite 10 of 77 studies using both datasets and never reporting transfer ⟨V⟩.

**Our corroboration** ⟨OWN⟩: our SIPaKMeD-trained cytology model scores **0.998 AUROC in-distribution** (leaky random split) but **0.73 / 0.59 / 0.53** on Herlev / HiCervix / APACC — a 27–47 point collapse consistent with the literature.

**Supporting papers:** Coskun et al. (2026); Ahmed et al. (PLOS Digital Health 2024); Cai et al. (LSA, arXiv:2503.06563); Liu et al. (eClinicalMedicine 2024); Tran et al.

---

### G10 — Naive late fusion can be **worse than the better single expert**, yet "does fusion ever hurt?" is never posed as a first-class research question
**Severity:** medium · **Maturity:** unaddressed · **Confidence:** medium

**Description.** **Kelebet et al. (BMC Med Inform Decis Mak 2025, 10.1186/s12911-025-03174-6)** is the closest structural analogue to the target system — VGG16 on histopathology images + Random Forest on patient card sheets, combined by **late fusion** — and the fused model (**92%**) is **worse than the tabular expert alone (96%)** ⟨V⟩. This is the citable precedent that fusion is not free.

**Our corroboration** ⟨OWN⟩: a dependence-sensitivity envelope over the real marginals (image 0.73 external, tabular 0.65) shows equal-weight sum-rule fusion yields only **+0.016 under independence** and **−0.030 under realistic positive dependence** (both experts are HPV-driven, so ρ>0 is the realistic region).

**Supporting papers:** Kelebet et al. (2025); Yin et al. (arXiv:2602.23614); Grill et al. (2017).

---

### G11 — Explanation ≠ clinical benefit; attribution XAI dominates cervical AI yet is essentially never clinician-evaluated, and **no system explains which modality drove a decision**
**Severity:** medium · **Maturity:** emerging · **Confidence:** high

**Description.** **Jabbour et al. (JAMA 2023**, randomised clinical-vignette survey, **n=457 clinicians)** found image-based AI explanations **did not mitigate** the harm of a systematically biased model ⟨V⟩. Independently, Din et al. (Frontiers in AI 2026) converge on the same negative result. Nothing in the corpus attributes a decision **across** modalities ("the image drove this, the questionnaire dissented"), and no reader study presents per-modality explanations or per-modality confidence ⟨V⟩.

**Supporting papers:** Jabbour et al. (JAMA 2023); Din et al. (Frontiers in AI 2026); Roy et al. (2025).

---

### G12 — Deferral in cervical screening **works clinically** but is threshold-based, uncalibrated and modality-blind; the calibrated learning-to-defer literature has never been applied to it
**Severity:** high · **Maturity:** emerging · **Confidence:** high

**Description.** Deferral is the proven clinical pattern: **Xue et al. (Modern Pathology 2023)** auto-triage negatives below a **single fixed raw-score threshold of 0.35** with a cytologist-in-the-loop ⟨V⟩ — no calibration, no modality-awareness. Meanwhile no learning-to-defer method (Mozannar & Sontag; Verma & Nalisnick) conditions the reject decision on **which modalities are present**, and no work validates abstention rates against real cytologist workload ⟨V⟩.

**Supporting papers:** Xue et al. (Modern Pathology 2023); Wang et al. (2025); Mozannar & Sontag (ICML 2020); Verma & Nalisnick.

---

## 2. Absence findings (verified affirmatively, paper-by-paper — not inferred from null searches)

These are **results**, not missing work:

1. **No public dataset pairs a cervical image with a risk questionnaire for the same patient — and the barrier is structural.** Every public cervical image set carries **zero patient metadata**: SIPaKMeD (4,049 cells / 966 cluster images), Herlev (917 cells), HiCervix (40,229 cells / 4,496 WSIs), CRIC (400 images / 118 patients), APACC (103,675 cells / 107 smears), Mendeley LBC (963 images / 460 patients), Cx22, CPSMI2025. **The decisive test:** UCI 384 ("Quality Assessment of Digital Colposcopies") has the **same authors, same hospital and same year** as UCI 383, yet has **no questionnaire attributes, no patient identifier, and mismatched cardinality** (287 images vs 858 patients) — the single most likely candidate pair in existence is a confirmed miss. The cause is consent/de-identification (CRIC: "it is not possible to provide demographic information on the participants… including age"; APACC: "The data collected from anonymous samples preclude the identification of patients"). **A paired corpus will not emerge from re-annotating existing cohorts; it requires new prospective consent.**
2. **The one cohort with both modalities is closed, mismatched, and — decisively — was never fused.** The NCI Guanacaste cohort (Hu et al., JNCI 2019) has cervigrams *and* enrollment risk-factor interviews for the same 9,406 women, but is controlled-access, uses film cervigrams from "a discontinued film camera technique" (not Pap cytology), and its published AVE algorithm is **image-only** (AUC 0.91), using risk factors merely as Table-1 descriptors. *Even where fusion was possible, nobody did it.*
3. **No cervical system anywhere supports missing-modality inference** — 0/10 across the cervical-multimodal cluster; independently confirmed 10/10 in the domain-shift and XAI/CDS clusters. Only the Multi-Modal GNN paper even names the deficit.
4. **No cervical-specific missing-modality method exists at all.**
5. **No method trains multimodal fusion across fully disjoint cohorts with different label spaces** (independently confirmed by 4 clusters). Note the field's own survey (Wu et al., TMLR) lists six open issues and **neither calibration-under-missing-modality nor the disjoint-cohort setting is among them.**
6. **No calibration/CP for a fused cervical system, and none on UCI at all** — searched directly, zero hits; all 8 cervical-tabular records return "not reported" for **both** calibration and uncertainty.
7. **No foundation model accepts a structured risk-factor vector.** Where pathology FMs are "multimodal", the second modality is **always free text**, never structured patient variables (UNI, CONCH, Virchow, Prov-GigaPath, TITAN). The two cervical FMs are strictly image-only (UniCAS: 48,532 WSIs, slide AUC 92.60%, CC BY 4.0; Smart-CCS: 127,471 WSIs / 48 centres, internal AUC 0.965 / external 0.950, Apache 2.0).
8. **A definitional conflation is widespread:** true cross-modal (image + non-image clinical) vs **multi-*view* of the same organ relabelled "multimodal"** (CerviFusionNet's four colposcopy captures; Multi-Modal GNN's image + its own mask + its own graph; Niu 2025's radiomics + deep features from the identical region). In the latter **a missing modality cannot arise by construction** — which partly explains why the field never confronted the problem.

---

## 3. Rejected gaps (insufficient or contrary evidence)

Per the instruction to reject unsupported gaps, the following were **considered and rejected**:

| Candidate gap | Why rejected |
|---|---|
| "Per-subset calibration is unreported in missing-modality work" | **Refuted by AECF** (arXiv:2505.15417), which claims *and measures* monotone confidence across all 2^M−1 subsets. Narrowed to the *medical/cervical* literatures only (G2). |
| "Modality dropout is unexplored for image+tabular" | **Refuted by Gu, Saito & Ma (MICCAI 2025, arXiv:2509.18284)** — image+tabular with learnable modality tokens and improved modality dropout, reporting SOTA "particularly… where only a single modality is available" (AUROC 0.836→0.840; 0.712→0.722). **Modality dropout must not be claimed as novel.** |
| "Randomly pairing subjects across cohorts sharing only a label has no dedicated critique" | **Refuted by Poh & Bengio's "chimeric users" papers** (MLMI 2005 / LNCS 2006, 10.1007/11677482_8; ICASSP 2006, 10.1109/icassp.2006.1661466) — at least two dedicated papers ask precisely whether one may fabricate virtual subjects by pairing traits of different people across disjoint databases. ⚠ Their **content is unverified** (publisher-elided abstracts): existence HIGH confidence, conclusions **unknown** — must be retrieved before any pseudo-pairing is attempted. |
| "Combining expert probabilities with weights is novel" | **Refuted by Genest & Zidek (Statistical Science 1986)** — a 40-year characterised literature (linear vs logarithmic opinion pools). |
| "Bayes-rule fusion that marginalises over absent modalities is novel" | **Refuted by ProbEn (ECCV 2022 oral, arXiv:2104.02904)**: "derived from Bayes' rule and first principles… through probabilistic marginalization, ProbEn elegantly handles missing modalities." |
| "Fusing disjoint samples is an unstudied problem" | **Tempered by statistical matching / data fusion** (official statistics, ~50 years; CRAN `StatMatch`). Note our setting is **strictly harder**: statistical matching needs a common target population **and** shared common variables Z — SIPaKMeD/Herlev vs UCI have **neither**. |

---

## 4. Evidence hygiene — leads located but **NOT** verified (must not be cited without retrieval)

| Lead | Status |
|---|---|
| MC2 (Research Square, modality-conditional CP) | real but content-unverified (publisher 403). **Do not cite its 0.73-vs-0.90 figure.** |
| Uddin et al. 2024 (Neuroscience Informatics) — apparently the only UCI paper self-acknowledging oversampling leakage | snippet only (ScienceDirect 403) |
| Systematic review reporting ~88.9–92.6% of risk-factor works use the single UCI dataset | ScienceDirect 403 |
| UAD-FM (npj Digital Medicine 2025) | Nature auth wall |
| Poh & Bengio chimeric-user papers | existence verified; **conclusions unknown** |

---

## 5. What the gap analysis implies (bridge to Phase 7)

Three gaps are **prerequisites the target system cannot skip** and are simultaneously **uncollided**: **G3** (two-prior composition), **G8** (leakage-audited honest baseline), **G4** (the quantified price of the forced independence assumption). Gaps G1/G7 are real but the *mechanism* that answers them (vacuous-opinion any-subset inference) is inherited, not invented. Gaps G2/G5/G6 are constrained by AECF, Podkopaev & Ramdas, and TMUR respectively, and must be claimed narrowly.

**The construct/unit-of-analysis mismatch (see `10_NOVELTY_STATEMENT.md` §2) is not in this list because it is not a gap in the literature — it is a defect in the naive framing of the task itself, and it is fatal to any system that maps a cytomorphology posterior and a histology-risk posterior onto one axis and calls the result P(abnormal).**
