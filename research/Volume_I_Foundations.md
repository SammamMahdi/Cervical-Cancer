# A Trustworthy, Generalizable Framework for Deep-Learning Cervical Cytology
## Volume I — Foundations

*A research monograph (thesis-format supplementary volume).*

---

### Note on evidence, citations, and quantitative claims

This volume is written to a doctoral-dissertation standard. In accordance with the research
methodology it advocates, the following editorial conventions are observed throughout:

- **No citation has been fabricated.** All references are to genuine, well-established works to the
  best of the author's knowledge. Where a specific peer-reviewed source could not be confidently
  identified, the text uses general scholarly phrasing ("a substantial body of work has shown…")
  rather than inventing a reference.
- **No numerical claim has been fabricated.** Epidemiological figures are attributed to their primary
  source (e.g., GLOBOCAN 2020 / Sung et al., 2021; World Health Organization). Because epidemiological
  estimates are periodically revised, exact figures **must be re-verified against the current primary
  source before final submission**; passages requiring such verification are marked *[verify]*.
- The text explicitly distinguishes four epistemic categories: **established knowledge** (textbook or
  consensus fact), **evidence-supported claim** (a finding reported in the literature),
  **reasonable interpretation** (the author's synthesis of evidence), and **speculation** (a
  forward-looking conjecture, always flagged).
- The framing inputs — research gaps G1–G7, research questions RQ1–RQ7, hypotheses H1–H7,
  contributions C1–C6, and the evaluation protocol — are treated as fixed and are reproduced verbatim
  in Chapter 4.

---

# Chapter 1 — Introduction

## 1.1 The global burden of cervical cancer

Cervical cancer remains one of the most significant yet most preventable malignancies affecting women
worldwide. According to the GLOBOCAN 2020 estimates (Sung et al., 2021), cervical cancer was the
fourth most commonly diagnosed cancer and the fourth leading cause of cancer death in women globally,
with on the order of 604,000 new cases and 342,000 deaths in 2020 *[verify against GLOBOCAN 2020/2022]*.
The disease is distinguished from most other major cancers by two features that frame the entire
scientific motivation of this work. First, its principal cause is known: persistent infection with
high-risk human papillomavirus (HPV) is a necessary cause of effectively all cervical carcinomas, a
relationship established by zur Hausen's work on HPV oncogenesis and by large molecular epidemiological
studies that detected HPV DNA in the overwhelming majority of cervical tumours (Walboomers et al.,
1999). Second, the disease has a long, detectable pre-invasive phase — cervical intraepithelial
neoplasia — during which intervention is highly effective. Together these facts make cervical cancer,
in principle, a controllable disease through vaccination, screening, and treatment of precancer.

The burden is, however, profoundly inequitable. The large majority of cervical-cancer deaths occur in
low- and middle-income countries (LMICs), where organised screening infrastructure, trained
cytopathologists, and treatment capacity are scarce (World Health Organization, 2020). This disparity
is the clinical backdrop against which automated, low-resource-deployable screening tools acquire their
significance. In 2020 the World Health Organization launched a Global Strategy to accelerate the
elimination of cervical cancer as a public-health problem, defining an incidence threshold below four
cases per 100,000 women-years and the "90–70–90" interim targets for 2030: 90% of girls fully
vaccinated against HPV by age 15; 70% of women screened with a high-performance test by ages 35 and 45;
and 90% of women with cervical disease receiving treatment (World Health Organization, 2020). Achieving
the screening target in resource-constrained settings is widely recognised as a workforce and
throughput problem as much as a coverage problem, and it is precisely this bottleneck that motivates
research into reliable computational assistance for cytological interpretation.

It is an **established** fact that cervical cancer is preventable and that screening reduces mortality;
it is a **reasonable interpretation**, supported by the structure of the WHO strategy, that scaling
screening to the required level in LMICs cannot be achieved by expanding the human cytology workforce
alone, and therefore that trustworthy automation has a meaningful role to play.

## 1.2 The clinical workflow of cervical cancer screening

Cervical screening is a multi-stage clinical pathway rather than a single test, and understanding this
pathway is essential to situating the contribution of an artificial-intelligence system correctly.
In contemporary practice the pathway typically comprises: (i) specimen collection from the
transformation zone of the cervix; (ii) primary testing, which may be HPV-based molecular testing,
cytological examination, or co-testing; (iii) triage of screen-positive women; (iv) colposcopic
examination with directed biopsy where indicated; and (v) histopathological confirmation, which is the
diagnostic reference standard. Cytology occupies the second and third stages of this pathway: it
functions both as a primary screen (in cytology-first programmes) and, increasingly, as a triage test
for HPV-positive women in HPV-first programmes.

The diagnostic reference standard for cervical disease is histopathology of a tissue biopsy, not
cytology. This distinction is not pedantic; it has direct methodological consequences for any AI system
trained on cytological labels, because cytological categories are screening categories defined to
stratify risk and direct further management, not definitive tissue diagnoses. An automated cytology
classifier therefore operates within a screening decision, where the relevant performance question is
not "what is the final diagnosis?" but "what is the appropriate next action, and with what confidence?"
This reframing — from diagnosis to risk-stratified triage under uncertainty — is a recurring theme of
this monograph and directly motivates the emphasis on calibration and abstention developed in later
chapters.

## 1.3 Cytology-based screening methods

Cytological screening originates with the Papanicolaou test (Papanicolaou & Traut, 1941), which
revolutionised cervical-cancer control in the twentieth century by enabling detection of pre-invasive
disease from exfoliated cells. Two preparation modalities dominate contemporary practice and are
central to the technical problem addressed in this work. **Conventional (smear) cytology** spreads
cellular material directly onto a glass slide; it is inexpensive and widely available but is subject to
uneven cell distribution, obscuring blood and inflammation, and air-drying artefact. **Liquid-based
cytology (LBC)**, introduced commercially in the 1990s (e.g., ThinPrep and SurePath systems), suspends
cells in a fixative and produces a cleaner, more uniform monolayer; it reduces unsatisfactory-slide
rates and enables ancillary molecular testing from the same vial. Crucially for this thesis, the two
modalities differ systematically in cell density, background, focus, staining response, and
segmentation amenability. These differences constitute a concrete, clinically meaningful source of
**domain shift** between datasets — a point developed rigorously in Chapter 2 and central to research
gap G2.

Cytological findings are reported using standardised grading systems (Section 2.3). The interpretive
task itself — scanning a slide containing tens of thousands of cells to identify a small number of
abnormal cells — is laborious, requires scarce expertise, and is subject to inter-observer variability
and fatigue. These properties make cytology an archetypal candidate for computational assistance, while
simultaneously imposing stringent reliability requirements: an automated system that fails silently on
the rare abnormal cell, or that is confidently wrong on slides from an unfamiliar laboratory, is not
merely unhelpful but potentially harmful.

## 1.4 The current role of artificial intelligence in cervical cytology

Automated assistance for cervical cytology is not new. Semi-automated image-analysis systems for
slide screening were developed and, in some jurisdictions, cleared for clinical use decades ago,
historically based on handcrafted morphometric features and classical pattern recognition. The modern
wave of interest, however, is driven by deep learning, which has displaced handcrafted-feature pipelines
across medical image analysis more broadly (Litjens et al., 2017). In cervical cytology specifically,
deep convolutional neural networks are now routinely reported to achieve high accuracy on
single-cell classification benchmarks. The role currently envisaged for such systems spans a spectrum:
as a *pre-screening* filter that prioritises slides or fields for human review; as a *second reader*
that flags potentially missed abnormalities; and, more ambitiously, as a *primary* automated classifier
in settings without sufficient cytotechnologist capacity.

It is an **established** observation that deep learning achieves strong within-dataset accuracy on
curated cervical-cytology benchmarks. It is, however, an **evidence-supported** and increasingly
recognised concern — and a central thesis of this work — that this within-dataset accuracy has been
repeatedly conflated with clinical readiness, and that the gap between the two has not been adequately
characterised, let alone closed. The role of AI in cytology is therefore best described at present as
*demonstrated in principle but not yet trustworthy in practice*; converting the former into the latter
is the overarching objective of this research.

## 1.5 The evolution of deep learning in cervical cytology

The trajectory of computational cervical cytology mirrors the broader evolution of medical-image
analysis but with instructive local features. The earliest automated approaches relied on explicit
cell and nucleus **segmentation** followed by extraction of handcrafted morphometric descriptors —
nuclear area, nuclear-to-cytoplasmic (N/C) ratio, chromatin texture, and shape descriptors — fed to
classical classifiers. The public **Herlev** dataset (Jantzen et al., 2005) and, later, the
**SIPaKMeD** dataset (Plissiti et al., 2018) were curated within this paradigm and provided per-cell
images with expert morphological labels; they remain the most widely used single-cell benchmarks and
anchor much of the empirical literature.

The advent of deep convolutional networks (Krizhevsky et al., 2012; LeCun, Bengio & Hinton, 2015)
shifted the field from feature engineering to representation learning. Early deep approaches to
cervical cytology — for example, the line of work exemplified by deep convolutional classification of
Pap-stained cells (e.g., Zhang et al., 2017) — demonstrated that end-to-end networks could match or
exceed handcrafted pipelines on Herlev-style benchmarks. Subsequent work progressively imported the
methodological advances of mainstream computer vision: deeper and more efficient backbones such as the
EfficientNet family (Tan & Le, 2019); transfer learning from ImageNet pretraining; attention and
transformer architectures; and, more recently, knowledge distillation (Hinton, Vinyals & Dean, 2015)
to compress high-capacity models for deployment. In parallel, explainability methods such as
class-activation mapping (Selvaraju et al., 2017) and post-hoc calibration analyses (Guo et al., 2017)
began to be applied, though typically as add-ons rather than as primary objects of study.

The **interpretation** offered here is that the field's methodological evolution has been largely
*accuracy-driven*: each new architecture or training trick is validated by an incremental gain in
within-dataset accuracy on Herlev or SIPaKMeD. The properties that determine clinical trustworthiness —
honest evaluation free of leakage, cross-domain generalisation, calibrated uncertainty, principled
handling of incompatible label spaces, and faithful explanation — have received comparatively little
systematic attention. This imbalance is the scientific opening that the present research is designed to
address.

## 1.6 Current state-of-the-art approaches

The contemporary state of the art in single-cell and slide-level cervical cytology can be summarised,
at a conceptual level, along four axes. First, **architecture**: high-accuracy results are reported with
modern convolutional backbones (e.g., EfficientNet, ResNet, DenseNet) and increasingly with vision
transformers and hybrid convolution–attention models; reported within-dataset accuracies on SIPaKMeD
and Herlev frequently exceed 0.95. Second, **learning strategy**: transfer learning from natural-image
pretraining is near-universal, and techniques such as data augmentation, class weighting, and ensemble
methods are commonly employed to address imbalance and limited data. Third, **scope**: most published
systems are single-cell classifiers; a smaller body of work addresses whole-slide or field-of-view
analysis, multiple-instance learning, and detection-then-classification pipelines. Fourth,
**auxiliary capabilities**: a minority of studies report explainability visualisations, calibration
metrics, or external validation.

Two characteristics of this state of the art are critical for the present thesis and are stated here as
**evidence-supported** observations consistent with the published record. (i) The dominant evaluation
paradigm is *single-dataset*: a model is trained and tested on partitions of one collection, and the
resulting accuracy is reported as if it were an intrinsic property of the model rather than of the
model-plus-dataset. (ii) When external or cross-dataset validation is performed at all, it is the
exception rather than the norm, and the label-space incompatibilities between datasets are frequently
handled by informal relabelling whose assumptions are not made explicit. The cumulative consequence,
developed in Section 1.9, is that the headline numbers reported by the state of the art are not, in
general, valid estimates of the quantity that matters clinically — performance on cells from a new
laboratory under a documented decision rule.

## 1.7 Persistent scientific limitations

Synthesising the foregoing, seven persistent and interrelated scientific limitations characterise the
current literature. These are stated here as the conceptual antecedents of the formal research gaps
G1–G7 (Chapters 3–4); they are introduced now to motivate the study, and each is substantiated against
the literature in Chapter 3.

1. **Evaluation integrity (antecedent of G1).** Reported accuracy is frequently inflated by data
   leakage — most commonly, placing cells originating from the same slide, field of view, or patient
   into both training and test partitions — and splits and seeds are rarely released, undermining
   reproducibility. This is a documented failure mode across machine-learning-based science
   (Kapoor & Narayanan, 2023) and is acutely relevant to cytology, where multiple cells share a single
   imaging context.
2. **Cross-domain generalisation (antecedent of G2).** Models trained on one preparation/staining
   pipeline degrade, often severely, when applied to another. The phenomenon is an instance of the
   broader problem of dataset shift and of "shortcut learning," in which models exploit
   preparation-specific cues rather than the causal morphological signal (Geirhos et al., 2020).
3. **Uncertainty and abstention (antecedent of G3).** Modern networks are typically miscalibrated and
   overconfident (Guo et al., 2017); cervical-cytology systems generally output point predictions with
   no calibrated confidence and no principled mechanism to abstain and defer to an expert — a
   prerequisite for safe screening triage.
4. **Label-space incompatibility (antecedent of G4).** Cytological taxonomies (e.g., SIPaKMeD
   morphology classes, Herlev dysplasia grades, the Bethesda System) do not share a label space, and
   naïve cross-mapping between them is clinically unsafe, obstructing both multi-dataset learning and
   honest cross-dataset evaluation.
5. **Imbalance and rare classes (antecedent of G5).** Clinically important high-grade and minority
   morphologies are heavily under-represented in public datasets, so models tend to under-detect
   precisely the categories whose detection is most consequential.
6. **Accuracy–efficiency trade-off (antecedent of G6).** The highest-accuracy models are often too
   computationally heavy for point-of-care or low-resource deployment, and compact alternatives are
   rarely validated head-to-head under an identical, leakage-safe protocol.
7. **Opacity and modality narrowness (antecedent of G7).** Predictions are usually image-only and
   delivered as opaque scores without faithful, quantified explanations or complementary morphometric
   evidence, limiting clinician trust and adoption.

The **reasonable interpretation** advanced throughout this monograph is that these limitations are not
independent defects to be patched one at a time but are *systemically coupled*: they share a common
root in an evaluation culture that optimises and reports a single within-dataset accuracy number. A
framework that addresses them coherently — rather than piecemeal — is therefore both scientifically
warranted and clinically necessary.

## 1.8 Barriers to clinical translation

Even where strong in-principle performance has been demonstrated, translation of cervical-cytology AI
into routine clinical use faces well-documented barriers that the limitations of Section 1.7 directly
exacerbate. **Generalisation risk** is foremost: a regulator or laboratory cannot adopt a system whose
performance on its own slide-preparation pipeline is unknown or unvalidated. **Calibration and
operating-point transfer**: a screening threshold chosen on one population may be badly mis-placed on
another, so that a nominally accurate model fails at the very operating point at which it is deployed.
**Trust and explainability**: cytopathologists are professionally accountable for reported results and
are unlikely to accept opaque recommendations without faithful, inspectable evidence. **Regulatory and
safety requirements**: medical-device frameworks increasingly demand evidence of robustness,
uncertainty quantification, and post-market monitoring, none of which a single accuracy figure
supplies. **Workflow integration and equity**: a tool intended to relieve the LMIC workforce bottleneck
must be deployable on modest hardware and must not silently fail on the populations it is meant to
serve. Each of these barriers maps onto one or more of the seven limitations above, reinforcing the
**interpretation** that clinical translation is gated less by raw accuracy than by *trustworthiness* —
the property that this thesis takes as its organising objective.

## 1.9 The insufficiency of current benchmarks

A benchmark is fit for purpose only if performance on it predicts performance in deployment. The
**central methodological argument** of this monograph is that the prevailing cervical-cytology
benchmarks, as commonly used, fail this criterion for three compounding reasons.

First, **leakage**: random per-cell partitioning of datasets such as SIPaKMeD places cells from a single
field of view into both training and test sets. Because such cells share staining, focus, and debris,
the test estimate is optimistically biased; it measures memorisation of an imaging context rather than
generalisation to new cells. Second, **single-dataset scope**: testing on a held-out partition of the
*same* collection estimates within-distribution performance only and is silent about the cross-domain
performance that deployment requires. Third, **metric narrowness**: a single accuracy figure conceals
calibration failure, conceals below-baseline behaviour at a fixed operating point, and conceals the
structural inability of a morphology-only label space to represent high-grade disease.

The **interpretation** that follows is that a near-ceiling accuracy reported under these conditions is
not evidence of clinical readiness and may be actively misleading. A scientifically adequate benchmark
for this domain must instead be *leakage-safe* (grouped by imaging context with a sealed test),
*multi-domain* (validated across preparation pipelines), *multi-metric and per-domain* (reporting
balanced accuracy, calibration, ranking, and uncertainty separately for each domain, against the
majority-class baseline), and *taxonomy-honest* (making cross-dataset label mappings explicit and
reporting the fraction of cells that cannot be mapped). Constructing and exercising such a benchmark is
the first contribution of this work (C1) and the subject of research question RQ1.

## 1.10 The necessity of trustworthy AI

"Trustworthy AI" is not a slogan but a set of measurable properties without which a screening system
cannot responsibly be deployed. Drawing on the broader discourse on trustworthy and reliable machine
learning, this thesis operationalises trustworthiness for cervical cytology as the conjunction of five
properties, each tied to a limitation of Section 1.7 and a contribution of Chapter 4:
**(i) honest, reproducible evaluation** (no leakage; released splits and seeds);
**(ii) demonstrated cross-domain generalisation**, or, where generalisation is genuinely limited, an
honest characterisation of that limit;
**(iii) calibrated confidence and principled abstention**, so that the system can defer when uncertain;
**(iv) clinically safe handling of incompatible label spaces**, including transparent reporting of what
the model structurally cannot represent; and
**(v) faithful, quantified explanation and complementary evidence**, so that a clinician can
interrogate why a decision was made.

The **argument** is that these properties are not optional refinements to be added once accuracy is
"good enough"; they are constitutive of clinical readiness and must be designed and evaluated as
first-class objectives. This stance — *reliability-first* rather than *accuracy-first* — is the
philosophical core of the research and is elaborated as a methodological commitment in Chapter 5.

## 1.11 Motivation for the present research

The motivation for this research is therefore neither to propose a marginally more accurate architecture
nor to add another result to the Herlev/SIPaKMeD leaderboard. It is to confront, coherently and with
methodological rigour, the gap between the *reported* and the *deployable* performance of
cervical-cytology AI. The work is motivated by three convictions, each grounded in the foregoing
analysis. First, that the dominant evaluation paradigm systematically overstates clinical readiness, and
that establishing an honest, leakage-safe, multi-domain benchmark is a prerequisite for genuine progress
(motivating RQ1). Second, that the failures of generalisation, calibration, taxonomy alignment,
minority detection, deployability, and explainability are *coupled* through this evaluation culture and
must be addressed within a single, internally consistent framework rather than in isolation (motivating
RQ2–RQ7). Third, that the appropriate scientific posture is one of *honest auditing followed by targeted
remediation*: first measure the true gap without flattering the model, then ask, for each identified
failure, whether a specific, evidence-based intervention demonstrably closes it.

## 1.12 Research significance

The significance of this research is both methodological and translational. **Methodologically**, it
contributes a reusable template for trustworthy evaluation of medical-image classifiers — leakage-safe
grouping, sealed testing, per-domain multi-metric reporting against baselines, explicit taxonomy
crosswalks with reported exclusion fractions, and full computational provenance — that is transferable
beyond cervical cytology to other domains where multiple instances share an imaging context and where
label spaces fragment across institutions. **Scientifically**, it provides a mechanistic account of why
high-accuracy cytology models fail across domains, separating label-mapping artefacts from genuine
domain shift and from operating-point miscalibration, and it tests, under one protocol, whether specific
interventions repair specific failures. **Translationally**, it foregrounds the properties — calibration,
abstention, efficiency, and faithful explanation — on which real adoption depends, and it does so for a
disease whose global burden falls disproportionately on settings that stand to benefit most from
reliable automation.

## 1.13 Expected scientific impact

The expected scientific impact, stated as **reasonable expectation** rather than established result, is
fourfold. (1) A demonstration, with quantified evidence, that within-dataset accuracy is a poor proxy
for cross-domain readiness in cervical cytology, and a methodology that exposes the discrepancy.
(2) A diagnostic framework that decomposes cross-dataset failure into identifiable, separately testable
mechanisms — labelling artefact, domain shift with preparation distance, and non-transferable operating
point — thereby converting an opaque "it does not generalise" into specific, falsifiable claims. (3)
Evidence on whether established remedies (e.g., stain normalisation, self-supervised pretraining,
partial-label cross-taxonomy learning, calibrated abstention, conditional synthesis for rare classes)
transfer to this domain and to what extent. (4) A fully provenance-tracked, reproducible artefact —
released splits, seeds, metrics, and figures — that lowers the barrier to honest comparison for
subsequent researchers.

## 1.14 Expected clinical impact

The expected clinical impact, likewise stated as **reasonable expectation**, is to move
cervical-cytology AI closer to responsible deployment by supplying the evidence and the safeguards that
clinical adoption requires. A system that reports calibrated confidence and abstains when uncertain can
function as a safe triage assistant rather than an unaccountable autonomous classifier; one that has
been honestly validated across preparation pipelines gives laboratories a defensible basis for adoption
decisions; and one that is compact and explainable is compatible with point-of-care use and with the
professional accountability of cytopathologists. The work does not claim to deliver a deployment-ready
device; it claims, more modestly and more defensibly, to establish the evaluation and reliability
foundations without which such a device cannot be responsibly built.

## 1.15 Scope and limitations of the study

To preserve scientific honesty, the scope of this study is delimited explicitly. **In scope**: a
single-cell morphological classification problem trained on a public conventional-smear dataset and
evaluated, under a leakage-safe protocol, across multiple external single-cell datasets spanning
preparation modalities; the construction and exercise of a trustworthy evaluation framework; and a set
of targeted interventions evaluated under that framework. **Out of scope**, and acknowledged as
limitations: whole-slide acquisition, scanning, and quality control; prospective clinical validation on
real patient cohorts; histopathological ground truth (the reference standard remains tissue diagnosis,
whereas the model is trained on cytological categories); the molecular (HPV) testing context within
which cytology increasingly operates; and the human-factors and regulatory studies required for a
deployed device. The morphology-only label space further imposes a *structural* limitation — an inability
to represent high-grade disease directly — which this work treats not as a defect to be hidden but as a
property to be measured and reported honestly. These boundaries are stated here so that the claims of
later volumes are read with the appropriate scope.

## 1.16 Organization of the thesis

The monograph is organised into volumes. **Volume I (Foundations)** — the present volume — establishes
the clinical and scientific motivation, the conceptual background, the formal problem formulation, the
research objectives, the methodological philosophy, and the conceptual framework. It deliberately
contains no implementation detail. **Volume II (Dataset Engineering)** will address data curation,
the leakage-safe grouped split, the cross-taxonomy label crosswalk, and exploratory and characterisation
analyses of the data. **Subsequent volumes** will develop the methodology (representation learning,
distillation, calibration, abstention, cross-taxonomy and minority-class learning), the experimental
protocol and results, and the discussion of clinical implications. Within Volume I, Chapter 2 develops
the background and problem formulation; Chapter 3 motivates each research gap against the literature;
Chapter 4 states the objectives, questions, hypotheses, and contributions; Chapter 5 sets out the
research philosophy and design; Chapter 6 presents the conceptual framework; and Chapter 7 summarises
and motivates the transition to Volume II.

## 1.17 Chapter summary

This chapter has established the motivation for the research. Cervical cancer is a preventable,
HPV-driven disease whose burden is large and inequitably distributed, and whose control depends on
screening pathways in which cytology plays a central but workforce-limited role (Sections 1.1–1.3).
Artificial intelligence, and deep learning in particular, has demonstrated strong within-dataset
accuracy on cytological benchmarks, but its role today is best characterised as demonstrated in
principle yet not trustworthy in practice (Sections 1.4–1.6). Seven persistent, systemically coupled
limitations — concerning evaluation integrity, cross-domain generalisation, uncertainty, label
incompatibility, imbalance, efficiency, and opacity — separate reported performance from deployable
performance (Section 1.7), and these limitations are precisely the barriers to clinical translation
(Section 1.8). The prevailing benchmarks are, as commonly used, insufficient to estimate deployable
performance (Section 1.9), motivating a reliability-first conception of trustworthy AI (Section 1.10)
and a research programme of honest auditing followed by targeted remediation (Sections 1.11–1.14),
within a clearly delimited scope (Section 1.15).

## 1.18 Remaining challenges

Several challenges remain open at the close of this introductory chapter and are taken up in the
chapters that follow. The clinical and cytological concepts invoked here — grading systems, preparation
modalities, the nature of domain shift, and the specific form of label incompatibility between cervical
datasets — require rigorous definition before a problem formulation can be stated; this is the task of
Chapter 2. The seven limitations require substantiation against the strongest existing solutions, and
their open status requires justification; this is the task of Chapter 3. And the loose notion of a
"reliability-first framework" requires translation into a precise set of objectives, falsifiable
hypotheses, and a defensible methodological philosophy; this occupies Chapters 4 and 5.

## 1.19 Transition to Chapter 2

Having motivated *why* trustworthy cervical-cytology AI is needed and *what* separates current research
from clinical readiness, the monograph now turns to the conceptual foundations required to formulate the
problem precisely. Chapter 2 develops the fundamentals of cervical cytology and cell morphology, the
grading systems and their incompatibilities, the mechanisms of domain shift between preparation
modalities, and the concepts of leakage, calibration, reliability, and explainability that recur
throughout the work, before consolidating them into a formal problem formulation that the remainder of
the thesis addresses.

*(All in-text citations are collected in the consolidated **References** section at the end of
Volume I; a corresponding `references.bib` accompanies the document.)*

---

# Chapter 2 — Background and Problem Formulation

This chapter establishes the conceptual foundation required to state the research problem precisely. It
proceeds from the biology and clinical practice of cervical cytology (Sections 2.1–2.4), through the
taxonomic systems and their incompatibilities (Sections 2.5–2.7), to the technical phenomena that govern
machine-learning behaviour in this domain — preparation-induced domain shift, data leakage, bias,
imbalance, and annotation uncertainty (Sections 2.8–2.12) — and finally to the reliability concepts that
the framework treats as first-class objectives: clinical risk, explainability, calibration, reliability,
and human–AI collaboration (Sections 2.13–2.18). Section 2.19 consolidates these into a formal problem
formulation. Consistent with Volume I's remit, no model architecture, loss, or training procedure is
described.

## 2.1 Fundamentals of cervical cytology

Cervical cytology is the microscopic examination of exfoliated cells collected from the uterine cervix,
performed to detect pre-invasive and invasive squamous and glandular lesions. The biological rationale
rests on the natural history of cervical carcinogenesis: persistent infection with high-risk HPV induces
progressive epithelial abnormality — from mild dysplasia through severe dysplasia and carcinoma in situ
to invasion — over a typically protracted interval, during which abnormal cells shed from the
transformation zone can be sampled and recognised (Section 1.1). The transformation zone, where columnar
epithelium meets squamous epithelium and undergoes metaplasia, is both the principal site of
HPV-associated neoplasia and the target of specimen collection. Cytological assessment is thus a
*screening* modality that stratifies risk and directs further investigation; it is **established** that
its reference standard is histopathology of tissue obtained at colposcopy, a distinction with direct
methodological consequences for label semantics (Section 2.5) and for the framing of the learning
problem (Section 2.19).

## 2.2 Cell morphology

The cytological diagnosis of dysplasia rests on a constellation of nuclear and cytoplasmic morphological
features that an automated system must, implicitly or explicitly, learn to recognise. The most
diagnostically salient is the **nuclear-to-cytoplasmic (N/C) ratio**: progressive dysplasia is
characterised by nuclear enlargement disproportionate to cytoplasmic area, so that the N/C ratio
increases with grade. Additional nuclear criteria include **hyperchromasia** (increased and abnormally
distributed chromatin), **chromatin texture** (coarsening and irregular clumping), **nuclear membrane
irregularity**, **anisonucleosis** (variation in nuclear size), and **multinucleation**. Cytoplasmic
features include **keratinisation** and the **perinuclear halo** (koilocytosis) characteristic of
productive HPV infection. Normal squamous cells — superficial and intermediate — present small, regular
nuclei within abundant cytoplasm; parabasal and metaplastic cells present larger nuclei within scant
cytoplasm without the chromatin abnormalities of dysplasia, a benign-but-confusable appearance that is a
recurring source of classifier error. It is an **established** principle of cytopathology that these
features are continuous and partially overlapping across categories rather than discretely separable,
which both explains the inherent inter-observer variability of cytology (Section 2.12) and bounds the
achievable separability of any classifier operating on the same images.

## 2.3 Cytological grading systems

Cytological findings are reported using standardised nomenclatures that map the continuous morphological
spectrum onto discrete reporting categories so as to direct management. Historically several systems
have been used — the original Papanicolaou class system (I–V), the dysplasia/carcinoma-in-situ
nomenclature, and the cervical intraepithelial neoplasia (CIN) grading borrowed from histopathology —
before convergence, in cytology, on the Bethesda System (Section 2.4). For the purposes of this thesis
the essential point is **conceptual**: a grading system is a *clinically motivated quantisation* of an
underlying continuum, and different systems quantise that continuum differently, at different
granularities, and with category boundaries drawn for different management purposes. Consequently two
datasets labelled under two systems do not, in general, share a label space — a fact that becomes the
formal label-incompatibility problem of Section 2.7.

## 2.4 The Bethesda System

The Bethesda System for Reporting Cervical Cytology (Nayar & Wilbur, 2015) is the predominant
contemporary reporting framework. It first classifies a specimen's adequacy and then reports squamous
and glandular findings. The squamous interpretive categories most relevant here are: **NILM** (negative
for intraepithelial lesion or malignancy); **ASC-US** (atypical squamous cells of undetermined
significance) and **ASC-H** (atypical squamous cells, cannot exclude high-grade lesion); **LSIL**
(low-grade squamous intraepithelial lesion, encompassing HPV cytopathic effect/koilocytosis and CIN1);
**HSIL** (high-grade squamous intraepithelial lesion, encompassing CIN2–CIN3/carcinoma in situ); and
**squamous cell carcinoma**. Glandular abnormalities (e.g., AGC, atypical glandular cells) form a
separate axis. Two features of this system are decisive for the present work. First, it is a
*risk-stratification* taxonomy: ASC-US, for example, denotes diagnostic uncertainty rather than a
specific morphology, and has no clean correspondence to any single morphological cell class. Second, its
distinction between LSIL and HSIL — the clinically pivotal boundary — is defined by criteria (degree of
nuclear abnormality, maturation) that a morphology-only single-cell taxonomy may be unable to represent.
These properties underpin both the label-incompatibility problem (Section 2.7) and the structural
"high-grade blind spot" acknowledged as a scope limitation in Section 1.15.

## 2.5 The SIPaKMeD taxonomy

The SIPaKMeD dataset (Plissiti et al., 2018) provides isolated single-cell images annotated under a
**five-class morphological** taxonomy: **superficial–intermediate** and **parabasal** (normal
squamous), **metaplastic** (benign), and **koilocytotic** and **dyskeratotic** (abnormal, reflecting
HPV cytopathic effect and abnormal keratinisation respectively). Critically, this is a taxonomy of
*cell appearance*, not of *cytological grade*: its categories describe what a cell looks like, not the
Bethesda interpretive category a cytopathologist would assign to the slide. A model trained on SIPaKMeD
therefore learns a mapping from images to morphological appearance classes, and any clinical
interpretation of its outputs requires an explicit, defensible mapping from appearance to a
risk-stratification tier. The dataset also provides, for each cell, expert nuclear and cytoplasmic
boundary annotations, which support ground-truth morphometric characterisation and faithfulness
evaluation (developed in later volumes) but are themselves a curation artefact not available at
deployment.

## 2.6 The Herlev taxonomy

The Herlev dataset (Jantzen et al., 2005) provides single-cell images under a **seven-class** taxonomy
spanning normal (superficial squamous, intermediate squamous, columnar) and dysplastic categories (mild,
moderate, severe dysplasia, and carcinoma in situ). Unlike SIPaKMeD's appearance taxonomy, Herlev's
categories are dysplasia *grades*, aligned with the histological CIN spectrum, and include a glandular
(columnar) class with no squamous-morphology counterpart. Herlev thus differs from SIPaKMeD in
granularity (seven vs five classes), in semantics (grade vs appearance), and in scope (inclusion of a
glandular category). These differences are not incidental; they are precisely the kind of taxonomic
divergence that frustrates naïve cross-dataset transfer and that the framework's label crosswalk
(Chapter 4) is designed to handle honestly.

## 2.7 The label-incompatibility problem

The foregoing sections motivate a problem that this thesis treats as fundamental: cervical-cytology
datasets are annotated under **incompatible label spaces**, and there exists no clinically safe,
information-preserving bijection between them. The incompatibility is of several kinds simultaneously:
(i) *granularity* mismatch (five vs seven vs Bethesda categories); (ii) *semantic* mismatch (appearance
vs grade vs risk-stratification); (iii) *scope* mismatch (glandular and indeterminate categories present
in one taxonomy and absent in another); and (iv) *projection* hazard (a single source class may map to
several target categories — e.g., a koilocytotic appearance corresponds to LSIL, but LSIL also subsumes
other appearances — so a forced one-to-one mapping is many-to-one in one direction and one-to-many,
hence undefined, in the other). The **reasonable interpretation**, developed formally in Section 2.19
and operationalised in Chapter 4, is that the *only* level at which heterogeneous cervical taxonomies can
be aligned deterministically and many-to-one — without inventing label information — is a coarse,
clinically meaningful tier such as a binary normal-versus-abnormal decision, with categories that have
no squamous-morphology home (glandular, indeterminate, high-grade beyond the model's representational
capacity) explicitly *excluded* and the excluded fraction reported as an honesty metric. Treating the
crosswalk as an explicit, documented object — rather than an informal relabelling — is what allows a
cross-dataset performance change to be attributed to the model rather than to the labels.

## 2.8 Conventional smear versus liquid-based cytology

As introduced in Section 1.3, the two dominant preparation modalities — conventional smear and
liquid-based cytology (LBC) — differ systematically. Conventional smears exhibit higher and more uneven
cell density, frequent overlapping and three-dimensional cell groups, variable background (blood,
inflammatory cells, debris), and air-drying artefact. LBC produces a thinner, more uniform monolayer on
a defined area, with cleaner background and more consistent fixation, at the cost of altered cell
architecture and presentation. From a machine-learning standpoint these are not cosmetic differences:
they change the marginal distribution of pixel statistics, the spatial density and separability of
cells, the segmentation amenability of nuclei and cytoplasm, and the staining response. They constitute,
therefore, a concrete and clinically grounded instance of **domain shift** (Section 2.9), and the
*distance* between a training preparation and a target preparation is a candidate explanatory variable
for cross-dataset performance degradation — a hypothesis the framework is designed to test.

## 2.9 Sources of domain shift

**Domain shift** (or dataset shift) denotes any difference between the joint distribution of inputs and
labels at training time and at deployment (Quiñonero-Candela et al., 2009; Moreno-Torres et al., 2012).
In cervical cytology the principal sources are: (i) **preparation shift** (smear vs LBC; Section 2.8);
(ii) **stain and scanner shift** — variation in Papanicolaou staining protocols, reagent lots,
microscope optics, and digitisation pipelines, analogous to the well-characterised stain variation in
histopathology (Reinhard et al., 2001; Macenko et al., 2009; Tellez et al., 2019; Stacke et al., 2021);
(iii) **segmentation/cropping shift** — differences in how single-cell images are produced, from manual
expert cropping to automated detection; (iv) **population and prevalence shift** — differences in patient
demographics, HPV-genotype distribution, and disease prevalence that alter the class prior; and (v)
**annotation shift** — differences in labelling taxonomy and criteria (Sections 2.5–2.7, 2.12). A
critical and **evidence-supported** concern is that high-capacity networks may achieve within-dataset
accuracy by exploiting *preparation-specific* cues that co-vary with the label in the training set but
do not generalise — an instance of **shortcut learning** (Geirhos et al., 2020). Distinguishing genuine
morphological competence from such shortcuts is a core diagnostic aim of the framework.

## 2.10 Data leakage in medical imaging

**Data leakage** denotes any flow of information from the test set into model selection or training that
inflates the estimated performance relative to deployment (Kapoor & Narayanan, 2023). In medical imaging
the most pernicious form is **grouping leakage**: multiple images that share a latent acquisition
context — the same patient, slide, or field of view — are split across training and test partitions, so
that the test set is not statistically independent of the training set. Because such images share
staining, focus, and incidental artefacts, a model can attain a high test score by recognising the
shared context rather than the diagnostic signal. Cervical cytology is acutely susceptible: a single
cluster image yields many cropped cells, and a random per-cell split routinely co-locates sibling cells
across partitions. The **established** methodological remedy is to partition at the level of the latent
group (patient/slide/field), with a sealed test set untouched during model development. The framework
adopts grouped partitioning with a sealed test as a non-negotiable evaluation commitment (Chapter 4) and
quantifies the leakage-induced inflation by comparison with a deliberately leaky control.

## 2.11 Dataset bias

Beyond domain shift and leakage, public cytology datasets carry **selection and curation biases** that
constrain what can be learned and validated. Curated single-cell benchmarks are assembled by expert
selection of representative, often unambiguous, cells; they thus under-represent the borderline,
overlapping, and artefact-laden cells that dominate real slides. Class composition reflects curation
convenience rather than clinical prevalence. Imaging conditions are typically homogeneous within a
dataset and heterogeneous across datasets. The **interpretation** is that a model and a metric estimated
on such a dataset describe performance on *curated, representative cells from one acquisition pipeline* —
a quantity that may substantially overstate performance on consecutive real-world material. Honest
reporting therefore requires that the curation context of every dataset be stated and that claims be
scoped accordingly (a practice this monograph follows for its own datasets in Volume II).

## 2.12 Class imbalance and annotation uncertainty

Two further data properties shape the problem. **Class imbalance**: clinically important categories —
high-grade lesions, glandular abnormalities, and rare morphologies — are under-represented relative to
normal and low-grade categories, so that a classifier minimising average error is biased against
detecting precisely the cells whose detection matters most. **Annotation uncertainty**: cytological
labels are themselves noisy. Inter-observer agreement in cervical cytology is known to be limited,
particularly for borderline categories such as ASC-US, as documented by large studies of interpretive
reproducibility (e.g., Stoler & Schiffman, 2001). Single-cell labels inherit and may amplify this
uncertainty, because a slide-level interpretation does not uniquely determine the category of each
constituent cell. The **consequences** are twofold and must be reflected in evaluation: first, label
noise imposes an irreducible ceiling on achievable accuracy, so that near-ceiling reported numbers
warrant scrutiny; second, the appropriate performance metrics are those robust to imbalance (balanced
accuracy, per-class recall, ranking metrics) reported against the majority-class baseline, rather than
raw accuracy alone.

## 2.13 The clinical risk of misclassification

Misclassification in cervical screening carries **asymmetric clinical cost**. A false negative — failing
to flag an abnormal cell — risks delayed diagnosis of a potentially progressive lesion and is the more
serious error in a screening context. A false positive risks unnecessary anxiety, colposcopy, and
over-treatment. This asymmetry has three implications adopted throughout the framework. First, the
operating point of any deployed classifier must be chosen with respect to clinical cost, not default
thresholds, and must be *transferable* to the deployment population — a requirement the framework shows
is frequently violated under domain shift. Second, sensitivity to abnormal categories, and per-class
recall on high-grade and minority categories, are first-order metrics. Third, a screening assistant
should be able to **abstain** when uncertain and defer to a human, converting a potentially confident
error into a flagged uncertainty — motivating the calibration and abstention machinery of Sections 2.16
and 2.17.

## 2.14 Explainability in medical AI

For a screening assistant to be adopted, its outputs must be *interrogable* by an accountable clinician.
**Explainability** in medical AI spans post-hoc attribution methods that localise the image evidence for
a decision (e.g., class-activation mapping; Selvaraju et al., 2017), complementary interpretable evidence
(e.g., morphometric measurements that a cytopathologist already reasons about, such as the N/C ratio),
and intrinsically interpretable models. The literature also issues a cautionary, **evidence-supported**
warning: post-hoc explanations can be unfaithful — appearing plausible while not reflecting the model's
actual computation — and should not be mistaken for guarantees of correct reasoning (Rudin, 2019;
Ghassemi et al., 2021). The framework therefore treats explanation as something to be *measured*
(faithfulness quantified against ground-truth structures and perturbation tests) rather than merely
displayed, and pairs image attribution with complementary morphometric evidence whose clinical meaning
is established.

## 2.15 Calibration

A classifier is **calibrated** if its reported confidence matches its empirical accuracy: among
predictions made with confidence p, a fraction p are correct. Modern deep networks are typically
**miscalibrated and over-confident** (Guo et al., 2017), and calibration is especially load-bearing in
screening, where a downstream decision threshold is only meaningful if the underlying probabilities are
honest. Calibration is conventionally summarised by the expected calibration error and visualised by
reliability diagrams, and can be improved post hoc (e.g., by temperature scaling). A central, and
under-appreciated, concern that this thesis foregrounds is that **calibration is domain-local**:
calibration established on one distribution need not, and frequently does not, transfer to another, so
that a model with excellent in-distribution calibration may be badly miscalibrated — and its fixed
operating point badly mis-placed — on a new domain. Measuring calibration *per domain*, rather than once
in-distribution, is therefore a methodological commitment of the framework.

## 2.16 Reliability and uncertainty quantification

**Reliability** generalises calibration to the broader requirement that a system's stated confidence be
trustworthy enough to act on, including the ability to recognise inputs on which it should not commit to
a prediction. Relevant tools include **uncertainty quantification** (distinguishing aleatoric from
epistemic uncertainty), **out-of-distribution detection** (recognising inputs unlike the training
distribution), and **conformal prediction**, which constructs prediction *sets* with a finite-sample,
distribution-free coverage guarantee under exchangeability (Vovk, Gammerman & Shafer, 2005; Angelopoulos
& Bates, 2021). The **interpretation** adopted here is that reliability, not raw accuracy, is the
property that gates safe deployment: a model that is less accurate but knows when it is uncertain, and
abstains accordingly, is clinically preferable to a more accurate model that fails silently. This
"reliability-first" stance, introduced in Section 1.10, is given methodological form in Chapter 5.

## 2.17 Human–AI collaboration and abstention

The deployment model envisaged in this work is **assistive**, not autonomous: the system functions
within a human–AI workflow in which it triages, prioritises, or second-reads, and defers genuinely
uncertain cases to a cytopathologist (Topol, 2019). This framing makes **abstention** a first-class
capability rather than a fallback: a calibrated confidence and a coverage-controlled abstention rule
together allow the system to operate at a high-precision regime on the cases it does commit to, while
routing the remainder to human review. Selective prediction — trading coverage for selective accuracy —
is, under this framing, the natural evaluation lens, and is preferred to a single accuracy figure
because it directly characterises the assistant's behaviour at the human–AI interface. The framework
therefore evaluates abstention behaviour explicitly and treats the conformal/OOD machinery of Section
2.16 as the means to realise it.

## 2.18 Synthesis of the conceptual background

Sections 2.1–2.17 establish a coherent picture. Cervical cytology is a screening task on a continuous
morphological spectrum, quantised differently by incompatible taxonomies, acquired under heterogeneous
preparation pipelines, and labelled with non-trivial uncertainty. Machine-learning systems trained on
curated single-cell benchmarks are exposed to leakage, bias, imbalance, and preparation-induced domain
shift, and are prone to shortcut learning, miscalibration, and unfaithful explanation. The clinical
context imposes asymmetric costs and demands calibrated, abstaining, interpretable assistance within a
human–AI workflow. These facts are not a list of independent concerns but the components of a single
problem: *to learn, evaluate, and deploy a cervical-cytology classifier whose reported performance is an
honest and transferable estimate of its clinical utility.* The remainder of the monograph is organised
around that problem, which Section 2.19 now states formally.

## 2.19 Formal problem formulation

The problem is formulated at the level of learning and evaluation, without reference to any particular
model class. (This is a conceptual/mathematical formulation, not an implementation.)

**2.19.1 Objects.** Let an input be a single-cell image x ∈ 𝒳. Let the *source taxonomy* be a finite set
of morphological appearance classes 𝒴_S (for the training dataset). Let each *target taxonomy* 𝒴_T^{(d)}
be the native label set of an external evaluation domain d ∈ 𝒟. A predictor is a function f: 𝒳 → Δ(𝒴_S),
mapping an image to a distribution over the source classes (Δ denotes the probability simplex).

**2.19.2 Distributions and domain shift.** Let P_S(x, y) denote the joint distribution of the source
domain and P_d(x, y) that of target domain d. Domain shift is the condition P_d ≠ P_S for some d ∈ 𝒟
(Section 2.9). The deployment quantity of interest is the performance of f under P_d for target domains,
not under a held-out partition of P_S.

**2.19.3 Evaluation tiers and the crosswalk.** Because 𝒴_S and 𝒴_T^{(d)} are incompatible (Section 2.7),
performance cannot be compared in either native space directly. Define a coarse, clinically meaningful
*evaluation tier* 𝒯 (e.g., the binary tier {normal, abnormal}) and two deterministic, many-to-one
crosswalks: g_S: 𝒴_S → 𝒯 ∪ {⊥} on the model side and, for each domain, h_d: 𝒴_T^{(d)} → 𝒯 ∪ {⊥} on the
label side, where ⊥ denotes *exclusion* of native categories that have no defensible image in 𝒯 (e.g.,
glandular, indeterminate, or high-grade categories beyond the representational capacity of 𝒴_S). The
model's tier decision is obtained by aggregating the appropriate source-class mass and thresholding at a
pre-registered operating point τ. The *excluded fraction* — the proportion of cells with h_d(y) = ⊥ —
is reported as an honesty metric, since silently discarding unmappable cells would inflate the tier
score.

**2.19.4 Evaluation desiderata.** A scientifically adequate evaluation of f is required to estimate, for
each domain d, a vector of quantities rather than a scalar: (i) threshold-free discrimination (e.g., the
area under the ROC curve of the tier score), which is invariant to the operating point; (ii)
operating-point performance at the pre-registered τ, reported against the majority-class baseline of
domain d; (iii) calibration of the tier score under P_d (Section 2.15); and (iv) reliability/abstention
behaviour under P_d (Sections 2.16–2.17). The internal estimate under P_S must itself be leakage-safe
(Section 2.10): partitions are formed over the latent grouping variable (field/slide), with a sealed
test set, so that the internal number is an unbiased estimate of within-domain generalisation.

**2.19.5 The scientific problem.** Given a predictor f trained on the source domain, the problem has two
parts. *Diagnostic*: characterise and decompose the discrepancy between the internal estimate under P_S
and the performance under {P_d}, attributing it to its sources — labelling/crosswalk artefact (testable
by whether the tier score is correctly oriented, i.e., ranks abnormal above normal, in every domain),
genuine domain shift (testable against preparation distance), and non-transferable operating
point/calibration (testable by the gap between threshold-free and operating-point performance, and by
per-domain recalibration). *Constructive*: determine whether specific, evidence-based interventions —
applied within the identical leakage-safe, multi-domain, multi-metric protocol — measurably reduce the
diagnosed failures, while preserving calibration and abstention guarantees. The seven research questions
of Chapter 4 instantiate these two parts; the contributions C1–C6 are their intended resolutions.

## 2.20 Chapter summary

This chapter has built the conceptual foundation for the thesis. It defined the biology and clinical
practice of cervical cytology and the morphological features on which diagnosis rests (Sections 2.1–2.2);
described the grading systems and the Bethesda System, and the SIPaKMeD and Herlev taxonomies, exposing
their incompatibility as a problem of granularity, semantics, scope, and projection hazard (Sections
2.3–2.7); characterised the technical phenomena that govern learning in this domain — preparation-induced
domain shift, leakage, bias, imbalance, and annotation uncertainty (Sections 2.8–2.12); and articulated
the reliability concepts the framework treats as first-class — clinical risk asymmetry, explainability
and its faithfulness caveat, calibration and its domain-locality, reliability and abstention, and the
human–AI collaborative setting (Sections 2.13–2.18). Section 2.19 consolidated these into a formal
problem formulation with a diagnostic and a constructive part, defined over incompatible taxonomies, a
deterministic crosswalk with reported exclusions, leakage-safe internal estimation, and per-domain
multi-metric evaluation.

## 2.21 Remaining challenges

The formulation of Section 2.19 raises challenges addressed subsequently. The seven sources of failure
it references must each be shown, against the strongest existing solutions, to be genuinely open rather
than already solved; this is the burden of Chapter 3. The diagnostic and constructive problems must be
turned into precise, falsifiable research questions and hypotheses with a pre-registered protocol; this
is Chapter 4. And the methodological commitments invoked here — leakage-safe evaluation, per-domain
reporting, reliability-first design, honest crosswalks — must be justified as a coherent research
philosophy rather than asserted; this is Chapter 5.

## 2.22 Transition to Chapter 3

With the problem now precisely formulated, the monograph turns from *what* the problem is to *why it
remains unsolved*. Chapter 3 examines each of the seven research gaps in turn, explaining why the problem
exists, summarising and critically appraising the strongest existing solutions, justifying why the gap
remains open, and articulating its scientific and clinical stakes, thereby motivating the research
objectives of Chapter 4 without yet describing how they will be met.

---

# Chapter 3 — Research Motivation

This chapter justifies the research programme by examining each of the seven research gaps (G1–G7) in
turn. For every gap, the analysis follows a fixed structure: (1) why the problem exists; (2) the
strongest existing solutions; (3) their limitations; (4) why the gap remains open; (5) why solving it
matters scientifically; (6) why it matters clinically; and (7) the connection to the corresponding
research question (RQ1–RQ7). The chapter establishes that each gap is genuinely open against the state
of the art, and that the gaps are coupled, motivating a single coherent framework rather than seven
isolated fixes. No implementation is described.

## 3.1 G1 — Non-reproducible, leakage-prone evaluation

**3.1.1 Why the problem exists.** Curated single-cell datasets are produced by cropping many cells from
a smaller number of cluster images or slides; the dominant evaluation convention partitions these cells
at random. Because sibling cells from one acquisition context share staining, focus, and artefacts,
random partitioning induces grouping leakage (Section 2.10). Compounding this, splits, seeds, and code
are seldom released, so reported numbers cannot be independently reproduced.

**3.1.2 Strongest existing solutions.** The broader machine-learning-for-science community has
articulated the problem clearly and proposed remedies: explicit leakage taxonomies and checklists
(Kapoor & Narayanan, 2023), group-aware cross-validation, and reproducibility and reporting standards
for medical imaging (Pineau et al., 2021; Varoquaux & Cheplygina, 2022). Within medical imaging,
patient-level splitting is an established recommendation.

**3.1.3 Limitations.** These remedies are methodological recommendations rather than properties of the
cervical-cytology benchmarks themselves; the field's *de facto* benchmarks continue to be used with
per-cell random splits, and quantification of the resulting inflation — how many points of accuracy are
attributable to leakage in this specific domain — is rarely reported. A recommendation that is not
instantiated as a released, leakage-safe benchmark exerts little corrective force.

**3.1.4 Why the gap remains open.** There is, to the author's knowledge, no widely adopted, released,
leakage-safe benchmark for single-cell cervical cytology with sealed test partitions and published seeds,
nor a published quantification of the leakage tax against a deliberately leaky control. The gap is open
not for want of awareness but for want of an instantiated artefact.

**3.1.5 Scientific stakes.** Without leakage-safe evaluation, the entire empirical edifice of the field
rests on biased estimates; comparisons between methods may reflect differential exploitation of leakage
rather than genuine merit. Closing the gap is a precondition for any trustworthy claim in the domain.

**3.1.6 Clinical stakes.** A regulator or laboratory cannot act on inflated estimates; an overstated
sensitivity translates directly into unquantified false-negative risk in deployment. Honest internal
estimation is the floor beneath every clinical claim.

**3.1.7 Connection to RQ1.** This gap motivates **RQ1**: whether a cluster-grouped, stratified split
with a sealed test and released seeds yields a reproducible benchmark whose internal score reflects
genuine within-dataset generalisation, and what the leakage tax is.

## 3.2 G2 — Single-dataset training and unsolved cross-domain generalisation

**3.2.1 Why the problem exists.** Datasets are acquired under one preparation/staining/segmentation
pipeline; models trained and tested within one dataset never encounter the preparation shift they will
face in deployment, and high-capacity networks readily fit preparation-specific shortcuts (Sections 2.9,
2.8; Geirhos et al., 2020).

**3.2.2 Strongest existing solutions.** Mainstream remedies include stain normalisation and stain
augmentation to reduce appearance shift (Reinhard et al., 2001; Macenko et al., 2009; Tellez et al.,
2019); domain-generalisation algorithms surveyed in the general literature (Zhou et al., 2022; Wang et
al., 2022); and self-supervised pretraining to learn more transferable representations (Chen et al.,
2020; Azizi et al., 2021).

**3.2.3 Limitations.** These techniques are largely validated in histopathology or natural images;
their transfer to single-cell cervical cytology, and specifically across the smear↔LBC↔auto-segmented
preparation axis, is sparsely and inconsistently evaluated. Crucially, cross-dataset evaluation is
frequently absent or, where present, confounded by undocumented label remapping (G4), so that the very
measurement needed to assess generalisation is unreliable.

**3.2.4 Why the gap remains open.** There is no established, leakage-safe, multi-preparation benchmark on
which generalisation techniques have been compared under one protocol; consequently the field lacks a
clear, reproducible answer to whether, and by how much, any technique narrows the internal→external gap
for cervical cytology.

**3.2.5 Scientific stakes.** Generalisation is the property that distinguishes a learned morphological
competence from a memorised acquisition signature. Characterising and reducing the gap, and ordering it
by preparation distance, would convert an opaque failure into a quantitative, explanatory account.

**3.2.6 Clinical stakes.** Cross-domain robustness is the single most decision-relevant property for
adoption: a laboratory must know how a model performs on *its* preparation pipeline. Unmeasured
generalisation is unmanaged clinical risk.

**3.2.7 Connection to RQ2.** This gap motivates **RQ2**: whether stain normalisation, self-supervised
pretraining, and domain-aware training produce a classifier that generalises across preparation domains,
and where (by preparation distance) generalisation succeeds or fails.

## 3.3 G3 — Overconfidence and the absence of uncertainty or abstention

**3.3.1 Why the problem exists.** Deep networks trained by maximum-likelihood are systematically
over-confident (Guo et al., 2017), and cytology systems typically emit point predictions without
calibrated confidence or any mechanism to abstain. Moreover calibration is domain-local (Section 2.15),
so even an in-distribution-calibrated model may be badly mis-placed at deployment.

**3.3.2 Strongest existing solutions.** Post-hoc calibration (temperature scaling; Guo et al., 2017),
ensemble and Bayesian uncertainty (Ovadia et al., 2019), out-of-distribution detection (Hendrycks &
Gimpel, 2017; Liu et al., 2020), and conformal prediction with distribution-free coverage guarantees
(Vovk et al., 2005; Angelopoulos & Bates, 2021) constitute the strongest general tools.

**3.3.3 Limitations.** These are seldom integrated into cervical-cytology systems, still less evaluated
*across domains*; the domain-locality of calibration — that the operating point fails precisely where it
is most needed — is rarely measured. Reported calibration, when present, is almost always
in-distribution only.

**3.3.4 Why the gap remains open.** There is no established cervical-cytology evaluation that reports
per-domain calibration and coverage-controlled abstention, nor a demonstration that abstention recovers
safe operation under shift. The machinery exists; its domain-aware application to this task does not.

**3.3.5 Scientific stakes.** Reliability under shift is a more stringent and more informative property
than accuracy; establishing whether calibration and abstention transfer is a substantive scientific
question with implications beyond cytology.

**3.3.6 Clinical stakes.** A screening assistant that cannot say "I am unsure" is unsafe; calibrated
abstention is what makes assistive, human-in-the-loop deployment defensible (Section 2.17).

**3.3.7 Connection to RQ3.** This gap motivates **RQ3**: whether per-domain temperature scaling,
conformal prediction, and an out-of-distribution gate deliver trustworthy confidence and safe abstention
in- and cross-distribution.

## 3.4 G4 — Fragmented, incompatible label taxonomies

**3.4.1 Why the problem exists.** Cervical datasets are labelled under incompatible taxonomies differing
in granularity, semantics, and scope (Sections 2.5–2.7); there is no information-preserving bijection
between them, and naïve one-to-one remapping invents label information.

**3.4.2 Strongest existing solutions.** The general literature offers label-space-unification strategies,
hierarchical label structures, and partial-label and weak-supervision learning for training under
incomplete or set-valued labels (e.g., Cour et al., 2011). Open-set and superclass formulations address
categories absent from the training space.

**3.4.3 Limitations.** Cross-dataset cytology comparisons frequently perform *informal* relabelling
whose assumptions are neither stated nor validated, conflating label-mapping artefacts with genuine
performance differences; and partial-label methods are seldom applied to reconcile cervical taxonomies.
The structural inability of a morphology-only space to represent high-grade disease is rarely acknowledged
as a reportable limitation.

**3.4.4 Why the gap remains open.** There is no established, deterministic, documented crosswalk for
cervical cytology that maps both predictions and native labels many-to-one to a clinically meaningful
tier with an explicit exclusion set and a reported excluded fraction, and no standard practice of
training across taxonomies via partial labels.

**3.4.5 Scientific stakes.** Only an explicit crosswalk permits attribution of a cross-dataset change to
the model rather than the labels — a prerequisite for the diagnostic problem of Section 2.19. Safe
cross-taxonomy learning could, in principle, unlock multi-dataset training.

**3.4.6 Clinical stakes.** Mis-mapping labels can systematically mis-state sensitivity to high-grade
disease; honest reporting of what a model cannot represent is itself a safety requirement.

**3.4.7 Connection to RQ4.** This gap motivates **RQ4**: whether a deterministic many-to-one crosswalk
and partial-label learning enable clinically safe cross-taxonomy evaluation and multi-dataset training
without the dangerous one-to-many projection.

## 3.5 G5 — Severe imbalance and missed rare or high-grade classes

**3.5.1 Why the problem exists.** Curation and clinical prevalence both under-represent high-grade,
glandular, and rare morphologies (Sections 2.11–2.12); average-error objectives are thereby biased
against the clinically critical minority.

**3.5.2 Strongest existing solutions.** Class-imbalance remedies are well studied: re-weighting and
re-sampling, class-balanced and focal losses (Buda et al., 2018; Cui et al., 2019; Lin et al., 2017),
and synthetic minority augmentation, including generative-adversarial synthesis of cells (Goodfellow et
al., 2014; Frid-Adar et al., 2018), with cytology-specific conditional synthesis recently proposed
(e.g., CellGAN — verify exact reference).

**3.5.3 Limitations.** Whether minority-class synthesis improves recall *without harming calibration* is
rarely evaluated jointly; synthesis can shift the operating point and inflate confidence. Imbalance
interventions are seldom assessed under a leakage-safe, calibration-aware protocol.

**3.5.4 Why the gap remains open.** There is no established evaluation in cervical cytology that reports
minority-class recall *and* calibration jointly under synthesis-based augmentation, leaving the
risk–benefit trade-off uncharacterised.

**3.5.5 Scientific stakes.** Understanding when synthetic augmentation helps, and at what calibration
cost, is a general question about the interaction of data augmentation and uncertainty.

**3.5.6 Clinical stakes.** Under-detection of high-grade lesions is the most consequential failure mode
of a screening tool; improving minority recall safely is of direct clinical value.

**3.5.7 Connection to RQ5.** This gap motivates **RQ5**: whether conditional synthesis of
under-represented classes improves minority recall while preserving calibration.

## 3.6 G6 — The accuracy–efficiency trade-off and point-of-care deployment

**3.6.1 Why the problem exists.** The highest-accuracy models are computationally heavy, while the
settings with the greatest need (LMICs; point-of-care) have the least compute (Section 1.1).

**3.6.2 Strongest existing solutions.** Efficient architectures (Tan & Le, 2019; Howard et al., 2017),
knowledge distillation (Hinton et al., 2015), pruning and quantisation provide compact models; some
cervical-cytology studies report lightweight networks.

**3.6.3 Limitations.** Compact and heavy models are rarely compared *head-to-head under one
leakage-safe protocol*, with efficiency (parameters, FLOPs, latency) reported alongside accuracy and
calibration; nor are specialised cervical architectures benchmarked against general backbones under
identical evaluation. Efficiency claims are therefore difficult to compare.

**3.6.4 Why the gap remains open.** There is no established apples-to-apples comparison of backbones and
published specialised architectures for cervical cytology under one leakage-safe, multi-metric protocol
with efficiency profiling.

**3.6.5 Scientific stakes.** Whether the failure modes of the field are architecture-specific or
protocol-level is testable only under such a comparison; distillation's effect on generalisation (as
opposed to in-distribution accuracy) is itself an open question.

**3.6.6 Clinical stakes.** Deployability on modest hardware is a precondition for impact in exactly the
settings where cervical-cancer burden is highest.

**3.6.7 Connection to RQ6.** This gap motivates **RQ6**: whether knowledge distillation yields a compact
model that matches larger backbones and published specialised architectures under one protocol.

## 3.7 G7 — Black-box, image-only predictions and limited clinical trust

**3.7.1 Why the problem exists.** Predictions are usually delivered as opaque image-only scores; the
features cytopathologists reason about (e.g., N/C ratio) are not surfaced, and post-hoc explanations may
be unfaithful (Section 2.14; Rudin, 2019; Ghassemi et al., 2021).

**3.7.2 Strongest existing solutions.** Attribution methods (Selvaraju et al., 2017; Sundararajan et
al., 2017; Lundberg & Lee, 2017), prototype-based interpretable models (Chen et al., 2019), and
multimodal fusion of image with tabular/morphometric features (Baltrušaitis et al., 2019) are the
strongest tools.

**3.7.3 Limitations.** Explanations are typically *displayed but not measured*: faithfulness is rarely
quantified against ground-truth structures or perturbation tests; and complementary morphometric
evidence is seldom integrated or evaluated for added value in this domain.

**3.7.4 Why the gap remains open.** There is no established cervical-cytology system that pairs faithful,
*quantified* explanation with complementary morphometric evidence under a reproducible protocol.

**3.7.5 Scientific stakes.** Whether explanations are faithful, and whether morphometric fusion adds
information beyond the image, are measurable scientific questions presently unanswered for this task.

**3.7.6 Clinical stakes.** Faithful, inspectable evidence is a precondition for the professional
accountability of cytopathologists and thus for adoption.

**3.7.7 Connection to RQ7.** This gap motivates **RQ7**: whether multimodal fusion improves performance
and whether the explanation suite provides faithful, cell-localised evidence.

## 3.8 The coupling of the gaps

The seven gaps are not independent. They share a common root in an evaluation culture that optimises and
reports a single within-dataset accuracy number (Section 1.7): leakage (G1) inflates that number;
single-dataset scope (G2) prevents it from estimating deployment performance; the absence of calibration
and abstention (G3) hides that the operating point fails under shift; taxonomic informality (G4)
contaminates the only measurement that could reveal cross-domain failure; imbalance (G5) is masked by
average accuracy; efficiency (G6) is reported without a common protocol; and opacity (G7) prevents the
human oversight that might catch the rest. The **reasonable interpretation** — and the organising thesis
of the work — is therefore that the gaps must be addressed by a *single, internally consistent
framework* whose foundation is an honest, leakage-safe, multi-domain, multi-metric, reliability-first
evaluation protocol, upon which targeted interventions can be assessed.

## 3.9 Chapter summary

This chapter has motivated the research by demonstrating, gap by gap, that each of the seven limitations
introduced in Chapter 1 is genuinely open against the strongest existing solutions, and by showing that
the gaps are coupled through a shared evaluation root. For each gap it identified why the problem exists,
appraised the best available remedies and their limitations, justified the gap's open status, articulated
its scientific and clinical stakes, and connected it to the corresponding research question. The cumulative
argument is that piecemeal fixes are insufficient and that a coherent, reliability-first framework is both
scientifically warranted and clinically necessary.

## 3.10 Remaining challenges

What remains is to convert this motivation into a precise research design: to state the overall and
specific objectives, the research questions and their falsifiable hypotheses, and the intended
contributions, each tied to an identified gap (Chapter 4); and to justify the methodological commitments
— leakage-safe evaluation, per-domain reporting, reliability-first design, honest crosswalks,
reproducibility and open science — as a coherent philosophy (Chapter 5).

## 3.11 Transition to Chapter 4

Chapter 4 now formalises the research objectives. It states the overall objective and the specific
objectives, reproduces the research questions (RQ1–RQ7), hypotheses (H1–H7), and contributions (C1–C6)
as fixed inputs, and the pre-registered evaluation protocol, and explains how each objective addresses an
identified limitation in the existing literature — still without describing how the objectives will be
implemented.

---

# Chapter 4 — Research Objectives

This chapter formalises the research design. It states the overall objective (Section 4.1) and the
specific objectives (Section 4.2); reproduces, as fixed inputs, the research questions (Section 4.3),
the hypotheses with their pre-registered success criteria (Section 4.4), and the contributions (Section
4.5); states the pre-registered evaluation protocol (Section 4.6); and consolidates the alignment between
gaps, objectives, questions, hypotheses, and contributions in a master mapping (Section 4.7). For each
objective the text explains which literature limitation it addresses. No implementation is described;
*how* the objectives are met is the subject of later volumes.

## 4.1 Overall objective

The overall objective of this research is:

> *To establish a trustworthy, reproducible, and clinically honest framework for deep-learning cervical
> cytology that (i) measures the true gap between within-dataset and cross-domain performance under a
> leakage-safe, multi-domain, multi-metric protocol, (ii) mechanistically diagnoses the sources of any
> cross-domain failure, and (iii) determines whether targeted, evidence-based interventions
> demonstrably reduce those failures while preserving calibration and safe abstention.*

This objective deliberately subordinates raw accuracy to *trustworthiness*, defined operationally in
Section 1.10 as the conjunction of honest evaluation, demonstrated (or honestly bounded) generalisation,
calibrated abstention, safe taxonomy handling, and faithful explanation.

## 4.2 Specific objectives

The overall objective decomposes into seven specific objectives (SO1–SO7), each addressing an identified
literature limitation (G1–G7) and each instantiated by a research question (RQ1–RQ7).

- **SO1 — Honest benchmark (addresses G1).** Establish a leakage-safe, cluster-grouped, sealed-test
  evaluation with released splits and seeds, and quantify the inflation attributable to leakage.
- **SO2 — Cross-domain generalisation (addresses G2).** Determine whether stain normalisation,
  self-supervised pretraining, and domain-aware training narrow the internal→external gap, and
  characterise the dependence of the residual gap on preparation distance.
- **SO3 — Calibrated, risk-aware operation (addresses G3).** Establish per-domain calibration and
  coverage-controlled abstention, and determine whether they restore safe operation under shift.
- **SO4 — Safe cross-taxonomy alignment (addresses G4).** Construct a deterministic many-to-one crosswalk
  with an explicit exclusion set and reported excluded fraction, and assess partial-label multi-dataset
  learning.
- **SO5 — Minority-class rescue (addresses G5).** Determine whether conditional synthesis improves
  minority/high-grade recall without degrading calibration.
- **SO6 — Deployable, architecture-agnostic accuracy (addresses G6).** Compare a distilled compact model
  against larger backbones and published specialised architectures under one protocol with efficiency
  profiling.
- **SO7 — Faithful, multimodal explanation (addresses G7).** Determine whether multimodal fusion adds
  value and whether the explanation suite provides faithful, quantified, cell-localised evidence.

## 4.3 Research questions (fixed input)

The research questions are reproduced verbatim from the project's pre-registered framing.

| ID | Gap | Research question |
|----|-----|-------------------|
| **RQ1** | G1 | Can a cluster-grouped, stratified 80/10/10 split with a sealed test and released seeds/splits provide a reproducible cervical-cytology benchmark whose in-distribution score reflects genuine within-dataset generalisation? |
| **RQ2** | G2 | Do stain normalisation, self-supervised pretraining, and domain-aware augmentation produce a classifier that generalises across slide-preparation domains? |
| **RQ3** | G3 | Do per-domain temperature scaling, conformal prediction, and an OOD/energy gate deliver trustworthy confidence and safe abstention in- and cross-distribution? |
| **RQ4** | G4 | Can a deterministic many-to-one crosswalk (Route A) plus partial-label learning enable safe cross-taxonomy evaluation and multi-dataset training without the dangerous one-to-many projection? |
| **RQ5** | G5 | Does conditional synthesis of under-represented morphology classes improve minority-class recall while preserving calibration? |
| **RQ6** | G6 | Can knowledge distillation produce a compact student that matches larger backbones and published cervical-cytology architectures under one protocol? |
| **RQ7** | G7 | Does late fusion with tabular morphometric features improve accuracy, and does the explanation suite provide faithful, cell-localised evidence? |

## 4.4 Hypotheses and pre-registered success criteria (fixed input)

Each research question is paired with a falsifiable hypothesis predicting that the proposed method
*succeeds* against a pre-registered, quantitative criterion. Pre-registering the criterion before any
result exists guards against post-hoc goalpost-moving; a hypothesis is corroborated only if its criterion
is met and is otherwise reported as refuted.

| ID | Hypothesis | Pre-registered success criterion |
|----|-----------|----------------------------------|
| **H1** | The grouped, sealed-test protocol yields a reproducible benchmark whose 5-class metrics are stable across seeds within their 95% bootstrap CIs and reproducible from the released split. | Bootstrap-CI-stable metrics across ≥3 seeds; split/seed-reproducible; zero cluster overlap. |
| **H2** | The domain-generalisation stack achieves external binary balanced accuracy and AUROC clearly above the majority baseline and above the un-normalised backbone on conventional and liquid-based targets. | External balanced-acc and AUROC above majority baseline AND above the no-stain/no-SSL arm; positive gap reduction vs baseline. |
| **H3** | Temperature scaling yields low ECE in- and cross-distribution; split-conformal sets meet target coverage on the sealed test; the OOD gate flags external shift so high-confidence errors are abstained. | Post-scaling ECE below pre; conformal empirical coverage ≥ 1−α; OOD AUROC > 0.5 with abstention improving selective accuracy. |
| **H4** | The deterministic crosswalk gives a clinically safe, fully-documented E0/E1 evaluation with a reported excluded fraction; partial-label multi-dataset training improves external performance over single-dataset training. | 100% deterministic many-to-one coverage with reported excluded fraction; partial-label external balanced-acc ≥ single-dataset baseline. |
| **H5** | Training with synthesised minority samples improves minority-class recall and balanced accuracy at equal-or-better ECE versus class-weighting/focal alone. | Minority recall and balanced acc up vs weights/focal at ECE no worse. |
| **H6** | The distilled compact student matches or exceeds its teacher and is within a small margin of the best backbone/specialised architecture in-distribution at a fraction of the parameters/latency. | Student ≥ teacher in-distribution; within a small delta of the best backbone/architecture at much lower params/FLOPs/latency. |
| **H7** | Late fusion gives a measurable in-distribution lift over the image-only model, and attributions localise to nuclear/cytoplasmic structure with quantified faithfulness. | Fusion balanced-acc ≥ image-only; attribution faithfulness reported (e.g., deletion/insertion, nucleus localisation); prototypes class-consistent. |

A consolidated *hypothesis-status* table (Hypothesis / Claim / Verdict / Evidence-location) will be
maintained in the experimental volume so that each hypothesis is explicitly adjudicated against its
criterion, including honest reporting of any refutation.

## 4.5 Expected contributions (fixed input)

The intended contributions, reproduced from the project framing, are the resolutions of the specific
objectives.

- **C1 — Leakage-safe, fully-reproducible benchmark.** A cluster-grouped, stratified, sealed-test split
  with released seeds/splits and bootstrap/McNemar/Wilcoxon significance, giving a trustworthy
  in-distribution benchmark. *(SO1/RQ1; addresses G1.)*
- **C2 — Domain-generalisable classifier across preparation types.** A stain-normalised,
  self-supervised-pretrained, domain-aware model that improves cross-dataset performance over
  un-normalised backbones. *(SO2/RQ2; addresses G2.)*
- **C3 — Calibrated, risk-aware deployment.** Per-domain temperature scaling, split/Mondrian conformal
  prediction, and an OOD gate that together give trustworthy confidence and safe abstention under domain
  shift. *(SO3/RQ3; addresses G3.)*
- **C4 — Safe cross-taxonomy learning and evaluation.** A deterministic many-to-one crosswalk plus
  partial-label routes that enable clinically safe multi-dataset training and honest E0/E1 evaluation
  with reported coverage. *(SO4/RQ4; addresses G4.)*
- **C5 — Minority-class rescue via conditional synthesis.** Conditional-synthesis augmentation that
  raises rare/minority morphology recall at maintained calibration. *(SO5/RQ5; addresses G5.)*
- **C6 — Deployable, explainable, full-provenance system.** A distilled compact model benchmarked against
  published architectures, with a faithful explanation suite, multimodal fusion, and every reported value
  and figure persisted as JSON/CSV for exact reproduction. *(SO6+SO7/RQ6+RQ7; addresses G6, G7.)*

## 4.6 Pre-registered evaluation protocol (fixed input)

The evaluation protocol is fixed in advance of any result and governs every empirical claim in later
volumes. Its commitments are:

- **Split.** Cluster-grouped 80/10/10 partition, stratified by class, with a fixed seed and a **sealed
  test** set untouched during model development; zero cluster overlap is asserted.
- **Fitting discipline.** Training and early-stopping/model-selection use the **train and validation**
  partitions only; the test partition is never consulted during development.
- **Internal evaluation.** All in-distribution metrics are reported on the sealed test partition.
- **External evaluation.** The frozen model is evaluated on external domains at the coarse tiers — E0
  (binary normal/abnormal), E1 (grade), and an E-morph candidate-set tier — with the **excluded
  fraction** reported per domain.
- **Primary metrics.** Balanced accuracy, macro-F1, one-vs-rest AUROC, expected calibration error, and
  conformal coverage, reported **per domain** and against the **majority-class baseline**.
- **Operating point.** A pre-registered threshold (τ = 0.5) for the binary tier, with per-domain
  temperature scaling for calibrated reporting; any oracle threshold is reported only as an upper bound,
  never as a deployable result.
- **Significance and stability.** 95% bootstrap confidence intervals, McNemar's test, the Wilcoxon
  signed-rank test, and ≥3 seeds.
- **Provenance.** Every reported value is persisted to machine-readable JSON/CSV and every figure to a
  companion data file, so that the entire empirical record is exactly reproducible.

## 4.7 Master alignment of gaps, objectives, questions, hypotheses, and contributions

The internal consistency of the research design is summarised in a single mapping, ensuring that every
gap is addressed by an objective, instantiated by a question, tested by a hypothesis, and resolved by a
contribution.

| Gap | Specific objective | Research question | Hypothesis | Contribution |
|-----|--------------------|-------------------|-----------|--------------|
| G1 (leakage/reproducibility) | SO1 | RQ1 | H1 | C1 |
| G2 (cross-domain generalisation) | SO2 | RQ2 | H2 | C2 |
| G3 (calibration/abstention) | SO3 | RQ3 | H3 | C3 |
| G4 (taxonomy incompatibility) | SO4 | RQ4 | H4 | C4 |
| G5 (imbalance/rare classes) | SO5 | RQ5 | H5 | C5 |
| G6 (efficiency/deployability) | SO6 | RQ6 | H6 | C6 |
| G7 (opacity/modality) | SO7 | RQ7 | H7 | C6 |

(The deployability and explainability objectives SO6 and SO7 are jointly realised in the single
deployable-and-explainable system contribution C6.)

## 4.8 How the objectives address the literature

Each objective is, by construction, the constructive counterpart of a documented limitation appraised in
Chapter 3. SO1 instantiates the leakage-safe, reproducible benchmark whose absence Section 3.1 identified.
SO2 supplies the multi-preparation generalisation evaluation and intervention study missing per Section
3.2. SO3 brings per-domain calibration and abstention — present as general tools but unapplied to this
task (Section 3.3) — to bear. SO4 replaces the informal relabelling criticised in Section 3.4 with an
explicit, documented crosswalk and partial-label learning. SO5 evaluates minority synthesis jointly with
calibration, the joint assessment whose absence Section 3.5 noted. SO6 provides the apples-to-apples,
efficiency-profiled comparison absent per Section 3.6. SO7 makes explanation *measured* rather than merely
displayed, and tests multimodal value, addressing Section 3.7. The objectives thus collectively close the
coupled gap-structure of Section 3.8 within a single framework.

## 4.9 Chapter summary

This chapter formalised the research design. It stated an overall objective that subordinates accuracy to
trustworthiness and decomposed it into seven specific objectives, each mapped to a gap. It reproduced, as
fixed inputs, the seven research questions, the seven hypotheses with pre-registered quantitative success
criteria, and the six contributions, and stated the pre-registered evaluation protocol governing all
empirical claims. A master mapping confirmed the internal consistency of the design — every gap addressed
by an objective, instantiated by a question, tested by a hypothesis, and resolved by a contribution — and
the chapter explained how each objective addresses a specific documented limitation.

## 4.10 Remaining challenges

The objectives and protocol assert a number of methodological commitments — leakage-safe and sealed
evaluation, per-domain multi-metric reporting against baselines, pre-registration of success criteria,
reliability-first prioritisation, honest crosswalks with reported exclusions, and full provenance. These
commitments require justification as a coherent scientific philosophy rather than a list of practices, and
they raise ethical, data-governance, and open-science considerations that have not yet been addressed.

## 4.11 Transition to Chapter 5

Chapter 5 supplies that justification. It articulates the research paradigm and the quantitative,
experimental, and reproducible-science principles underpinning the design; develops the leakage-safe and
cross-domain validation philosophies and the reliability-first stance as principled commitments; and
addresses clinical-translation, ethical, FAIR-data, and open-science considerations — still without
describing model architectures or implementation.

---

# Chapter 5 — Research Philosophy and Design

This chapter justifies the methodological commitments asserted in Chapter 4 as a coherent research
philosophy rather than a list of practices. It states the research paradigm (Section 5.1) and the
quantitative, experimental, and reproducible-science principles that follow from it (Sections 5.2–5.4);
articulates the evidence-based stance toward AI development (Section 5.5); develops the leakage-safe and
cross-domain validation philosophies and the reliability-first prioritisation as principled positions
(Sections 5.6–5.8); and addresses clinical-translation, ethical, FAIR-data, and open-science
considerations (Sections 5.9–5.12). Consistent with Volume I, no model architecture or implementation is
described; the philosophy described here *constrains* the implementation developed in later volumes.

## 5.1 Research paradigm

The work is conducted within a **post-positivist, empirical-quantitative paradigm**. It assumes that
claims about a model's behaviour are empirical claims about measurable quantities, that such claims are
provisional and revisable, and that the scientific value of a claim derives from the rigour with which it
has been *exposed to potential refutation* (Popper, 1959). This paradigm has a direct and unusual
consequence for the present subject matter: because the field's central claims (e.g., "this model is
accurate") have typically been tested under conditions that cannot refute them (leakage-prone,
single-dataset evaluation; Chapter 3), the paradigm demands that the research first *reconstruct the
conditions of refutation* — an honest, leakage-safe, multi-domain protocol — before any claim can be
meaningfully advanced. The paradigm thus motivates the audit-then-remediate structure of the work: the
diagnostic part (Section 2.19.5) establishes refutable measurements; the constructive part subjects each
proposed intervention to a pre-registered criterion that it could fail.

## 5.2 Quantitative methodology

The methodology is quantitative throughout. Performance is characterised not by a scalar but by a vector
of pre-specified quantities (Section 4.6) — threshold-free discrimination, operating-point performance
against a baseline, calibration, and reliability/abstention — reported per domain. Uncertainty in every
estimate is quantified (95% bootstrap confidence intervals), and comparisons between methods are
subjected to appropriate statistical tests (McNemar for paired classifications, Wilcoxon signed-rank for
paired metric differences), with stability assessed across ≥3 seeds. The methodological principle is that
a difference is not a finding unless it is *both* of plausible magnitude *and* unlikely under the null,
and that an estimate without an interval is not a measurement. This discipline is adopted specifically to
counter the field's tendency to report point estimates of single metrics without uncertainty.

## 5.3 Experimental philosophy

The experimental philosophy is **controlled and ablative**. Interventions are evaluated by changing one
factor at a time under an otherwise identical protocol, so that a measured change can be attributed to
the factor rather than to confounds. Two further commitments follow. First, every intervention is paired
with an honest *control* — most importantly, a deliberately leaky split as the control for the
leakage-safe split, so that the leakage tax is measured rather than assumed. Second, *negative results
are first-class*: a pre-registered hypothesis that fails its criterion is reported as refuted, with the
same prominence as a corroboration, because suppressing inconvenient results would reintroduce exactly
the bias the framework exists to remove. The experimental philosophy therefore privileges *attributable*
and *honest* evidence over favourable headline numbers.

## 5.4 Reproducible-science principles

Reproducibility is treated as a constitutive property of a scientific claim, not an optional courtesy.
The work adopts the now-standard reproducibility prescriptions for machine learning and for ML in
medical imaging (Pineau et al., 2021; Varoquaux & Cheplygina, 2022): deterministic seeding; release of
the exact data partitions; persistence of every reported value and figure to machine-readable companion
files; and recording of the computational environment. The operational test of reproducibility adopted
here is stringent: the headline results must be regenerable bit-for-bit from the released split and seed,
and every figure must be reconstructable from its persisted data. This **provenance-first** practice is
what allows the diagnostic claims of the work to be independently checked and is itself a contribution
(C1, C6).

## 5.5 Evidence-based AI development

The work adopts an **evidence-based** stance toward methodological choices, by analogy with evidence-based
medicine. A design decision (e.g., to adopt stain normalisation, self-supervised pretraining, or
conditional synthesis) is justified by the strongest available evidence for its effect, is then *tested*
in the target setting rather than assumed to transfer, and is retained only if it meets its pre-registered
criterion. This stance is deliberately sceptical of methodological fashion: the inclusion of a technique
is contingent on demonstrated benefit under the protocol, and the framework is prepared to report that a
widely used technique does *not* help in this domain. Evidence-based development thus operationalises the
post-positivist paradigm (Section 5.1) at the level of engineering choices.

## 5.6 Leakage-safe evaluation philosophy

The leakage-safe philosophy holds that an evaluation is only as valid as the independence of its test
set. Because cervical cytology generates many cells per acquisition context, statistical independence
requires partitioning at the level of the latent grouping variable (field/slide), not the cell (Section
2.10). The philosophy makes three commitments: (i) *group-aware partitioning* with asserted zero overlap;
(ii) a *sealed test* set, consulted exactly once, after all development is complete, so that it cannot
leak into model selection; and (iii) *measurement of the leakage tax* via a leaky control, so that the
magnitude of the bias the field has been incurring is quantified rather than merely asserted. This
philosophy is non-negotiable in the design because every downstream claim inherits the validity, or
invalidity, of the internal estimate.

## 5.7 Cross-domain validation philosophy

The cross-domain philosophy holds that the quantity of clinical interest is performance on data from a
*different* acquisition pipeline than the training data, and that this quantity must be measured directly
rather than extrapolated from within-dataset performance. Three commitments follow: (i) *external
validation across multiple preparation modalities* (conventional smear, liquid-based, auto-segmented),
chosen to span the preparation axis along which domain shift is expected to vary; (ii) *honest taxonomy
alignment* via an explicit, deterministic crosswalk with a reported excluded fraction (Section 2.7), so
that a cross-domain change is attributable to the model rather than the labels; and (iii) *decomposition*
of any observed degradation into labelling artefact, genuine domain shift (ordered by preparation
distance), and operating-point/calibration failure (Section 2.19.5). The philosophy converts "it does not
generalise" from an excuse into a set of falsifiable, separately testable claims.

## 5.8 Reliability-first artificial intelligence

The reliability-first stance — introduced in Sections 1.10 and 2.16 — is the philosophical core of the
design. It holds that, for a screening assistant, the property that gates safe deployment is not raw
accuracy but *trustworthy behaviour under uncertainty and shift*: calibrated confidence, distribution-free
coverage guarantees, recognition of out-of-distribution inputs, and the ability to abstain and defer.
Methodologically, reliability-first implies that calibration and abstention are *designed and evaluated as
primary objectives*, measured per domain, and that a model which is less accurate but reliably uncertain
is preferred to a more accurate model that fails silently. This inverts the field's customary
prioritisation and is, in the author's **interpretation**, the single most consequential commitment of the
research, because it aligns the optimisation and reporting targets with the conditions of clinical use.

## 5.9 Clinical-translation considerations

The design is shaped by the requirements of eventual clinical translation, even though prospective
clinical validation is out of scope (Section 1.15). It adopts an *assistive, human-in-the-loop* deployment
model (Section 2.17), in which abstention routes uncertain cases to expert review; it foregrounds
*efficiency* so that the system is compatible with low-resource and point-of-care settings; it aligns
reporting with emerging standards for AI prediction models and AI in medical imaging (e.g., TRIPOD-style
reporting; the CLAIM checklist for medical-imaging AI; Mongan et al., 2020), so that the evidence produced
is intelligible to clinical and regulatory audiences; and it states scope and structural limitations
explicitly, so that claims are not over-generalised beyond the evidence. These considerations constrain
*what is measured and reported*, not how the model is built.

## 5.10 Ethical considerations

The research uses publicly available, de-identified single-cell image datasets and therefore does not
involve direct human-subjects experimentation; nonetheless, ethical commitments shape the design. The
foremost is *non-maleficence through honesty*: because over-stated performance in a screening tool can
translate into missed disease, the work treats inflated or non-reproducible claims as an ethical, not
merely scientific, failure, and commits to reporting limitations and negative results faithfully. The
work further attends to *equity*, prioritising deployability in the low-resource settings that bear the
greatest cervical-cancer burden, and to *appropriate use*, framing the system as a decision *support*
tool under professional oversight rather than an autonomous diagnostic device. The broader principles of
trustworthy AI — human agency and oversight, technical robustness and safety, transparency, and
accountability (e.g., European Commission High-Level Expert Group on AI, 2019) — inform these commitments.

## 5.11 FAIR data principles

The work adopts the FAIR principles — that research artefacts be **F**indable, **A**ccessible,
**I**nteroperable, and **R**eusable (Wilkinson et al., 2016) — for its data products and metadata.
Concretely: the exact partitions, seeds, label crosswalks, and per-figure data are released in
machine-readable formats with documentation, so that they are findable and accessible; the crosswalk and
metric definitions are made explicit so that results are interoperable across datasets and reusable by
other investigators; and dataset provenance and composition are documented in the spirit of *datasheets
for datasets* (Gebru et al., 2021) and model behaviour in the spirit of *model cards* (Mitchell et al.,
2019). FAIR practice is the data-governance counterpart of the reproducibility philosophy (Section 5.4).

## 5.12 Open-science practices

Finally, the work commits to **open-science** practices: pre-registration of hypotheses and success
criteria before results exist (Section 4.4); release of code, splits, seeds, metrics, and figure data;
transparent reporting of the computational environment; and documentation sufficient for independent
reproduction and critique. These practices are not ornamental: they are the mechanism by which the
work's central methodological contribution — an honest, checkable evaluation of cervical-cytology AI —
becomes usable by, and falsifiable by, the wider community, thereby exerting the corrective force that
mere recommendations (Section 3.1.2) could not.

## 5.13 Chapter summary

This chapter justified the methodological commitments of the research as a coherent philosophy. Within a
post-positivist, refutation-oriented paradigm (Section 5.1), the work adopts a quantitative,
uncertainty-aware methodology (Section 5.2) and a controlled, ablative, negative-result-honest
experimental philosophy (Section 5.3), underpinned by stringent reproducibility and provenance practices
(Section 5.4) and an evidence-based stance toward methodological choices (Section 5.5). It elevated
leakage-safe evaluation (Section 5.6), direct cross-domain validation with honest taxonomy alignment and
failure decomposition (Section 5.7), and reliability-first prioritisation (Section 5.8) to principled
commitments, and situated the design within clinical-translation (Section 5.9), ethical (Section 5.10),
FAIR-data (Section 5.11), and open-science (Section 5.12) considerations. Together these constitute the
methodological constraints within which the framework of Chapter 6 — and its later implementation — must
operate.

## 5.14 Remaining challenges

The philosophy articulated here must now be assembled into a single conceptual framework that shows *how
the parts relate*: how the gaps, questions, and contributions connect, how the diagnostic and constructive
problems are sequenced, how knowledge is expected to be generated and validated, and what the scientific
reasoning behind the framework's structure is. Expressing these relationships at a conceptual level —
without lapsing into implementation — is the task of Chapter 6.

## 5.15 Transition to Chapter 6

Chapter 6 presents the conceptual framework. It describes the overall research workflow, the relationships
among research gaps, questions, and contributions, the expected knowledge-generation and validation
process, and the scientific reasoning that organises the framework, using conceptual diagrams described in
text. It remains strictly conceptual: no neural-network architecture and no implementation are discussed.

---

# Chapter 6 — Conceptual Framework

This chapter presents the research framework at a conceptual level. It describes the overall workflow
(Section 6.1), the relationships among the research gaps (Section 6.2), questions (Section 6.3), and
contributions (Section 6.4), the expected knowledge-generation process (Section 6.5), the validation
philosophy that binds the framework together (Section 6.6), and the scientific reasoning that organises it
(Section 6.7). Conceptual diagrams are described in text (the field convention of a schematic figure is
preserved by an explicit textual specification of each figure's content). The chapter contains no
neural-network architecture and no implementation detail; it specifies *what relates to what and why*,
which the later volumes will then realise.

## 6.1 Overall research workflow

The framework is organised as a two-phase, audit-then-remediate workflow, reflecting the post-positivist
paradigm of Section 5.1.

> **Figure 6.1 (described).** *The audit-then-remediate workflow.* A horizontal flow with two phases.
> **Phase A (Audit / diagnostic)**, left: a single source dataset feeds a *leakage-safe partitioning*
> block (group-aware split with a sealed test) → a *frozen reference model* block → an *honest evaluation*
> block that fans out to (i) internal sealed-test evaluation and (ii) external multi-domain evaluation
> through the *deterministic crosswalk* (with excluded-fraction reporting). The evaluation block emits a
> *diagnosis* node that decomposes any internal→external gap into three labelled arrows — *labelling
> artefact*, *domain shift (ordered by preparation distance)*, and *operating-point / calibration
> failure*. **Phase B (Remediate / constructive)**, right: the diagnosis conditions a set of *targeted
> interventions* (domain-generalisation, calibration-and-abstention, cross-taxonomy learning,
> minority-class synthesis, efficient distillation, multimodal explanation), each evaluated **under the
> identical protocol** of Phase A and each adjudicated against its pre-registered criterion. A feedback
> arrow returns each intervention's verdict to a running *hypothesis-status ledger*.

The essential conceptual feature is that **the evaluation protocol is shared between the two phases**: the
same leakage-safe, multi-domain, multi-metric, reliability-first measurement that exposes the failures in
Phase A is the measurement against which remedies are judged in Phase B. This shared substrate is what
makes the constructive claims commensurable with the diagnostic ones and is the structural embodiment of
the reliability-first philosophy.

## 6.2 Relationships among the research gaps

The seven gaps are not a flat list but a *coupled structure* with a single root, as argued in Section 3.8.

> **Figure 6.2 (described).** *The gap-coupling graph.* A directed graph with a central root node,
> "single-within-dataset-accuracy evaluation culture." From the root, edges point to G1 (leakage inflates
> the number) and G2 (single-dataset scope prevents it estimating deployment). G1 and G2 jointly enable a
> *masking* cluster: G3 (calibration/abstention absence hides operating-point failure under shift), G4
> (taxonomy informality contaminates the cross-domain measurement), and G5 (imbalance is hidden by average
> accuracy). G6 (efficiency without a common protocol) and G7 (opacity preventing human oversight) attach
> as orthogonal but reinforcing concerns. A dashed "containment" boundary encloses all seven, labelled
> "addressable only by a shared honest protocol."

The conceptual claim encoded by this graph is **interpretation**, not established fact: the gaps persist
together because they are symptoms of one methodological pathology, and therefore a remedy that treats any
one in isolation — for example, a new architecture (touching G6) evaluated under the leaky, single-dataset
regime — cannot resolve the others and may not even resolve its own, because the measurement remains
invalid. The framework's response is to repair the root (the protocol) first, then address the symptoms.

## 6.3 Relationships among the research questions

The research questions inherit the structure of the gaps and are themselves ordered by logical dependency.

> **Figure 6.3 (described).** *The question-dependency diagram.* RQ1 (honest benchmark) sits at the base,
> with upward dependency arrows to every other question, signifying that no other question can be answered
> validly until RQ1 establishes a trustworthy measurement. Above RQ1, RQ4 (honest crosswalk) is a second
> enabling node, with arrows to RQ2, RQ3, and RQ6, signifying that valid cross-domain answers require an
> honest taxonomy alignment. RQ2 (generalisation), RQ3 (calibration/abstention), and RQ5 (minority rescue)
> form a middle tier of *constructive* questions, mutually linked by a shared "preserve calibration"
> constraint (an edge from RQ3 to RQ2 and RQ5 labelled "without degrading reliability"). RQ6 (deployable
> comparison) and RQ7 (explanation/fusion) sit at the top as *deployment-readiness* questions, each
> depending on the validity established below.

The dependency structure expresses a methodological ordering: RQ1 and RQ4 are *enabling* (they make valid
measurement possible); RQ2, RQ3, RQ5 are *constructive under a reliability constraint* (interventions must
help without harming calibration); and RQ6, RQ7 are *integrative* (they assess the deployable,
explainable system). This ordering is reflected in the volume structure of the monograph and in the
sequencing of the experimental programme.

## 6.4 Relationships among the contributions

The six contributions map onto the question structure and accumulate into a single deliverable.

> **Figure 6.4 (described).** *The contribution-accumulation diagram.* C1 (leakage-safe benchmark) is the
> foundation block. C4 (safe cross-taxonomy crosswalk) rests on C1. C2 (generalisation), C3
> (calibration/abstention), and C5 (minority rescue) are three pillars resting jointly on C1 and C4. C6
> (deployable, explainable, full-provenance system) is the capstone spanning all pillars, with a base
> band labelled "full provenance" running beneath every block to indicate that reproducibility is a
> property of the whole, not an add-on.

The conceptual point is that the contributions are *cumulative and load-bearing*: C1 and C4 carry the
epistemic weight (without them, C2/C3/C5 cannot be validly demonstrated), while C6 integrates the pillars
into a system whose every claim is reproducible. The framework therefore delivers not seven separate
results but one coherent, honestly evaluated artefact.

## 6.5 Expected knowledge-generation process

Knowledge is generated in the framework by a repeated cycle of *measure → diagnose → intervene →
re-measure → adjudicate*, governed by the pre-registered criteria of Section 4.4.

> **Figure 6.5 (described).** *The knowledge-generation cycle.* A closed loop: a *measurement* node (the
> shared protocol) produces evidence; a *diagnosis* node interprets it into mechanism-level claims; an
> *intervention* node applies an evidence-based remedy; a *re-measurement* node evaluates the remedy under
> the identical protocol; and an *adjudication* node compares the result against the pre-registered
> criterion, writing a corroborated/refuted verdict into the hypothesis-status ledger before the loop
> repeats. An external arrow exits the loop to "released artefacts (splits, seeds, metrics, figures)."

Two epistemic safeguards are built into this cycle. First, *pre-registration*: the criterion against
which an intervention is judged is fixed before the result is seen, so the loop cannot be biased by
post-hoc reinterpretation. Second, *symmetry of outcome*: a refuted hypothesis is recorded and reported
with the same weight as a corroborated one, so the knowledge produced is the honest set of what does and
does not work in this domain under an honest protocol — which is precisely the knowledge the field
currently lacks.

## 6.6 Validation philosophy

Validation in the framework is *internal-and-external, multi-metric, and per-domain*, and is identical in
the diagnostic and constructive phases (Section 6.1). Conceptually, validation answers four questions for
every model or intervention, in order: (i) *Is the internal estimate honest?* — addressed by leakage-safe,
sealed-test evaluation. (ii) *Does it transfer?* — addressed by external evaluation across preparation
domains through the honest crosswalk. (iii) *Is it reliable where it transfers?* — addressed by per-domain
calibration and coverage-controlled abstention. (iv) *Is it deployable and inspectable?* — addressed by
efficiency profiling and quantified, faithful explanation. A model that succeeds on (i) but fails (ii)–(iv)
is, under this philosophy, explicitly *not* validated, regardless of its headline accuracy — the precise
inversion of the field's customary standard that the framework is designed to enforce.

## 6.7 Scientific reasoning behind the framework

The framework's structure follows from three reasoned premises, stated as the **interpretation** that
organises the work. *First premise*: the validity of every empirical claim in the domain is bounded by the
validity of its evaluation, so repairing evaluation is logically prior to any methodological contribution
(motivating the audit-first structure and the foundational role of C1/C4). *Second premise*: cross-domain
failure is not monolithic but decomposable into labelling-artefact, domain-shift, and
operating-point/calibration components that are *separately testable* and *separately remediable*
(motivating the diagnostic decomposition and the targeted, ablative interventions). *Third premise*: for a
screening assistant, trustworthiness — calibrated, abstaining, inspectable behaviour under shift — is the
operative criterion of success, not raw accuracy (motivating reliability-first prioritisation and the
four-question validation philosophy). From these premises the entire architecture of the framework — shared
protocol, two-phase workflow, cumulative contributions, pre-registered adjudication — follows as the
minimal structure that honours all three. The framework is, in this sense, not an arbitrary assemblage of
components but the conceptual consequence of taking honest evaluation, mechanistic diagnosis, and clinical
reliability seriously and simultaneously.

## 6.8 Chapter summary

This chapter specified the research framework conceptually. It described a two-phase audit-then-remediate
workflow built on a shared evaluation protocol (Section 6.1); represented the gaps as a coupled structure
with a single methodological root (Section 6.2), the questions as a dependency hierarchy with RQ1 and RQ4
as enabling nodes (Section 6.3), and the contributions as a cumulative, load-bearing structure with C6 as
capstone over a provenance base (Section 6.4); set out the pre-registered measure–diagnose–intervene–
re-measure–adjudicate knowledge-generation cycle with its pre-registration and outcome-symmetry safeguards
(Section 6.5); articulated an internal-and-external, multi-metric, per-domain, four-question validation
philosophy (Section 6.6); and grounded the whole in three reasoned premises about evaluation validity,
failure decomposability, and reliability-first success (Section 6.7). The framework is conceptual
throughout; its realisation is deferred to later volumes.

## 6.9 Remaining challenges

What remains for Volume I is to consolidate its argument — motivation, conceptual background, problem
formulation, objectives, philosophy, and framework — into a coherent summary, to make explicit the
theoretical foundations on which the implementation will rest, and to motivate the transition to dataset
engineering (Volume II) without anticipating its implementation. The principal substantive challenge
carried forward is that the framework's commitments are demanding: instantiating a genuinely leakage-safe,
multi-domain, reliability-first protocol, and an honest crosswalk, is itself non-trivial engineering and
data work, which Volume II must address before any model is trained.

## 6.10 Transition to Chapter 7

Chapter 7 concludes Volume I. It summarises the scientific motivation, the theoretical foundations, the
identified gaps, the research objectives, and the expected contributions established across Chapters 1–6,
and motivates the transition to Volume II (Dataset Engineering), which will operationalise the data-level
commitments — leakage-safe grouped partitioning, the cross-taxonomy crosswalk, and exploratory and
characterisation analyses — that the framework requires, again without pre-empting implementation detail.

---

# Chapter 7 — Summary

This chapter concludes Volume I. It consolidates the scientific motivation (Section 7.1), the theoretical
foundations (Section 7.2), the identified research gaps (Section 7.3), the research objectives (Section
7.4), and the expected contributions (Section 7.5) established across Chapters 1–6, and motivates the
transition to Volume II (Section 7.6). In keeping with the remit of Volume I, it introduces no
implementation detail.

## 7.1 Scientific motivation

Cervical cancer is a preventable, HPV-driven malignancy whose large and inequitably distributed burden
falls disproportionately on low-resource settings, and whose control depends on screening pathways in
which cytology plays a central but workforce-limited role (Chapter 1). Deep learning has demonstrated
strong within-dataset accuracy on cytological benchmarks, yet its clinical role remains, at present,
demonstrated in principle but not trustworthy in practice. The motivating diagnosis of this work is that a
single, systemic flaw — an evaluation culture that optimises and reports one within-dataset accuracy
number — both inflates reported performance and conceals the failures of generalisation, calibration,
taxonomy alignment, minority detection, deployability, and explainability that separate reported from
deployable performance. The research is therefore motivated not to add another accuracy result but to
reconstruct honest measurement, diagnose the true gap, and test targeted remedies — a posture of *honest
auditing followed by evidence-based remediation*.

## 7.2 Theoretical foundations

Chapter 2 established the conceptual foundations: the biology and clinical practice of cervical cytology
and the morphological features on which diagnosis rests; the grading systems and the Bethesda System; the
SIPaKMeD and Herlev taxonomies and the formal sense in which they are incompatible (granularity, semantics,
scope, and projection hazard); the preparation modalities and the domain shift between them; and the
technical phenomena of leakage, bias, imbalance, and annotation uncertainty, alongside the reliability
concepts — clinical-risk asymmetry, explainability and its faithfulness caveat, calibration and its
domain-locality, reliability and abstention, and the human–AI collaborative setting — that the framework
treats as first-class. These were consolidated into a formal problem formulation (Section 2.19) with a
*diagnostic* part (decompose the internal→external discrepancy into labelling artefact, domain shift, and
operating-point/calibration failure) and a *constructive* part (test whether targeted interventions reduce
the diagnosed failures while preserving reliability), defined over incompatible taxonomies, a deterministic
crosswalk with reported exclusions, leakage-safe internal estimation, and per-domain multi-metric
evaluation.

## 7.3 Identified research gaps

Chapter 3 substantiated seven coupled research gaps against the strongest existing solutions: G1,
non-reproducible, leakage-prone evaluation; G2, single-dataset training and unsolved cross-domain
generalisation; G3, overconfidence and the absence of uncertainty or abstention; G4, fragmented,
incompatible label taxonomies; G5, severe imbalance and missed rare or high-grade classes; G6, the
accuracy–efficiency trade-off obstructing point-of-care deployment; and G7, black-box, image-only
predictions limiting clinical trust. For each, the chapter showed why the problem exists, appraised the
best available remedies and their limitations, justified the gap's open status, and articulated its
scientific and clinical stakes. Crucially, it argued that the gaps are not independent but symptoms of a
single methodological root, and therefore demand a coherent framework rather than seven isolated fixes.

## 7.4 Research objectives

Chapter 4 formalised the design. The overall objective subordinates raw accuracy to *trustworthiness* and
decomposes into seven specific objectives (SO1–SO7), each mapped to a gap and instantiated by a research
question (RQ1–RQ7). Seven hypotheses (H1–H7) were stated with pre-registered, quantitative success
criteria, so that each can be honestly adjudicated as corroborated or refuted; and a pre-registered
evaluation protocol — cluster-grouped sealed-test partitioning, train/validation-only fitting, external
E0/E1/E-morph evaluation with reported excluded fractions, per-domain multi-metric reporting against
baselines, pre-registered operating point with per-domain calibration, statistical significance and
multi-seed stability, and full provenance — was fixed to govern every empirical claim. A master mapping
confirmed that every gap is addressed by an objective, instantiated by a question, tested by a hypothesis,
and resolved by a contribution.

## 7.5 Expected scientific contributions

The work's intended contributions (Chapter 4, Section 6.4) are cumulative and load-bearing: C1, a
leakage-safe, fully-reproducible benchmark; C2, a domain-generalisable classifier across preparation
types; C3, calibrated, risk-aware deployment with abstention; C4, safe cross-taxonomy learning and
evaluation; C5, minority-class rescue via conditional synthesis; and C6, a deployable, explainable,
full-provenance system that integrates the pillars. C1 and C4 carry the epistemic weight by making valid
measurement possible; C2, C3, and C5 are constructive interventions under a reliability constraint; and C6
is the integrative capstone over a provenance base. Chapters 5 and 6 then justified the methodological
philosophy — post-positivist, quantitative, ablative, reproducible, evidence-based, leakage-safe,
cross-domain, and reliability-first — and assembled it into a conceptual framework: a two-phase
audit-then-remediate workflow on a shared protocol, a coupled gap-structure, a question-dependency
hierarchy, a cumulative contribution-structure, and a pre-registered knowledge-generation cycle with
outcome-symmetric adjudication.

## 7.6 Transition to Volume II (Dataset Engineering)

Volume I has established *why* the research is needed, *what* problem it addresses, *which* objectives and
hypotheses it pursues, *how* (philosophically) it will proceed, and the *conceptual framework* that
organises it. It has deliberately withheld all implementation. The framework, however, imposes demanding
data-level commitments that must be operationalised before any model is trained: a genuinely leakage-safe,
group-aware partitioning with a sealed test; a deterministic, documented cross-taxonomy crosswalk with an
explicit exclusion set and reported excluded fractions; multi-domain external data spanning the preparation
axis; and exploratory and characterisation analyses sufficient to understand the data's composition, biases,
and class structure before modelling. **Volume II — Dataset Engineering** takes up exactly these commitments.
It will detail the datasets and their provenance, the construction and verification of the leakage-safe
grouped split, the design and validation of the label crosswalk, and the exploratory and class-characterisation
analyses that the framework requires — thereby converting the conceptual commitments of Volume I into the
concrete data foundation on which the methodology of subsequent volumes will be built. With the foundations
established, the monograph proceeds to that engineering.

## 7.7 Chapter summary

This concluding chapter consolidated Volume I. It restated the scientific motivation (a preventable disease,
a workforce bottleneck, and an evaluation culture that overstates readiness), the theoretical foundations
(cytology, taxonomy incompatibility, domain shift, leakage, and the reliability concepts), the seven coupled
research gaps, the seven specific objectives with their questions and pre-registered hypotheses, and the six
cumulative contributions, before motivating the transition to Volume II. Volume I thereby provides the
complete intellectual foundation — motivation, background, problem formulation, objectives, philosophy, and
conceptual framework — upon which the dataset engineering, methodology, experiments, and clinical discussion
of the subsequent volumes will be constructed.

---

# References

*Consolidated bibliography for Volume I. All entries are genuine, well-established works to the best of
the author's knowledge; no reference has been fabricated. Entries flagged **[verify]** require
confirmation of exact bibliographic details (and, for epidemiological sources, of current figures)
against the primary source before final submission. A machine-readable `references.bib` accompanies this
document.*

1. Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and
   distribution-free uncertainty quantification. *arXiv:2107.07511*. *[arXiv: no peer-reviewed venue;
   widely cited tutorial.]*
2. Azizi, S., Mustafa, B., Ryan, F., et al. (2021). Big self-supervised models advance medical image
   classification. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*. **[verify]**
3. Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). Multimodal machine learning: A survey and
   taxonomy. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(2), 423–443.
4. Buda, M., Maki, A., & Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in
   convolutional neural networks. *Neural Networks*, 106, 249–259.
5. Chen, C., Li, O., Tao, D., Barnett, A., Rudin, C., & Su, J. K. (2019). This looks like that: Deep
   learning for interpretable image recognition (ProtoPNet). *Advances in Neural Information Processing
   Systems (NeurIPS)*.
6. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning
   of visual representations (SimCLR). *Proceedings of the International Conference on Machine Learning
   (ICML)*.
7. Cour, T., Sapp, B., & Taskar, B. (2011). Learning from partial labels. *Journal of Machine Learning
   Research*, 12, 1501–1536. **[verify]**
8. Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on effective
   number of samples. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
   (CVPR)*.
9. European Commission, High-Level Expert Group on Artificial Intelligence. (2019). *Ethics guidelines
   for trustworthy AI*. Brussels: European Commission.
10. Frid-Adar, M., Diamant, I., Klang, E., Amitai, M., Goldberger, J., & Greenspan, H. (2018). GAN-based
    synthetic medical image augmentation for increased CNN performance in liver lesion classification.
    *Neurocomputing*, 321, 321–331. **[verify]**
11. Gebru, T., Morgenstern, J., Vecchione, B., et al. (2021). Datasheets for datasets. *Communications of
    the ACM*, 64(12), 86–92.
12. Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A.
    (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2, 665–673.
13. Ghassemi, M., Oakden-Rayner, L., & Beam, A. L. (2021). The false hope of current approaches to
    explainable artificial intelligence in health care. *The Lancet Digital Health*, 3(11), e745–e750.
14. Goodfellow, I., Pouget-Abadie, J., Mirza, M., et al. (2014). Generative adversarial nets. *Advances
    in Neural Information Processing Systems (NeurIPS)*.
15. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks.
    *Proceedings of the International Conference on Machine Learning (ICML)*.
16. Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution
    examples in neural networks. *Proceedings of the International Conference on Learning Representations
    (ICLR)*.
17. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network.
    *arXiv:1503.02531* (NeurIPS Deep Learning Workshop). *[arXiv: no peer-reviewed venue.]*
18. Howard, A. G., Zhu, M., Chen, B., et al. (2017). MobileNets: Efficient convolutional neural networks
    for mobile vision applications. *arXiv:1704.04861*. *[arXiv.]*
19. Jantzen, J., Norup, J., Dounias, G., & Bjerregaard, B. (2005). Pap-smear benchmark data for pattern
    classification (Herlev dataset). *Proceedings of NiSIS*. **[verify exact venue/pages]**
20. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based
    science. *Patterns*, 4(9), 100804.
21. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional
    neural networks. *Advances in Neural Information Processing Systems (NeurIPS)*.
22. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444.
23. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object
    detection. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.
24. Litjens, G., Kooi, T., Bejnordi, B. E., et al. (2017). A survey on deep learning in medical image
    analysis. *Medical Image Analysis*, 42, 60–88.
25. Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based out-of-distribution detection. *Advances
    in Neural Information Processing Systems (NeurIPS)*. **[verify]**
26. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions (SHAP).
    *Advances in Neural Information Processing Systems (NeurIPS)*.
27. Macenko, M., Niethammer, M., Marron, J. S., et al. (2009). A method for normalizing histology slides
    for quantitative analysis. *Proceedings of the IEEE International Symposium on Biomedical Imaging
    (ISBI)*.
28. Mitchell, M., Wu, S., Zaldivar, A., et al. (2019). Model cards for model reporting. *Proceedings of
    the Conference on Fairness, Accountability, and Transparency (FAT\*)*.
29. Mongan, J., Moy, L., & Kahn, C. E. (2020). Checklist for Artificial Intelligence in Medical Imaging
    (CLAIM): A guide for authors and reviewers. *Radiology: Artificial Intelligence*, 2(2), e200029.
30. Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V., & Herrera, F. (2012). A unifying
    view on dataset shift in classification. *Pattern Recognition*, 45(1), 521–530. **[verify]**
31. Nayar, R., & Wilbur, D. C. (Eds.). (2015). *The Bethesda System for Reporting Cervical Cytology:
    Definitions, Criteria, and Explanatory Notes* (3rd ed.). Springer.
32. Ovadia, Y., Fertig, E., Ren, J., et al. (2019). Can you trust your model's uncertainty? Evaluating
    predictive uncertainty under dataset shift. *Advances in Neural Information Processing Systems
    (NeurIPS)*.
33. Papanicolaou, G. N., & Traut, H. F. (1941). The diagnostic value of vaginal smears in carcinoma of
    the uterus. *American Journal of Obstetrics and Gynecology*, 42, 193–206.
34. Pineau, J., Vincent-Lamarre, P., Sinha, K., et al. (2021). Improving reproducibility in machine
    learning research (a report from the NeurIPS 2019 reproducibility program). *Journal of Machine
    Learning Research*, 22, 1–20.
35. Plissiti, M. E., Dimitrakopoulos, P., Sfikas, G., Nikou, C., Krikoni, O., & Charchanti, A. (2018).
    SIPaKMeD: A new dataset for feature and image based classification of normal and pathological cervical
    cells in Pap smear images. *Proceedings of the IEEE International Conference on Image Processing
    (ICIP)*.
36. Popper, K. R. (1959). *The Logic of Scientific Discovery*. Hutchinson.
37. Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (Eds.). (2009). *Dataset
    Shift in Machine Learning*. MIT Press.
38. Reinhard, E., Ashikhmin, M., Gooch, B., & Shirley, P. (2001). Color transfer between images. *IEEE
    Computer Graphics and Applications*, 21(5), 34–41.
39. Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use
    interpretable models instead. *Nature Machine Intelligence*, 1, 206–215.
40. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM:
    Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE
    International Conference on Computer Vision (ICCV)*.
41. Stacke, K., Eilertsen, G., Unger, J., & Lundström, C. (2021). Measuring domain shift for deep learning
    in histopathology. *IEEE Journal of Biomedical and Health Informatics*, 25(2), 325–336. **[verify]**
42. Stoler, M. H., & Schiffman, M. (2001). Interobserver reproducibility of cervical cytologic and
    histologic interpretations: Realistic estimates from the ASCUS-LSIL Triage Study (ALTS). *JAMA*,
    285(11), 1500–1505.
43. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks (Integrated
    Gradients). *Proceedings of the International Conference on Machine Learning (ICML)*.
44. Sung, H., Ferlay, J., Siegel, R. L., et al. (2021). Global Cancer Statistics 2020: GLOBOCAN estimates
    of incidence and mortality worldwide for 36 cancers in 185 countries. *CA: A Cancer Journal for
    Clinicians*, 71(3), 209–249. **[verify current figures against GLOBOCAN]**
45. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks.
    *Proceedings of the International Conference on Machine Learning (ICML)*.
46. Tellez, D., Litjens, G., Bándi, P., et al. (2019). Quantifying the effects of data augmentation and
    stain color normalization in convolutional neural networks for computational pathology. *Medical Image
    Analysis*, 58, 101544.
47. Topol, E. J. (2019). High-performance medicine: The convergence of human and artificial intelligence.
    *Nature Medicine*, 25, 44–56.
48. Varoquaux, G., & Cheplygina, V. (2022). Machine learning for medical imaging: Methodological failures
    and recommendations for the future. *npj Digital Medicine*, 5, 48.
49. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
50. Walboomers, J. M. M., Jacobs, M. V., Manos, M. M., et al. (1999). Human papillomavirus is a necessary
    cause of invasive cervical cancer worldwide. *The Journal of Pathology*, 189(1), 12–19.
51. Wang, J., Lan, C., Liu, C., et al. (2022). Generalizing to unseen domains: A survey on domain
    generalization. *IEEE Transactions on Knowledge and Data Engineering*. **[verify volume/pages]**
52. Wilkinson, M. D., Dumontier, M., Aalbersberg, I. J., et al. (2016). The FAIR Guiding Principles for
    scientific data management and stewardship. *Scientific Data*, 3, 160018.
53. World Health Organization. (2020). *Global strategy to accelerate the elimination of cervical cancer
    as a public health problem*. Geneva: WHO.
54. Zhang, L., Lu, L., Nogues, I., Summers, R. M., Liu, S., & Yao, J. (2017). DeepPap: Deep convolutional
    networks for cervical cell classification. *IEEE Journal of Biomedical and Health Informatics*, 21(6),
    1633–1643. **[verify]**
55. Zhou, K., Liu, Z., Qiao, Y., Xiang, T., & Loy, C. C. (2022). Domain generalization: A survey. *IEEE
    Transactions on Pattern Analysis and Machine Intelligence*. **[verify volume/pages]**
56. zur Hausen, H. (2002). Papillomaviruses and cancer: From basic studies to clinical application.
    *Nature Reviews Cancer*, 2(5), 342–350.

---

*End of Volume I — Foundations.*

