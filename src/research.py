"""Predefined research framing for the NEW pipeline (single source of truth).

This pins down---*before any run*---the research gaps in the cervical-cytology deep-learning
literature that this pipeline closes, the falsifiable hypotheses that predict each method *works*,
and the resulting contributions. `run.py eda` (and the notebook) dump this to
results/metrics/research_framing.json so every figure/number traces back to the gap it addresses.

Framing rules:
  * Each RQ is anchored to a published literature gap, not to any prior in-house model.
  * Each hypothesis predicts the proposed method SUCCEEDS against a concrete, pre-registered target.
    We do not pre-register experiments designed to fail; a predicted failure is instead a design
    requirement the method must satisfy.

Grounded in: plans/PIPELINE_UPGRADE_PLAN.md (U1--U9), plans/DATA_PLAN.md,
plans/LABEL_ALIGNMENT_METHOD.md (Routes A/B/C), plans/MULTIMODAL_PLAN.md,
plans/Ablation and Evaluation plan.md.
"""
import os
import config as C
import runlog

# ---------------------------------------------------------------------------
# Gaps in the cervical-cytology DL literature that this pipeline targets.
# ---------------------------------------------------------------------------
RESEARCH_GAPS = [
    dict(id="G1", title="Non-reproducible, leakage-prone evaluation",
         gap="Cytology classifiers are routinely reported at near-ceiling accuracy under random "
             "per-cell/per-FOV splits that leak same-slide cells across train and test, and splits/"
             "seeds are seldom released. There is no widely-used leakage-safe, reproducible benchmark."),
    dict(id="G2", title="Single-dataset training; cross-domain generalization unsolved",
         gap="Models are trained and tested on one collection prepared under one stain/segmentation "
             "protocol; robustness across slide-preparation domains (conventional smear, liquid-based, "
             "auto-segmented) is largely unmeasured and unaddressed."),
    dict(id="G3", title="Overconfidence; no uncertainty or abstention",
         gap="Cytology DL models output uncalibrated, overconfident probabilities with no principled "
             "abstention, which is unsafe for screening triage where a trustworthy operating point and "
             "'defer to expert' option are required."),
    dict(id="G4", title="Fragmented, incompatible label taxonomies",
         gap="SIPaKMeD morphology, Herlev dysplasia and Bethesda grades do not share a label space; "
             "naive one-to-many cross-mapping is clinically unsafe, which blocks learning from multiple "
             "datasets and honest cross-dataset evaluation."),
    dict(id="G5", title="Severe imbalance; rare high-grade / minority classes missed",
         gap="High-grade lesions and rare morphologies are heavily under-represented, so models "
             "under-detect exactly the clinically important minority classes."),
    dict(id="G6", title="Accuracy-efficiency trade-off blocks point-of-care use",
         gap="High-accuracy models are too heavy for low-resource / point-of-care screening; compact "
             "models that retain accuracy are needed but rarely validated head-to-head."),
    dict(id="G7", title="Black-box, image-only predictions limit clinical trust",
         gap="Predictions are rarely accompanied by faithful, quantified explanations or complementary "
             "morphometric evidence, limiting cytopathologist trust and adoption."),
]

# ---------------------------------------------------------------------------
# Research questions: each closes a gap; each hypothesis predicts SUCCESS with a
# concrete pre-registered target. `where` names the code that delivers the method.
# ---------------------------------------------------------------------------
RESEARCH_QUESTIONS = [
    dict(
        id="RQ1", label="A leakage-safe, reproducible benchmark", gap="G1",
        question="Can a cluster-grouped, stratified 80/10/10 split with a sealed test and fully "
                 "released seeds/splits provide a reproducible cervical-cytology benchmark whose "
                 "in-distribution score reflects genuine within-dataset generalization?",
        hypothesis="H1: the grouped, sealed-test protocol yields a reproducible benchmark whose "
                   "5-class metrics (balanced acc, macro-F1, OvR-AUROC) are stable across seeds within "
                   "their 95% bootstrap CIs and are reproducible bit-for-bit from the released split.",
        success_metric="bootstrap-CI-stable metrics across >=3 seeds; split/seed-reproducible; zero cluster overlap.",
        where="data/splits.py, eval/metrics.py (bootstrap_ci), run.evaluate (sealed test)",
    ),
    dict(
        id="RQ2", label="Domain-generalizable classifier", gap="G2",
        question="Do stain normalization (U4), self-supervised pretraining (SSL), and domain-aware "
                 "augmentation produce a classifier that generalizes across slide-preparation domains?",
        hypothesis="H2: the domain-generalization stack achieves external binary balanced accuracy and "
                   "AUROC clearly above the majority baseline and above the un-normalized backbone on "
                   "conventional-smear and liquid-based targets, narrowing the internal->external gap.",
        success_metric="external balanced-acc and AUROC above majority baseline AND above the no-stain/no-SSL arm; "
                        "positive gap reduction vs baseline.",
        where="data/stain.py, selfsup.py, ablation.py (A4 stain, A9 SSL-init), run.evaluate (prep-stratified)",
    ),
    dict(
        id="RQ3", label="Calibrated, risk-aware predictions with abstention", gap="G3",
        question="Do per-domain temperature scaling, conformal prediction, and an OOD/energy gate "
                 "deliver trustworthy confidence and safe abstention in- and cross-distribution?",
        hypothesis="H3: temperature scaling yields low ECE in- and cross-distribution; split-conformal "
                   "sets meet their target coverage (e.g. 90%) on the sealed test; the OOD gate flags "
                   "external shift so high-confidence errors are abstained rather than emitted.",
        success_metric="post-scaling ECE below pre; conformal empirical coverage >= 1-alpha; OOD AUROC>0.5 with "
                        "abstention improving selective accuracy.",
        where="eval/calibration.py, eval/conformal.py, route_b.py (OOD gate), run.evaluate",
    ),
    dict(
        id="RQ4", label="Learning across incompatible taxonomies", gap="G4",
        question="Can a deterministic many-to-one crosswalk (Route A) plus partial-label learning "
                 "(Routes B/C) enable safe cross-taxonomy evaluation and multi-dataset training without "
                 "the dangerous one-to-many projection?",
        hypothesis="H4: the deterministic crosswalk gives a clinically safe, fully-documented E0/E1 "
                   "evaluation with a reported excluded fraction; partial-label multi-dataset training "
                   "(Routes B/C) improves external performance over single-dataset training.",
        success_metric="100% deterministic many-to-one coverage with reported excluded fraction; Routes B/C "
                        "external balanced-acc >= single-dataset baseline.",
        where="data/labelmap.py (Route A), route_b.py, route_c.py, eval/metrics.py",
    ),
    dict(
        id="RQ5", label="Rescuing rare and minority classes", gap="G5",
        question="Does conditional synthesis (CellGAN) of under-represented morphology classes improve "
                 "minority-class recall while preserving calibration?",
        hypothesis="H5: training with CellGAN-synthesised minority samples improves minority-class recall "
                   "and balanced accuracy at equal-or-better ECE versus class-weighting/focal alone.",
        success_metric="minority-class recall and balanced acc up vs A6 (weights/focal) at ECE no worse.",
        where="cellgan.py, ablation.py (synthetic-augment arm), eval/metrics.py",
    ),
    dict(
        id="RQ6", label="Deployable, architecture-agnostic accuracy", gap="G6",
        question="Can knowledge distillation produce a compact student that matches larger backbones and "
                 "published cervical-cytology SOTA (CCanNet/MaxCerVixT/CerviXpert) under one protocol?",
        hypothesis="H6: the distilled B0 student matches or exceeds its teacher and is within a small "
                   "margin of the best backbone/SOTA in-distribution at a fraction of the params/latency, "
                   "under the identical leakage-safe protocol.",
        success_metric="student >= teacher in-distribution; within small delta of best backbone/SOTA at much "
                        "lower params/FLOPs/latency.",
        where="models/kd.py, models/backbones.py, models/sota.py, ablation.model_comparison(), eval/efficiency.py",
    ),
    dict(
        id="RQ7", label="Explainable, multimodal decisions", gap="G7",
        question="Does late fusion with tabular morphometric features improve accuracy, and does the "
                 "explanation suite (Grad-CAM/IG/SHAP/ProtoPNet) provide faithful, cell-localised evidence?",
        hypothesis="H7: late fusion gives a measurable in-distribution lift over the image-only model, and "
                   "attributions localise to nuclear/cytoplasmic structure with quantified faithfulness "
                   "(deletion/insertion), giving clinically meaningful evidence.",
        success_metric="fusion balanced-acc >= image-only; attribution deletion/insertion faithfulness reported; "
                        "ProtoPNet prototypes class-consistent.",
        where="explain/features.py (late fusion), explain/{cam,attribution}.py, models/protopnet.py",
    ),
]

# ---------------------------------------------------------------------------
# Contributions (solutions that close the gaps above).
# ---------------------------------------------------------------------------
CONTRIBUTIONS = [
    dict(id="C1", title="Leakage-safe, fully-reproducible cytology benchmark",
         text="A cluster-grouped, stratified, sealed-test split with released seeds/splits and "
              "bootstrap/McNemar/Wilcoxon significance, giving a trustworthy in-distribution benchmark."),
    dict(id="C2", title="Domain-generalizable classifier across preparation types",
         text="A stain-normalised, SSL-pretrained, domain-aware model that improves cross-dataset "
              "(conventional/LBC/auto-segmented) performance over un-normalised backbones."),
    dict(id="C3", title="Calibrated, risk-aware deployment",
         text="Per-domain temperature scaling + split/Mondrian conformal prediction + an OOD gate that "
              "together give trustworthy confidence and safe abstention under domain shift."),
    dict(id="C4", title="Safe cross-taxonomy learning and evaluation",
         text="A deterministic many-to-one crosswalk plus partial-label routes (B/C) that enable "
              "clinically safe multi-dataset training and honest E0/E1 evaluation with reported coverage."),
    dict(id="C5", title="Minority-class rescue via conditional synthesis",
         text="CellGAN-based augmentation that raises rare/minority morphology recall at maintained calibration."),
    dict(id="C6", title="Deployable + explainable + full provenance",
         text="A distilled compact model benchmarked against published SOTA, with a faithful "
              "explanation suite, multimodal fusion, and every value/figure persisted as JSON/CSV."),
]

# Pre-registered evaluation commitments (so the protocol is fixed before results exist).
PROTOCOL = dict(
    split="cluster-grouped 80/10/10, stratified by class, seed=%d, sealed test" % C.SEED,
    train_val="training and early stopping use TRAIN+VAL only",
    internal_test="all in-distribution metrics are reported on the SEALED 10%% test split",
    external="frozen model -> E0 (binary), E1 (grade), E-morph (candidate-set); excluded fraction reported",
    primary_metrics=["balanced accuracy", "macro-F1", "OvR-AUROC", "ECE", "conformal coverage"],
    operating_point="pre-registered tau=0.5 for E0; per-domain temperature scaling for calibrated reporting",
    significance=["95% bootstrap CI", "McNemar", "Wilcoxon", ">=3 seeds"],
    provenance="every value -> results/metrics/*.json|csv ; every figure -> companion *.csv|json",
)


def framing():
    return dict(research_gaps=RESEARCH_GAPS, research_questions=RESEARCH_QUESTIONS,
                contributions=CONTRIBUTIONS, protocol=PROTOCOL)


def dump(path=None):
    path = path or os.path.join(C.METRICDIR, "research_framing.json")
    runlog.dump_json(path, framing())
    return path


if __name__ == "__main__":
    p = dump()
    fr = framing()
    print(f"Wrote {p}")
    print(f"{len(fr['research_gaps'])} gaps, {len(fr['research_questions'])} RQs, "
          f"{len(fr['contributions'])} contributions.")
    for rq in fr["research_questions"]:
        print(f"  {rq['id']} ({rq['label']}) <- {rq['gap']}: {rq['hypothesis'][:64]}...")
