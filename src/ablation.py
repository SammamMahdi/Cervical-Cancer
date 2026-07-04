"""Ablation runner (Ablation & Evaluation plan §2). Each arm toggles ONE factor, all else fixed;
trains (resumable, isolated checkpoints) then evaluates under the standard protocol, and collects
in-dist macro-F1 + mean external E0 + the generalization gap into one comparison table.

Plumbed arms:
  A1  split protocol     : grouped (baseline) vs random/leaky        -> quantifies leakage inflation
  A2  knowledge distill. : fixed-KD (baseline) vs scratch vs buggy-KD
  A5  augmentation       : photometric (baseline) vs none vs geometric
  A6  imbalance handling : class-weights (baseline) vs focal vs none
  A8  backbone           : B0 (baseline) vs MobileNetV3 vs ResNet50
Covered elsewhere: A3 resolution (set C.IMG_SIZE), A4 stain (data/stain.py — wire into transform),
A7 calibration (already reported per run as ece_pre vs ece_post by run.evaluate).

Each arm is a FULL training run — long. Resumable; re-run to continue. Run a subset via
run_all(arms=[...]).  CLI:  python ablation.py --arms A2_scratch A2_buggy_kd
"""
import os, argparse
import numpy as np

import config as C
import runlog
import plotting as P
import train
import run

ARMS = {
    "A1_random_split":  dict(split_kind="random"),
    "A2_scratch":       dict(distill=False),
    "A2_buggy_kd":      dict(kd_mode="buggy"),
    "A3_res160":        dict(img_size=160, teacher_img_size=160),  # clean low-res ablation (teacher too)
    "A4_macenko":       dict(stain="macenko"),
    "A4_reinhard":      dict(stain="reinhard"),
    "A5_aug_none":      dict(aug="none"),
    "A5_aug_geometric": dict(aug="geometric"),
    "A6_focal":         dict(imbalance="focal"),
    "A6_none":          dict(imbalance="none"),
    "A6_oversample":    dict(imbalance="oversample"),
    "A8_mobilenet_v3":  dict(student_backbone="mobilenet_v3_large"),
    "A8_resnet50":      dict(student_backbone="resnet50"),
    "A8_swin_t":        dict(student_backbone="swin_t"),
    "A8_maxvit_t":      dict(student_backbone="maxvit_t"),
    "A8_shufflenet_v2": dict(student_backbone="shufflenet_v2_x1_0"),
    "A8_densenet121":   dict(student_backbone="densenet121"),
    "A9_ssl_init":      dict(ssl_init=True),     # needs selfsup.pretrain() run first -> ssl_<bb>.pt
}

# A8/§3 model-comparison family (trained on the same split, evaluated under the same protocol)
COMPARISON_BACKBONES = ["efficientnet_b0", "efficientnet_b2", "mobilenet_v3_large",
                        "shufflenet_v2_x1_0", "resnet50", "densenet121", "swin_t", "maxvit_t"]


def _collect(name, summ):
    ext = summ["cross"]["E0"]
    mean_ext = float(np.mean([v["accuracy"] for v in ext.values()])) if ext else None
    e0_int = summ["indist"]["e0_internal"]["accuracy"]
    return dict(arm=name,
                indist_acc=summ["indist"]["accuracy"],
                indist_macro_f1=summ["indist"]["macro_f1"],
                e0_internal=e0_int,
                mean_external_e0=mean_ext,
                generalization_gap=(e0_int - mean_ext) if mean_ext is not None else None,
                ece_pre=summ["calibration"].get("ece_pre"),
                ece_post=summ["calibration"].get("ece_post"))


def run_baseline():
    """The full pipeline as-is (the main student.pt) — the reference row."""
    return _collect("baseline_fullpipe", run.evaluate(which="student"))


def run_arm(name, kwargs, smoke=False):
    sfx = "_" + name
    train.main(tag_suffix=sfx, smoke=smoke, **kwargs)
    summ = run.evaluate(which=f"student{sfx}",
                        backbone=kwargs.get("student_backbone"),
                        split_kind=kwargs.get("split_kind", "grouped"),
                        img_size=kwargs.get("img_size", C.IMG_SIZE),
                        stain=kwargs.get("stain", "none"))
    return _collect(name, summ)


def run_all(arms=None, smoke=False, include_baseline=True):
    rows = []
    if include_baseline:
        try: rows.append(run_baseline())
        except Exception as e: print(f"[baseline] skipped: {e}")
    for name in (arms or list(ARMS)):
        try:
            rows.append(run_arm(name, ARMS[name], smoke))
        except Exception as e:
            print(f"[{name}] failed: {e}")
        runlog.dump_json(os.path.join(C.METRICDIR, "ablation_table.json"), rows)
    f1 = {r["arm"]: r["indist_macro_f1"] for r in rows if r.get("indist_macro_f1") is not None}
    if f1:
        P.bars(f1, os.path.join(C.FIGDIR, "ablation_macro_f1"),
               ylabel="in-dist macro-F1", title="Ablation — in-dist macro-F1")
    print("Saved results/metrics/ablation_table.json + figures/ablation_macro_f1")
    return rows


def model_comparison(backbones=None, smoke=False, out="model_comparison.json"):
    """§3 apples-to-apples comparison: train each backbone on the SAME grouped split, evaluate under
    the SAME protocol; collect in-dist macro-F1 + mean external E0 + efficiency. Saves <out>."""
    import eval.efficiency as EFF
    from models.backbones import build
    rows = []
    for bb in (backbones or COMPARISON_BACKBONES):
        try:
            r = run_arm(f"cmp_{bb}", dict(student_backbone=bb), smoke=smoke)
            try:
                r["efficiency"] = EFF.profile(build(bb, C.NUM_CLASSES, pretrained=False), bb)
            except Exception as e:
                print(f"[cmp_{bb}] efficiency skipped: {e}")
            rows.append(r)
        except Exception as e:
            print(f"[cmp_{bb}] failed: {e}")
        runlog.dump_json(os.path.join(C.METRICDIR, out), rows)
    return rows


def sota_comparison(smoke=False):
    """RQ6: published-SOTA reimplementations (CCanNet / CerviXpert / MaxCerVixT) trained and evaluated
    under the IDENTICAL leakage-safe protocol as the backbones. Saves sota_comparison.json."""
    from models.sota import SOTA_MODELS
    return model_comparison(backbones=list(SOTA_MODELS), smoke=smoke, out="sota_comparison.json")


# ---------------------------------------------------------------------------
# A10 — CytoBridge taxonomy-bridged partial-label arms (Route-C+, G4/RQ4/C4 + G2/RQ2/C2)
# A SEPARATE entry point from ARMS/run_arm (those call train.main); these call route_c.train_partial.
# Each arm trains a shared encoder with PartialLabelLoss over SIPaKMeD singletons + external candidate
# sets, then reports sealed SIPaKMeD 5-class metrics (the 8-leaf head collapsed to its 5 squamous leaves)
# + the taxonomy coverage (how many external cells became representable).
# ---------------------------------------------------------------------------
_ALL_EXT = ("herlev", "apacc", "hicervix", "cric")
ROUTE_C_ARMS = {
    # A10a SANITY: 8-leaf head, NO external -> just SIPaKMeD singletons; must NOT regress 5-class (RQ6).
    "A10a_8leaf_sipakmed_only": dict(taxonomy="bethesda8", external_sources=()),
    # A10b SPINE: 8-leaf head + ALL external partial labels, externals down-weighted (G4/RQ4 + G2/RQ2).
    "A10b_8leaf_multisource":   dict(taxonomy="bethesda8", external_sources=_ALL_EXT,
                                     w_src={"sipakmed": 1.0, "external": 0.3}),
    # A10c CONTROL: old 5-leaf partial-label multi-source (drops glandular/microbe/high-grade) — isolates
    #              the value of the 3 new non-squamous nodes vs partial-labeling alone.
    "A10c_5leaf_multisource":   dict(taxonomy="sipakmed5", external_sources=_ALL_EXT),
    # A10d w_src sweep point (stronger external weight) — chosen on VAL only in the full study.
    "A10d_8leaf_wsrc05":        dict(taxonomy="bethesda8", external_sources=_ALL_EXT,
                                     w_src={"sipakmed": 1.0, "external": 0.5}),
    # A11 SCALE-AUG: A10b spine + train-time scale jitter (aug='photometric+scale'). The scale-stress
    #               gate showed the frozen student collapses 0.97->0.50 under 0.3x-3x rescale; this arm
    #               forces scale-invariance so apparent cell size stops driving the abnormal prediction.
    "A11_scale_aug":            dict(taxonomy="bethesda8", external_sources=_ALL_EXT,
                                     w_src={"sipakmed": 1.0, "external": 0.3},
                                     aug="photometric+scale"),
}


def run_partial_arm(name, kwargs, smoke=False, epochs=None):
    """Train one Route-C+ arm and collect sealed-test 5-class metrics + taxonomy coverage."""
    import route_c as RC
    ep = (2 if smoke else epochs)
    res = RC.train_partial(tag=f"student_{name}", epochs=ep, save=not smoke, **kwargs)
    st = res["sealed_test"]; ds = res["dataset"]
    return dict(arm=name, taxonomy=res["config"]["taxonomy"], n_leaves=res["config"]["n_leaves"],
                indist_acc=st["accuracy"], indist_balanced_acc=st["balanced_accuracy"],
                indist_macro_f1=st["macro_f1"], cohen_kappa=st.get("cohen_kappa"),
                n_sipakmed=ds["n_sipakmed_singleton"], n_external=ds["n_external"],
                external_per_source=ds["external_per_source"])


def run_route_c(arms=None, smoke=False, epochs=None):
    """Run the A10 taxonomy-bridged partial-label arms; saves route_c_arms.json + a macro-F1 bar."""
    rows = []
    for name in (arms or list(ROUTE_C_ARMS)):
        try:
            rows.append(run_partial_arm(name, ROUTE_C_ARMS[name], smoke=smoke, epochs=epochs))
        except Exception as e:
            print(f"[{name}] failed: {e}")
        runlog.dump_json(os.path.join(C.METRICDIR, "route_c_arms.json"), rows)
    f1 = {r["arm"]: r["indist_macro_f1"] for r in rows if r.get("indist_macro_f1") is not None}
    if f1:
        P.bars(f1, os.path.join(C.FIGDIR, "route_c_macro_f1"),
               ylabel="sealed-test macro-F1", title="Route-C+ (A10) — sealed SIPaKMeD 5-class")
    print("Saved results/metrics/route_c_arms.json + figures/route_c_macro_f1")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None, help="subset of arm names (default: all)")
    ap.add_argument("--no-baseline", dest="baseline", action="store_false")
    ap.add_argument("--compare", action="store_true", help="run the §3 model-comparison instead of ablations")
    ap.add_argument("--sota", action="store_true", help="run the RQ6 published-SOTA comparison")
    ap.add_argument("--route-c", dest="route_c", action="store_true",
                    help="run the A10 taxonomy-bridged partial-label arms (CytoBridge spine)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.route_c:
        run_route_c(arms=a.arms, smoke=a.smoke)
    elif a.sota:
        sota_comparison(smoke=a.smoke)
    elif a.compare:
        model_comparison(smoke=a.smoke)
    else:
        run_all(arms=a.arms, smoke=a.smoke, include_baseline=a.baseline)
