"""End-to-end orchestrator for the new pipeline (the thin driver; notebook calls these).

Subcommands:
  eda       -> cluster/split report + class-balance figures + label-map manifest
  train     -> train.main (teacher two-stage -> distilled student) on the leakage-safe split
  evaluate  -> sealed-test in-dist (5-class) + cross-dataset E0/E1 + temperature calibration + efficiency
  all       -> eda -> train -> evaluate
"""
import os, json, argparse
import numpy as np
import torch

import config as C
import runlog
import plotting as P
from data.dataset import make_loaders, build_transform, fit_stain
from data.splits import make_split, split_report
from data.clustering import load_cells, summarize
import data.labelmap as LM
from models.backbones import build
import eval.metrics as M
import eval.calibration as CAL
import eval.efficiency as EFF


def _load_model(which="student", backbone=None):
    import ckpt as CK
    cfg = C.TEACHER if which.startswith("teacher") else C.STUDENT
    bb = backbone or cfg["backbone"]
    m = build(bb, C.NUM_CLASSES, cfg["hidden"], cfg["dropout"], pretrained=False).to(C.DEVICE)
    meta = CK.load_into(m, which, map_location=C.DEVICE)   # latest checkpoint (root or run_*/ snapshot)
    print(f"[run] loaded {which} <- {meta.get('path')}")
    m.eval(); return m


@torch.no_grad()
def _logits(model, loader, desc="logits"):
    from tqdm.auto import tqdm
    ys, ls = [], []
    for x, y in tqdm(loader, desc=desc, leave=False, unit="batch"):
        x = x.to(C.DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(C.DEVICE == "cuda")):
            ls.append(model(x).float().cpu().numpy())
        ys.append(np.asarray(y))
    return np.concatenate(ys), np.concatenate(ls)


def _e0_metrics(y5, probs5):
    """Internal binary (E0) metrics from the 5-class SIPaKMeD test predictions."""
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix)
    p_abn = LM.p_abnormal(probs5)
    pred = (p_abn >= C.ABNORMAL_THRESHOLD).astype(int)
    true = np.isin(y5, C.ABNORMAL_IDX).astype(int)
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    return dict(accuracy=float(accuracy_score(true, pred)),
                balanced_accuracy=M.balanced_accuracy(true, pred),
                macro_f1=float(f1_score(true, pred, average="macro", zero_division=0)),
                sensitivity=float(tp / (tp + fn) if (tp + fn) else 0.0),
                specificity=float(tn / (tn + fp) if (tn + fp) else 0.0),
                auroc=float(roc_auc_score(true, p_abn)) if len(set(true)) > 1 else None)


def eda():
    # pre-register the research framing (RQs / hypotheses / contributions / protocol)
    import research
    research.dump()
    # SIPaKMeD cluster + leakage-safe split report (training source)
    cells = load_cells(); summ = summarize(cells)
    split = make_split(); rep = split_report(split)
    runlog.dump_json(os.path.join(C.METRICDIR, "eda_summary.json"),
                     {"clusters": summ, "split": rep, "labelmap": LM.manifest()})
    P.class_balance(summ["cells_per_class"], os.path.join(C.FIGDIR, "eda_class_balance"),
                    title="SIPaKMeD class balance (cells)")
    P.class_balance(summ["clusters_per_class"], os.path.join(C.FIGDIR, "eda_cluster_balance"),
                    ylabel="clusters", title="SIPaKMeD clusters per class")
    # ALL-dataset EDA (SIPaKMeD + every external): counts / coverage / size / corrupt / montages
    import data.eda as EDA
    print("All-dataset EDA:")
    all_rec = EDA.run_all()
    # CLASS CHARACTERIZATION: what defines each 5-class morphology and each binary group
    cls = None
    try:
        import class_analysis
        cls = class_analysis.run()
    except Exception as e:
        print(f"[class-analysis] skipped: {e}")
    print(json.dumps(rep, indent=2))
    return {"sipakmed_split": rep, "all_datasets": all_rec, "class_analysis": cls}


def evaluate(which="student", backbone=None, split_kind="grouped", img_size=C.IMG_SIZE, stain="none",
             num_workers=C.NUM_WORKERS):
    model = _load_model(which, backbone)
    _, val_ld, test_ld, _ = make_loaders(split_kind=split_kind, img_size=img_size, stain=stain,
                                         num_workers=num_workers)

    # ---- in-distribution, SEALED test (5-class) ----
    y, probs = M.predict(model, test_ld)
    md = M.indist_metrics(y, probs)
    ece, bc, ba = M.expected_calibration_error(probs, y); md["ece"] = ece
    md["e0_internal"] = _e0_metrics(y, probs)
    # 95% bootstrap CIs on the sealed test (Ablation §1.3)
    from sklearn.metrics import accuracy_score, f1_score
    pred = probs.argmax(1)
    md["accuracy_ci95"] = M.bootstrap_ci(y, pred, accuracy_score)
    md["macro_f1_ci95"] = M.bootstrap_ci(y, pred, lambda a, b: f1_score(a, b, average="macro", zero_division=0))
    runlog.dump_json(os.path.join(C.METRICDIR, f"indist_{which}.json"), md)
    P.confusion_matrix(md["confusion_matrix"], C.CLASSES, os.path.join(C.FIGDIR, f"cm_{which}"),
                       title=f"Confusion — {which} (sealed test)")
    from sklearn.metrics import roc_curve, roc_auc_score
    oh = np.eye(C.NUM_CLASSES)[y]; curves = {}
    for ci, cl in enumerate(C.CLASSES):
        fpr, tpr, _ = roc_curve(oh[:, ci], probs[:, ci]); curves[cl] = (fpr, tpr, roc_auc_score(oh[:, ci], probs[:, ci]))
    P.roc_curves(curves, os.path.join(C.FIGDIR, f"roc_{which}"), title=f"ROC OvR — {which}")
    if np.isfinite(bc).any():
        P.reliability(bc, ba, os.path.join(C.FIGDIR, f"reliability_{which}"),
                      title=f"Reliability — {which} (ECE={ece:.3f})", label=which)

    # ---- calibration (temperature scaling fit on VAL, applied to test) ----
    yv, lv = _logits(model, val_ld, desc="calib val"); yt, lt = _logits(model, test_ld, desc="calib test")
    calrep = CAL.calibrate_and_report(lv, yv, lt, yt)
    runlog.dump_json(os.path.join(C.METRICDIR, f"calibration_{which}.json"), calrep)

    # ---- cross-dataset: E0 (binary) + E1 (grade) + E-morph (candidate-set hit) ----
    tf = build_transform(train=False, img_size=img_size, stain_norm=fit_stain(stain, split_kind=split_kind))
    ext = {  # name -> (kind, root, mapper)
        "herlev":   ("folder", C.HERLEV_DIR, LM.herlev_map),
        "apacc":    ("folder", C.APACC_DIR, LM.apacc_map),
        "hicervix": ("csv", None, None),
        "cric":     ("point", None, None),     # Tier-2 FOV+point-annotation (DATA_PLAN §4.2)
    }
    def _build(name, kind, root, mapper, tier):
        if kind == "folder":
            return M.FolderExternal(root, mapper, tf, tier=tier)
        if kind == "csv":
            return M.HiCervixExternal(tf, tier=tier)
        from data.external_tier2 import cric_dataset
        return cric_dataset(tier=tier, transform=tf)
    cross = {"E0": {}, "E1": {}, "Emorph": {}}
    for name, (kind, root, mapper) in ext.items():
        try:
            for tier in ("E0", "E1"):
                cross[tier][name] = M.cross_dataset_eval(name, _build(name, kind, root, mapper, tier), model,
                                                         tier=tier, num_workers=num_workers)
            if kind != "point":               # E-morph supports folder/CSV loaders (CRIC point-crops skipped)
                cross["Emorph"][name] = M.emorph_eval(name, model, tf, num_workers=num_workers)
            P.confusion_matrix(cross["E0"][name]["confusion_matrix"], ["Normal", "Abnormal"],
                               os.path.join(C.FIGDIR, f"cm_cross_{name}"), title=f"Cross {name} — E0 ({which})")
        except Exception as e:
            print(f"[cross {name}] skipped: {e}")
    delta = M.internal_external_delta(md["e0_internal"], cross["E0"])

    # ---- preparation-stratified read (§4): conventional smear vs LBC (preparation = dominant shift) ----
    PREP = {"conventional": ["herlev", "apacc", "cric"], "lbc": ["hicervix"]}
    prep = {}
    for g, names in PREP.items():
        vals = [cross["E0"][n]["accuracy"] for n in names if n in cross["E0"]]
        if vals:
            prep[g] = float(sum(vals) / len(vals))

    # ---- efficiency ----
    eff = EFF.profile(model, which)

    summary = {"indist": md, "calibration": calrep, "cross": cross,
               "internal_external_delta": delta, "prep_stratified_E0": prep,
               "stain": stain, "img_size": img_size, "efficiency": eff}
    runlog.dump_json(os.path.join(C.METRICDIR, f"eval_summary_{which}.json"), summary)
    print(f"[{which}] 5-class acc={md['accuracy']:.4f} macroF1={md['macro_f1']:.4f} | "
          f"E0 internal acc={md['e0_internal']['accuracy']:.4f} | ECE {calrep.get('ece_pre')}->{calrep.get('ece_post')}")
    for name in cross["E0"]:
        e0 = cross["E0"][name]; e1 = cross["E1"].get(name, {}); em = cross["Emorph"].get(name, {})
        print(f"  {name}: E0 acc={e0.get('accuracy'):.3f} sens={e0.get('sensitivity'):.3f} "
              f"spec={e0.get('specificity'):.3f} excl={e0.get('excluded_fraction'):.2f} | "
              f"E1 HG-recall={e1.get('highgrade_recall')} | E-morph acc={em.get('accuracy')}")
    return summary


def evaluate8(tag="student_partiallabel", backbone="efficientnet_b0", img_size=C.IMG_SIZE,
              stain="none", num_workers=C.NUM_WORKERS, save=True):
    """Cross-dataset E0/E2 evaluation for an 8-LEAF (Bethesda) checkpoint (Wave-2b).

    Loads an 8-logit head (labelmap.N_LEAVES8) from the `{tag}` checkpoint and scores it on the
    SAME external loaders as the 5-class `evaluate()` (Herlev/APACC FolderExternal, HiCervix CSV,
    CRIC point-crops). The 8-leaf head does NOT change the external GROUND TRUTH; only the decision
    score (P(abnormal)=labelmap.p_abnormal8, mass on leaves {3,4,5,7}). Produces:
      1. cross_E0  : per-dataset acc/balacc/sens/spec/auroc/ece at C.ABNORMAL_THRESHOLD.
      2. E2        : HiCervix non-squamous detection AUROC (pos=AGC|microbe, score=mass on {5,6,7}).
      3. coverage  : excluded fraction BEFORE (5-class None) vs AFTER (8-leaf candidate) per dataset.
      4. shared_E0 : E0 on cells representable under BOTH maps (coverage gain != accuracy control).
    Writes results/metrics/eval8_<tag>.json. Does NOT touch the 5-class evaluate path.
    """
    import ckpt as CK
    model = build(backbone, LM.N_LEAVES8, C.STUDENT["hidden"], C.STUDENT["dropout"],
                  pretrained=False).to(C.DEVICE)
    state, meta, path = CK.load(tag, map_location=C.DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[eval8] loaded 8-leaf {tag} <- {path}  (n_leaves={LM.N_LEAVES8})")

    tf = build_transform(train=False, img_size=img_size, stain_norm=fit_stain(stain))

    def _ext_e0(name):
        if name == "herlev":   return M.FolderExternal(C.HERLEV_DIR, LM.herlev_map, tf, tier="E0")
        if name == "apacc":    return M.FolderExternal(C.APACC_DIR, LM.apacc_map, tf, tier="E0")
        if name == "hicervix": return M.HiCervixExternal(tf, tier="E0")
        from data.external_tier2 import cric_dataset
        return cric_dataset(tier="E0", transform=tf)

    # ---- (1) cross-dataset E0 with the 8-leaf P(abnormal) ----
    cross_e0 = {}
    for name in ("herlev", "apacc", "hicervix", "cric"):
        try:
            ds = _ext_e0(name)
            cross_e0[name] = M.cross_dataset_eval8(name, ds, model, num_workers=num_workers, save=save)
            r = cross_e0[name]
            print(f"  [E0-8] {name}: acc={r['accuracy']:.3f} balacc={r['balanced_accuracy']:.3f} "
                  f"sens={r['sensitivity']:.3f} spec={r['specificity']:.3f} auroc={r['auroc']}")
        except Exception as e:
            print(f"[eval8 E0 {name}] skipped: {e}")

    # ---- (2) E2 non-squamous detection (HiCervix) ----
    e2 = {}
    try:
        e2 = M.nonsquamous_eval8(model, tf, num_workers=num_workers, save=save)
        print(f"  [E2] non-squamous AUROC={e2.get('auroc')} "
              f"(pos={e2.get('n_pos')} neg={e2.get('n_neg')} excl={e2.get('excluded')})")
    except Exception as e:
        print(f"[eval8 E2] skipped: {e}")

    # ---- (3) excluded fraction: 5-class crosswalk vs 8-leaf taxonomy ----
    coverage = {}
    try:
        coverage = M.excluded_fraction_5_vs_8(save=save)
        for n, c in coverage.items():
            print(f"  [coverage] {n}: excl_before={c['frac_before']:.3f} -> excl_after={c['frac_after']:.3f}")
    except Exception as e:
        print(f"[eval8 coverage] skipped: {e}")

    # ---- (4) shared-cell E0 control (cells representable under BOTH maps) ----
    shared_e0 = {}
    def _shared_ds(name):
        if name == "herlev":   return M.SharedCellFolderE0(C.HERLEV_DIR, LM.herlev_map, LM.herlev8_candidates, tf)
        if name == "apacc":    return M.SharedCellFolderE0(C.APACC_DIR, LM.apacc_map, LM.apacc8_candidates, tf)
        if name == "hicervix": return M.SharedCellHiCervixE0(tf)
        return M.cric_shared_e0_dataset(tf)
    for name in ("herlev", "apacc", "hicervix", "cric"):
        try:
            ds = _shared_ds(name)
            shared_e0[name] = M.cross_dataset_eval8(f"{name}_shared", ds, model, num_workers=num_workers, save=False)
            r = shared_e0[name]
            print(f"  [shared-E0] {name}: n={r['n']} acc={r['accuracy']:.3f} auroc={r['auroc']}")
        except Exception as e:
            print(f"[eval8 shared {name}] skipped: {e}")

    summary = dict(tag=tag, checkpoint=path, n_leaves=LM.N_LEAVES8, backbone=backbone,
                   abnormal_leaves=sorted(LM._ABNORMAL8), nonsquamous_leaves=[5, 6, 7],
                   threshold=float(C.ABNORMAL_THRESHOLD),
                   cross_E0=cross_e0, E2_nonsquamous=e2, coverage=coverage, shared_E0=shared_e0,
                   img_size=img_size, stain=stain)
    if save:
        runlog.dump_json(os.path.join(C.METRICDIR, f"eval8_{tag}.json"), summary)
        print(f"[eval8] wrote {os.path.join(C.METRICDIR, f'eval8_{tag}.json')}")
    return summary


def per_domain_calibration(which="student", backbone=None, img_size=C.IMG_SIZE, stain="none",
                           num_workers=C.NUM_WORKERS):
    """C: per-domain operating-point + score recalibration on P(abnormal). For the sealed internal test
    and each external dataset, report fixed-tau=0.5 vs ORACLE-threshold accuracy (the recoverable gap)
    and binary ECE before/after a Platt fit. The oracle threshold and Platt fit use each domain's own
    labels => an UPPER BOUND (a deployed system calibrates on a small labelled target sample). Writes
    per_domain_calibration.json + a recovery-bar figure."""
    import eval.calibration as CAL
    from torch.utils.data import DataLoader
    model = _load_model(which, backbone)
    domains = {}
    # internal sealed test
    _, _vl, test_ld, _ = make_loaders(img_size=img_size, stain=stain, num_workers=num_workers)
    y, probs = M.predict(model, test_ld, desc="sipakmed test")
    domains["sipakmed_internal"] = (LM.p_abnormal(probs), np.isin(y, C.ABNORMAL_IDX).astype(int))
    # externals (E0), same wiring as evaluate()
    tf = build_transform(train=False, img_size=img_size, stain_norm=fit_stain(stain))
    def _ext_ds(name):
        if name == "herlev":   return M.FolderExternal(C.HERLEV_DIR, LM.herlev_map, tf, tier="E0")
        if name == "apacc":    return M.FolderExternal(C.APACC_DIR, LM.apacc_map, tf, tier="E0")
        if name == "hicervix": return M.HiCervixExternal(tf, tier="E0")
        from data.external_tier2 import cric_dataset
        return cric_dataset(tier="E0", transform=tf)
    for name in ("herlev", "apacc", "hicervix", "cric"):
        try:
            ds = _ext_ds(name)
            yb, pr = M.predict(model, DataLoader(ds, batch_size=C.BATCH_SIZE, num_workers=num_workers), desc=name)
            domains[name] = (LM.p_abnormal(pr), np.asarray(yb).astype(int))
        except Exception as e:
            print(f"[calib {name}] skipped: {e}")
    report = {n: CAL.recalibrate_binary(p, yb) for n, (p, yb) in domains.items()}
    runlog.dump_json(os.path.join(C.METRICDIR, "per_domain_calibration.json"),
                     {"which": which, "note": "oracle/Platt tuned on each domain's own labels = upper bound",
                      "domains": report})
    # plot BALANCED-accuracy recovery (consistent with the oracle objective, which maximizes balanced acc)
    P.bars({n: round(r["balacc_oracle"] - r["balacc_tau50"], 4) for n, r in report.items()},
           os.path.join(C.FIGDIR, "calib_threshold_recovery"), ylabel="balanced-acc recovered",
           title="Per-domain balanced-acc recoverable by re-thresholding (oracle - tau=0.5)")
    print("Per-domain operating point (acc tau=0.5 -> oracle | ECE pre->post):")
    for n, r in report.items():
        print(f"  {n:18s} acc {r['acc_tau50']:.3f} -> {r['acc_oracle']:.3f} (tau*={r.get('tau_oracle')}) | "
              f"ECE {r['ece_pre']:.3f} -> {r['ece_post']:.3f}")
    return report


def cric_crop_sweep(which="student", backbone=None, crops=None, img_size=C.IMG_SIZE, stain="none"):
    """CRIC crop-size sensitivity (RQ4 robustness): evaluate CRIC E0 at several point-crop sizes with the
    frozen model. CRIC is a dense preparation (nuclei ~70px apart), so small crops magnify the nucleus
    while large crops include neighbours; this sweep makes that trade-off explicit instead of fixing one
    arbitrary size. Writes cric_crop_sensitivity.json/.csv + a figure. Needs a trained model."""
    from data.external_tier2 import cric_dataset, cric_geometry
    geom = cric_geometry(save=True)
    model = _load_model(which, backbone)
    tf = build_transform(train=False, img_size=img_size, stain_norm=fit_stain(stain))
    crops = crops or C.CRIC_CROP_SWEEP
    rows = []
    for c in crops:
        ds = cric_dataset(tier="E0", crop=c, transform=tf)
        r = M.cross_dataset_eval("cric", ds, model, tier="E0")
        rows.append(dict(crop=c, n=ds.n, excluded=ds.excluded, accuracy=r.get("accuracy"),
                         balanced_accuracy=r.get("balanced_accuracy"), auroc=r.get("auroc"),
                         sensitivity=r.get("sensitivity"), specificity=r.get("specificity")))
        print(f"  crop={c:3d}px  n={ds.n:5d}  acc={r.get('accuracy'):.3f}  "
              f"bal={r.get('balanced_accuracy'):.3f}  auroc={r.get('auroc')}")
    import math
    valid = [r for r in rows if r["auroc"] is not None and not math.isnan(r["auroc"])]
    best = (max(valid, key=lambda d: d["auroc"]) if valid
            else max(rows, key=lambda d: (d["balanced_accuracy"] or -1)))
    out = {"geometry": geom, "sweep": rows, "best_by_auroc": best, "default_crop": C.CRIC_CROP}
    runlog.dump_json(os.path.join(C.METRICDIR, "cric_crop_sensitivity.json"), out)
    xs = [r["crop"] for r in rows]
    P.line_plot(xs, {"auroc": [r["auroc"] for r in rows], "balanced_accuracy": [r["balanced_accuracy"] for r in rows]},
                os.path.join(C.FIGDIR, "cric_crop_sensitivity"), xlabel="point-crop size (px)",
                ylabel="E0 metric", title="CRIC crop-size sensitivity", baseline=0.5)
    print(f"[cric sweep] best AUROC at crop={best['crop']}px (auroc={best['auroc']}); "
          f"median NN={geom['nn_distance_px']['median']:.0f}px; default={C.CRIC_CROP}px")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eda", "train", "evaluate", "all", "cric-sweep"])
    ap.add_argument("--which", default="student")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.cmd in ("eda", "all"): eda()
    if args.cmd in ("train", "all"):
        import train; train.main(smoke=args.smoke)
    if args.cmd in ("evaluate", "all"): evaluate(args.which)
    if args.cmd == "cric-sweep": cric_crop_sweep(args.which)


if __name__ == "__main__":
    main()
