"""MEASURE — the Mahalanobis three-tier abstention gate on the FROZEN baseline student (RQ3).

Post-hoc, no training. Pipeline:
  1. FIT (SIPaKMeD TRAIN, label-free for the gate's purpose -- TRAIN labels only build the class
     Gaussians):  MahalanobisScorer (per-class means + shared LedoitWolf cov) on penultimate features.
  2. FIT (SIPaKMeD VAL only): the gate thresholds -- 95th-pct Maha FLAG threshold + conformal qhat
     (eval.conformal, APS) for the prediction-set size. NO external labels are ever used to set
     thresholds, so the gate is LABEL-FREE / deployable.
  3. For each external (herlev / apacc / hicervix / cric) reuse the SAME eval loaders as
     run_ood_mahalanobis.main: extract penultimate feats -> Maha score; predict -> p_abnormal +
     conformal set size; apply gate.decide; compute selective (risk-coverage) external balanced-acc /
     accuracy / ECE at deferral budgets {0,10,20,40%} (defer the highest-Maha fraction).
  4. FLAG-recall on whole-set-OOD domains: CRIC (every cell OOD) + HiCervix NON-SQUAMOUS cells
     (glandular/microbe -- truly outside the 5 squamous morphologies).

Honesty: the headline is whether deferring the highest-Maha cells RAISES the released-subset balanced
accuracy. deferral-rate is always reported next to released-accuracy so a degenerate gate is visible.

Writes results/metrics/abstention_baseline.json. ADDITIVE -- does not touch the 5-class train/eval path.
"""
import os, json
import numpy as np
from torch.utils.data import DataLoader, Subset

import config as C
import run
import eval.metrics as M
import eval.ood as OOD
import eval.conformal as CONF
import eval.gate as GATE
from data.dataset import CellDataset, build_transform, fit_stain
from data.splits import get_split
from data import labelmap as LM

MAX_N = 2000
SEED = C.SEED
BUDGETS = (0.0, 0.10, 0.20, 0.40)


def _subset_idx(n, max_n):
    if n <= max_n:
        return None
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def _loader(ds, num_workers, batch_size=C.BATCH_SIZE):
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False)


def _feats_probs_labels(model, ds, num_workers, max_n=None, desc="ext"):
    """One subset, two passes (features for Maha; softmax probs for p_abnormal + conformal sets).

    Subsamples to <= max_n with the SAME seeded indices for both passes so Maha/probs/labels align.
    Returns (feats [n,D], probs [n,K], y_bin [n]).
    """
    idx = _subset_idx(len(ds), max_n) if max_n else None
    sub = Subset(ds, idx.tolist()) if idx is not None else ds
    ld = _loader(sub, num_workers)
    feats, y_f = OOD.extract_features(model, ld, desc=f"{desc} feats")
    y, probs = M.predict(model, ld, desc=f"{desc} probs")
    y = np.asarray(y).astype(int)
    y_bin = (y >= 1).astype(int)            # external tier ints already E0 {0,1}; >=1 -> abnormal
    return feats, probs, y_bin


def main(which="student", num_workers=2, stain="none", img_size=C.IMG_SIZE,
         alpha=0.1, maha_pct=95.0, p_margin=0.15):
    model = run._load_model(which)

    split = get_split("grouped")
    stain_norm = fit_stain(stain, split=split)
    tf_eval = build_transform(train=False, img_size=img_size, stain_norm=stain_norm)

    # ---- (1) FIT MahalanobisScorer on SIPaKMeD TRAIN penultimate features ----
    train_ds = CellDataset(split["train"], tf_eval)
    f_tr, y_tr = OOD.extract_features(model, _loader(train_ds, num_workers), desc="sipakmed train (fit)")
    scorer = OOD.MahalanobisScorer().fit(f_tr, y_tr)
    print(f"[fit] train feats={f_tr.shape} classes={scorer.classes_.tolist()} "
          f"LedoitWolf shrinkage={scorer.shrinkage_:.4f}")

    # ---- (2) FIT gate thresholds on SIPaKMeD VAL ONLY (label-free) ----
    val_ds = CellDataset(split["val"], tf_eval)
    val_ld = _loader(val_ds, num_workers)
    f_val, y_val = OOD.extract_features(model, val_ld, desc="sipakmed val feats")
    yv, probs_val = M.predict(model, val_ld, desc="sipakmed val probs")
    s_val = scorer.score(f_val)
    # conformal qhat (APS, marginal) on VAL 5-class probs vs VAL true labels
    qhat = CONF.fit_conformal(probs_val, yv, alpha=alpha, method="aps", mondrian=False)
    val_sets = CONF.predict_sets(probs_val, qhat, method="aps", mondrian=False)
    val_set_size = np.array([len(s) for s in val_sets], dtype=float)
    thr = GATE.fit_thresholds(s_val, val_p_abnormal=LM.p_abnormal(probs_val),
                              val_conf_set_size=val_set_size, maha_pct=maha_pct, p_margin=p_margin)
    print(f"[fit] VAL n={s_val.size} maha_p{int(maha_pct)}={thr['maha_flag']:.3f} "
          f"conformal qhat={qhat:.4f} set_size_max={thr['set_size_max']} p_margin={thr['p_margin']}")

    # internal VAL reference selective curve (sanity: should be near-flat, already in-distribution)
    val_decisions = GATE.decide(s_val, LM.p_abnormal(probs_val), val_set_size, thr)

    # ---- (3) externals: SAME wiring as run_ood_mahalanobis ----
    tf = build_transform(train=False, img_size=img_size, stain_norm=fit_stain(stain, split_kind="grouped"))

    def _build_ext(name):
        if name == "herlev":
            return M.FolderExternal(C.HERLEV_DIR, LM.herlev_map, tf, tier="E0")
        if name == "apacc":
            return M.FolderExternal(C.APACC_DIR, LM.apacc_map, tf, tier="E0")
        if name == "hicervix":
            return M.HiCervixExternal(tf, tier="E0")
        if name == "cric":
            from data.external_tier2 import cric_dataset
            return cric_dataset(tier="E0", transform=tf)
        raise ValueError(name)

    domains = {}
    for name in ("herlev", "apacc", "hicervix", "cric"):
        try:
            ds = _build_ext(name)
            total = len(ds)
            feats, probs, y_bin = _feats_probs_labels(model, ds, num_workers, max_n=MAX_N, desc=name)
            maha = scorer.score(feats)
            p_abn = LM.p_abnormal(probs)
            sets = CONF.predict_sets(probs, qhat, method="aps", mondrian=False)
            css = np.array([len(s) for s in sets], dtype=float)

            decisions = GATE.decide(maha, p_abn, css, thr)
            rates = GATE.decision_rates(decisions)
            selective = GATE.selective_metrics(maha, p_abn, y_bin, budgets=BUDGETS)
            frecall = GATE.flag_recall(maha, thr)

            domains[name] = dict(
                n=int(maha.size), n_total_available=int(total),
                abnormal_rate=float(y_bin.mean()),
                maha_mean=float(maha.mean()), maha_median=float(np.median(maha)),
                decision_rates=rates,
                selective=selective,
                flag_recall_wholeset=frecall,
            )
            r0 = selective["0pct"]; r40 = selective["40pct"]
            print(f"[ext] {name:9s} n={maha.size:5d}/{total:6d}  flag={rates['flag']:.2f} "
                  f"defer={rates['defer']:.2f}  balacc 0%->40%: {r0['balanced_accuracy']:.3f}"
                  f"->{r40['balanced_accuracy']:.3f}  flag_recall={frecall:.3f}")
        except Exception as e:
            domains[name] = dict(error=f"{type(e).__name__}: {e}")
            print(f"[ext {name}] FAILED: {type(e).__name__}: {e}")

    # ---- (4) HiCervix NON-SQUAMOUS FLAG-recall (truly outside the 5 squamous morphologies) ----
    nonsq = {}
    try:
        ns_ds = M.HiCervixNonSquamous(tf)
        # keep only the positive (non-squamous: glandular/microbe) cells
        pos_idx = [i for i, lab in enumerate(ns_ds.labels) if lab == 1]
        if pos_idx:
            sub_idx = _subset_idx(len(pos_idx), MAX_N)
            use = [pos_idx[i] for i in sub_idx] if sub_idx is not None else pos_idx
            ns_sub = Subset(ns_ds, use)
            f_ns, _ = OOD.extract_features(model, _loader(ns_sub, num_workers), desc="hicervix nonsq feats")
            maha_ns = scorer.score(f_ns)
            nonsq = dict(n=int(maha_ns.size), n_total_pos=int(len(pos_idx)),
                         maha_mean=float(maha_ns.mean()),
                         flag_recall=GATE.flag_recall(maha_ns, thr))
            print(f"[nonsq] hicervix non-squamous n={maha_ns.size}/{len(pos_idx)} "
                  f"maha_mean={maha_ns.mean():.3f} flag_recall={nonsq['flag_recall']:.3f}")
        else:
            nonsq = dict(n=0, note="no non-squamous positive cells found")
    except Exception as e:
        nonsq = dict(error=f"{type(e).__name__}: {e}")
        print(f"[nonsq] FAILED: {type(e).__name__}: {e}")

    out = dict(
        which=which, max_n=MAX_N, seed=SEED,
        gate="Mahalanobis three-tier abstention (FLAG=high-Maha OOD, DEFER=uncertain, RELEASE=confident)",
        thresholds=dict(maha_flag=thr["maha_flag"], maha_pct=thr["maha_pct"],
                        p_margin=thr["p_margin"], set_size_max=thr["set_size_max"],
                        conformal_qhat=float(qhat), conformal_alpha=float(alpha)),
        thresholds_fit_on="SIPaKMeD VAL only (label-free for externals)",
        ledoitwolf_shrinkage=float(scorer.shrinkage_), feature_dim=int(f_tr.shape[1]),
        sipakmed_val=dict(n=int(s_val.size), decision_rates=GATE.decision_rates(val_decisions),
                          maha_mean=float(s_val.mean()), maha_p95=float(thr["maha_flag"])),
        budgets=list(BUDGETS),
        external=domains,
        hicervix_nonsquamous_flag_recall=nonsq,
        cric_flag_recall=domains.get("cric", {}).get("flag_recall_wholeset"),
    )
    os.makedirs(C.METRICDIR, exist_ok=True)
    outpath = os.path.join(C.METRICDIR, "abstention_baseline.json")
    with open(outpath, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("\n==== SELECTIVE (released balanced-acc) vs deferral budget ====")
    print(f"{'domain':10s} {'0%':>8s} {'10%':>8s} {'20%':>8s} {'40%':>8s}  flag_recall")
    for name, d in domains.items():
        if "selective" not in d:
            print(f"{name:10s}  (error: {d.get('error')})"); continue
        s = d["selective"]
        row = "  ".join(f"{s[f'{int(b*100)}pct']['balanced_accuracy']:.3f}" for b in BUDGETS)
        print(f"{name:10s} {row}   fr={d['flag_recall_wholeset']:.3f}")
    print(f"CRIC flag_recall={out['cric_flag_recall']}  "
          f"HiCervix-nonsq flag_recall={nonsq.get('flag_recall')}")
    print(f"wrote {outpath}")
    return out


if __name__ == "__main__":
    main()
