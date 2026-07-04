"""GATE — Mahalanobis three-tier abstention gate (RQ3, label-free deployment triage).

This is the POST-HOC abstention gate sitting on top of the FROZEN 5-class student. It turns three
already-measured, label-free signals into a per-cell routing decision:

    * maha_score      : class-conditional Mahalanobis OOD distance (eval.ood.MahalanobisScorer),
                        fit on SIPaKMeD TRAIN penultimate features. Higher == farther from every
                        in-distribution class == more out-of-distribution.
    * p_abnormal      : softmax mass on the abnormal classes (labelmap.p_abnormal). |p-0.5| small ==
                        the binary call is uncertain.
    * conf_set_size   : split-conformal (eval.conformal) prediction-set size. Larger == the model is
                        not sure which of the 5 morphologies it is, i.e. high epistemic ambiguity.

The gate is THREE-TIER (decide):
    FLAG     : maha_score >= a high SIPaKMeD-VAL Maha percentile (default 95th)  -> out-of-distribution,
               route to human review (the model's feature space has never seen this cell).
    DEFER    : not flagged, but the decision is uncertain -- |p_abnormal - 0.5| small OR conf_set_size
               large -> hand to a human for a second read.
    RELEASE  : everything else -- the model is confident AND in-distribution; emit the automated call.

ALL thresholds are fit on SIPaKMeD VAL ONLY (no external labels are ever used to set them), so the gate
is LABEL-FREE and deployable. The honest question this module answers (selective_metrics): does deferring
the highest-Maha cells actually RAISE the balanced accuracy of the cells we keep (RELEASE)? If yes the
gate has value; if released-accuracy is flat or falls as we defer, the gate is degenerate -- so we ALWAYS
report deferral-rate alongside released-accuracy so a degenerate gate cannot hide.

Pure-numpy decision logic (no torch / no sklearn here); the heavy feature/score extraction lives in
eval.ood + eval.conformal. This module is ADDITIVE -- it does not touch the 5-class train/eval path.
"""
import numpy as np

import config as C

RELEASE, DEFER, FLAG = "RELEASE", "DEFER", "FLAG"


# --------------------------------------------------------------------------- #
#  Thresholds (fit on SIPaKMeD VAL only -- label-free)
# --------------------------------------------------------------------------- #
def fit_thresholds(val_maha, val_p_abnormal=None, val_conf_set_size=None,
                   maha_pct=95.0, p_margin=0.15, set_size_max=None):
    """Derive the gate thresholds from SIPaKMeD-VAL signals (NO external labels).

    val_maha          : (Nv,) Mahalanobis scores on SIPaKMeD VAL.
    maha_pct          : FLAG everything at/above this VAL Maha percentile (default 95th).
    p_margin          : DEFER when |p_abnormal - 0.5| < p_margin (binary call too close to call).
    set_size_max      : DEFER when conf_set_size > set_size_max. If None and val_conf_set_size is
                        given, default to the 90th VAL percentile (rounded down, >=1); else 1.

    Returns a thresholds dict consumed by `decide`.
    """
    val_maha = np.asarray(val_maha, dtype=np.float64)
    maha_flag = float(np.percentile(val_maha, maha_pct)) if val_maha.size else float("inf")
    if set_size_max is None:
        if val_conf_set_size is not None and len(val_conf_set_size):
            set_size_max = int(max(1, np.floor(np.percentile(np.asarray(val_conf_set_size), 90.0))))
        else:
            set_size_max = 1
    return dict(maha_flag=maha_flag, maha_pct=float(maha_pct),
                p_margin=float(p_margin), set_size_max=int(set_size_max))


# --------------------------------------------------------------------------- #
#  Three-tier decision
# --------------------------------------------------------------------------- #
def decide(maha_score, p_abnormal, conf_set_size, thresholds):
    """Per-cell three-tier routing decision -> array of {RELEASE, DEFER, FLAG}.

    maha_score / p_abnormal / conf_set_size : (N,) arrays (conf_set_size may be None -> ignored).
    thresholds                              : dict from `fit_thresholds`.

    Precedence: FLAG (OOD) > DEFER (uncertain) > RELEASE. Vectorised; returns an (N,) object array.
    """
    maha = np.asarray(maha_score, dtype=np.float64)
    p = np.asarray(p_abnormal, dtype=np.float64)
    n = maha.shape[0]
    css = (np.asarray(conf_set_size, dtype=np.float64) if conf_set_size is not None
           else np.zeros(n, dtype=np.float64))

    flag = maha >= thresholds["maha_flag"]
    uncertain = (np.abs(p - 0.5) < thresholds["p_margin"]) | (css > thresholds["set_size_max"])

    out = np.full(n, RELEASE, dtype=object)
    out[uncertain & ~flag] = DEFER
    out[flag] = FLAG
    return out


# --------------------------------------------------------------------------- #
#  Selective (risk-coverage) metrics
# --------------------------------------------------------------------------- #
def _balanced_accuracy(y_true, y_pred):
    """Mean per-class recall over classes PRESENT in y_true (binary E0). NaN if a class is empty."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    recalls = []
    for c in np.unique(y_true):
        m = y_true == c
        recalls.append((y_pred[m] == c).mean())
    return float(np.mean(recalls)) if recalls else float("nan")


def _binary_ece(p_abn, y_bin, n_bins=15):
    """Binary ECE: top-confidence max(p,1-p) vs accuracy, equal-width bins (matches calibration.binary_ece)."""
    p = np.asarray(p_abn, float)
    y = np.asarray(y_bin, int)
    if p.size == 0:
        return float("nan")
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def selective_metrics(maha_score, p_abnormal, y_bin, budgets=(0.0, 0.10, 0.20, 0.40),
                      threshold=None):
    """Risk-coverage curve: DEFER the highest-Maha fraction, score the RELEASED remainder.

    For each deferral budget b, defer the b-fraction of cells with the LARGEST maha_score and keep the
    rest (the "released" subset). On the released subset report binary E0 balanced-accuracy + accuracy +
    ECE at the pre-registered P(abnormal) threshold, and the released/deferred counts. Deferral here is
    LABEL-FREE: the ranking is purely by maha_score, so this is what a deployed system could actually do.

    maha_score / p_abnormal / y_bin : (N,) external arrays (y_bin = binary E0 ground truth 0/1).
    budgets                         : deferral fractions to sweep.
    Returns {f"{int(b*100)}pct": {deferral_rate, released_n, balanced_accuracy, accuracy, ece, ...}}.
    """
    thr = float(C.ABNORMAL_THRESHOLD if threshold is None else threshold)
    maha = np.asarray(maha_score, dtype=np.float64)
    p = np.asarray(p_abnormal, dtype=np.float64)
    y = np.asarray(y_bin, dtype=int)
    n = maha.shape[0]
    order = np.argsort(-maha)                      # highest Maha first (deferred first)

    out = {}
    for b in budgets:
        n_def = int(np.floor(b * n))
        deferred = set(order[:n_def].tolist())
        keep = np.array([i not in deferred for i in range(n)], dtype=bool)
        yk, pk = y[keep], p[keep]
        pred = (pk >= thr).astype(int)
        out[f"{int(round(b * 100))}pct"] = dict(
            budget=float(b),
            deferral_rate=float(n_def / n) if n else 0.0,
            released_n=int(keep.sum()),
            deferred_n=int(n_def),
            balanced_accuracy=_balanced_accuracy(yk, pred) if keep.any() else float("nan"),
            accuracy=float((pred == yk).mean()) if keep.any() else float("nan"),
            ece=_binary_ece(pk, yk) if keep.any() else float("nan"),
            released_abnormal_rate=float(yk.mean()) if keep.any() else float("nan"),
        )
    return out


def flag_recall(maha_score, thresholds):
    """FLAG-recall on a whole-set-OOD domain: fraction of cells whose Maha >= the VAL FLAG threshold.

    For CRIC (every cell is out-of-distribution) and HiCervix non-squamous cells, the IDEAL gate FLAGs
    100% of them. This is the recall of the FLAG tier when the ground-truth answer is "flag everything".
    """
    maha = np.asarray(maha_score, dtype=np.float64)
    if maha.size == 0:
        return float("nan")
    return float((maha >= thresholds["maha_flag"]).mean())


def decision_rates(decisions):
    """Fraction of cells in each tier {RELEASE, DEFER, FLAG} (degenerate-gate visibility)."""
    d = np.asarray(decisions, dtype=object)
    n = d.shape[0]
    if n == 0:
        return dict(release=float("nan"), defer=float("nan"), flag=float("nan"), n=0)
    return dict(release=float((d == RELEASE).mean()), defer=float((d == DEFER).mean()),
                flag=float((d == FLAG).mean()), n=int(n))


# --------------------------------------------------------------------------- #
#  smoke test: synthetic Maha + p_abnormal -> released accuracy should RISE
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(C.SEED)
    N = 2000
    # construct a regime where high-Maha cells are the WRONG ones (the gate should help)
    maha = rng.gamma(2.0, 2.0, size=N)
    y = rng.integers(0, 2, size=N)
    # p_abnormal correct on low-Maha cells, noisy/flipped on high-Maha cells
    base = np.where(y == 1, 0.85, 0.15)
    noise = (maha - maha.mean()) / (maha.std() + 1e-9)        # high Maha -> more corruption
    p = np.clip(base - np.sign(base - 0.5) * 0.6 * np.clip(noise, 0, None) / 3.0, 0.01, 0.99)
    p = np.clip(p + rng.normal(0, 0.05, size=N), 0.01, 0.99)

    val_maha = rng.gamma(2.0, 2.0, size=500)
    thr = fit_thresholds(val_maha, maha_pct=95.0)
    dec = decide(maha, p, np.ones(N), thr)
    print("decision rates:", decision_rates(dec))

    sm = selective_metrics(maha, p, y)
    print("risk-coverage (defer highest-Maha):")
    for k, v in sm.items():
        print(f"  defer {k:>5}: rate={v['deferral_rate']:.2f} "
              f"released_bal_acc={v['balanced_accuracy']:.3f} ece={v['ece']:.3f} n={v['released_n']}")
    print("CRIC-style flag_recall (whole-set OOD):", round(flag_recall(maha + 6.0, thr), 3))
    # the gate has value iff deferring the highest-Maha cells does not LOWER released accuracy
    assert sm["40pct"]["balanced_accuracy"] >= sm["0pct"]["balanced_accuracy"] - 1e-9
    print("OK")
