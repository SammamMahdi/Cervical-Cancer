"""Model-pipeline bake-off for the NHANES HPV-risk tabular expert.

Answers "which pipeline works best for this data" empirically and honestly:
  * 10 model families, each in a LEAKAGE-SAFE sklearn Pipeline (impute/encode/scale fit INSIDE folds;
    native-missing GBDTs get the raw matrix). Calibration wrappers respect folds too.
  * TWO evaluation protocols:
      (1) repeated stratified 5-fold OOF  -> in-distribution AUROC / PR-AUC / Brier / ECE + bootstrap 95% CI
      (2) leave-one-cycle-out (LOCO)      -> temporal generalisation to a held-out survey cycle
        (this is the realistic deployment question: does a model trained on 2007-2014 hold up on 2015-16?)
  * Calibration study (raw vs isotonic vs sigmoid) + a screening operating point (sensitivity >= 0.80/0.90)
    for the winner.

Outputs -> <repo>/Tabular results/figures/nhanes_*  and  metrics/nhanes_pipeline_results.json
Run:  python nhanes_pipeline.py
"""
import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

import config as C
import nhanes_data as ND

FIGDIR = ND.FIG
METDIR = ND.MET
RNG = 42

from nhanes_data import NUM, CAT           # feature lists derived from the questionnaire spec


# ---------- preprocessing ----------
def prep(scale):
    num = [SimpleImputer(strategy="median")]
    if scale:
        num.append(StandardScaler())
    return ColumnTransformer([
        ("num", Pipeline([(f"s{i}", t) for i, t in enumerate(num)]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
    ])


def _try_lgbm():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                              random_state=RNG, verbose=-1)
    except Exception:
        return None


def _try_cat():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6,
                                  auto_class_weights="Balanced", random_seed=RNG, verbose=False)
    except Exception:
        return None


def _try_xgb(spw):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                             eval_metric="logloss", random_state=RNG, tree_method="hist")
    except Exception:
        return None


def _try_tabpfn():
    """Local TabPFN v2 (n<10k). Skips silently if weights are unavailable."""
    try:
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
        # smoke test so we fail fast (weights/license) rather than mid-CV
        rng = np.random.default_rng(0)
        clf.fit(rng.normal(size=(32, 3)), np.array([0, 1] * 16))
        clf.predict_proba(rng.normal(size=(4, 3)))
        return TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
    except Exception as e:
        print(f"  [TabPFN] unavailable ({type(e).__name__}) -> skipping")
        return None


def build_models(y):
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    m = {
        "LogReg":        (Pipeline([("p", prep(True)),  ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]), "impute"),
        "kNN":           (Pipeline([("p", prep(True)),  ("m", KNeighborsClassifier(n_neighbors=51))]), "impute"),
        "RandomForest":  (Pipeline([("p", prep(False)), ("m", RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample", random_state=RNG, n_jobs=-1))]), "impute"),
        "ExtraTrees":    (Pipeline([("p", prep(False)), ("m", ExtraTreesClassifier(n_estimators=600, min_samples_leaf=5, class_weight="balanced_subsample", random_state=RNG, n_jobs=-1))]), "impute"),
        "HistGB":        (HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, l2_regularization=1.0, class_weight="balanced", random_state=RNG), "native"),
    }
    if (lg := _try_lgbm()) is not None:   m["LightGBM"] = (lg, "native")
    if (cb := _try_cat()) is not None:    m["CatBoost"] = (cb, "native")
    if (xg := _try_xgb(spw)) is not None: m["XGBoost"] = (xg, "native")
    if (tp := _try_tabpfn()) is not None: m["TabPFN"] = (tp, "native")
    return m


# ---------- metrics ----------
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def boot_ci(y, p, fn, n=1000, seed=RNG):
    rng = np.random.default_rng(seed); vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(fn(y[b], p[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _Xin(X, kind):
    """Native models get the raw 12-col frame (cat as int codes, NaN kept); others get the frame for the
    ColumnTransformer inside the pipeline."""
    return X[NUM + CAT]


# ---------- evaluation ----------
def oof_predict(est, kind, X, y, splitter):
    Xin = _Xin(X, kind)
    oof = np.full(len(y), np.nan)
    for tr, te in splitter.split(Xin, y):
        est_c = _clone(est)
        est_c.fit(Xin.iloc[tr], y[tr])
        oof[te] = est_c.predict_proba(Xin.iloc[te])[:, 1]
    return oof


def _clone(est):
    from sklearn.base import clone
    try:
        return clone(est)
    except Exception:
        return est


def repeated_oof(est, kind, X, y, repeats=3):
    """Average OOF probabilities over `repeats` stratified 5-fold runs (stable estimate)."""
    accum = np.zeros(len(y)); cnt = np.zeros(len(y))
    for r in range(repeats):
        sk = StratifiedKFold(5, shuffle=True, random_state=RNG + r)
        oof = oof_predict(est, kind, X, y, sk)
        m = ~np.isnan(oof); accum[m] += oof[m]; cnt[m] += 1
    return accum / np.maximum(cnt, 1)


def loco(est, kind, X, y, cycles):
    """Leave-one-cycle-out OOF (temporal generalisation)."""
    Xin = _Xin(X, kind); oof = np.full(len(y), np.nan)
    per = {}
    for cyc in sorted(set(cycles)):
        te = (cycles == cyc).to_numpy(); tr = ~te
        est_c = _clone(est); est_c.fit(Xin[tr], y[tr])
        p = est_c.predict_proba(Xin[te])[:, 1]; oof[te] = p
        if len(np.unique(y[te])) > 1:
            per[cyc] = float(roc_auc_score(y[te], p))
    return oof, per


def main():
    print("Loading NHANES ...")
    X, y, feat, meta = ND.load(verbose=False)
    cycles = _load_cycles()
    print(f"  n={meta['n']} HPV+={meta['n_pos']} ({100*meta['pos_rate']:.1f}%)  features={len(feat)}")
    models = build_models(y)
    print(f"  models: {list(models)}")

    results = {}
    for name, (est, kind) in models.items():
        print(f"  [{name}] repeated-OOF ...", end="", flush=True)
        p = repeated_oof(est, kind, X, y)
        auc = roc_auc_score(y, p); ap = average_precision_score(y, p)
        br = brier_score_loss(y, p); ec = ece(y, p)
        lo, hi = boot_ci(y, p, roc_auc_score)
        print(f" AUROC={auc:.3f} [{lo:.3f},{hi:.3f}]  PR={ap:.3f}  ECE={ec:.3f}")
        _, per = loco(est, kind, X, y, cycles)
        loco_auc = float(np.mean(list(per.values()))) if per else float("nan")
        results[name] = dict(auroc=auc, auroc_ci=[lo, hi], pr_auc=ap, brier=br, ece=ec,
                             loco_auroc=loco_auc, loco_per_cycle=per)

    best = max(results, key=lambda k: results[k]["auroc"])
    print(f"\n  >>> best by in-distribution AUROC: {best} ({results[best]['auroc']:.3f})")

    # calibration study + operating point on the winner
    cal = calibration_study(models[best][0], models[best][1], X, y)
    op = operating_points(models[best][0], models[best][1], X, y)
    summary = dict(meta=meta, n_features=len(feat), features=feat, univariate=_univariate(X, y, feat),
                   models=results, best=best, calibration=cal, operating_points=op)
    json.dump(summary, open(os.path.join(METDIR, "nhanes_pipeline_results.json"), "w"), indent=2)

    make_figures(results, best, X, y, feat, cal, op, models)
    print(f"\n  saved metrics -> Tabular results/metrics/nhanes_pipeline_results.json")
    print(f"  saved figures -> Tabular results/figures/nhanes_*")
    return summary


def _load_cycles():
    """Cycle label per row, aligned to ND.load()'s row order (stored on X.attrs)."""
    X, y, feat, meta = ND.load(verbose=False)
    return pd.Series(X.attrs["cycle"])


def _univariate(X, y, feat):
    out = {}
    for c in feat:
        v = X[c].to_numpy(float); m = ~np.isnan(v)
        if m.sum() < 100 or len(np.unique(y[m])) < 2:
            continue
        a = roc_auc_score(y[m], v[m]); out[c] = float(max(a, 1 - a))
    return out


def calibration_study(est, kind, X, y):
    Xin = _Xin(X, kind); sk = StratifiedKFold(5, shuffle=True, random_state=RNG)
    res = {}
    for method in ["raw", "isotonic", "sigmoid"]:
        oof = np.full(len(y), np.nan)
        for tr, te in sk.split(Xin, y):
            if method == "raw":
                m = _clone(est); m.fit(Xin.iloc[tr], y[tr]); p = m.predict_proba(Xin.iloc[te])[:, 1]
            else:
                m = CalibratedClassifierCV(_clone(est), method=method, cv=3)
                m.fit(Xin.iloc[tr], y[tr]); p = m.predict_proba(Xin.iloc[te])[:, 1]
            oof[te] = p
        res[method] = dict(auroc=float(roc_auc_score(y, oof)), ece=float(ece(y, oof)),
                           brier=float(brier_score_loss(y, oof)))
    return res


def operating_points(est, kind, X, y):
    p = repeated_oof(est, kind, X, y)
    fpr, tpr, thr = roc_curve(y, p)
    out = {}
    for target in [0.80, 0.90]:
        i = np.where(tpr >= target)[0]
        if len(i):
            j = i[0]
            out[f"sens>={target}"] = dict(threshold=float(thr[j]), sensitivity=float(tpr[j]),
                                          specificity=float(1 - fpr[j]))
    return out


def make_figures(results, best, X, y, feat, cal, op, models):
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120, "savefig.dpi": 600,
                         "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, name + ".png"), bbox_inches="tight")
        fig.savefig(os.path.join(FIGDIR, name + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    # 1 univariate
    uni = _univariate(X, y, feat); items = sorted(uni.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#333")
    ax.axvline(0.5, color="k", lw=0.8, ls="--"); ax.set_xlabel("univariate AUROC")
    ax.set_title("NHANES: single-feature signal vs high-risk HPV"); save(fig, "nhanes_01_univariate")

    # 2 bake-off AUROC with CI
    order = sorted(results, key=lambda k: results[k]["auroc"])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    yv = np.arange(len(order))
    aur = [results[k]["auroc"] for k in order]
    lo = [results[k]["auroc"] - results[k]["auroc_ci"][0] for k in order]
    hi = [results[k]["auroc_ci"][1] - results[k]["auroc"] for k in order]
    ax.errorbar(aur, yv, xerr=[lo, hi], fmt="o", color="#111", ecolor="#888", capsize=3)
    ax.set_yticks(yv); ax.set_yticklabels(order); ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xlabel("OOF AUROC (repeated 5-fold, 95% CI)")
    ax.set_title("NHANES HPV-risk: model bake-off"); save(fig, "nhanes_02_bakeoff")

    # 3 leave-one-cycle-out for the best model
    per = results[best]["loco_per_cycle"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ks = sorted(per); ax.bar(ks, [per[k] for k in ks], color="#333")
    ax.axhline(0.5, color="k", lw=0.8, ls="--"); ax.set_ylim(0.5, max(0.8, max(per.values()) + 0.03))
    ax.set_ylabel("AUROC on held-out cycle"); ax.set_title(f"Temporal generalisation (LOCO) — {best}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); save(fig, "nhanes_03_loco")

    # 4 reliability for winner raw vs isotonic
    est, kind = models[best]; sk = StratifiedKFold(5, shuffle=True, random_state=RNG)
    Xin = _Xin(X, kind)
    fig, ax = plt.subplots(figsize=(5, 5))
    for method, col in [("raw", "#c00"), ("isotonic", "#060")]:
        oof = np.full(len(y), np.nan)
        for tr, te in sk.split(Xin, y):
            if method == "raw":
                m = _clone(est); m.fit(Xin.iloc[tr], y[tr]); oof[te] = m.predict_proba(Xin.iloc[te])[:, 1]
            else:
                m = CalibratedClassifierCV(_clone(est), method="isotonic", cv=3)
                m.fit(Xin.iloc[tr], y[tr]); oof[te] = m.predict_proba(Xin.iloc[te])[:, 1]
        edges = np.linspace(0, 1, 11); xs, ys = [], []
        for i in range(10):
            mm = (oof >= edges[i]) & (oof < edges[i + 1])
            if mm.sum() > 20:
                xs.append(oof[mm].mean()); ys.append(y[mm].mean())
        ax.plot(xs, ys, "o-", color=col, label=f"{method} (ECE={cal[method]['ece']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("predicted P(HPV+)"); ax.set_ylabel("observed")
    ax.legend(); ax.set_title(f"Reliability — {best}"); save(fig, "nhanes_04_reliability")

    # 5 PR curve + operating points
    from sklearn.metrics import precision_recall_curve
    p = repeated_oof(est, kind, X, y)
    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot(rec, prec, color="#111"); ax.axhline(y.mean(), color="k", ls="--", lw=0.8, label=f"prevalence {y.mean():.2f}")
    for k, v in op.items():
        ax.scatter([v["sensitivity"]], [_prec_at(p, y, v["threshold"])], s=40, zorder=5,
                   label=f"{k}: spec={v['specificity']:.2f}")
    ax.set_xlabel("recall (sensitivity)"); ax.set_ylabel("precision")
    ax.set_title(f"PR + operating points — {best} (AP={results[best]['pr_auc']:.3f})")
    ax.legend(fontsize=8); save(fig, "nhanes_05_pr_operating")


def _prec_at(p, y, thr):
    pred = p >= thr
    return float(y[pred].mean()) if pred.sum() else float("nan")


if __name__ == "__main__":
    main()
