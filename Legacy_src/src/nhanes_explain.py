"""Explainability for the frozen NHANES tabular expert — SHAP (attribution) + EBM shape functions
("what the model believes"). Complements the discrimination/calibration story with interpretability.

  * SHAP (TreeExplainer on the frozen monotonic LightGBM): global mean-|SHAP| importance, beeswarm,
    and dependence plots for the top exposure features.
  * EBM glass-box: per-feature response (shape) functions — the logit contribution vs feature value,
    directly readable as the model's believed risk relationship.

Outputs -> Tabular results/NHANES/figures/explain_*  and  metrics/nhanes_explain.json
Run:  python nhanes_explain.py
"""
import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
from sklearn.impute import SimpleImputer

import nhanes_data as ND
from nhanes_diagnostics import lgbm_mono, base_cols
FIG, MET = ND.FIG, ND.MET
RNG = 42


def _save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, name + "." + ext), bbox_inches="tight")
    plt.close(fig)


def main():
    X, y, feat, meta = ND.load(target="high_risk", verbose=False)
    cols = base_cols(feat)
    Xc = X[cols].to_numpy(float)
    print(f"n={len(y)} features={len(cols)}  — fitting frozen monotonic LightGBM for explanation")
    model = lgbm_mono(cols); model.fit(Xc, y)

    # ---------- SHAP ----------
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(Xc)
    if isinstance(sv, list):                     # [neg, pos] -> positive class
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                             # (n, f, classes)
        sv = sv[:, :, 1]
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(mean_abs)[::-1]
    top = [cols[i] for i in order[:15]]
    print("top SHAP features:", ", ".join(f"{c}={mean_abs[cols.index(c)]:.3f}" for c in top[:6]))

    # global importance bar
    fig, ax = plt.subplots(figsize=(6, 5))
    idx = order[:15][::-1]
    ax.barh([cols[i] for i in idx], mean_abs[idx], color="#1a1a1a")
    ax.set_xlabel("mean |SHAP| (log-odds)"); ax.set_title("NHANES expert — global SHAP importance")
    _save(fig, "explain_shap_importance")

    # beeswarm
    plt.figure(figsize=(7, 5.5))
    shap.summary_plot(sv, X[cols], feature_names=cols, show=False, max_display=15)
    fig = plt.gcf(); fig.suptitle("SHAP beeswarm — direction & magnitude", y=1.01, fontsize=11)
    _save(fig, "explain_shap_beeswarm")

    # dependence plots for the 4 strongest features
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for ax, fi in zip(axs.ravel(), order[:4]):
        v = Xc[:, fi]; m = ~np.isnan(v)
        ax.scatter(v[m], sv[m, fi], s=5, alpha=0.3, color="#1a6")
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_xlabel(cols[fi]); ax.set_ylabel("SHAP (log-odds)")
        lo, hi = np.nanpercentile(v, [1, 99]); ax.set_xlim(lo, hi)
    fig.suptitle("SHAP dependence — top exposure features (monotonic where constrained)", y=1.0)
    _save(fig, "explain_shap_dependence")

    # ---------- EBM shape functions ----------
    ebm_shapes = ebm_explain(Xc, y, cols)

    json.dump(dict(shap_importance={cols[i]: float(mean_abs[i]) for i in order},
                   ebm_top=ebm_shapes),
              open(os.path.join(MET, "nhanes_explain.json"), "w"), indent=2)
    print("saved -> figures/explain_*  and  metrics/nhanes_explain.json")


def ebm_explain(Xc, y, cols):
    from interpret.glassbox import ExplainableBoostingClassifier
    Xi = SimpleImputer(strategy="median").fit_transform(Xc)
    ebm = ExplainableBoostingClassifier(feature_names=cols, interactions=0, outer_bags=8,
                                        random_state=RNG, n_jobs=-1)
    ebm.fit(Xi, y)
    ge = ebm.explain_global()
    names = list(ebm.term_names_)
    importances = np.asarray(ge.data()["scores"], float)
    order = np.argsort(importances)[::-1]
    top_idx = [i for i in order if ge.data(i).get("type") in ("univariate", "continuous", None)][:6]

    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    top = {}
    for ax, i in zip(axs.ravel(), top_idx):
        d = ge.data(i); xs = np.asarray(d["names"], float); ys = np.asarray(d["scores"], float)
        xc = (xs[:-1] + xs[1:]) / 2 if len(xs) == len(ys) + 1 else xs[:len(ys)]
        ax.plot(xc, ys, drawstyle="steps-mid", color="#1a1a1a", lw=1.8)
        ax.axhline(0, color="#c0392b", lw=0.8, ls="--")
        ax.set_title(names[i], fontsize=10); ax.set_xlabel(names[i]); ax.set_ylabel("logit contribution")
        top[names[i]] = float(importances[i])
    fig.suptitle("EBM glass-box shape functions — the model's believed risk relationships", y=1.0, fontsize=12)
    _save(fig, "explain_ebm_shapes")
    print("EBM top terms:", ", ".join(f"{names[i]}" for i in top_idx))
    return top


if __name__ == "__main__":
    main()
