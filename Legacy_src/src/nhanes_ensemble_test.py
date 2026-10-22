"""Does a hybrid / ensemble model help on NHANES HPV? — a controlled test.

Ensembling only helps when base models make DECORRELATED errors. Our top models are all tree ensembles on
the same 12 features, so this measures (a) how correlated their out-of-fold predictions are and (b) whether
any fusion strategy beats the best single model:

  base learners : LightGBM, XGBoost, CatBoost, RandomForest (tuned)  +  LogisticRegression, kNN (diverse/weaker)
  fusions       : soft-vote (trees) · soft-vote (all) · soft-vote (diverse trio) · rank-average ·
                  stacking (LR meta, cross-fitted) · in-sample optimal convex weights (upper bound)

Outputs -> Tabular results/NHANES/figures/nhanes_ensemble_*  and  metrics/nhanes_ensemble_test.json
Run:  python nhanes_ensemble_test.py
"""
import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

import nhanes_data as ND
import nhanes_pipeline as NP
from nhanes_pipeline import NUM, CAT, prep
RNG = 42
MET, FIG = ND.MET, ND.FIG


def tuned():
    """Rebuild the 4 tuned tree models from the HPO results, plus 2 diverse weaker learners."""
    p = json.load(open(os.path.join(MET, "nhanes_hpo_results.json")))["models"]
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    m = {}
    m["LightGBM"] = LGBMClassifier(**p["LightGBM"]["best_params"], class_weight="balanced",
                                   random_state=RNG, verbose=-1, n_jobs=-1)
    m["XGBoost"] = XGBClassifier(**p["XGBoost"]["best_params"], tree_method="hist",
                                 eval_metric="logloss", random_state=RNG, n_jobs=-1)
    m["CatBoost"] = CatBoostClassifier(**p["CatBoost"]["best_params"], auto_class_weights="Balanced",
                                       random_seed=RNG, verbose=False, thread_count=-1)
    rf_p = p["RandomForest"]["best_params"]
    m["RandomForest"] = Pipeline([("imp", SimpleImputer(strategy="median")),
                                  ("rf", RandomForestClassifier(**rf_p, class_weight="balanced_subsample",
                                                                random_state=RNG, n_jobs=-1))])
    m["LogReg"] = Pipeline([("p", prep(True)), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    m["kNN"] = Pipeline([("p", prep(True)), ("m", KNeighborsClassifier(n_neighbors=51))])
    return m


def auc(y, p):
    return float(roc_auc_score(y, p))


def main():
    X, y, feat, meta = ND.load(verbose=False)
    print(f"NHANES n={len(y)} HPV+={y.sum()} ({y.mean():.3f})\n")
    models = tuned()

    # base OOF probabilities (each already out-of-fold -> combining them is leakage-safe)
    oof = {name: NP.repeated_oof(est, "native", X, y) for name, est in models.items()}
    singles = {name: auc(y, p) for name, p in oof.items()}
    best_single = max(singles, key=singles.get)
    print("single-model AUROC:")
    for k, v in sorted(singles.items(), key=lambda kv: -kv[1]):
        print(f"   {k:14s} {v:.4f}")

    # correlation of base predictions (why ensembling will/won't help)
    names = list(oof); Z = np.column_stack([oof[n] for n in names])
    corr = np.corrcoef(Z.T)
    print("\nmean pairwise corr among the 4 trees:",
          f"{np.mean([corr[i, j] for i in range(4) for j in range(4) if i < j]):.3f}")

    trees = ["LightGBM", "XGBoost", "CatBoost", "RandomForest"]
    Ztr = np.column_stack([oof[n] for n in trees])
    Zall = Z
    fus = {}
    fus["soft_vote_trees"] = auc(y, Ztr.mean(1))
    fus["soft_vote_all"] = auc(y, Zall.mean(1))
    fus["soft_vote_diverse_trio"] = auc(y, np.column_stack([oof[best_single], oof["LogReg"], oof["kNN"]]).mean(1))
    # rank-average (scale-free combination, often best for AUROC)
    R = np.column_stack([rankdata(oof[n]) for n in trees]) / len(y)
    fus["rank_average_trees"] = auc(y, R.mean(1))
    # stacking: LR meta-learner on base OOF, itself cross-fitted (leakage-safe)
    meta_oof = cross_val_predict(LogisticRegression(max_iter=2000), Zall, y,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=RNG),
                                 method="predict_proba")[:, 1]
    fus["stacking_LR_meta"] = auc(y, meta_oof)

    # in-sample optimal convex weights over all 6 (UPPER BOUND — weights fit on the same OOF)
    def neg_auc(w):
        w = np.clip(w, 0, None); s = w.sum()
        return -auc(y, Zall @ (w / s if s > 0 else np.ones_like(w) / len(w)))
    w0 = np.ones(Zall.shape[1]) / Zall.shape[1]
    wopt = minimize(neg_auc, w0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4}).x
    wopt = np.clip(wopt, 0, None); wopt /= wopt.sum()
    fus["optimal_convex_UPPERBOUND"] = auc(y, Zall @ wopt)

    print("\nfusion AUROC (best single = %s %.4f):" % (best_single, singles[best_single]))
    for k, v in fus.items():
        print(f"   {k:26s} {v:.4f}  (vs best single {v - singles[best_single]:+.4f})")

    out = dict(meta=meta, singles=singles, best_single=best_single, fusions=fus,
               optimal_weights=dict(zip(names, [round(float(x), 3) for x in wopt])),
               tree_mean_corr=float(np.mean([corr[i, j] for i in range(4) for j in range(4) if i < j])))
    json.dump(out, open(os.path.join(MET, "nhanes_ensemble_test.json"), "w"), indent=2)
    make_figs(names, corr, singles, fus, best_single)
    print("\nsaved -> metrics/nhanes_ensemble_test.json , figures/nhanes_ensemble_*")


def make_figs(names, corr, singles, fus, best_single):
    plt.rcParams.update({"font.size": 9.5, "savefig.dpi": 600,
                         "axes.spines.top": False, "axes.spines.right": False})
    # correlation heatmap
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=0.5, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if corr[i, j] > 0.8 else "black")
    fig.colorbar(im, fraction=0.046); ax.set_title("Base-model prediction correlation")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "nhanes_ensemble_corr.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "nhanes_ensemble_corr.pdf"), bbox_inches="tight"); plt.close(fig)

    # single vs fusion bar
    labels = [f"single: {best_single}"] + list(fus)
    vals = [singles[best_single]] + [fus[k] for k in fus]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cols = ["#c0392b" if labels[i].startswith("single") else "#1a1a1a" for i in order]
    ax.barh([labels[i] for i in order], [vals[i] for i in order], color=cols)
    ax.axvline(singles[best_single], color="#c0392b", ls="--", lw=1,
               label=f"best single {singles[best_single]:.4f}")
    ax.set_xlim(0.66, 0.725); ax.set_xlabel("repeated-OOF AUROC")
    ax.set_title("Hybrid / ensemble vs best single model"); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "nhanes_ensemble_comparison.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "nhanes_ensemble_comparison.pdf"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
