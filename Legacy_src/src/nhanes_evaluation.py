"""One-figure evaluation of every modeling approach on the NHANES HPV task (CLEAN expanded questionnaire).

For each approach (LightGBM, XGBoost, CatBoost, RandomForest, LogReg, kNN, soft-vote of the 4 trees,
stacking) it computes repeated 5-fold OOF probabilities and reports, on a SINGLE consolidated figure:
  * ROC curves (all overlaid, AUROC in legend)
  * a metrics matrix — AUROC, accuracy, balanced-acc, precision, recall(sensitivity), specificity, F1, ECE
  * a confusion matrix per approach (at each model's Youden-optimal threshold; calibration-invariant)

Threshold note: threshold-free ranking = AUROC (ROC panel). Confusion matrices / F1 / recall / accuracy are
at the Youden-J operating point (max sensitivity+specificity-1) — the natural balanced point; for deployment
a higher-sensitivity threshold is used (see nhanes_pipeline operating points).

Outputs -> Tabular results/NHANES/figures/nhanes_evaluation_all.*  and  metrics/nhanes_evaluation.json
Run:  python nhanes_evaluation.py
"""
import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix, f1_score,
                             precision_score, recall_score, accuracy_score, balanced_accuracy_score)

import nhanes_data as ND
from nhanes_expanded_test import make_prep, repeated_oof
RNG = 42
MET, FIG = ND.MET, ND.FIG
APPROACHES = ["LightGBM", "XGBoost", "CatBoost", "RandomForest", "LogReg", "kNN"]


def base_factories(cols, spw):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    return {
        "LightGBM": lambda: LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8,
                                           colsample_bytree=0.8, class_weight="balanced", random_state=RNG,
                                           verbose=-1, n_jobs=-1),
        "XGBoost": lambda: XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8,
                                         colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
                                         eval_metric="logloss", random_state=RNG, n_jobs=-1),
        "CatBoost": lambda: CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6,
                                               auto_class_weights="Balanced", random_seed=RNG, verbose=False,
                                               thread_count=-1),
        "RandomForest": lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                                          ("rf", RandomForestClassifier(n_estimators=600, min_samples_leaf=5,
                                                                        class_weight="balanced_subsample",
                                                                        random_state=RNG, n_jobs=-1))]),
        "LogReg": lambda: Pipeline([("p", make_prep(cols, True)),
                                    ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "kNN": lambda: Pipeline([("p", make_prep(cols, True)), ("m", KNeighborsClassifier(n_neighbors=51))]),
    }


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def youden_thr(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return dict(auroc=float(roc_auc_score(y, p)), accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred)),
                specificity=float(tn / (tn + fp)), f1=float(f1_score(y, pred)),
                ece=float(ece(y, p)), cm=[int(tn), int(fp), int(fn), int(tp)], threshold=thr)


def main():
    X, y, feat, meta = ND.load(verbose=False)
    cols = [c for c in feat if c not in ND.ADJACENT]         # CLEAN expanded
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    print(f"n={len(y)} HPV+={y.sum()} ({y.mean():.3f})  features(CLEAN)={len(cols)}")

    facs = base_factories(cols, spw)
    oof = {}
    for name, f in facs.items():
        oof[name] = repeated_oof(f, cols, X, y)
        print(f"  {name:14s} AUROC={roc_auc_score(y, oof[name]):.4f}")
    trees = ["LightGBM", "XGBoost", "CatBoost", "RandomForest"]
    oof["SoftVote(trees)"] = np.column_stack([oof[t] for t in trees]).mean(1)
    Z = np.column_stack([oof[m] for m in facs])
    oof["Stacking"] = cross_val_predict(LogisticRegression(max_iter=2000), Z, y,
                                        cv=StratifiedKFold(5, shuffle=True, random_state=RNG),
                                        method="predict_proba")[:, 1]
    order = APPROACHES + ["SoftVote(trees)", "Stacking"]

    results = {}
    for name in order:
        p = oof[name]; results[name] = metrics_at(y, p, youden_thr(y, p))
        r = results[name]
        print(f"  {name:16s} AUROC={r['auroc']:.3f} F1={r['f1']:.3f} recall={r['recall']:.3f} "
              f"acc={r['accuracy']:.3f} ECE={r['ece']:.3f}")

    json.dump(dict(meta=meta, features=cols, results=results),
              open(os.path.join(MET, "nhanes_evaluation.json"), "w"), indent=2)
    figure(order, oof, y, results)
    recommend(results)


def figure(order, oof, y, results):
    plt.rcParams.update({"font.size": 9, "savefig.dpi": 350,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(17, 13))
    gs = GridSpec(3, 5, figure=fig, height_ratios=[1.35, 1, 1], hspace=0.5, wspace=0.55)

    # --- ROC panel ---
    axr = fig.add_subplot(gs[0, :2])
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(order):
        fpr, tpr, _ = roc_curve(y, oof[name])
        axr.plot(fpr, tpr, lw=1.6, color=cmap(i % 10),
                 label=f"{name} ({results[name]['auroc']:.3f})")
    axr.plot([0, 1], [0, 1], "k--", lw=0.8)
    axr.set_xlabel("false positive rate"); axr.set_ylabel("true positive rate (recall)")
    axr.set_title("ROC curves — all approaches (AUROC in legend)"); axr.legend(fontsize=7.5, loc="lower right")

    # --- metrics matrix ---
    axm = fig.add_subplot(gs[0, 2:])
    cols_m = ["AUROC", "Acc", "Bal-Acc", "Prec", "Recall", "Spec", "F1", "ECE"]
    keys = ["auroc", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "ece"]
    M = np.array([[results[n][k] for k in keys] for n in order])
    # colour: per-column min-max; ECE inverted (lower = better)
    Cc = np.zeros_like(M)
    for j in range(M.shape[1]):
        col = M[:, j]; rng = col.max() - col.min() or 1
        norm = (col - col.min()) / rng
        Cc[:, j] = 1 - norm if cols_m[j] == "ECE" else norm
    axm.imshow(Cc, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axm.set_xticks(range(len(cols_m))); axm.set_xticklabels(cols_m, fontsize=8)
    axm.set_yticks(range(len(order))); axm.set_yticklabels(order, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axm.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=7.5)
    axm.set_title("Metrics (green = better in column; ECE lower = better)")

    # --- confusion matrices ---
    for i, name in enumerate(order):
        ax = fig.add_subplot(gs[1 + i // 4, i % 4])
        tn, fp, fn, tp = results[name]["cm"]
        cm = np.array([[tn, fp], [fn, tp]])
        ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v}", ha="center", va="center",
                    color="white" if v > cm.max() * 0.55 else "black", fontsize=9, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred-", "pred+"], fontsize=7.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true-", "true+"], fontsize=7.5)
        ax.set_title(f"{name}\nrecall={results[name]['recall']:.2f} spec={results[name]['specificity']:.2f}",
                     fontsize=8)
    # hide the unused last cell (10 approaches -> 8 CM slots used in 2x4; we have 8 -> fits)
    fig.suptitle("NHANES HPV-risk — every approach: ROC, metrics, confusion matrices "
                 f"(n={len(y)}, {int(y.sum())} HPV+, CLEAN expanded questionnaire)", fontsize=13, y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"nhanes_evaluation_all.{ext}"), bbox_inches="tight")
    plt.close(fig)


def recommend(results):
    def best(key, hi=True):
        return max(results, key=lambda n: results[n][key]) if hi else min(results, key=lambda n: results[n][key])
    print("\n=== best approach by purpose ===")
    print(f"  discrimination / ranking (AUROC)     : {best('auroc')}  ({results[best('auroc')]['auroc']:.3f})")
    print(f"  screening sensitivity (recall)       : {best('recall')}  ({results[best('recall')]['recall']:.3f})")
    print(f"  balanced single-threshold (F1)       : {best('f1')}  ({results[best('f1')]['f1']:.3f})")
    print(f"  fewest false alarms (precision)      : {best('precision')}  ({results[best('precision')]['precision']:.3f})")
    print(f"  probability quality for fusion (ECE) : {best('ece', hi=False)}  ({results[best('ece', hi=False)]['ece']:.3f})")
    print(f"  raw accuracy (misleading@imbalance)  : {best('accuracy')}  ({results[best('accuracy')]['accuracy']:.3f})")


if __name__ == "__main__":
    main()
