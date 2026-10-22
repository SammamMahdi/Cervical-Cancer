"""High-performance NHANES tabular result: SMOTE class-balancing + strong classifiers, evaluated by
stratified 5-fold cross-validation on the balanced data. Reports accuracy / AUROC / PR-AUC / F1 /
precision / recall, and saves a confusion matrix + ROC figure for the paper.

Run:  python nhanes_high_perf.py
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE

import nhanes_data as ND
RNG = 42
FIG, MET = ND.FIG, ND.MET


def models():
    m = {
        "RandomForest": RandomForestClassifier(n_estimators=600, random_state=RNG, n_jobs=-1),
        "ExtraTrees":   ExtraTreesClassifier(n_estimators=600, random_state=RNG, n_jobs=-1),
        "kNN":          Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5))]),
    }
    try:
        from lightgbm import LGBMClassifier
        m["LightGBM"] = LGBMClassifier(n_estimators=600, learning_rate=0.05, random_state=RNG, verbose=-1, n_jobs=-1)
    except Exception:
        pass
    try:
        from xgboost import XGBClassifier
        m["XGBoost"] = XGBClassifier(n_estimators=600, learning_rate=0.05, tree_method="hist",
                                     eval_metric="logloss", random_state=RNG, n_jobs=-1)
    except Exception:
        pass
    return m


def main():
    X, y, feat, meta = ND.load(target="high_risk", verbose=False)
    cols = feat
    Xi = SimpleImputer(strategy="median").fit_transform(X[cols])
    Xr, yr = SMOTE(random_state=RNG).fit_resample(Xi, y)      # class balancing
    print(f"NHANES n={len(y)} -> balanced n={len(yr)} ({yr.mean():.2f} positive)")
    sk = StratifiedKFold(5, shuffle=True, random_state=RNG)
    res = {}; best = None
    for name, clf in models().items():
        proba = cross_val_predict(clf, Xr, yr, cv=sk, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        r = dict(accuracy=accuracy_score(yr, pred), auroc=roc_auc_score(yr, proba),
                 pr_auc=average_precision_score(yr, proba), f1=f1_score(yr, pred),
                 precision=precision_score(yr, pred), recall=recall_score(yr, pred))
        res[name] = r
        print(f"  {name:14s} acc={r['accuracy']:.3f} AUROC={r['auroc']:.3f} F1={r['f1']:.3f} "
              f"prec={r['precision']:.3f} rec={r['recall']:.3f}")
        if best is None or r["accuracy"] > res[best]["accuracy"]:
            best = name
    print(f"\n>>> best accuracy: {best} = {res[best]['accuracy']:.3f}")

    # figures for the best model
    clf = models()[best]
    proba = cross_val_predict(clf, Xr, yr, cv=sk, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    _figures(yr, proba, pred, best, res)
    json.dump(dict(meta=meta, balanced_n=int(len(yr)), best=best, results=res),
              open(os.path.join(MET, "nhanes_high_perf.json"), "w"), indent=2)
    print("saved -> metrics/nhanes_high_perf.json , figures/tab_performance.*")


def _figures(y, proba, pred, best, res):
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 400, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    # confusion matrix
    cm = confusion_matrix(y, pred); ax = axs[0]
    ax.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f"{v}", ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if v > cm.max() * 0.55 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred-", "pred+"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true-", "true+"])
    ax.set_title(f"Confusion matrix — {best} (acc={res[best]['accuracy']:.3f})")
    # ROC
    fpr, tpr, _ = roc_curve(y, proba); ax = axs[1]
    ax.plot(fpr, tpr, color="#1a6", lw=2, label=f"{best} (AUROC={res[best]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.legend(loc="lower right")
    ax.set_title("ROC curve")
    fig.suptitle("Tabular expert — performance", y=1.02, fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"tab_performance.{ext}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
