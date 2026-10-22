"""Full result suite for the NHANES tabular expert (class-weighted, no oversampling; tree ensembles).

Generates every figure + table the paper needs for the tabular expert, and caches the expert's OOF
probabilities for the fusion stage.

Figures -> Tabular results/NHANES/figures/tabx_*   |  metrics -> metrics/nhanes_tabular_full.json
  tabx_bakeoff       model comparison (accuracy + AUROC bars)
  tabx_eval          best model: confusion matrix + ROC + precision-recall
  tabx_reliability   best model calibration (reliability + ECE)
  tabx_importance    ExtraTrees feature importances
  tabx_shap          SHAP beeswarm (attribution)
Run:  python nhanes_tabular_full.py
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix, roc_curve,
                             precision_recall_curve, classification_report)
from imblearn.over_sampling import SMOTE

import nhanes_data as ND
RNG = 42
FIG, MET = ND.FIG, ND.MET
INK, POS = "#1a1a1a", "#c0392b"


def _save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, name + "." + ext), bbox_inches="tight")
    plt.close(fig)


def ece(y, p, bins=12):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def make_models(spw=1.0):
    m = {"ExtraTrees": ExtraTreesClassifier(n_estimators=600, class_weight="balanced", random_state=RNG, n_jobs=-1),
         "RandomForest": RandomForestClassifier(n_estimators=600, class_weight="balanced", random_state=RNG, n_jobs=-1),
         "kNN": Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5))])}
    try:
        from lightgbm import LGBMClassifier
        m["LightGBM"] = LGBMClassifier(n_estimators=600, learning_rate=0.05, class_weight="balanced",
                                       random_state=RNG, verbose=-1, n_jobs=-1)
    except Exception: pass
    try:
        from xgboost import XGBClassifier
        m["XGBoost"] = XGBClassifier(n_estimators=600, learning_rate=0.05, scale_pos_weight=spw,
                                     tree_method="hist", eval_metric="logloss", random_state=RNG, n_jobs=-1)
    except Exception: pass
    return m


def main():
    X, y, feat, meta = ND.load(target="high_risk", verbose=False)
    Xi = SimpleImputer(strategy="median").fit_transform(X[feat])
    Xr, yr = Xi, y                       # no oversampling; class imbalance handled by class weights
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    sk = StratifiedKFold(5, shuffle=True, random_state=RNG)
    print(f"n={len(y)} (no oversampling; {y.mean():.3f} positive)")

    # ---- bake-off ----
    res = {}
    for name, clf in make_models(spw).items():
        proba = cross_val_predict(clf, Xr, yr, cv=sk, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        res[name] = dict(accuracy=accuracy_score(yr, pred), auroc=roc_auc_score(yr, proba),
                         pr_auc=average_precision_score(yr, proba), f1=f1_score(yr, pred),
                         precision=precision_score(yr, pred), recall=recall_score(yr, pred))
        print(f"  {name:13s} acc={res[name]['accuracy']:.3f} AUROC={res[name]['auroc']:.3f}")
    best = max(res, key=lambda k: res[k]["auroc"])
    print(f">>> best {best}")

    # ---- Fig bakeoff ----
    order = sorted(res, key=lambda k: res[k]["accuracy"])
    fig, ax = plt.subplots(figsize=(7, 4)); xpos = np.arange(len(order)); w = 0.38
    ax.bar(xpos - w/2, [res[k]["accuracy"] for k in order], w, label="accuracy", color=INK)
    ax.bar(xpos + w/2, [res[k]["auroc"] for k in order], w, label="AUROC", color=POS)
    for i, k in enumerate(order):
        ax.text(i - w/2, res[k]["accuracy"] + .006, f"{res[k]['accuracy']:.3f}", ha="center", fontsize=7.5)
        ax.text(i + w/2, res[k]["auroc"] + .006, f"{res[k]['auroc']:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(xpos); ax.set_xticklabels(order); ax.set_ylim(0.6, 1.0); ax.legend()
    ax.set_ylabel("score (5-fold CV)"); ax.set_title("Tabular expert: model comparison (5-fold CV, no oversampling)")
    _save(fig, "tabx_bakeoff")

    # ---- best model OOF for eval figures ----
    clf = make_models(spw)[best]
    proba = cross_val_predict(clf, Xr, yr, cv=sk, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(yr, pred)
    fpr, tpr, _ = roc_curve(yr, proba); prec, rec, _ = precision_recall_curve(yr, proba)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
    axs[0].imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        axs[0].text(c, r, f"{v}", ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if v > cm.max()*0.55 else "black")
    axs[0].set_xticks([0,1]); axs[0].set_xticklabels(["pred-","pred+"])
    axs[0].set_yticks([0,1]); axs[0].set_yticklabels(["true-","true+"])
    axs[0].set_title(f"Confusion matrix (acc={res[best]['accuracy']:.3f})")
    axs[1].plot(fpr, tpr, color="#1a6", lw=2, label=f"AUROC={res[best]['auroc']:.3f}")
    axs[1].plot([0,1],[0,1],"k--",lw=.8); axs[1].set_xlabel("false positive rate"); axs[1].set_ylabel("true positive rate")
    axs[1].legend(loc="lower right"); axs[1].set_title("ROC curve")
    axs[2].plot(rec, prec, color="#c0392b", lw=2, label=f"PR-AUC={res[best]['pr_auc']:.3f}")
    axs[2].set_xlabel("recall"); axs[2].set_ylabel("precision"); axs[2].legend(loc="lower left"); axs[2].set_title("Precision-recall")
    fig.suptitle(f"Tabular expert — {best} evaluation", y=1.02, fontsize=12); _save(fig, "tabx_eval")

    # ---- reliability (calibrated) ----
    cal = cross_val_predict(CalibratedClassifierCV(make_models(spw)[best], method="isotonic", cv=3),
                            Xr, yr, cv=sk, method="predict_proba")[:, 1]
    edges = np.linspace(0, 1, 11); xs, ys = [], []
    for i in range(10):
        mm = (cal >= edges[i]) & (cal < edges[i+1])
        if mm.sum() > 30: xs.append(cal[mm].mean()); ys.append(yr[mm].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0,1],[0,1],"k--",lw=.8,label="perfect"); ax.plot(xs, ys, "o-", color="#1a6", label=f"ECE={ece(yr,cal):.3f}")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title(f"Calibration — {best} (isotonic)"); ax.legend(loc="lower right"); _save(fig, "tabx_reliability")

    # ---- cache the expert OOF for the fusion stage (written EARLY, before slow SHAP) ----
    json.dump(dict(p=cal.tolist(), y=yr.astype(int).tolist(), prior=float(yr.mean())),
              open(os.path.join(MET, "nhanes_tabular_oof.json"), "w"))
    print("refreshed fusion OOF cache")

    # ---- feature importance ----
    et = ExtraTreesClassifier(n_estimators=600, class_weight="balanced", random_state=RNG, n_jobs=-1).fit(Xr, yr)
    imp = et.feature_importances_; idx = np.argsort(imp)[::-1][:15][::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([feat[i] for i in idx], imp[idx], color=INK)
    ax.set_xlabel("ExtraTrees importance"); ax.set_title("Tabular expert: feature importance"); _save(fig, "tabx_importance")

    # ---- SHAP ----
    try:
        import shap
        ns = 800                                   # small subsample: TreeExplainer on 600-tree ExtraTrees is slow
        et_s = ExtraTreesClassifier(n_estimators=200, class_weight="balanced", random_state=RNG, n_jobs=-1).fit(Xr, yr)
        sv = shap.TreeExplainer(et_s).shap_values(Xr[:ns], check_additivity=False)
        if isinstance(sv, list): sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3: sv = sv[:, :, 1]
        plt.figure(figsize=(7, 5.5))
        shap.summary_plot(sv, Xr[:ns], feature_names=feat, show=False, max_display=15)
        fig = plt.gcf(); fig.suptitle("SHAP beeswarm — tabular expert", y=1.01, fontsize=11); _save(fig, "tabx_shap")
    except Exception as e:
        print("SHAP skipped:", type(e).__name__)

    # ---- per-class report + cache OOF for fusion ----
    rep = classification_report(yr, pred, target_names=["HPV-", "HPV+"], output_dict=True)
    json.dump(dict(meta=meta, balanced_n=int(len(yr)), best=best, results=res, report=rep),
              open(os.path.join(MET, "nhanes_tabular_full.json"), "w"), indent=2)
    print("saved figures tabx_* , metrics/nhanes_tabular_full.json")


if __name__ == "__main__":
    main()
