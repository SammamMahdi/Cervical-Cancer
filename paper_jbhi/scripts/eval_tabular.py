"""HPV-risk (tabular) expert: full evaluation for the paper.

NHANES 2003-2016 female participants with a vaginal HPV swab; target = any of 14 high-risk genotypes.
Six classifier families, class-weighted (NO oversampling), median imputation fitted INSIDE each fold,
stratified 5-fold CV. The best family (by AUROC) is isotonic-calibrated with nested CV and its out-of-fold
probabilities are saved for the fusion stage.

Run (Windows python):  python paper_jbhi/scripts/eval_tabular.py
Outputs: ../results/tabular_eval.json, ../results/tabular_oof.npz, ../results/tabular_shap.npz
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "Legacy_src", "src"))
import nhanes_data as ND
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, accuracy_score,
                             f1_score, precision_score, recall_score, confusion_matrix)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

OUT = os.path.join(HERE, "..", "results"); os.makedirs(OUT, exist_ok=True)
SEED = 42; RNG = np.random.default_rng(0)


def models(spw):
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Logistic regression": Pipeline([("imp", imp()), ("sc", StandardScaler()),
                                         ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "k-NN": Pipeline([("imp", imp()), ("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5))]),
        "RandomForest": Pipeline([("imp", imp()), ("m", RandomForestClassifier(
            n_estimators=600, class_weight="balanced", random_state=SEED, n_jobs=-1))]),
        "ExtraTrees": Pipeline([("imp", imp()), ("m", ExtraTreesClassifier(
            n_estimators=600, class_weight="balanced", random_state=SEED, n_jobs=-1))]),
        "LightGBM": Pipeline([("imp", imp()), ("m", LGBMClassifier(
            n_estimators=600, learning_rate=0.05, class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=-1))]),
        "XGBoost": Pipeline([("imp", imp()), ("m", XGBClassifier(
            n_estimators=600, learning_rate=0.05, scale_pos_weight=spw, tree_method="hist",
            eval_metric="logloss", random_state=SEED, n_jobs=-1))]),
    }


def ece(y, p, bins=10):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def ci(fn, y, p, n=1000):
    v = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(fn(y[i], p[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def scores(y, p, thr=0.5):
    pred = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return dict(auroc=float(roc_auc_score(y, p)), auroc_ci95=ci(roc_auc_score, y, p),
                pr_auc=float(average_precision_score(y, p)), pr_auc_ci95=ci(average_precision_score, y, p),
                brier=float(brier_score_loss(y, p)), ece=ece(y, p), accuracy=float(accuracy_score(y, pred)),
                sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                precision=float(precision_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred)),
                confusion=[[int(tn), int(fp)], [int(fn), int(tp)]])


def main():
    t0 = time.time()
    X, y, feat, meta = ND.load(verbose=False); cyc = X.attrs["cycle"]; Xv = X[feat].to_numpy(float)
    spw = float((y == 0).sum() / (y == 1).sum())
    cohort = dict(meta=meta, per_cycle={c: dict(n=int((cyc == c).sum()), hpv_pos_rate=float(y[cyc == c].mean()))
                                        for c in sorted(set(cyc))},
                  missing_pct={f: float(np.isnan(Xv[:, j]).mean() * 100) for j, f in enumerate(feat)},
                  age_mean=float(np.nanmean(X["age"])), age_sd=float(np.nanstd(X["age"])),
                  features=feat, questionnaire=[dict(section=q[0], feature=q[1], question=q[6], coding=q[7])
                                                for q in ND.QUESTIONNAIRE], engineered=ND.ENGINEERED)
    print(f"n={meta['n']} pos={meta['pos_rate']:.3f} features={len(feat)}")

    sk = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res, oof = {}, {}
    for name, clf in models(spw).items():
        p = cross_val_predict(clf, Xv, y, cv=sk, method="predict_proba")[:, 1]
        res[name] = scores(y, p); oof[name] = p
        print(f"  {name:20s} AUROC={res[name]['auroc']:.3f} {res[name]['auroc_ci95']} PR={res[name]['pr_auc']:.3f} "
              f"acc={res[name]['accuracy']:.3f} ({time.time() - t0:.0f}s)")
    best = max(res, key=lambda k: res[k]["auroc"])
    print(">>> best:", best)

    cal = cross_val_predict(CalibratedClassifierCV(models(spw)[best], method="isotonic", cv=3),
                            Xv, y, cv=sk, method="predict_proba")[:, 1]
    calib = dict(raw=dict(ece=res[best]["ece"], brier=res[best]["brier"]), isotonic=scores(y, cal, thr=float(y.mean())))
    print(f"  calibrated: ECE {res[best]['ece']:.3f} -> {calib['isotonic']['ece']:.3f}  AUROC {calib['isotonic']['auroc']:.3f}")

    np.savez_compressed(os.path.join(OUT, "tabular_oof.npz"), y=y, p_cal=cal, cycle=cyc,
                        **{f"p_{k.replace(' ', '_').replace('-', '')}": v for k, v in oof.items()})
    json.dump(dict(cohort=cohort, results=res, best=best, calibration=calib),
              open(os.path.join(OUT, "tabular_eval.json"), "w"), indent=2)

    # permutation importance on held-out folds (AUROC drop), averaged over folds
    # (n_jobs=1: loky cannot re-spawn large forests on Windows; the forest itself is multi-threaded)
    imps = []
    for tr, te in sk.split(Xv, y):
        m = models(spw)[best].fit(Xv[tr], y[tr])
        r = permutation_importance(m, Xv[te], y[te], scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
        imps.append(r.importances_mean)
    imp = np.mean(imps, 0); imp_sd = np.std(imps, 0)
    perm = sorted([dict(feature=f, auroc_drop=float(imp[j]), sd=float(imp_sd[j])) for j, f in enumerate(feat)],
                  key=lambda d: -d["auroc_drop"])
    print("  top features:", [d["feature"] for d in perm[:6]], f"({time.time() - t0:.0f}s)")

    json.dump(dict(cohort=cohort, results=res, best=best, calibration=calib, permutation_importance=perm,
                   cv="stratified 5-fold, seed 42", seconds=time.time() - t0),
              open(os.path.join(OUT, "tabular_eval.json"), "w"), indent=2)

    # SHAP (TreeSHAP) on a 200-tree forest of the best family, 800-subject subsample
    try:
        import shap
        sub = RNG.choice(len(y), 800, replace=False)
        imp_ = SimpleImputer(strategy="median").fit(Xv); Xi = imp_.transform(Xv)
        fm = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1,
                                    min_samples_leaf=5).fit(Xi, y)
        sv = shap.TreeExplainer(fm).shap_values(Xi[sub], check_additivity=False)
        sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        np.savez_compressed(os.path.join(OUT, "tabular_shap.npz"), shap=sv, X=Xi[sub], features=np.array(feat))
        print(f"  SHAP done ({time.time() - t0:.0f}s)")
    except Exception as e:
        print("  SHAP skipped:", type(e).__name__, e)
    print("saved tabular_eval.json / tabular_oof.npz")


if __name__ == "__main__":
    main()
