"""Tabular risk-factor expert (UCI Cervical Cancer Risk Factors) for late fusion.

Trains ONE calibrated probabilistic head per screening target (Hinselmann, Schiller, Citology,
Biopsy) and reports leakage-safe, out-of-fold metrics so each head's `P_tab(positive)` can be
fused with the frozen image expert's `P_img(abnormal)` (see FUSION_METHOD.md).

Grounding decisions (see FUSION_METHOD.md §4):
  * DROP the `Dx:*` columns (Dx:Cancer/Dx:CIN/Dx:HPV/Dx) -> they are prior diagnoses = target leakage.
  * DROP the two ~92%-missing columns (`STDs: Time since first/last diagnosis`).
  * Keep remaining NaNs -> HistGradientBoostingClassifier ingests them natively (no imputation leakage).
  * Severe imbalance -> class-balanced sample weights + probability calibration; judge by PR-AUC /
    sensitivity, NOT accuracy. (The 98%+ accuracies in the UCI literature are SMOTE-before-split leakage.)
  * Reliability r = clip(2*(AUROC-0.5), 0, 1) from each head's OWN out-of-fold AUROC -> needs NO paired
    data, and feeds the fusion gate's reliability weighting.

Pure sklearn (HistGradientBoostingClassifier + CalibratedClassifierCV); xgboost optional. Runs on CPU.
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

import config as C

UCI_CSV = os.path.join(C.PROJECT, "data", "cervical+cancer+risk+factors",
                       "risk_factors_cervical_cancer.csv")

TARGETS = ["Hinselmann", "Schiller", "Citology", "Biopsy"]
LEAKAGE_DROP = ["Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx"]              # prior diagnoses -> outcome leakage
HIGH_MISSING_DROP = ["STDs: Time since first diagnosis",
                     "STDs: Time since last diagnosis"]            # ~92% missing

N_SPLITS = 5
SPEC_TARGET = 0.90          # report sensitivity at this specificity


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def load_uci(csv_path=UCI_CSV):
    """Return (X: DataFrame of leakage-safe numeric features with NaNs, Y: DataFrame of 4 targets)."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"UCI csv not found at {csv_path}")
    df = pd.read_csv(csv_path, na_values="?")
    df = df.apply(pd.to_numeric, errors="coerce")
    Y = df[TARGETS].astype(int).copy()
    drop = set(TARGETS) | set(LEAKAGE_DROP) | set(HIGH_MISSING_DROP)
    feat_cols = [c for c in df.columns if c not in drop]
    X = df[feat_cols].copy()
    return X, Y, feat_cols


# --------------------------------------------------------------------------- #
#  Model factory
# --------------------------------------------------------------------------- #
def _base_estimator():
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
        l2_regularization=1.0, min_samples_leaf=20,
        early_stopping=False, random_state=C.SEED)


def _calibrated(base=None, method="sigmoid", cv=3):
    """Wrap the base GBDT in Platt/isotonic calibration (sigmoid is safe at low positive counts)."""
    return CalibratedClassifierCV(estimator=base or _base_estimator(), method=method, cv=cv)


def _balanced_weights(y):
    y = np.asarray(y).astype(int)
    n, npos = len(y), int(y.sum())
    nneg = n - npos
    w = np.where(y == 1, n / (2.0 * max(npos, 1)), n / (2.0 * max(nneg, 1)))
    return w


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def binary_ece(p, y, n_bins=15):
    p = np.asarray(p, float); y = np.asarray(y, int)
    pred = (p >= 0.5).astype(int); conf = np.maximum(p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def sensitivity_at_specificity(p, y, spec=SPEC_TARGET):
    y = np.asarray(y, int)
    if len(set(y)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, p)
    ok = fpr <= (1.0 - spec)
    if not ok.any():
        return 0.0, 1.0
    i = np.argmax(tpr[ok])                       # best TPR among allowed-FPR points
    idx = np.where(ok)[0][i]
    return float(tpr[idx]), float(thr[idx])


def reliability_from_auroc(auroc):
    """r = clip(2*(AUROC-0.5), 0, 1). CONVENTION (not an estimated quantity) — used ONLY to weight the
    Dempster-Shafer CONFLICT flag in fusion.py, NOT the point estimate. 0.5 -> 0 (useless); 1.0 -> 1."""
    if auroc is None or np.isnan(auroc):
        return 0.0
    return float(np.clip(2.0 * (auroc - 0.5), 0.0, 1.0))


def bootstrap_auroc_ci(y, p, n_boot=2000, seed=C.SEED):
    """Bootstrap 95% CI for AUROC (patients resampled) — the honest uncertainty at tiny positive counts."""
    y = np.asarray(y, int); p = np.asarray(p, float)
    if len(set(y)) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(y); boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


# --------------------------------------------------------------------------- #
#  Train + honest out-of-fold evaluation, per target
# --------------------------------------------------------------------------- #
def train_and_eval(target, X=None, Y=None, save=True):
    if X is None or Y is None:
        X, Y, _ = load_uci()
    if target == "any4":                          # composite: any screening test positive (11.9%)
        y = (Y[TARGETS].sum(1) > 0).to_numpy().astype(int)
    else:
        y = Y[target].to_numpy().astype(int)
    Xv = X.to_numpy(dtype=np.float64)            # HGB handles NaN
    n_pos = int(y.sum())

    # ---- out-of-fold calibrated probabilities (leakage-safe) ----
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=C.SEED)
    oof = np.full(len(y), np.nan)
    for tr, va in skf.split(Xv, y):
        clf = _calibrated()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xv[tr], y[tr], sample_weight=_balanced_weights(y[tr]))
        oof[va] = clf.predict_proba(Xv[va])[:, 1]

    auroc = float(roc_auc_score(y, oof)) if len(set(y)) > 1 else float("nan")
    ci_lo, ci_hi = bootstrap_auroc_ci(y, oof)
    ap = float(average_precision_score(y, oof)) if len(set(y)) > 1 else float("nan")
    brier = float(brier_score_loss(y, oof))
    ece = binary_ece(oof, y)
    sens, thr = sensitivity_at_specificity(oof, y)
    metrics = dict(
        target=target, n=int(len(y)), n_pos=n_pos, base_rate=float(y.mean()),
        auroc=auroc, auroc_ci95=[ci_lo, ci_hi], auroc_above_chance=bool(ci_lo > 0.5),
        ap_pr_auc=ap, brier=brier, ece=ece,
        sensitivity_at_spec90=sens, threshold_at_spec90=thr,
        reliability=reliability_from_auroc(auroc))

    # ---- final deployable model: calibrated on ALL data ----
    final = _calibrated()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(Xv, y, sample_weight=_balanced_weights(y))
    if save:
        os.makedirs(C.CKPTDIR, exist_ok=True)
        joblib.dump(dict(model=final, features=list(X.columns), metrics=metrics),
                    os.path.join(C.CKPTDIR, f"tabular_{target}.joblib"))
    flag = "" if metrics["auroc_above_chance"] else "  <= CHANCE (drop from fusion)"
    print(f"[tabular {target:11s}] n_pos={n_pos:3d} ({100*y.mean():.1f}%)  "
          f"AUROC={auroc:.3f} CI[{ci_lo:.3f},{ci_hi:.3f}]  PR-AUC={ap:.3f}  ECE={ece:.3f}  "
          f"sens@spec90={sens:.3f}  r={metrics['reliability']:.2f}{flag}")
    return metrics


def train_all(save=True):
    X, Y, feat_cols = load_uci()
    print(f"UCI: {len(X)} patients, {len(feat_cols)} leakage-safe features "
          f"(dropped {LEAKAGE_DROP + HIGH_MISSING_DROP})")
    results = {t: train_and_eval(t, X, Y, save=save) for t in TARGETS + ["any4"]}
    if save:
        os.makedirs(C.METRICDIR, exist_ok=True)
        with open(os.path.join(C.METRICDIR, "tabular_metrics.json"), "w") as f:
            json.dump(results, f, indent=2)
    return results


# --------------------------------------------------------------------------- #
#  Deployment helpers
# --------------------------------------------------------------------------- #
def load_expert(target):
    path = os.path.join(C.CKPTDIR, f"tabular_{target}.joblib")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no trained tabular expert at {path} — run tabular.train_all() first")
    return joblib.load(path)


def p_positive(expert, X_row):
    """P_tab(positive) for one or many patients. X_row: DataFrame/2D array aligned to expert['features']."""
    model = expert["model"]
    X = X_row[expert["features"]].to_numpy(dtype=np.float64) if isinstance(X_row, pd.DataFrame) \
        else np.atleast_2d(np.asarray(X_row, dtype=np.float64))
    return model.predict_proba(X)[:, 1]


if __name__ == "__main__":
    C.set_seed()
    train_all()
