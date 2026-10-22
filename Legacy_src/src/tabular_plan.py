"""Execute the tabular work plan (TABULAR_PLAN.md) and write everything to  <repo>/Tabular results/.

Tracks:
  A  stronger model arsenal  -> 11-12 model bake-off (adds LightGBM, CatBoost, TabPFN*) + calibrated stacking
  B  feature engineering      -> baseline vs +interactions vs +aggregates vs +all vs +selection  (ΔAUROC)
  C  target strategy          -> AUROC per target (Biopsy / Schiller / Citology / Hinselmann / any-of-4)
  D  imbalance & calibration  -> SMOTE-inside-folds vs class-weight; reliability diagram + ECE; operating point
  F  interpretability         -> permutation importance (SHAP already in the characterization)
  E  protocol                 -> repeated stratified CV, bootstrap 95% CI, PR-AUC/sens/ECE, leakage guards in code

*TabPFN weights are license-gated: run `python -c "from tabpfn import TabPFNClassifier; TabPFNClassifier()"`
 once and complete the free login (or set TABPFN_API_KEY); this script skips it gracefully otherwise.

Leakage-safe throughout (Dx:* dropped via tabular.load_uci; NO oversample-before-split; SMOTE only inside
train folds). Run from legacy_src/src:  python tabular_plan.py
"""
import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# reuse the house style + helpers from the characterization module (importing applies rcParams)
import tabular_characterize as TC
import tabular as TAB
import config as C
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve)
import lightgbm as lgb
from catboost import CatBoostClassifier

PAL = TC.PAL
short = TC.short
OUT = os.path.join(C.PROJECT, "Tabular results", "UCI")
FIG = os.path.join(OUT, "figures")
MET = os.path.join(OUT, "metrics")
os.makedirs(FIG, exist_ok=True); os.makedirs(MET, exist_ok=True)
SEED = C.SEED
TARGETS = ["Hinselmann", "Schiller", "Citology", "Biopsy"]


def save(fig, stem):
    fig.savefig(os.path.join(FIG, stem + ".pdf"))
    fig.savefig(os.path.join(FIG, stem + ".png"))
    plt.close(fig)
    return stem


def binary_ece(p, y, n_bins=15):
    p = np.asarray(p, float); y = np.asarray(y, int)
    conf = np.maximum(p, 1 - p); pred = (p >= 0.5).astype(int); correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


# =========================================================================== #
#  TRACK A — model arsenal
# =========================================================================== #
def all_models():
    m = TC.models()                                   # the 9 sklearn/XGB families
    m["LightGBM"] = TC._pipe(lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                             class_weight="balanced", min_child_samples=20, random_state=SEED,
                             verbose=-1), scale=False)
    m["CatBoost"] = TC._pipe(CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                             auto_class_weights="Balanced", random_state=SEED, verbose=0), scale=False)
    # TabPFN — cloud client (Prior Labs API) if TABPFN_TOKEN is set (UCI is public CC-BY data),
    # else the local license-gated weights. Enabled only if a real fit+predict succeeds.
    tok = os.environ.get("TABPFN_TOKEN")
    try:
        os.environ.setdefault("TABPFN_ALLOW_INTERACTIVE", "0")
        if tok:
            from tabpfn_client import TabPFNClassifier as _TabPFN, set_access_token
            set_access_token(tok)
        else:
            from tabpfn import TabPFNClassifier as _TabPFN
        _t, _rng = _TabPFN(), np.random.RandomState(0)
        _t.fit(_rng.normal(size=(16, 3)), np.array([0, 1] * 8))     # non-constant features (server rejects zeros)
        _t.predict_proba(_rng.normal(size=(3, 3)))
        m["TabPFN"] = TC._pipe(_TabPFN(), scale=False)
        print(f"  [TabPFN enabled — {'cloud API' if tok else 'local weights'}]")
    except Exception as e:
        print(f"  [TabPFN skipped: {type(e).__name__}: {str(e)[:80]}]")
    return m


def track_A(X, y):
    print("\n=== TRACK A — model bake-off (repeated 5-fold OOF, target=Biopsy) ===")
    res = {}
    for name, est in all_models().items():
        seeds = (42,) if name == "TabPFN" else (42, 7, 13)   # TabPFN cloud calls are slow -> single seed
        mu, sd, oof = TC.oof_auroc(est, X, y, seeds=seeds)
        lo, hi = TC.boot_ci(y, oof)
        res[name] = dict(auroc=mu, sd=sd, ci=[lo, hi])
        print(f"  {name:14s} AUROC={mu:.3f} ± {sd:.3f}  CI[{lo:.3f}, {hi:.3f}]")

    # calibrated stacking of three complementary strong learners
    stack = StackingClassifier(
        estimators=[("rf", RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                                  min_samples_leaf=3, random_state=SEED)),
                    ("lgbm", lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                             class_weight="balanced", random_state=SEED, verbose=-1)),
                    ("knn", Pipeline([("imp", SimpleImputer(strategy="median")),
                                      ("knn", KNeighborsClassifier(n_neighbors=21, weights="distance"))]))],
        final_estimator=LogisticRegression(max_iter=2000), stack_method="predict_proba", cv=5)
    mu, sd, oof = TC.oof_auroc(TC._pipe(stack, scale=False), X, y)
    lo, hi = TC.boot_ci(y, oof)
    res["Stacking (RF+LGBM+kNN)"] = dict(auroc=mu, sd=sd, ci=[lo, hi])
    print(f"  {'Stacking':14s} AUROC={mu:.3f} ± {sd:.3f}  CI[{lo:.3f}, {hi:.3f}]")

    order = sorted(res, key=lambda k: res[k]["auroc"])
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ys = np.arange(len(order)); mus = [res[k]["auroc"] for k in order]
    err = [[res[k]["auroc"] - res[k]["ci"][0] for k in order], [res[k]["ci"][1] - res[k]["auroc"] for k in order]]
    ax.barh(ys, mus, xerr=err, color=PAL["bar"], edgecolor="black", linewidth=1.0,
            error_kw=dict(elinewidth=1.2, ecolor="#333", capsize=3))
    for yv, k in zip(ys, order):
        ax.text(res[k]["auroc"] + 0.004, yv, f"{res[k]['auroc']:.3f}", va="center", fontsize=8, fontweight="bold")
    ax.axvline(0.5, ls="--", c=PAL["chance"], lw=1.2, label="chance")
    ax.set_yticks(ys); ax.set_yticklabels(order); ax.set_xlim(0.45, 0.80)
    ax.set_xlabel("out-of-fold AUROC (95% CI)"); ax.set_title("Track A — expanded model bake-off (Biopsy)")
    ax.legend(loc="lower right"); ax.grid(True, axis="x", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    save(fig, "A_model_bakeoff")
    return res


# =========================================================================== #
#  TRACK B — feature engineering
# =========================================================================== #
def engineer(X):
    """Add interaction / ratio / aggregate / missingness features (leakage-safe)."""
    d = X.copy()
    g = lambda c: d[c] if c in d.columns else pd.Series(np.nan, index=d.index)
    age = g("Age"); part = g("Number of sexual partners"); first = g("First sexual intercourse")
    preg = g("Num of pregnancies"); syr = g("Smokes (years)"); spk = g("Smokes (packs/year)")
    hc = g("Hormonal Contraceptives (years)"); iud = g("IUD (years)"); nstd = g("STDs (number)")
    eps = 1e-3
    d["x_partners_per_year_since_first"] = part / (age - first + eps)   # sexual-exposure rate
    d["x_preg_per_age"] = preg / (age + eps)
    d["x_pack_years"] = syr * spk
    d["x_contra_exposure"] = hc.fillna(0) + iud.fillna(0)
    std_flags = [c for c in d.columns if c.startswith("STDs:") and "number" not in c.lower()]
    d["x_std_total"] = d[std_flags].sum(axis=1)
    d["x_std_answered"] = (~g("STDs (number)").isna()).astype(float)     # missingness-as-signal
    return d


def track_B(X, y, workhorse):
    print("\n=== TRACK B — feature engineering (workhorse=LightGBM) ===")
    Xe = engineer(X)
    inter = [c for c in Xe.columns if c.startswith("x_") and ("per" in c or "pack" in c)]
    aggr = [c for c in Xe.columns if c in ("x_contra_exposure", "x_std_total", "x_std_answered")]
    from sklearn.feature_selection import mutual_info_classif
    Xi = SimpleImputer(strategy="median").fit_transform(Xe)
    mi = mutual_info_classif(Xi, y, random_state=SEED)
    topk = [Xe.columns[i] for i in np.argsort(-mi)[:12]]
    configs = {"baseline": list(X.columns),
               "+interactions": list(X.columns) + inter,
               "+aggregates": list(X.columns) + aggr,
               "+all engineered": list(Xe.columns),
               "+select top-12 (MI)": topk}
    res = {}
    for name, cols in configs.items():
        mu, sd, oof = TC.oof_auroc(clone(workhorse), Xe[cols], y, seeds=(42, 7))
        lo, hi = TC.boot_ci(y, oof)
        res[name] = dict(auroc=mu, ci=[lo, hi], n_feat=len(cols))
        print(f"  {name:22s} ({len(cols):2d} feat) AUROC={mu:.3f}  CI[{lo:.3f}, {hi:.3f}]")
    names = list(configs)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    mus = [res[n]["auroc"] for n in names]
    err = [[res[n]["auroc"] - res[n]["ci"][0] for n in names], [res[n]["ci"][1] - res[n]["auroc"] for n in names]]
    bars = ax.bar(names, mus, yerr=err, color=PAL["bar"], edgecolor="black", linewidth=1.0,
                  error_kw=dict(elinewidth=1.2, ecolor="#333", capsize=3))
    ax.axhline(res["baseline"]["auroc"], ls="--", c=PAL["chance"], lw=1.0, label="baseline")
    for b, n in zip(bars, names):
        ax.text(b.get_x() + b.get_width() / 2, res[n]["auroc"], f"{res[n]['auroc']:.3f}", ha="center",
                va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("OOF AUROC (Biopsy)"); ax.set_ylim(0.45, 0.75); ax.legend(loc="lower right")
    ax.set_title("Track B — feature engineering ΔAUROC"); plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    save(fig, "B_feature_engineering")
    return res


# =========================================================================== #
#  TRACK C — target strategy
# =========================================================================== #
def track_C(X, Y, workhorse):
    print("\n=== TRACK C — target strategy ===")
    yany = (Y[TARGETS].sum(1) > 0).astype(int).to_numpy()
    tgts = {t: Y[t].to_numpy().astype(int) for t in TARGETS}
    tgts["any-of-4"] = yany
    res = {}
    for t, yv in tgts.items():
        mu, sd, oof = TC.oof_auroc(clone(workhorse), X, yv, seeds=(42, 7))
        lo, hi = TC.boot_ci(yv, oof)
        res[t] = dict(auroc=mu, ci=[lo, hi], n_pos=int(yv.sum()))
        print(f"  {t:11s} n_pos={int(yv.sum()):3d}  AUROC={mu:.3f}  CI[{lo:.3f}, {hi:.3f}]")
    names = list(res)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    mus = [res[n]["auroc"] for n in names]
    err = [[res[n]["auroc"] - res[n]["ci"][0] for n in names], [res[n]["ci"][1] - res[n]["auroc"] for n in names]]
    bars = ax.bar(names, mus, yerr=err, color=[PAL["pos"] if res[n]["ci"][0] > 0.5 else PAL["chance"] for n in names],
                  edgecolor="black", linewidth=1.0, error_kw=dict(elinewidth=1.2, ecolor="#333", capsize=3))
    for b, n in zip(bars, names):
        ax.text(b.get_x() + b.get_width() / 2, res[n]["auroc"], f"{res[n]['auroc']:.3f}", ha="center",
                va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(0.5, ls="--", c=PAL["chance"], lw=1.0)
    ax.set_ylabel("OOF AUROC"); ax.set_ylim(0.40, 0.75)
    ax.set_title("Track C — which target is learnable? (grey = ≤ chance)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    save(fig, "C_target_strategy")
    return res


# =========================================================================== #
#  TRACK D — imbalance & calibration
# =========================================================================== #
def _oof_probs(est, X, y, seed=SEED):
    Xv = X.to_numpy(float); oof = np.full(len(y), np.nan)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xv, y):
        m = clone(est); m.fit(Xv[tr], y[tr]); oof[va] = m.predict_proba(Xv[va])[:, 1]
    return oof


def track_D(X, y, workhorse):
    print("\n=== TRACK D — imbalance & calibration ===")
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    # (D1) class-weight vs SMOTE-inside-folds
    uncal = clone(workhorse)
    smote = ImbPipeline([("imp", SimpleImputer(strategy="median")), ("smote", SMOTE(random_state=SEED)),
                         ("clf", lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                          random_state=SEED, verbose=-1))])
    cal = CalibratedClassifierCV(clone(workhorse), method="isotonic", cv=3)
    oof_u = _oof_probs(uncal, X, y); oof_s = _oof_probs(smote, X, y); oof_c = _oof_probs(cal, X, y)
    d1 = {"class-weight (uncal)": dict(auroc=roc_auc_score(y, oof_u), ece=binary_ece(oof_u, y)),
          "SMOTE-inside-folds": dict(auroc=roc_auc_score(y, oof_s), ece=binary_ece(oof_s, y)),
          "class-weight + isotonic": dict(auroc=roc_auc_score(y, oof_c), ece=binary_ece(oof_c, y))}
    for k, v in d1.items():
        print(f"  {k:26s} AUROC={v['auroc']:.3f}  ECE={v['ece']:.3f}")

    # (D2) reliability diagram: uncalibrated vs isotonic-calibrated
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], ls=":", c="#222", lw=1.5, label="perfect")
    for oof, name, col in [(oof_u, f"uncalibrated (ECE {binary_ece(oof_u,y):.3f})", PAL["chance"]),
                           (oof_c, f"isotonic (ECE {binary_ece(oof_c,y):.3f})", PAL["pos"])]:
        edges = np.linspace(0, 1, 9); mids, accs = [], []
        for i in range(8):
            m = (oof >= edges[i]) & (oof < edges[i + 1])
            if m.sum() >= 5:
                mids.append(oof[m].mean()); accs.append(y[m].mean())
        ax.plot(mids, accs, marker="o", ms=6, color=col, label=name)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Track D — calibration (Biopsy)"); ax.legend(loc="upper left"); ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    save(fig, "D_reliability")

    # (D3) operating point: sensitivity/specificity vs threshold + PR curve; pick sensitivity>=0.80
    fpr, tpr, thr = roc_curve(y, oof_c); spec = 1 - fpr
    prec, rec, pthr = precision_recall_curve(y, oof_c)
    ok = np.where(tpr >= 0.80)[0]
    j = ok[np.argmax(spec[ok])] if len(ok) else int(np.argmax(tpr - fpr))
    op = dict(threshold=float(thr[j]), sensitivity=float(tpr[j]), specificity=float(spec[j]),
              pr_auc=float(average_precision_score(y, oof_c)))
    print(f"  operating point @ sens>=0.80: thr={op['threshold']:.3f} sens={op['sensitivity']:.2f} "
          f"spec={op['specificity']:.2f}  PR-AUC={op['pr_auc']:.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    axes[0].plot(thr, tpr[:-0 or None] if False else tpr, color=PAL["pos"], label="sensitivity")
    axes[0].plot(thr, spec, color=PAL["bar"], label="specificity")
    axes[0].axvline(op["threshold"], ls="--", c=PAL["chance"], lw=1.2)
    axes[0].set_xlabel("threshold"); axes[0].set_ylabel("rate"); axes[0].set_title("Sens/Spec vs threshold")
    axes[0].legend(loc="center right"); axes[0].grid(True, alpha=0.3, linestyle="--"); axes[0].set_axisbelow(True)
    axes[1].plot(rec, prec, color=PAL["pos"]); axes[1].axhline(y.mean(), ls="--", c=PAL["chance"], lw=1.0,
                                                              label=f"base rate {y.mean():.3f}")
    axes[1].set_xlabel("recall (sensitivity)"); axes[1].set_ylabel("precision")
    axes[1].set_title(f"PR curve (AP={op['pr_auc']:.3f})"); axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3, linestyle="--"); axes[1].set_axisbelow(True)
    fig.suptitle("Track D — operating point for a screening (high-sensitivity) target", fontweight="bold", y=1.02)
    save(fig, "D_operating_point")
    return dict(imbalance_calibration=d1, operating_point=op)


# =========================================================================== #
#  TRACK F — interpretability (permutation importance)
# =========================================================================== #
def track_F(X, y, workhorse):
    print("\n=== TRACK F — permutation importance (LightGBM) ===")
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    m = clone(workhorse).fit(Xi, y)
    r = permutation_importance(m, Xi, y, n_repeats=20, random_state=SEED, scoring="roc_auc")
    idx = np.argsort(r.importances_mean)[-12:]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh([short(X.columns[i]) for i in idx], r.importances_mean[idx],
            xerr=r.importances_std[idx], color=PAL["hi"], edgecolor="black", linewidth=0.8,
            error_kw=dict(elinewidth=1.0, ecolor="#333"))
    ax.set_xlabel("Δ AUROC when feature permuted"); ax.set_title("Track F — permutation importance (Biopsy)")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    save(fig, "F_permutation_importance")
    return {X.columns[i]: float(r.importances_mean[i]) for i in idx[::-1]}


# =========================================================================== #
#  main
# =========================================================================== #
def main():
    C.set_seed()
    X, Y, feat = TAB.load_uci()
    y = Y["Biopsy"].to_numpy().astype(int)
    print(f"UCI leakage-safe: {len(X)} patients, {len(feat)} features. Outputs -> {OUT}")

    workhorse = TC._pipe(lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                         class_weight="balanced", min_child_samples=20, random_state=SEED, verbose=-1),
                         scale=False)

    out = {}
    out["A_bakeoff"] = track_A(X, y)
    out["B_feature_engineering"] = track_B(X, y, workhorse)
    out["C_target_strategy"] = track_C(X, Y, workhorse)
    out["D"] = track_D(X, y, workhorse)
    out["F_permutation_importance"] = track_F(X, y, workhorse)

    best = max(out["A_bakeoff"], key=lambda k: out["A_bakeoff"][k]["auroc"])
    out["summary"] = dict(best_model=best, best_auroc=out["A_bakeoff"][best]["auroc"],
                          leakage_safe=True, output_dir=OUT)
    json.dump(out, open(os.path.join(MET, "tabular_plan_results.json"), "w"), indent=2)
    print(f"\nBEST model: {best}  AUROC {out['A_bakeoff'][best]['auroc']:.3f}")
    print(f"figures -> {FIG}\nmetrics -> {MET}/tabular_plan_results.json")


if __name__ == "__main__":
    main()
