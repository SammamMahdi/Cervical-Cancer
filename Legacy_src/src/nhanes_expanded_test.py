"""Re-test individual models + voting on the EXPANDED questionnaire.

Compares three feature sets under one fixed set of models and the same repeated 5-fold OOF protocol:
  OLD12  : the original 12 items (baseline ~0.717)
  CLEAN  : the full questionnaire MINUS the HPV-adjacent genital-warts item (the honest expanded model)
  FULL   : the full questionnaire WITH genital warts (shown for transparency; warts = low-risk HPV 6/11)

For each set: individual models (LightGBM, XGBoost, CatBoost, RandomForest, LogReg) + soft-vote of the
4 trees + stacking. Answers "did the extra questionnaire items raise accuracy, and how much of any FULL
gain is the HPV-adjacent item?"

Outputs -> Tabular results/NHANES/figures/nhanes_expanded_*  and  metrics/nhanes_expanded_test.json
Run:  python nhanes_expanded_test.py
"""
import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

import nhanes_data as ND
RNG = 42
MET, FIG = ND.MET, ND.FIG

OLD12 = ["age", "poverty_ratio", "age_first_sex", "n_partners_life", "n_pregnancies", "age_menarche",
         "race", "education", "marital", "ever_pregnant", "smoked_100", "smoking_status"]


def make_prep(cols, scale):
    num = [c for c in cols if c in ND.NUM]; cat = [c for c in cols if c in ND.CAT]
    steps_num = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    return ColumnTransformer([
        ("num", Pipeline([(f"s{i}", t) for i, t in enumerate(steps_num)]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])


def factories(cols, spw):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    return {
        "LightGBM": lambda: LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31,
                                           subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                                           random_state=RNG, verbose=-1, n_jobs=-1),
        "XGBoost": lambda: XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8,
                                         colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
                                         eval_metric="logloss", random_state=RNG, n_jobs=-1),
        "CatBoost": lambda: CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6,
                                               auto_class_weights="Balanced", random_seed=RNG,
                                               verbose=False, thread_count=-1),
        "RandomForest": lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                                          ("rf", RandomForestClassifier(n_estimators=600, min_samples_leaf=5,
                                                                        class_weight="balanced_subsample",
                                                                        random_state=RNG, n_jobs=-1))]),
        "LogReg": lambda: Pipeline([("p", make_prep(cols, True)),
                                    ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    }


def repeated_oof(factory, cols, X, y, repeats=3):
    Xc = X[cols]; accum = np.zeros(len(y)); cnt = np.zeros(len(y))
    for r in range(repeats):
        sk = StratifiedKFold(5, shuffle=True, random_state=RNG + r)
        for tr, te in sk.split(Xc, y):
            est = factory(); est.fit(Xc.iloc[tr], y[tr])
            accum[te] += est.predict_proba(Xc.iloc[te])[:, 1]; cnt[te] += 1
    return accum / np.maximum(cnt, 1)


def eval_set(name, cols, X, y, spw):
    print(f"\n=== {name}  ({len(cols)} features) ===")
    facs = factories(cols, spw)
    oof = {}
    for m, f in facs.items():
        p = repeated_oof(f, cols, X, y)
        oof[m] = p; print(f"   {m:14s} AUROC={roc_auc_score(y, p):.4f}")
    trees = ["LightGBM", "XGBoost", "CatBoost", "RandomForest"]
    vote = np.column_stack([oof[t] for t in trees]).mean(1)
    Z = np.column_stack([oof[m] for m in facs])
    stack = cross_val_predict(LogisticRegression(max_iter=2000), Z, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=RNG),
                              method="predict_proba")[:, 1]
    res = {m: float(roc_auc_score(y, p)) for m, p in oof.items()}
    res["soft_vote_trees"] = float(roc_auc_score(y, vote))
    res["stacking"] = float(roc_auc_score(y, stack))
    print(f"   {'soft_vote_trees':14s} AUROC={res['soft_vote_trees']:.4f}")
    print(f"   {'stacking':14s} AUROC={res['stacking']:.4f}")
    return res


def main():
    X, y, feat, meta = ND.load(verbose=False)
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    print(f"NHANES full questionnaire: n={len(y)} HPV+={y.sum()} ({y.mean():.3f})  features={len(feat)}")

    clean = [c for c in feat if c not in ND.ADJACENT]
    full = list(feat)
    sets = {"OLD12": OLD12, "CLEAN_expanded": clean, "FULL_with_warts": full}
    out = {}
    for name, cols in sets.items():
        out[name] = eval_set(name, cols, X, y, spw)

    # univariate signal of the NEW items (what the expansion added)
    new_items = [c for c in feat if c not in OLD12]
    uni = {}
    for c in new_items:
        v = X[c].to_numpy(float); m = ~np.isnan(v)
        if m.sum() > 200 and len(np.unique(y[m])) > 1:
            a = roc_auc_score(y[m], v[m]); uni[c] = float(max(a, 1 - a))
    print("\nunivariate AUROC of NEW items:")
    for c, a in sorted(uni.items(), key=lambda kv: -kv[1]):
        print(f"   {c:26s} {a:.3f}")

    json.dump(dict(meta=meta, sets=out, new_item_univariate=uni),
              open(os.path.join(MET, "nhanes_expanded_test.json"), "w"), indent=2)
    make_fig(out, uni)
    print("\nsaved -> metrics/nhanes_expanded_test.json , figures/nhanes_expanded_*")


def make_fig(out, uni):
    plt.rcParams.update({"font.size": 9, "savefig.dpi": 600,
                         "axes.spines.top": False, "axes.spines.right": False})
    # panel 1: model x feature-set
    models = ["LightGBM", "XGBoost", "CatBoost", "RandomForest", "LogReg", "soft_vote_trees", "stacking"]
    sets = list(out); x = np.arange(len(models)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, s in enumerate(sets):
        ax.bar(x + (i - 1) * w, [out[s][m] for m in models], w, label=s)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylim(0.66, 0.78); ax.axhline(0.717, color="#888", ls="--", lw=1, label="old-12 tuned 0.717")
    ax.set_ylabel("repeated-OOF AUROC"); ax.legend(fontsize=8, ncol=2)
    ax.set_title("Individual models + voting across feature sets")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "nhanes_expanded_comparison.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "nhanes_expanded_comparison.pdf"), bbox_inches="tight"); plt.close(fig)
    # panel 2: new-item univariate
    items = dict(sorted(uni.items(), key=lambda kv: kv[1]))
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.barh(list(items), list(items.values()), color="#1a1a1a"); ax.axvline(0.5, color="#c0392b", ls="--", lw=1)
    ax.set_xlabel("univariate AUROC"); ax.set_title("Signal in the newly-added questionnaire items")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "nhanes_expanded_new_signal.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "nhanes_expanded_new_signal.pdf"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
