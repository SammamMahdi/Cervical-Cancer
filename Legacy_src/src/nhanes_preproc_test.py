"""Does preprocessing improve AUROC? — a controlled ablation.

Holds the model fixed (tuned LightGBM) and varies ONLY the preprocessing, under the same repeated
stratified 5-fold OOF protocol. The point: for gradient-boosted trees, scaling / monotonic transforms are
rank-preserving and therefore CANNOT change AUROC; the only preprocessing with a real chance of helping is
how missing values are handled (native NaN + missingness flags) and, marginally, categorical encoding.

Outputs -> Tabular results/NHANES/figures/nhanes_preproc_ablation.*  and  metrics/nhanes_preproc_test.json
Run:  python nhanes_preproc_test.py
"""
import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import (StandardScaler, PowerTransformer, QuantileTransformer,
                                   TargetEncoder, FunctionTransformer)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

import nhanes_data as ND
from nhanes_pipeline import NUM, CAT
RNG = 42
MET, FIG = ND.MET, ND.FIG


def lgbm():
    from lightgbm import LGBMClassifier
    p = os.path.join(MET, "nhanes_hpo_results.json")
    params = json.load(open(p))["models"]["LightGBM"]["best_params"] if os.path.exists(p) else {}
    return LGBMClassifier(**params, class_weight="balanced", random_state=RNG, verbose=-1, n_jobs=-1)


def keep_nan():
    return FunctionTransformer(lambda a: a)          # identity, preserves NaN for native LGBM


def variants():
    cols = NUM + CAT
    miss = FeatureUnion([("raw", keep_nan()),
                         ("flag", MissingIndicator(features="all"))])
    te = ColumnTransformer([("num", keep_nan(), list(range(len(NUM)))),
                            ("cat", TargetEncoder(target_type="binary", cv=5),
                             list(range(len(NUM), len(cols))))])
    return {
        "A_median_impute":         Pipeline([("imp", SimpleImputer(strategy="median")), ("m", lgbm())]),
        "B_native_NaN":            Pipeline([("m", lgbm())]),
        "C_native_NaN+missflags":  Pipeline([("mi", miss), ("m", lgbm())]),
        "D_KNN_impute":            Pipeline([("imp", KNNImputer(n_neighbors=10)), ("m", lgbm())]),
        "E_MICE_impute":           Pipeline([("imp", IterativeImputer(max_iter=10, random_state=RNG)), ("m", lgbm())]),
        "F_scale+impute":          Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", lgbm())]),
        "G_yeojohnson+impute":     Pipeline([("imp", SimpleImputer(strategy="median")), ("pt", PowerTransformer()), ("m", lgbm())]),
        "H_quantile+impute":       Pipeline([("imp", SimpleImputer(strategy="median")), ("qt", QuantileTransformer(output_distribution="normal", random_state=RNG)), ("m", lgbm())]),
        "I_targetencode_cats":     Pipeline([("te", te), ("m", lgbm())]),
    }


def repeated_oof(build, X, y, repeats=3):
    Xin = X[NUM + CAT].to_numpy(float)              # numeric matrix, NaN preserved
    accum = np.zeros(len(y)); cnt = np.zeros(len(y))
    for r in range(repeats):
        sk = StratifiedKFold(5, shuffle=True, random_state=RNG + r)
        for tr, te in sk.split(Xin, y):
            from sklearn.base import clone
            est = clone(build); est.fit(Xin[tr], y[tr])
            accum[te] += est.predict_proba(Xin[te])[:, 1]; cnt[te] += 1
    return accum / np.maximum(cnt, 1)


def main():
    X, y, feat, meta = ND.load(verbose=False)
    print(f"NHANES n={len(y)} HPV+={y.sum()} ({y.mean():.3f})\n")
    vs = variants(); res = {}
    base = None
    for name, pipe in vs.items():
        p = repeated_oof(pipe, X, y)
        auc = float(roc_auc_score(y, p)); ap = float(average_precision_score(y, p))
        if base is None:
            base = auc
        res[name] = dict(auroc=auc, pr_auc=ap, delta_vs_baseline=round(auc - base, 4))
        print(f'{name:26s} AUROC={auc:.4f}  (d {auc-base:+.4f})  PR-AUC={ap:.3f}')

    json.dump({"meta": meta, "baseline": "A_median_impute", "results": res},
              open(os.path.join(MET, "nhanes_preproc_test.json"), "w"), indent=2)
    make_fig(res, base)
    print("\nsaved -> metrics/nhanes_preproc_test.json , figures/nhanes_preproc_ablation.*")


def make_fig(res, base):
    plt.rcParams.update({"font.size": 9.5, "savefig.dpi": 600,
                         "axes.spines.top": False, "axes.spines.right": False})
    order = sorted(res, key=lambda k: res[k]["auroc"])
    fig, ax = plt.subplots(figsize=(7, 4.6))
    vals = [res[k]["auroc"] for k in order]
    cols = ["#c0392b" if res[k]["auroc"] > base + 1e-4 else
            ("#999" if abs(res[k]["auroc"] - base) <= 1e-4 else "#1a1a1a") for k in order]
    ax.barh(order, vals, color=cols)
    ax.axvline(base, color="#c0392b", ls="--", lw=1, label=f"baseline (median impute) {base:.4f}")
    ax.set_xlim(0.68, 0.725); ax.set_xlabel("repeated-OOF AUROC")
    ax.set_title("Preprocessing ablation (model fixed = tuned LightGBM)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "nhanes_preproc_ablation.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "nhanes_preproc_ablation.pdf"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
