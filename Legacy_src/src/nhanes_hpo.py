"""Optuna hyperparameter search for the NHANES HPV-risk models, then the definitive best result.

- Tunes the 4 families that led the bake-off (LightGBM, XGBoost, CatBoost, RandomForest) with a TPE
  sampler; TQDM progress bar per study (Optuna's `show_progress_bar`).
- Objective = 5-fold stratified OOF AUROC (leakage-safe: imputation for RF is inside the fold pipeline;
  native GBDTs take the raw NaN matrix).
- CPU only: benchmarking showed GPU is 4-18x SLOWER for GBDTs on 9k rows (kernel/transfer overhead), so
  every model runs multi-threaded on CPU.
- Re-evaluates each tuned winner with the full harness (repeated-OOF AUROC + 95% CI, leave-one-cycle-out,
  isotonic-calibrated ECE, screening operating points) and reports the single best.

Outputs -> Tabular results/NHANES/figures/hpo_*  and  metrics/nhanes_hpo_results.json
Run:  python nhanes_hpo.py [--trials 60]
"""
import os
import json
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import nhanes_data as ND
import nhanes_pipeline as NP
from nhanes_pipeline import NUM, CAT

optuna.logging.set_verbosity(optuna.logging.WARNING)
RNG = 42
FIG, MET = ND.FIG, ND.MET


def cv_auroc(est, X, y, folds=5, seed=RNG):
    """5-fold OOF AUROC; est is a fully-formed estimator/pipeline (fit inside each fold)."""
    Xin = X[NUM + CAT]
    sk = StratifiedKFold(folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in sk.split(Xin, y):
        e = clone(est); e.fit(Xin.iloc[tr], y[tr]); oof[te] = e.predict_proba(Xin.iloc[te])[:, 1]
    return roc_auc_score(y, oof)


# ---------- search spaces ----------
def make_lgbm(t):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=t.suggest_int("n_estimators", 100, 3000, step=50),
        learning_rate=t.suggest_float("learning_rate", 2e-3, 0.3, log=True),
        num_leaves=t.suggest_int("num_leaves", 7, 511, log=True),
        max_depth=t.suggest_int("max_depth", 2, 16),
        min_child_samples=t.suggest_int("min_child_samples", 2, 250),
        min_split_gain=t.suggest_float("min_split_gain", 1e-8, 1.0, log=True),
        subsample=t.suggest_float("subsample", 0.4, 1.0),
        subsample_freq=1,
        colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha=t.suggest_float("reg_alpha", 1e-8, 30.0, log=True),
        reg_lambda=t.suggest_float("reg_lambda", 1e-8, 30.0, log=True),
        class_weight="balanced", random_state=RNG, verbose=-1, n_jobs=-1)


def make_xgb(t, spw):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=t.suggest_int("n_estimators", 100, 3000, step=50),
        learning_rate=t.suggest_float("learning_rate", 2e-3, 0.3, log=True),
        max_depth=t.suggest_int("max_depth", 2, 14),
        min_child_weight=t.suggest_float("min_child_weight", 1e-2, 50, log=True),
        subsample=t.suggest_float("subsample", 0.4, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
        colsample_bylevel=t.suggest_float("colsample_bylevel", 0.3, 1.0),
        gamma=t.suggest_float("gamma", 1e-8, 10.0, log=True),
        reg_alpha=t.suggest_float("reg_alpha", 1e-8, 30.0, log=True),
        reg_lambda=t.suggest_float("reg_lambda", 1e-8, 30.0, log=True),
        scale_pos_weight=t.suggest_float("scale_pos_weight", 1.0, 2 * spw),
        tree_method="hist", eval_metric="logloss", random_state=RNG, n_jobs=-1)


def make_cat(t):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=t.suggest_int("iterations", 100, 2000, step=50),
        learning_rate=t.suggest_float("learning_rate", 2e-3, 0.3, log=True),
        depth=t.suggest_int("depth", 3, 11),
        l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.5, 60.0, log=True),
        bagging_temperature=t.suggest_float("bagging_temperature", 0.0, 5.0),
        random_strength=t.suggest_float("random_strength", 1e-3, 20, log=True),
        border_count=t.suggest_int("border_count", 32, 255),
        auto_class_weights="Balanced", random_seed=RNG, verbose=False, thread_count=-1)


def make_rf(t):
    rf = RandomForestClassifier(
        n_estimators=t.suggest_int("n_estimators", 200, 1500, step=50),
        max_depth=t.suggest_int("max_depth", 3, 40),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 60),
        min_samples_split=t.suggest_int("min_samples_split", 2, 40),
        max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.8, 1.0]),
        max_samples=t.suggest_float("max_samples", 0.4, 1.0),
        bootstrap=True, class_weight="balanced_subsample", random_state=RNG, n_jobs=-1)
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", rf)])   # impute inside fold


def tune(name, builder, X, y, trials, timeout):
    print(f"\n=== tuning {name} ({trials} trials, TPE, <= {timeout//60} min) ===")
    sampler = optuna.samplers.TPESampler(seed=RNG, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda t: cv_auroc(builder(t), X, y), n_trials=trials, timeout=timeout,
                   show_progress_bar=True)                                    # <- TQDM bar
    print(f"    best 5-fold AUROC = {study.best_value:.4f}  ({len(study.trials)} trials done)")
    return study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=1500, help="per-model time cap (s)")
    args = ap.parse_args(); trials = args.trials

    X, y, feat, meta = ND.load(verbose=False)
    cyc = NP._load_cycles()
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    print(f"NHANES n={len(y)} HPV+={y.sum()} ({y.mean():.3f})  |  CPU, {trials} trials/model")

    builders = {
        "LightGBM":     lambda t: make_lgbm(t),
        "XGBoost":      lambda t: make_xgb(t, spw),
        "CatBoost":     lambda t: make_cat(t),
        "RandomForest": lambda t: make_rf(t),
    }
    studies = {name: tune(name, b, X, y, trials, args.timeout) for name, b in builders.items()}

    # rebuild the best estimator per model and run the FULL harness
    def best_est(name, study):
        t = optuna.trial.FixedTrial(study.best_params)
        return builders[name](t)

    final = {}
    for name, study in studies.items():
        est = best_est(name, study)
        kind = "impute" if name == "RandomForest" else "native"
        p = NP.repeated_oof(est, kind, X, y)
        auc = roc_auc_score(y, p); lo, hi = NP.boot_ci(y, p, roc_auc_score)
        _, per = NP.loco(est, kind, X, y, cyc)
        loco_auc = float(np.mean(list(per.values())))
        cal = NP.calibration_study(est, kind, X, y)
        op = NP.operating_points(est, kind, X, y)
        final[name] = dict(best_params=study.best_params, cv_auroc_5fold=float(study.best_value),
                           repeated_oof_auroc=float(auc), auroc_ci=[lo, hi],
                           loco_auroc=loco_auc, ece_isotonic=cal["isotonic"]["ece"],
                           pr_auc=float(NP.average_precision_score(y, p)),
                           operating_points=op, loco_per_cycle=per)
        print(f"  [{name}] tuned repeated-OOF AUROC={auc:.4f} [{lo:.3f},{hi:.3f}] "
              f"LOCO={loco_auc:.3f} ECE(iso)={cal['isotonic']['ece']:.3f}")

    best = max(final, key=lambda k: final[k]["repeated_oof_auroc"])
    print(f"\n>>> BEST tuned model: {best}  AUROC={final[best]['repeated_oof_auroc']:.4f} "
          f"{final[best]['auroc_ci']}")

    summary = dict(meta=meta, trials=trials, best=best, models=final,
                   note="CPU only; GPU benchmarked 4-18x slower for GBDTs at n=9348")
    json.dump(summary, open(os.path.join(MET, "nhanes_hpo_results.json"), "w"), indent=2)
    make_figures(studies, final, best)
    print("saved -> metrics/nhanes_hpo_results.json , figures/hpo_*")
    return summary


def make_figures(studies, final, best):
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 600,
                         "axes.spines.top": False, "axes.spines.right": False})

    # optimization history (best-so-far per model)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, study in studies.items():
        vals = [t.value for t in study.trials if t.value is not None]
        best_so_far = np.maximum.accumulate(vals)
        ax.plot(range(1, len(best_so_far) + 1), best_so_far, label=name, lw=1.8)
    ax.set_xlabel("trial"); ax.set_ylabel("best 5-fold AUROC so far")
    ax.set_title("Optuna optimization history"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "hpo_history.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "hpo_history.pdf"), bbox_inches="tight"); plt.close(fig)

    # tuned comparison with CI
    order = sorted(final, key=lambda k: final[k]["repeated_oof_auroc"])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    yv = np.arange(len(order))
    aur = [final[k]["repeated_oof_auroc"] for k in order]
    lo = [aur[i] - final[k]["auroc_ci"][0] for i, k in enumerate(order)]
    hi = [final[k]["auroc_ci"][1] - aur[i] for i, k in enumerate(order)]
    cols = ["#c0392b" if k == best else "#1a1a1a" for k in order]
    ax.errorbar(aur, yv, xerr=[lo, hi], fmt="o", ecolor="#999", capsize=3, linestyle="none")
    for i, k in enumerate(order):
        ax.scatter(final[k]["repeated_oof_auroc"], i, color=cols[i], zorder=5)
    ax.set_yticks(yv); ax.set_yticklabels(order); ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xlabel("tuned repeated-OOF AUROC (95% CI)")
    ax.set_title(f"Tuned models — best: {best}")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "hpo_tuned_comparison.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "hpo_tuned_comparison.pdf"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
