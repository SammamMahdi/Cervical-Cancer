"""U2b: hyperparameter optimization with Optuna (TPE) + Hyperband pruning.

Maximizes group-aware VALIDATION macro-F1; the 10% test stays sealed. Hyperband kills weak trials
early so many configs fit per GPU-hour. Searches lr, weight decay, optimizer, dropout, label
smoothing, and (with --kd) the KD temperature T / alpha. Saves the search space, budget, and best
config to results/metrics/hpo_study.json for reproducibility.

Usage:
    python hpo.py --trials 30 --epochs 15          # tune the student (CE)
    python hpo.py --trials 40 --epochs 15 --kd     # tune the distilled student (needs teacher.pt)
"""
import os, argparse
import numpy as np
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.metrics import f1_score

import config as C
import runlog
from data.dataset import make_loaders, class_weights
from models.backbones import build
from models.kd import DistillLoss
from train import _make_optimizer, _autocast
import eval.metrics as M


def _val_macro_f1(model, loader):
    y, probs = M.predict(model, loader)
    return f1_score(y, probs.argmax(1), average="macro")


def objective(trial, train_ld, val_ld, cw, max_epochs, teacher=None):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    optim = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    ls = trial.suggest_float("label_smoothing", 0.0, 0.1)

    model = build(C.STUDENT["backbone"], C.NUM_CLASSES, C.STUDENT["hidden"], dropout).to(C.DEVICE)
    opt = _make_optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr, optim, wd)
    if teacher is not None:
        T = trial.suggest_float("kd_T", 2.0, 6.0)
        alpha = trial.suggest_float("kd_alpha", 0.05, 0.5)
        distill = DistillLoss(temperature=T, alpha=alpha, label_smoothing=ls); crit = None
    else:
        crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=ls); distill = None

    best = 0.0
    for ep in range(1, max_epochs + 1):
        model.train()
        if teacher is not None: teacher.eval()
        for x, y in train_ld:
            x = x.to(C.DEVICE); y = y.to(C.DEVICE); opt.zero_grad(set_to_none=True)
            with _autocast():
                lo = model(x)
                if distill is not None:
                    with torch.no_grad(): tl = teacher(x)
                    loss, _, _ = distill(lo, tl, y)
                else:
                    loss = crit(lo, y)
            loss.backward(); opt.step()
        f1 = _val_macro_f1(model, val_ld); best = max(best, f1)
        trial.report(f1, ep)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best


def _map_overrides(best_params):
    """Optuna best-params -> student config overrides (kd_T->temperature, kd_alpha->alpha)."""
    ov = {}
    for k, v in best_params.items():
        if k == "kd_T": ov["temperature"] = v
        elif k == "kd_alpha": ov["alpha"] = v
        else: ov[k] = v
    return ov


def retrain_and_evaluate(best_params=None, tag_suffix="_hpo"):
    """U2b final step: re-fit the SINGLE best config (student) and evaluate ONCE on the sealed test
    + external sets. Reads results/metrics/hpo_study.json if best_params not given."""
    import train, run
    if best_params is None:
        d = runlog.load_json(os.path.join(C.METRICDIR, "hpo_study.json"))
        best_params = (d or {}).get("best_params", {})
    import ckpt as CK
    ov = _map_overrides(best_params)
    use_kd = ("temperature" in ov) or (CK.latest_path("teacher") is not None)
    print(f"[hpo] refitting best config (kd={use_kd}): {ov}")
    train.train_student(overrides=ov, tag_suffix=tag_suffix, distill=use_kd)
    return run.evaluate(which=f"student{tag_suffix}")


def run_hpo(n_trials=30, max_epochs=15, use_kd=False, seed=C.SEED, retrain=False):
    C.set_seed(seed)
    train_ld, val_ld, _, counts = make_loaders()
    cw = class_weights(counts).to(C.DEVICE)

    teacher = None
    if use_kd:
        import ckpt as CK
        if CK.latest_path("teacher") is not None:
            teacher = build(C.TEACHER["backbone"], C.NUM_CLASSES, C.TEACHER["hidden"],
                            C.TEACHER["dropout"], pretrained=False).to(C.DEVICE)
            CK.load_into(teacher, "teacher", map_location=C.DEVICE); teacher.eval()
            for p in teacher.parameters(): p.requires_grad = False
        else:
            print("[hpo] no teacher checkpoint found -> tuning student without KD")

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=seed),
        pruner=HyperbandPruner(min_resource=3, max_resource=max_epochs, reduction_factor=3))
    study.optimize(lambda t: objective(t, train_ld, val_ld, cw, max_epochs, teacher), n_trials=n_trials)

    space = ["lr", "weight_decay", "optimizer", "dropout", "label_smoothing"] + (["kd_T", "kd_alpha"] if teacher else [])
    out = dict(objective="group-aware val macro-F1 (test sealed)", n_trials=len(study.trials),
               max_epochs=max_epochs, sampler="TPE", pruner="Hyperband(rf=3)", seed=seed,
               search_space=space, best_value=float(study.best_value), best_params=study.best_params,
               pruned=sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials))
    runlog.dump_json(os.path.join(C.METRICDIR, "hpo_study.json"), out)
    print(f"[hpo] best val macro-F1 = {study.best_value:.4f}  ({out['pruned']}/{out['n_trials']} pruned)")
    print("[hpo] best params:", study.best_params)
    if retrain:
        out["post_hpo_eval"] = retrain_and_evaluate(study.best_params)
    return out


def run_tabular_hpo(n_trials=40, seed=C.SEED):
    """Separate Optuna study over the XGBoost tabular branch (§2b / U7); maximizes val macro-F1 on
    the leakage-safe split. Saves results/metrics/hpo_tabular.json."""
    from sklearn.metrics import f1_score
    import xgboost as xgb
    from explain.features import load_features
    d = load_features(); X, y, sp = d["X"], d["y"], d["split"]
    Xtr, ytr, Xva, yva = X[sp["train"]], y[sp["train"]], X[sp["val"]], y[sp["val"]]

    def obj(t):
        params = dict(max_depth=t.suggest_int("max_depth", 3, 10),
                      learning_rate=t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                      n_estimators=t.suggest_int("n_estimators", 100, 600),
                      subsample=t.suggest_float("subsample", 0.6, 1.0),
                      colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                      tree_method="hist")  # XGBClassifier infers objective/num_class from y
        m = xgb.XGBClassifier(**params, random_state=seed, n_jobs=4)
        m.fit(Xtr, ytr)
        return f1_score(yva, m.predict(Xva), average="macro")

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(obj, n_trials=n_trials)
    out = dict(objective="val macro-F1 (tabular XGBoost)", n_trials=len(study.trials),
               best_value=float(study.best_value), best_params=study.best_params)
    runlog.dump_json(os.path.join(C.METRICDIR, "hpo_tabular.json"), out)
    print(f"[tabular hpo] best val macro-F1 = {study.best_value:.4f}  {study.best_params}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--kd", action="store_true")
    ap.add_argument("--retrain", action="store_true", help="refit best config + evaluate on sealed test + external")
    ap.add_argument("--tabular", action="store_true", help="run the XGBoost tabular Optuna study instead")
    a = ap.parse_args()
    if a.tabular:
        run_tabular_hpo(a.trials)
    else:
        run_hpo(a.trials, a.epochs, a.kd, retrain=a.retrain)
