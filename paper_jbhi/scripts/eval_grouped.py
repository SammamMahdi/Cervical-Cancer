"""Metrics for the slide-grouped robustness run (Protocol B), same definitions as eval_image.py.
Run (Windows python):  python paper_jbhi/scripts/eval_grouped.py   -> ../results/grouped_eval.json
"""
import os, json
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, roc_auc_score,
                             average_precision_score, confusion_matrix, precision_recall_fscore_support)

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
CLASSES = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]
RNG = np.random.default_rng(0)


def ece(y, P, bins=15):
    conf, pred = P.max(1), P.argmax(1); acc = (pred == y).astype(float); e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def ci(fn, y, P, n=1000):
    v = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 5:
            v.append(fn(y[i], P[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def metrics(y, P):
    pred = P.argmax(1); p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(5), zero_division=0)
    yb = np.isin(y, [0, 1]).astype(int); pb = P[:, [0, 1]].sum(1); predb = (pb >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, predb, labels=[0, 1]).ravel()
    auc = lambda yy, PP: roc_auc_score(yy, PP, multi_class="ovr", average="macro")
    return dict(n=int(len(y)), accuracy=float(accuracy_score(y, pred)),
                accuracy_ci95=ci(lambda a, b: accuracy_score(a, b.argmax(1)), y, P),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average="macro")),
                cohen_kappa=float(cohen_kappa_score(y, pred)), macro_auroc_ovr=float(auc(y, P)), macro_auroc_ci95=ci(auc, y, P),
                ece=ece(y, P), per_class={c: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(s[i]))
                                          for i, c in enumerate(CLASSES)},
                confusion=confusion_matrix(y, pred, labels=range(5)).tolist(),
                binary=dict(auroc=float(roc_auc_score(yb, pb)), ap=float(average_precision_score(yb, pb)),
                            sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp))))


if __name__ == "__main__":
    g = np.load(os.path.join(RES, "grouped_preds.npz")); y = g["y_test"]
    h = json.load(open(os.path.join(RES, "grouped_histories.json")))
    out = dict(protocol="B: slide-grouped, cropped cells only", n_test=int(len(y)),
               teacher=metrics(y, g["P_teacher"]), student=metrics(y, g["P_student"]),
               epochs=dict(stage1=len(h["teacher_stage1"]["loss"]), stage2=len(h["teacher_stage2"]["loss"]),
                           kd=len(h["student_kd"]["accuracy"])), train_minutes=h["train_seconds"] / 60)
    json.dump(out, open(os.path.join(RES, "grouped_eval.json"), "w"), indent=2)
    for m in ("teacher", "student"):
        e = out[m]
        print(f"{m}: acc={e['accuracy']:.4f} {e['accuracy_ci95']} F1={e['macro_f1']:.4f} AUROC={e['macro_auroc_ovr']:.4f} "
              f"{e['macro_auroc_ci95']} binAUROC={e['binary']['auroc']:.4f} sens={e['binary']['sensitivity']:.3f} "
              f"spec={e['binary']['specificity']:.3f} ECE={e['ece']:.3f}")
    print("epochs", out["epochs"], "train min", round(out["train_minutes"], 1))
