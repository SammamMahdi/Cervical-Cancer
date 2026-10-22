"""Evaluate the trained teacher (EfficientNet-B4) and distilled student (EfficientNet-B0) on a FIXED test set.

The notebook split SIPaKMeD 70/30 at file level (Keras, seed 42) and then cut the 30% holdout 2:1 into
validation/test by batches. Keras re-shuffles the holdout on every iteration, so that val/test boundary moved
between runs. Here we rebuild the identical file-level split and fix the test set as the last third of the
holdout file list (504 files). No holdout file was ever used for gradient updates.

Run in WSL:  python "/mnt/e/Cervaical Cancer/paper_jbhi/scripts/eval_image.py"
Outputs: ../results/image_eval.json, ../results/image_preds.npz
"""
import os, json, time
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras.src.utils import dataset_utils
from tf_keras.src.utils.image_dataset import ALLOWLIST_FORMATS
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score,
                             roc_auc_score, average_precision_score, precision_recall_fscore_support,
                             confusion_matrix)

ROOT = "/mnt/e/Cervaical Cancer/data/sipakmed"
MODELS = {"teacher": "/mnt/e/Cervaical Cancer/Legacy_src/teacher_sipakmed_final.keras",
          "student": "/mnt/e/Cervaical Cancer/Legacy_src/student_sipakmed_final_distilled.keras"}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results"); os.makedirs(OUT, exist_ok=True)
SEED, IMG, HOLDOUT, BATCH = 42, (380, 380), 0.3, 8
CLASSES = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]
ABNORMAL = [0, 1]   # Bethesda-style collapse: Dyskeratotic + Koilocytotic = abnormal (LSIL/HSIL spectrum)
RNG = np.random.default_rng(0)


def split_files():
    paths, labels, names = dataset_utils.index_directory(ROOT, "inferred", formats=ALLOWLIST_FORMATS,
                                                         class_names=None, shuffle=True, seed=SEED)
    tr_p, tr_y = dataset_utils.get_training_or_validation_split(paths, labels, HOLDOUT, "training")
    ho_p, ho_y = dataset_utils.get_training_or_validation_split(paths, labels, HOLDOUT, "validation")
    n_ho_batches = int(np.ceil(len(ho_p) / BATCH))
    n_val = int(round(n_ho_batches * (0.2 / HOLDOUT))) * BATCH
    return names, (tr_p, np.array(tr_y)), (ho_p[:n_val], np.array(ho_y[:n_val])), (ho_p[n_val:], np.array(ho_y[n_val:]))


def load(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG, method="bilinear")
    img.set_shape((IMG[0], IMG[1], 3))
    return img


def predict(model, paths):
    ds = tf.data.Dataset.from_tensor_slices(list(paths)).map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return model.predict(ds.batch(32).prefetch(tf.data.AUTOTUNE), verbose=0)


def ece(y, P, bins=15):
    conf, pred = P.max(1), P.argmax(1); acc = (pred == y).astype(float); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def boot_ci(fn, y, P, n=1000):
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < len(CLASSES):
            continue
        vals.append(fn(y[idx], P[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def metrics(y, P):
    pred = P.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(5), zero_division=0)
    yb = np.isin(y, ABNORMAL).astype(int); pb = P[:, ABNORMAL].sum(1); predb = (pb >= 0.5).astype(int)
    tn, fp, fn_, tp = confusion_matrix(yb, predb, labels=[0, 1]).ravel()
    acc_fn = lambda yy, PP: accuracy_score(yy, PP.argmax(1))
    auc_fn = lambda yy, PP: roc_auc_score(yy, PP, multi_class="ovr", average="macro")
    return dict(
        n=int(len(y)), accuracy=float(accuracy_score(y, pred)), accuracy_ci95=boot_ci(acc_fn, y, P),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average="macro")),
        cohen_kappa=float(cohen_kappa_score(y, pred)),
        macro_auroc_ovr=float(auc_fn(y, P)), macro_auroc_ci95=boot_ci(auc_fn, y, P),
        ece=ece(y, P), brier=float(np.mean(np.sum((P - np.eye(5)[y]) ** 2, 1))),
        per_class={c: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(s[i]))
                   for i, c in enumerate(CLASSES)},
        confusion=confusion_matrix(y, pred, labels=range(5)).tolist(),
        binary=dict(auroc=float(roc_auc_score(yb, pb)), ap=float(average_precision_score(yb, pb)),
                    sensitivity=float(tp / (tp + fn_)), specificity=float(tn / (tn + fp)),
                    accuracy=float((tp + tn) / len(yb)), prevalence=float(yb.mean())))


def model_cost(model):
    params = int(model.count_params())
    x1 = tf.zeros((1, IMG[0], IMG[1], 3)); x32 = tf.zeros((32, IMG[0], IMG[1], 3))
    for _ in range(5):
        model(x1, training=False); model(x32, training=False)
    t = time.perf_counter(); [model(x1, training=False) for _ in range(50)]; lat = (time.perf_counter() - t) / 50
    t = time.perf_counter(); [model(x32, training=False) for _ in range(10)]; thr = 320 / (time.perf_counter() - t)
    return dict(params=params, params_millions=params / 1e6, latency_ms_batch1=lat * 1000, throughput_img_per_s=thr)


def main():
    names, (tr_p, tr_y), (va_p, va_y), (te_p, te_y) = split_files()
    print(f"classes={names}\ntrain={len(tr_p)} val={len(va_p)} test={len(te_p)}")
    assert [n.replace("im_", "") for n in names] == CLASSES

    # composition + parent-slide overlap diagnostics (full-field vs cropped cells)
    def slide_key(p):
        cls = p.split("/sipakmed/")[1].split("/")[0]; base = os.path.splitext(os.path.basename(p))[0]
        return cls + ":" + base.split("_")[0]
    tr_keys = {slide_key(p) for p in tr_p}
    te_crop = np.array(["/CROPPED/" in p for p in te_p])
    overlap = np.array([slide_key(p) in tr_keys for p in te_p])
    comp = dict(test_crops=int(te_crop.sum()), test_fullfield=int((~te_crop).sum()),
                test_crops_with_parent_slide_in_train=int((te_crop & overlap).sum()),
                train_crops=int(sum("/CROPPED/" in p for p in tr_p)),
                train_fullfield=int(sum("/CROPPED/" not in p for p in tr_p)))
    print(comp)

    res = dict(split=dict(train=len(tr_p), val=len(va_p), test=len(te_p), seed=SEED, composition=comp),
               classes=CLASSES)
    store = dict(test_paths=np.array(te_p), y_test=te_y, holdout_paths=np.array(va_p + te_p),
                 y_holdout=np.concatenate([va_y, te_y]), test_is_crop=te_crop, test_parent_in_train=overlap)
    for tag, path in MODELS.items():
        model = keras.models.load_model(path, compile=False)
        P = predict(model, te_p); Ph = predict(model, va_p + te_p)
        m = metrics(te_y, P)
        m["by_image_type"] = {k: float(accuracy_score(te_y[msk], P[msk].argmax(1)))
                              for k, msk in (("cropped_cells", te_crop), ("full_field", ~te_crop)) if msk.any()}
        m["full_holdout"] = {k: v for k, v in metrics(np.concatenate([va_y, te_y]), Ph).items()
                             if k in ("n", "accuracy", "macro_auroc_ovr", "ece", "macro_f1")}
        m["cost"] = model_cost(model)
        m["file_size_mb"] = os.path.getsize(path) / 1e6
        res[tag] = m; store[f"P_{tag}"] = P; store[f"Ph_{tag}"] = Ph
        print(f"[{tag}] acc={m['accuracy']:.4f} macroAUROC={m['macro_auroc_ovr']:.4f} ECE={m['ece']:.4f} "
              f"bin-AUROC={m['binary']['auroc']:.4f} params={m['cost']['params_millions']:.2f}M "
              f"lat={m['cost']['latency_ms_batch1']:.1f}ms by_type={m['by_image_type']}")
        del model; keras.backend.clear_session()

    json.dump(res, open(os.path.join(OUT, "image_eval.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(OUT, "image_preds.npz"), **store)
    print("saved results/image_eval.json + image_preds.npz")


if __name__ == "__main__":
    main()
