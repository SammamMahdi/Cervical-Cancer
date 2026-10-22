"""External validation of the cytology expert on the Herlev Pap-smear data set.

SIPaKMeD and Herlev differ in acquisition, staining and class vocabulary, so the models are applied with no
retraining and the five SIPaKMeD classes are collapsed to the Bethesda-style binary decision the fusion uses:
    abnormal = dyskeratotic + koilocytotic
Herlev classes map as: normal_{columnar,intermediate,superficiel} -> normal;
{light,moderate,severe}_dysplastic and carcinoma_in_situ -> abnormal.

Run (CPU, so it can share the machine with training):
  CUDA_VISIBLE_DEVICES= python eval_external.py   -> ../results/external_eval.json
"""
import os, sys, glob, json
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
HERLEV = "/mnt/e/Cervaical Cancer/data/raw/herlev"
MODELS = {"student (Protocol A)": "/mnt/e/Cervaical Cancer/Legacy_src/student_sipakmed_final_distilled.keras",
          "student (Protocol B, slide-grouped)": os.path.join(HERE, "..", "models", "grouped_student.keras")}
NORMAL = ("normal_columnar", "normal_intermediate", "normal_superficiel")
ABNORMAL = ("light_dysplastic", "moderate_dysplastic", "severe_dysplastic", "carcinoma_in_situ")
ABN_IDX = [0, 1]
RNG = np.random.default_rng(0)


def files_and_labels():
    paths, y, cls = [], [], []
    for c in NORMAL + ABNORMAL:
        fs = sorted(sum([glob.glob(os.path.join(HERLEV, "**", c, f"*.{e}"), recursive=True)
                         for e in ("bmp", "BMP", "jpg", "png")], []))
        fs = [f for f in fs if "-d." not in os.path.basename(f)]      # drop the segmentation masks
        paths += fs; y += [int(c in ABNORMAL)] * len(fs); cls += [c] * len(fs)
    return paths, np.array(y), np.array(cls)


def predict(model, paths, size=(380, 380), batch=32):
    def load(p):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(img, size, method="bilinear"); img.set_shape(size + (3,))
        return img
    ds = tf.data.Dataset.from_tensor_slices(paths).map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return model.predict(ds.batch(batch).prefetch(tf.data.AUTOTUNE), verbose=0)


def ece(y, p, bins=10):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def boot_ci(y, p, fn, n=1000):
    v = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(fn(y[i], p[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    paths, y, cls = files_and_labels()
    print(f"Herlev: {len(paths)} images, {y.sum()} abnormal ({y.mean():.3f}); classes "
          f"{ {c: int((cls == c).sum()) for c in sorted(set(cls))} }")
    out = dict(dataset="Herlev", n=len(paths), prevalence=float(y.mean()),
               per_class={c: int((cls == c).sum()) for c in sorted(set(cls))}, models={})
    for tag, path in MODELS.items():
        if not os.path.exists(path):
            print("missing model:", path); continue
        m = keras.models.load_model(path, compile=False)
        P = predict(m, paths)
        if P.min() < 0 or P.max() > 1:                       # the variant students output logits
            P = tf.nn.softmax(P, -1).numpy()
        p_abn = P[:, ABN_IDX].sum(1)
        pred = (p_abn >= 0.5).astype(int)
        tn, fp, fn_, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rec = dict(auroc=float(roc_auc_score(y, p_abn)), auroc_ci95=boot_ci(y, p_abn, roc_auc_score),
                   ap=float(average_precision_score(y, p_abn)), ece=ece(y, p_abn),
                   accuracy=float((tp + tn) / len(y)), sensitivity=float(tp / (tp + fn_)),
                   specificity=float(tn / (tn + fp)), mean_score=float(p_abn.mean()),
                   per_class_mean_score={c: float(p_abn[cls == c].mean()) for c in sorted(set(cls))})
        out["models"][tag] = rec
        print(f"[{tag}] AUROC {rec['auroc']:.4f} {['%.3f' % v for v in rec['auroc_ci95']]} "
              f"sens {rec['sensitivity']:.3f} spec {rec['specificity']:.3f} ECE {rec['ece']:.3f}")
        np.savez_compressed(os.path.join(RES, "external_" + ("protocolA" if "Protocol A" in tag else "protocolB") + ".npz"),
                            y=y, p=p_abn, cls=cls)
        del m; keras.backend.clear_session()
    json.dump(out, open(os.path.join(RES, "external_eval.json"), "w"), indent=2)
    print("saved results/external_eval.json")


if __name__ == "__main__":
    main()
