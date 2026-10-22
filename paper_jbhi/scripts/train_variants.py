"""Seed repeats and knowledge-distillation ablation on the slide-grouped protocol.

Addresses two reviewer requests:
  * multiple independent training runs, so intervals reflect training instability and not only test sampling;
  * an ablation of the distillation loss, because the notebook applied the temperature to the teacher's
    *probabilities* rather than its logits, which is not standard knowledge distillation.

KD modes
  none      student trained on hard labels only (alpha = 1)
  prob_T    as in the main paper: KL between softmax(softmax(z)/T) of teacher and student
  logit_T   standard KD: KL between softmax(z/T) of teacher and student
  logit_T2  standard KD with the usual T^2 scaling of the distillation term

Run (WSL, detached):  python train_variants.py [seeds|kd|all]
Outputs: ../results/variants/<tag>.json and <tag>_preds.npz
"""
import os, sys, json, time, glob
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_CACHE_MAXSIZE", "4294967296")
import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers

ROOT = "/mnt/e/Cervaical Cancer/data/sipakmed"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "variants"); MOD = os.path.join(HERE, "..", "models")
os.makedirs(OUT, exist_ok=True); os.makedirs(MOD, exist_ok=True)
CLASSES = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]
IMG, BATCH = (380, 380), 8


def log(*a):
    print(" ".join(str(x) for x in a), flush=True)


def grouped_split(seed):
    """70/20/10 split of cropped cells by SOURCE SLIDE within each class."""
    rng = np.random.default_rng(seed); split = {"train": [], "val": [], "test": []}
    for ci, c in enumerate(CLASSES):
        files = sorted(glob.glob(f"{ROOT}/im_{c}/im_{c}/CROPPED/*.bmp"))
        slides = sorted({os.path.basename(f).split("_")[0] for f in files})
        rng.shuffle(slides); n = len(slides)
        n_te, n_va = max(1, round(0.1 * n)), max(1, round(0.2 * n))
        part = {s: ("test" if i < n_te else "val" if i < n_te + n_va else "train") for i, s in enumerate(slides)}
        for f in files:
            split[part[os.path.basename(f).split("_")[0]]].append((f, ci))
    for k in split:
        rng.shuffle(split[k])
    return split


def load(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG, method="bilinear"); img.set_shape((IMG[0], IMG[1], 3))
    return img, label


def make_ds(items, shuffle, seed):
    p = [a for a, _ in items]; y = [b for _, b in items]
    ds = tf.data.Dataset.from_tensor_slices((p, y))
    if shuffle:
        ds = ds.shuffle(len(p), seed=seed, reshuffle_each_iteration=True)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)


def build_teacher():
    aug = keras.Sequential([layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.2),
                            layers.RandomZoom(0.1)], name="augmentation")
    base = keras.applications.EfficientNetB4(include_top=False, weights="imagenet", input_shape=IMG + (3,))
    base.trainable = False
    inp = keras.Input(IMG + (3,)); x = aug(inp); x = keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False); x = layers.GlobalAveragePooling2D()(x); x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x); x = layers.Dropout(0.5)(x)
    return keras.Model(inp, layers.Dense(5, activation="softmax", name="pred")(x)), base


def build_student_logits():
    """Student whose head is linear; probabilities are taken with an explicit softmax where needed."""
    base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=IMG + (3,))
    inp = layers.Input(IMG + (3,)); x = base(inp); x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x); x = layers.Dense(512, activation="relu")(x); x = layers.Dropout(0.3)(x)
    return keras.Model(inp, layers.Dense(5, activation=None)(x), name="Student_B0_logits")


class Distiller(keras.Model):
    def __init__(self, student, teacher, mode, alpha=0.1, T=4.0):
        super().__init__(); self.student, self.teacher, self.mode, self.alpha, self.T = student, teacher, mode, alpha, T

    def compile(self, optimizer, metrics):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        self.kl = keras.losses.KLDivergence(reduction=keras.losses.Reduction.NONE)

    def _kd(self, t_prob, s_logit):
        if self.mode == "prob_T":                      # paper's implementation: temperature on probabilities
            return self.kl(tf.nn.softmax(t_prob / self.T, -1), tf.nn.softmax(tf.nn.softmax(s_logit, -1) / self.T, -1))
        t_logit = tf.math.log(tf.clip_by_value(t_prob, 1e-8, 1.0))     # recover logits up to a constant
        d = self.kl(tf.nn.softmax(t_logit / self.T, -1), tf.nn.softmax(s_logit / self.T, -1))
        return d * (self.T ** 2) if self.mode == "logit_T2" else d

    def train_step(self, data):
        x, y = data; t = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            s = self.student(x, training=True)
            ce = tf.nn.compute_average_loss(self.ce(y, s))
            if self.mode == "none":
                loss, kd = ce, tf.constant(0.0)
            else:
                kd = tf.nn.compute_average_loss(self._kd(t, s)); loss = self.alpha * ce + (1 - self.alpha) * kd
        g = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.student.trainable_variables))
        self.compiled_metrics.update_state(y, tf.nn.softmax(s, -1))
        r = {m.name: m.result() for m in self.metrics}; r.update(ce=ce, kd=kd); return r

    def test_step(self, data):
        x, y = data; s = self.student(x, training=False)
        self.compiled_metrics.update_state(y, tf.nn.softmax(s, -1))
        r = {m.name: m.result() for m in self.metrics}
        r.update(ce=tf.nn.compute_average_loss(self.ce(y, s))); return r


def metrics_of(y, P):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    yb = np.isin(y, [0, 1]).astype(int)
    return dict(accuracy=float(accuracy_score(y, P.argmax(1))), macro_f1=float(f1_score(y, P.argmax(1), average="macro")),
                macro_auroc=float(roc_auc_score(y, P, multi_class="ovr")),
                binary_auroc=float(roc_auc_score(yb, P[:, [0, 1]].sum(1))), n=int(len(y)))


def train_teacher(split, seed, tag):
    tr, va = make_ds(split["train"], True, seed), make_ds(split["val"], False, seed)
    ytr = np.array([y for _, y in split["train"]])
    cw = {i: float(len(ytr) / (5 * max(1, (ytr == i).sum()))) for i in range(5)}
    teacher, base = build_teacher()
    teacher.compile(keras.optimizers.Adam(1e-3), keras.losses.SparseCategoricalCrossentropy(), ["accuracy"])
    teacher.fit(tr, validation_data=va, epochs=30, class_weight=cw, verbose=2, callbacks=[
        keras.callbacks.EarlyStopping("val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.3, patience=4, min_lr=1e-6)])
    base.trainable = True
    teacher.compile(keras.optimizers.Adam(1e-5), keras.losses.SparseCategoricalCrossentropy(), ["accuracy"])
    teacher.fit(tr, validation_data=va, epochs=50, class_weight=cw, verbose=2, callbacks=[
        keras.callbacks.EarlyStopping("val_loss", patience=6, restore_best_weights=True)])
    teacher.save(os.path.join(MOD, f"{tag}_teacher.keras"))
    return teacher


def train_student(split, seed, teacher, mode):
    tr, va = make_ds(split["train"], True, seed), make_ds(split["val"], False, seed)
    st = build_student_logits(); d = Distiller(st, teacher, mode)
    d.compile(keras.optimizers.Adam(5e-5), [keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    d.fit(tr, validation_data=va, epochs=40, verbose=2, callbacks=[
        keras.callbacks.EarlyStopping("val_accuracy", patience=8, restore_best_weights=True, mode="max")])
    return st


def evaluate(model, split, softmax):
    te = make_ds(split["test"], False, 0)
    P = model.predict(te, verbose=0)
    if softmax:
        P = tf.nn.softmax(P, -1).numpy()
    return np.array([y for _, y in split["test"]]), P


def run_seeds(seeds=(1, 2)):
    for seed in seeds:
        t0 = time.time(); tag = f"seed{seed}"
        log(f"=== {tag}: grouped teacher + student (paper KD) ===")
        tf.random.set_seed(seed); np.random.seed(seed)
        split = grouped_split(seed)
        teacher = train_teacher(split, seed, tag)
        yt, Pt = evaluate(teacher, split, softmax=False)
        st = train_student(split, seed, teacher, "prob_T")
        ys, Ps = evaluate(st, split, softmax=True)
        res = dict(tag=tag, seed=seed, kd_mode="prob_T", teacher=metrics_of(yt, Pt), student=metrics_of(ys, Ps),
                   minutes=(time.time() - t0) / 60, n_train=len(split["train"]), n_test=len(split["test"]))
        json.dump(res, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)
        np.savez_compressed(os.path.join(OUT, f"{tag}_preds.npz"), y=ys, P_student=Ps, P_teacher=Pt,
                            paths=np.array([p for p, _ in split["test"]]))
        log(f"{tag}: teacher {res['teacher']['accuracy']:.4f} student {res['student']['accuracy']:.4f} "
            f"({res['minutes']:.0f} min)")
        del teacher, st; keras.backend.clear_session()


def run_kd(modes=("none", "logit_T", "logit_T2")):
    """Ablation on the seed-42 teacher already trained for the paper."""
    seed = 42; split = grouped_split(seed)
    teacher = keras.models.load_model(os.path.join(MOD, "grouped_teacher.keras"), compile=False)
    for mode in modes:
        t0 = time.time(); tag = f"kd_{mode}"
        log(f"=== {tag} ===")
        tf.random.set_seed(seed); np.random.seed(seed)
        st = train_student(split, seed, teacher, mode)
        y, P = evaluate(st, split, softmax=True)
        res = dict(tag=tag, seed=seed, kd_mode=mode, student=metrics_of(y, P), minutes=(time.time() - t0) / 60)
        json.dump(res, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)
        np.savez_compressed(os.path.join(OUT, f"{tag}_preds.npz"), y=y, P_student=P,
                            paths=np.array([p for p, _ in split["test"]]))
        log(f"{tag}: student {res['student']['accuracy']:.4f} macroAUROC {res['student']['macro_auroc']:.4f} "
            f"({res['minutes']:.0f} min)")
        del st; keras.backend.clear_session()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("seeds", "all"):
        run_seeds()
    if what in ("kd", "all"):
        run_kd()
    log("VARIANTS_FINISHED")
