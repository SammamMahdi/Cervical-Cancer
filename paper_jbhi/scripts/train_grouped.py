"""Slide-grouped robustness run: same teacher->student recipe as the notebook, but
  * cropped single cells only (SIPaKMeD CROPPED/ folders; full-field slide images excluded), and
  * a 70/20/10 split made at the level of SOURCE SLIDES within each class, so no test cell shares a
    slide with any training cell.

Run in WSL (background):  python "/mnt/e/Cervaical Cancer/paper_jbhi/scripts/train_grouped.py"
Outputs: ../models/grouped_{teacher,student}.keras, ../results/grouped_*.json, ../results/grouped_preds.npz
"""
import os, json, glob, time
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_CACHE_MAXSIZE", "4294967296")
import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers

ROOT = "/mnt/e/Cervaical Cancer/data/sipakmed"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results"); MOD = os.path.join(HERE, "..", "models")
os.makedirs(OUT, exist_ok=True); os.makedirs(MOD, exist_ok=True)
CLASSES = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]
SEED, IMG, BATCH = 42, (380, 380), 8
LOG = open(os.path.join(OUT, "grouped_train.log"), "a")


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def grouped_split():
    rng = np.random.default_rng(SEED); split = {"train": [], "val": [], "test": []}
    for ci, c in enumerate(CLASSES):
        files = sorted(glob.glob(f"{ROOT}/im_{c}/im_{c}/CROPPED/*.bmp"))
        slides = sorted({os.path.basename(f).split("_")[0] for f in files})
        rng.shuffle(slides); n = len(slides)
        n_te, n_va = max(1, round(0.1 * n)), max(1, round(0.2 * n))
        part = {s: ("test" if i < n_te else "val" if i < n_te + n_va else "train") for i, s in enumerate(slides)}
        for f in files:
            split[part[os.path.basename(f).split("_")[0]]].append((f, ci))
        log(f"{c}: {len(files)} cells from {n} slides -> slides train/val/test = {n - n_te - n_va}/{n_va}/{n_te}")
    for k in split:
        rng.shuffle(split[k])
    return split


def load(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG, method="bilinear"); img.set_shape((IMG[0], IMG[1], 3))
    return img, label


def make_ds(items, shuffle):
    p = [a for a, _ in items]; y = [b for _, b in items]
    ds = tf.data.Dataset.from_tensor_slices((p, y))
    if shuffle:
        ds = ds.shuffle(len(p), seed=SEED, reshuffle_each_iteration=True)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)


def build_teacher():
    aug = keras.Sequential([layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.2),
                            layers.RandomZoom(0.1)], name="augmentation")
    base = keras.applications.EfficientNetB4(include_top=False, weights="imagenet", input_shape=IMG + (3,))
    base.trainable = False
    inp = keras.Input(IMG + (3,)); x = aug(inp); x = keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False); x = layers.GlobalAveragePooling2D()(x); x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x); x = layers.Dropout(0.5)(x)
    out = layers.Dense(5, activation="softmax", name="pred")(x)
    return keras.Model(inp, out, name="Teacher_B4_grouped"), base


def build_student():
    base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=IMG + (3,))
    inp = layers.Input(IMG + (3,)); x = base(inp); x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x); x = layers.Dense(512, activation="relu")(x); x = layers.Dropout(0.3)(x)
    return keras.Model(inp, layers.Dense(5, activation="softmax")(x), name="Student_B0_grouped")


class Distiller(keras.Model):
    """Identical to the notebook's distiller (alpha=0.1, T=4, KL between temperature-scaled softmaxes)."""
    def __init__(self, student, teacher):
        super().__init__(); self.student, self.teacher = student, teacher

    def compile(self, optimizer, metrics, alpha=0.1, temperature=4.0):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.sl = keras.losses.SparseCategoricalCrossentropy(reduction=keras.losses.Reduction.NONE)
        self.dl = keras.losses.KLDivergence(reduction=keras.losses.Reduction.NONE)
        self.alpha, self.T = alpha, temperature

    def train_step(self, data):
        x, y = data; t = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            s = self.student(x, training=True)
            sl = tf.nn.compute_average_loss(self.sl(y, s))
            dl = tf.nn.compute_average_loss(self.dl(tf.nn.softmax(t / self.T, -1), tf.nn.softmax(s / self.T, -1)))
            loss = self.alpha * sl + (1 - self.alpha) * dl
        g = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.student.trainable_variables))
        self.compiled_metrics.update_state(y, s)
        r = {m.name: m.result() for m in self.metrics}; r.update(student_loss=sl, distillation_loss=dl); return r

    def test_step(self, data):
        x, y = data; s = self.student(x, training=False)
        self.compiled_metrics.update_state(y, s)
        r = {m.name: m.result() for m in self.metrics}
        r.update(student_loss=tf.nn.compute_average_loss(self.sl(y, s))); return r


def hist(h):
    return {k: [float(v) for v in vs] for k, vs in h.history.items()}


def main():
    tf.random.set_seed(SEED); np.random.seed(SEED)
    split = grouped_split()
    log("cells train/val/test =", {k: len(v) for k, v in split.items()})
    json.dump({k: [p for p, _ in v] for k, v in split.items()}, open(os.path.join(OUT, "grouped_split.json"), "w"))
    tr, va, te = make_ds(split["train"], True), make_ds(split["val"], False), make_ds(split["test"], False)
    ytr = np.array([y for _, y in split["train"]])
    cw = {i: float(len(ytr) / (5 * max(1, (ytr == i).sum()))) for i in range(5)}

    t0 = time.time(); teacher, base = build_teacher()
    teacher.compile(keras.optimizers.Adam(1e-3), keras.losses.SparseCategoricalCrossentropy(), ["accuracy"])
    ck1 = os.path.join(MOD, "grouped_teacher_stage1.keras")
    h1 = teacher.fit(tr, validation_data=va, epochs=30, class_weight=cw, verbose=2, callbacks=[
        keras.callbacks.EarlyStopping("val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.3, patience=4, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(ck1, "val_loss", save_best_only=True)])
    log(f"stage1 done {time.time() - t0:.0f}s")
    base.trainable = True
    teacher.compile(keras.optimizers.Adam(1e-5), keras.losses.SparseCategoricalCrossentropy(), ["accuracy"])
    ck2 = os.path.join(MOD, "grouped_teacher.keras")
    h2 = teacher.fit(tr, validation_data=va, epochs=50, class_weight=cw, verbose=2, callbacks=[
        keras.callbacks.EarlyStopping("val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ck2, "val_loss", save_best_only=True)])
    teacher.save(ck2); log(f"stage2 done {time.time() - t0:.0f}s")

    student = build_student(); d = Distiller(student, teacher)
    d.compile(keras.optimizers.Adam(5e-5), [keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    h3 = d.fit(tr, validation_data=va, epochs=40, verbose=2,
               callbacks=[keras.callbacks.EarlyStopping("val_accuracy", patience=8, restore_best_weights=True, mode="max")])
    student.save(os.path.join(MOD, "grouped_student.keras")); log(f"KD done {time.time() - t0:.0f}s")

    y_te = np.array([y for _, y in split["test"]])
    P_t = teacher.predict(te, verbose=0); P_s = student.predict(te, verbose=0)
    np.savez_compressed(os.path.join(OUT, "grouped_preds.npz"), y_test=y_te, P_teacher=P_t, P_student=P_s,
                        test_paths=np.array([p for p, _ in split["test"]]))
    json.dump(dict(teacher_stage1=hist(h1), teacher_stage2=hist(h2), student_kd=hist(h3), class_weight=cw,
                   train_seconds=time.time() - t0), open(os.path.join(OUT, "grouped_histories.json"), "w"), indent=1)
    log(f"teacher test acc={np.mean(P_t.argmax(1) == y_te):.4f}  student test acc={np.mean(P_s.argmax(1) == y_te):.4f}")
    log("GROUPED_RUN_FINISHED")


if __name__ == "__main__":
    main()
