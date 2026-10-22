"""Grad-CAM for the trained teacher (B4) and student (B0) on held-out test cells.

The notebook's Grad-CAM fed the `top_conv` activations straight into the classification head, skipping
EfficientNet's `top_bn` and `top_activation` layers, i.e. it evaluated a different network (every example was
"predicted" Parabasal). Here the forward pass is rebuilt layer-by-layer from `top_conv` through the real head, and
the rebuilt prediction is asserted to match model.predict before any map is drawn.

Run in WSL:  python "/mnt/e/Cervaical Cancer/paper_jbhi/scripts/gradcam.py"   -> ../figures/img_gradcam.{pdf,png}
"""
import os, sys
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import tf_keras as keras
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pubstyle import plt, save, size  # noqa: E402
import matplotlib.cm as cm  # noqa: E402

RES = os.path.join(HERE, "..", "results")
MODELS = {"Teacher B4": "/mnt/e/Cervaical Cancer/Legacy_src/teacher_sipakmed_final.keras",
          "Student B0": "/mnt/e/Cervaical Cancer/Legacy_src/student_sipakmed_final_distilled.keras"}
CLS = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]
SHORT = ["Dysk.", "Koilo.", "Metap.", "Parab.", "Sup.-Int."]


def load(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return tf.image.resize(img, (380, 380), method="bilinear").numpy()


class CAM:
    def __init__(self, model):
        self.model = model
        names = [l.name for l in model.layers]
        bi = next(i for i, l in enumerate(model.layers) if l.name.startswith("efficientnet"))
        self.base = model.layers[bi]
        self.pre = model.layers[1:bi]                       # augmentation / preprocessing (inference = identity)
        self.head = model.layers[bi + 1:]
        self.to_conv = keras.Model(self.base.inputs, self.base.get_layer("top_conv").output)
        self.top = [self.base.get_layer("top_bn"), self.base.get_layer("top_activation")]
        print("layers:", names)

    def forward(self, x, watch=None):
        for l in self.pre:
            x = l(x, training=False)
        conv = self.to_conv(x, training=False)
        if watch is not None:
            watch.watch(conv)
        h = conv
        for l in self.top + self.head:
            h = l(h, training=False)
        return conv, h

    def logits(self, x, watch=None):
        """Forward pass that returns the PRE-SOFTMAX class scores (Grad-CAM on saturated softmax outputs vanishes)."""
        for l in self.pre:
            x = l(x, training=False)
        conv = self.to_conv(x, training=False)
        if watch is not None:
            watch.watch(conv)
        h = conv
        for l in self.top + self.head[:-1]:
            h = l(h, training=False)
        last = self.head[-1]; W, b = last.kernel, last.bias
        return conv, tf.matmul(h, W) + b

    def __call__(self, img):
        x = tf.convert_to_tensor(img[None])
        with tf.GradientTape() as tape:
            conv, z = self.logits(x, tape); k = int(tf.argmax(z[0])); score = z[:, k]
        pk = float(tf.nn.softmax(z[0])[k])
        g = tape.gradient(score, conv)[0]; w = tf.reduce_mean(g, axis=(0, 1))
        cam = tf.nn.relu(tf.reduce_sum(conv[0] * w, -1)).numpy()
        cam = cam / (cam.max() + 1e-9)
        return tf.image.resize(cam[..., None], (380, 380)).numpy()[..., 0], k, pk


def main():
    d = np.load(os.path.join(RES, "image_preds.npz"), allow_pickle=True)
    paths, y, P = d["test_paths"], d["y_test"], d["P_student"]
    is_crop = np.array(["/CROPPED/" in str(p) for p in paths])
    rng = np.random.default_rng(7); pick = []
    for c in range(5):                                       # one correctly-classified test cell per class
        idx = np.where(is_crop & (y == c) & (P.argmax(1) == c))[0]; pick.append(int(rng.choice(idx)))
    cams = {}
    for tag, path in MODELS.items():
        model = keras.models.load_model(path, compile=False); cam = CAM(model)
        imgs = np.stack([load(str(paths[i])) for i in pick])
        ref = model.predict(imgs, verbose=0); _, man = cam.forward(tf.convert_to_tensor(imgs))
        _, z = cam.logits(tf.convert_to_tensor(imgs))
        err = float(np.abs(ref - man.numpy()).max()); err2 = float(np.abs(ref - tf.nn.softmax(z).numpy()).max())
        print(f"[{tag}] max |manual - predict| = {err:.2e}; max |softmax(logits) - predict| = {err2:.2e}")
        assert err < 1e-3 and err2 < 1e-3, "rebuilt forward pass does not match the model"
        cams[tag] = [cam(im) for im in imgs]
        del model; keras.backend.clear_session()

    fig, axs = plt.subplots(3, 5, figsize=size(11.5, 7.36))
    for j, i in enumerate(pick):
        im = load(str(paths[i])) / 255.0
        axs[0, j].imshow(im); axs[0, j].set_title(SHORT[y[i]], fontsize=8)
        for r, tag in enumerate(MODELS, start=1):
            m, k, pk = cams[tag][j]
            axs[r, j].imshow(im); axs[r, j].imshow(cm.jet(m)[..., :3], alpha=0.45)
            axs[r, j].set_title(f"{SHORT[k]} ({pk:.2f})", fontsize=8)
    for r, lab in enumerate(["input cell"] + list(MODELS)):
        axs[r, 0].set_ylabel(lab, fontsize=8.7)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
    save(fig, "img_gradcam"); print("saved img_gradcam")


if __name__ == "__main__":
    main()
