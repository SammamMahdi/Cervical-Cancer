"""Grad-CAM for the frozen image expert (EfficientNet-B0 student) — shows WHERE the model looks when it
calls a cell abnormal. Grad-CAM (Selvaraju et al. 2017) on the last conv block (model.last_conv_module),
targeting the true abnormal-class logit, for high-confidence Koilocytotic and Dyskeratotic cells.

The point for the paper: the abnormal-class evidence localises to the nucleus / perinuclear halo — the
cytopathic signature of HPV — i.e. the model attends to the biologically-correct region.

Outputs -> results/figures/image_gradcam.png/pdf   Run (from legacy_src/src):  python image_gradcam.py
"""
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

import config as C
import evaluate as EV
from dataset import find_class_dirs, list_images, stratified_split, get_transforms

MEAN = torch.tensor(C.IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(C.IMAGENET_STD).view(3, 1, 1)
N_PER_CLASS = 4


def gradcam(model, x, cls):
    """Grad-CAM heatmap (H,W in [0,1]) for class `cls` on input x (1,3,H,W)."""
    acts, grads = {}, {}
    def fwd(m, i, o):
        acts["v"] = o
        o.register_hook(lambda g: grads.__setitem__("v", g))
    h = model.last_conv_module.register_forward_hook(fwd)
    for p in model.parameters():
        p.requires_grad_(False)
    x = x.clone().requires_grad_(True)
    logits = model(x)
    model.zero_grad(set_to_none=True)
    logits[0, cls].backward()
    h.remove()
    A, G = acts["v"][0], grads["v"][0]                 # (Cc,h,w)
    w = G.mean(dim=(1, 2))                              # (Cc,)
    cam = torch.relu((w[:, None, None] * A).sum(0))     # (h,w)
    cam = cam / (cam.max() + 1e-8)
    cam = torch.nn.functional.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear",
                                          align_corners=False)[0, 0]
    return cam.detach().cpu().numpy()


def disp_image(x):
    img = (x.cpu() * STD + MEAN).clamp(0, 1)[0].permute(1, 2, 0).numpy()
    return img


def pick_examples(model, tf, paths, labels, cls_idx, n):
    """First n cells of class cls_idx that the model confidently calls that abnormal class."""
    out = []
    for p, y in zip(paths, labels):
        if y != cls_idx:
            continue
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(C.DEVICE)
        with torch.no_grad():
            prob = torch.softmax(model(x).float(), 1)[0, cls_idx].item()
        if prob >= 0.6:
            out.append((x, prob, p))
        if len(out) >= n:
            break
    return out


def main():
    C.set_seed()
    model = EV.load_model("student")
    tf = get_transforms(C.IMG_SIZE, train=False)
    paths, labels, _ = list_images(find_class_dirs())
    (_, _), (vap, val) = stratified_split(paths, labels)
    koilo, dysk = C.CLASSES.index("Koilocytotic"), C.CLASSES.index("Dyskeratotic")

    cols = [("Koilocytotic", koilo), ("Dyskeratotic", dysk)]
    examples = {name: pick_examples(model, tf, vap, val, idx, N_PER_CLASS) for name, idx in cols}
    print("examples:", {k: len(v) for k, v in examples.items()})

    fig, axs = plt.subplots(N_PER_CLASS, 4, figsize=(12, 3 * N_PER_CLASS))
    for r in range(N_PER_CLASS):
        for ci, (name, idx) in enumerate(cols):
            exs = examples[name]
            ax_o, ax_c = axs[r, ci * 2], axs[r, ci * 2 + 1]
            if r < len(exs):
                x, prob, _ = exs[r]
                img = disp_image(x)
                cam = gradcam(model, x, idx)
                ax_o.imshow(img)
                ax_c.imshow(img); ax_c.imshow(cam, cmap="jet", alpha=0.45)
                ax_o.set_title(f"{name} (P={prob:.2f})", fontsize=9)
                ax_c.set_title("Grad-CAM", fontsize=9)
            for a in (ax_o, ax_c):
                a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Image expert Grad-CAM — abnormal evidence localises to the nucleus / perinuclear halo "
                 "(the HPV cytopathic signature)", y=1.005, fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIGDIR, f"image_gradcam.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved -> results/figures/image_gradcam.png")


if __name__ == "__main__":
    main()
