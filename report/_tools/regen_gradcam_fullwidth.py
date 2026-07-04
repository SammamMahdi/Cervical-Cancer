"""Regenerate the Grad-CAM figure as a tall, full-width 2-row panel:
   row 1 = original cell, row 2 = Grad-CAM overlay (true -> pred, confidence).
Reuses the legacy model + house style; same seed -> same representative cells.
Writes report/cvpr_paper/figures/gradcam_student.(png|pdf).
"""
import os, sys, glob, random
import numpy as np
import cv2
from PIL import Image
import torch

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "Legacy_src", "src")
sys.path.insert(0, os.path.abspath(SRC))
import config as C          # noqa: E402
import plotting as P        # noqa: E402  (applies house style on import)
import dataset as D         # noqa: E402
import evaluate as E        # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cvpr_paper", "figures", "gradcam_student"))


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.acts = None; self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o): self.acts = o.detach()
    def _bwd(self, m, gi, go): self.grads = go[0].detach()

    def __call__(self, x, cls=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if cls is None:
            cls = logits.argmax(1)
        logits.gather(1, cls.view(-1, 1)).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1))
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.cpu().numpy(), cls.cpu().numpy(), torch.softmax(logits, 1).detach().cpu().numpy()


def _overlay(img01, cam):
    cam = cv2.resize(cam.astype(np.float32), (img01.shape[1], img01.shape[0]))
    hm = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[:, :, ::-1] / 255.0
    return np.clip(0.45 * hm + 0.55 * img01, 0, 1)


def main():
    model = E.load_model("student"); model.eval()
    cam_extractor = GradCAM(model, model.last_conv_module)
    tf = D.get_transforms(C.IMG_SIZE, train=False)
    cdirs = D.find_class_dirs(); rng = random.Random(C.SEED)
    origs, overlays, titles = [], [], []
    for idx, rc in enumerate(C.RAW_CLASSES):
        files = [p for p in glob.glob(os.path.join(cdirs[rc], "**", "*"), recursive=True)
                 if p.lower().endswith(D.IMG_EXT) and "crop" in p.lower()]
        pil = Image.open(rng.choice(files)).convert("RGB")
        x = tf(pil).unsqueeze(0).to(C.DEVICE)
        cam, cls, prob = cam_extractor(x)
        disp = np.asarray(pil.resize((C.IMG_SIZE, C.IMG_SIZE))) / 255.0
        origs.append(disp)
        overlays.append(_overlay(disp, cam[0]))
        titles.append(f"{P.pretty(C.CLASSES[idx])} $\\rightarrow$ {P.pretty(C.CLASSES[int(cls[0])])}\n(conf {prob[0, int(cls[0])]:.2f})")
        print(titles[-1].replace(chr(10), " "))

    n = len(origs)
    # tall, full-width: 2 rows x n cols
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7.2), constrained_layout=True)
    for j in range(n):
        axes[0, j].imshow(np.clip(origs[j], 0, 1))
        axes[0, j].set_title(titles[j], fontsize=15, fontweight="bold")
        axes[1, j].imshow(np.clip(overlays[j], 0, 1))
        for r in (0, 1):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("Input cell", fontsize=17, fontweight="bold")
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=17, fontweight="bold")
    P.save(fig, OUT)
    print("Saved ->", OUT + ".png")


if __name__ == "__main__":
    main()
