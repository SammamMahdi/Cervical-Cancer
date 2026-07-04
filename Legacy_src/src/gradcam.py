"""Grad-CAM for the trained EfficientNet (last conv block), house-style overlay panels.

Saves results/figures/gradcam_<which>.(pdf|png): one sample cell per SIPaKMeD class with its
Grad-CAM heatmap overlaid, titled true -> predicted (confidence).
"""
import os, glob, random
import numpy as np
import cv2
from PIL import Image
import torch

import config as C
import plotting as P
import dataset as D
import evaluate as E


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.acts = None; self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o): self.acts = o.detach()
    def _bwd(self, m, gi, go): self.grads = go[0].detach()

    def __call__(self, x, cls=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)                              # float32 (no autocast — needs grads)
        if cls is None:
            cls = logits.argmax(1)
        logits.gather(1, cls.view(-1, 1)).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)        # GAP of grads
        cam = torch.relu((w * self.acts).sum(1))             # (B,H,W)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.cpu().numpy(), cls.cpu().numpy(), torch.softmax(logits, 1).detach().cpu().numpy()


def _overlay(img01, cam):
    cam = cv2.resize(cam.astype(np.float32), (img01.shape[1], img01.shape[0]))
    hm = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[:, :, ::-1] / 255.0   # BGR->RGB
    return np.clip(0.45 * hm + 0.55 * img01, 0, 1)


def run_gradcam(which="student", out=None):
    model = E.load_model(which); model.eval()
    cam_extractor = GradCAM(model, model.last_conv_module)
    tf = D.get_transforms(C.IMG_SIZE, train=False)
    cdirs = D.find_class_dirs(); rng = random.Random(C.SEED)
    panels, titles = [], []
    for idx, rc in enumerate(C.RAW_CLASSES):
        files = [p for p in glob.glob(os.path.join(cdirs[rc], "**", "*"), recursive=True)
                 if p.lower().endswith(D.IMG_EXT) and "crop" in p.lower()]
        pil = Image.open(rng.choice(files)).convert("RGB")
        x = tf(pil).unsqueeze(0).to(C.DEVICE)
        cam, cls, prob = cam_extractor(x)
        disp = np.asarray(pil.resize((C.IMG_SIZE, C.IMG_SIZE))) / 255.0
        panels.append(_overlay(disp, cam[0]))
        titles.append(f"{P.pretty(C.CLASSES[idx])} -> {P.pretty(C.CLASSES[int(cls[0])])} ({prob[0, int(cls[0])]:.2f})")
    out = out or os.path.join(C.FIGDIR, f"gradcam_{which}")
    path = P.image_panels(panels, titles, out, suptitle=f"Grad-CAM — {which} (true -> predicted)")
    print("Saved Grad-CAM ->", path)
    return path


if __name__ == "__main__":
    run_gradcam("student")
