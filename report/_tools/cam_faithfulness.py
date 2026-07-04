"""Quantitative Grad-CAM comparison: new (224px) vs legacy (380px) student, on the SAME cells.

Eyeballing n=1 cell per class is not a verdict. This measures, over N random cropped cells, what the
heatmaps actually do:
  * accuracy / mean confidence on those cells,
  * concentration   = fraction of Grad-CAM energy in the top-20% hottest pixels (higher = more focused),
  * deletion-drop   = drop in the predicted-class prob when the top-20% Grad-CAM pixels are masked
                      (higher = the highlighted region was genuinely important = more FAITHFUL),
  * cam_grid        = the conv feature-map size the CAM is computed at (explains crispness).

Run twice (separate processes — each pipeline owns a module named `config`):
    python cam_faithfulness.py new
    python cam_faithfulness.py legacy
-> results/metrics/cam_faithfulness_<which>.json
"""
import os, sys, glob, json, random
import numpy as np
from PIL import Image
import torch

WHICH = sys.argv[1] if len(sys.argv) > 1 else "new"
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
N_CELLS = 80
SEED = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406); IMAGENET_STD = (0.229, 0.224, 0.225)

# --- shared pool of cropped cells (same files for both models) ---
crop_dirs = glob.glob(os.path.join(PROJ, "data", "raw", "sipakmed", "**", "CROPPED"), recursive=True)
paths = []
for d in crop_dirs:
    paths += [p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
              if p.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")) and "crop" in p.lower()]
paths = sorted(set(paths))
random.Random(SEED).shuffle(paths)
paths = paths[:N_CELLS]


def _class_name(p):
    for c in ["Superficial-Intermediate", "Parabasal", "Metaplastic", "Koilocytotic", "Dyskeratotic"]:
        if f"im_{c}" in p:
            return c
    return "?"


# --- load the requested model + its native resolution ---
if WHICH == "new":
    sys.path.insert(0, os.path.join(PROJ, "src"))
    import config as C
    import run
    model = run._load_model("student"); IMG = C.IMG_SIZE; CLASSES = C.CLASSES
    if len(sys.argv) > 2:                      # allow explaining the 224-trained student at a higher res
        IMG = int(sys.argv[2])
else:
    sys.path.insert(0, os.path.join(PROJ, "Legacy_src", "src"))
    import config as C
    import evaluate as E
    model = E.load_model("student"); IMG = C.IMG_SIZE; CLASSES = C.CLASSES
DEV = next(model.parameters()).device
model.eval()


def _tensor(pil):
    a = np.asarray(pil.convert("RGB").resize((IMG, IMG)), np.float32) / 255.0
    a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float()


class GradCAM:
    def __init__(self, model, layer):
        self.a = self.g = None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "a", o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "g", go[0].detach()))

    def __call__(self, x, cls):
        self.model_zero(); logits = model(x.to(DEV))
        p = torch.softmax(logits.float(), 1)
        logits[0, cls].backward()
        w = self.g.mean((2, 3), keepdim=True)
        cam = torch.relu((w * self.a).sum(1))[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), p.detach()

    def model_zero(self):
        for pm in model.parameters():
            pm.grad = None


cam_fn = GradCAM(model, model.last_conv_module)
cam_grid = None
acc = conf = concentration = deletion = n = 0

for p in paths:
    try:
        pil = Image.open(p)
    except Exception:
        continue
    x = _tensor(pil)
    logits = model(x.to(DEV))
    probs = torch.softmax(logits.float(), 1)[0]
    pred = int(probs.argmax()); c0 = float(probs[pred])
    cam, _ = cam_fn(x, pred)
    if cam_grid is None:
        cam_grid = cam.shape
    # concentration: top-20% energy fraction
    flat = cam.flatten(); flat = flat / (flat.sum() + 1e-8)
    k = max(1, int(0.20 * flat.size)); top = np.sort(flat)[-k:]
    concentration += float(top.sum())
    # deletion: mask the top-20% CAM pixels (upsampled to image), re-run, measure prob drop
    import torch.nn.functional as F
    big = F.interpolate(torch.from_numpy(cam)[None, None], size=(IMG, IMG), mode="bilinear",
                        align_corners=False)[0, 0].numpy()
    thr = np.quantile(big, 0.80)
    mask = torch.from_numpy((big < thr).astype(np.float32))[None, None]   # keep bottom 80%
    xm = x * mask
    with torch.no_grad():
        cm = float(torch.softmax(model(xm.to(DEV)).float(), 1)[0, pred])
    deletion += (c0 - cm)
    conf += c0; acc += int(CLASSES[pred] == _class_name(p)); n += 1

rep = dict(which=WHICH, img_size=IMG, n=n, cam_grid=list(cam_grid),
           accuracy=round(acc / n, 4), mean_confidence=round(conf / n, 4),
           cam_concentration_top20=round(concentration / n, 4),
           deletion_faithfulness=round(deletion / n, 4))
out = os.path.join(PROJ, "results", "metrics", f"cam_faithfulness_{WHICH}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(rep, open(out, "w"), indent=2)
print(json.dumps(rep, indent=2))
