"""MINI-GATE B - model-based scale-stress curve.

Measure how the frozen student's predictions degrade when the INPUT cell is rescaled (zoom in/out)
within the fixed C.IMG_SIZE model frame, simulating cross-dataset scale shift. We reuse the same
deterministic eval preprocessing as data.dataset (ToTensor + ImageNet normalize) but swap AspectPad
for a scale-aware pad/crop so the cell occupies f x its native fraction of the frame.

  f < 1.0 : shrink the cell (longest side -> size*f) then aspect-pad to size  -> cell is SMALLER
  f = 1.0 : exactly AspectPad(size)                                            -> baseline
  f > 1.0 : enlarge the cell (longest side -> size*f) then CENTER-CROP to size -> cell is BIGGER

Reports 5-class accuracy and mean P(abnormal) vs f, plus the accuracy drop at 0.3x and 3x relative
to 1.0x. Writes results/metrics/mini_gate_b_scale_stress.json and a line figure.
"""
import os, json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

import config as C
import runlog
import plotting as P
import run as RUN
from data.splits import get_split
from data.dataset import _class_of
import data.labelmap as LM


class ScalePad:
    """Render the cell at relative scale `f` inside a fixed `size`x`size` frame.

    longest-side -> round(size*f); if the scaled image is smaller than the frame -> aspect-pad
    (cell shrinks); if larger -> center-crop to the frame (cell zooms in, periphery clipped).
    f=1.0 reduces exactly to data.dataset.AspectPad(size).
    """
    def __init__(self, size, f, fill=0):
        self.size = size; self.f = f; self.fill = fill

    def __call__(self, img):
        w, h = img.size
        target_long = max(1, round(self.size * self.f))
        s = target_long / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        img = img.resize((nw, nh), Image.BILINEAR)
        # pad (if smaller than frame on an axis) then center-crop (if larger) -> always size x size
        if nw < self.size or nh < self.size:
            l = max(0, (self.size - nw) // 2); t = max(0, (self.size - nh) // 2)
            img = TF.pad(img, [l, t, max(0, self.size - nw - l), max(0, self.size - nh - t)], fill=self.fill)
        # now image is >= size on every axis where it was enlarged; center-crop to exact frame
        img = TF.center_crop(img, [self.size, self.size])
        return img


def build_scale_transform(f, img_size=C.IMG_SIZE):
    return T.Compose([ScalePad(img_size, f),
                      T.ToTensor(),
                      T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)])


class ScaledCellDataset(Dataset):
    def __init__(self, paths, f, img_size=C.IMG_SIZE):
        self.paths = paths
        self.labels = [_class_of(p) for p in paths]
        self.transform = build_scale_transform(f, img_size)
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (8, 8))
        return self.transform(img), self.labels[i]


@torch.no_grad()
def _predict(model, loader):
    device = next(model.parameters()).device
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(C.DEVICE == "cuda")):
            logits = model(xb).float()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        ps.append(probs); ys.append(np.asarray(yb))
    return np.concatenate(ys), np.concatenate(ps)


def run(scales=(0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0), which="student",
        img_size=C.IMG_SIZE, num_workers=C.NUM_WORKERS):
    model = RUN._load_model(which)
    split = get_split("grouped")
    test_paths = split["test"]
    n = len(test_paths)
    print(f"[mini-gate-B] frozen {which}; sealed SIPaKMeD test cells: n={n}; scales={list(scales)}")

    curve = []
    for f in scales:
        ds = ScaledCellDataset(test_paths, f, img_size)
        ld = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=num_workers)
        y, probs = _predict(model, ld)
        pred = probs.argmax(1)
        acc = float((pred == y).mean())
        p_abn = LM.p_abnormal(probs)
        mean_pabn = float(p_abn.mean())
        # binary (E0) accuracy too, for context
        true_abn = np.isin(y, C.ABNORMAL_IDX).astype(int)
        pred_abn = np.isin(pred, C.ABNORMAL_IDX).astype(int)
        e0_acc = float((pred_abn == true_abn).mean())
        curve.append(dict(scale=float(f), accuracy=acc, e0_accuracy=e0_acc,
                          mean_p_abnormal=mean_pabn))
        print(f"  f={f:<4} acc={acc:.4f}  E0acc={e0_acc:.4f}  mean_P(abn)={mean_pabn:.4f}")

    by_f = {round(c["scale"], 3): c for c in curve}
    acc1 = by_f[1.0]["accuracy"]
    pabn1 = by_f[1.0]["mean_p_abnormal"]

    def drop(f):
        if round(f, 3) in by_f:
            return float(acc1 - by_f[round(f, 3)]["accuracy"])
        return None

    accs = [c["accuracy"] for c in curve]
    pabns = [c["mean_p_abnormal"] for c in curve]
    summary = dict(
        which=which, img_size=img_size, n_test=n, scales=list(map(float, scales)),
        baseline_f1p0=dict(accuracy=acc1, mean_p_abnormal=pabn1),
        curve=curve,
        acc_drop_at_0p3x=drop(0.3),
        acc_drop_at_3x=drop(3.0),
        acc_drop_at_0p5x=drop(0.5),
        acc_drop_at_2x=drop(2.0),
        accuracy_range=float(max(accs) - min(accs)),
        accuracy_min=float(min(accs)), accuracy_max=float(max(accs)),
        p_abnormal_range=float(max(pabns) - min(pabns)),
        p_abnormal_min=float(min(pabns)), p_abnormal_max=float(max(pabns)),
    )
    out_json = os.path.join(C.METRICDIR, "mini_gate_b_scale_stress.json")
    runlog.dump_json(out_json, summary)

    xs = [c["scale"] for c in curve]
    try:
        fig_path = P.line_plot(
            xs,
            {"accuracy_5class": accs, "mean_P_abnormal": pabns, "E0_accuracy": [c["e0_accuracy"] for c in curve]},
            os.path.join(C.FIGDIR, "mini_gate_b_scale_stress"),
            xlabel="relative input scale f (cell size in frame)",
            ylabel="metric", title="Mini-Gate B: scale-stress curve (frozen student)")
    except Exception as e:
        fig_path = None
        print(f"[mini-gate-B] figure skipped: {e}")

    summary["json_path"] = out_json
    summary["figure_path"] = fig_path
    print(f"[mini-gate-B] acc@1.0={acc1:.4f}  drop@0.3x={summary['acc_drop_at_0p3x']}  "
          f"drop@3x={summary['acc_drop_at_3x']}  acc range={summary['accuracy_range']:.4f}  "
          f"P(abn) range={summary['p_abnormal_range']:.4f}")
    return summary


if __name__ == "__main__":
    run()
