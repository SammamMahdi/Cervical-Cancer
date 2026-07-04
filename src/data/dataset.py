"""224px data loading for the new pipeline (DATA_PLAN §4-5).

- Input 224x224 (B0 native; fixes the legacy 380 mismatch), ASPECT-PRESERVING pad (no nuclear-shape
  distortion).
- Cytology-appropriate, train-only augmentation, tiered for the A5 ablation
  (none / geometric / photometric). Stain augmentation (MixStyle/HED) is added by stain.py (A4/U4).
- Loaders are built from the persisted leakage-safe split (splits.py); val/test never augmented.
"""
import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import config as C

IMG_EXT = (".bmp", ".png", ".jpg", ".jpeg")


class AspectPad:
    """Resize longest side to `size`, pad to a square (no distortion of nuclear morphology)."""
    def __init__(self, size, fill=0):
        self.size = size; self.fill = fill
    def __call__(self, img):
        w, h = img.size
        s = self.size / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        nw, nh = img.size
        l = (self.size - nw) // 2; t = (self.size - nh) // 2
        return TF.pad(img, [l, t, self.size - nw - l, self.size - nh - t], fill=self.fill)


class ScaleJitterPad:
    """Aspect-preserving rescale-then-pad with a RANDOM per-call scale factor (train-only).

    Replaces the deterministic ``AspectPad`` when scale augmentation is on: the cell is fit to a frame
    of ``round(size * f)`` (longest side) for ``f`` drawn log-uniformly in ``[lo, hi]``, then either
    center-padded (f<=1, cell appears SMALLER inside the model frame) or center-cropped (f>1, cell
    appears LARGER / partly out of frame). This forces scale-invariance — the scale-stress gate showed
    the frozen student collapses 0.97->0.50 under 0.3x-3x rescale, reading small cells as more abnormal.

    ``lo``/``hi`` are multiplicative bounds on apparent cell size (e.g. (0.5, 2.0)). The draw is
    log-uniform so up- and down-scaling are symmetric. ``fill`` matches AspectPad's pad colour.
    """
    def __init__(self, size, lo=0.5, hi=2.0, fill=0):
        self.size = int(size); self.lo = float(lo); self.hi = float(hi); self.fill = fill
        if not (0 < self.lo <= self.hi):
            raise ValueError(f"scale_jitter bounds must satisfy 0<lo<=hi, got ({lo},{hi})")

    def __call__(self, img):
        import math, random
        S = self.size
        f = math.exp(random.uniform(math.log(self.lo), math.log(self.hi)))   # log-uniform in [lo,hi]
        target = max(1, int(round(S * f)))                       # longest-side target inside the frame
        w, h = img.size
        s = target / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        nw, nh = img.size
        # center the rescaled cell in an S x S canvas: pad when smaller, crop when larger (per-axis).
        l = (S - nw) // 2; t = (S - nh) // 2
        img = TF.pad(img, [max(0, l), max(0, t),
                           max(0, S - nw - l), max(0, S - nh - t)], fill=self.fill)
        # img is now >= S on each axis; center-crop back to exactly S x S (handles the crop side, f>1).
        cw, ch = img.size
        cl = (cw - S) // 2; ct = (ch - S) // 2
        return img.crop((cl, ct, cl + S, ct + S))


_DEFAULT_SCALE_JITTER = (0.5, 2.0)   # apparent-scale bounds when scale aug is on (log-uniform)


def build_transform(train, aug="photometric", img_size=C.IMG_SIZE, stain_norm=None,
                    scale_jitter=None):
    """aug in {'none','geometric','photometric'} and the '+scale' suffix (A11), e.g.
    'photometric+scale' / 'geometric+scale' / 'none+scale'. stain_norm = a fitted Macenko/Reinhard
    normalizer (U4/A4), applied to the raw RGB before pad/normalize. Eval -> deterministic.

    Scale augmentation (TRAIN ONLY): when ``aug`` ends in '+scale' OR ``scale_jitter=(lo,hi)`` is given,
    the deterministic AspectPad is replaced by ``ScaleJitterPad`` which randomly rescales the cell inside
    the model frame (log-uniform in [lo,hi]) to force scale-invariance (the scale-stress gate). An
    explicit ``scale_jitter`` kwarg overrides the suffix bounds; the suffix alone uses
    ``_DEFAULT_SCALE_JITTER``. Default OFF: with no suffix and scale_jitter=None the output is
    byte-identical to before."""
    # ---- resolve scale-jitter request (suffix and/or kwarg); strip '+scale' from aug ----
    aug_base = aug
    sj = scale_jitter
    if isinstance(aug, str) and aug.endswith("+scale"):
        aug_base = aug[: -len("+scale")] or "none"
        if sj is None:
            sj = _DEFAULT_SCALE_JITTER
    scale_on = train and (sj is not None)                       # eval is always deterministic

    def _pad():                                                 # AspectPad, or ScaleJitterPad when on
        if scale_on:
            return ScaleJitterPad(img_size, lo=sj[0], hi=sj[1])
        return AspectPad(img_size)

    pre = []
    if stain_norm is not None:
        from data.stain import StainTransform
        pre = [StainTransform(stain_norm)]
    norm = [T.ToTensor(), T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)]
    if not train or aug_base == "none":
        return T.Compose([*pre, _pad(), *norm])
    geo = [
        _pad(),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomRotation(180),                                   # full 0-360deg (cells have no canonical orientation)
        T.RandomAffine(degrees=0, scale=(0.85, 1.15), translate=(0.05, 0.05)),
    ]
    if aug_base == "geometric":
        return T.Compose([*pre, *geo, *norm])
    photo = [
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),  # mild scanner/stain variation
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),
    ]
    return T.Compose([*pre, *geo, *photo, *norm])


def _class_of(path):
    for ci, cname in enumerate(C.CLASSES):
        if f"im_{cname}" in path:
            return ci
    raise ValueError(f"cannot determine class for {path}")


class CellDataset(Dataset):
    def __init__(self, paths, transform, labels=None):
        self.paths = paths
        self.labels = labels if labels is not None else [_class_of(p) for p in paths]
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (8, 8))   # corrupt/unreadable -> placeholder (keeps loop running)
        return self.transform(img), self.labels[i]


def fit_stain(stain="none", split=None, split_kind="grouped"):
    """Return a Macenko/Reinhard normalizer fit on the TRAIN split only (U4/A4), or None.
    Shared by make_loaders (in-dist) and the external-eval transforms so stain norm is consistent."""
    if stain not in ("macenko", "reinhard"):
        return None
    from data.splits import get_split
    from data.stain import ReinhardNormalizer, MacenkoNormalizer, fit_on_paths
    split = split or get_split(split_kind)
    norm = MacenkoNormalizer() if stain == "macenko" else ReinhardNormalizer()
    fit_on_paths(norm, split["train"])
    return norm


def make_loaders(split=None, split_kind="grouped", batch_size=C.BATCH_SIZE, aug="photometric",
                 num_workers=C.NUM_WORKERS, img_size=C.IMG_SIZE, stain="none", oversample=False):
    """Train/val/test loaders. split_kind='grouped' (leakage-safe, default) or 'random' (leaky A1).
    stain in {'none','macenko','reinhard'} (U4/A4) — fit on TRAIN ONLY. oversample=True (A6) uses a
    class-balanced WeightedRandomSampler on train. Returns (train_ld, val_ld, test_ld, counts)."""
    from data.splits import get_split
    from collections import Counter
    split = split or get_split(split_kind)
    stain_norm = fit_stain(stain, split=split)                 # fit on TRAIN ONLY (no leakage)
    pf = 2 if num_workers > 0 else None
    def ld(paths, train):
        ds = CellDataset(paths, build_transform(train, aug, img_size, stain_norm=stain_norm))
        sampler = None
        if train and oversample:                                # A6: class-balanced oversampling
            from torch.utils.data import WeightedRandomSampler
            labs = [_class_of(p) for p in paths]; cc = Counter(labs)
            w = [1.0 / cc[l] for l in labs]
            sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        return DataLoader(ds, batch_size=batch_size, shuffle=(train and sampler is None),
                          sampler=sampler, num_workers=num_workers, pin_memory=False,
                          prefetch_factor=pf, persistent_workers=False)
    from collections import Counter
    counts = Counter(_class_of(p) for p in split["train"])
    counts = {C.CLASSES[i]: counts.get(i, 0) for i in range(C.NUM_CLASSES)}
    return ld(split["train"], True), ld(split["val"], False), ld(split["test"], False), counts


def class_weights(counts):
    """Inverse-frequency weights: total / (k * count_i)."""
    total = sum(counts.values()); k = len(counts)
    return torch.tensor([total / (k * max(1, counts[c])) for c in C.CLASSES], dtype=torch.float32)


if __name__ == "__main__":
    tr, va, te, counts = make_loaders()
    xb, yb = next(iter(tr))
    print("train counts:", counts)
    print("batch:", tuple(xb.shape), "labels:", yb[:8].tolist(), "| val batches:", len(va), "test batches:", len(te))
