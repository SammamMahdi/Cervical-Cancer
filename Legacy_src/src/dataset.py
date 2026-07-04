"""SIPaKMeD data loading — faithful to the legacy notebook.

The notebook ran `image_dataset_from_directory` over the 5 `im_<Class>` folders and found
5015 images (it recursively picked up BOTH the cluster FOV .bmp images AND the CROPPED single
cells). We reproduce that: per class, recursively gather ALL images; legacy random 80/20 split.
"""
import os, glob, random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import config as C

IMG_EXT = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def find_class_dirs(root=C.SIPAKMED_DIR, raw_classes=C.RAW_CLASSES):
    """Locate each im_<Class> directory anywhere under root (mirrors nest differently)."""
    mapping = {}
    for rc in raw_classes:
        hits = [d for d in glob.glob(os.path.join(root, "**", rc), recursive=True) if os.path.isdir(d)]
        if not hits:
            # some mirrors strip the 'im_' prefix
            alt = rc.replace("im_", "")
            hits = [d for d in glob.glob(os.path.join(root, "**", alt), recursive=True) if os.path.isdir(d)]
        if not hits:
            raise FileNotFoundError(f"class folder for {rc} not found under {root}")
        mapping[rc] = sorted(hits, key=len)[0]   # shallowest match
    return mapping


def list_images(class_dirs):
    """Return (paths, labels, per_class_counts). Recursively gathers all images per class."""
    paths, labels = [], []
    counts = {}
    for idx, rc in enumerate(C.RAW_CLASSES):
        d = class_dirs[rc]
        files = [p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                 if p.lower().endswith(IMG_EXT)]
        counts[C.CLASSES[idx]] = len(files)
        paths += files
        labels += [idx] * len(files)
    return paths, labels, counts


def get_transforms(img_size=C.IMG_SIZE, train=True):
    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),                       # legacy: flip horizontal_and_vertical
            T.RandomRotation(72),                            # legacy RandomRotation(0.2) ~= +/-72 deg
            T.RandomAffine(degrees=0, scale=(0.9, 1.1)),     # legacy RandomZoom(0.1)
            T.ToTensor(),
            T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


class SipakmedDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def stratified_split(paths, labels, val_frac=C.VAL_SPLIT, seed=C.SEED):
    """Legacy-style random split, stratified per class (deterministic)."""
    rng = random.Random(seed)
    by_cls = {}
    for p, y in zip(paths, labels):
        by_cls.setdefault(y, []).append(p)
    tr_p, tr_y, va_p, va_y = [], [], [], []
    for y, ps in by_cls.items():
        ps = ps[:]; rng.shuffle(ps)
        n_val = int(round(len(ps) * val_frac))
        va = ps[:n_val]; tr = ps[n_val:]
        tr_p += tr; tr_y += [y] * len(tr)
        va_p += va; va_y += [y] * len(va)
    return (tr_p, tr_y), (va_p, va_y)


def make_loaders(batch_size=C.BATCH_SIZE, img_size=C.IMG_SIZE, val_frac=C.VAL_SPLIT,
                 num_workers=C.NUM_WORKERS, seed=C.SEED):
    cdirs = find_class_dirs()
    paths, labels, counts = list_images(cdirs)
    (trp, trl), (vap, val) = stratified_split(paths, labels, val_frac, seed)
    train_ds = SipakmedDataset(trp, trl, get_transforms(img_size, train=True))
    val_ds = SipakmedDataset(vap, val, get_transforms(img_size, train=False))
    pf = 2 if num_workers > 0 else None
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=False, drop_last=False, prefetch_factor=pf,
                          persistent_workers=False)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=False, prefetch_factor=pf, persistent_workers=False)
    return train_ld, val_ld, counts


def class_weights(counts):
    """Legacy static weights: total / (num_classes * count_i)."""
    total = sum(counts.values()); k = len(counts)
    return torch.tensor([total / (k * counts[c]) for c in C.CLASSES], dtype=torch.float32)
