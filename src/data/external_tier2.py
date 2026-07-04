"""Tier-2 point-annotated external datasets -> cell crops (DATA_PLAN §4.2).

Some external datasets ship only whole-FOV (field-of-view) images with point annotations
(a centroid (x,y) + a Bethesda/native label per cell) instead of pre-cropped cells. To evaluate
the SIPaKMeD-trained cell classifier on them we crop a fixed square patch centered on each
annotated point, clamped to the FOV bounds, and push it through the same deterministic eval
transform used for in-distribution test data (build_transform(train=False)).

Native labels are collapsed to the shared safe tiers via the deterministic crosswalk in
data.labelmap (many-to-one, never one-to-many). Rows whose tier label maps to None (EXCLUDE:
glandular / ASC-US indeterminate / high-grade with no morphology home, depending on tier) are
SKIPPED and counted in .excluded — the "excluded fraction" honesty metric — never silently used.

  Tier E0  -> binary  : normal(0) / abnormal(1)
  Tier E1  -> grade   : normal(0) / low(1) / high(2)

CRIC (Center for Recognition and Inspection of Cells) is the concrete Tier-2 source here:
whole-slide FOV PNGs + a per-cell nucleus-centroid CSV with Bethesda classes. We reuse the
HiCervix Bethesda crosswalk (labelmap.hicervix_map / .hicervix_candidates) since both are
Bethesda-system labels.
"""
import os
import csv

import torch
from PIL import Image
from torch.utils.data import Dataset

import config as C
from data.dataset import build_transform
from data import labelmap

# ---- E1 grade -> integer (E0 reuses normal/abnormal -> 0/1) ----
_E1_TO_INT = {"normal": 0, "low": 1, "high": 2}

CRIC_ROOT = os.path.join(C.DATA_RAW, "cric", "cric_dataset_complete")
CRIC_CSV = os.path.join(CRIC_ROOT, "classes", "classifications.csv")


def _tier_int(e0, e1, tier):
    """Map a (e0, e1) crosswalk result to the integer label for the requested tier, or None to skip."""
    if tier == "E0":
        if e0 == "normal":
            return 0
        if e0 == "abnormal":
            return 1
        return None                       # EXCLUDE
    if tier == "E1":
        if e1 is None:
            return None
        return _E1_TO_INT.get(e1, None)
    raise ValueError(f"unknown tier {tier!r} (expected 'E0' or 'E1')")


class PointCropExternal(Dataset):
    """Crop fixed square patches centered on point annotations from FOV images.

    Args:
      image_dir : directory holding the FOV images.
      annotations : iterable of (image_filename, x, y, native_label) tuples.
      label_mapper : fn(native_label) -> (e0, e1) (a labelmap.*_map function).
      tier : 'E0' (normal/abnormal -> 0/1) or 'E1' (normal/low/high -> 0/1/2).
      crop : square patch side length in pixels (default 80), centered on (x,y), clamped to bounds.
      transform : eval transform; defaults to build_transform(train=False).

    Behaviour: rows whose mapped tier label is None (EXCLUDE) are dropped and counted in .excluded;
    rows whose FOV image is missing are also excluded. .n = number of kept (usable) crops.
    __getitem__(i) -> (tensor, tier_int).
    """

    def __init__(self, image_dir, annotations, label_mapper, tier="E0", crop=80, transform=None):
        self.image_dir = image_dir
        self.tier = tier
        self.crop = int(crop)
        self.transform = transform if transform is not None else build_transform(train=False)
        self.items = []          # (image_path, x, y, label_int)
        self.excluded = 0
        for fname, x, y, native in annotations:
            e0, e1 = label_mapper(native)
            lab = _tier_int(e0, e1, tier)
            if lab is None:
                self.excluded += 1
                continue
            path = os.path.join(image_dir, fname)
            if not os.path.isfile(path):
                self.excluded += 1
                continue
            self.items.append((path, int(round(float(x))), int(round(float(y))), int(lab)))
        self.n = len(self.items)

    def __len__(self):
        return self.n

    def _crop_patch(self, img, x, y):
        """Return a `crop`x`crop` patch centered on (x,y), clamped so the box stays inside the image."""
        w, h = img.size
        half = self.crop // 2
        left = x - half
        top = y - half
        # clamp the top-left so a full-size box fits inside the FOV (centroids near edges still yield
        # a `crop`x`crop` patch rather than a truncated one).
        left = max(0, min(left, max(0, w - self.crop)))
        top = max(0, min(top, max(0, h - self.crop)))
        right = min(w, left + self.crop)
        bottom = min(h, top + self.crop)
        return img.crop((left, top, right, bottom))

    def __getitem__(self, i):
        path, x, y, lab = self.items[i]
        try:
            with Image.open(path) as im:
                patch = self._crop_patch(im.convert("RGB"), x, y)
        except Exception:
            patch = Image.new("RGB", (self.crop, self.crop))   # corrupt FOV -> placeholder
        return self.transform(patch), lab


def _read_cric_rows(csv_path=CRIC_CSV):
    """Yield (image_filename, x, y, bethesda_class) from the CRIC CSV.

    Columns (per SHARED CONTRACT, 0-indexed): image_id, image_filename(1), doi, cell_id,
    bethesda_class(4), nucleus_x(5), nucleus_y(6). The on-disk file carries a header row, so any
    row whose x/y do not parse as numbers is treated as a header/garbage row and skipped.
    """
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            fname = row[1].strip()
            native = row[4].strip()
            try:
                x = float(row[5]); y = float(row[6])
            except ValueError:
                continue                          # header line ("nucleus_x") or malformed row
            yield (fname, x, y, native)


def cric_dataset(tier="E0", crop=None, root=CRIC_ROOT, csv_path=None, transform=None):
    """Build a PointCropExternal over the CRIC dataset (DATA_PLAN §4.2).

    FOV PNGs live directly under `root`; the annotation CSV is root/classes/classifications.csv.
    Bethesda classes are mapped via labelmap.hicervix_map (SCC/HSIL/ASC-H -> high, LSIL -> low,
    NILM/negative -> normal, ASC-US/glandular -> EXCLUDE). Returns a PointCropExternal exposing
    .n (kept) and .excluded.
    """
    csv_path = csv_path or os.path.join(root, "classes", "classifications.csv")
    crop = crop if crop is not None else C.CRIC_CROP    # data-driven default (one-cell territory)
    annotations = list(_read_cric_rows(csv_path))
    return PointCropExternal(root, annotations, labelmap.hicervix_map,
                             tier=tier, crop=crop, transform=transform)


def cric_geometry(csv_path=None, save=True):
    """Training-free: measure CRIC cell density (nearest-neighbour nucleus distance per FOV) to justify
    the crop size. Writes results/metrics/cric_crop_geometry.json. A crop ~ median NN distance captures
    roughly one cell's territory; larger crops include neighbours (CRIC is a dense preparation)."""
    import numpy as np
    import runlog
    from collections import defaultdict
    csv_path = csv_path or CRIC_CSV
    byfov = defaultdict(list)
    for fname, x, y, _native in _read_cric_rows(csv_path):
        byfov[fname].append((x, y))
    nn = []
    for pts in byfov.values():
        if len(pts) < 2:
            continue
        Pm = np.array(pts, float)
        for i in range(len(Pm)):
            d = np.sqrt(((Pm - Pm[i]) ** 2).sum(1)); d[i] = np.inf
            nn.append(float(d.min()))
    nn = np.array(nn)
    nnd = (dict(median=float(np.median(nn)), mean=float(nn.mean()),
                p25=float(np.percentile(nn, 25)), p75=float(np.percentile(nn, 75)))
           if nn.size else dict(median=None, mean=None, p25=None, p75=None))
    rep = dict(n_fovs=len(byfov), n_nuclei=int(sum(len(v) for v in byfov.values())),
               nn_distance_px=nnd, chosen_crop=C.CRIC_CROP, sweep=C.CRIC_CROP_SWEEP,
               note="crop ~ median NN distance = one cell's territory; larger -> neighbour clutter")
    if save:
        runlog.dump_json(os.path.join(C.METRICDIR, "cric_crop_geometry.json"), rep)
    return rep


def cric_candidates(native_label):
    """E-morph candidate set for a CRIC Bethesda class (reuse the HiCervix crosswalk)."""
    return labelmap.hicervix_candidates(native_label)


if __name__ == "__main__":
    ds = cric_dataset("E0")
    print(f"CRIC E0: kept n={ds.n}  excluded={ds.excluded}  (total={ds.n + ds.excluded})")
    if ds.n:
        x0, y0 = ds[0]
        print(f"sample crop: tensor {tuple(x0.shape)}  dtype={x0.dtype}  label={y0}")
    ds1 = cric_dataset("E1")
    print(f"CRIC E1: kept n={ds1.n}  excluded={ds1.excluded}")
