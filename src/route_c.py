"""Route-C: partial-label / marginal+exclusion training (LABEL_ALIGNMENT_METHOD).

Motivation
----------
Routes A/B align coarse external labels (Herlev / APACC / HiCervix / CRIC) to the 5-class
SIPaKMeD morphology space only to *evaluate* them (the E0/E1/E-morph crosswalk in
``data.labelmap``). Route-C instead lets those coarse labels *train* the model — without
ever forcing a single leaf.

The device is the **candidate set** S ⊆ {0..4}. An external cell whose native label is, e.g.,
"LSIL" is consistent with EITHER Koilocytotic or Dyskeratotic but we do not know which, so its
candidate set is S = {Koilocytotic, Dyskeratotic}. We do not pick one; we supervise the *marginal
probability mass* the model assigns to S::

    L_partial = -log( sum_{k in S} softmax(logits)_k )                       (marginal CE)

No leaf inside S is penalised relative to the others — the model is free to distribute mass within
S however its features dictate, and only the *total* mass on S is pushed up. A SIPaKMeD cell has a
singleton S = {true class}, for which this collapses *exactly* to standard cross-entropy
( -log softmax_true ). An optional **exclusion** term additionally pushes mass *away* from a set of
forbidden classes F (classes the native label rules out)::

    L_excl    = -log( sum_{k not in F} softmax(logits)_k )

This is the partial-label / "superset learning" objective of Shi et al. (Medical Image Analysis,
2021) and the marginal-likelihood view behind PRODEN (Lv et al., ICML 2020): coarse labels
contribute a real gradient to the shared encoder while remaining honest about what they do (and do
not) determine. High-grade / glandular / indeterminate native labels have NO candidate set in the
5 morphology classes (``data.labelmap.*_candidates`` -> None) and are dropped — never coerced.

Datasets
--------
- SIPaKMeD TRAIN cells (leakage-safe grouped split)  -> exact singleton candidate sets.
- External cells -> candidate sets via ``labelmap.EXTERNAL_CANDIDATES`` (herlev/apacc/hicervix) and,
  for CRIC, the Bethesda keyword crosswalk ``labelmap.hicervix_candidates`` (Negative -> normal,
  LSIL -> low; ASC-US/ASC-H/HSIL/SCC/glandular -> None -> dropped).

Evaluation is **5-class, standard CE metrics, on the SEALED SIPaKMeD test split only** (the external
cells are training signal, never test). The trained encoder+head is saved as ``student_partiallabel.pt``.

Public API
----------
- PartialLabelLoss(nn.Module)      : marginal (+ optional exclusion) cross-entropy on a multi-hot mask.
- PartialLabelDataset(Dataset)     : yields (image_tensor, candidate_mask[5] multi-hot float).
- candidate_mask(candset)          : frozenset|iterable of class idx -> multi-hot tensor[5].
- build_partial_dataset(...)       : mixed SIPaKMeD-singleton + external-candidate dataset (+ counts).
- train_partial(...)               : train shared encoder + 5-logit head; eval on sealed SIPaKMeD test.
"""
from __future__ import annotations

import os
import csv
import glob
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import config as C
import runlog
from data.clustering import load_cells
from data.splits import get_split
from data.dataset import build_transform
from data import labelmap as L
from models.backbones import build, count_params
import eval.metrics as M

IMG_EXT = (".bmp", ".png", ".jpg", ".jpeg")
_NEG_INF = -1e30   # log(0) surrogate for masked-out logits (BF16/fp32 safe, finite gradient)


# --------------------------------------------------------------------------- #
#  Loss
# --------------------------------------------------------------------------- #
class PartialLabelLoss(nn.Module):
    r"""Partial-label marginal (+ optional exclusion) cross-entropy.

    For a cell with candidate set S (multi-hot mask ``m`` over the 5 classes), the loss is the
    negative log of the softmax mass on S::

        p_S   = sum_{k in S} softmax(logits)_k
        L     = -log p_S = -( logsumexp_{k in S} logits_k  -  logsumexp_k logits_k )

    computed in log-space (mask the non-candidate logits to -inf inside the numerator's
    ``logsumexp``) for numerical stability. With a SINGLETON mask this is identically
    ``F.cross_entropy`` (modulo label smoothing): -log softmax_{true}.

    Optional **exclusion** term (weight ``excl_weight`` > 0): given a forbidden mask ``f`` (classes
    the native label rules OUT), additionally minimise the mass on the forbidden set by maximising
    the mass on its complement::

        L_excl = -log( sum_{k not in F} softmax(logits)_k )

    Total = L_marginal + excl_weight * L_excl. Rows with an all-zero forbidden mask contribute 0 to
    the exclusion term. Cells with a full / empty candidate mask (no usable supervision) contribute
    0 and are not counted in the mean (they should be filtered upstream, but this is defensive).

    Parameters
    ----------
    excl_weight : float
        Weight on the exclusion term (0 disables it; the default — exclusion is implied by a
        proper candidate set already, so it is opt-in for datasets that supply explicit forbidden
        sets rather than candidate sets).
    label_smoothing : float
        Uniform smoothing applied to the *candidate set* target: with smoothing ``e`` the loss
        becomes ``(1-e)*(-log p_S) + e*(-mean_{k in S} log softmax_k)`` so mass is encouraged to
        spread within S (a mild regulariser; 0 reproduces the pure marginal CE).
    reduction : {'mean','sum','none'}
        Reduction over the batch (mean ignores zero-supervision rows in the denominator).
    """

    def __init__(self, excl_weight: float = 0.0, label_smoothing: float = 0.0,
                 reduction: str = "mean") -> None:
        super().__init__()
        if excl_weight < 0:
            raise ValueError(f"excl_weight must be >= 0, got {excl_weight}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0,1), got {label_smoothing}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"invalid reduction: {reduction}")
        self.excl_weight = float(excl_weight)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

    @staticmethod
    def _masked_logsumexp(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """logsumexp over the True entries of ``mask`` (bool/float multi-hot), row-wise.

        Non-selected logits are set to a large negative constant so they vanish from the
        log-sum-exp; rows whose mask is all-zero return ``_NEG_INF`` (caller handles them).
        """
        sel = mask > 0.5
        masked = torch.where(sel, logits, torch.full_like(logits, _NEG_INF))
        return torch.logsumexp(masked, dim=1)

    def forward(self, logits: torch.Tensor, cand_mask: torch.Tensor,
                forbidden_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``logits`` (N,5) ; ``cand_mask`` (N,5) multi-hot of the candidate set S.

        ``forbidden_mask`` (N,5) multi-hot of classes ruled OUT (used only if excl_weight>0).
        Returns the reduced partial-label loss.
        """
        logits = logits.float()
        cand_mask = cand_mask.to(logits.dtype)

        denom = torch.logsumexp(logits, dim=1)                 # log Z  (full partition)
        num = self._masked_logsumexp(logits, cand_mask)        # log sum_{k in S} exp(logit_k)
        nll = denom - num                                      # = -log p_S  (>= 0)

        # rows with no usable candidate set (all-zero or all-one mask) carry no signal.
        n_sel = cand_mask.sum(dim=1)
        usable = (n_sel > 0.5) & (n_sel < logits.shape[1] - 0.5)
        loss = torch.where(usable, nll, torch.zeros_like(nll))

        if self.label_smoothing > 0.0:
            # spread term: -mean_{k in S} log softmax_k  = -(mean_{k in S} logit_k) + log Z
            log_prob = F.log_softmax(logits, dim=1)
            mean_logp = (log_prob * cand_mask).sum(dim=1) / n_sel.clamp_min(1.0)
            spread = torch.where(usable, -mean_logp, torch.zeros_like(nll))
            e = self.label_smoothing
            loss = (1.0 - e) * loss + e * spread

        if self.excl_weight > 0.0 and forbidden_mask is not None:
            forbidden_mask = forbidden_mask.to(logits.dtype)
            allowed = 1.0 - forbidden_mask                     # complement of F
            has_f = forbidden_mask.sum(dim=1) > 0.5
            allowed_lse = self._masked_logsumexp(logits, allowed)
            excl = torch.where(has_f, denom - allowed_lse, torch.zeros_like(nll))  # -log p_allowed
            loss = loss + self.excl_weight * excl

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        # mean over the rows that actually carried supervision (>=1 to avoid 0/0)
        denom_n = usable.sum().clamp_min(1)
        return loss.sum() / denom_n


# --------------------------------------------------------------------------- #
#  Candidate masks
# --------------------------------------------------------------------------- #
def candidate_mask(candset, n_leaves=None) -> torch.Tensor:
    """iterable of class/leaf indices -> multi-hot float tensor of length ``n_leaves`` (default NUM_CLASSES).

    For the 8-leaf Bethesda taxonomy (CytoBridge) pass ``n_leaves=labelmap.N_LEAVES8``; SIPaKMeD
    singleton indices (0..4) are valid in both spaces because the first 5 leaves ARE the SIPaKMeD classes."""
    n = C.NUM_CLASSES if n_leaves is None else int(n_leaves)
    m = torch.zeros(n, dtype=torch.float32)
    if candset is None:
        return m
    for k in candset:
        m[int(k)] = 1.0
    return m


def _singleton_mask(class_idx) -> torch.Tensor:
    return candidate_mask((int(class_idx),))


# --------------------------------------------------------------------------- #
#  External cell enumeration (path, candidate frozenset)
# --------------------------------------------------------------------------- #
def _list_images(d):
    out = []
    try:
        names = sorted(os.listdir(d))
    except (FileNotFoundError, NotADirectoryError):
        return out
    for f in names:
        if f.lower().endswith("-d.bmp"):        # Herlev segmentation mask -> skip
            continue
        if f.lower().endswith(IMG_EXT):
            out.append(os.path.join(d, f))
    return out


def _folder_external(root, mapper):
    """Walk a folder-structured dataset (Herlev/APACC); yield (path, candset) with candset!=None."""
    out = []
    for dp, _dirs, _files in os.walk(root):
        imgs = _list_images(dp)
        if not imgs:
            continue
        cs = mapper(os.path.basename(dp))
        if cs is None:
            continue
        for p in imgs:
            out.append((p, cs))
    return out


def _hicervix_external(splits=("train", "val"), mapper=None):
    """HiCervix rows from the given CSV split(s); yield (path, candset) using level_1/class_name."""
    mapper = mapper or L.hicervix_candidates
    out = []
    for sp in splits:
        csv_path = os.path.join(C.HICERVIX_DIR, f"{sp}.csv")
        img_root = os.path.join(C.HICERVIX_DIR, sp, sp)
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row.get("level_1") or "").strip() or (row.get("class_name") or "").strip()
                cs = mapper(key)
                if cs is None:
                    continue
                name = (row.get("image_name") or "").strip()
                path = os.path.join(img_root, name)
                if name and os.path.isfile(path):
                    out.append((path, cs))
    return out


def _cric_external(mapper=None):
    """CRIC cells from classifications.csv; one row per annotated cell -> the full-frame image.

    The CRIC release labels individual cells (bethesda_system) at (nucleus_x, nucleus_y) inside a
    full slide-field PNG; we do not have pre-cropped cells, so each annotated cell contributes its
    parent image with the cell's Bethesda candidate set (Negative -> normal, LSIL -> low; the rest
    -> None -> dropped). Mapped via the canonical Bethesda keyword crosswalk (labelmap.hicervix_candidates).

    The CSV ships WITH a header in this release; we detect and skip it, and also tolerate a
    header-less file (contract layout: image_filename @ idx1, bethesda_system @ idx4).
    """
    mapper = mapper or L.hicervix_candidates
    base = os.path.join(C.DATA_RAW, "cric", "cric_dataset_complete")
    csv_path = os.path.join(base, "classes", "classifications.csv")
    out = []
    if not os.path.isfile(csv_path):
        return out
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return out
    # detect header: first row col4 is not a known/parseable class token while later rows are
    start = 1 if rows[0][4].strip().lower() in ("bethesda_system", "bethesda_class") else 0
    for r in rows[start:]:
        if len(r) <= 6:
            continue
        fname = r[1].strip()
        cls = r[4].strip()
        cs = mapper(cls)        # Bethesda keyword crosswalk also covers CRIC tokens
        if cs is None or not fname:
            continue
        path = os.path.join(base, fname)
        if os.path.isfile(path):
            out.append((path, cs))
    return out


def _point_crop(img, x, y, size):
    """Crop a ``size``x``size`` patch centered on (x,y), clamped to image bounds.

    Mirrors ``data.external_tier2.PointCropExternal._crop_patch`` so CRIC TRAINING crops are identical
    to CRIC EVAL crops (one single-cell window per nucleus, not the whole 1376x1020 field)."""
    w, h = img.size
    half = int(size) // 2
    left = max(0, min(x - half, max(0, w - int(size))))
    top = max(0, min(y - half, max(0, h - int(size))))
    return img.crop((left, top, min(w, left + int(size)), min(h, top + int(size))))


def _cric_external_points(mapper=None, crop=None):
    """CRIC TRAINING cells as POINT-CROPS: one (path, candset, (x,y,size)) per annotated nucleus, so
    each record is a single-cell window consistent with the eval crop -- NOT the whole field (which
    would attach many conflicting per-cell labels to the same image)."""
    from data.external_tier2 import _read_cric_rows, CRIC_ROOT, CRIC_CSV
    mapper = mapper or L.hicervix_candidates
    crop = int(crop if crop is not None else C.CRIC_CROP)
    out = []
    if not os.path.isfile(CRIC_CSV):
        return out
    for fname, x, y, native in _read_cric_rows(CRIC_CSV):
        cs = mapper(native)
        if cs is None:
            continue
        path = os.path.join(CRIC_ROOT, fname)
        if os.path.isfile(path):
            out.append((path, cs, (int(round(float(x))), int(round(float(y))), crop)))
    return out


def external_cells(sources=("herlev", "apacc", "hicervix", "cric"), hicervix_splits=("train", "val")):
    """Enumerate (path, candidate_frozenset, crop) for all requested external sources (None dropped).

    ``crop`` is None for pre-cropped single-cell datasets and (x,y,size) for CRIC point-crops (so CRIC
    contributes single-cell windows, consistent with eval, not the whole field). Returns (cells, counts).
    """
    cells, counts = [], {}
    for name in sources:
        if name == "herlev":
            got = [(p, cs, None) for p, cs in _folder_external(C.HERLEV_DIR, L.herlev_candidates)]
        elif name == "apacc":
            got = [(p, cs, None) for p, cs in _folder_external(C.APACC_DIR, L.apacc_candidates)]
        elif name == "hicervix":
            got = [(p, cs, None) for p, cs in _hicervix_external(hicervix_splits)]
        elif name == "cric":
            got = _cric_external_points()                       # point-crops: (path, cs, (x,y,size))
        else:
            raise ValueError(f"unknown external source {name!r}")
        cells.extend(got)
        counts[name] = len(got)
    return cells, counts


def external_cells8(sources=("herlev", "apacc", "hicervix", "cric"), hicervix_splits=("train", "val")):
    """Like ``external_cells`` but candidate sets are over the 8 Bethesda leaves, and each record is
    tagged with its source. Glandular/microbe/high-grade cells (previously dropped) now map to the new
    non-squamous nodes. Returns (cells, per_source_counts) with cells = list[(path, frozenset8, source)]."""
    cells, counts = [], {}
    for name in sources:
        if name == "herlev":
            got = [(p, cs, None) for p, cs in _folder_external(C.HERLEV_DIR, L.herlev8_candidates)]
        elif name == "apacc":
            got = [(p, cs, None) for p, cs in _folder_external(C.APACC_DIR, L.apacc8_candidates)]
        elif name == "hicervix":
            got = [(p, cs, None) for p, cs in _hicervix_external(hicervix_splits, mapper=L.bethesda8_candidates)]
        elif name == "cric":
            got = _cric_external_points(mapper=L.bethesda8_candidates)   # point-crops: (path, cs, (x,y,size))
        else:
            raise ValueError(f"unknown external source {name!r}")
        cells.extend((p, cs, name, crop) for (p, cs, crop) in got)
        counts[name] = len(got)
    return cells, counts


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
class PartialLabelDataset(Dataset):
    """Yields (image_tensor, candidate_mask[NUM_CLASSES] multi-hot float).

    Built from a list of (path, candset) records where candset is a frozenset / iterable of
    SIPaKMeD class indices (singleton for SIPaKMeD train cells, multi-class for external cells).
    The mask is precomputed lazily per item so the dataset is cheap to construct.
    """

    def __init__(self, records, transform, n_leaves=None):
        # records: (path, candset[, weight[, crop]]); weight->1.0, crop->None when absent.
        # crop, when present, is (x, y, size) -> point-crop a single cell out of a FOV image (CRIC).
        self.records = [(r[0], r[1], (r[2] if len(r) > 2 else 1.0), (r[3] if len(r) > 3 else None))
                        for r in records]
        self.transform = transform
        self.n_leaves = C.NUM_CLASSES if n_leaves is None else int(n_leaves)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        path, candset, w, crop = self.records[i]
        img = Image.open(path).convert("RGB")
        if crop is not None:
            img = _point_crop(img, crop[0], crop[1], crop[2])
        return (self.transform(img), candidate_mask(candset, self.n_leaves),
                torch.tensor(float(w), dtype=torch.float32))


def build_partial_dataset(aug="photometric", img_size=C.IMG_SIZE, split_kind="grouped",
                          external_sources=("herlev", "apacc", "hicervix", "cric"),
                          hicervix_splits=("train", "val"), train=True, transform=None,
                          taxonomy="sipakmed5", w_src=None,
                          include_sipakmed=True):
    """Mixed partial-label dataset: SIPaKMeD TRAIN singletons + external candidate sets.

    SIPaKMeD train cells (leakage-safe grouped split) get an EXACT singleton candidate set {true class}.
    External cells get their candidate set via the labelmap crosswalks (None -> dropped). Each record
    carries a per-source loss WEIGHT from ``w_src`` (default: SIPaKMeD 1.0, externals 1.0 for the 5-class
    path / 0.3 for the 8-leaf path), so coarse external labels can be down-weighted.

    ``taxonomy``:
      - 'sipakmed5' (default): 5-leaf candidates (``EXTERNAL_CANDIDATES`` + ``hicervix_candidates``).
      - 'bethesda8'         : 8-leaf candidates (``EXTERNAL_CANDIDATES8`` + ``bethesda8_candidates``),
        un-excluding glandular/microbe/high-grade external cells via the 3 new non-squamous nodes.
    ``include_sipakmed=False`` builds an external-only dataset (for the train→eval source matrix).
    Returns (dataset, stats).
    """
    if transform is None:
        transform = build_transform(train=train, aug=aug, img_size=img_size)
    is8 = (taxonomy == "bethesda8")
    n_leaves = L.N_LEAVES8 if is8 else C.NUM_CLASSES
    w_src = dict(w_src or {})
    w_sip = float(w_src.get("sipakmed", 1.0))
    ext_default = float(w_src.get("external", 0.3 if is8 else 1.0))

    split = get_split(split_kind)
    # drive-letter-AGNOSTIC + case/sep-normalized: the split may store f:\ paths while load_cells()
    # returns e:\ (or F:\) for the same volume depending on how python was invoked. splitdrive strips
    # the drive so the project-relative path is what matches.
    _norm = lambda p: os.path.normcase(os.path.splitdrive(os.path.normpath(p))[1])
    train_paths = set(_norm(p) for p in split["train"])
    cells = load_cells()
    sipak = ([(c.path, frozenset((c.class_idx,)), w_sip) for c in cells if _norm(c.path) in train_paths]
             if include_sipakmed else [])

    if is8:
        ext, ext_counts = external_cells8(external_sources, hicervix_splits=hicervix_splits)
        ext_records = [(p, cs, float(w_src.get(src, ext_default)), crop) for (p, cs, src, crop) in ext]
    else:
        ext, ext_counts = external_cells(external_sources, hicervix_splits=hicervix_splits)
        ext_records = [(p, cs, ext_default, crop) for (p, cs, crop) in ext]

    records = sipak + ext_records
    ds = PartialLabelDataset(records, transform, n_leaves=n_leaves)
    # per-record source tag (for the optional source-balanced sampler in train_partial)
    if is8:
        ds.record_sources = ["sipakmed"] * len(sipak) + [src for (_p, _cs, src, _crop) in ext]
    else:
        ds.record_sources = ["sipakmed"] * len(sipak) + ["external"] * len(ext)
    stats = dict(
        n_total=len(records), n_sipakmed_singleton=len(sipak), n_external=len(ext_records),
        external_per_source=ext_counts, external_sources=list(external_sources),
        hicervix_splits=list(hicervix_splits), split_kind=split_kind,
        taxonomy=taxonomy, n_leaves=n_leaves, w_src={**{"sipakmed": w_sip, "external": ext_default}, **w_src},
    )
    return ds, stats


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #
def _autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(C.USE_BF16 and C.DEVICE == "cuda"))


def _sipakmed_test_loader(split_kind="grouped", aug="photometric", img_size=C.IMG_SIZE,
                          batch_size=C.BATCH_SIZE):
    """Sealed SIPaKMeD test loader yielding (image, int class label) for standard 5-class metrics."""
    from data.dataset import CellDataset
    split = get_split(split_kind)
    tf = build_transform(train=False, aug=aug, img_size=img_size)
    ds = CellDataset(split["test"], tf)              # labels inferred from path
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=C.NUM_WORKERS, pin_memory=False)


def train_partial(backbone="efficientnet_b0", epochs=None, lr=None, hidden=None, dropout=None,
                  excl_weight=0.0, label_smoothing=0.0, aug="photometric", img_size=C.IMG_SIZE,
                  split_kind="grouped", external_sources=("herlev", "apacc", "hicervix", "cric"),
                  hicervix_splits=("train", "val"), batch_size=C.BATCH_SIZE, weight_decay=1e-4,
                  optimizer="adamw", pretrained=True, tag="student_partiallabel", save=True,
                  device=None, logger=None, taxonomy="sipakmed5", w_src=None,
                  balanced_sampler=False, sipakmed_frac=0.4):
    """Train a shared-encoder 5-logit classifier with PartialLabelLoss over the mixed dataset.

    SIPaKMeD train cells supervise exact leaves (singleton CE); external cells supervise marginal
    candidate-set mass. Early-eval is standard 5-class CE metrics on the SEALED SIPaKMeD test split.
    Saves ``{tag}.pt`` (a plain state_dict, like ``student.pt``) under C.CKPTDIR. Returns a dict
    with the dataset stats, per-epoch history, and the final sealed-test metrics.
    """
    C.set_seed()
    device = device or C.DEVICE
    scfg = dict(C.STUDENT)
    hidden = scfg["hidden"] if hidden is None else hidden
    dropout = scfg["dropout"] if dropout is None else dropout
    lr = scfg["lr"] if lr is None else lr
    epochs = scfg["epochs"] if epochs is None else epochs

    n_leaves = L.N_LEAVES8 if taxonomy == "bethesda8" else C.NUM_CLASSES

    # ---- data ----
    ds, stats = build_partial_dataset(aug=aug, img_size=img_size, split_kind=split_kind,
                                      external_sources=external_sources, hicervix_splits=hicervix_splits,
                                      taxonomy=taxonomy, w_src=w_src)
    if balanced_sampler:
        # source-balanced sampling: prevent the large external pools (HiCervix 32k) from swamping the
        # 3238 SIPaKMeD singletons (flat mixing collapses the in-distribution 5-class signal). Target:
        # SIPaKMeD gets ``sipakmed_frac`` of the sampled mass; the rest is split EQUALLY across the
        # external sources (so HiCervix doesn't dominate the external portion either).
        from collections import Counter
        srcs = ds.record_sources
        cnt = Counter(srcs)
        ext_srcs = [s for s in cnt if s != "sipakmed"]
        targets = {"sipakmed": float(sipakmed_frac)}
        if ext_srcs:
            per = (1.0 - float(sipakmed_frac)) / len(ext_srcs)
            for s in ext_srcs:
                targets[s] = per
        weights = [targets[s] / cnt[s] for s in srcs]
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=C.NUM_WORKERS,
                            pin_memory=False, drop_last=False)
        print(f"[route-c] source-balanced sampler: sipakmed_frac={sipakmed_frac}, "
              f"per-source counts={dict(cnt)}")
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=C.NUM_WORKERS,
                            pin_memory=False, drop_last=False)
    test_ld = _sipakmed_test_loader(split_kind=split_kind, aug=aug, img_size=img_size,
                                    batch_size=batch_size)

    # ---- model / loss / optim ----
    model = build(backbone, n_leaves, hidden, dropout, pretrained=pretrained).to(device)
    crit = PartialLabelLoss(excl_weight=excl_weight, label_smoothing=label_smoothing, reduction="none")

    def _sealed_metrics():
        """5-class sealed-test metrics; for the 8-leaf head, collapse softmax to the 5 squamous leaves."""
        yy, pp = M.predict(model, test_ld)
        pp = np.asarray(pp)
        if pp.shape[1] != C.NUM_CLASSES:                       # 8-leaf -> renormalize over squamous leaves
            p5 = pp[:, :C.NUM_CLASSES]
            pp = p5 / p5.sum(1, keepdims=True).clip(min=1e-9)
        return yy, pp
    params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    total, trainable = count_params(model)
    print(f"[route-c] {backbone} params={total/1e6:.1f}M | dataset: {stats['n_total']} "
          f"(sipakmed {stats['n_sipakmed_singleton']} + external {stats['n_external']} "
          f"{stats['external_per_source']})")
    if logger:
        logger.event(f"route-c data: {stats}")

    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time(); run_loss, n = 0.0, 0
        for xb, mb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with _autocast():
                logits = model(xb)
                row = crit(logits.float(), mb)                # per-row partial-label NLL (reduction='none')
            ns = mb.sum(1)                                     # candidate-set size per row
            usable = ((ns > 0.5) & (ns < n_leaves - 0.5)).to(row.dtype)   # match reduction='mean' denom
            w = wb * usable
            loss = (row * w).sum() / w.sum().clamp_min(1.0)    # per-source weighted mean over usable rows
            loss.backward(); opt.step()
            run_loss += loss.item() * xb.size(0); n += xb.size(0)
        tr_loss = run_loss / max(1, n); dt = time.time() - t0

        # sealed-test 5-class metrics (8-leaf head collapsed to the 5 squamous leaves)
        y, probs = _sealed_metrics()
        im = M.indist_metrics(y, probs)
        rec = dict(epoch=ep, partial_loss=tr_loss, epoch_time=dt,
                   test_accuracy=im["accuracy"], test_balanced_accuracy=im["balanced_accuracy"],
                   test_macro_f1=im["macro_f1"], test_kappa=im["cohen_kappa"])
        history.append(rec)
        print(f"[route-c] ep {ep:02d}/{epochs}  partial_loss={tr_loss:.4f}  "
              f"sealed-test acc={im['accuracy']:.4f} balacc={im['balanced_accuracy']:.4f} "
              f"macroF1={im['macro_f1']:.4f}  ({dt:.0f}s)")
        if logger:
            logger.epoch(tag, rec)
        if save:        # per-epoch checkpoint so a long multi-source run survives a crash
            os.makedirs(C.CKPTDIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(C.CKPTDIR, f"{tag}_last.pt"))

    # final sealed-test metrics (full dict)
    y, probs = _sealed_metrics()
    final_metrics = M.indist_metrics(y, probs)

    out = dict(backbone=backbone, params=int(total), config=dict(
        epochs=epochs, lr=lr, hidden=hidden, dropout=dropout, excl_weight=excl_weight,
        label_smoothing=label_smoothing, optimizer=optimizer, weight_decay=weight_decay,
        batch_size=batch_size, aug=aug, img_size=img_size, split_kind=split_kind,
        external_sources=list(external_sources), hicervix_splits=list(hicervix_splits),
        pretrained=pretrained, taxonomy=taxonomy, n_leaves=n_leaves),
        dataset=stats, history=history, sealed_test=final_metrics)

    if save:
        os.makedirs(C.CKPTDIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(C.CKPTDIR, f"{tag}.pt"))
        runlog.dump_json(os.path.join(C.METRICDIR, f"{tag}.json"), out)
        print(f"[route-c] saved {os.path.join(C.CKPTDIR, tag + '.pt')} + metrics json")
    out["model"] = model
    return out


# --------------------------------------------------------------------------- #
#  CPU smoke
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(C.SEED)
    K = C.NUM_CLASSES
    N = 16

    crit = PartialLabelLoss(reduction="none")
    logits = torch.randn(N, K)
    targets = torch.randint(0, K, (N,))

    # (1) SINGLETON candidate masks must reduce EXACTLY to standard cross-entropy.
    singleton = torch.zeros(N, K)
    singleton[torch.arange(N), targets] = 1.0
    partial = crit(logits, singleton)                       # (N,)
    ce = F.cross_entropy(logits, targets, reduction="none")  # (N,)
    max_abs = (partial - ce).abs().max().item()
    assert torch.allclose(partial, ce, atol=1e-5), f"singleton != CE (max abs diff {max_abs:.2e})"
    print(f"singleton partial-loss == CE : max|diff| = {max_abs:.2e}  OK")

    # mean reduction equals mean CE too.
    crit_mean = PartialLabelLoss(reduction="mean")
    pm = crit_mean(logits, singleton).item()
    cem = F.cross_entropy(logits, targets).item()
    assert abs(pm - cem) < 1e-5, f"mean singleton {pm} != mean CE {cem}"
    print(f"mean singleton {pm:.4f} == mean CE {cem:.4f}  OK")

    # (2) MULTI-candidate masks: finite, non-negative, and <= the CE of the worst leaf in S
    #     (marginal mass on a SET is always >= mass on any single member -> lower NLL).
    multi = torch.zeros(N, K)
    multi[:, C.ABNORMAL_IDX] = 1.0                           # S = {Koilocytotic, Dyskeratotic}
    pmulti = crit(logits, multi)
    assert torch.isfinite(pmulti).all(), "multi-candidate loss not finite"
    assert (pmulti >= -1e-6).all(), "marginal NLL should be >= 0"
    # for each row, marginal NLL(S) <= min_{k in S} CE(k)
    logp = F.log_softmax(logits, dim=1)
    ce_members = -logp[:, C.ABNORMAL_IDX]                    # (N, |S|)
    assert (pmulti <= ce_members.min(dim=1).values + 1e-5).all(), "marginal mass <= best member NLL violated"
    print(f"multi-candidate partial-loss finite, mean={pmulti.mean().item():.4f}  OK")

    # (3) exclusion term: forbidding a class is finite and adds non-negative cost.
    crit_excl = PartialLabelLoss(excl_weight=1.0, reduction="mean")
    forbidden = torch.zeros(N, K); forbidden[:, 0] = 1.0     # forbid class 0
    lex = crit_excl(logits, multi, forbidden_mask=forbidden).item()
    assert np.isfinite(lex), "exclusion loss not finite"
    print(f"exclusion-augmented partial-loss = {lex:.4f}  (finite)  OK")

    # (4) label-smoothing variant stays finite and reduces to CE-ish for singletons.
    crit_ls = PartialLabelLoss(label_smoothing=0.1, reduction="mean")
    lls = crit_ls(logits, singleton).item()
    assert np.isfinite(lls), "label-smoothed loss not finite"
    print(f"label-smoothed singleton partial-loss = {lls:.4f}  (finite)  OK")

    # (5) candidate_mask helper + a tiny random-model forward to exercise the train path shapes.
    m = candidate_mask(frozenset(C.ABNORMAL_IDX))
    assert m.tolist() == [0, 0, 0, 1, 1], m.tolist()
    net = build("efficientnet_b0", C.NUM_CLASSES, hidden=32, dropout=0.0, pretrained=False).eval()
    x = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        lo = net(x)
    cm = torch.zeros(4, K); cm[:, C.ABNORMAL_IDX] = 1.0
    sl = PartialLabelLoss()(lo, cm)
    assert torch.isfinite(sl), "model-path partial-loss not finite"
    print(f"random-model partial-loss on (4,5) logits = {sl.item():.4f}  OK")

    print("route_c CPU smoke: OK")
