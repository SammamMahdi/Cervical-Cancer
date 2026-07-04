"""U3 / U5 evaluation: in-distribution (5-class) + cross-dataset (E0/E1) metrics.

Two regimes, one module:
  * In-distribution  -- the held-out SIPaKMeD test split, scored as a 5-class problem
    (accuracy, balanced accuracy, macro-F1, Cohen's kappa, macro one-vs-rest AUROC, the
    full per-class classification report, confusion matrix), plus calibration (ECE),
    bootstrap confidence intervals and the paired McNemar test for model-vs-model deltas.
  * Cross-dataset    -- Herlev / APACC / HiCervix scored under the safe many-to-one label
    crosswalk (data.labelmap). The 5-class head's prediction is collapsed to E0
    (normal/abnormal) or E1 (normal/low/high); P(abnormal) drives the AUROC and the
    pre-registered C.ABNORMAL_THRESHOLD operating point. High-grade recall is reported for
    E1 and is ~0 BY CONSTRUCTION (the model has no HSIL output) -- that is the honest point.

All heavy inference goes through `predict` (eval / no_grad / BF16 autocast / softmax).
Datasets (FolderExternal, HiCervixExternal) expose `.n` (kept) and `.excluded` (dropped:
unmapped folder or mapped tier == None) so the "excluded fraction" honesty metric is exact.
"""
import os
import json
import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score,
    roc_auc_score, classification_report, confusion_matrix, recall_score,
)

import config as C
from data.dataset import build_transform
from data import labelmap as L


def balanced_accuracy(y_true, y_pred):
    """Balanced accuracy = mean per-class recall over classes PRESENT in y_true. sklearn warns
    ("y_pred contains classes not in y_true") when the model predicts a class absent from y_true
    (degenerate but harmless — e.g. a smoke run or a single-class cross-dataset slice); the value is
    still correct, so we silence only that specific benign warning here."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return float(balanced_accuracy_score(y_true, y_pred))

IMG_EXT = (".bmp", ".png", ".jpg", ".jpeg")


# --------------------------------------------------------------------------- #
#  Inference
# --------------------------------------------------------------------------- #
def predict(model, loader, desc="eval"):
    """Run `model` over `loader` -> (y_true [N] int64, probs [N,5] float64).

    Eval mode, no_grad, BF16 autocast on CUDA, softmax over the 5-class logits. A tqdm bar (labelled
    `desc`) tracks batch progress (useful on the large external sets).
    """
    model.eval()
    device = next(model.parameters()).device
    ys, ps = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False, unit="batch"):
            xb, yb = batch[0], batch[1]
            xb = xb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(C.DEVICE == "cuda")):
                logits = model(xb)
            # softmax in fp32 for numerically-stable, comparable probabilities
            probs = torch.softmax(logits.float(), dim=1)
            ps.append(probs.cpu().numpy())
            ys.append(np.asarray(yb))
    y = np.concatenate(ys).astype(np.int64)
    probs = np.concatenate(ps).astype(np.float64)
    return y, probs


# --------------------------------------------------------------------------- #
#  In-distribution (5-class) metrics
# --------------------------------------------------------------------------- #
def indist_metrics(y, probs):
    """5-class metrics dict for the SIPaKMeD held-out test set."""
    y = np.asarray(y).astype(int)
    probs = np.asarray(probs)
    pred = probs.argmax(1)
    labels = list(range(C.NUM_CLASSES))

    # macro one-vs-rest AUROC; guard against a class missing from y_true in tiny subsets
    try:
        auroc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        auroc = float("nan")

    report = classification_report(
        y, pred, labels=labels, target_names=C.CLASSES,
        output_dict=True, zero_division=0,
    )
    return dict(
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=balanced_accuracy(y, pred),
        macro_f1=float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        cohen_kappa=float(cohen_kappa_score(y, pred, labels=labels)),
        roc_auc_macro_ovr=auroc,
        per_class=report,
        confusion_matrix=confusion_matrix(y, pred, labels=labels).tolist(),
    )


def expected_calibration_error(probs, y, n_bins=15):
    """Top-label ECE (U5). Returns (ece, bin_conf, bin_acc) with len-n_bins arrays.

    Empty bins contribute 0 to the ECE and NaN to the returned bin curves (so a reliability
    diagram can skip them) -- this keeps the weighted ECE exact without phantom points.
    """
    probs = np.asarray(probs)
    y = np.asarray(y).astype(int)
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = np.full(n_bins, np.nan)
    bin_acc = np.full(n_bins, np.nan)
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # last bin closed on the right so conf == 1.0 is included
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        c = conf[m].mean()
        a = correct[m].mean()
        bin_conf[b] = c
        bin_acc[b] = a
        ece += (m.sum() / n) * abs(a - c)
    return float(ece), bin_conf, bin_acc


# --------------------------------------------------------------------------- #
#  Uncertainty: bootstrap CI + McNemar
# --------------------------------------------------------------------------- #
def bootstrap_ci(y, pred, metric_fn, n=1000, seed=C.SEED):
    """Paired percentile bootstrap of metric_fn(y, pred). Returns (mean, lo, hi) at 95%.

    `pred` may be hard labels OR per-row probabilities; metric_fn receives the resampled
    (y, pred) so it works for accuracy/F1 (labels) as well as AUROC-style probability fns.
    """
    y = np.asarray(y)
    pred = np.asarray(pred)
    rng = np.random.default_rng(seed)
    N = len(y)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        try:
            stats.append(float(metric_fn(y[idx], pred[idx])))
        except ValueError:
            # degenerate resample (e.g. a class vanished for AUROC) -> skip
            continue
    if not stats:
        return float("nan"), float("nan"), float("nan")
    stats = np.asarray(stats)
    return float(stats.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def mcnemar(y, pred_a, pred_b):
    """Paired McNemar test for two models on the SAME items.

    b = A correct & B wrong, c = A wrong & B correct (discordant pairs). Uses the exact
    binomial test when discordant pairs are few (b+c < 25), else the continuity-corrected
    chi-square. Returns dict(b, c, statistic, p_value).
    """
    y = np.asarray(y)
    a_ok = (np.asarray(pred_a) == y)
    b_ok = (np.asarray(pred_b) == y)
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    nd = b + c
    if nd == 0:
        return dict(b=b, c=c, statistic=0.0, p_value=1.0)
    if nd < 25:
        from scipy.stats import binomtest
        p = float(binomtest(min(b, c), nd, 0.5).pvalue)
        # report the smaller discordant count as the (exact-test) statistic
        return dict(b=b, c=c, statistic=float(min(b, c)), p_value=p)
    from scipy.stats import chi2
    stat = (abs(b - c) - 1.0) ** 2 / nd      # continuity-corrected chi-square, 1 dof
    return dict(b=b, c=c, statistic=float(stat), p_value=float(chi2.sf(stat, 1)))


# --------------------------------------------------------------------------- #
#  External datasets (collapsed-tier labels)
# --------------------------------------------------------------------------- #
_TIER_LABEL = {
    "E0": {"normal": 0, "abnormal": 1},
    "E1": {"normal": 0, "low": 1, "high": 2},
}


def _tier_int(tier, e0, e1):
    """(e0, e1) crosswalk pair -> integer label for `tier`, or None if excluded."""
    val = e0 if tier == "E0" else e1
    if val is None:
        return None
    return _TIER_LABEL[tier][val]


def _list_images(d):
    out = []
    for f in sorted(os.listdir(d)):
        if f.lower().endswith("-d.bmp"):       # Herlev segmentation mask -> skip
            continue
        if f.lower().endswith(IMG_EXT):
            fp = os.path.join(d, f)
            if os.path.getsize(fp) > 0:        # skip 0-byte / failed-download files
                out.append(fp)
    return out


def _safe_rgb(path):
    """Open an image as RGB; on a corrupt/unreadable file return a tiny black placeholder so a single
    bad cell never crashes a whole evaluation (the eval transform resizes it)."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (8, 8))


class FolderExternal(Dataset):
    """Folder-structured external dataset (Herlev / APACC) collapsed to a single tier.

    os.walk(root); for every directory whose basename `mapper(basename)` -> (e0, e1) has a
    NON-None label for `tier`, add its images (skipping Herlev '-d.bmp' masks). Images under
    unmapped folders, or whose folder maps to None for this tier, are counted in `.excluded`.
    __getitem__ -> (transformed tensor, int tier label). `.n` == kept items.
    """
    def __init__(self, root, mapper, transform, tier="E0"):
        assert tier in _TIER_LABEL, f"bad tier {tier}"
        self.root = root
        self.transform = transform
        self.tier = tier
        self.paths, self.labels = [], []
        self.excluded = 0
        for dirpath, _dirs, _files in os.walk(root):
            imgs = _list_images(dirpath)
            if not imgs:
                continue
            base = os.path.basename(dirpath)
            try:
                e0, e1 = mapper(base)
            except Exception:
                e0 = e1 = None
            lbl = _tier_int(tier, e0, e1)
            if lbl is None:
                self.excluded += len(imgs)          # unmapped folder or tier == EXCLUDE
                continue
            self.paths.extend(imgs)
            self.labels.extend([lbl] * len(imgs))
        self.n = len(self.paths)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = _safe_rgb(self.paths[i])
        return self.transform(img), self.labels[i]


class HiCervixExternal(Dataset):
    """HiCervix test set collapsed to a single tier via labelmap.hicervix_map.

    Reads C.HICERVIX_DIR/test.csv; image at <hicervix>/test/test/<image_name>. Maps each row
    by level_1 (falling back to class_name). Rows whose tier label is None, or whose image is
    missing on disk, are counted in `.excluded`. `.n` == kept items.
    """
    def __init__(self, transform, tier="E0", csv_path=None, img_root=None):
        assert tier in _TIER_LABEL, f"bad tier {tier}"
        import csv
        self.transform = transform
        self.tier = tier
        csv_path = csv_path or os.path.join(C.HICERVIX_DIR, "test.csv")
        img_root = img_root or os.path.join(C.HICERVIX_DIR, "test", "test")
        self.paths, self.labels = [], []
        self.excluded = 0
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                lvl1 = (row.get("level_1") or "").strip()
                key = lvl1 if lvl1 else (row.get("class_name") or "").strip()
                e0, e1 = L.hicervix_map(key)
                lbl = _tier_int(tier, e0, e1)
                name = row.get("image_name", "").strip()
                path = os.path.join(img_root, name)
                if lbl is None or not name or not os.path.isfile(path):
                    self.excluded += 1
                    continue
                self.paths.append(path)
                self.labels.append(lbl)
        self.n = len(self.paths)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = _safe_rgb(self.paths[i])
        return self.transform(img), self.labels[i]


# --------------------------------------------------------------------------- #
#  Cross-dataset evaluation
# --------------------------------------------------------------------------- #
def cross_dataset_eval(name, dataset, model, tier="E0", batch_size=C.BATCH_SIZE,
                       num_workers=C.NUM_WORKERS, save=True):
    """Score the 5-class `model` on an external `dataset` under the E0/E1 crosswalk.

    The model keeps its 5-class head; its argmax prediction is collapsed via labelmap and
    P(abnormal) (= softmax mass on the abnormal classes) drives AUROC and the pre-registered
    C.ABNORMAL_THRESHOLD operating point. Returns (and, by default, persists to
    C.METRICDIR/cross_<name>_<tier>.json) a metrics dict.
    """
    assert tier in _TIER_LABEL, f"bad tier {tier}"
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    y, probs = predict(model, loader, desc=f"{name} {tier}")
    pabn = L.p_abnormal(probs)                      # (N,) P(abnormal)
    pred_idx = probs.argmax(1)

    # ----- collapse predictions and ground truth to the binary E0 frame -----
    # Ground-truth tier ints: E0 already {0:normal,1:abnormal}; for E1 collapse low/high -> abnormal(1).
    y = np.asarray(y).astype(int)
    y_bin = (y >= 1).astype(int)                    # any non-normal tier label == abnormal

    # E0 operating-point prediction: pre-registered threshold on P(abnormal)
    pred_bin = (pabn >= C.ABNORMAL_THRESHOLD).astype(int)

    cm = confusion_matrix(y_bin, pred_bin, labels=[0, 1])
    # sensitivity = recall of abnormal(1); specificity = recall of normal(0)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")

    # external calibration (§1.2): uncalibrated binary ECE (top-class conf vs accuracy, 15 bins)
    _conf = np.maximum(pabn, 1.0 - pabn); _corr = (pred_bin == y_bin).astype(float)
    _ed = np.linspace(0.0, 1.0, 16); _ece = 0.0
    for _b in range(15):
        _m = ((_conf > _ed[_b]) & (_conf <= _ed[_b + 1])) if _b else ((_conf >= _ed[_b]) & (_conf <= _ed[_b + 1]))
        if _m.any():
            _ece += float(_m.mean()) * abs(float(_corr[_m].mean()) - float(_conf[_m].mean()))

    try:
        auroc = float(roc_auc_score(y_bin, pabn)) if len(np.unique(y_bin)) == 2 else float("nan")
    except ValueError:
        auroc = float("nan")

    n = int(dataset.n)
    excluded = int(getattr(dataset, "excluded", 0))
    out = dict(
        dataset=name,
        tier=tier,
        n=n,
        excluded=excluded,
        excluded_fraction=float(excluded / (n + excluded)) if (n + excluded) else 0.0,
        accuracy=float(accuracy_score(y_bin, pred_bin)),
        balanced_accuracy=balanced_accuracy(y_bin, pred_bin),
        macro_f1=float(f1_score(y_bin, pred_bin, labels=[0, 1], average="macro", zero_division=0)),
        sensitivity=sensitivity,
        specificity=specificity,
        auroc=auroc,
        threshold=float(C.ABNORMAL_THRESHOLD),
        ece=float(_ece),
        confusion_matrix=cm.tolist(),
    )

    # E1 honesty metric: recall on the high-grade subset -- ~0 by construction (no HSIL head).
    if tier == "E1":
        high_mask = (y == _TIER_LABEL["E1"]["high"])
        n_high = int(high_mask.sum())
        # model can only ever say normal/low -> it NEVER predicts high; recall against the
        # binary abnormal call (the best the model can do for a high-grade cell).
        if n_high:
            out["highgrade_recall"] = float(pred_bin[high_mask].mean())  # frac flagged abnormal
        else:
            out["highgrade_recall"] = float("nan")
        out["n_highgrade"] = n_high

    if save:
        os.makedirs(C.METRICDIR, exist_ok=True)
        with open(os.path.join(C.METRICDIR, f"cross_{name}_{tier}.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    return out


def wilcoxon(a, b):
    """Wilcoxon signed-rank test for paired continuous metrics across folds/images (Ablation §1.3).
    a, b: paired per-unit scores (e.g. per-fold macro-F1 of two models). Returns dict(statistic, p_value)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) != len(b) or len(a) < 1 or np.allclose(a, b):
        return dict(statistic=0.0, p_value=1.0, n=int(len(a)))
    from scipy.stats import wilcoxon as _w
    try:
        st, p = _w(a, b)
        return dict(statistic=float(st), p_value=float(p), n=int(len(a)))
    except ValueError:
        return dict(statistic=0.0, p_value=1.0, n=int(len(a)))


# --------------------------------------------------------------------------- #
#  E-morph: candidate-set-hit (lenient morphology) evaluation
# --------------------------------------------------------------------------- #
class CandidateExternal(Dataset):
    """E-morph loader: yields (tensor, native_key_idx). Each native key maps to a morphology
    candidate set (frozenset of SIPaKMeD class indices). Native labels with NO candidate set
    (high-grade / glandular / indeterminate) are counted in `.excluded`. Exposes `.cand` (list of
    candidate frozensets aligned to key idx) and `.keys` (native label strings)."""
    def __init__(self, name, transform):
        self.transform = transform
        self.paths, self.key_idx, self.keys, self.cand = [], [], [], []
        self.excluded = 0
        key2idx = {}

        def add(path, native_key, candset):
            if candset is None:
                self.excluded += 1; return
            if native_key not in key2idx:
                key2idx[native_key] = len(self.keys); self.keys.append(native_key); self.cand.append(candset)
            self.paths.append(path); self.key_idx.append(key2idx[native_key])

        if name in ("herlev", "apacc"):
            root = C.HERLEV_DIR if name == "herlev" else C.APACC_DIR
            mapper = L.EXTERNAL_CANDIDATES[name]
            for dp, _d, _f in os.walk(root):
                imgs = _list_images(dp)
                if not imgs:
                    continue
                cs = mapper(os.path.basename(dp))
                for p in imgs:
                    add(p, os.path.basename(dp), cs)
        elif name == "hicervix":
            import csv
            img_root = os.path.join(C.HICERVIX_DIR, "test", "test")
            with open(os.path.join(C.HICERVIX_DIR, "test.csv"), newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    key = (row.get("level_1") or "").strip() or (row.get("class_name") or "").strip()
                    cs = L.hicervix_candidates(key)
                    nm = row.get("image_name", "").strip(); path = os.path.join(img_root, nm)
                    if not nm or not os.path.isfile(path):
                        if cs is not None: self.excluded += 1
                        continue
                    add(path, key, cs)
        else:
            raise ValueError(f"unknown external dataset {name}")
        self.n = len(self.paths)

    def __len__(self): return self.n
    def __getitem__(self, i):
        return self.transform(_safe_rgb(self.paths[i])), self.key_idx[i]


def emorph_eval(name, model, transform=None, batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS, save=True):
    """E-morph candidate-set-hit accuracy: a 5-class prediction is correct if it lies in the native
    label's morphology candidate set. Reports overall accuracy, per-native-label accuracy, and the
    excluded fraction (native labels with no morphology home). Returns/saves a metrics dict."""
    transform = transform or build_transform(train=False)
    ds = CandidateExternal(name, transform)
    if ds.n == 0:
        out = dict(dataset=name, tier="Emorph", n=0, excluded=int(ds.excluded), note="no mappable cells")
        return out
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    model.eval(); device = next(model.parameters()).device
    preds, kidx = [], []
    with torch.no_grad():
        for xb, kb in tqdm(loader, desc=f"{name} E-morph", leave=False, unit="batch"):
            xb = xb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(C.DEVICE == "cuda")):
                logits = model(xb)
            preds.append(logits.argmax(1).cpu().numpy()); kidx.append(np.asarray(kb))
    preds = np.concatenate(preds); kidx = np.concatenate(kidx)
    hit = np.array([int(preds[i]) in ds.cand[kidx[i]] for i in range(len(preds))], dtype=float)

    per_label = {}
    for ki, key in enumerate(ds.keys):
        m = (kidx == ki)
        if m.any():
            per_label[key] = dict(n=int(m.sum()), accuracy=float(hit[m].mean()))
    out = dict(dataset=name, tier="Emorph", n=int(ds.n), excluded=int(ds.excluded),
               excluded_fraction=float(ds.excluded / (ds.n + ds.excluded)) if (ds.n + ds.excluded) else 0.0,
               accuracy=float(hit.mean()), per_label_accuracy=per_label)
    if save:
        os.makedirs(C.METRICDIR, exist_ok=True)
        with open(os.path.join(C.METRICDIR, f"cross_{name}_Emorph.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    return out


def internal_external_delta(internal_e0, external_dict):
    """Per-metric Δ (external − internal) for each external dataset.

    `internal_e0`  : dict of the model's INTERNAL E0 metrics (same keys as cross_dataset_eval).
    `external_dict`: {dataset_name: cross_dataset_eval(...) dict}.
    Returns {dataset_name: {metric: {internal, external, delta}}} over the shared numeric keys
    -- the generalization-gap table (a negative delta == drop out-of-distribution).
    """
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "auroc"]
    out = {}
    for name, ext in external_dict.items():
        row = {}
        for k in keys:
            iv = internal_e0.get(k)
            ev = ext.get(k)
            if iv is None or ev is None:
                continue
            row[k] = dict(internal=float(iv), external=float(ev), delta=float(ev) - float(iv))
        out[name] = row
    return out


# --------------------------------------------------------------------------- #
#  8-leaf (Bethesda) cross-dataset evaluation  (Wave-2b)
# --------------------------------------------------------------------------- #
#  The 8-leaf model keeps an 8-logit head; it is scored on the SAME external loaders as the
#  5-class path (FolderExternal / HiCervixExternal / cric_dataset point-crops, all with binary
#  E0 ground-truth ints). Only the SCORE changes: P(abnormal) = labelmap.p_abnormal8(probs8) =
#  softmax mass on the abnormal leaves {3,4,5,7} (abnormal squamous + glandular + indeterminate;
#  microbe(6) is treated as benign). Everything below is additive: the 5-class functions are untouched.
def predict8(model, loader, desc="eval8"):
    """Run an 8-leaf `model` over `loader` -> (y_true [N] int64, probs8 [N,n_leaves] float64).

    Identical inference contract to `predict` (eval / no_grad / BF16 autocast / fp32 softmax) but
    the softmax is over the full leaf head (8 logits) rather than 5.
    """
    model.eval()
    device = next(model.parameters()).device
    ys, ps = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False, unit="batch"):
            xb, yb = batch[0], batch[1]
            xb = xb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(C.DEVICE == "cuda")):
                logits = model(xb)
            probs = torch.softmax(logits.float(), dim=1)
            ps.append(probs.cpu().numpy())
            ys.append(np.asarray(yb))
    y = np.concatenate(ys).astype(np.int64)
    probs = np.concatenate(ps).astype(np.float64)
    return y, probs


def cross_dataset_eval8(name, dataset, model, batch_size=C.BATCH_SIZE,
                        num_workers=C.NUM_WORKERS, threshold=None, save=False):
    """Score an 8-leaf `model` on an external `dataset` under the E0 (normal/abnormal) crosswalk.

    The dataset yields the SAME binary E0 ground-truth ints as the 5-class path (so coverage of the
    GROUND TRUTH is unchanged); the only difference is the decision score: P(abnormal) is the 8-leaf
    softmax mass on leaves {3,4,5,7} (`labelmap.p_abnormal8`). Reports acc / balacc / sens / spec /
    auroc / ece (binary, 15-bin) + confusion matrix at the pre-registered C.ABNORMAL_THRESHOLD.
    """
    thr = float(C.ABNORMAL_THRESHOLD if threshold is None else threshold)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    y, probs8 = predict8(model, loader, desc=f"{name} E0(8)")
    pabn = L.p_abnormal8(probs8)                      # (N,) mass on abnormal leaves {3,4,5,7}
    y = np.asarray(y).astype(int)
    y_bin = (y >= 1).astype(int)
    pred_bin = (pabn >= thr).astype(int)

    cm = confusion_matrix(y_bin, pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")

    _conf = np.maximum(pabn, 1.0 - pabn); _corr = (pred_bin == y_bin).astype(float)
    _ed = np.linspace(0.0, 1.0, 16); _ece = 0.0
    for _b in range(15):
        _m = ((_conf > _ed[_b]) & (_conf <= _ed[_b + 1])) if _b else ((_conf >= _ed[_b]) & (_conf <= _ed[_b + 1]))
        if _m.any():
            _ece += float(_m.mean()) * abs(float(_corr[_m].mean()) - float(_conf[_m].mean()))

    try:
        auroc = float(roc_auc_score(y_bin, pabn)) if len(np.unique(y_bin)) == 2 else float("nan")
    except ValueError:
        auroc = float("nan")

    n = int(getattr(dataset, "n", len(dataset)))
    excluded = int(getattr(dataset, "excluded", 0))
    out = dict(
        dataset=name, tier="E0", head="8leaf", n=n, excluded=excluded,
        excluded_fraction=float(excluded / (n + excluded)) if (n + excluded) else 0.0,
        accuracy=float(accuracy_score(y_bin, pred_bin)),
        balanced_accuracy=balanced_accuracy(y_bin, pred_bin),
        macro_f1=float(f1_score(y_bin, pred_bin, labels=[0, 1], average="macro", zero_division=0)),
        sensitivity=sensitivity, specificity=specificity, auroc=auroc,
        threshold=thr, ece=float(_ece), confusion_matrix=cm.tolist(),
    )
    if save:
        os.makedirs(C.METRICDIR, exist_ok=True)
        with open(os.path.join(C.METRICDIR, f"cross8_{name}_E0.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    return out


class HiCervixNonSquamous(Dataset):
    """HiCervix test rows for the E2 non-squamous-detection task (8-leaf only).

    Positive (1) = level_1 in {AGC (glandular), microbe}; negative (0) = squamous (level_1 in
    {negative, ASC}). Rows whose level_1 is neither, or whose image is missing, are dropped and
    counted in `.excluded`. The 8-leaf score for AUROC is the softmax mass on the NON-SQUAMOUS
    leaves {GLANDULAR(5), MICROBE(6), INDETERMINATE(7)} (computed by the caller). `.n` == kept items.
    """
    NONSQ = ("agc", "glandular", "adenocarc", "adc", "endocervical", "endometrial", "ais",
             "microbe", "actino", "fungi", "trich", "tri", "hsv", "organism", "candida")
    SQ = ("negative", "nilm", "normal", "inflammat", "reactive", "benign",
          "asc", "atypical", "lsil", "low", "hsil", "high", "scc", "carcinoma")

    def __init__(self, transform, csv_path=None, img_root=None):
        import csv
        self.transform = transform
        csv_path = csv_path or os.path.join(C.HICERVIX_DIR, "test.csv")
        img_root = img_root or os.path.join(C.HICERVIX_DIR, "test", "test")
        self.paths, self.labels = [], []
        self.excluded = 0
        self.n_pos = self.n_neg = 0
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row.get("level_1") or "").strip().lower() or (row.get("class_name") or "").strip().lower()
                if any(k in key for k in self.NONSQ):
                    lbl = 1
                elif any(k in key for k in self.SQ):
                    lbl = 0
                else:
                    self.excluded += 1; continue
                name = (row.get("image_name") or "").strip()
                path = os.path.join(img_root, name)
                if not name or not os.path.isfile(path):
                    self.excluded += 1; continue
                self.paths.append(path); self.labels.append(lbl)
                if lbl: self.n_pos += 1
                else:   self.n_neg += 1
        self.n = len(self.paths)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.transform(_safe_rgb(self.paths[i])), self.labels[i]


def nonsquamous_eval8(model, transform, batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS, save=False):
    """E2: HiCervix non-squamous detection with the 8-leaf head.

    positive = AGC|microbe, negative = squamous; score = softmax mass on non-squamous leaves
    {5,6,7}. Reports AUROC (the headline) + acc/sens/spec at a 0.5 score threshold, n_pos/n_neg.
    """
    ds = HiCervixNonSquamous(transform)
    if ds.n == 0 or ds.n_pos == 0 or ds.n_neg == 0:
        return dict(task="E2_nonsquamous", n=int(ds.n), n_pos=int(ds.n_pos), n_neg=int(ds.n_neg),
                    excluded=int(ds.excluded), auroc=None, note="degenerate (need both classes)")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    y, probs8 = predict8(model, loader, desc="hicervix E2(8)")
    nonsq = sorted((L.GLANDULAR8, L.MICROBE8, L.INDETERMINATE8))   # {5,6,7}
    score = np.asarray(probs8)[:, nonsq].sum(1)
    y = np.asarray(y).astype(int)
    pred = (score >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    try:
        auroc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None
    except ValueError:
        auroc = None
    out = dict(task="E2_nonsquamous", n=int(ds.n), n_pos=int(ds.n_pos), n_neg=int(ds.n_neg),
               excluded=int(ds.excluded), nonsquamous_leaves=nonsq, auroc=auroc,
               accuracy=float(accuracy_score(y, pred)),
               balanced_accuracy=balanced_accuracy(y, pred),
               sensitivity=float(tp / (tp + fn)) if (tp + fn) else float("nan"),
               specificity=float(tn / (tn + fp)) if (tn + fp) else float("nan"),
               confusion_matrix=cm.tolist(), threshold=0.5)
    if save:
        os.makedirs(C.METRICDIR, exist_ok=True)
        with open(os.path.join(C.METRICDIR, "cross8_hicervix_E2.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    return out


# -- coverage: 5-class (None-excluded) vs 8-leaf (non-empty candidate) excluded fraction --
def _folder_excluded_counts(root, map5, map8):
    """Walk a Herlev/APACC folder tree; return (total, excl5, excl8) image counts.

    excl5 counts images whose folder maps to None for E0 under the 5-class crosswalk; excl8 counts
    images whose 8-leaf candidate set is empty/None. Denominator (total) is the SAME for both."""
    total = excl5 = excl8 = 0
    for dp, _d, _f in os.walk(root):
        imgs = _list_images(dp)
        if not imgs:
            continue
        base = os.path.basename(dp)
        try:
            e0, _e1 = map5(base)
        except Exception:
            e0 = None
        cs8 = map8(base)
        n = len(imgs)
        total += n
        if e0 is None:
            excl5 += n
        if not cs8:
            excl8 += n
    return total, excl5, excl8


def excluded_fraction_5_vs_8(save=False):
    """Coverage gain from the 8-leaf head: per external dataset, the fraction of cells with NO
    home under the 5-class crosswalk (E0 -> None) vs the 8-leaf taxonomy (empty candidate set),
    over a CONSTANT denominator. Folder datasets (Herlev/APACC) walk the tree; HiCervix/CRIC read
    their label tables. Returns {dataset: {total, excluded_before, excluded_after, frac_before,
    frac_after}}.
    """
    out = {}

    def _record(name, total, e5, e8):
        out[name] = dict(total=int(total), excluded_before=int(e5), excluded_after=int(e8),
                         frac_before=float(e5 / total) if total else 0.0,
                         frac_after=float(e8 / total) if total else 0.0)

    # Herlev / APACC (folder structured)
    t, e5, e8 = _folder_excluded_counts(C.HERLEV_DIR, L.herlev_map, L.herlev8_candidates)
    _record("herlev", t, e5, e8)
    t, e5, e8 = _folder_excluded_counts(C.APACC_DIR, L.apacc_map, L.apacc8_candidates)
    _record("apacc", t, e5, e8)

    # HiCervix (level_1 label table)
    import csv as _csv
    csv_path = os.path.join(C.HICERVIX_DIR, "test.csv")
    if os.path.isfile(csv_path):
        t = e5 = e8 = 0
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                key = (row.get("level_1") or "").strip() or (row.get("class_name") or "").strip()
                e0, _e1 = L.hicervix_map(key)
                cs8 = L.bethesda8_candidates(key)
                t += 1
                if e0 is None: e5 += 1
                if not cs8:    e8 += 1
        _record("hicervix", t, e5, e8)

    # CRIC (Bethesda token table)
    try:
        from data.external_tier2 import _read_cric_rows, CRIC_CSV
        if os.path.isfile(CRIC_CSV):
            t = e5 = e8 = 0
            for _fn, _x, _y, native in _read_cric_rows(CRIC_CSV):
                e0, _e1 = L.hicervix_map(native)
                cs8 = L.bethesda8_candidates(native)
                t += 1
                if e0 is None: e5 += 1
                if not cs8:    e8 += 1
            _record("cric", t, e5, e8)
    except Exception:
        pass

    if save:
        os.makedirs(C.METRICDIR, exist_ok=True)
        with open(os.path.join(C.METRICDIR, "excluded_fraction_5_vs_8.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    return out


class SharedCellFolderE0(Dataset):
    """Folder external (Herlev/APACC) restricted to cells representable under BOTH the 5-class
    crosswalk (E0 != None) AND the 8-leaf taxonomy (non-empty candidate). Yields (tensor, E0 int).

    This is the SHARED-CELL E0 CONTROL: scoring the 8-leaf head on exactly the cells the 5-class
    head could already see isolates the accuracy change from the coverage change. `.n`/`.excluded`
    track kept vs dropped-as-not-shared."""
    def __init__(self, root, map5, map8, transform):
        self.transform = transform
        self.paths, self.labels = [], []
        self.excluded = 0
        for dp, _d, _f in os.walk(root):
            imgs = _list_images(dp)
            if not imgs:
                continue
            base = os.path.basename(dp)
            try:
                e0, _e1 = map5(base)
            except Exception:
                e0 = None
            cs8 = map8(base)
            shared = (e0 is not None) and bool(cs8)
            if not shared:
                self.excluded += len(imgs); continue
            lbl = 1 if e0 == "abnormal" else 0
            self.paths.extend(imgs); self.labels.extend([lbl] * len(imgs))
        self.n = len(self.paths)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.transform(_safe_rgb(self.paths[i])), self.labels[i]


class SharedCellHiCervixE0(Dataset):
    """HiCervix test rows representable under BOTH maps (E0 != None AND non-empty 8-leaf candidate).
    Yields (tensor, E0 int). Shared-cell E0 control for the LBC preparation."""
    def __init__(self, transform, csv_path=None, img_root=None):
        import csv
        self.transform = transform
        csv_path = csv_path or os.path.join(C.HICERVIX_DIR, "test.csv")
        img_root = img_root or os.path.join(C.HICERVIX_DIR, "test", "test")
        self.paths, self.labels = [], []
        self.excluded = 0
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row.get("level_1") or "").strip() or (row.get("class_name") or "").strip()
                e0, _e1 = L.hicervix_map(key)
                cs8 = L.bethesda8_candidates(key)
                name = (row.get("image_name") or "").strip()
                path = os.path.join(img_root, name)
                shared = (e0 is not None) and bool(cs8)
                if not shared or not name or not os.path.isfile(path):
                    self.excluded += 1; continue
                self.paths.append(path); self.labels.append(1 if e0 == "abnormal" else 0)
        self.n = len(self.paths)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.transform(_safe_rgb(self.paths[i])), self.labels[i]


def cric_shared_e0_dataset(transform, crop=None):
    """CRIC point-crops restricted to cells shared under BOTH maps (E0 != None AND non-empty 8-leaf
    candidate). Reuses external_tier2.PointCropExternal but with a label_mapper that returns
    (EXCLUDE, EXCLUDE) for any token not representable in both spaces."""
    from data.external_tier2 import PointCropExternal, _read_cric_rows, CRIC_ROOT, CRIC_CSV

    def _shared_mapper(native):
        e0, e1 = L.hicervix_map(native)
        cs8 = L.bethesda8_candidates(native)
        if e0 is None or not cs8:
            return (None, None)
        return (e0, e1)

    crop = crop if crop is not None else C.CRIC_CROP
    annotations = list(_read_cric_rows(CRIC_CSV)) if os.path.isfile(CRIC_CSV) else []
    return PointCropExternal(CRIC_ROOT, annotations, _shared_mapper, tier="E0",
                             crop=crop, transform=transform)


if __name__ == "__main__":
    # Light, dependency-free smoke test (no model / no disk inference).
    rng = np.random.default_rng(C.SEED)
    N = 400
    y = rng.integers(0, C.NUM_CLASSES, N)
    logits = rng.normal(size=(N, C.NUM_CLASSES))
    logits[np.arange(N), y] += 2.0                         # make argmax mostly correct
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)

    m = indist_metrics(y, probs)
    print("indist acc=%.3f bal_acc=%.3f macroF1=%.3f kappa=%.3f auroc=%.3f"
          % (m["accuracy"], m["balanced_accuracy"], m["macro_f1"],
             m["cohen_kappa"], m["roc_auc_macro_ovr"]))

    ece, bc, ba = expected_calibration_error(probs, y, n_bins=15)
    print("ECE=%.4f (bins with data: %d)" % (ece, np.isfinite(bc).sum()))

    pred = probs.argmax(1)
    mean, lo, hi = bootstrap_ci(y, pred, accuracy_score, n=300)
    print("bootstrap acc=%.3f [%.3f, %.3f]" % (mean, lo, hi))

    pred_b = pred.copy()
    flip = rng.choice(N, 30, replace=False)
    pred_b[flip] = (pred_b[flip] + 1) % C.NUM_CLASSES
    mc = mcnemar(y, pred, pred_b)
    print("mcnemar b=%d c=%d stat=%.3f p=%.4f" % (mc["b"], mc["c"], mc["statistic"], mc["p_value"]))

    # tier-int / delta sanity
    assert _tier_int("E0", "abnormal", "high") == 1
    assert _tier_int("E1", None, None) is None
    d = internal_external_delta(
        {"accuracy": 0.9, "auroc": 0.95},
        {"herlev": {"accuracy": 0.7, "auroc": 0.8}},
    )
    print("delta herlev acc:", d["herlev"]["accuracy"]["delta"])
    print("OK")
