"""Evaluation protocol: in-distribution (5-class) metrics + cross-dataset (binary) eval.

In-distribution (SIPaKMeD val): accuracy, balanced accuracy, macro-F1, Cohen's kappa, per-class
precision/recall/F1, confusion matrix, one-vs-rest ROC-AUC, ECE + reliability diagram.
Cross-dataset: collapse the 5-class prediction to binary Normal/Abnormal and score external sets.
All figures use the house SciencePlots style (plotting.py).
"""
import os, json, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (classification_report, confusion_matrix, balanced_accuracy_score,
                             f1_score, cohen_kappa_score, roc_auc_score, roc_curve, accuracy_score)

import config as C
import models as M
import plotting as P
import runlog
from dataset import make_loaders, get_transforms, IMG_EXT

# SIPaKMeD 5-class -> binary collapse  (indices: 0 Dysk,1 Koilo,2 Metap,3 Parab,4 Superf-Interm)
ABNORMAL_IDX = [C.CLASSES.index("Koilocytotic"), C.CLASSES.index("Dyskeratotic")]   # [1, 0]


def load_model(which="student"):
    cfg = C.TEACHER if which == "teacher" else C.STUDENT
    m = M.EffNetClassifier(cfg["backbone"], C.NUM_CLASSES, cfg["hidden"], cfg["dropout"], pretrained=False).to(C.DEVICE)
    ckpt = "teacher_b4.pt" if which == "teacher" else "student_b0.pt"
    m.load_state_dict(torch.load(os.path.join(C.CKPTDIR, ckpt), map_location=C.DEVICE))
    m.eval()
    return m


@torch.no_grad()
def predict(model, loader):
    ys, ps = [], []
    for x, y in loader:
        x = x.to(C.DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(C.DEVICE == "cuda")):
            logits = model(x)
        ps.append(torch.softmax(logits.float(), 1).cpu().numpy()); ys.append(np.asarray(y))
    return np.concatenate(ys), np.concatenate(ps)


def expected_calibration_error(probs, labels, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0; bc, ba = [], []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            ece += abs(correct[m].mean() - conf[m].mean()) * m.mean()
            bc.append(float(conf[m].mean())); ba.append(float(correct[m].mean()))
    return float(ece), np.array(bc), np.array(ba)


def indist_eval(which="student"):
    """Full 5-class metrics + figures on the SIPaKMeD validation split."""
    _, val_ld, _ = make_loaders()
    model = load_model(which)
    y, probs = predict(model, val_ld)
    pred = probs.argmax(1)
    onehot = np.eye(C.NUM_CLASSES)[y]
    rep = classification_report(y, pred, target_names=C.CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred)
    ece, bc, ba = expected_calibration_error(probs, y)
    metrics = dict(
        model=which, accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average="macro")),
        cohen_kappa=float(cohen_kappa_score(y, pred)),
        roc_auc_macro_ovr=float(roc_auc_score(onehot, probs, multi_class="ovr", average="macro")),
        ece=ece, per_class=rep, confusion_matrix=cm.tolist())

    # raw predictions in JSON -> any in-dist figure can be rebuilt without re-running the model
    runlog.dump_json(os.path.join(C.METRICDIR, f"predictions_{which}.json"),
                     dict(model=which, classes=C.CLASSES, y_true=y.tolist(), probs=probs.tolist()))

    # figures (house style)
    _indist_figures(which, y, probs, cm, ece, bc, ba)

    with open(os.path.join(C.METRICDIR, f"indist_metrics_{which}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[in-dist {which}] acc={metrics['accuracy']:.4f} bal_acc={metrics['balanced_accuracy']:.4f} "
          f"macroF1={metrics['macro_f1']:.4f} kappa={metrics['cohen_kappa']:.4f} "
          f"AUROC={metrics['roc_auc_macro_ovr']:.4f} ECE={ece:.4f}")
    return metrics


def _indist_figures(which, y, probs, cm, ece, bc, ba):
    """Draw confusion matrix + one-vs-rest ROC + reliability from arrays (house style)."""
    y = np.asarray(y); probs = np.asarray(probs)
    onehot = np.eye(C.NUM_CLASSES)[y]
    P.confusion_matrix(cm, C.CLASSES, os.path.join(C.FIGDIR, f"cm_{which}"),
                       title=f"Confusion matrix — {which} (SIPaKMeD val)")
    curves = {}
    for ci, cls in enumerate(C.CLASSES):
        fpr, tpr, _ = roc_curve(onehot[:, ci], probs[:, ci])
        curves[cls] = (fpr, tpr, roc_auc_score(onehot[:, ci], probs[:, ci]))
    P.roc_curves(curves, os.path.join(C.FIGDIR, f"roc_{which}"), title=f"ROC (one-vs-rest) — {which}")
    if len(bc):
        P.reliability(bc, ba, os.path.join(C.FIGDIR, f"reliability_{which}"),
                      title=f"Reliability — {which} (ECE={ece:.3f})", label=which)


def figures_from_json(which="student"):
    """Rebuild all in-distribution figures from results/metrics/predictions_<which>.json (no model needed)."""
    d = runlog.load_json(os.path.join(C.METRICDIR, f"predictions_{which}.json"))
    if d is None:
        print(f"no predictions_{which}.json found"); return
    y = np.asarray(d["y_true"]); probs = np.asarray(d["probs"]); pred = probs.argmax(1)
    from sklearn.metrics import confusion_matrix as _cm
    cm = _cm(y, pred)
    ece, bc, ba = expected_calibration_error(probs, y)
    _indist_figures(which, y, probs, cm, ece, bc, ba)
    print(f"Rebuilt in-dist figures for {which} from JSON.")


# ---------------- cross-dataset (binary Normal/Abnormal) ----------------
def _is_mask(fn):
    """Herlev segmentation masks are companion `*-d.bmp` files — exclude them from classification."""
    return fn.lower().endswith("-d.bmp")


class FolderBinary(Dataset):
    """Walk all folders under root; any folder whose name maps (via name_map -> 0/1) contributes its
    images. Robust to arbitrary nesting in different dataset mirrors. Segmentation masks skipped."""
    def __init__(self, root, name_map, transform):
        self.transform = transform; self.items = []
        for dirpath, _, filenames in os.walk(root):
            lab = name_map(os.path.basename(dirpath))
            if lab is None:
                continue
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                if fn.lower().endswith(IMG_EXT) and not _is_mask(fn) and os.path.getsize(fp) > 0:
                    self.items.append((fp, lab))

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, lab = self.items[i]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (8, 8))   # corrupt/unreadable file -> placeholder (keeps eval running)
        return self.transform(img), lab


def cross_dataset_eval(name, root=None, name_map=None, batch_size=C.BATCH_SIZE, dataset=None):
    """Evaluate the SIPaKMeD-trained STUDENT on an external set, collapsed to binary (0=Normal,1=Abnormal).
    Pass (root, name_map) for a FOLDER dataset, OR a prebuilt `dataset` (e.g. HiCervix CSV-labeled)."""
    if dataset is None:
        if not os.path.isdir(root):
            print(f"[cross {name}] SKIP — not found at {root}"); return None
        dataset = FolderBinary(root, name_map, get_transforms(C.IMG_SIZE, train=False))
    ds = dataset
    if len(ds) == 0:
        print(f"[cross {name}] SKIP — no mapped images"); return None
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=C.NUM_WORKERS, pin_memory=False)
    model = load_model("student")
    y, probs = predict(model, ld)
    p_abn = probs[:, ABNORMAL_IDX].sum(1)           # model P(abnormal)
    pred = (p_abn >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    res = dict(dataset=name, n=int(len(ds)),
               accuracy=float(accuracy_score(y, pred)),
               balanced_accuracy=float(balanced_accuracy_score(y, pred)),
               sensitivity=float(tp / (tp + fn) if (tp + fn) else 0),
               specificity=float(tn / (tn + fp) if (tn + fp) else 0),
               auroc=float(roc_auc_score(y, p_abn)) if len(set(y)) > 1 else None,
               confusion_matrix=cm.tolist())
    P.confusion_matrix(cm, ["Normal", "Abnormal"], os.path.join(C.FIGDIR, f"cm_cross_{name}"),
                       title=f"Cross-dataset {name} — binary (student)")
    runlog.dump_json(os.path.join(C.METRICDIR, f"predictions_cross_{name}.json"),
                     dict(dataset=name, y_true=y.tolist(), p_abnormal=p_abn.tolist()))
    with open(os.path.join(C.METRICDIR, f"cross_{name}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[cross {name}] n={res['n']} acc={res['accuracy']:.3f} bal_acc={res['balanced_accuracy']:.3f} "
          f"sens={res['sensitivity']:.3f} spec={res['specificity']:.3f} AUROC={res['auroc']}")
    return res


# binary name maps per external dataset (subfolder name -> 'normal'/'abnormal'/None)
def herlev_map(sub):
    s = sub.lower()
    normal = ("superficiel", "superficial", "intermediate", "columnar", "normal")
    abnormal = ("dysplasia", "dysplastic", "carcinoma", "abnormal", "severe", "moderate", "mild", "light")
    if any(k in s for k in normal): return 0
    if any(k in s for k in abnormal): return 1
    return None

def apacc_map(sub):
    s = sub.lower()
    # NOTE: test "unhealthy" BEFORE "healthy" — "unhealthy" contains the substring "healthy".
    if "unhealthy" in s or "abnormal" in s: return 1
    if "healthy" in s or "normal" in s: return 0
    return None   # rubbish / rubbish_part2 / bothcells dropped


def hicervix_map(label):
    """HiCervix hierarchical-Bethesda label -> binary (0=Normal, 1=Abnormal, None=exclude glandular)."""
    s = str(label).lower().strip()
    if any(k in s for k in ("nilm", "negative", "normal", "inflammat", "reactive", "benign")): return 0
    if any(k in s for k in ("glandular", "endocervical", "endometrial", "agc", "adenocarc", "ais")): return None
    if any(k in s for k in ("lsil", "hsil", "asc-h", "asch", "asc-us", "ascus", "scc",
                            "carcinoma", "squamous", "atypical", "asc", "dysplas")): return 1
    return None


class HiCervixBinary(Dataset):
    """HiCervix test split (CSV-labeled): images at <hicervix>/test/test/<image_name>, labels from
    test.csv (hierarchical Bethesda). Collapsed to binary via hicervix_map(level_1 or class_name)."""
    def __init__(self, transform, csv_path=None, img_root=None):
        import csv
        self.transform = transform; self.items = []; self.excluded = 0
        hdir = os.path.join(C.DATA_RAW, "hicervix")
        csv_path = csv_path or os.path.join(hdir, "test.csv")
        img_root = img_root or os.path.join(hdir, "test", "test")
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row.get("level_1") or "").strip() or (row.get("class_name") or "").strip()
                lab = hicervix_map(key)
                name = (row.get("image_name") or "").strip()
                path = os.path.join(img_root, name)
                if lab is None or not name or not os.path.isfile(path) or os.path.getsize(path) == 0:
                    self.excluded += 1; continue
                self.items.append((path, lab))

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, lab = self.items[i]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (8, 8))
        return self.transform(img), lab


def hicervix_dataset():
    """Build the HiCervix binary eval dataset (for cross_dataset_eval(..., dataset=...))."""
    return HiCervixBinary(get_transforms(C.IMG_SIZE, train=False))


if __name__ == "__main__":
    indist_eval("teacher")
    indist_eval("student")
