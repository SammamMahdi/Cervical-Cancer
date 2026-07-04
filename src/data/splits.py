"""U1: leakage-safe, cluster-grouped 80/10/10 train/val/test split + grouped CV folds.

Groups = source FOV/cluster id (no cluster shared across train, val, test). Stratified by class.
Built with sklearn StratifiedGroupKFold. Persists the split to results/splits/ as JSON (paths +
cluster ids) so every run is reproducible and the test stays sealed. Asserts ZERO cluster overlap.
"""
import os, json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import config as C
from data.clustering import load_cells, summarize

SPLIT_PATH = os.path.join(C.SPLITDIR, "sipakmed_split.json")

# The split JSON stores ABSOLUTE paths; if it was written under a drive letter that is no longer
# mounted (this project is reachable as both e:\ and f:\ at different times, and only one is live at
# once), those paths fail to open and loaders silently feed BLANK images. Reroot every stored path
# onto the CURRENT project drive at load time so every consumer (test loader, route_c, cam, ...) is safe.
_CUR_DRIVE = os.path.splitdrive(C.PROJECT)[0]


def _reroot(p):
    drive, rest = os.path.splitdrive(os.path.normpath(p))
    if drive and _CUR_DRIVE and drive.lower() != _CUR_DRIVE.lower():
        return _CUR_DRIVE + rest
    return p


def _reroot_split(split):
    for k in ("train", "val", "test"):
        if isinstance(split.get(k), list):
            split[k] = [_reroot(p) for p in split[k]]
    return split


def _assert_disjoint_clusters(cells, idx_sets):
    cluster_sets = [set(cells[i].cluster_id for i in idxs) for idxs in idx_sets.values()]
    names = list(idx_sets.keys())
    arr = list(cluster_sets)
    for a in range(len(arr)):
        for b in range(a + 1, len(arr)):
            overlap = arr[a] & arr[b]
            assert not overlap, f"CLUSTER LEAKAGE between {names[a]} and {names[b]}: {list(overlap)[:5]}"


def make_split(seed=C.SEED, fractions=C.SPLIT, force=False):
    """Create (or load) the group-aware 80/10/10 split. Returns dict with train/val/test index lists."""
    if os.path.exists(SPLIT_PATH) and not force:
        return load_split()

    cells = load_cells()
    y = np.array([c.class_idx for c in cells])
    groups = np.array([c.cluster_id for c in cells])
    n = len(cells)

    # Strategy: StratifiedGroupKFold into 10 folds -> 8 train / 1 val / 1 test (group-disjoint).
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = [test_idx for _, test_idx in skf.split(np.zeros(n), y, groups)]
    test_idx = folds[0]
    val_idx = folds[1]
    train_idx = np.concatenate(folds[2:])

    idx_sets = {"train": train_idx, "val": val_idx, "test": test_idx}
    _assert_disjoint_clusters(cells, idx_sets)

    split = {
        "seed": seed, "fractions": fractions, "grouping": "source FOV/cluster",
        "n_cells": n, "n_clusters": len(set(groups)),
        "classes": C.CLASSES,
        "train": [cells[i].path for i in train_idx],
        "val":   [cells[i].path for i in val_idx],
        "test":  [cells[i].path for i in test_idx],
        "train_clusters": sorted(set(groups[train_idx].tolist())),
        "val_clusters":   sorted(set(groups[val_idx].tolist())),
        "test_clusters":  sorted(set(groups[test_idx].tolist())),
    }
    os.makedirs(C.SPLITDIR, exist_ok=True)
    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)
    return split


def cv_folds(seed=C.SEED, n_splits=C.SPLIT["n_cv_folds"]):
    """Grouped CV folds over TRAIN+VAL (for development / bootstrap CIs); test stays sealed."""
    split = make_split(seed=seed)
    devpaths = split["train"] + split["val"]
    cells = load_cells()
    by_path = {os.path.normcase(c.path): c for c in cells}   # normcase: drive-letter/case-robust
    dev = [by_path[os.path.normcase(p)] for p in devpaths if os.path.normcase(p) in by_path]
    y = np.array([c.class_idx for c in dev]); groups = np.array([c.cluster_id for c in dev])
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = []
    for tr, va in skf.split(np.zeros(len(dev)), y, groups):
        # group-disjoint by construction; assert anyway
        assert not (set(groups[tr]) & set(groups[va])), "CV fold cluster leakage"
        out.append(([dev[i].path for i in tr], [dev[i].path for i in va]))
    return out


RANDOM_SPLIT_PATH = os.path.join(C.SPLITDIR, "sipakmed_random_split.json")


def make_random_split(seed=C.SEED, fractions=C.SPLIT, force=False):
    """A1 BASELINE (leaky): per-CELL stratified random 80/10/10 with NO grouping — same-cluster cells
    leak across train/val/test. Used only to quantify the leakage inflation vs the grouped split."""
    if os.path.exists(RANDOM_SPLIT_PATH) and not force:
        with open(RANDOM_SPLIT_PATH, "r", encoding="utf-8") as f:
            return _reroot_split(json.load(f))
    import random as _r
    cells = load_cells(); rng = _r.Random(seed)
    by_cls = {}
    for c in cells:
        by_cls.setdefault(c.class_idx, []).append(c.path)
    tr, va, te = [], [], []
    for _ci, paths in by_cls.items():
        paths = paths[:]; rng.shuffle(paths); n = len(paths)
        ntr = int(round(n * fractions["train"])); nva = int(round(n * fractions["val"]))
        tr += paths[:ntr]; va += paths[ntr:ntr + nva]; te += paths[ntr + nva:]
    split = {"seed": seed, "fractions": fractions, "grouping": "NONE (per-cell random — leaky A1 baseline)",
             "n_cells": len(cells), "classes": C.CLASSES, "train": tr, "val": va, "test": te}
    with open(RANDOM_SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)
    return split


def get_split(kind="grouped", seed=C.SEED):
    """kind: 'grouped' (leakage-safe, default) | 'random' (leaky A1 baseline)."""
    return make_split(seed=seed) if kind == "grouped" else make_random_split(seed=seed)


def load_split():
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        return _reroot_split(json.load(f))


def split_report(split=None):
    """Human/JSON summary used by EDA and the notebook."""
    split = split or load_split()
    cells = load_cells()
    by_path = {os.path.normcase(c.path): c for c in cells}   # normcase: drive-letter/case-robust (legacy bug)
    rep = {"n_cells": split["n_cells"], "n_clusters": split.get("n_clusters")}
    for part in ("train", "val", "test"):
        paths = split[part]
        from collections import Counter
        cells_part = [by_path[os.path.normcase(p)] for p in paths if os.path.normcase(p) in by_path]
        cls = Counter(c.class_name for c in cells_part)
        clusters = set(c.cluster_id for c in cells_part)
        rep[part] = {"cells": len(paths), "clusters": len(clusters), "per_class": dict(cls)}
    # zero-overlap proof (grouped split only; random A1 split has no *_clusters keys)
    tc, vc, sc = (set(split.get(f"{p}_clusters", [])) for p in ("train", "val", "test"))
    rep["cluster_overlap"] = {"train&val": len(tc & vc), "train&test": len(tc & sc), "val&test": len(vc & sc)}
    return rep


if __name__ == "__main__":
    s = make_split(force=True)
    print("clusters:", summarize(load_cells()))
    print(json.dumps(split_report(s), indent=2))
