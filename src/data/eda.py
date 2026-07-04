"""All-dataset EDA for the new pipeline (DATA_PLAN.md §6).

Profiles EVERY present dataset (SIPaKMeD train source + Herlev/APACC/HiCervix/CRIC external) with a
single, consistent enumeration that reuses the eval-path label maps (data/labelmap.py), so EDA counts
match what the model is actually evaluated on. For each dataset we record, and PERSIST as JSON+CSV+figure:

  * native-class counts (+ class-balance bar)            -> results/figures/eda/<ds>_classes.{pdf,png,csv}
  * E0 mapping coverage normal/abnormal/EXCLUDED + frac  -> in <ds>.json and eda_overview.csv
  * image size stats (w/h min/max/mean/median, formats)  -> sampled, in <ds>.json
  * corrupt / zero-byte file count (legacy-issue guard)  -> in <ds>.json
  * sample montage (a few cells per native class)        -> results/figures/eda/<ds>_montage.{pdf,png}

Plus a combined results/metrics/eda/eda_all.json and results/metrics/eda/eda_overview.csv.
Run: `python -m data.eda`  (or `run.py eda`).
"""
import os, glob, csv as _csv
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

import config as C
import plotting as P
import runlog
import data.labelmap as LM
from data.clustering import load_cells, summarize

EDA_METRICDIR = os.path.join(C.METRICDIR, "eda")
EDA_FIGDIR = os.path.join(C.FIGDIR, "eda")
for _d in (EDA_METRICDIR, EDA_FIGDIR):
    os.makedirs(_d, exist_ok=True)

IMG_EXT = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _is_img(f):
    return f.lower().endswith(IMG_EXT)


def _safe_open(path):
    """Robust load guarding the legacy failures (0-byte, truncated, unreadable). Returns PIL or None."""
    try:
        if os.path.getsize(path) == 0:
            return None
        im = Image.open(path); im.load()
        return im.convert("RGB")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  Per-dataset enumeration: -> list of (path, native_class, e0_label_or_excluded)
#  e0 in {"normal","abnormal", None(excluded)}.  Reuses labelmap (eval-consistent).
# --------------------------------------------------------------------------- #
def _enum_sipakmed():
    out = []
    for c in load_cells():
        out.append((c.path, c.class_name, LM.pred_to_e0(c.class_idx)))
    return out


def _enum_folder(root, mapper):
    out = []
    if not os.path.isdir(root):
        return out
    for dirpath, _dirs, files in os.walk(root):
        imgs = [os.path.join(dirpath, f) for f in files if _is_img(f) and not f.lower().endswith("-d.bmp")]
        if not imgs:
            continue
        base = os.path.basename(dirpath)
        try:
            e0, _e1 = mapper(base)
        except Exception:
            e0 = None
        for p in imgs:
            out.append((p, base, e0))
    return out


def _enum_hicervix():
    out = []
    csv_path = os.path.join(C.HICERVIX_DIR, "test.csv")
    img_root = os.path.join(C.HICERVIX_DIR, "test", "test")
    if not os.path.isfile(csv_path):
        return out
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            lvl1 = (row.get("level_1") or "").strip()
            key = lvl1 or (row.get("class_name") or "").strip()
            e0, _e1 = LM.hicervix_map(key)
            name = (row.get("image_name") or "").strip()
            out.append((os.path.join(img_root, name), key or "unknown", e0))
    return out


def _enum_cric():
    from data.external_tier2 import _read_cric_rows, CRIC_ROOT
    out = []
    csv_path = os.path.join(CRIC_ROOT, "classes", "classifications.csv")
    if not os.path.isfile(csv_path):
        return out
    for fname, _x, _y, native in _read_cric_rows(csv_path):
        e0, _e1 = LM.hicervix_map(native)
        out.append((os.path.join(CRIC_ROOT, fname), native or "unknown", e0))
    return out


ENUMERATORS = {
    "sipakmed": _enum_sipakmed,
    "herlev":   lambda: _enum_folder(C.HERLEV_DIR, LM.herlev_map),
    "apacc":    lambda: _enum_folder(C.APACC_DIR, LM.apacc_map),
    "hicervix": _enum_hicervix,
    "cric":     _enum_cric,
}


def _size_stats(paths, sample=400, seed=C.SEED):
    """Sample up to `sample` images; return w/h stats, format counts, and corrupt count."""
    import random
    rng = random.Random(seed)
    uniq = list(dict.fromkeys(paths))
    pick = uniq if len(uniq) <= sample else rng.sample(uniq, sample)
    ws, hs, fmts, corrupt = [], [], Counter(), 0
    for p in pick:
        im = _safe_open(p)
        if im is None:
            corrupt += 1; continue
        ws.append(im.width); hs.append(im.height)
        fmts[os.path.splitext(p)[1].lower()] += 1
    def _st(a):
        a = np.array(a, float)
        return None if not len(a) else dict(min=int(a.min()), max=int(a.max()),
                                            mean=round(float(a.mean()), 1), median=int(np.median(a)))
    return dict(sampled=len(pick), corrupt_or_zero=corrupt, width=_st(ws), height=_st(hs),
                formats=dict(fmts))


def _montage(items, name, per_class=2, seed=C.SEED):
    """One small montage: up to `per_class` sample cells per native class."""
    import random
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for p, cls, _e0 in items:
        by_cls[cls].append(p)
    imgs, titles = [], []
    for cls in sorted(by_cls):
        for p in rng.sample(by_cls[cls], min(per_class, len(by_cls[cls]))):
            im = _safe_open(p)
            if im is None:
                continue
            imgs.append(np.asarray(im.resize((128, 128))) / 255.0)
            titles.append(cls[:14])
    if imgs:
        P.montage(imgs, titles, os.path.join(EDA_FIGDIR, f"{name}_montage"),
                  suptitle=f"{name} — sample cells", ncol=min(len(imgs), 6))


def profile_dataset(name):
    items = ENUMERATORS[name]()
    n = len(items)
    native = Counter(cls for _p, cls, _e0 in items)
    e0c = Counter(("excluded" if e0 is None else e0) for _p, _cls, e0 in items)
    excl = e0c.get("excluded", 0)
    rec = dict(
        dataset=name, n_items=n,
        native_classes=dict(native.most_common()),
        e0_coverage=dict(normal=e0c.get("normal", 0), abnormal=e0c.get("abnormal", 0),
                         excluded=excl, excluded_fraction=round(excl / n, 4) if n else 0.0),
        size_stats=_size_stats([p for p, _c, _e in items]),
    )
    # per-class balance figure (+ csv via plotting)
    if native:
        P.class_balance(dict(native.most_common()), os.path.join(EDA_FIGDIR, f"{name}_classes"),
                        ylabel="images", title=f"{name} — native class balance")
    # E0 coverage figure
    P.class_balance({k: e0c.get(k, 0) for k in ("normal", "abnormal", "excluded")},
                    os.path.join(EDA_FIGDIR, f"{name}_e0coverage"), ylabel="images",
                    title=f"{name} — E0 mapping coverage")
    _montage(items, name)
    runlog.dump_json(os.path.join(EDA_METRICDIR, f"{name}.json"), rec)
    # per-class csv
    P.write_csv(os.path.join(EDA_METRICDIR, f"{name}_classes"),
                ["native_class", "count", "e0"],
                [[cls, cnt, _first_e0(items, cls)] for cls, cnt in native.most_common()])
    return rec


def _first_e0(items, cls):
    for _p, c, e0 in items:
        if c == cls:
            return "excluded" if e0 is None else e0
    return ""


def run_all(datasets=None):
    datasets = datasets or list(ENUMERATORS)
    all_rec = {}
    # SIPaKMeD also gets the cluster/split summary (training source)
    for name in datasets:
        try:
            all_rec[name] = profile_dataset(name)
            r = all_rec[name]
            print(f"  {name:9s} n={r['n_items']:6d}  excl_frac={r['e0_coverage']['excluded_fraction']:.3f}  "
                  f"corrupt={r['size_stats']['corrupt_or_zero']}  classes={len(r['native_classes'])}")
        except Exception as e:
            print(f"  {name:9s} FAILED: {e}")
            all_rec[name] = dict(dataset=name, error=str(e))
    # SIPaKMeD clustering detail
    try:
        all_rec["sipakmed_clusters"] = summarize(load_cells())
    except Exception as e:
        print("  sipakmed clusters failed:", e)
    runlog.dump_json(os.path.join(EDA_METRICDIR, "eda_all.json"), all_rec)
    # combined overview CSV
    rows = []
    for name in datasets:
        r = all_rec.get(name, {})
        if "error" in r:
            continue
        ss = r.get("size_stats", {})
        rows.append([name, r["n_items"], len(r["native_classes"]),
                     r["e0_coverage"]["normal"], r["e0_coverage"]["abnormal"],
                     r["e0_coverage"]["excluded"], r["e0_coverage"]["excluded_fraction"],
                     ss.get("corrupt_or_zero", 0),
                     (ss.get("width") or {}).get("mean"), (ss.get("height") or {}).get("mean")])
    P.write_csv(os.path.join(EDA_METRICDIR, "eda_overview"),
                ["dataset", "n_items", "n_native_classes", "e0_normal", "e0_abnormal",
                 "e0_excluded", "excluded_fraction", "corrupt_or_zero", "mean_width", "mean_height"], rows)
    return all_rec


if __name__ == "__main__":
    print("All-dataset EDA ->", EDA_METRICDIR)
    run_all()
    print("Done. JSON+CSV in", EDA_METRICDIR, "| figures in", EDA_FIGDIR)
