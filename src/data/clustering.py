"""U1 prerequisite: rebuild the SIPaKMeD cell -> source-cluster (FOV) map.

The SIPaKMeD download ships, per class folder:
  im_<Class>/im_<Class>/<FOV>.bmp                 -> 966 field-of-view (cluster) images
  im_<Class>/im_<Class>/CROPPED/<FOV>_<cell>.bmp  -> 4,049 cropped single cells

The cropped filename encodes its source FOV (<FOV>_<cell>), so each of the 4,049 cells maps to one
of 966 clusters with NO need to parse the .dat boundary tables. The cluster id is the leakage-safe
grouping unit for the 80/10/10 split (no FOV shared across train/val/test).
"""
import os, glob
from dataclasses import dataclass
import config as C

IMG_EXT = (".bmp", ".png", ".jpg", ".jpeg")


@dataclass
class Cell:
    path: str
    class_name: str
    class_idx: int
    fov: str          # FOV number within the class, e.g. "001"
    cluster_id: str   # globally-unique group key, e.g. "Dyskeratotic/001"


def _cropped_dir(class_name):
    """Locate the CROPPED folder for a class (robust to the nested im_X/im_X layout)."""
    base = C.RAW_CLASS_DIRS[class_name]
    hits = [d for d in glob.glob(os.path.join(C.SIPAKMED_DIR, "**", base, "**", "CROPPED"), recursive=True)
            if os.path.isdir(d)]
    if not hits:
        # fall back: any CROPPED dir whose path contains the class folder name
        hits = [d for d in glob.glob(os.path.join(C.SIPAKMED_DIR, "**", "CROPPED"), recursive=True)
                if os.path.isdir(d) and base in d]
    if not hits:
        raise FileNotFoundError(f"CROPPED folder for {class_name} not found under {C.SIPAKMED_DIR}")
    return sorted(hits, key=len)[0]


def load_cells():
    """Return list[Cell] for all 4,049 cropped SIPaKMeD cells with their cluster (FOV) group."""
    cells = []
    for ci, cname in enumerate(C.CLASSES):
        d = _cropped_dir(cname)
        files = [p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith(IMG_EXT)]
        for p in files:
            stem = os.path.splitext(os.path.basename(p))[0]
            fov = stem.split("_")[0]                      # "001_03" -> "001"
            cells.append(Cell(path=p, class_name=cname, class_idx=ci,
                              fov=fov, cluster_id=f"{cname}/{fov}"))
    if not cells:
        raise RuntimeError("no cropped SIPaKMeD cells found — check data/raw/sipakmed")
    return cells


def summarize(cells):
    """Counts used by the EDA / split assertions (DATA_PLAN §6.1)."""
    from collections import Counter, defaultdict
    per_class = Counter(c.class_name for c in cells)
    clusters = set(c.cluster_id for c in cells)
    clusters_per_class = Counter(cid.split("/")[0] for cid in clusters)
    cells_per_cluster = Counter(c.cluster_id for c in cells)
    sizes = list(cells_per_cluster.values())
    return dict(
        n_cells=len(cells), n_clusters=len(clusters),
        cells_per_class=dict(per_class), clusters_per_class=dict(clusters_per_class),
        cells_per_cluster_min=min(sizes), cells_per_cluster_max=max(sizes),
        cells_per_cluster_mean=round(sum(sizes) / len(sizes), 2),
    )


if __name__ == "__main__":
    import json
    cells = load_cells()
    print(json.dumps(summarize(cells), indent=2))
