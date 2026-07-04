"""Extensive EDA on SIPaKMeD (class balance, cluster-vs-cropped, image-size stats, sample montage).

Outputs SciencePlots-styled figures to results/figures/ and a JSON + Markdown summary to
results/metrics/. Run after the SIPaKMeD download completes.
"""
import os, glob, json, random
from collections import Counter
import numpy as np
from PIL import Image

import config as C
import dataset as D
import plotting as P


def analyze(sample_per_class=80):
    cdirs = D.find_class_dirs()
    total, cropped, cluster = {}, {}, {}
    dims = {}            # class -> list of (w,h)
    montage = []         # one sample image (RGB 0..1) per class
    montage_titles = []
    rng = random.Random(C.SEED)

    for idx, rc in enumerate(C.RAW_CLASSES):
        cls = C.CLASSES[idx]
        files = [p for p in glob.glob(os.path.join(cdirs[rc], "**", "*"), recursive=True)
                 if p.lower().endswith(D.IMG_EXT)]
        total[cls] = len(files)
        cropped[cls] = sum(1 for p in files if "crop" in p.lower())
        cluster[cls] = total[cls] - cropped[cls]
        samp = rng.sample(files, min(sample_per_class, len(files)))
        ds = []
        for p in samp:
            try:
                with Image.open(p) as im:
                    ds.append(im.size)   # (w,h)
            except Exception:
                pass
        dims[cls] = ds
        # one montage image (prefer a cropped cell)
        pick = next((p for p in files if "crop" in p.lower()), files[0])
        arr = np.asarray(Image.open(pick).convert("RGB").resize((200, 200))) / 255.0
        montage.append(arr); montage_titles.append(P.pretty(cls))

    grand = sum(total.values())
    alldims = [wh for ds in dims.values() for wh in ds]
    ws = [w for w, h in alldims]; hs = [h for w, h in alldims]

    # ---- figures (house style) ----
    P.class_balance(total, os.path.join(C.FIGDIR, "eda_class_balance"),
                    ylabel="images", title="SIPaKMeD class balance")
    P.image_panels(montage, montage_titles, os.path.join(C.FIGDIR, "eda_sample_cells"),
                   suptitle="SIPaKMeD sample cells (one per class)")

    summary = dict(
        n_classes=C.NUM_CLASSES, total_images=grand,
        per_class_total=total, per_class_cropped=cropped, per_class_cluster=cluster,
        imbalance_ratio=round(max(total.values()) / max(1, min(total.values())), 3),
        image_width=dict(min=int(min(ws)), max=int(max(ws)), mean=round(float(np.mean(ws)), 1)) if ws else {},
        image_height=dict(min=int(min(hs)), max=int(max(hs)), mean=round(float(np.mean(hs)), 1)) if hs else {},
    )
    with open(os.path.join(C.METRICDIR, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- markdown summary ----
    md = ["# SIPaKMeD — EDA summary\n",
          f"- **Total images:** {grand} across {C.NUM_CLASSES} classes",
          f"- **Class imbalance ratio (max/min):** {summary['imbalance_ratio']}",
          f"- **Image size:** W {summary['image_width']}, H {summary['image_height']} (px)\n",
          "| Class | total | cropped (cells) | cluster (FOV) |", "|---|--:|--:|--:|"]
    for cls in C.CLASSES:
        md.append(f"| {cls} | {total[cls]} | {cropped[cls]} | {cluster[cls]} |")
    md += ["", "Figures: `eda_class_balance.(pdf|png)`, `eda_sample_cells.(pdf|png)` in results/figures/."]
    with open(os.path.join(C.METRICDIR, "eda_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(json.dumps(summary, indent=2))
    print("EDA figures + summary written to results/")
    return summary


if __name__ == "__main__":
    analyze()
