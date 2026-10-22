"""Aggregate cell-level predictions to a slide-level abnormality score.

A reviewer objection to the first version: the cytology expert predicts P(abnormal CELL) at a 41.8% cell-level
prevalence, while the deployment prevalence used for re-anchoring is the fraction of WOMEN with abnormal
cytology. Those are different random variables. Here the held-out cells of each source slide are pooled into one
slide-level score, so the image expert predicts P(abnormal SLIDE) and the prior it is re-anchored from is the
fraction of abnormal slides, which is the same kind of quantity as the deployment prevalence.

Aggregators compared: mean, max, mean of the top-k cells, and noisy-OR.

Run: python paper_jbhi/scripts/slide_level.py   -> ../results/slide_level.json, slide_preds.npz
"""
import os, json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
ABN = (0, 1)   # dyskeratotic, koilocytotic


def slide_id(path):
    p = str(path).replace("\\", "/")
    cls = p.split("/sipakmed/")[1].split("/")[0]
    return cls + ":" + os.path.basename(p).split("_")[0]


def aggregate(p, how, k=3):
    p = np.sort(p)[::-1]
    if how == "mean":
        return p.mean()
    if how == "max":
        return p.max()
    if how.startswith("top"):
        return p[:min(k, len(p))].mean()
    if how == "noisy_or":
        return 1 - np.prod(1 - np.clip(p, 0, 0.999))
    raise ValueError(how)


def main():
    d = np.load(os.path.join(RES, "grouped_preds.npz"), allow_pickle=True)
    paths, y, P = d["test_paths"], d["y_test"], d["P_student"]
    p_cell = P[:, ABN].sum(1)
    sid = np.array([slide_id(p) for p in paths])
    slides = sorted(set(sid))
    y_slide = np.array([int(y[sid == s][0] in ABN) for s in slides])
    n_cells = np.array([(sid == s).sum() for s in slides])
    print(f"{len(slides)} held-out slides ({y_slide.sum()} abnormal), "
          f"{n_cells.min()}-{n_cells.max()} cells each (median {int(np.median(n_cells))})")

    out = dict(n_slides=len(slides), n_abnormal=int(y_slide.sum()), n_cells_total=int(len(y)),
               cells_per_slide=dict(min=int(n_cells.min()), max=int(n_cells.max()), median=float(np.median(n_cells))),
               cell_level=dict(auroc=float(roc_auc_score(np.isin(y, ABN).astype(int), p_cell)),
                               prior=float(np.isin(y, ABN).mean())),
               slide_prior=float(y_slide.mean()), aggregators={})
    best, best_auc = None, -1
    for how in ("mean", "max", "top3", "noisy_or"):
        s = np.array([aggregate(p_cell[sid == sl], how) for sl in slides])
        auc = float(roc_auc_score(y_slide, s)); ap = float(average_precision_score(y_slide, s))
        out["aggregators"][how] = dict(auroc=auc, ap=ap, mean_score=float(s.mean()))
        print(f"  {how:9s} slide AUROC {auc:.4f}  AP {ap:.4f}  mean score {s.mean():.3f}")
        if auc > best_auc:
            best, best_auc = how, auc
    out["chosen"] = best
    scores = np.array([aggregate(p_cell[sid == sl], best) for sl in slides])
    np.savez_compressed(os.path.join(RES, "slide_preds.npz"), y=y_slide, p=scores,
                        slides=np.array(slides), n_cells=n_cells, aggregator=best)
    print(f"chosen aggregator: {best} (slide AUROC {best_auc:.4f}); slide-level abnormal prior "
          f"{out['slide_prior']:.3f} vs cell-level {out['cell_level']['prior']:.3f}")
    json.dump(out, open(os.path.join(RES, "slide_level.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
