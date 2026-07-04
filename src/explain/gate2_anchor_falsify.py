"""Gate 2 — zero-training falsification of NucleusAnchor as a scale-shortcut fix (CPU only).

Reproduces the diagnosed scale shortcut on SIPaKMeD (corr(cytoplasm_area, crop_pixel_area) ~ +0.989),
then asks whether NUCLEUS-anchoring the input removes it. The anchoring scale per cell is
    s_i = target_frac * frame_i / nucleus_equiv_diameter_i           (crop pixel frame)
and the apparent cytoplasm area after anchoring is  cytoplasm_area_i * s_i^2.

GO  if |corr(anchored_apparent_cytoplasm, crop_pixel_area)| drops substantially below ~0.989.
NO-GO if it stays high (then the confound is a within-dataset tautology that input anchoring cannot fix).
"""
import json
import numpy as np
from PIL import Image

import config as C
import explain.features as feat
from data.anchor import NucleusAnchor


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def run(target_frac=0.35):
    d = feat.load_features(split_kind="grouped", standardize=False)
    X, names, paths = d["X"], d["names"], d["paths"]
    i_cyt = names.index("cytoplasm_area")
    i_ned = names.index("nucleus_equivdiameter")
    i_nca = names.index("nucleus_area")
    i_nc = names.index("nc_ratio")

    cyt_area = X[:, i_cyt].astype(float)
    nuc_equiv = X[:, i_ned].astype(float)
    nuc_area = X[:, i_nca].astype(float)
    nc_ratio = X[:, i_nc].astype(float)
    n = len(paths)

    W = np.empty(n); H = np.empty(n)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            W[i], H[i] = im.size
    crop_area = W * H
    frame = np.maximum(W, H)

    anc = NucleusAnchor(target_frac=target_frac)
    s = np.array([anc.scale_factor(nuc_equiv[i], frame[i]) for i in range(n)])
    anchored_cyt = cyt_area * s ** 2
    anchored_cyt_frac = anchored_cyt / frame ** 2   # apparent size as fraction of fixed anchored frame

    out = dict(
        n=int(n), target_frac=target_frac,
        corr_cyt_crop=round(corr(cyt_area, crop_area), 4),
        corr_anchored_crop=round(corr(anchored_cyt, crop_area), 4),
        corr_anchored_frac_crop=round(corr(anchored_cyt_frac, crop_area), 4),
        corr_nc_crop=round(corr(nc_ratio, crop_area), 4),
        corr_nucarea_crop=round(corr(nuc_area, crop_area), 4),
        median_anchor_scale=round(float(np.nanmedian(s)), 4),
    )
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
