"""Ground-truth Grad-CAM faithfulness against SIPaKMeD nucleus contours (A).

Each CROPPED cell ships a `<base>_nuc.dat` polygon (nucleus boundary, `x,y` per line) in the crop's
pixel frame. We rasterise it to a mask, push it through the SAME aspect-preserving pad/resize the
model input uses (so the mask aligns with the CAM grid), and measure whether the heatmap actually
lands on the nucleus:

  nucleus_hit_rate    fraction of cells whose CAM PEAK pixel is inside the nucleus (pointing game).
  energy_in_nucleus   mean fraction of total CAM energy inside the nucleus mask.
  nucleus_area_frac   mean nucleus area as a fraction of the frame (the uninformed baseline).
  concentration_ratio energy_in_nucleus / nucleus_area_frac  (>1 = CAM concentrates on the nucleus
                      beyond its size = genuine nuclear localization; ~1 = uninformative/diffuse).

This is a clinical, ground-truth check — stronger than the deletion proxy. Writes
results/metrics/nucleus_faithfulness_<which>.json + two bar figures.
"""
import os, glob, random
import numpy as np
from PIL import Image, ImageDraw
import torch

import config as C


def _aspect_pad_mask(mask_pil, size):
    """Replicate data.dataset.AspectPad on a binary mask (resize longest side, center-pad to square)."""
    w, h = mask_pil.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    m = mask_pil.resize((nw, nh), Image.NEAREST)
    out = Image.new("L", (size, size), 0)
    out.paste(m, ((size - nw) // 2, (size - nh) // 2))
    return np.asarray(out, bool)


def _nucleus_mask(crop_path, size):
    nuc = crop_path.rsplit(".", 1)[0] + "_nuc.dat"
    if not os.path.exists(nuc):
        return None
    pts = []
    with open(nuc) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                x, y = line.split(",")
                pts.append((float(x), float(y)))
            except ValueError:
                pass
    if len(pts) < 3:
        return None
    with Image.open(crop_path) as im:
        w, h = im.size
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).polygon(pts, fill=1)
    return _aspect_pad_mask(m, size)


def run_nucleus_faithfulness(which="student", img_size=380, n_per_class=20,
                             methods=("gradcam", "layercam", "hirescam"), fine_layer=False):
    import explain.cam as cam
    import ckpt as CK
    from models.backbones import build
    cfg = C.TEACHER if which.startswith("teacher") else C.STUDENT
    model = build(cfg["backbone"], C.NUM_CLASSES, cfg["hidden"], cfg["dropout"], pretrained=False).to(C.DEVICE)
    CK.load_into(model, which, map_location=C.DEVICE)   # latest checkpoint (root or run_*/ snapshot)
    model.eval()
    target = cam._finer_layer(model) if fine_layer else None

    rng = random.Random(C.SEED)
    cells = {}
    for ci, cname in enumerate(C.CLASSES):
        pool = [p for p in glob.glob(os.path.join(C.SIPAKMED_DIR, f"im_{cname}", "**", "CROPPED", "*.bmp"),
                                     recursive=True)
                if os.path.exists(p.rsplit(".", 1)[0] + "_nuc.dat")]
        rng.shuffle(pool)
        cells[ci] = pool[:n_per_class]

    results = {}
    for method in methods:
        cam_obj = cam.CAM(model, target_layer=target, method=method)
        hit = energy = ratio = area = 0.0; n = 0
        try:
            for ci, paths in cells.items():
                for p in paths:
                    mask = _nucleus_mask(p, img_size)
                    if mask is None or mask.sum() == 0:
                        continue
                    _disp, x = cam._load_display_and_tensor(p, img_size=img_size)
                    x = x.to(C.DEVICE)
                    with torch.no_grad():
                        pred = int(torch.softmax(model(x.float()), 1)[0].argmax())
                    cmap = cam_obj(x, class_idx=pred)               # (H,W) in [0,1], input resolution
                    tot = float(cmap.sum()) + 1e-8
                    e_in = float(cmap[mask].sum()) / tot
                    a_frac = float(mask.mean())
                    peak = np.unravel_index(int(cmap.argmax()), cmap.shape)
                    hit += bool(mask[peak]); energy += e_in; area += a_frac
                    ratio += e_in / (a_frac + 1e-8); n += 1
        finally:
            cam_obj.remove()                       # always remove hooks
        if n == 0:
            print(f"  {method:9s} no cells with a nucleus contour; skipped"); continue
        results[method] = dict(n=n, nucleus_hit_rate=round(hit / n, 4), energy_in_nucleus=round(energy / n, 4),
                               nucleus_area_frac=round(area / n, 4), concentration_ratio=round(ratio / n, 4))
        print(f"  {method:9s} n={n:3d}  hit={results[method]['nucleus_hit_rate']:.3f}  "
              f"energy_in_nuc={results[method]['energy_in_nucleus']:.3f}  "
              f"ratio={results[method]['concentration_ratio']:.2f} (area={results[method]['nucleus_area_frac']:.3f})")

    import runlog, plotting as P
    suffix = (f"_{img_size}px" if img_size != C.IMG_SIZE else "") + ("_fine" if fine_layer else "")
    rep = dict(which=which, img_size=img_size, fine_layer=fine_layer, n_per_class=n_per_class, methods=results)
    runlog.dump_json(os.path.join(C.METRICDIR, f"nucleus_faithfulness_{which}{suffix}.json"), rep)
    tag = f"@{img_size}px" + ("/fine" if fine_layer else "")
    P.bars({m: results[m]["energy_in_nucleus"] for m in results},
           os.path.join(C.FIGDIR, f"nucleus_energy_in_nucleus{suffix}"),
           ylabel="CAM energy in nucleus", title=f"Nuclear localization ({which} {tag})")
    P.bars({m: results[m]["nucleus_hit_rate"] for m in results},
           os.path.join(C.FIGDIR, f"nucleus_hit_rate{suffix}"),
           ylabel="nucleus hit-rate", title=f"CAM peak in nucleus ({which} {tag})")
    return rep


if __name__ == "__main__":
    run_nucleus_faithfulness()
