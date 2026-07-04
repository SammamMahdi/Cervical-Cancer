"""MINI-GATE A — feature-space Mahalanobis OOD (replacement for the falsified free-energy signal).

Fits class-conditional Gaussians (per-class means + ONE shared Ledoit-Wolf shrinkage covariance)
on SIPaKMeD TRAIN penultimate features (models.backbones.Classifier.features_vec), then scores
the sealed SIPaKMeD test split (internal) and each external set (Herlev, HiCervix, APACC, CRIC)
with the min-over-classes squared Mahalanobis distance. Higher score == more OOD.

REUSES the exact external-loader construction from run_ood_gate3.py (FolderExternal / HiCervix /
cric_dataset). Externals are sub-sampled to <= MAX_N (seeded). Reports mean score per domain and
internal-vs-external AUROC, side by side with the previously-measured ENERGY AUROCs.

Energy baseline (already falsified, from MEMORY): herlev 0.722 / hicervix 0.905 / apacc 0.537 / cric 0.683.
"""
import os, json
import numpy as np
from torch.utils.data import DataLoader, Subset

import config as C
import run
import eval.metrics as M
import eval.ood as OOD
from data.dataset import CellDataset, build_transform, fit_stain
from data.splits import get_split
from data import labelmap as LM

MAX_N = 2000
SEED = C.SEED
ENERGY_AUROC = {"herlev": 0.722, "hicervix": 0.905, "apacc": 0.537, "cric": 0.683}


def _subset_loader(ds, max_n, num_workers, batch_size=C.BATCH_SIZE):
    n = len(ds)
    if n > max_n:
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(n, size=max_n, replace=False))
        ds = Subset(ds, idx.tolist())
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False)


def _class_of(p):
    from data.dataset import _class_of as cof
    return cof(p)


def main(which="student", num_workers=2, stain="none", img_size=C.IMG_SIZE):
    model = run._load_model(which)

    # ---- SIPaKMeD split (leakage-safe grouped) ----
    split = get_split("grouped")
    stain_norm = fit_stain(stain, split=split)
    tf_eval = build_transform(train=False, img_size=img_size, stain_norm=stain_norm)

    # ---- FIT: SIPaKMeD TRAIN features (deterministic eval transform, NOT augmented) ----
    train_ds = CellDataset(split["train"], tf_eval)
    train_ld = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                          num_workers=num_workers, pin_memory=False)
    f_tr, y_tr = OOD.extract_features(model, train_ld, desc="sipakmed train (fit)")
    scorer = OOD.MahalanobisScorer().fit(f_tr, y_tr)
    print(f"[fit] train feats={f_tr.shape} classes={scorer.classes_.tolist()} "
          f"LedoitWolf shrinkage={scorer.shrinkage_:.4f}")

    # ---- internal: sealed SIPaKMeD TEST features ----
    test_ds = CellDataset(split["test"], tf_eval)
    test_ld = DataLoader(test_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                         num_workers=num_workers, pin_memory=False)
    f_te, _ = OOD.extract_features(model, test_ld, desc="sipakmed test")
    s_int = scorer.score(f_te)
    print(f"[internal] sipakmed test n={s_int.size} meanMaha={s_int.mean():.4f}")

    # ---- externals: SAME wiring as run_ood_gate3.py ----
    tf = build_transform(train=False, img_size=img_size, stain_norm=fit_stain(stain, split_kind="grouped"))

    def _build_ext(name):
        if name == "herlev":
            return M.FolderExternal(C.HERLEV_DIR, LM.herlev_map, tf, tier="E0")
        if name == "apacc":
            return M.FolderExternal(C.APACC_DIR, LM.apacc_map, tf, tier="E0")
        if name == "hicervix":
            return M.HiCervixExternal(tf, tier="E0")
        if name == "cric":
            from data.external_tier2 import cric_dataset
            return cric_dataset(tier="E0", transform=tf)
        raise ValueError(name)

    def _summary(s):
        s = np.asarray(s, dtype=np.float64)
        if s.size == 0:
            return dict(n=0, mean=float("nan"), std=float("nan"), median=float("nan"))
        return dict(n=int(s.size), mean=float(s.mean()), std=float(s.std()),
                    median=float(np.median(s)), p25=float(np.percentile(s, 25)),
                    p75=float(np.percentile(s, 75)))

    domains = {"sipakmed_internal": _summary(s_int)}
    aurocs = {}

    for name in ("herlev", "hicervix", "apacc", "cric"):
        try:
            ds = _build_ext(name)
            total = len(ds)
            ld = _subset_loader(ds, MAX_N, num_workers)
            f_ext, _ = OOD.extract_features(model, ld, desc=name, max_n=MAX_N)
            s_ext = scorer.score(f_ext)
            domains[name] = _summary(s_ext)
            domains[name]["n_total_available"] = int(total)
            auroc = OOD.ood_auroc(s_int, s_ext)
            aurocs[name] = auroc
            print(f"[external] {name:9s} n={s_ext.size:5d}/{total:6d} "
                  f"meanMaha={s_ext.mean():.4f}  AUROC(int-vs-ext)={auroc:.4f}  "
                  f"(energy was {ENERGY_AUROC.get(name)})")
        except Exception as e:
            print(f"[external {name}] FAILED: {type(e).__name__}: {e}")

    # ---- side-by-side Mahalanobis vs energy ----
    compare = {n: dict(mahalanobis=aurocs.get(n), energy=ENERGY_AUROC.get(n),
                       beats_energy=bool(aurocs.get(n, 0) > ENERGY_AUROC.get(n, 1)))
               for n in ("herlev", "hicervix", "apacc", "cric")}

    out = dict(
        which=which, max_n=MAX_N,
        scorer="class-conditional Mahalanobis, shared LedoitWolf cov, score=min_k Maha^2",
        ledoitwolf_shrinkage=scorer.shrinkage_,
        feature_dim=int(f_tr.shape[1]),
        per_domain=domains,
        ood_auroc_internal_vs_external=aurocs,
        energy_auroc_baseline=ENERGY_AUROC,
        mahalanobis_vs_energy=compare,
    )
    os.makedirs(C.METRICDIR, exist_ok=True)
    outpath = os.path.join(C.METRICDIR, "mini_gate_a_mahalanobis.json")
    with open(outpath, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n==== SUMMARY (Mahalanobis vs energy AUROC) ====")
    for n in ("herlev", "hicervix", "apacc", "cric"):
        c = compare[n]
        print(f"  {n:9s} maha={c['mahalanobis']!s:>7}  energy={c['energy']:.3f}  "
              f"beats_energy={c['beats_energy']}")
    print(f"wrote {outpath}")
    return out


if __name__ == "__main__":
    main()
