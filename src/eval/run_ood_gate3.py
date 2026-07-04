"""GATE 3 driver: per-domain mean free-energy + internal-vs-external OOD AUROC.

Loads the latest frozen student (run._load_model('student')) and computes label-free
free-energy E(x) = -logsumexp(raw logits) on:
  * the sealed SIPaKMeD test split (internal), and
  * each external set (Herlev, HiCervix, APACC, CRIC),
REUSING the exact external loader construction from run.evaluate() / eval.metrics.

External sets are sub-sampled to <= MAX_N images (reported as n) for speed. The premise:
mean energy should rise from internal to external and track the measured external-ECE
ordering (SIPaKMeD < Herlev < HiCervix < APACC < CRIC), and internal-vs-external AUROC
should exceed ~0.65 for the strongly-shifted domains.
"""
import os, json
import numpy as np
from torch.utils.data import DataLoader, Subset

import config as C
import run
import eval.metrics as M
import eval.ood as OOD
from data.dataset import make_loaders, build_transform, fit_stain
from data import labelmap as LM

MAX_N = 2000          # sub-sample cap per external set (speed); internal test used in full
SEED = C.SEED


def _subset_loader(ds, max_n, num_workers, batch_size=C.BATCH_SIZE):
    """Deterministic sub-sample of `ds` to <= max_n items -> a non-shuffled DataLoader."""
    n = len(ds)
    if n > max_n:
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(n, size=max_n, replace=False))
        ds = Subset(ds, idx.tolist())
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False)


def main(which="student", num_workers=2, stain="none", img_size=C.IMG_SIZE):
    model = run._load_model(which)

    # ---- internal: sealed SIPaKMeD test split (full) ----
    _, _vl, test_ld, _ = make_loaders(split_kind="grouped", img_size=img_size, stain=stain,
                                      num_workers=num_workers)
    e_int, _ = OOD.energy_logits(model, test_ld, desc="sipakmed test")
    print(f"[internal] sipakmed test n={e_int.size} meanE={e_int.mean():.4f}")

    # ---- externals: SAME wiring as run.evaluate() / metrics ----
    tf = build_transform(train=False, img_size=img_size,
                         stain_norm=fit_stain(stain, split_kind="grouped"))

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

    domains = {"sipakmed_internal": OOD.summarize_energy(e_int)}
    energies = {"sipakmed_internal": e_int}
    aurocs = {}

    for name in ("herlev", "hicervix", "apacc", "cric"):
        try:
            ds = _build_ext(name)
            total = len(ds)
            ld = _subset_loader(ds, MAX_N, num_workers)
            e_ext, _ = OOD.energy_logits(model, ld, desc=f"{name}", max_n=MAX_N)
            energies[name] = e_ext
            domains[name] = OOD.summarize_energy(e_ext)
            domains[name]["n_total_available"] = int(total)
            auroc = OOD.ood_auroc(e_int, e_ext)
            aurocs[name] = auroc
            print(f"[external] {name:9s} n={e_ext.size:5d}/{total:6d} "
                  f"meanE={e_ext.mean():.4f}  AUROC(int-vs-ext)={auroc:.4f}")
        except Exception as e:
            print(f"[external {name}] FAILED: {type(e).__name__}: {e}")

    # ---- measured external ECE (from sealed cross_*_E0.json) for the ordering check ----
    ece = {}
    for name in ("herlev", "hicervix", "apacc", "cric"):
        p = os.path.join(C.METRICDIR, f"cross_{name}_E0.json")
        if os.path.isfile(p):
            ece[name] = json.load(open(p)).get("ece")
    ece["sipakmed_internal"] = 0.035   # internal sealed test ECE (from task / indist)

    # ---- ordering check: mean energy vs ECE ordering ----
    order_by_energy = sorted([d for d in domains if d in ece],
                             key=lambda d: domains[d]["mean"])
    order_by_ece = sorted([d for d in domains if d in ece], key=lambda d: ece[d])

    out = dict(
        which=which, max_n=MAX_N, T=1.0,
        energy_definition="E(x) = -logsumexp(raw_logits, axis=1)  [lower=in-dist]",
        per_domain=domains,
        ood_auroc_internal_vs_external=aurocs,
        measured_external_ece=ece,
        order_by_mean_energy=order_by_energy,
        order_by_measured_ece=order_by_ece,
        cric_is_highest_energy=bool(
            "cric" in domains and
            domains["cric"]["mean"] == max(domains[d]["mean"] for d in domains)),
    )
    os.makedirs(C.METRICDIR, exist_ok=True)
    outpath = os.path.join(C.METRICDIR, "ood_energy_gate3.json")
    with open(outpath, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n==== SUMMARY ====")
    print("mean energy ordering :", " < ".join(order_by_energy))
    print("measured ECE ordering:", " < ".join(order_by_ece))
    print("AUROCs:", {k: round(v, 3) for k, v in aurocs.items()})
    print(f"wrote {outpath}")
    return out


if __name__ == "__main__":
    main()
