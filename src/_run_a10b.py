"""Launch A10b: 8-leaf multi-source partial-label training + cross-dataset evaluate8.

The decisive measurement of whether un-excluding the 40% (BethHead-8 + partial labels over
SIPaKMeD + Herlev + APACC + HiCervix + CRIC) improves cross-dataset E0/E2 vs the baseline.

Run as a script (NOT `python -c`) so DataLoader workers spawn safely on Windows (num_workers>0
needs the __main__ guard). Background-launched; logs to results/logs/a10b.log.
"""
import os, time, json
import config as C

C.NUM_WORKERS = 4    # CLI run: parallelize the ~49k-cell load (Jupyter default is 0)
EPOCHS = 15
TAG = "student_A10b"


def main():
    import route_c, run
    t0 = time.time()
    print(f"[a10b] start: 8-leaf multisource, epochs={EPOCHS}, num_workers={C.NUM_WORKERS}", flush=True)
    out = route_c.train_partial(
        taxonomy="bethesda8",
        external_sources=("herlev", "apacc", "hicervix", "cric"),
        w_src={"sipakmed": 1.0, "external": 0.5},
        balanced_sampler=True, sipakmed_frac=0.4,   # keep SIPaKMeD ~40% of each batch (anti-collapse)
        epochs=EPOCHS, tag=TAG, save=True)
    st = out["sealed_test"]; ds = out["dataset"]
    print(f"[a10b] TRAIN DONE in {(time.time()-t0)/60:.1f} min | dataset n={ds['n_total']} "
          f"(sipakmed {ds['n_sipakmed_singleton']} + external {ds['n_external']}) | "
          f"sealed-test acc={st['accuracy']:.4f} macroF1={st['macro_f1']:.4f}", flush=True)

    # cross-dataset 8-leaf evaluation (the payoff)
    run.evaluate8(TAG)
    p = os.path.join(C.METRICDIR, f"eval8_{TAG}.json")
    ev = json.load(open(p, encoding="utf-8"))
    print("[a10b] EVAL8 DONE ->", p, flush=True)
    for name, m in ev.get("cross_E0", {}).items():
        print(f"[a10b]   {name:9s} E0 acc={m['accuracy']:.3f} balacc={m['balanced_accuracy']:.3f} "
              f"auroc={m['auroc']:.3f} ece={m['ece']:.3f}", flush=True)
    e2 = ev.get("E2_nonsquamous", {})
    if e2:
        print(f"[a10b]   E2 non-squamous AUROC={e2.get('auroc')}", flush=True)
    print("[a10b] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
