"""Preflight + dataset verification for the new pipeline.

One command that (1) verifies every dataset is downloaded (and fetches any missing one by reusing the
idempotent Legacy_src downloader), (2) guards EVERY failure mode we hit during the legacy runs, and
(3) checks the GPU/BF16 environment. Writes results/metrics/preflight.json. Safe to re-run.

Legacy failures explicitly guarded here:
  * APACC zero-byte / truncated image crash      -> scan all datasets for 0-byte & unreadable files.
  * `ssl.py` shadowing the stdlib (broke torch)  -> assert no src/ssl.py (must be selfsup.py).
  * drive-letter case mismatch (F: vs f:)        -> assert split paths normcase-match on disk.
  * GPU unused / RAM instead of VRAM             -> assert CUDA visible; report device + VRAM.
  * Blackwell BF16 / num_workers on Windows      -> report bf16 support + recommended num_workers.
"""
import os, glob, sys
import config as C
import runlog

DATASETS = {
    "sipakmed": C.SIPAKMED_DIR, "herlev": C.HERLEV_DIR, "apacc": C.APACC_DIR,
    "hicervix": C.HICERVIX_DIR, "cric": os.path.join(C.DATA_RAW, "cric"),
}
IMG_EXT = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _count_images(root):
    if not os.path.isdir(root):
        return 0
    return sum(1 for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
               if p.lower().endswith(IMG_EXT))


def ensure_downloaded(force=False):
    """Verify each dataset; fetch any missing one via the idempotent legacy downloader."""
    present = {name: _count_images(root) for name, root in DATASETS.items()}
    missing = [n for n, c in present.items() if c == 0]
    if missing:
        print(f"[preflight] missing datasets: {missing} -> invoking legacy downloader")
        legacy_src = os.path.join(C.PROJECT, "Legacy_src", "src")
        if legacy_src not in sys.path:
            sys.path.insert(0, legacy_src)
        try:
            import download_datasets as DL
            DL.download_all(force=force)
            present = {name: _count_images(root) for name, root in DATASETS.items()}
        except Exception as e:
            print(f"[preflight] auto-download failed ({e}); fetch manually into data/raw/")
    return present, [n for n, c in present.items() if c == 0]


def scan_corrupt(sample_per_ds=None):
    """Scan datasets for 0-byte / unreadable images (the APACC crash class). Returns per-dataset counts."""
    from PIL import Image
    report = {}
    for name, root in DATASETS.items():
        if not os.path.isdir(root):
            report[name] = dict(checked=0, zero_byte=0, unreadable=0); continue
        files = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                 if p.lower().endswith(IMG_EXT)]
        if sample_per_ds and len(files) > sample_per_ds:
            import random
            files = random.Random(C.SEED).sample(files, sample_per_ds)
        zb, unl = 0, 0
        for p in files:
            try:
                if os.path.getsize(p) == 0:
                    zb += 1; continue
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                unl += 1
        report[name] = dict(checked=len(files), zero_byte=zb, unreadable=unl)
    return report


def check_code_hazards():
    """Guard the import/path bugs we hit (ssl.py shadowing; drive-letter case)."""
    hz = {}
    hz["ssl_shadow"] = not os.path.exists(os.path.join(os.path.dirname(__file__), "ssl.py"))
    # drive-letter case: split paths must normcase-match files on disk
    try:
        from data.splits import load_split
        sp = load_split()
        bad = [p for p in (sp.get("train", [])[:50]) if not os.path.exists(p)
               and os.path.exists(os.path.normcase(p))]
        hz["split_paths_case_ok"] = (len(bad) == 0)
    except Exception:
        hz["split_paths_case_ok"] = None   # split not built yet
    return hz


def check_env():
    import torch
    env = dict(torch=torch.__version__, cuda_available=torch.cuda.is_available(),
               device=C.DEVICE, recommended_num_workers_notebook=0, num_workers_cli_ok=4)
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda
        try:
            free, total = torch.cuda.mem_get_info()
            env["vram_total_gb"] = round(total / 1e9, 1); env["vram_free_gb"] = round(free / 1e9, 1)
        except Exception:
            pass
        try:
            env["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception:
            env["bf16_supported"] = None
    return env


def run(scan_full=False):
    present, missing = ensure_downloaded()
    rep = dict(
        datasets={n: dict(images=c, present=(c > 0)) for n, c in present.items()},
        missing=missing,
        corrupt_scan=scan_corrupt(sample_per_ds=(None if scan_full else 600)),
        code_hazards=check_code_hazards(),
        env=check_env(),
    )
    rep["ok"] = (not missing
                 and all(v["zero_byte"] == 0 and v["unreadable"] == 0 for v in rep["corrupt_scan"].values())
                 and rep["code_hazards"].get("ssl_shadow") is True
                 and rep["env"]["cuda_available"])
    runlog.dump_json(os.path.join(C.METRICDIR, "preflight.json"), rep)
    print("Datasets:", {n: present[n] for n in present})
    print("Missing:", missing or "none")
    bad = {n: v for n, v in rep["corrupt_scan"].items() if v["zero_byte"] or v["unreadable"]}
    print("Corrupt/zero-byte:", bad or "none")
    print("Env:", {k: rep["env"].get(k) for k in ("device", "gpu", "vram_total_gb", "bf16_supported")})
    print("Code hazards:", rep["code_hazards"])
    print("PREFLIGHT OK" if rep["ok"] else "PREFLIGHT: issues above (see preflight.json)")
    return rep


if __name__ == "__main__":
    run(scan_full="--full" in sys.argv)
