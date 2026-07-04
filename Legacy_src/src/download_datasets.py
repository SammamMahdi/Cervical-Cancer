"""Best-effort, IDEMPOTENT download of the external evaluation datasets (Herlev, APACC, HiCervix).

Each function skips if the dataset is already on disk (pass force=True to re-download), is robust on
failure (prints guidance, returns False), so re-running the notebook's download cell never re-fetches
gigabytes you already have. Cross-dataset eval runs on whatever lands in data/raw/.
"""
import os
import glob
import config as C

HERLEV_DIR = os.path.join(C.DATA_RAW, "herlev")
APACC_DIR = os.path.join(C.DATA_RAW, "apacc")
HICERVIX_DIR = os.path.join(C.DATA_RAW, "hicervix")

_IMG = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _count_images(dest):
    if not os.path.isdir(dest):
        return 0
    return sum(1 for p in glob.glob(os.path.join(dest, "**", "*"), recursive=True)
               if p.lower().endswith(_IMG))


def _already(tag, dest, force):
    """True (skip) if `dest` already holds images and not forcing."""
    n = _count_images(dest)
    if n > 0 and not force:
        print(f"[{tag}] already present at {dest} ({n} images) — skipping (force=True to re-download)")
        return True
    return False


def _kaggle():
    import kaggle; kaggle.api.authenticate(); return kaggle.api


def download_herlev(dest=HERLEV_DIR, ref="niladrishekharray/herlev-pap-smear-2005", force=False):
    """Herlev single-cell Pap smear, original 2005 (folder-per-class). `*-d.bmp` masks skipped at eval."""
    os.makedirs(dest, exist_ok=True)
    if _already("Herlev", dest, force):
        return True
    try:
        api = _kaggle()
        if ref is None:
            hits = list(api.dataset_list(search="herlev pap smear"))
            if not hits:
                print("[Herlev] no Kaggle dataset found by search; pass ref='owner/slug' manually"); return False
            ref = str(hits[0].ref)
        print(f"[Herlev] downloading {ref} ...")
        api.dataset_download_files(ref, path=dest, unzip=True, quiet=True)
        print(f"[Herlev] -> {dest}"); return True
    except Exception as e:
        print(f"[Herlev] FAILED ({e}). Manual: MDE-Lab smear2005.zip -> data/raw/herlev/"); return False


def download_hicervix(dest=HICERVIX_DIR, ref="suyash1567/hicervix", force=False):
    """HiCervix (single-cell crops under <split>/<split>/ + per-split CSVs, hierarchical Bethesda) from
    the Kaggle mirror. Zenodo original is firewall-blocked in some environments."""
    os.makedirs(dest, exist_ok=True)
    if _already("HiCervix", dest, force):
        return True
    try:
        api = _kaggle()
        print(f"[HiCervix] downloading {ref} (~4 GB) ...")
        api.dataset_download_files(ref, path=dest, unzip=True, quiet=False)
        print(f"[HiCervix] -> {dest}"); return True
    except Exception as e:
        print(f"[HiCervix] FAILED ({e}). Manual: kaggle datasets download {ref} -> data/raw/hicervix/")
        return False


def download_apacc(dest=APACC_DIR, splits=("test",), classes=("healthy", "unhealthy"), force=False):
    """APACC single-cell crops from OSF (fp2xe): dataset_full/cell-images/<split>/<class>/ (mapped
    classes only; skips rubbish/bothcells and the multi-GB whole-slide .mrxs)."""
    os.makedirs(dest, exist_ok=True)
    if _already("APACC", dest, force):
        return True
    try:
        from osfclient.api import OSF
        store = OSF().project("fp2xe").storage("osfstorage")
        full = next(fo for fo in store.folders if fo.name == "dataset_full")
        cells = next(fo for fo in full.folders if fo.name == "cell-images")
        n = 0
        for split in cells.folders:
            if split.name not in splits:
                continue
            for cls in split.folders:
                if cls.name not in classes:
                    continue
                outdir = os.path.join(dest, split.name, cls.name)
                os.makedirs(outdir, exist_ok=True)
                for f in cls.files:
                    if not f.name.lower().endswith(_IMG):
                        continue
                    out = os.path.join(outdir, f.name)
                    if not os.path.exists(out):
                        with open(out, "wb") as fh:
                            f.write_to(fh)
                    n += 1
                    if n % 200 == 0:
                        print(f"[APACC] {n} cell images ...", flush=True)
                print(f"[APACC] {split.name}/{cls.name}: done ({n} total so far)", flush=True)
        print(f"[APACC] downloaded {n} cell images -> {dest}"); return n > 0
    except Exception as e:
        print(f"[APACC] SKIP ({e}). Download https://osf.io/fp2xe/ -> dataset_full/cell-images "
              "into data/raw/apacc/"); return False


def download_all(force=False):
    return dict(herlev=download_herlev(force=force),
                hicervix=download_hicervix(force=force),
                apacc=download_apacc(force=force))


if __name__ == "__main__":
    download_all()
