import os, glob, kaggle
dest = r"f:\Cervaical Cancer\data\raw\cric"
os.makedirs(dest, exist_ok=True)
ref = "rakibhossainsajib/cric-cervix-dataset"
print("downloading", ref, "->", dest)
kaggle.api.authenticate()
kaggle.api.dataset_download_files(ref, path=dest, unzip=True, quiet=False)
# inventory
imgs = glob.glob(os.path.join(dest, "**", "*.png"), recursive=True)
csvs = glob.glob(os.path.join(dest, "**", "*.csv"), recursive=True)
print(f"\nCRIC: {len(imgs)} png, {len(csvs)} csv")
for c in csvs:
    print("CSV:", os.path.relpath(c, dest))
    with open(c, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i < 3: print("   ", line.rstrip()[:140])
            else: break
