"""Config for the PyTorch re-implementation of the legacy SIPaKMeD pipeline.

Values mirror the original tf_keras notebook (efficientnetb4-sipakmed.ipynb) so the
re-implementation reproduces the listed ~97% on the SAME random-split task.
Framework: PyTorch (cu128) on the RTX 5060 Ti (Blackwell, BF16).
"""
import os
import torch

# ---- paths ----
PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))   # f:\Cervaical Cancer
DATA_RAW = os.path.join(PROJECT, "data", "raw")
SIPAKMED_DIR = os.path.join(DATA_RAW, "sipakmed")          # im_<Class>/ folders (or nested)
RESULTS = os.path.join(PROJECT, "Legacy_src", "results")
FIGDIR = os.path.join(RESULTS, "figures")
METRICDIR = os.path.join(RESULTS, "metrics")
CKPTDIR = os.path.join(RESULTS, "checkpoints")
LOGDIR = os.path.join(RESULTS, "logs")             # timestamped per-run JSON logs
for d in (RESULTS, FIGDIR, METRICDIR, CKPTDIR, LOGDIR):
    os.makedirs(d, exist_ok=True)

# ---- reproducibility ----
SEED = 42

# ---- data / classes (5 SIPaKMeD morphology classes; alphabetical folder order) ----
RAW_CLASSES = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic",
               "im_Parabasal", "im_Superficial-Intermediate"]
CLASSES = [c.replace("im_", "") for c in RAW_CLASSES]
NUM_CLASSES = len(CLASSES)

# ---- image / batch (legacy used 380x380 for BOTH teacher and student) ----
IMG_SIZE = 380
BATCH_SIZE = 32
VAL_SPLIT = 0.2          # legacy random split (faithful reproduction; NOT the upgrade's group split)
NUM_WORKERS = 0          # Windows + Jupyter: spawn-based workers can't start in a notebook kernel
                         # (they hang the first epoch and starve the GPU). 0 = load in main process.
                         # Also lighter on the 15 GB host RAM. For CLI runs you can raise this.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---- teacher (EfficientNet-B4) two-stage schedule (from notebook) ----
TEACHER = dict(
    backbone="efficientnet_b4", hidden=512, dropout=0.5,
    stage1=dict(lr=1e-3, epochs=30, patience=8),     # frozen backbone
    stage2=dict(lr=1e-5, epochs=50, patience=6),     # fine-tune (unfrozen)
    reduce_lr=dict(factor=0.3, patience=4, min_lr=1e-6),
)

# ---- student (EfficientNet-B0) distillation (from notebook) ----
STUDENT = dict(
    backbone="efficientnet_b0", hidden=512, dropout=0.3,
    lr=5e-5, epochs=40, patience=8,
    temperature=4.0, alpha=0.1,    # alpha = weight on hard CE; (1-alpha) on soft KL
)

# ---- device ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = True       # Blackwell BF16 autocast

def set_seed(seed=SEED):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
