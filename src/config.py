"""Config for the NEW (upgraded) cervical-cytology pipeline.

Implements the committed design in plans/: train a 5-class SIPaKMeD model on the 4,049 CROPPED
single cells with a LEAKAGE-SAFE, cluster-grouped 80/10/10 split (U1), correct knowledge
distillation (U2), cross-dataset E0/E1 evaluation (U3), stain normalization (U4), calibration (U5),
efficiency profiling (U6). PyTorch on the RTX 5060 Ti (Blackwell, BF16).

This package lives at src/ and is independent of Legacy_src/ (the faithful legacy reproduction).
"""
import os
import torch

# ---------------- paths ----------------
PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # f:\Cervaical Cancer
DATA_RAW = os.path.join(PROJECT, "data", "raw")
SIPAKMED_DIR = os.path.join(DATA_RAW, "sipakmed")
HERLEV_DIR = os.path.join(DATA_RAW, "herlev")
APACC_DIR = os.path.join(DATA_RAW, "apacc")
HICERVIX_DIR = os.path.join(DATA_RAW, "hicervix")

RESULTS = os.path.join(PROJECT, "results")               # NEW pipeline results (root, not Legacy_src)
SPLITDIR = os.path.join(RESULTS, "splits")
FIGDIR = os.path.join(RESULTS, "figures")
METRICDIR = os.path.join(RESULTS, "metrics")
CKPTDIR = os.path.join(RESULTS, "checkpoints")
LOGDIR = os.path.join(RESULTS, "logs")
for _d in (RESULTS, SPLITDIR, FIGDIR, METRICDIR, CKPTDIR, LOGDIR):
    os.makedirs(_d, exist_ok=True)

# ---------------- reproducibility ----------------
SEED = 42

# ---------------- SIPaKMeD classes (NORMAL first, then ABNORMAL -> clean E0/E1 collapse) ----------------
# idx 0,1,2 = Normal ; idx 3,4 = Abnormal (Koilo=LSIL bridge, Dysk=low-grade by default)
CLASSES = ["Superficial-Intermediate", "Parabasal", "Metaplastic", "Koilocytotic", "Dyskeratotic"]
RAW_CLASS_DIRS = {c: f"im_{c}" for c in CLASSES}
NUM_CLASSES = len(CLASSES)
NORMAL_IDX = [0, 1, 2]
ABNORMAL_IDX = [3, 4]

# ---------------- split (leakage-safe, U1) ----------------
SPLIT = dict(train=0.8, val=0.1, test=0.1, n_cv_folds=5)   # group-aware, grouped by source FOV/cluster

# ---------------- input / preprocessing (U / DATA_PLAN §4) ----------------
IMG_SIZE = 224           # EfficientNet-B0 native; student trains/evaluates here (aspect-preserving pad)
TEACHER_IMG_SIZE = 380   # EfficientNet-B4 native: train the teacher at its own resolution (the student
                         # stays at 224; KD upsamples the student batch to the teacher's resolution).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------- training defaults (HPO overrides these; U2b) ----------------
BATCH_SIZE = 64          # student @224px -> larger batch fits 16 GB
TEACHER_BATCH_SIZE = 32  # teacher @380px is ~3x heavier; 32 fits 16 GB (matches the legacy B4 run)
NUM_WORKERS = 0          # Windows + Jupyter safe (see Legacy_src diagnosis); raise for CLI runs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = True

# weight EMA (smooths the honest, small-val curve and usually lifts final accuracy; the EMA weights are
# what get evaluated, early-stopped on, and saved as 'best')
USE_EMA = True
EMA_DECAY = 0.999

TEACHER = dict(
    backbone="efficientnet_b4", hidden=512, dropout=0.4,
    stage1=dict(lr=1e-3, epochs=50, patience=8),     # frozen backbone (cap raised so patience decides)
    stage2=dict(lr=1e-5, epochs=120, patience=10),   # fine-tune to convergence (was 50 -> hit the cap)
    reduce_lr=dict(factor=0.3, patience=4, min_lr=1e-7),
    optimizer="adamw", weight_decay=1e-4, label_smoothing=0.05,
)
STUDENT = dict(
    backbone="efficientnet_b0", hidden=512, dropout=0.3,
    lr=5e-4, epochs=60, patience=10,
    optimizer="adamw", weight_decay=1e-4, label_smoothing=0.05,
    temperature=4.0, alpha=0.5,    # KD: L = alpha*CE + (1-alpha)*T^2*KL  (Hinton, fixed)
)

# ablation backbone registry (A8): name -> torchvision builder key
ABLATION_BACKBONES = ["efficientnet_b0", "efficientnet_b2", "mobilenet_v3_large", "resnet50"]

# ---------------- evaluation tiers (U3 / DATA_PLAN §3) ----------------
ABNORMAL_THRESHOLD = 0.5     # pre-registered P(abnormal) operating point for E0

# CRIC point-crop size (px). Data-driven: CRIC is a DENSE preparation — annotated nuclei sit ~70px
# apart (median nearest-neighbour) / ~87px (mean) (results/metrics/cric_crop_geometry.json), so an
# ~80px window captures roughly one cell's territory without heavy neighbour overlap. Isolated
# whole-cell crops at SIPaKMeD's ~200px scale are impossible here (they would include 2-3 neighbours).
# Confirm/refine empirically with run.cric_crop_sweep -> cric_crop_sensitivity.json.
CRIC_CROP = 80
CRIC_CROP_SWEEP = [48, 64, 80, 96, 112, 144]


def set_seed(seed=SEED, deterministic=False):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
