"""U3 / Route-A deterministic label crosswalk (DATA_PLAN §3).

Collapse BOTH the model's 5-class prediction AND each external dataset's native labels to a shared,
safe MANY-TO-ONE level:
  E0  Binary  : normal / abnormal           (primary cross-dataset metric)
  E1  Grade   : normal / low / high         (diagnostic; model has NO high-grade output by design)

Every mapping is many-to-one (never the dangerous one-to-many 5-class projection). Unmappable native
classes (glandular, ASC-US indeterminate, debris) map to EXCLUDE (None) and are reported as the
"excluded fraction" (an honesty metric), never silently dropped.
"""
import numpy as np
import config as C

LABELMAP_VERSION = "v1-2026-06"
EXCLUDE = None   # cells that have no home in the SIPaKMeD space

# ---------------- model prediction -> tiers ----------------
def pred_to_e0(class_idx):
    """5-class index -> 'normal'/'abnormal'."""
    return "abnormal" if class_idx in C.ABNORMAL_IDX else "normal"

def pred_to_e1(class_idx):
    """5-class index -> 'normal'/'low'. The model can NEVER output 'high' (structural blind spot)."""
    return "low" if class_idx in C.ABNORMAL_IDX else "normal"

def p_abnormal(probs):
    """P(abnormal) = softmax(Koilocytotic)+softmax(Dyskeratotic). probs: (N,5) -> (N,)."""
    probs = np.asarray(probs)
    return probs[:, C.ABNORMAL_IDX].sum(1)


# ---------------- external native label -> (E0, E1) ----------------
def _kw(s, *keys):
    s = s.lower()
    return any(k in s for k in keys)

def herlev_map(folder):
    """Herlev 7-class folder name -> (E0, E1)."""
    s = folder.lower()
    if _kw(s, "columnar"):                                   return (EXCLUDE, EXCLUDE)   # glandular
    if _kw(s, "superficiel", "superficial", "intermediate"): return ("normal", "normal")
    if _kw(s, "light", "mild"):                              return ("abnormal", "low")  # CIN1
    if _kw(s, "moderate", "severe", "carcinoma", "in_situ"): return ("abnormal", "high")
    if _kw(s, "normal"):                                     return ("normal", "normal")
    return (EXCLUDE, EXCLUDE)

def apacc_map(folder):
    """APACC 4-class folder name -> (E0, E1). NOTE: test 'unhealthy' BEFORE 'healthy' (substring)."""
    s = folder.lower()
    if _kw(s, "unhealthy", "abnormal"): return ("abnormal", EXCLUDE)   # no grade in APACC
    if _kw(s, "healthy", "normal"):     return ("normal", "normal")
    return (EXCLUDE, EXCLUDE)                                          # rubbish / bothcells

def hicervix_map(label):
    """HiCervix hierarchical-Bethesda leaf/label -> (E0, E1). Keyword Bethesda crosswalk."""
    s = str(label).lower().strip()
    if _kw(s, "nilm", "negative", "normal", "inflammat", "reactive", "benign"): return ("normal", "normal")
    if _kw(s, "glandular", "endocervical", "endometrial", "agc", "adenocarc", "ais"): return (EXCLUDE, EXCLUDE)
    if _kw(s, "asc-h", "asch"):       return ("abnormal", "high")
    if _kw(s, "hsil", "high"):        return ("abnormal", "high")
    if _kw(s, "scc", "carcinoma", "squamous cell carc"): return ("abnormal", "high")
    if _kw(s, "lsil", "low"):         return ("abnormal", "low")
    if _kw(s, "asc-us", "ascus", "asc us"): return ("abnormal", EXCLUDE)   # indeterminate
    if _kw(s, "atypical", "asc"):     return ("abnormal", EXCLUDE)
    return (EXCLUDE, EXCLUDE)

EXTERNAL_MAPS = {"herlev": herlev_map, "apacc": apacc_map, "hicervix": hicervix_map}


# ---------------- E-morph: candidate-set hit (lenient morphology) ----------------
# A 5-class prediction counts correct if it lies in the native label's morphology candidate set.
# Many-to-one and safe: high-grade / glandular / indeterminate native labels have NO candidate
# (None -> excluded), since the model's 5 morphology classes cannot represent them.
NORMAL_CAND = frozenset(C.NORMAL_IDX)                                   # {SI, Parabasal, Metaplastic}
LOW_CAND = frozenset(C.ABNORMAL_IDX)                                    # {Koilocytotic, Dyskeratotic}
SI_ONLY = frozenset([C.CLASSES.index("Superficial-Intermediate")])     # {SI}

def herlev_candidates(folder):
    s = folder.lower()
    if _kw(s, "columnar"):                                   return None        # glandular
    if _kw(s, "superficiel", "superficial", "intermediate"): return SI_ONLY     # specific -> SI
    if _kw(s, "light", "mild"):                              return LOW_CAND     # CIN1/LSIL
    if _kw(s, "moderate", "severe", "carcinoma", "in_situ"): return None        # high-grade, no home
    if _kw(s, "normal"):                                     return NORMAL_CAND
    return None

def apacc_candidates(folder):
    s = folder.lower()
    if _kw(s, "unhealthy", "abnormal"): return LOW_CAND      # only abnormal candidates the model has
    if _kw(s, "healthy", "normal"):     return NORMAL_CAND
    return None

def hicervix_candidates(label):
    s = str(label).lower()
    if _kw(s, "nilm", "negative", "normal", "inflammat", "reactive", "benign"): return NORMAL_CAND
    if _kw(s, "lsil", "low"): return LOW_CAND
    return None   # asc-us / hsil / asc-h / scc / glandular -> no morphology home

EXTERNAL_CANDIDATES = {"herlev": herlev_candidates, "apacc": apacc_candidates, "hicervix": hicervix_candidates}

def pred_in_candidates(pred_idx, cand):
    return cand is not None and int(pred_idx) in cand


# ---------------- Bethesda-aligned 8-leaf head (Wave-1 candidate taxonomy) ----------------
# A wider, Bethesda-aligned leaf set that ADDS three non-squamous nodes (GLANDULAR, MICROBE,
# INDETERMINATE) on top of the 5 SIPaKMeD morphology leaves. This does NOT touch the trained
# 5-class model (C.NUM_CLASSES stays 5); it is a leaf-name taxonomy used to recount how many
# HiCervix cells become *representable* (have >=1 candidate leaf) once the head is widened.
# Keyword/level_1 driven, deterministic, many-to-(small set), mirroring the existing maps.
BETHESDA8_LEAVES = [
    "SI", "Parabasal", "Metaplastic",        # normal squamous (SIPaKMeD)
    "Koilocytotic", "Dyskeratotic",          # abnormal squamous (SIPaKMeD)
    "GLANDULAR", "MICROBE", "INDETERMINATE",  # new non-squamous Bethesda nodes
]

# convenient leaf-name groupings over the 8 leaves
_NORMAL_SQ = ["SI", "Parabasal", "Metaplastic"]
_ABNORMAL_SQ = ["Koilocytotic", "Dyskeratotic"]


def hicervix8_candidates(level_1_label):
    """HiCervix level_1 (Bethesda category) -> candidate leaf names among BETHESDA8_LEAVES.

    Returns a set of candidate leaf names (possibly empty if still unrepresentable).
    Deterministic / keyword-based, keyed on the HiCervix level_1 label:
      negative -> normal squamous leaves {SI, Parabasal, Metaplastic}
      ASC      -> {INDETERMINATE} + abnormal squamous leaves (atypical squamous, indeterminate grade)
      AGC      -> {GLANDULAR}     (atypical glandular cells -> new glandular node)
      microbe  -> {MICROBE}       (organisms: ACTINO/FUNGI/TRI/HSV/... -> new microbe node)
    Anything else returns an empty set (still excluded).
    """
    s = str(level_1_label).lower().strip()
    # microbe / organisms
    if _kw(s, "microbe", "actino", "fungi", "trich", "tri", "hsv", "organism", "candida"):
        return set(["MICROBE"])
    # glandular (atypical glandular cells, adeno, endocervical/endometrial)
    if _kw(s, "agc", "glandular", "adenocarc", "adc", "endocervical", "endometrial", "ais"):
        return set(["GLANDULAR"])
    # atypical squamous cells of undetermined significance -> indeterminate + abnormal squamous morphology
    if _kw(s, "asc-us", "ascus", "asc us"):
        return set(["INDETERMINATE"]) | set(_ABNORMAL_SQ)
    if _kw(s, "asc-h", "asch"):
        return set(["INDETERMINATE"]) | set(_ABNORMAL_SQ)
    if _kw(s, "asc", "atypical"):
        return set(["INDETERMINATE"]) | set(_ABNORMAL_SQ)
    # explicit squamous lesion grades
    if _kw(s, "lsil", "low"):
        return set(_ABNORMAL_SQ)
    if _kw(s, "hsil", "high", "scc", "carcinoma"):
        return set(["INDETERMINATE"]) | set(_ABNORMAL_SQ)
    # negative / normal
    if _kw(s, "negative", "nilm", "normal", "inflammat", "reactive", "benign"):
        return set(_NORMAL_SQ)
    return set()


# ---- index-based 8-leaf crosswalks for partial-label TRAINING (Route-C+, CytoBridge) ----
# The first 5 leaf indices equal the SIPaKMeD class indices (C.CLASSES), so a SIPaKMeD singleton
# {class_idx} is a valid candidate in the 8-leaf space unchanged; the 3 new nodes (5,6,7) give the
# previously-excluded external cells a training home.
LEAF8_IDX = {name: i for i, name in enumerate(BETHESDA8_LEAVES)}   # name -> 0..7
N_LEAVES8 = len(BETHESDA8_LEAVES)                                   # 8
_NORMAL_SQ8 = frozenset(LEAF8_IDX[n] for n in _NORMAL_SQ)          # {0,1,2}
_ABN_SQ8 = frozenset(LEAF8_IDX[n] for n in _ABNORMAL_SQ)          # {3,4}
GLANDULAR8, MICROBE8, INDETERMINATE8 = LEAF8_IDX["GLANDULAR"], LEAF8_IDX["MICROBE"], LEAF8_IDX["INDETERMINATE"]
# E0 abnormality over 8 leaves: abnormal squamous + glandular(AGC) + indeterminate(ASC); microbe is benign.
_ABNORMAL8 = _ABN_SQ8 | frozenset([GLANDULAR8, INDETERMINATE8])


def _names_to_idxset(names):
    """set of leaf NAMES -> frozenset of leaf indices, or None when empty (drop, per Route-C)."""
    if not names:
        return None
    return frozenset(LEAF8_IDX[n] for n in names)


def herlev8_candidates(folder):
    """Herlev folder name -> candidate leaf-index set over the 8 leaves (None -> drop)."""
    s = folder.lower()
    if _kw(s, "columnar"):                                    return frozenset([GLANDULAR8])     # glandular (was dropped)
    if _kw(s, "superficiel", "superficial", "intermediate"): return frozenset([LEAF8_IDX["SI"]])
    if _kw(s, "light", "mild"):                              return _ABN_SQ8                     # LSIL/CIN1
    if _kw(s, "moderate", "severe", "carcinoma", "in_situ"): return frozenset([INDETERMINATE8]) | _ABN_SQ8  # high-grade (was dropped)
    if _kw(s, "normal"):                                     return _NORMAL_SQ8
    return None


def apacc8_candidates(folder):
    """APACC folder name -> candidate leaf-index set over the 8 leaves (None -> drop)."""
    s = folder.lower()
    if _kw(s, "unhealthy", "abnormal"): return _ABN_SQ8        # generic abnormal squamous
    if _kw(s, "healthy", "normal"):     return _NORMAL_SQ8
    return None


def bethesda8_candidates(token):
    """HiCervix level_1 OR CRIC bethesda token -> candidate leaf-index set (None -> drop).

    Reuses the keyword logic of ``hicervix8_candidates`` (which returns leaf NAMES) and converts to
    indices; covers negative/ASC/AGC/microbe (HiCervix level_1) and Negative/LSIL/HSIL/ASC-H/ASC-US/SCC
    (CRIC). Previously-excluded glandular/microbe/high-grade now map to the new non-squamous nodes."""
    return _names_to_idxset(hicervix8_candidates(token))


EXTERNAL_CANDIDATES8 = {"herlev": herlev8_candidates, "apacc": apacc8_candidates,
                        "hicervix": bethesda8_candidates, "cric": bethesda8_candidates}


def pred8_to_e0(leaf_idx):
    """8-leaf prediction index -> 'normal'/'abnormal' (E0). Microbe is benign->normal."""
    return "abnormal" if int(leaf_idx) in _ABNORMAL8 else "normal"


def pred8_to_e1(leaf_idx):
    """8-leaf prediction index -> 'normal'/'low' (E1; model still has no 'high'). Non-squamous->EXCLUDE."""
    i = int(leaf_idx)
    if i in _NORMAL_SQ8 or i == MICROBE8: return "normal"
    if i in _ABN_SQ8:                     return "low"
    return EXCLUDE                                              # glandular / indeterminate: no squamous grade


def p_abnormal8(probs8):
    """P(abnormal) from 8-leaf softmax probs (N,8) -> (N,): mass on abnormal squamous+glandular+indeterminate."""
    probs8 = np.asarray(probs8)
    return probs8[:, sorted(_ABNORMAL8)].sum(1)


def is_nonsquamous8(leaf_idx):
    """True for the 3 new non-squamous nodes (GLANDULAR/MICROBE/INDETERMINATE) -> route-to-review."""
    return int(leaf_idx) in (GLANDULAR8, MICROBE8, INDETERMINATE8)


def manifest():
    """Serializable record of the label map (versioned, persisted for reproducibility)."""
    return dict(
        version=LABELMAP_VERSION,
        classes=C.CLASSES, normal_idx=C.NORMAL_IDX, abnormal_idx=C.ABNORMAL_IDX,
        pred_E0={c: pred_to_e0(i) for i, c in enumerate(C.CLASSES)},
        pred_E1={c: pred_to_e1(i) for i, c in enumerate(C.CLASSES)},
        external=list(EXTERNAL_MAPS),
        note="High-grade recall is 0 by construction (no HSIL output); report excluded fraction per dataset.",
    )
