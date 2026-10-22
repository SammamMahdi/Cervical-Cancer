"""NHANES loader for a leakage-safe cervical HPV-risk tabular expert — full questionnaire edition.

Builds ONE analytic table by pooling NHANES cycles (2003-2016): the ACTUAL patient questionnaire
(demographics + the Sexual Behaviour Questionnaire + reproductive + smoking + HPV vaccination) plus one
blood biomarker (HSV-2), predicting a LAB LABEL (high-risk HPV DNA positivity on the self-collected
vaginal swab).

  * LABEL   : high_risk_hpv = 1 if ANY of the 14 oncogenic types {16,18,31,33,35,39,45,51,52,56,58,59,66,68}
              is positive on the Roche Linear Array; inadequate/absent samples dropped.
  * FEATURES: every item in QUESTIONNAIRE below (question text + answer coding shown). NHANES renames
              variables across cycles, so each item is resolved by its STABLE QUESTION TEXT (label) with
              name fall-backs. Genital warts is included but FLAGGED HPV-adjacent (warts = low-risk HPV
              6/11; target = high-risk types) and reported with/without via ADJACENT.
  * EXCLUDED: SXQ753 "Ever told by a doctor you had HPV" — that is a self-report of the TARGET itself
              (the leakage variable used by the published 0.95-AUC papers). Never used as a feature.

Files are downloaded (and cached) from the CDC public site. Run:  python nhanes_data.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
import config as C

NHANES_DIR = os.path.join(C.PROJECT, "data", "nhanes")
os.makedirs(NHANES_DIR, exist_ok=True)
_S = requests.Session(); _S.headers.update({"User-Agent": "Mozilla/5.0 research"})

RESULTS = os.path.join(C.PROJECT, "Tabular results", "NHANES")
FIG = os.path.join(RESULTS, "figures"); MET = os.path.join(RESULTS, "metrics")
os.makedirs(FIG, exist_ok=True); os.makedirs(MET, exist_ok=True)

# cycle -> (start-year, file suffix); HPV swab file is HPVSWR_{suf} except 2003-04 (L37SWR_C)
CYCLES = {"2003-2004": ("2003", "C"), "2005-2006": ("2005", "D"), "2007-2008": ("2007", "E"),
          "2009-2010": ("2009", "F"), "2011-2012": ("2011", "G"), "2013-2014": ("2013", "H"),
          "2015-2016": ("2015", "I")}
HPV_FILE = {"2003-2004": "L37SWR_C"}
ALL_CYCLES = tuple(CYCLES)
HIGH_RISK = [16, 18, 31, 33, 35, 39, 45, 51, 52, 56, 58, 59, 66, 68]

# ---- THE QUESTIONNAIRE ---------------------------------------------------------------------------
# (section, feature, file, kind, label_pattern, name_fallbacks, question, answers)
#   kind: num/age/count = numeric | bin = yes-no | cat = nominal | sero = 1pos/2neg lab | derived
QUESTIONNAIRE = [
    # -- Demographics --
    ("Demographics", "age", "DEMO", "num", None, ["RIDAGEYR"],
     "What is your age?", "age in years (18-59)"),
    ("Demographics", "race", "DEMO", "cat", None, ["RIDRETH1"],
     "Race / Hispanic origin", "1=MexicanAmerican 2=OtherHispanic 3=White 4=Black 5=Other"),
    ("Demographics", "education", "DEMO", "cat", None, ["DMDEDUC2"],
     "Highest education completed", "1=<9th 2=9-11th 3=HS/GED 4=SomeCollege 5=CollegeGrad+"),
    ("Demographics", "marital", "DEMO", "cat", None, ["DMDMARTL"],
     "Marital status", "1=Married 2=Widowed 3=Divorced 4=Separated 5=NeverMarried 6=LivingWithPartner"),
    ("Demographics", "poverty_ratio", "DEMO", "num", None, ["INDFMPIR"],
     "Ratio of family income to the poverty threshold", "0.0-5.0 (higher = wealthier)"),
    # -- Sexual behaviour --
    ("Sexual behaviour", "ever_sex", "SXQ", "bin", "ever had vaginal, anal, or oral sex", ["SXD021", "SXQ021"],
     "Ever had vaginal, anal, or oral sex?", "1=Yes 2=No"),
    ("Sexual behaviour", "age_first_sex", "SXQ", "age", "how old when first had sex", ["SXD031"],
     "How old were you when you had sex for the first time?", "age in years"),
    ("Sexual behaviour", "age_first_oral", "SXQ", "age", "how old when first had oral sex", ["SXD621"],
     "How old were you when you first performed oral sex?", "age in years (asked 2009+)"),
    ("Sexual behaviour", "ever_oral_man", "SXQ", "bin", "ever performed oral sex on a man", ["SXQ703"],
     "Ever performed oral sex on a man?", "1=Yes 2=No (asked 2009+)"),
    ("Sexual behaviour", "ever_anal_man", "SXQ", "bin", "ever had anal sex with a man", ["SXQ706"],
     "Ever had anal sex with a man?", "1=Yes 2=No (asked 2009+)"),
    ("Sexual behaviour", "n_partners_life", "SXQ", "count", "# male sex partners/lifetime", ["SXD101", "SXQ101"],
     "How many male sex partners have you had in your lifetime?", "count"),
    ("Sexual behaviour", "n_partners_year", "SXQ", "count", "# male sex partners/year", ["SXD450"],
     "How many male sex partners in the past 12 months?", "count"),
    ("Sexual behaviour", "n_vaginal_partners_life", "SXQ", "count", "# male vaginal sex partners/lifetime", ["SXQ724"],
     "How many male vaginal-sex partners in your lifetime?", "count (asked 2009+)"),
    ("Sexual behaviour", "n_oral_partners_life", "SXQ", "count", "# male oral sex partners/lifetime", ["SXQ624"],
     "How many male oral-sex partners in your lifetime?", "count (asked 2009+)"),
    ("Sexual behaviour", "condomless_year", "SXQ", "count", "without condom", ["SXQ251"],
     "In the past year, how many times did you have sex without a condom?", "count"),
    ("Sexual behaviour", "new_partner_year", "SXQ", "bin", "had sex with new partner", ["SXQ648"],
     "Have you had sex with a new partner in the past year?", "1=Yes 2=No (asked 2009+)"),
    ("Sexual behaviour", "orientation", "SXQ", "cat", "sexual orientation (female)", ["SXQ294"],
     "Which best describes your sexual orientation?", "1=Heterosexual 2=Homosexual 3=Bisexual 4=Other"),
    # -- STI history (doctor-diagnosed) --
    ("STI history", "herpes_hx", "SXQ", "bin", "genital herpes", ["SXQ260"],
     "Has a doctor ever told you that you had genital herpes?", "1=Yes 2=No"),
    ("STI history", "gonorrhea_hx", "SXQ", "bin", "had gonorrhea", ["SXQ270"],
     "Has a doctor ever told you that you had gonorrhea?", "1=Yes 2=No"),
    ("STI history", "chlamydia_hx", "SXQ", "bin", "had chlamydia", ["SXQ272"],
     "Has a doctor ever told you that you had chlamydia?", "1=Yes 2=No"),
    ("STI history", "genital_warts_hx", "SXQ", "bin", "genital warts", ["SXQ265"],
     "Has a doctor ever told you that you had genital warts?  [HPV-ADJACENT — reported with/without]",
     "1=Yes 2=No  (warts = low-risk HPV 6/11; target = high-risk types)"),
    # -- Reproductive --
    ("Reproductive", "age_menarche", "RHQ", "age", None, ["RHQ010"],
     "How old were you when your periods started?", "age in years"),
    ("Reproductive", "ever_pregnant", "RHQ", "bin", None, ["RHQ131"],
     "Have you ever been pregnant?", "1=Yes 2=No"),
    ("Reproductive", "n_pregnancies", "RHQ", "count", None, ["RHQ160", "RHD160"],
     "How many times have you been pregnant?", "count (0 if never pregnant)"),
    # -- Smoking --
    ("Smoking", "smoked_100", "SMQ", "bin", None, ["SMQ020"],
     "Have you smoked at least 100 cigarettes in your entire life?", "1=Yes 2=No"),
    ("Smoking", "smoking_status", "SMQ", "derived", None, [],
     "Current smoking status (derived from SMQ020 + SMQ040)", "0=Never 1=Former 2=Current"),
    # -- Vaccination --
    ("Vaccination", "hpv_vaccine", "IMQ", "bin", "hpv", ["IMQ040", "IMQ060"],
     "Have you ever received the HPV vaccine (Cervarix or Gardasil)?",
     "1=Yes 2=No  (question introduced 2007; set 2=No for pre-2007 cycles)"),
    # -- Blood biomarker (not a questionnaire item; a shared-transmission proxy, NOT HPV) --
    ("Biomarker", "hsv2", "HSV", "sero", None, ["LBXHE2"],
     "Herpes Simplex Virus type-2 antibody (laboratory blood test)",
     "1=Positive 2=Negative  (different virus; marks shared sexual-transmission risk)"),
]
ADJACENT = ["genital_warts_hx"]                          # dropped in the "clean" model variant
ENGINEERED = ["years_active", "partners_per_active_year", "pregnancies_per_active_year"]
EXCLUDED = {"SXQ753": "Ever told by a doctor you had HPV — a self-report of the TARGET (leakage)."}

FILES_NEEDED = ("DEMO", "SXQ", "RHQ", "SMQ", "IMQ", "HSV")
FEATURES = [q[1] for q in QUESTIONNAIRE] + ENGINEERED
_KIND = {q[1]: q[3] for q in QUESTIONNAIRE}
NUM = [q[1] for q in QUESTIONNAIRE if q[3] in ("num", "age", "count")] + ENGINEERED
CAT = [q[1] for q in QUESTIONNAIRE if q[3] in ("cat", "bin", "sero", "derived")]


def _download(fname, year):
    path = os.path.join(NHANES_DIR, fname + ".XPT")
    if not os.path.exists(path):
        url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{fname}.XPT"
        r = _S.get(url, timeout=180)
        if r.status_code != 200 or r.content[:2] != b"HE":
            return None
        open(path, "wb").write(r.content)
    try:
        return pd.read_sas(path, format="xport")
    except Exception:
        return None


def _labels(fname, year):
    """Variable -> lowercased question text (from the XPT header, via pyreadstat)."""
    _download(fname, year)
    path = os.path.join(NHANES_DIR, fname + ".XPT")
    if not os.path.exists(path):
        return {}
    try:
        import pyreadstat
        _, meta = pyreadstat.read_xport(path, metadataonly=True)
        return {v: str(l).lower() for v, l in zip(meta.column_names, meta.column_labels)}
    except Exception:
        return {}


def _clean(s, kind):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if kind in ("cat", "bin"):
        s = s.mask(s.isin({7, 9, 77, 99}))
    elif kind == "sero":
        s = s.mask(s.isin({3, 7, 9}))                    # 3 = indeterminate
    elif kind == "age":
        s = s.mask(s.isin({77, 99, 777, 999})); s = s.mask((s < 8) | (s > 60))
    elif kind == "count":
        s = s.mask(s.isin({7777, 9999, 77777, 99999})); s = s.mask(s > 500)
    return s


def _resolve(df, labels, fallbacks, label_pat):
    for c in fallbacks:
        if c in df.columns:
            return c
    if label_pat:
        for v, l in labels.items():
            if label_pat in l and v in df.columns and v not in EXCLUDED:
                return v
    return None


def _smoking_status(smq):
    if "SMQ020" not in smq.columns:
        return None
    s20 = pd.to_numeric(smq["SMQ020"], errors="coerce")
    s40 = pd.to_numeric(smq.get("SMQ040"), errors="coerce") if "SMQ040" in smq.columns else pd.Series(np.nan, index=smq.index)
    st = pd.Series(np.nan, index=smq.index)
    st[s20 == 2] = 0; st[(s20 == 1) & (s40 == 3)] = 1; st[(s20 == 1) & (s40.isin([1, 2]))] = 2
    return pd.DataFrame({"SEQN": smq["SEQN"], "smoking_status": st})


def _hpv_label(hpv, kind="high_risk"):
    """Positive = a target genotype tests ==1; inadequate/absent -> NaN (dropped, never coerced negative).
    kind: 'high_risk' (14 oncogenic) · 'any' (any type) · '16_18' · 'other_hr' (high-risk except 16/18).
    Adequacy is assay-level (over ALL typed columns) so subgroup endpoints share one denominator."""
    all_cols = [c for c in hpv.columns if c.startswith("LBDR") and c[4:].isdigit()]
    if not all_cols:
        return None
    allsub = hpv[all_cols].apply(pd.to_numeric, errors="coerce")
    adequate = allsub.notna().any(axis=1) & (allsub != 3).any(axis=1)
    if kind == "any":
        types = [int(c[4:]) for c in all_cols]
    elif kind == "16_18":
        types = [16, 18]
    elif kind == "other_hr":
        types = [t for t in HIGH_RISK if t not in (16, 18)]
    else:
        types = HIGH_RISK
    cols = [f"LBDR{t}" for t in types if f"LBDR{t}" in hpv.columns]
    pos = (hpv[cols].apply(pd.to_numeric, errors="coerce") == 1).any(axis=1)
    y = pos.astype(float); y[~adequate] = np.nan
    return pd.DataFrame({"SEQN": hpv["SEQN"], "high_risk_hpv": y})


def hpv_swab(cyc, kind="high_risk"):
    year, suf = CYCLES[cyc]
    hpv = _download(HPV_FILE.get(cyc, f"HPVSWR_{suf}"), year)
    return _hpv_label(hpv, kind) if hpv is not None else None


def _add_engineered(X):
    ya = (X["age"] - X["age_first_sex"]).clip(lower=0)
    X["years_active"] = ya
    X["partners_per_active_year"] = X["n_partners_life"] / (ya + 1)
    X["pregnancies_per_active_year"] = X["n_pregnancies"] / (ya + 1)
    return X


def load(cycles=ALL_CYCLES, target="high_risk", engineered=True, verbose=True):
    """Return (X, y, feature_names, meta). Pools cycles; female HPV-swab subjects only.
    target: 'high_risk' (primary) · 'any' · '16_18' · 'other_hr' (genotype-subgroup diagnostics)."""
    frames = []
    for cyc in cycles:
        year, suf = CYCLES[cyc]
        lab = hpv_swab(cyc, target)
        if lab is None:
            if verbose:
                print(f"  [{cyc}] HPV swab unavailable -> skip")
            continue
        feat, labs = {}, {}
        for f in FILES_NEEDED:
            df = _download(f"{f}_{suf}", year)
            if df is not None:
                feat[f] = df; labs[f] = _labels(f"{f}_{suf}", year)
        row = lab.dropna(subset=["high_risk_hpv"]).copy()
        for _, name, fkey, kind, pat, fb, _, _ in QUESTIONNAIRE:
            if kind == "derived":
                continue
            src = _resolve(feat[fkey], labs.get(fkey, {}), fb, pat) if fkey in feat else None
            if src is not None:
                m = feat[fkey][["SEQN", src]].drop_duplicates("SEQN")
                row = row.merge(m, on="SEQN", how="left"); row[name] = _clean(row.pop(src), kind)
            else:
                row[name] = np.nan
        if "SMQ" in feat and (smk := _smoking_status(feat["SMQ"])) is not None:
            row = row.merge(smk, on="SEQN", how="left")
        else:
            row["smoking_status"] = np.nan
        row.loc[row["ever_pregnant"] == 2, "n_pregnancies"] = row.loc[row["ever_pregnant"] == 2, "n_pregnancies"].fillna(0)
        if cyc in ("2003-2004", "2005-2006"):
            row["hpv_vaccine"] = 2.0                      # vaccine did not exist -> definitionally unvaccinated
        row["cycle"] = cyc
        frames.append(row)
        if verbose:
            print(f"  [{cyc}] n={len(row):4d}  HPV+ rate={row['high_risk_hpv'].mean():.3f}")
    if not frames:
        raise RuntimeError("no NHANES cycles could be assembled")
    data = pd.concat(frames, ignore_index=True)
    X = data[[q[1] for q in QUESTIONNAIRE]].copy()
    if engineered:
        X = _add_engineered(X)
    feat_names = list(X.columns)
    y = data["high_risk_hpv"].astype(int).to_numpy()
    meta = dict(n=len(data), n_pos=int(y.sum()), pos_rate=float(y.mean()), target=target,
                cycles=[c for c in cycles if c in set(data["cycle"])], n_features=len(feat_names))
    X.attrs["cycle"] = data["cycle"].to_numpy()
    return X, y, feat_names, meta


def write_questionnaire_md(path=None):
    """Emit the questionnaire as an actual Q&A document, with per-item coverage in the cohort."""
    X, y, feat, meta = load(verbose=False)
    cov = {c: 100 * (1 - X[c].isna().mean()) for c in X.columns}
    path = path or os.path.join(RESULTS, "QUESTIONNAIRE.md")
    lines = ["# NHANES Cervical HPV-Risk Questionnaire (female respondents)",
             "",
             f"Administered items pooled across NHANES 2003-2016. **n = {meta['n']:,}** women with a valid "
             f"high-risk-HPV swab result; **{100*meta['pos_rate']:.1f}%** positive. Each item shows its "
             "question, answer coding, source NHANES variable, and % answered in this cohort.",
             "",
             "> **Outcome (predicted, not asked):** high-risk HPV DNA on the self-collected vaginal swab "
             "(any oncogenic type 16/18/31/33/35/39/45/51/52/56/58/59/66/68).",
             "",
             "> **Deliberately EXCLUDED:** `SXQ753` *\"Ever told by a doctor you had HPV\"* — a self-report of "
             "the outcome itself; using it is the target-leakage behind the published 0.90-0.95 AUC claims.",
             ""]
    sec = None; i = 0
    for section, name, fkey, kind, pat, fb, q, ans in QUESTIONNAIRE:
        if section != sec:
            sec = section; lines += ["", f"## {section}", ""]
        i += 1
        adj = "  ⚠️_HPV-adjacent_" if name in ADJACENT else ""
        lines.append(f"**Q{i}. {q}**{adj}  ")
        lines.append(f"*Answers:* {ans}  ")
        lines.append(f"*Variable:* `{name}` ({fkey}) · *answered:* {cov.get(name,0):.0f}%")
        lines.append("")
    lines += ["## Engineered (derived from the answers above)", ""]
    for e in ENGINEERED:
        lines.append(f"- `{e}`")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    return path


if __name__ == "__main__":
    X, y, feat, meta = load()
    print("\n=== NHANES full-questionnaire table ===")
    print(f"  n={meta['n']}  HPV+={meta['n_pos']} ({100*meta['pos_rate']:.1f}%)  "
          f"features={meta['n_features']}  cycles={len(meta['cycles'])}")
    print("\n  coverage per item:")
    for c in feat:
        print(f"    {c:26s} {100*(1-X[c].isna().mean()):5.1f}% answered")
    import json
    json.dump(meta, open(os.path.join(MET, "nhanes_meta.json"), "w"), indent=2)
    p = write_questionnaire_md()
    print(f"\n  saved -> metrics/nhanes_meta.json  and  {os.path.relpath(p, C.PROJECT)}")
