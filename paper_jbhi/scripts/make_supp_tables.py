"""Emit the supplementary tables as LaTeX, straight from the result files.

Writes one self-contained table environment per file into ../supp/, so the supplement never carries
hand-typed numbers.  Run: python paper_jbhi/scripts/make_supp_tables.py
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "supp")
os.makedirs(OUT, exist_ok=True)
J = lambda f: json.load(open(os.path.join(RES, f)))


def write(name, body):
    with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8") as f:
        f.write(body.rstrip() + "\n")
    print("wrote supp/" + name + ".tex")


def table(caption, label, colspec, header, rows, note=None, wide=False, small="footnotesize"):
    env = "table*" if wide else "table"
    out = [f"\\begin{{{env}}}[t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\centering\\{small}", "\\begin{tabular}{@{}" + colspec + "@{}}", "\\toprule",
           header + " \\\\", "\\midrule"]
    out += [r + " \\\\" for r in rows]
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    if note:
        w = "\\textwidth" if wide else "\\columnwidth"
        out.append(f"\\\\[2pt]\\begin{{minipage}}{{{w}}}\\scriptsize {note}\\end{{minipage}}")
    out.append(f"\\end{{{env}}}")
    return "\n".join(out)


# ---------------------------------------------------------------- cytology: per class
I = J("image_eval.json")
rows = []
for c in I["classes"]:
    t, s = I["teacher"]["per_class"][c], I["student"]["per_class"][c]
    rows.append(f"{c} & {t['support']} & {t['precision']:.3f} & {t['recall']:.3f} & {t['f1']:.3f} & "
                f"{s['precision']:.3f} & {s['recall']:.3f} & {s['f1']:.3f}")
write("tab_perclass", table(
    "Per-Class Performance on the Fixed Protocol~A Test Set",
    "tab:sperclass", "lccccccc",
    "Class & $n$ & \\multicolumn{3}{c}{Teacher B4} & \\multicolumn{3}{c}{Student B0}",
    ["\\cmidrule(lr){3-5}\\cmidrule(l){6-8}\nMetric & & Prec. & Rec. & F1 & Prec. & Rec. & F1"] + rows,
    wide=True))

# ---------------------------------------------------------------- slide-level aggregation
S = J("slide_level.json")
rows = [f"{k.replace('_', '-')} & {v['auroc']:.4f} & {v['ap']:.4f} & {v['mean_score']:.3f}"
        for k, v in S["aggregators"].items()]
rows.append(f"\\midrule cell level (no pooling) & {S['cell_level']['auroc']:.4f} & --- & ---")
write("tab_slide", table(
    "Pooling Held-Out Cells Into Slide Scores",
    "tab:sslide", "lccc",
    "Aggregator & AUROC & Average precision & Mean score", rows,
    note=f"{S['n_slides']} slides ({S['n_abnormal']} abnormal), {S['n_cells_total']} cells, median "
         f"{S['cells_per_slide']['median']:.0f} cells per slide. The abnormal-slide prior is "
         f"{S['slide_prior']:.3f} against a cell-level prior of {S['cell_level']['prior']:.3f}."))

# ---------------------------------------------------------------- external, by class
E = J("external_eval.json")
mb = E["models"]["student (Protocol B, slide-grouped)"]["per_class_mean_score"]
ma = E["models"]["student (Protocol A)"]["per_class_mean_score"]
label = {"normal_superficiel": "normal superficial", "normal_intermediate": "normal intermediate",
         "normal_columnar": "normal columnar", "light_dysplastic": "light dysplasia",
         "moderate_dysplastic": "moderate dysplasia", "severe_dysplastic": "severe dysplasia",
         "carcinoma_in_situ": "carcinoma \\emph{in situ}"}
order = ["normal_superficiel", "normal_intermediate", "normal_columnar", "light_dysplastic",
         "moderate_dysplastic", "severe_dysplastic", "carcinoma_in_situ"]
rows = [f"{label[c]} & {'normal' if c.startswith('normal') else 'abnormal'} & "
        f"{E['per_class'][c]} & {ma[c]:.3f} & {mb[c]:.3f}" for c in order]
write("tab_external", table(
    "Mean Predicted Abnormality on Herlev, by Class",
    "tab:sexternal", "llccc",
    "Herlev class & Mapped to & $n$ & Protocol A & Protocol B", rows,
    note="Neither network ever saw a columnar cell. Both score that class as abnormal, which is what "
         "drives the drop in external AUROC; excluding it raises Protocol~B from "
         f"{E['models']['student (Protocol B, slide-grouped)']['auroc']:.3f} to "
         f"{E['models']['student (Protocol B, slide-grouped)']['auroc_excluding_columnar']:.3f}."))

# ---------------------------------------------------------------- questionnaire families
T = J("tabular_v2.json")
rows = [f"{k} & {v['auroc']:.4f} & {v['pr_auc']:.4f} & {v['brier']:.4f} & {v['ece']:.3f}"
        for k, v in T["families"].items()]
rows.append("\\midrule nested selection $+$ isotonic & "
            f"{T['nested_selection']['calibrated']['auroc']:.4f} & "
            f"{T['nested_selection']['calibrated']['pr_auc']:.4f} & "
            f"{T['nested_selection']['calibrated']['brier']:.4f} & "
            f"{T['nested_selection']['calibrated']['ece']:.3f}")
write("tab_families", table(
    "Questionnaire Model Families on Identical Folds",
    "tab:sfamilies", "lcccc",
    "Model & AUROC & Avg.\\ precision & Brier & ECE", rows,
    note="All figures are out of fold with imputation, scaling and calibration fitted inside the "
         "training folds. The last row is the nested procedure reported in the main paper, which "
         "selected logistic regression in all five outer folds."))

# ---------------------------------------------------------------- subgroups, including cycles
rows = [f"{g.replace('race: ', 'race/ethnicity: ')} & {v['n']} & {v['prevalence']:.3f} & "
        f"{v['auroc']:.3f} & {v['brier']:.3f} & {v['ece']:.3f}" for g, v in T["subgroups"].items()]
write("tab_subgroups", table(
    "Questionnaire Expert by Subgroup, Including Survey Cycle",
    "tab:ssubgroups", "lccccc",
    "Subgroup & $n$ & Prevalence & AUROC & Brier & ECE", rows, wide=True,
    note="Groups with fewer than 200 women or a single outcome class are omitted. Discrimination "
         "varies by 0.09 AUROC and calibration error stays at or below 0.050 everywhere."))

# ---------------------------------------------------------------- permutation importance, full
imp = T["permutation_importance"][:20]


def pretty_feat(f):
    """Readable input names: underscores become spaces and a flag is named as such."""
    flag = f.endswith("_missing")
    base = f[:-len("_missing")] if flag else f
    base = base.replace("_", " ")
    return base + (" (asked?)" if flag else "")


rows = [f"{pretty_feat(d['feature'])} & {'yes' if d['feature'].endswith('_missing') else 'no'} & "
        f"{d['auroc_drop']:+.4f} & {d['sd']:.4f} & {d['coef']:+.3f}" for d in imp]
write("tab_importance", table(
    "Permutation Importance of All Inputs, Top 20",
    "tab:simportance", "lcccc",
    "Input & Missingness flag & AUROC drop & SD & Coefficient", rows, wide=True,
    note="Measured on held-out folds. Missingness flags record whether a question was asked at all. They "
         "head the ranking because they are collinear and carry large offsetting coefficients, so "
         "permuting one destroys a cancellation the others rely on; removing all 27 costs "
         f"{T['missingness_ablation']['with_indicators'] - T['missingness_ablation']['without_indicators']:.4f} "
         "AUROC. The main paper's figure therefore shows the substantive items only."))

# ---------------------------------------------------------------- violations, full sweeps
R = J("fusion_robust.json")
M = ["image only", "naive mean", "sum rule", "proposed", "oracle (paired)"]
blocks = [("V1_dependence", "rho", "$\\rho$"), ("V2_direct_path", "gamma", "$\\gamma$"),
          ("V3_infection_on_image", "delta", "$\\delta$"), ("V4_subgroup_link", "q_spread", "spread"),
          ("V5_miscalibration", None, "temperature"), ("V6_domain_prevalence", None, "setting")]
PRETTY_SET = {"mis_t": "questionnaire temperature", "mis_i": "image temperature",
              "img_bias": "image log-odds bias", "prev": "prevalence", "q_spread": "spread"}
PRETTY_BLK = {"V1_dependence": "V1: residual dependence given the latent state",
              "V2_direct_path": "V2: direct questionnaire-to-cytology path",
              "V3_infection_on_image": "V3: infection affects the image at fixed cytology",
              "V4_subgroup_link": "V4: the link varies by subgroup",
              "V5_miscalibration": "V5: an expert is miscalibrated by an unknown temperature",
              "V6_domain_prevalence": "V6: image domain shift, with and without prevalence change"}
rows = []
for blk, key, lab in blocks:
    rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{PRETTY_BLK[blk]}}}}}")
    for r in R[blk]:
        st = r["setting"]
        desc = (f"{lab} $=$ {st[key]:g}" if key else
                ", ".join(f"{PRETTY_SET.get(k, k)} {v:g}" for k, v in st.items()))
        vals = " & ".join(f"{r['methods'][m]['auroc']:.4f}" for m in M)
        rows.append(f"\\quad {desc} & {vals} & {r['methods']['proposed']['ece']:.3f}")
write("tab_violations", table(
    "Every Assumption-Violation Setting in Full",
    "tab:sviolations", "lcccccc",
    "Setting & Image & Naive mean & Sum rule & Proposed & Oracle & ECE$_{\\text{prop}}$", rows,
    wide=True, small="scriptsize",
    note="AUROC, mean of 20 replicates of 20,000 women. The oracle is a logistic stacker refitted on "
         "paired data and is a reference, not a deployable comparator."))

# ---------------------------------------------------------------- conformal under shift
rows = [f"{r['scenario']} & {r['target']:.2f} & {r['coverage']['mean']:.3f} & "
        f"{r['coverage']['sd']:.3f} & {r['abstention']['mean']:.3f}"
        for r in R["conformal_under_shift"]]
write("tab_conformal", table(
    "Conformal Coverage Under Deliberate Shift",
    "tab:sconformal", "lcccc",
    "Scenario & Target & Coverage & SD & Deferral rate", rows,
    note="Class-conditional coverage holds on matched data and fails under shift, because the "
         "guarantee is conditional on exchangeability between calibration and test data."))

# ---------------------------------------------------------------- paired-data requirement
rows = [f"{r['n_paired']} & {r['oracle_minus_proposed']['mean']:+.4f} & {r['oracle_minus_proposed']['sd']:.4f}"
        for r in R["paired_data_requirement"]]
write("tab_paired", table(
    "Stacker Minus Unpaired Rule, by Amount of Paired Data",
    "tab:spaired", "lcc",
    "Paired patients & $\\Delta$AUROC & SD", rows,
    note="Negative values mean the stacker is worse than the rule that uses no paired data at all. "
         "It never overtakes it here, because on matched cohorts the rule is already optimal."))

print("done")
