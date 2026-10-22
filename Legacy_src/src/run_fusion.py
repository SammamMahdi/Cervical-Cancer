"""run_fusion.py — the end-to-end any-modality screener (IDEAL_PIPELINE.md stages 4-7).

ONE entry point, `Screener.screen(image=?, questionnaire=?)`, that ties the two FROZEN experts
together exactly as the pipeline document specifies:

    image        --> image expert  (SIPaKMeD EfficientNet-B0 student, evaluate.load_model)   --> P_img
    questionnaire --> tabular expert (calibrated UCI head, tabular.load_expert)               --> P_tab
    (either / or / both)   --> calibrate + prior-shift re-anchor to ONE deployment prevalence tau
                           --> asymmetric cascade + agreement/disagreement gate (fusion.predict)
                           --> {RELEASE, DEFER, FLAG}

Design choices are read from the experts' OWN data (no paired cohort needed):
  * deployment prevalence tau         : declared (default 0.10);
  * image operating threshold op_thr  : Youden's J on the image expert's anchored SIPaKMeD-val scores;
  * reliabilities (conflict weighting) : from each expert's own held-out AUROC (external for the image);
  * priors pi_k                        : SIPaKMeD abnormal-cell rate; the UCI target base rate.

Only depends on legacy_src/src modules: config, fusion, tabular, evaluate, dataset.
Run:  python run_fusion.py        (from legacy_src/src)
"""
import os
import json
import numpy as np
from sklearn.metrics import roc_curve

import config as C
import fusion as F
import tabular as TAB


class Screener:
    def __init__(self, target="Biopsy", deploy_prev=0.10, img_prior=0.40,
                 boundary_margin=0.10, conflict_flag=0.45):
        self.target = target
        self.deploy_prev = float(deploy_prev)
        self.img_prior = float(img_prior)
        self.boundary_margin = float(boundary_margin)
        self.conflict_flag = float(conflict_flag)

        # --- tabular expert (already trained + calibrated; joblib on disk) ---
        self.tab_expert = TAB.load_expert(target)
        tab_prior = float(self.tab_expert["metrics"]["base_rate"])
        self.priors = dict(img=self.img_prior, tab=tab_prior)
        self.reliabilities = dict(
            img=self._image_reliability(),
            tab=float(self.tab_expert["metrics"].get("reliability", 0.0)))

        # --- image operating threshold from the image expert's OWN val set ---
        self.op_thr = self._fit_op_thr()
        self._img = None            # lazy-loaded torch model (only when an image is actually screened)

    # ---------------- design choices read from the experts' own data ---------------- #
    def _image_val(self):
        """(y_bin, p_abn) on the SIPaKMeD val split, from the saved student predictions."""
        p = os.path.join(C.METRICDIR, "predictions_student.json")
        if not os.path.isfile(p):
            return None, None
        d = json.load(open(p))
        y = np.asarray(d["y_true"]); probs = np.asarray(d["probs"])
        abn = [C.CLASSES.index("Koilocytotic"), C.CLASSES.index("Dyskeratotic")]
        return np.isin(y, abn).astype(int), probs[:, abn].sum(1)

    def _image_reliability(self):
        """Honest EXTERNAL AUROC -> reliability in [0,1] (feeds the conflict flag only)."""
        p = os.path.join(C.METRICDIR, "cross_herlev.json")
        au = json.load(open(p)).get("auroc") if os.path.isfile(p) else 0.73
        return TAB.reliability_from_auroc(au)

    def _fit_op_thr(self):
        """Operating threshold = Youden's J on the image expert's ANCHORED val scores."""
        y_bin, p_abn = self._image_val()
        if y_bin is None or len(set(y_bin)) < 2:
            return 0.5
        anchored = np.array([F.prior_shift(float(p), self.img_prior, self.deploy_prev) for p in p_abn])
        fpr, tpr, thr = roc_curve(y_bin, anchored)
        return float(np.clip(thr[np.argmax(tpr - fpr)], 0.02, 0.98))

    # ---------------- expert inference ---------------- #
    def p_tabular(self, questionnaire_row):
        """P_tab(positive) for one patient (a 1-row DataFrame aligned to the expert's features)."""
        return float(TAB.p_positive(self.tab_expert, questionnaire_row)[0])

    def p_image(self, image_tensor):
        """P_img(abnormal) for one preprocessed image tensor (C,H,W) or a batch (N,C,H,W)."""
        import torch
        import evaluate as EV
        if self._img is None:
            self._img = EV.load_model("student")
        x = image_tensor if image_tensor.ndim == 4 else image_tensor.unsqueeze(0)
        _, probs = EV.predict(self._img, [(x, torch.zeros(len(x)))])
        return float(probs[0, EV.ABNORMAL_IDX].sum())

    # ---------------- the one entry point ---------------- #
    def screen(self, image=None, questionnaire=None, p_img=None, p_tab=None, mode="cascade"):
        """Run whichever modality is present and combine per IDEAL_PIPELINE.md §5.

        Pass raw inputs (`image`, `questionnaire`) OR pre-computed expert scores (`p_img`, `p_tab`).
        Returns fusion.predict's dict, plus the raw expert probabilities used.
        """
        if p_img is None and image is not None:
            p_img = self.p_image(image)
        if p_tab is None and questionnaire is not None:
            p_tab = self.p_tabular(questionnaire)
        inputs = {}
        if p_img is not None:
            inputs["p_img"] = float(p_img)
        if p_tab is not None:
            inputs["p_tab"] = float(p_tab)
        r = F.predict(inputs, self.priors, self.deploy_prev, mode=mode,
                      reliabilities=self.reliabilities, op_thr=self.op_thr,
                      boundary_margin=self.boundary_margin, conflict_flag=self.conflict_flag)
        r["raw"] = dict(p_img=p_img, p_tab=p_tab)
        r["target"] = self.target
        return r


# --------------------------------------------------------------------------- #
#  demo: run all three routes on REAL frozen experts
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    C.set_seed()
    s = Screener(target="Biopsy")
    print(f"Screener[{s.target}]  deploy_prev={s.deploy_prev}  op_thr={s.op_thr:.3f}")
    print(f"  priors={s.priors}")
    print(f"  reliabilities: img={s.reliabilities['img']:.2f}  tab={s.reliabilities['tab']:.2f}")

    def show(tag, r):
        raw = r["raw"]
        ri = "-" if raw["p_img"] is None else f"{raw['p_img']:.3f}"
        rt = "-" if raw["p_tab"] is None else f"{raw['p_tab']:.3f}"
        k = "-" if r["conflict"] is None else f"{r['conflict']:.3f}"
        print(f"  {tag:26s} p_img={ri} p_tab={rt} -> P={r['p_fused']:.3f} "
              f"[{r['route']:12s} {r['decision']:8s} conflict={k}]")

    # --- tabular-only: real UCI patients (a weak lone triage -> should DEFER) ---
    X, _, _ = TAB.load_uci()
    print("\n-- tabular-only (real UCI patients) --")
    for i in range(3):
        show(f"patient {i}", s.screen(questionnaire=X.iloc[[i]]))

    # --- image-only + both: real SIPaKMeD cells (guarded: needs checkpoint + data + torch) ---
    try:
        from dataset import make_loaders
        _, val_ld, _ = make_loaders(batch_size=8)
        xb, _ = next(iter(val_ld))
        print("\n-- image-only (real SIPaKMeD val cells) --")
        for i in range(3):
            show(f"cell {i}", s.screen(image=xb[i]))
        print("\n-- BOTH present (arbitrary cell + arbitrary patient; UNPAIRED demo, NOT a joint claim) --")
        show("cell0 + patient0", s.screen(image=xb[0], questionnaire=X.iloc[[0]]))
        show("cell1 + patient1", s.screen(image=xb[1], questionnaire=X.iloc[[1]]))
    except Exception as e:
        print(f"\n[image routes SKIP -- {type(e).__name__}: {e}]")

    print("\ndone.")
