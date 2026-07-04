"""Explainability orchestrator (U8 + Axis 1-4). `import explain; explain.explain_all('student')`.

Runs every explainer on the trained model and saves a summary; each axis is independently guarded
so a missing model/lib never aborts the rest.
  Axis 1 where : CAM family (explain.cam) + pixel attributions (explain.attribution)
  Axis 2 why   : SHAP on the handcrafted-feature branch + U7 fusion (explain.features)
  Axis 3 cases : nearest-exemplar + counterfactuals (explain.exemplar)
  Axis 4 trust : conformal prediction sets (eval.conformal)  [calibration is in run.evaluate]
"""
import os
import config as C
import runlog


def explain_all(which="student", attributions=True, exemplars=True, conformal=True,
                cam_img_size=None, cam_fine_layer=False):
    summary = {}

    try:
        import explain.cam as cam
        summary["cam"] = cam.run_cam(which=which, img_size=cam_img_size, fine_layer=cam_fine_layer)
    except Exception as e:
        print("[cam]", e)

    if attributions:
        try:
            import explain.attribution as attr
            summary["attribution"] = attr.run_attributions(which=which, methods=("ig", "saliency", "occlusion"))
        except Exception as e:
            print("[attribution]", e)

    try:
        import explain.features as feat
        summary["features"] = feat.run_all()
    except Exception as e:
        print("[features]", e)

    if exemplars:
        try:
            import explain.exemplar as ex
            from run import _load_model
            from data.splits import get_split
            model = _load_model(which)
            summary["exemplar"] = ex.nearest_exemplars(model, get_split("grouped")["test"][0], k=5)
        except Exception as e:
            print("[exemplar]", e)

    if conformal:
        try:
            import eval.conformal as CF
            import eval.metrics as M
            from run import _load_model
            from data.dataset import make_loaders
            model = _load_model(which)
            _, val_ld, test_ld, _ = make_loaders()
            yv, pv = M.predict(model, val_ld); yt, pt = M.predict(model, test_ld)
            summary["conformal"] = CF.report(pv, yv, pt, yt, alpha=0.1, method="aps", mondrian=True)
        except Exception as e:
            print("[conformal]", e)

    runlog.dump_json(os.path.join(C.METRICDIR, f"explain_summary_{which}.json"), summary)
    print("Explainability summary ->", os.path.join(C.METRICDIR, f"explain_summary_{which}.json"))
    return summary
