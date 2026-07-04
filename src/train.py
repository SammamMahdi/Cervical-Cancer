"""New-pipeline training: EfficientNet-B4 teacher (two-stage) -> fixed-KD EfficientNet-B0 student,
on the LEAKAGE-SAFE cluster-grouped split (U1), 224px, AdamW + label smoothing, optional focal /
class weights (A6). Resumable (best+last checkpoints, crash-safe JSON log, train/val curves).

The 10% TEST split is SEALED here — training and early-stopping use train + val only.

Usage:
    python train.py                 # full run (auto-resume); teacher two-stage -> distilled student
    python train.py --smoke         # 2-epoch sanity check (isolated checkpoints)
    python train.py --fresh         # restart from scratch
    python train.py --no-distill    # train the student from scratch (A2 arm)
    python train.py --imbalance focal     # A6 arm   |   --aug geometric  (A5 arm)
    python train.py --student-backbone mobilenet_v3_large   # A8 arm
"""
import os, glob, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

import config as C
import runlog
from data.dataset import make_loaders, class_weights
from models.backbones import build, count_params
from models.kd import DistillLoss, BuggyDistillLoss, FocalLoss


class EMA:
    """Exponential moving average of model weights. The EMA copy is what we evaluate, early-stop on,
    and save as 'best' — it smooths the (small, honest) validation curve and usually lifts accuracy.
    Tracks ALL state_dict entries (float params/buffers via EMA; int buffers like num_batches_tracked
    are copied) so the shadow loads cleanly with strict=True."""
    def __init__(self, model, decay):
        self.decay = decay; self.step = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        # bias-corrected warmup: early on the EMA tracks the model closely (so val/early-stop are
        # meaningful), ramping to the full decay -> avoids the ~1/(1-decay) "untrained EMA" window.
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return {k: v.detach().clone() for k, v in self.shadow.items()}   # defensive copy

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}
        self.step = 10 ** 6                                              # resume past warmup


def _autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(C.USE_BF16 and C.DEVICE == "cuda"))


def _make_optimizer(params, lr, optimizer="adamw", weight_decay=1e-4):
    if optimizer == "adamw": return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if optimizer == "sgd":   return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def _make_criterion(imbalance="weights", class_weight=None, label_smoothing=0.0):
    """A6 arms: 'none' | 'weights' | 'focal' (oversampling handled at the sampler level, omitted here)."""
    if imbalance == "focal":
        return FocalLoss(gamma=2.0, weight=class_weight)
    w = class_weight if imbalance == "weights" else None
    return nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)


@torch.no_grad()
def evaluate(model, loader, loss_fn=None):
    model.eval(); tot, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(C.DEVICE, non_blocking=True); y = y.to(C.DEVICE, non_blocking=True)
        with _autocast():
            logits = model(x)
            if loss_fn is not None:
                loss_sum += loss_fn(logits.float(), y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item(); tot += x.size(0)
    return (loss_sum / tot if loss_fn is not None else None), correct / tot


# ---------------- checkpoints (best + last per stage) ----------------
def _last_path(tag): return os.path.join(C.CKPTDIR, f"{tag}_last.pt")
def _best_path(tag): return os.path.join(C.CKPTDIR, f"{tag}_best.pt")


def _save_last(tag, *, epoch, model, opt, sched, history, best, wait, done, ema=None):
    torch.save({"tag": tag, "epoch": epoch, "done": done, "wait": wait,
                "model_state": model.state_dict(), "opt_state": opt.state_dict(),
                "sched_state": sched.state_dict() if sched is not None else None,
                "ema_state": ema.state_dict() if ema is not None else None,
                "history": history, "best": best,
                "rng": {"torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "numpy": np.random.get_state(), "python": __import__("random").getstate()}},
               _last_path(tag))


def _save_best(tag, *, epoch, score, state_dict):
    """Save the chosen weights as 'best' (EMA weights when EMA is on, else the raw model)."""
    torch.save({"epoch": epoch, "score": float(score), "model_state": state_dict}, _best_path(tag))


def _restore_rng(rng):
    torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in rng["cuda"]])
    np.random.set_state(rng["numpy"]); __import__("random").setstate(rng["python"])


def _eval_weights(model, ema, val_ld):
    """Evaluate on the EMA weights (swapping them in then restoring the raw training weights), or the
    raw model when EMA is off. The EMA estimate is what we monitor / early-stop / save."""
    if ema is None:
        return evaluate(model, val_ld, nn.CrossEntropyLoss())
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.copy_to(model)
    out = evaluate(model, val_ld, nn.CrossEntropyLoss())
    model.load_state_dict(backup, strict=True)        # restore raw weights for the next epoch
    return out


def fit(model, train_ld, val_ld, *, epochs, lr, optimizer="adamw", weight_decay=1e-4, patience=8,
        monitor="val_loss", loss_fn=None, distill=None, teacher=None, scheduler_cfg=None,
        tag="model", logger=None, resume=True, use_ema=False, teacher_input_size=None):
    opt = _make_optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr, optimizer, weight_decay)
    sched = None
    if scheduler_cfg:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=scheduler_cfg["factor"],
                                                           patience=scheduler_cfg["patience"], min_lr=scheduler_cfg["min_lr"])
    ema = EMA(model, C.EMA_DECAY) if use_ema else None
    hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [], "lr": [], "epoch_time": []}
    best = {"score": None, "epoch": -1}; wait = 0; start_ep = 1

    if resume and os.path.exists(_last_path(tag)):
        ck = torch.load(_last_path(tag), map_location=C.DEVICE, weights_only=False)
        if ck.get("done"):
            if os.path.exists(_best_path(tag)):
                model.load_state_dict(torch.load(_best_path(tag), map_location=C.DEVICE)["model_state"])
            tqdm.write(f"[{tag}] already complete (best {monitor}={ck['best'].get('score')}) - skipping.")
            if logger: logger.event(f"{tag}: already complete")
            return ck["history"], ck["best"]
        model.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["opt_state"])
        if sched is not None and ck["sched_state"] is not None: sched.load_state_dict(ck["sched_state"])
        if ema is not None and ck.get("ema_state") is not None: ema.load_state_dict(ck["ema_state"])
        hist = ck["history"]; best = ck["best"]; wait = ck["wait"]; start_ep = ck["epoch"] + 1
        _restore_rng(ck["rng"])
        tqdm.write(f"[{tag}] resuming at epoch {start_ep}/{epochs} (best {monitor}={best.get('score')})")
        if logger: logger.event(f"{tag}: resumed at epoch {start_ep}")

    ep = start_ep - 1                      # bind `ep` so the tail is safe if no epoch runs (resume edge)
    for ep in range(start_ep, epochs + 1):
        model.train()
        if teacher is not None: teacher.eval()
        t0 = time.time(); tr_loss, tr_corr, n = 0.0, 0, 0
        pbar = tqdm(train_ld, desc=f"[{tag}] ep {ep}/{epochs}", leave=False, ncols=100)
        for x, y in pbar:
            x = x.to(C.DEVICE, non_blocking=True); y = y.to(C.DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with _autocast():
                s_logits = model(x)
                if distill is not None:
                    # teacher may train at a higher resolution than the student -> upsample the student
                    # batch to the teacher's native size for faithful soft targets (KD allows size mismatch)
                    xt = F.interpolate(x, size=teacher_input_size, mode="bilinear", align_corners=False) \
                        if (teacher_input_size and teacher_input_size != x.shape[-1]) else x
                    with torch.no_grad(): t_logits = teacher(xt)
                    loss, _, _ = distill(s_logits, t_logits, y)
                else:
                    loss = loss_fn(s_logits, y)
            loss.backward(); opt.step()
            if ema is not None: ema.update(model)
            tr_loss += loss.item() * x.size(0); tr_corr += (s_logits.argmax(1) == y).sum().item(); n += x.size(0)
            pbar.set_postfix(loss=f"{tr_loss/n:.3f}", acc=f"{tr_corr/n:.3f}")

        vl, va = _eval_weights(model, ema, val_ld)          # EMA weights when use_ema
        tr_loss /= n; tr_acc = tr_corr / n; lr_now = opt.param_groups[0]["lr"]; dt = time.time() - t0
        for k, v in [("loss", tr_loss), ("val_loss", vl), ("accuracy", tr_acc), ("val_accuracy", va), ("lr", lr_now), ("epoch_time", dt)]:
            hist[k].append(v)
        if sched is not None: sched.step(vl)
        score = vl if monitor == "val_loss" else va
        improved = (best["score"] is None or (score < best["score"] if monitor == "val_loss" else score > best["score"]))
        if improved:
            best = {"score": float(score), "epoch": ep}; wait = 0
            _save_best(tag, epoch=ep, score=score, state_dict=(ema.state_dict() if ema is not None else model.state_dict()))
        else:
            wait += 1
        _save_last(tag, epoch=ep, model=model, opt=opt, sched=sched, history=hist, best=best, wait=wait, done=False, ema=ema)
        runlog.dump_json(os.path.join(C.METRICDIR, f"history_{tag}.json"), hist)
        tqdm.write(f"[{tag}] ep {ep:02d}/{epochs}  loss={tr_loss:.4f} val_loss={vl:.4f}  "
                   f"acc={tr_acc:.4f} val_acc={va:.4f}  lr={lr_now:.1e}  ({dt:.0f}s){'  *best' if improved else ''}")
        if logger:
            logger.epoch(tag, dict(epoch=ep, loss=tr_loss, val_loss=vl, accuracy=tr_acc, val_accuracy=va,
                                   lr=lr_now, epoch_time=dt, improved=bool(improved)))
        if wait >= patience:
            tqdm.write(f"[{tag}] early stop at ep {ep} (best {monitor}={best['score']:.4f} @ ep{best['epoch']})")
            break

    if os.path.exists(_best_path(tag)):
        model.load_state_dict(torch.load(_best_path(tag), map_location=C.DEVICE)["model_state"])
    _save_last(tag, epoch=ep, model=model, opt=opt, sched=sched, history=hist, best=best, wait=wait, done=True, ema=ema)
    if logger: logger.event(f"{tag}: done (best {monitor}={best['score']:.4f} @ ep{best['epoch']})")
    return hist, best


def _wipe(patterns):
    for pat in patterns:
        for p in glob.glob(os.path.join(C.CKPTDIR, pat)): os.remove(p)


def main(smoke=False, fresh=False, resume=True, distill=True, imbalance="weights", aug="photometric",
         teacher_backbone=None, student_backbone=None, kd_mode="fixed", tag_suffix="", split_kind="grouped",
         stain="none", img_size=C.IMG_SIZE, teacher_img_size=C.TEACHER_IMG_SIZE, ssl_init=None):
    C.set_seed(); torch.backends.cudnn.benchmark = True
    sfx = ("_smoke" if smoke else "") + tag_suffix
    ema_on = C.USE_EMA and not smoke         # EMA needs warmup; skip for 2-epoch smoke sanity checks
    if fresh and not smoke: _wipe(["*_last.pt", "*_best.pt"]); print("--fresh: cleared checkpoints")
    if smoke: _wipe([f"*_smoke{tag_suffix}_last.pt", f"*_smoke{tag_suffix}_best.pt"]); resume = False

    run_id = runlog.now_id()
    logger = runlog.RunLogger(C.LOGDIR, run_id, config=dict(
        split="cluster-grouped 80/10/10", img_size=img_size, teacher_img_size=teacher_img_size,
        batch_size=C.BATCH_SIZE, ema=ema_on,
        teacher=C.TEACHER, student=C.STUDENT, distill=distill, kd_mode=kd_mode, imbalance=imbalance,
        aug=aug, teacher_backbone=teacher_backbone or C.TEACHER["backbone"],
        student_backbone=student_backbone or C.STUDENT["backbone"], smoke=smoke))
    print(f"Run id: {run_id}  ->  {logger.path}")

    # student loaders @ img_size (224); teacher gets its own loaders @ teacher_img_size (380, B4-native)
    # with a smaller batch (B4 is ~3x heavier). Reuse the student loaders when the resolutions match.
    train_ld, val_ld, test_ld, counts = make_loaders(aug=aug, split_kind=split_kind, stain=stain,
                                                     img_size=img_size, oversample=(imbalance == "oversample"))
    if teacher_img_size != img_size:
        t_train_ld, t_val_ld, _, _ = make_loaders(aug=aug, split_kind=split_kind, stain=stain,
                                                  img_size=teacher_img_size, batch_size=C.TEACHER_BATCH_SIZE,
                                                  oversample=(imbalance == "oversample"))
    else:
        t_train_ld, t_val_ld = train_ld, val_ld
    print(f"Split: {split_kind} | stain: {stain} | student img: {img_size} | teacher img: {teacher_img_size} "
          f"| EMA: {ema_on} | Train counts:", counts, "| total", sum(counts.values()),
          "(test:", len(test_ld.dataset), "cells)")
    cw = class_weights(counts).to(C.DEVICE)

    # ---------------- TEACHER (two-stage) ----------------
    tcfg = dict(C.TEACHER); tb = teacher_backbone or tcfg["backbone"]
    teacher = build(tb, C.NUM_CLASSES, tcfg["hidden"], tcfg["dropout"]).to(C.DEVICE)
    print(f"Teacher [{tb}] params: {count_params(teacher)[0]/1e6:.1f}M")
    ce_w = _make_criterion(imbalance, cw, tcfg.get("label_smoothing", 0.0))
    s1, s2 = tcfg["stage1"], tcfg["stage2"]
    e1, e2 = (2, 2) if smoke else (s1["epochs"], s2["epochs"])
    teacher.set_backbone_trainable(False)
    h1, _ = fit(teacher, t_train_ld, t_val_ld, epochs=e1, lr=s1["lr"], optimizer=tcfg["optimizer"],
                weight_decay=tcfg["weight_decay"], patience=s1["patience"], monitor="val_loss",
                loss_fn=ce_w, scheduler_cfg=tcfg["reduce_lr"], tag=f"teacher-s1{sfx}", logger=logger,
                resume=resume, use_ema=ema_on)
    teacher.set_backbone_trainable(True)
    h2, _ = fit(teacher, t_train_ld, t_val_ld, epochs=e2, lr=s2["lr"], optimizer=tcfg["optimizer"],
                weight_decay=tcfg["weight_decay"], patience=s2["patience"], monitor="val_loss",
                loss_fn=ce_w, scheduler_cfg=tcfg["reduce_lr"], tag=f"teacher-s2{sfx}", logger=logger,
                resume=resume, use_ema=ema_on)
    torch.save(teacher.state_dict(), os.path.join(C.CKPTDIR, f"teacher{sfx}.pt"))
    teacher_hist = {k: h1[k] + h2[k] for k in h1}
    _, t_val = evaluate(teacher, t_val_ld); print(f"== Teacher val_acc = {t_val:.4f} ==")

    # ---------------- STUDENT ----------------
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    scfg = dict(C.STUDENT); sb = student_backbone or scfg["backbone"]
    student = build(sb, C.NUM_CLASSES, scfg["hidden"], scfg["dropout"]).to(C.DEVICE)
    print(f"Student [{sb}] params: {count_params(student)[0]/1e6:.1f}M")
    if ssl_init:  # A9: warm-start the student backbone from SimCLR pretraining (selfsup.pretrain)
        from selfsup import load_ssl_init
        ssl_path = ssl_init if isinstance(ssl_init, str) else os.path.join(C.CKPTDIR, f"ssl_{sb}.pt")
        load_ssl_init(student, ssl_path)
    e_s = 2 if smoke else scfg["epochs"]
    if distill:
        KD = BuggyDistillLoss if kd_mode == "buggy" else DistillLoss
        kd = KD(temperature=scfg["temperature"], alpha=scfg["alpha"], label_smoothing=scfg.get("label_smoothing", 0.0))
        stu_hist, _ = fit(student, train_ld, val_ld, epochs=e_s, lr=scfg["lr"], optimizer=scfg["optimizer"],
                          weight_decay=scfg["weight_decay"], patience=scfg["patience"], monitor="val_accuracy",
                          distill=kd, teacher=teacher, teacher_input_size=teacher_img_size,
                          tag=f"student-kd{sfx}", logger=logger, resume=resume, use_ema=ema_on)
    else:  # A2 from-scratch arm
        ce_s = _make_criterion(imbalance, cw, scfg.get("label_smoothing", 0.0))
        stu_hist, _ = fit(student, train_ld, val_ld, epochs=e_s, lr=scfg["lr"], optimizer=scfg["optimizer"],
                          weight_decay=scfg["weight_decay"], patience=scfg["patience"], monitor="val_accuracy",
                          loss_fn=ce_s, tag=f"student-scratch{sfx}", logger=logger, resume=resume, use_ema=ema_on)
    torch.save(student.state_dict(), os.path.join(C.CKPTDIR, f"student{sfx}.pt"))
    _, s_val = evaluate(student, val_ld); print(f"== Student val_acc = {s_val:.4f} ==")

    out = dict(run_id=run_id, teacher_history=teacher_hist, student_history=stu_hist,
               teacher_val_acc=t_val, student_val_acc=s_val, class_counts=counts, smoke=smoke,
               config=dict(distill=distill, kd_mode=kd_mode, imbalance=imbalance, aug=aug))
    runlog.dump_json(os.path.join(C.METRICDIR, f"train_histories{sfx}.json"), out)
    logger.finalize(dict(teacher_val_acc=t_val, student_val_acc=s_val))
    make_curves(teacher_hist, stu_hist, sfx)
    print("Saved histories + curves + run log."); return out


def train_student(overrides=None, tag_suffix="_hpo", resume=True, fresh=True, distill=True, aug="photometric"):
    """Train ONLY the student (reusing the already-trained teacher.pt) with hyperparameter overrides.
    Used for the post-HPO refit of the single best config. Saves student<tag_suffix>.pt."""
    C.set_seed(); torch.backends.cudnn.benchmark = True
    sfx = tag_suffix
    if fresh: _wipe([f"student*{sfx}_last.pt", f"student*{sfx}_best.pt"])
    train_ld, val_ld, test_ld, counts = make_loaders(aug=aug)
    cw = class_weights(counts).to(C.DEVICE)
    scfg = {**C.STUDENT, **(overrides or {})}
    run_id = runlog.now_id()
    logger = runlog.RunLogger(C.LOGDIR, run_id, config=dict(mode="student-refit", overrides=overrides or {}, distill=distill))

    teacher = None
    if distill:
        import ckpt as CK
        if CK.latest_path("teacher") is None:
            raise FileNotFoundError("teacher checkpoint required for distillation — train the full pipeline first")
        teacher = build(C.TEACHER["backbone"], C.NUM_CLASSES, C.TEACHER["hidden"], C.TEACHER["dropout"], pretrained=False).to(C.DEVICE)
        CK.load_into(teacher, "teacher", map_location=C.DEVICE); teacher.eval()
        for p in teacher.parameters(): p.requires_grad = False

    student = build(scfg["backbone"], C.NUM_CLASSES, scfg["hidden"], scfg["dropout"]).to(C.DEVICE)
    print(f"Student refit [{scfg['backbone']}] overrides={overrides}")
    if distill:
        kd = DistillLoss(temperature=scfg["temperature"], alpha=scfg["alpha"], label_smoothing=scfg.get("label_smoothing", 0.0))
        hist, _ = fit(student, train_ld, val_ld, epochs=scfg["epochs"], lr=scfg["lr"], optimizer=scfg["optimizer"],
                      weight_decay=scfg["weight_decay"], patience=scfg["patience"], monitor="val_accuracy",
                      distill=kd, teacher=teacher, teacher_input_size=C.TEACHER_IMG_SIZE,
                      tag=f"student-kd{sfx}", logger=logger, resume=resume, use_ema=C.USE_EMA)
    else:
        ce = _make_criterion("weights", cw, scfg.get("label_smoothing", 0.0))
        hist, _ = fit(student, train_ld, val_ld, epochs=scfg["epochs"], lr=scfg["lr"], optimizer=scfg["optimizer"],
                      weight_decay=scfg["weight_decay"], patience=scfg["patience"], monitor="val_accuracy",
                      loss_fn=ce, tag=f"student-scratch{sfx}", logger=logger, resume=resume, use_ema=C.USE_EMA)
    torch.save(student.state_dict(), os.path.join(C.CKPTDIR, f"student{sfx}.pt"))
    _, s_val = evaluate(student, val_ld); print(f"== Student refit val_acc = {s_val:.4f} ==")
    runlog.dump_json(os.path.join(C.METRICDIR, f"train_histories{sfx}.json"),
                     dict(run_id=run_id, student_history=hist, student_val_acc=s_val, overrides=overrides or {}))
    logger.finalize(dict(student_val_acc=s_val))
    import plotting as P
    P.training_curves(hist, os.path.join(C.FIGDIR, f"student_training_curves{sfx}"), title="Student refit (best HPO config)")
    return s_val


def make_curves(teacher_hist, stu_hist, sfx=""):
    import plotting as P
    P.training_curves(teacher_hist, os.path.join(C.FIGDIR, f"teacher_training_curves{sfx}"), title="Teacher (B4) training")
    P.training_curves(stu_hist, os.path.join(C.FIGDIR, f"student_training_curves{sfx}"), title="Student (B0, distilled) training")


def curves_from_json(sfx=""):
    d = runlog.load_json(os.path.join(C.METRICDIR, f"train_histories{sfx}.json"))
    if d is None: print("no train_histories json"); return
    make_curves(d["teacher_history"], d["student_history"], sfx); print("Rebuilt curves from JSON.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--no-distill", dest="distill", action="store_false")
    ap.add_argument("--kd-mode", choices=["fixed", "buggy"], default="fixed")
    ap.add_argument("--imbalance", choices=["none", "weights", "focal"], default="weights")
    ap.add_argument("--aug", choices=["none", "geometric", "photometric"], default="photometric")
    ap.add_argument("--teacher-backbone", default=None)
    ap.add_argument("--student-backbone", default=None)
    main(**vars(ap.parse_args()))
