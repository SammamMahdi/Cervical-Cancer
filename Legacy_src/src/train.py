"""Train the legacy pipeline in PyTorch: EfficientNet-B4 teacher (two-stage) -> distilled B0 student.

Faithful to the notebook (random 80/20 split, same head/hyperparameters) to reproduce ~97%,
on the RTX 5060 Ti with BF16 autocast.

Robustness / reproducibility features:
  * Resumable: every epoch writes a `<stage>_last.pt` recovery checkpoint (model + optimizer +
    scheduler + history + RNG state). Re-running resumes each stage exactly where it stopped.
  * Only TWO checkpoints are kept per stage: `<stage>_best.pt` (best val weights) and
    `<stage>_last.pt` (latest, for recovery). No per-epoch checkpoint pile-up.
  * Timestamped JSON log: results/logs/run_<timestamp>.json (all metrics + events, crash-safe).
  * Per-stage + consolidated JSON histories (results/metrics/*.json) so every diagram can be
    rebuilt from disk without re-training.

Usage:
    python train.py               # full run (auto-resumes if checkpoints exist)
    python train.py --fresh       # ignore existing checkpoints, start over
    python train.py --no-resume   # don't resume, but keep existing files
    python train.py --smoke       # 2-epoch sanity check (isolated *_smoke checkpoints)
"""
import os, glob, time, argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

import config as C
import runlog
from dataset import make_loaders, class_weights
from models import build_teacher, build_student, count_params
from kd import DistillLoss


def _autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(C.USE_BF16 and C.DEVICE == "cuda"))


@torch.no_grad()
def evaluate(model, loader, loss_fn=None):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(C.DEVICE, non_blocking=True); y = y.to(C.DEVICE, non_blocking=True)
        with _autocast():
            logits = model(x)
            if loss_fn is not None:
                loss_sum += loss_fn(logits.float(), y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item(); tot += x.size(0)
    return (loss_sum / tot if loss_fn is not None else None), correct / tot


# ---------------- checkpoint helpers (only best + last per stage) ----------------
def _last_path(tag): return os.path.join(C.CKPTDIR, f"{tag}_last.pt")
def _best_path(tag): return os.path.join(C.CKPTDIR, f"{tag}_best.pt")


def _save_last(tag, *, epoch, model, opt, sched, history, best, wait, done):
    """Recovery checkpoint — overwrites the single <tag>_last.pt every epoch."""
    torch.save({
        "tag": tag, "epoch": epoch, "done": done, "wait": wait,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "sched_state": sched.state_dict() if sched is not None else None,
        "history": history, "best": best,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": __import__("random").getstate(),
        },
    }, _last_path(tag))


def _save_best(tag, *, epoch, model, score):
    """Best-weights checkpoint — overwrites the single <tag>_best.pt only when val improves."""
    torch.save({"epoch": epoch, "score": float(score), "model_state": model.state_dict()}, _best_path(tag))


def _restore_rng(rng):
    # checkpoint may have been loaded with map_location=cuda -> coerce RNG back to CPU ByteTensor
    torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in rng["cuda"]])
    np.random.set_state(rng["numpy"]); __import__("random").setstate(rng["python"])


def fit(model, train_ld, val_ld, *, epochs, lr, patience, monitor="val_loss",
        loss_fn=None, distill=None, teacher=None, scheduler_cfg=None, tag="model",
        logger=None, resume=True):
    """Resumable train loop: early stopping (restore best) + optional ReduceLROnPlateau / distillation."""
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    sched = None
    if scheduler_cfg:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=scheduler_cfg["factor"], patience=scheduler_cfg["patience"],
            min_lr=scheduler_cfg["min_lr"])
    hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [], "lr": [], "epoch_time": []}
    best = {"score": None, "epoch": -1}
    wait = 0; start_ep = 1

    # ---- resume ----
    if resume and os.path.exists(_last_path(tag)):
        ck = torch.load(_last_path(tag), map_location=C.DEVICE, weights_only=False)
        if ck.get("done"):
            if os.path.exists(_best_path(tag)):
                model.load_state_dict(torch.load(_best_path(tag), map_location=C.DEVICE)["model_state"])
            tqdm.write(f"[{tag}] already complete (best {monitor}={ck['best'].get('score')}) - loaded best, skipping.")
            if logger: logger.event(f"{tag}: already complete, skipped")
            return ck["history"], ck["best"]
        model.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["opt_state"])
        if sched is not None and ck["sched_state"] is not None: sched.load_state_dict(ck["sched_state"])
        hist = ck["history"]; best = ck["best"]; wait = ck["wait"]; start_ep = ck["epoch"] + 1
        _restore_rng(ck["rng"])
        tqdm.write(f"[{tag}] resuming at epoch {start_ep}/{epochs} (best {monitor}={best.get('score')})")
        if logger: logger.event(f"{tag}: resumed at epoch {start_ep}")

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
                    with torch.no_grad():
                        t_logits = teacher(x)
                    loss, _, _ = distill(s_logits, t_logits, y)
                else:
                    loss = loss_fn(s_logits, y)
            loss.backward(); opt.step()
            tr_loss += loss.item() * x.size(0); tr_corr += (s_logits.argmax(1) == y).sum().item(); n += x.size(0)
            pbar.set_postfix(loss=f"{tr_loss/n:.3f}", acc=f"{tr_corr/n:.3f}")

        vl, va = evaluate(model, val_ld, nn.CrossEntropyLoss())
        tr_loss /= n; tr_acc = tr_corr / n; lr_now = opt.param_groups[0]["lr"]; dt = time.time() - t0
        hist["loss"].append(tr_loss); hist["val_loss"].append(vl)
        hist["accuracy"].append(tr_acc); hist["val_accuracy"].append(va)
        hist["lr"].append(lr_now); hist["epoch_time"].append(dt)
        if sched is not None: sched.step(vl)

        score = vl if monitor == "val_loss" else va
        improved = (best["score"] is None or
                    (score < best["score"] if monitor == "val_loss" else score > best["score"]))
        if improved:
            best = {"score": float(score), "epoch": ep}; wait = 0
            _save_best(tag, epoch=ep, model=model, score=score)
        else:
            wait += 1
        # recovery checkpoint + JSON mirror (every epoch)
        _save_last(tag, epoch=ep, model=model, opt=opt, sched=sched, history=hist, best=best, wait=wait, done=False)
        runlog.dump_json(os.path.join(C.METRICDIR, f"history_{tag}.json"), hist)

        tqdm.write(f"[{tag}] ep {ep:02d}/{epochs}  loss={tr_loss:.4f} val_loss={vl:.4f}  "
                   f"acc={tr_acc:.4f} val_acc={va:.4f}  lr={lr_now:.1e}  ({dt:.0f}s){'  *best' if improved else ''}")
        if logger:
            logger.epoch(tag, dict(epoch=ep, loss=tr_loss, val_loss=vl, accuracy=tr_acc,
                                   val_accuracy=va, lr=lr_now, epoch_time=dt, improved=bool(improved)))
        if wait >= patience:
            tqdm.write(f"[{tag}] early stop at ep {ep} (best {monitor}={best['score']:.4f} @ ep{best['epoch']})")
            break

    # finished: restore best weights, mark stage done
    if os.path.exists(_best_path(tag)):
        model.load_state_dict(torch.load(_best_path(tag), map_location=C.DEVICE)["model_state"])
    _save_last(tag, epoch=ep, model=model, opt=opt, sched=sched, history=hist, best=best, wait=wait, done=True)
    if logger: logger.event(f"{tag}: done (best {monitor}={best['score']:.4f} @ ep{best['epoch']})")
    return hist, best


def _wipe(patterns):
    for pat in patterns:
        for p in glob.glob(os.path.join(C.CKPTDIR, pat)):
            os.remove(p)


def main(smoke=False, fresh=False, resume=True):
    C.set_seed()
    torch.backends.cudnn.benchmark = True
    sfx = "_smoke" if smoke else ""
    if fresh and not smoke:
        _wipe(["*_last.pt", "*_best.pt"]); print("--fresh: cleared existing checkpoints")
    if smoke:
        _wipe(["*_smoke_last.pt", "*_smoke_best.pt"]); resume = False

    run_id = runlog.now_id()
    logger = runlog.RunLogger(C.LOGDIR, run_id, config=dict(
        seed=C.SEED, img_size=C.IMG_SIZE, batch_size=C.BATCH_SIZE, val_split=C.VAL_SPLIT,
        teacher=C.TEACHER, student=C.STUDENT, device=C.DEVICE, bf16=C.USE_BF16, smoke=smoke))
    print(f"Run id: {run_id}  ->  {logger.path}")

    train_ld, val_ld, counts = make_loaders()
    print("Class counts:", counts, "| total", sum(counts.values()))
    cw = class_weights(counts).to(C.DEVICE)

    # ---------------- TEACHER (EfficientNet-B4), two-stage ----------------
    teacher = build_teacher(C.TEACHER, C.NUM_CLASSES).to(C.DEVICE)
    tot, _ = count_params(teacher); print(f"Teacher params: {tot/1e6:.1f}M")
    ce_w = nn.CrossEntropyLoss(weight=cw)
    s1, s2 = C.TEACHER["stage1"], C.TEACHER["stage2"]
    e1, e2 = (2, 2) if smoke else (s1["epochs"], s2["epochs"])

    teacher.set_backbone_trainable(False)
    h1, _ = fit(teacher, train_ld, val_ld, epochs=e1, lr=s1["lr"], patience=s1["patience"],
                monitor="val_loss", loss_fn=ce_w, scheduler_cfg=C.TEACHER["reduce_lr"],
                tag=f"teacher-s1{sfx}", logger=logger, resume=resume)
    teacher.set_backbone_trainable(True)
    h2, _ = fit(teacher, train_ld, val_ld, epochs=e2, lr=s2["lr"], patience=s2["patience"],
                monitor="val_loss", loss_fn=ce_w, tag=f"teacher-s2{sfx}", logger=logger, resume=resume)
    torch.save(teacher.state_dict(), os.path.join(C.CKPTDIR, f"teacher_b4{sfx}.pt"))
    teacher_hist = {k: h1[k] + h2[k] for k in h1}
    _, teacher_val_acc = evaluate(teacher, val_ld)
    print(f"== Teacher final val_acc = {teacher_val_acc:.4f} ==")

    # ---------------- STUDENT (EfficientNet-B0), distillation ----------------
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    student = build_student(C.STUDENT, C.NUM_CLASSES).to(C.DEVICE)
    tot_s, _ = count_params(student); print(f"Student params: {tot_s/1e6:.1f}M")
    distill = DistillLoss(temperature=C.STUDENT["temperature"], alpha=C.STUDENT["alpha"])
    e_s = 2 if smoke else C.STUDENT["epochs"]
    stu_hist, _ = fit(student, train_ld, val_ld, epochs=e_s, lr=C.STUDENT["lr"],
                      patience=C.STUDENT["patience"], monitor="val_accuracy",
                      distill=distill, teacher=teacher, tag=f"student-kd{sfx}", logger=logger, resume=resume)
    torch.save(student.state_dict(), os.path.join(C.CKPTDIR, f"student_b0{sfx}.pt"))
    _, student_val_acc = evaluate(student, val_ld)
    print(f"== Student final val_acc = {student_val_acc:.4f} ==")

    # ---- consolidated JSON (source for all curves) ----
    out = dict(run_id=run_id, teacher_history=teacher_hist, student_history=stu_hist,
               teacher_val_acc=teacher_val_acc, student_val_acc=student_val_acc,
               class_counts=counts, smoke=smoke)
    runlog.dump_json(os.path.join(C.METRICDIR, "train_histories.json"), out)
    logger.finalize(dict(teacher_val_acc=teacher_val_acc, student_val_acc=student_val_acc))
    print("Saved histories ->", os.path.join(C.METRICDIR, "train_histories.json"))
    print("Saved run log   ->", logger.path)

    make_curves(teacher_hist, stu_hist)
    print("Saved curves    -> results/figures/{teacher,student}_training_curves.(pdf|png)")


def make_curves(teacher_hist, stu_hist):
    import plotting as P
    P.training_curves(teacher_hist, os.path.join(C.FIGDIR, "teacher_training_curves"),
                      title="Teacher (EfficientNet-B4) training")
    P.training_curves(stu_hist, os.path.join(C.FIGDIR, "student_training_curves"),
                      title="Student (EfficientNet-B0, distilled) training")


def curves_from_json():
    """Rebuild training curves from results/metrics/train_histories.json (no training needed)."""
    d = runlog.load_json(os.path.join(C.METRICDIR, "train_histories.json"))
    if d is None:
        print("no train_histories.json found"); return
    make_curves(d["teacher_history"], d["student_history"])
    print("Rebuilt curves from JSON.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2-epoch sanity check (isolated checkpoints)")
    ap.add_argument("--fresh", action="store_true", help="ignore/clear existing checkpoints, start over")
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="do not resume from checkpoints")
    main(**vars(ap.parse_args()))
