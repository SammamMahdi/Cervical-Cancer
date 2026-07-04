"""Knowledge-distillation and class-imbalance losses (U2 / A2 / A6).

Public API
----------
- DistillLoss      : CORRECT Hinton (NeurIPS 2014) KD on logits with T^2 scaling.
- BuggyDistillLoss : reproduces the LEGACY BUG (A2 ablation) — intentionally wrong.
- FocalLoss        : standard multiclass focal loss for imbalance (A6).

All three are drop-in ``nn.Module`` losses operating on raw class **logits**
(shape ``[B, NUM_CLASSES]``) and integer ``targets`` (shape ``[B]``).

Defaults mirror ``config.STUDENT`` (temperature=4.0, alpha=0.5).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillLoss(nn.Module):
    r"""Correct Hinton knowledge-distillation loss (U2).

    Combines a supervised cross-entropy term on the student logits with a
    temperature-softened KL-divergence between student and (frozen) teacher
    distributions::

        ce    = CE(student_logits, targets, weight, label_smoothing)
        kd    = T^2 * KL( log_softmax(student/T) || softmax(teacher/T) )
        total = alpha * ce + (1 - alpha) * kd

    The ``T^2`` factor restores the gradient magnitude of the soft-target term
    (it is otherwise scaled by ``1/T^2`` through the softmax), exactly as in
    Hinton et al., "Distilling the Knowledge in a Neural Network" (NeurIPS 2014
    Deep Learning Workshop). KL uses ``reduction='batchmean'`` so the loss is a
    proper per-example average.

    Parameters
    ----------
    temperature : float
        Softmax temperature ``T`` (>0). Higher = softer targets.
    alpha : float
        Weight on the hard-label CE term; ``1 - alpha`` weights the soft KD term.
    label_smoothing : float
        Label smoothing passed to the CE term.
    class_weight : Tensor | None
        Optional per-class weight tensor for CE (e.g. inverse-frequency).
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        label_smoothing: float = 0.0,
        class_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.label_smoothing = float(label_smoothing)
        # register as buffer so it follows .to(device) / state_dict (None is fine).
        self.register_buffer("class_weight", class_weight)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(total, ce, kd)`` scalar losses."""
        T = self.temperature
        # hard-label supervised term.
        ce = F.cross_entropy(
            student_logits,
            targets,
            weight=self.class_weight,
            label_smoothing=self.label_smoothing,
        )
        # soft-target term on LOGITS, with the gradient-restoring T^2 factor.
        # teacher is a fixed target -> detach to be safe even if it carries grad.
        s_log = F.log_softmax(student_logits / T, dim=1)
        t_prob = F.softmax(teacher_logits.detach() / T, dim=1)
        kd = (T * T) * F.kl_div(s_log, t_prob, reduction="batchmean")

        total = self.alpha * ce + (1.0 - self.alpha) * kd
        return total, ce, kd


class BuggyDistillLoss(nn.Module):
    r"""LEGACY-BUG distillation loss — INTENTIONALLY WRONG (A2 ablation only).

    This reproduces the distillation bug found in the legacy pipeline so the
    ablation study can quantify the damage. It is wrong in two ways:

    1. It applies the temperature to ALREADY-SOFTMAXED probabilities
       (``softmax(p / T)`` instead of ``softmax(logits / T)``). Because the
       inputs are probabilities in ``[0, 1]``, dividing by ``T`` and re-softmaxing
       barely changes a near-uniform distribution — the "soft targets" collapse
       toward uniform and carry almost no teacher information.
    2. It OMITS the ``T^2`` scaling, so the KD gradient is additionally damped by
       ``~1/T^2``.

    The net effect is a KD term that is near-zero and contributes essentially no
    learning signal. DO NOT use this in production — it exists purely as the
    negative control against :class:`DistillLoss`.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        label_smoothing: float = 0.0,
        class_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("class_weight", class_weight)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(total, ce, kd)`` — the kd term is broken-by-design."""
        T = self.temperature
        ce = F.cross_entropy(
            student_logits,
            targets,
            weight=self.class_weight,
            label_smoothing=self.label_smoothing,
        )
        # BUG: softmax the LOGITS to probabilities FIRST, then divide by T and
        # re-softmax. Tempering already-normalized probabilities does almost
        # nothing, and there is NO T^2 term -> kd ~ 0.
        s_prob = F.softmax(student_logits, dim=1)
        t_prob = F.softmax(teacher_logits.detach(), dim=1)
        s_log = F.log_softmax(s_prob / T, dim=1)        # wrong: softmax of probs
        t_soft = F.softmax(t_prob / T, dim=1)           # wrong: softmax of probs
        kd = F.kl_div(s_log, t_soft, reduction="batchmean")  # no T^2

        total = self.alpha * ce + (1.0 - self.alpha) * kd
        return total, ce, kd


class FocalLoss(nn.Module):
    r"""Standard multiclass focal loss (A6).

    Down-weights well-classified examples to focus training on hard / minority
    classes (Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017)::

        FL = -alpha_c * (1 - p_t)^gamma * log(p_t)

    where ``p_t`` is the softmax probability of the true class and the optional
    per-class ``weight`` plays the role of the class balancing term ``alpha_c``.

    Parameters
    ----------
    gamma : float
        Focusing parameter (>=0). ``0`` recovers (weighted) cross-entropy.
    weight : Tensor | None
        Optional per-class weight (inverse-frequency style balancing).
    reduction : {'mean', 'sum', 'none'}
        Output reduction.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"invalid reduction: {reduction}")
        self.gamma = float(gamma)
        self.reduction = reduction
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the focal loss between ``logits`` and integer ``targets``."""
        # per-example CE (no reduction) already folds in the class weight.
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        # p_t = exp(-ce_unweighted); recover it from the UNWEIGHTED log-prob so the
        # focusing modulator is independent of the class weighting.
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = logp_t.exp()
        focal = (1.0 - pt).pow(self.gamma) * ce  # ce carries the class weight

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


if __name__ == "__main__":
    # smoke-test: confirm shapes, finiteness, and that the correct KD term is
    # orders of magnitude larger than the buggy one.
    torch.manual_seed(0)
    try:
        import config as C  # noqa: F401

        num_classes = C.NUM_CLASSES
        T = C.STUDENT["temperature"]
        alpha = C.STUDENT["alpha"]
        ls = C.STUDENT.get("label_smoothing", 0.0)
    except Exception:  # standalone fallback
        num_classes, T, alpha, ls = 5, 4.0, 0.5, 0.05

    B = 32
    # teacher is made noticeably more confident than the student so a real KD
    # signal exists for the correct loss to capture.
    student = torch.randn(B, num_classes)
    teacher = torch.randn(B, num_classes) * 3.0
    targets = torch.randint(0, num_classes, (B,))
    cw = torch.rand(num_classes) + 0.5  # arbitrary positive class weights

    good = DistillLoss(temperature=T, alpha=alpha, label_smoothing=ls, class_weight=cw)
    bad = BuggyDistillLoss(temperature=T, alpha=alpha, label_smoothing=ls, class_weight=cw)
    focal = FocalLoss(gamma=2.0, weight=cw)

    g_total, g_ce, g_kd = good(student, teacher, targets)
    b_total, b_ce, b_kd = bad(student, teacher, targets)
    f_loss = focal(student, targets)

    print(f"num_classes={num_classes}  T={T}  alpha={alpha}  label_smoothing={ls}")
    print(f"DistillLoss      : total={g_total.item():.4f}  ce={g_ce.item():.4f}  kd={g_kd.item():.4f}")
    print(f"BuggyDistillLoss : total={b_total.item():.4f}  ce={b_ce.item():.4f}  kd={b_kd.item():.6f}")
    print(f"FocalLoss        : loss={f_loss.item():.4f}")

    # all finite
    for name, v in [("g_total", g_total), ("b_total", b_total), ("f_loss", f_loss)]:
        assert torch.isfinite(v).all(), f"{name} is not finite"

    # gradient flows through the student for every loss
    for fn in (
        lambda: good(student.requires_grad_(True), teacher, targets)[0],
        lambda: bad(student.detach().requires_grad_(True), teacher, targets)[0],
        lambda: focal(student.detach().requires_grad_(True), targets),
    ):
        student.grad = None
        out = fn()
        out.backward()

    # the core assertion of the A2 ablation: correct KD >> buggy KD.
    assert g_kd.item() > 10.0 * abs(b_kd.item()) + 1e-6, (
        f"expected correct kd ({g_kd.item():.4f}) >> buggy kd ({b_kd.item():.6f})"
    )
    print(f"OK: DistillLoss kd ({g_kd.item():.4f}) >> BuggyDistillLoss kd ({b_kd.item():.6f})")
