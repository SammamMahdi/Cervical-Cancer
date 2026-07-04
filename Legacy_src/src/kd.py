"""Knowledge-distillation loss (Hinton et al., NIPS 2014 DL Workshop).

L = alpha * CE(student_logits, y) + (1-alpha) * T^2 * KL(softmax(teacher/T) || softmax(student/T))

Note: the original notebook applied temperature to already-softmaxed probabilities and
omitted the T^2 factor (KL term ~0.005, so distillation barely contributed). This is the
*correct* formulation on logits with the T^2 scaling; the student still reaches ~the same
accuracy on the easy random-split task while the KD term is now meaningful.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.1):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        hard = self.ce(student_logits, targets)
        T = self.T
        soft = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction="batchmean",
        ) * (T * T)
        loss = self.alpha * hard + (1.0 - self.alpha) * soft
        return loss, hard.detach(), soft.detach()
