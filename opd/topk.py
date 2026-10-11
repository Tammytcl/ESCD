"""Pure tensor helpers for Top-K on-policy distillation."""

from __future__ import annotations

import torch


def merge_topk_union(teacher_ids: torch.Tensor, student_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return teacher-first Top-K union IDs and a validity mask.

    The output keeps a fixed ``2K`` width. Student IDs already present in the
    teacher Top-K are retained as padded entries with mask value zero.
    """

    if teacher_ids.shape != student_ids.shape:
        raise ValueError(f"Top-K ID shapes must match, got {teacher_ids.shape} and {student_ids.shape}.")
    duplicate = (student_ids.unsqueeze(-1) == teacher_ids.unsqueeze(-2)).any(dim=-1)
    union_ids = torch.cat((teacher_ids, student_ids), dim=-1)
    union_mask = torch.cat((torch.ones_like(teacher_ids, dtype=torch.bool), ~duplicate), dim=-1)
    return union_ids, union_mask


def renormalized_teacher_student_kl(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(teacher || student) after renormalizing on a shared support."""

    if teacher_log_probs.shape != student_log_probs.shape or mask.shape != teacher_log_probs.shape:
        raise ValueError("Teacher log-probs, student log-probs, and mask must have identical shapes.")
    masked_teacher = teacher_log_probs.masked_fill(~mask, float("-inf"))
    masked_student = student_log_probs.masked_fill(~mask, float("-inf"))
    teacher_local_log_probs = masked_teacher - torch.logsumexp(masked_teacher, dim=-1, keepdim=True)
    student_local_log_probs = masked_student - torch.logsumexp(masked_student, dim=-1, keepdim=True)
    teacher_local_probs = teacher_local_log_probs.exp().masked_fill(~mask, 0.0)
    return (teacher_local_probs * (teacher_local_log_probs - student_local_log_probs)).masked_fill(~mask, 0.0).sum(-1)


def teacher_topk_generalized_kl(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute non-renormalized generalized KL on the teacher Top-K support."""

    if teacher_log_probs.shape != student_log_probs.shape:
        raise ValueError("Teacher and student log-probs must have identical shapes.")
    if mask is None:
        mask = torch.ones_like(teacher_log_probs, dtype=torch.bool)
    if mask.shape != teacher_log_probs.shape:
        raise ValueError("Mask and log-probs must have identical shapes.")
    teacher_probs = teacher_log_probs.exp()
    student_probs = student_log_probs.exp()
    terms = teacher_probs * (teacher_log_probs - student_log_probs) - teacher_probs + student_probs
    return terms.masked_fill(~mask, 0.0).sum(-1)
