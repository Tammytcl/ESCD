"""Portable BPM distribution utilities used by offline research scripts."""
from __future__ import annotations

import torch


def _kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(torch.float32).tiny
    return torch.where(
        left > 0,
        left * (left.clamp_min(eps).log() - right.clamp_min(eps).log()),
        torch.zeros_like(left),
    ).sum(dim=-1)


def bpm_divergence(
    target: torch.Tensor,
    student: torch.Tensor,
    *,
    beta: float,
    stop_mask: torch.Tensor,
    rkl_skew: float = 0.1,
) -> torch.Tensor:
    if target.shape != student.shape or target.ndim != 2:
        raise ValueError("BPM target and Student distributions must have equal 2D shapes")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("BPM beta must be in [0, 1]")
    if not 0.0 < rkl_skew < 1.0:
        raise ValueError("BPM rkl_skew must be in (0, 1)")
    if beta == 0.0:
        losses = _kl(target, student)
    elif beta == 1.0:
        losses = _kl(student, (1.0 - rkl_skew) * target + rkl_skew * student)
    else:
        mixture = beta * target + (1.0 - beta) * student
        losses = beta * _kl(target, mixture) + (1.0 - beta) * _kl(student, mixture)
    return torch.where(stop_mask, _kl(target, student), losses)


def aggregate_bpm_event_probs(
    values: torch.Tensor,
    group_index: torch.Tensor,
    *,
    group_count: int | None = None,
) -> torch.Tensor:
    if values.ndim != 2 or group_index.shape != values.shape:
        raise ValueError("BPM grouped values and indices must have equal 2D shapes")
    if values.numel() and bool((group_index < 0).any()):
        raise ValueError("BPM group indices must be non-negative")
    inferred = int(group_index.max().item()) + 1 if group_index.numel() else 0
    width = max(inferred, int(group_count or 0))
    if width <= 0:
        raise ValueError("BPM grouped payload must contain at least one group")
    if group_index.numel() and bool((group_index >= width).any()):
        raise ValueError("BPM group index exceeds declared group count")
    output = values.new_zeros((values.size(0), width))
    output.scatter_add_(1, group_index.to(device=values.device, dtype=torch.long), values)
    if not bool(torch.allclose(output.sum(dim=-1), values.sum(dim=-1), rtol=1e-5, atol=1e-6)):
        raise AssertionError("BPM grouped event aggregation lost probability mass")
    return output
