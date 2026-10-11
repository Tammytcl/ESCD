"""Small shared helpers for paper-faithful cross-tokenizer baselines."""

from __future__ import annotations

import math
from typing import Any

import torch

from .base import AlignmentResult


def tokenizer_vocab_size(tokenizer: Any) -> int:
    """Return the effective tokenizer vocabulary, excluding model padding."""

    vocab = tokenizer.get_vocab()
    max_id_size = max((int(token_id) for token_id in vocab.values()), default=-1) + 1
    try:
        reported_size = int(len(tokenizer))
    except (TypeError, AttributeError):
        reported_size = 0
    size = max(max_id_size, reported_size)
    if size <= 0:
        raise ValueError("Tokenizer vocabulary must be non-empty")
    return size


def export_tokenizer_runtime_state(student_tokenizer: Any, teacher_tokenizer: Any) -> dict[str, Any]:
    return {
        "student_vocab_size": tokenizer_vocab_size(student_tokenizer),
        "teacher_vocab_size": tokenizer_vocab_size(teacher_tokenizer),
        "student_eos_token_id": (
            int(student_tokenizer.eos_token_id)
            if getattr(student_tokenizer, "eos_token_id", None) is not None
            else None
        ),
        "teacher_eos_token_id": (
            int(teacher_tokenizer.eos_token_id)
            if getattr(teacher_tokenizer, "eos_token_id", None) is not None
            else None
        ),
    }


def require_runtime_int(runtime_state: dict[str, Any], name: str) -> int:
    value = runtime_state.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"cross-tokenizer runtime_state requires positive integer {name!r}")
    return value


def resolve_vocab_size(runtime_state: dict[str, Any], name: str, tokenizer: Any | None) -> int:
    """Resolve GPU JSON state, with a CPU-tokenizer fallback for unit tools."""

    value = runtime_state.get(name)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if tokenizer is not None:
        return tokenizer_vocab_size(tokenizer)
    return require_runtime_int(runtime_state, name)


def finite_float(value: Any, *, name: str, minimum: float | None = None) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        qualifier = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be finite{qualifier}, got {parsed}")
    return parsed


def aligned_segment_indices(
    alignment: AlignmentResult,
    student_loss_mask: torch.Tensor,
    *,
    device: torch.device,
    drop_terminal_segment: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select byte-aligned groups fully covered by the Student loss mask."""

    student_loss_mask = student_loss_mask.to(device=device, dtype=torch.bool).flatten()
    if student_loss_mask.numel() != alignment.student_length:
        raise ValueError("Student loss-mask length does not match the alignment")
    segments = alignment.segments[:-1] if drop_terminal_segment and alignment.segments else alignment.segments
    if alignment.skipped or not segments:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, empty, empty

    teacher_starts = torch.tensor(
        [segment.teacher_start for segment in segments], dtype=torch.long, device=device
    )
    teacher_ends = torch.tensor(
        [segment.teacher_end for segment in segments], dtype=torch.long, device=device
    )
    student_starts = torch.tensor(
        [segment.student_start for segment in segments], dtype=torch.long, device=device
    )
    student_ends = torch.tensor(
        [segment.student_end for segment in segments], dtype=torch.long, device=device
    )
    prefix = torch.cat(
        (torch.zeros(1, dtype=torch.long, device=device), student_loss_mask.long().cumsum(dim=0))
    )
    selected = (
        prefix[student_ends] - prefix[student_starts] == student_ends - student_starts
    ).nonzero(as_tuple=False).flatten()
    return (
        teacher_starts.index_select(0, selected),
        teacher_ends.index_select(0, selected),
        student_starts.index_select(0, selected),
        student_ends.index_select(0, selected),
    )


def merge_observed_probabilities(
    probabilities: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    response_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply GOLD's default observed-token merge to aligned groups.

    For a group ``[i, ..., j]``, GOLD keeps the complete distribution at
    ``i`` and multiplies it by probabilities of the actually observed tokens
    at ``i+1..j``.  The result is intentionally unnormalized.
    """

    if starts.shape != ends.shape or starts.ndim != 1:
        raise ValueError("GOLD group starts/ends must be equal one-dimensional tensors")
    if probabilities.ndim != 2 or probabilities.size(0) != response_token_ids.numel():
        raise ValueError("GOLD probabilities must contain one row per response token")
    if starts.numel() == 0:
        return probabilities.new_empty((0, probabilities.size(-1)))
    rows: list[torch.Tensor] = []
    for start_tensor, end_tensor in zip(starts, ends, strict=True):
        start = int(start_tensor)
        end = int(end_tensor)
        if not 0 <= start < end <= probabilities.size(0):
            raise ValueError(f"Invalid GOLD alignment group [{start}, {end})")
        merged = probabilities[start]
        if end - start > 1:
            continuation_rows = torch.arange(start + 1, end, device=probabilities.device)
            continuation_ids = response_token_ids[start + 1 : end].to(
                device=probabilities.device, dtype=torch.long
            )
            continuation = probabilities[continuation_rows, continuation_ids].clamp_min(1e-8)
            merged = merged * continuation.prod()
        rows.append(merged)
    return torch.stack(rows)
