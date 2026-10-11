from argparse import Namespace
from collections.abc import Callable, Iterator
import logging
import math
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from torch.utils.checkpoint import checkpoint

from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.misc import load_function
from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from slime.utils.types import RolloutBatch

from opd.cross_tokenizer import (
    AlignmentResult,
    CandidateScores,
    FusedScoringContext,
    StudentScoringContext,
    TeacherScoringContext,
    TensorPayload,
    VocabularyMapping,
    build_cross_tokenizer,
    localize_payload_by_student_positions,
)
from opd.topk import merge_topk_union, renormalized_teacher_student_kl, teacher_topk_generalized_kl

from .cp_utils import (
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
)

TEACHER_LOGP_FAILED_SENTINEL = -100.0
logger = logging.getLogger(__name__)

_CROSS_TOKENIZER_TEACHER_HEAD_SHARD: torch.Tensor | None = None
_CROSS_TOKENIZER_TEACHER_VOCAB_SIZE: int | None = None
_CROSS_TOKENIZER_TEACHER_VALIDATED = False


def set_cross_tokenizer_teacher_head_shard(
    weight: torch.Tensor, *, vocab_size: int
) -> None:
    """Install a frozen causal-hidden Teacher output shard for one actor."""

    global _CROSS_TOKENIZER_TEACHER_HEAD_SHARD, _CROSS_TOKENIZER_TEACHER_VOCAB_SIZE
    global _CROSS_TOKENIZER_TEACHER_VALIDATED
    if weight.ndim != 2 or weight.requires_grad:
        raise ValueError("Cross-tokenizer Teacher head must be a frozen rank-two tensor")
    _CROSS_TOKENIZER_TEACHER_HEAD_SHARD = weight
    _CROSS_TOKENIZER_TEACHER_VOCAB_SIZE = int(vocab_size)
    _CROSS_TOKENIZER_TEACHER_VALIDATED = False


class _ReplicatedAllReduceSum(torch.autograd.Function):
    """All-reduce values while keeping one local backward contribution per TP rank."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
        ctx.group = group
        if group is None or not dist.is_initialized() or dist.get_world_size(group) == 1:
            return value
        output = value.clone()
        dist.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


class _ContextParallelResponseGather(torch.autograd.Function):
    """Assemble response rows across CP ranks and route remote loss gradients home.

    Cross-tokenizer plugins normally score independent local rows.  Algorithms
    such as SimCT additionally need the realized-token scores inside a minimal
    aligned unit, which may cross a CP boundary.  This generic primitive keeps
    that communication out of the algorithm and never gathers full vocabulary
    logits: only the plugin-selected row values are exchanged.
    """

    @staticmethod
    def forward(
        ctx,
        local_rows: torch.Tensor,
        local_positions: torch.Tensor,
        response_length: int,
        group: dist.ProcessGroup | None,
    ) -> torch.Tensor:
        positions = local_positions.to(device=local_rows.device, dtype=torch.long).flatten()
        if local_rows.ndim == 0 or local_rows.size(0) != positions.numel():
            raise ValueError("CP response values must have one leading row per local position")
        response_length = int(response_length)
        if positions.numel() and bool(((positions < 0) | (positions >= response_length)).any()):
            raise ValueError("CP response positions fall outside the global response")
        global_rows = local_rows.new_zeros((response_length, *local_rows.shape[1:]))
        global_rows.index_add_(0, positions, local_rows)
        distributed = (
            group is not None
            and dist.is_initialized()
            and dist.get_world_size(group) > 1
        )
        if distributed:
            dist.all_reduce(global_rows, group=group)
        ctx.group = group
        ctx.distributed = distributed
        ctx.save_for_backward(positions)
        return global_rows

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (positions,) = ctx.saved_tensors
        global_gradient = grad_output.contiguous()
        if ctx.distributed:
            global_gradient = global_gradient.clone()
            dist.all_reduce(global_gradient, group=ctx.group)
        return global_gradient.index_select(0, positions), None, None, None


def _gather_context_parallel_response_rows(
    local_rows: torch.Tensor,
    *,
    local_positions: torch.Tensor,
    response_length: int,
) -> torch.Tensor:
    return _ContextParallelResponseGather.apply(
        local_rows,
        local_positions,
        int(response_length),
        mpu.get_context_parallel_group(),
    )


class _VocabularyParallelAllGather(torch.autograd.Function):
    """Gather vocabulary shards while routing each rank's backward slice home.

    ULD/GOLD need a complete probability vector on every tensor-parallel rank.
    ``torch.distributed.all_gather`` itself is not autograd-aware; this wrapper
    gives each rank only the gradient slice for the local vocabulary shard.
    The loss is already replicated across TP ranks, so an additional backward
    all-reduce would multiply the gradient by the TP world size.
    """

    @staticmethod
    def forward(ctx, value: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
        ctx.group = group
        ctx.world_size = _tp_world_size(group)
        ctx.rank = _tp_rank(group)
        if ctx.world_size == 1:
            return value
        gathered = [torch.empty_like(value) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, value, group=group)
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        if ctx.world_size == 1:
            return grad_output, None
        chunks = grad_output.chunk(ctx.world_size, dim=-1)
        return chunks[ctx.rank].contiguous(), None


def _tp_world_size(group: dist.ProcessGroup | None) -> int:
    return dist.get_world_size(group) if group is not None and dist.is_initialized() else 1


def _tp_rank(group: dist.ProcessGroup | None) -> int:
    return dist.get_rank(group) if group is not None and dist.is_initialized() else 0


def _validate_global_token_ids(
    local_logits: torch.Tensor,
    token_ids: torch.Tensor,
    group: dist.ProcessGroup | None,
    *,
    field_name: str,
) -> None:
    if token_ids.numel() == 0:
        return
    global_vocab_size = local_logits.size(-1) * _tp_world_size(group)
    invalid = (token_ids < 0) | (token_ids >= global_vocab_size)
    if bool(invalid.any()):
        invalid_ids = token_ids[invalid]
        raise ValueError(
            f"{field_name} contains token IDs outside [0, {global_vocab_size}): "
            f"min={int(invalid_ids.min())}, max={int(invalid_ids.max())}"
        )


def _vocab_parallel_log_normalizer(
    local_logits: torch.Tensor,
    group: dist.ProcessGroup | None,
    *,
    row_indices: torch.Tensor | None = None,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Return TP-global log-normalizers without a full FP32 logits copy.

    ``chunk_size`` is a row count, not a vocabulary width.  Qwen3.5-2B emits
    BF16 ``[sequence, 124160]`` logits at TP2.  Casting or reducing every row
    together creates 10--15 GiB FP32 temporaries, while vocabulary-chunking
    still leaves one autograd branch per vocabulary chunk alive until
    backward.  Row chunks bound both the forward workspace and saved backward
    state.  CP8 then limits production responses to roughly four chunks per
    rank instead of twenty-eight chunks at CP1.

    ``row_indices`` additionally avoids normalizing filtered BPM rows.  It is
    deliberately implemented with row ``index_select`` inside each chunk so
    the complete logits tensor is never duplicated.
    """
    if local_logits.ndim != 2:
        raise ValueError("Vocabulary log-normalizer requires [rows, local_vocab] logits")
    if chunk_size <= 0:
        raise ValueError(f"log-normalizer chunk_size must be positive, got {chunk_size}")
    if local_logits.size(-1) == 0:
        raise ValueError("Vocabulary logits must have a non-empty final dimension")
    if row_indices is not None:
        row_indices = row_indices.to(device=local_logits.device, dtype=torch.long).flatten()
        if row_indices.numel() and bool(
            ((row_indices < 0) | (row_indices >= local_logits.size(0))).any()
        ):
            raise ValueError("Log-normalizer row index is outside the logits tensor")
        num_rows = int(row_indices.numel())
    else:
        num_rows = int(local_logits.size(0))

    # Retain an environment override for production memory tuning while
    # keeping the explicit argument deterministic in tests.
    if chunk_size == 1024:
        chunk_size = int(os.environ.get("OPD_LOGSUMEXP_CHUNK_ROWS", str(chunk_size)))
        if chunk_size <= 0:
            raise ValueError("OPD_LOGSUMEXP_CHUNK_ROWS must be positive")

    normalizers: list[torch.Tensor] = []
    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        chunk = (
            local_logits[start:end]
            if row_indices is None
            else local_logits.index_select(0, row_indices[start:end])
        )
        if chunk.dtype in (torch.float16, torch.bfloat16):
            chunk = chunk.float()
        local_max = chunk.detach().amax(dim=-1, keepdim=True)
        if _tp_world_size(group) > 1:
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=group)
        local_sum = (chunk - local_max).exp().sum(dim=-1, keepdim=True)
        global_sum = _ReplicatedAllReduceSum.apply(local_sum, group)
        normalizers.append(local_max + global_sum.log())
    if not normalizers:
        return local_logits.new_empty((0, 1), dtype=torch.float32)
    return torch.cat(normalizers, dim=0)


def _distributed_selected_log_probs(
    local_logits: torch.Tensor,
    token_ids: torch.Tensor,
    group: dist.ProcessGroup | None,
    *,
    row_indices: torch.Tensor | None = None,
    log_normalizer: torch.Tensor | None = None,
) -> torch.Tensor:
    selected_logits = _distributed_selected_logits(
        local_logits,
        token_ids,
        group,
        row_indices=row_indices,
    )
    if log_normalizer is None:
        log_normalizer = _vocab_parallel_log_normalizer(
            local_logits, group, row_indices=row_indices
        )
    if log_normalizer.shape != (token_ids.size(0), 1):
        raise ValueError("Selected log-probabilities received an incompatible log-normalizer")
    return selected_logits - log_normalizer


def _distributed_selected_logits(
    local_logits: torch.Tensor,
    token_ids: torch.Tensor,
    group: dist.ProcessGroup | None,
    *,
    row_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather raw global-vocabulary logits without applying a row normalizer."""

    if local_logits.ndim != 2 or token_ids.ndim != 2:
        raise ValueError("Selected logits require logits and token IDs to be two-dimensional")
    if row_indices is not None:
        row_indices = row_indices.to(device=local_logits.device, dtype=torch.long).flatten()
        if row_indices.numel() != token_ids.size(0):
            raise ValueError("Selected-logit row count differs from token-ID rows")
        if row_indices.numel() and bool(
            ((row_indices < 0) | (row_indices >= local_logits.size(0))).any()
        ):
            raise ValueError("Selected-logit row index is outside the logits tensor")
    elif token_ids.size(0) != local_logits.size(0):
        raise ValueError("Selected-logit token IDs must contain one row per logits row")
    local_vocab_size = local_logits.size(-1)
    local_start = _tp_rank(group) * local_vocab_size
    local_end = local_start + local_vocab_size
    owned = (token_ids >= local_start) & (token_ids < local_end)
    local_ids = (token_ids - local_start).clamp(min=0, max=local_vocab_size - 1)
    local_selected = (
        local_logits.gather(-1, local_ids)
        if row_indices is None
        else local_logits[row_indices.unsqueeze(-1), local_ids]
    )
    if local_selected.dtype in (torch.float16, torch.bfloat16):
        local_selected = local_selected.float()
    # Multiplication is not a safe mask for logits: ``0 * +/-inf`` is NaN and
    # can poison a TP all-reduce even though this rank does not own the token.
    local_selected = local_selected.masked_fill(~owned, 0.0)
    return _ReplicatedAllReduceSum.apply(local_selected, group)


def _distributed_topk(
    local_logits: torch.Tensor,
    k: int,
    group: dist.ProcessGroup | None,
    *,
    with_log_probs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    world_size = _tp_world_size(group)
    local_vocab_size = local_logits.size(-1)
    if k <= 0 or k > local_vocab_size * world_size:
        raise ValueError(f"opd_top_k must be in [1, {local_vocab_size * world_size}], got {k}.")
    local_k = min(k, local_vocab_size)
    local_values, local_ids = torch.topk(local_logits.detach(), k=local_k, dim=-1)
    local_ids = local_ids + _tp_rank(group) * local_vocab_size
    if world_size > 1:
        gathered_values = [torch.empty_like(local_values) for _ in range(world_size)]
        gathered_ids = [torch.empty_like(local_ids) for _ in range(world_size)]
        dist.all_gather(gathered_values, local_values, group=group)
        dist.all_gather(gathered_ids, local_ids, group=group)
        candidate_values = torch.cat(gathered_values, dim=-1)
        candidate_ids = torch.cat(gathered_ids, dim=-1)
    else:
        candidate_values = local_values
        candidate_ids = local_ids
    topk_values, topk_offsets = torch.topk(candidate_values, k=k, dim=-1)
    topk_ids = candidate_ids.gather(-1, topk_offsets)
    if not with_log_probs:
        return topk_ids, None
    log_normalizer = _vocab_parallel_log_normalizer(local_logits, group)
    return topk_ids, topk_values - log_normalizer


def _distributed_mapped_topk(
    local_logits: torch.Tensor,
    source_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    k: int,
    group: dist.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select source-vocabulary candidates and return their target-vocabulary IDs.

    ``k == 0`` preserves the complete mapped vocabulary. Positive ``k`` uses
    only O(k * TP) communication, mirroring the native OPD Top-K path.
    """

    if source_token_ids.ndim != 1 or target_token_ids.shape != source_token_ids.shape:
        raise ValueError("Cross-tokenizer source/target vocabulary mappings must be equal 1-D tensors")
    if source_token_ids.numel() == 0:
        raise ValueError("Cross-tokenizer shared vocabulary cannot be empty")
    if k < 0:
        raise ValueError(f"cross_tokenizer_top_k must be non-negative, got {k}")
    _validate_global_token_ids(
        local_logits,
        source_token_ids,
        group,
        field_name="mapped_topk source_ids",
    )

    rows = local_logits.size(0)
    output_k = source_token_ids.numel() if k == 0 else min(k, source_token_ids.numel())
    if rows == 0:
        shape = (0, output_k)
        return (
            torch.empty(shape, dtype=torch.long, device=local_logits.device),
            torch.empty(shape, dtype=local_logits.dtype, device=local_logits.device),
            torch.empty(shape, dtype=torch.bool, device=local_logits.device),
        )
    if k == 0:
        expanded_source_ids = source_token_ids.unsqueeze(0).expand(rows, -1)
        mapped_log_probs = _distributed_selected_log_probs(local_logits, expanded_source_ids, group)
        mapped_target_ids = target_token_ids.unsqueeze(0).expand(rows, -1)
        return mapped_target_ids, mapped_log_probs, torch.ones_like(mapped_target_ids, dtype=torch.bool)

    local_vocab_size = local_logits.size(-1)
    local_start = _tp_rank(group) * local_vocab_size
    local_end = local_start + local_vocab_size
    owned = (source_token_ids >= local_start) & (source_token_ids < local_end)
    owned_global_source_ids = source_token_ids[owned]
    owned_source_ids = owned_global_source_ids - local_start
    owned_target_ids = target_token_ids[owned]

    local_values = torch.full(
        (rows, output_k), float("-inf"), dtype=local_logits.dtype, device=local_logits.device
    )
    local_sources = torch.zeros((rows, output_k), dtype=torch.long, device=local_logits.device)
    local_targets = torch.zeros((rows, output_k), dtype=torch.long, device=local_logits.device)
    if owned_source_ids.numel() > 0:
        # Avoid materializing [rows, complete_shared_vocab] in bounded mode.
        # Long responses otherwise create a multi-GB temporary even for K=128.
        chunk_size = 1024
        for chunk_start in range(0, owned_source_ids.numel(), chunk_size):
            chunk_end = min(chunk_start + chunk_size, owned_source_ids.numel())
            chunk_ids = owned_source_ids[chunk_start:chunk_end]
            chunk_values = local_logits.detach().index_select(-1, chunk_ids)
            chunk_sources = owned_global_source_ids[chunk_start:chunk_end].unsqueeze(0).expand(rows, -1)
            chunk_targets = owned_target_ids[chunk_start:chunk_end].unsqueeze(0).expand(rows, -1)
            merged_values = torch.cat((local_values, chunk_values), dim=-1)
            merged_sources = torch.cat((local_sources, chunk_sources), dim=-1)
            merged_targets = torch.cat((local_targets, chunk_targets), dim=-1)
            local_values, selected_offsets = torch.topk(merged_values, k=output_k, dim=-1)
            local_sources = merged_sources.gather(-1, selected_offsets)
            local_targets = merged_targets.gather(-1, selected_offsets)

    world_size = _tp_world_size(group)
    if world_size > 1:
        gathered_values = [torch.empty_like(local_values) for _ in range(world_size)]
        gathered_sources = [torch.empty_like(local_sources) for _ in range(world_size)]
        gathered_targets = [torch.empty_like(local_targets) for _ in range(world_size)]
        dist.all_gather(gathered_values, local_values, group=group)
        dist.all_gather(gathered_sources, local_sources, group=group)
        dist.all_gather(gathered_targets, local_targets, group=group)
        candidate_values = torch.cat(gathered_values, dim=-1)
        candidate_sources = torch.cat(gathered_sources, dim=-1)
        candidate_targets = torch.cat(gathered_targets, dim=-1)
    else:
        candidate_values = local_values
        candidate_sources = local_sources
        candidate_targets = local_targets

    topk_values, topk_offsets = torch.topk(candidate_values, k=output_k, dim=-1)
    topk_sources = candidate_sources.gather(-1, topk_offsets)
    topk_targets = candidate_targets.gather(-1, topk_offsets)
    valid_mask = torch.isfinite(topk_values)
    selected_log_probs = _distributed_selected_log_probs(local_logits, topk_sources, group)
    return topk_targets, selected_log_probs, valid_mask


class _MegatronVocabularyDistribution:
    """TP-aware implementation of the model-agnostic scoring facade."""

    def __init__(
        self,
        local_logits: torch.Tensor,
        group: dist.ProcessGroup | None,
        row_indices: torch.Tensor | None = None,
    ) -> None:
        if local_logits.ndim != 2:
            raise ValueError(f"Vocabulary distributions require [rows, local_vocab] logits, got {local_logits.shape}")
        self.local_logits = local_logits
        self.group = group
        self.row_indices = row_indices

    @property
    def length(self) -> int:
        return self.local_logits.size(0) if self.row_indices is None else self.row_indices.numel()

    def rows(self, row_indices: torch.Tensor) -> "_MegatronVocabularyDistribution":
        row_indices = row_indices.to(device=self.local_logits.device, dtype=torch.long)
        if row_indices.ndim != 1:
            raise ValueError("Vocabulary distribution row indices must be one-dimensional")
        if row_indices.numel() and bool(((row_indices < 0) | (row_indices >= self.length)).any()):
            raise ValueError("Vocabulary distribution row index is out of bounds")
        absolute_rows = (
            row_indices
            if self.row_indices is None
            else self.row_indices.index_select(0, row_indices)
        )
        # Keep a lazy row view.  BPM supervises almost every token in a long
        # response; eagerly index_select'ing [active_rows, local_vocab] would
        # duplicate nearly the complete 10+ GiB logits tensor.
        return _MegatronVocabularyDistribution(self.local_logits, self.group, absolute_rows)

    def _selected_row_log_probs(self, token_ids: torch.Tensor) -> torch.Tensor:
        assert self.row_indices is not None
        if self.row_indices.numel() == 0:
            return self.local_logits.new_empty(token_ids.shape)
        return _distributed_selected_log_probs(
            self.local_logits,
            token_ids,
            self.group,
            row_indices=self.row_indices,
        )

    def _materialized_logits(self) -> torch.Tensor:
        return (
            self.local_logits
            if self.row_indices is None
            else self.local_logits.index_select(0, self.row_indices)
        )

    def _effective_logits(
        self,
        *,
        vocab_size: int | None,
        temperature: float,
    ) -> torch.Tensor:
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"distribution temperature must be positive and finite, got {temperature}")
        logits = self._materialized_logits()
        if temperature != 1.0:
            logits = logits / temperature
        if vocab_size is None:
            return logits
        vocab_size = int(vocab_size)
        global_padded_vocab_size = logits.size(-1) * _tp_world_size(self.group)
        if not 0 < vocab_size <= global_padded_vocab_size:
            raise ValueError(
                f"vocab_size must be in [1, {global_padded_vocab_size}], got {vocab_size}"
            )
        local_start = _tp_rank(self.group) * logits.size(-1)
        local_ids = torch.arange(
            local_start,
            local_start + logits.size(-1),
            device=logits.device,
            dtype=torch.long,
        )
        valid = local_ids < vocab_size
        return logits if bool(valid.all()) else logits.masked_fill(~valid.unsqueeze(0), float("-inf"))

    def selected_log_probs(
        self,
        token_ids: torch.Tensor,
        *,
        vocab_size: int | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        token_ids = token_ids.to(device=self.local_logits.device, dtype=torch.long)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(-1)
        if token_ids.size(0) != self.length:
            raise ValueError("selected_log_probs row count differs from the vocabulary distribution")
        _validate_global_token_ids(
            self.local_logits,
            token_ids,
            self.group,
            field_name="selected_log_probs token_ids",
        )
        if vocab_size is not None and token_ids.numel() and bool((token_ids >= int(vocab_size)).any()):
            raise ValueError("selected_log_probs token ID is outside the effective vocabulary")
        if self.row_indices is not None and vocab_size is None and float(temperature) == 1.0:
            return self._selected_row_log_probs(token_ids)
        logits = self._effective_logits(vocab_size=vocab_size, temperature=temperature)
        return _distributed_selected_log_probs(logits, token_ids, self.group)

    def selected_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(device=self.local_logits.device, dtype=torch.long)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(-1)
        if token_ids.size(0) != self.length:
            raise ValueError("selected_logits row count differs from the vocabulary distribution")
        _validate_global_token_ids(
            self.local_logits,
            token_ids,
            self.group,
            field_name="selected_logits token_ids",
        )
        return _distributed_selected_logits(
            self.local_logits,
            token_ids,
            self.group,
            row_indices=self.row_indices,
        )

    def full_probabilities(
        self,
        *,
        vocab_size: int,
        temperature: float = 1.0,
        preserve_float32: bool = False,
    ) -> torch.Tensor:
        """Materialize an exact TP-global softmax over valid (unpadded) IDs."""

        vocab_size = int(vocab_size)
        temperature = float(temperature)
        global_padded_vocab_size = self.local_logits.size(-1) * _tp_world_size(self.group)
        if not 0 < vocab_size <= global_padded_vocab_size:
            raise ValueError(
                f"vocab_size must be in [1, {global_padded_vocab_size}], got {vocab_size}"
            )
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"distribution temperature must be positive and finite, got {temperature}")

        logits = self._effective_logits(vocab_size=vocab_size, temperature=temperature)
        log_normalizer = _vocab_parallel_log_normalizer(logits, self.group)
        local_probabilities = (logits.float() - log_normalizer).exp()
        if not preserve_float32 and logits.dtype in (torch.float16, torch.bfloat16):
            # Match the source implementations' autocast probability dtype and
            # avoid doubling already-dense ULD/GOLD payload memory.
            local_probabilities = local_probabilities.to(logits.dtype)
        full_probabilities = _VocabularyParallelAllGather.apply(local_probabilities, self.group)
        return full_probabilities[:, :vocab_size]

    def projected_log_probs(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        vocab_size: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Project onto explicit token cells and an exact log-space complement."""

        token_ids = token_ids.to(device=self.local_logits.device, dtype=torch.long)
        token_mask = token_mask.to(device=self.local_logits.device, dtype=torch.bool)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0).expand(self.length, -1)
        if token_mask.ndim == 1:
            token_mask = token_mask.unsqueeze(0).expand(self.length, -1)
        if token_ids.shape != token_mask.shape or token_ids.size(0) != self.length:
            raise ValueError("Projected token IDs/mask must contain one equal-width row per distribution row")
        if token_ids.numel() and bool(
            ((token_ids < 0) | ((token_ids >= int(vocab_size)) & token_mask)).any()
        ):
            raise ValueError("Projected token ID is outside the effective vocabulary")

        logits = self._effective_logits(
            vocab_size=int(vocab_size), temperature=float(temperature)
        )
        log_normalizer = _vocab_parallel_log_normalizer(logits, self.group)
        selected_logits = _distributed_selected_logits(logits, token_ids, self.group)
        explicit_log_probs = (selected_logits - log_normalizer).masked_fill(
            ~token_mask, float("-inf")
        )

        local_vocab_size = logits.size(-1)
        local_start = _tp_rank(self.group) * local_vocab_size
        owned = (
            token_mask
            & (token_ids >= local_start)
            & (token_ids < local_start + local_vocab_size)
        )
        local_ids = (token_ids - local_start).clamp(
            min=0, max=local_vocab_size - 1
        )
        support_mask = torch.zeros_like(logits, dtype=torch.bool)
        support_mask.scatter_(1, local_ids, owned)
        complement_logits = logits.masked_fill(support_mask, float("-inf"))
        support_counts = token_mask.sum(dim=-1)
        if bool((support_counts > int(vocab_size)).any()):
            raise ValueError("Projected explicit support is wider than the effective vocabulary")
        complete_support = support_counts == int(vocab_size)
        # A complete explicit support has a mathematically empty complement.
        # Give the reduction a finite dummy row to avoid all-(-inf) arithmetic,
        # then restore the exact log(0) result. The plugin must assign zero
        # target mass to that coordinate.
        complement_logits = torch.where(
            complete_support.unsqueeze(-1),
            torch.zeros_like(complement_logits),
            complement_logits,
        )
        complement_log_normalizer = _vocab_parallel_log_normalizer(
            complement_logits, self.group
        )
        complement_log_prob = (complement_log_normalizer - log_normalizer).masked_fill(
            complete_support.unsqueeze(-1), float("-inf")
        )
        if not bool(torch.isfinite(complement_log_prob[~complete_support]).all()):
            raise FloatingPointError(
                "Projected Student complement contains non-finite logits"
            )
        return torch.cat((explicit_log_probs, complement_log_prob), dim=-1)

    def topk(self, k: int) -> CandidateScores:
        logits = self._materialized_logits()
        output_ids, _ = _distributed_topk(logits, k, self.group, with_log_probs=False)
        # Top-K ID selection is non-differentiable. Re-score those IDs through
        # the original logits so Student-side algorithms get correct gradients.
        log_probs = _distributed_selected_log_probs(logits, output_ids, self.group)
        return CandidateScores(output_ids, log_probs, torch.ones_like(output_ids, dtype=torch.bool))

    def mapped_topk(
        self,
        source_ids: torch.Tensor,
        output_ids: torch.Tensor,
        *,
        k: int,
        vocab_size: int | None = None,
        temperature: float = 1.0,
    ) -> CandidateScores:
        mapped_ids, log_probs, mask = _distributed_mapped_topk(
            self._effective_logits(vocab_size=vocab_size, temperature=temperature),
            source_ids.to(device=self.local_logits.device, dtype=torch.long),
            output_ids.to(device=self.local_logits.device, dtype=torch.long),
            k,
            self.group,
        )
        return CandidateScores(mapped_ids, log_probs, mask)


class _CausalHiddenVocabularyDistribution:
    """Lazy frozen-head vocabulary facade for transported Teacher states."""

    def __init__(
        self,
        hidden_states: torch.Tensor,
        head_shard: torch.Tensor,
        group: dist.ProcessGroup | None,
        row_indices: torch.Tensor | None = None,
    ) -> None:
        if hidden_states.ndim != 2 or head_shard.ndim != 2:
            raise ValueError("Causal-hidden Teacher states/head must both be rank two")
        if hidden_states.size(1) != head_shard.size(1):
            raise ValueError("Causal-hidden Teacher width differs from its frozen head")
        self.hidden_states = hidden_states
        self.head_shard = head_shard
        self.group = group
        self.row_indices = row_indices

    @property
    def length(self) -> int:
        return (
            self.hidden_states.size(0)
            if self.row_indices is None
            else self.row_indices.numel()
        )

    def rows(self, row_indices: torch.Tensor) -> "_CausalHiddenVocabularyDistribution":
        row_indices = row_indices.to(device=self.hidden_states.device, dtype=torch.long).flatten()
        if row_indices.numel() and bool(
            ((row_indices < 0) | (row_indices >= self.length)).any()
        ):
            raise ValueError("Causal-hidden Teacher row index is out of bounds")
        absolute_rows = (
            row_indices
            if self.row_indices is None
            else self.row_indices.index_select(0, row_indices)
        )
        return _CausalHiddenVocabularyDistribution(
            self.hidden_states, self.head_shard, self.group, absolute_rows
        )

    def _materialize(self) -> _MegatronVocabularyDistribution:
        hidden = (
            self.hidden_states
            if self.row_indices is None
            else self.hidden_states.index_select(0, self.row_indices)
        )
        logits = torch.nn.functional.linear(hidden, self.head_shard)
        return _MegatronVocabularyDistribution(logits, self.group)

    def selected_log_probs(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._materialize().selected_log_probs(*args, **kwargs)

    def selected_logits(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._materialize().selected_logits(*args, **kwargs)

    def full_probabilities(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        # BPM forms conditional prefix masses by summing large, overlapping
        # vocabulary subsets. Casting each probability back to BF16 can make
        # numerically nested events appear non-monotone. Keep this row-bounded
        # causal-hidden path in FP32; regular dense ULD/GOLD callers retain the
        # memory-saving backend default above.
        kwargs["preserve_float32"] = True
        return self._materialize().full_probabilities(*args, **kwargs)

    def projected_log_probs(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._materialize().projected_log_probs(*args, **kwargs)

    def topk(self, *args: Any, **kwargs: Any) -> CandidateScores:
        return self._materialize().topk(*args, **kwargs)

    def mapped_topk(self, *args: Any, **kwargs: Any) -> CandidateScores:
        return self._materialize().mapped_topk(*args, **kwargs)


def _build_cross_tokenizer_scoring_algorithm(args: Namespace):
    """Rebuild one scoring adapter from JSON-only CPU-exported state."""

    return build_cross_tokenizer(
        args.cross_tokenizer_algorithm,
        algorithm_kwargs=getattr(args, "cross_tokenizer_algorithm_kwargs", {}),
        runtime_state=getattr(args, "cross_tokenizer_runtime_state", {}),
    )


def get_opd_student_topk(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    opd_student_topk_ids: list[torch.Tensor] | None = None,
) -> dict[str, list[torch.Tensor]]:
    del with_entropy, non_loss_data, opd_student_topk_ids
    group = mpu.get_tensor_model_parallel_group()
    ids = []
    for response_logits, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        topk_ids, _ = _distributed_topk(
            response_logits, args.opd_top_k, group, with_log_probs=False
        )
        ids.append(topk_ids)
    return {"topk_ids": ids}


def get_opd_teacher_data(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    opd_student_topk_ids: list[torch.Tensor] | None = None,
) -> dict[str, list[torch.Tensor]]:
    del with_entropy, non_loss_data
    group = mpu.get_tensor_model_parallel_group()
    response_chunks = list(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )
    )
    if args.opd_loss_type == "union_kl" and opd_student_topk_ids is None:
        raise ValueError("union_kl requires the student Top-K probe before the teacher forward.")

    output_ids = []
    output_log_probs = []
    output_masks = []
    student_ids_iter = opd_student_topk_ids or [None] * len(response_chunks)
    for (response_logits, _), student_ids in zip(response_chunks, student_ids_iter, strict=True):
        teacher_ids, teacher_log_probs = _distributed_topk(response_logits, args.opd_top_k, group)
        assert teacher_log_probs is not None
        if args.opd_loss_type == "union_kl":
            student_ids = student_ids.to(device=response_logits.device, dtype=torch.long)
            union_ids, union_mask = merge_topk_union(teacher_ids, student_ids)
            teacher_log_probs = _distributed_selected_log_probs(response_logits, union_ids, group)
            output_ids.append(union_ids)
            output_log_probs.append(teacher_log_probs)
            output_masks.append(union_mask)
        elif args.opd_loss_type == "teacher_topk_gkl":
            output_ids.append(teacher_ids)
            output_log_probs.append(teacher_log_probs)
            output_masks.append(torch.ones_like(teacher_ids, dtype=torch.bool))
        else:
            raise ValueError(f"Unsupported Top-K OPD loss type: {args.opd_loss_type}")
    return {"opd_ids": output_ids, "opd_log_probs": output_log_probs, "opd_mask": output_masks}


def get_cross_tokenizer_teacher_data(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    cross_tokenizer_alignments: list[dict[str, Any]],
    cross_tokenizer_student_loss_masks: list[torch.Tensor],
    cross_tokenizer_student_response_token_ids: list[torch.Tensor] | None = None,
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    opd_student_topk_ids: list[torch.Tensor] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Build cross-tokenizer Teacher targets in Student response coordinates."""

    del with_entropy, non_loss_data, opd_student_topk_ids
    group = mpu.get_tensor_model_parallel_group()
    algorithm = _build_cross_tokenizer_scoring_algorithm(args)
    teacher_vocab_ids = torch.as_tensor(
        args.cross_tokenizer_teacher_vocab_ids, dtype=torch.long, device=logits.device
    )
    student_vocab_ids = torch.as_tensor(
        args.cross_tokenizer_student_vocab_ids, dtype=torch.long, device=logits.device
    )
    response_chunks = list(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
            # Cross-tokenizer algorithms receive native T=1 log-probabilities
            # and apply KD temperature in their own public supervision space.
            temperature=1.0,
        )
    )
    if cross_tokenizer_student_response_token_ids is None:
        cross_tokenizer_student_response_token_ids = [None] * len(cross_tokenizer_alignments)
    if len(cross_tokenizer_student_response_token_ids) != len(cross_tokenizer_alignments):
        raise ValueError("Student response IDs and cross-tokenizer alignments have different batch sizes")
    if not (
        len(response_chunks)
        == len(cross_tokenizer_alignments)
        == len(cross_tokenizer_student_loss_masks)
    ):
        raise ValueError("Teacher responses, alignments and Student loss masks have different batch sizes")

    mapping = VocabularyMapping(teacher_ids=teacher_vocab_ids, student_ids=student_vocab_ids)
    output_payloads: list[TensorPayload] = []
    output_skipped_masks: list[torch.Tensor] = []
    for (response_logits, response_tokens), raw_alignment, student_loss_mask, student_response_token_ids in zip(
        response_chunks,
        cross_tokenizer_alignments,
        cross_tokenizer_student_loss_masks,
        cross_tokenizer_student_response_token_ids,
        strict=True,
    ):
        alignment = AlignmentResult.from_dict(raw_alignment)
        if response_logits.size(0) != alignment.teacher_length:
            raise ValueError(
                f"Teacher response length {response_logits.size(0)} does not match alignment "
                f"length {alignment.teacher_length}"
            )
        student_loss_mask = student_loss_mask.to(
            device=response_logits.device, dtype=torch.bool
        ).flatten()
        if student_loss_mask.numel() != alignment.student_length:
            raise ValueError(
                f"Student loss-mask length {student_loss_mask.numel()} does not match alignment "
                f"length {alignment.student_length}"
            )
        if alignment.skipped:
            # Skip is a backend guarantee, not an algorithm convention. Do not
            # invoke plugin code for a sample that must contribute zero loss.
            payload = {}
        else:
            payload = algorithm.build_teacher_payload(
                TeacherScoringContext(
                    distribution=_MegatronVocabularyDistribution(response_logits, group),
                    response_token_ids=response_tokens,
                    student_response_token_ids=(
                        student_response_token_ids.to(device=response_logits.device, dtype=torch.long)
                        if isinstance(student_response_token_ids, torch.Tensor)
                        else None
                    ),
                    alignment=alignment,
                    vocabulary_mapping=mapping,
                    candidate_top_k=args.cross_tokenizer_top_k,
                    student_loss_mask=student_loss_mask,
                    temperature=args.cross_tokenizer_temperature,
                )
            )
        if not isinstance(payload, dict) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in payload.items()
        ):
            raise TypeError("Cross-tokenizer Teacher payload must be dict[str, torch.Tensor]")
        output_payloads.append({key: value.detach() for key, value in payload.items()})
        output_skipped_masks.append(
            torch.full(
                (alignment.student_length,),
                alignment.skipped,
                dtype=torch.bool,
                device=response_logits.device,
            )
        )

    return {
        "cross_payload": output_payloads,
        "cross_skipped_mask": output_skipped_masks,
    }


def compute_topk_opd_token_losses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    batch: RolloutBatch,
    teacher_ids: list[torch.Tensor],
    teacher_log_probs: list[torch.Tensor],
    teacher_masks: list[torch.Tensor],
) -> list[torch.Tensor]:
    group = mpu.get_tensor_model_parallel_group()
    response_chunks = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
    )
    losses = []
    for (response_logits, _), ids, teacher_lp, mask in zip(
        response_chunks, teacher_ids, teacher_log_probs, teacher_masks, strict=True
    ):
        ids = ids.to(device=response_logits.device, dtype=torch.long)
        teacher_lp = teacher_lp.to(device=response_logits.device, dtype=response_logits.dtype)
        mask = mask.to(device=response_logits.device, dtype=torch.bool)
        student_lp = _distributed_selected_log_probs(response_logits, ids, group)
        if args.opd_loss_type == "union_kl":
            token_loss = renormalized_teacher_student_kl(teacher_lp, student_lp, mask)
        elif args.opd_loss_type == "teacher_topk_gkl":
            token_loss = teacher_topk_generalized_kl(teacher_lp, student_lp, mask)
        else:
            raise ValueError(f"Unsupported Top-K OPD loss type: {args.opd_loss_type}")
        losses.append(token_loss)
    return losses


def _response_positions_for_cp(
    *,
    total_length: int,
    response_length: int,
    args: Namespace,
    device: torch.device,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Return global response offsets represented by this CP rank.

    Megatron's context-parallel layout assigns two zig-zag sequence chunks to
    each rank in both THD and BSHD modes. ``get_responses`` concatenates those
    two local response pieces; this function returns their positions in the
    original full Student response so opaque Teacher payload rows can be
    routed to their owner.
    """

    if mpu.get_context_parallel_world_size() == 1:
        return torch.arange(response_length, device=device, dtype=torch.long)
    _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
        total_length,
        response_length,
        args.qkv_format,
        max_seq_len,
    )
    prompt_length = total_length - response_length
    parts = [
        torch.arange(
            start - prompt_length,
            end - prompt_length,
            device=device,
            dtype=torch.long,
        )
        for start, end in token_offsets
        if end > start
    ]
    return (
        torch.cat(parts, dim=0)
        if parts
        else torch.empty((0,), device=device, dtype=torch.long)
    )


def _causal_hidden_teacher_distribution(
    hidden_states: torch.Tensor,
    response_token_ids: torch.Tensor,
    observed_log_probs: torch.Tensor | None,
    *,
    expected_rows: int,
) -> _CausalHiddenVocabularyDistribution:
    """Build and once validate the generic causal-hidden Teacher facade."""

    if (
        _CROSS_TOKENIZER_TEACHER_HEAD_SHARD is None
        or _CROSS_TOKENIZER_TEACHER_VOCAB_SIZE is None
    ):
        raise RuntimeError("The causal-hidden cross-tokenizer Teacher head was not initialized")
    hidden = hidden_states.to(
        device=_CROSS_TOKENIZER_TEACHER_HEAD_SHARD.device,
        dtype=_CROSS_TOKENIZER_TEACHER_HEAD_SHARD.dtype,
    )
    if hidden.ndim != 2 or hidden.size(0) != int(expected_rows):
        raise ValueError("Causal-hidden Teacher rows do not match the alignment")
    token_ids = response_token_ids.to(device=hidden.device, dtype=torch.long).flatten()
    if token_ids.numel() != hidden.size(0):
        raise ValueError("Causal-hidden Teacher token IDs do not match transported rows")
    distribution = _CausalHiddenVocabularyDistribution(
        hidden,
        _CROSS_TOKENIZER_TEACHER_HEAD_SHARD,
        mpu.get_tensor_model_parallel_group(),
    )

    global _CROSS_TOKENIZER_TEACHER_VALIDATED
    if observed_log_probs is not None and not _CROSS_TOKENIZER_TEACHER_VALIDATED:
        observed = observed_log_probs.to(device=hidden.device, dtype=torch.float32).flatten()
        if observed.numel() != hidden.size(0) or not bool(torch.isfinite(observed).all()):
            raise ValueError("Causal-hidden Teacher observed log-probabilities are invalid")
        validation_count = min(8, hidden.size(0))
        validation_positions = torch.linspace(
            0,
            hidden.size(0) - 1,
            steps=validation_count,
            device=hidden.device,
        ).round().to(dtype=torch.long).unique(sorted=True)
        with torch.no_grad():
            reconstructed = distribution.rows(validation_positions).selected_log_probs(
                token_ids.index_select(0, validation_positions).unsqueeze(-1),
                vocab_size=_CROSS_TOKENIZER_TEACHER_VOCAB_SIZE,
                temperature=1.0,
            ).squeeze(-1).float()
        absolute_error = (
            reconstructed - observed.index_select(0, validation_positions)
        ).abs()
        max_tolerance = float(
            os.environ.get(
                "OPD_CROSS_TOKENIZER_HEAD_LOGPROB_ATOL",
                os.environ.get("OPD_BPM_HEAD_LOGPROB_ATOL", "0.25"),
            )
        )
        mean_tolerance = float(
            os.environ.get(
                "OPD_CROSS_TOKENIZER_HEAD_LOGPROB_MEAN_ATOL",
                os.environ.get("OPD_BPM_HEAD_LOGPROB_MEAN_ATOL", "0.06"),
            )
        )
        if not math.isfinite(max_tolerance) or max_tolerance <= 0:
            raise ValueError("Teacher head maximum log-probability tolerance must be positive and finite")
        if not math.isfinite(mean_tolerance) or mean_tolerance <= 0:
            raise ValueError("Teacher head mean log-probability tolerance must be positive and finite")
        if not bool(torch.isfinite(absolute_error).all()):
            raise RuntimeError("Causal-hidden Teacher self-check produced a non-finite error")
        max_error = float(absolute_error.max())
        mean_error = float(absolute_error.mean())
        if max_error > max_tolerance or mean_error > mean_tolerance:
            raise RuntimeError(
                "Causal-hidden Teacher self-check failed: "
                f"max_abs_logprob_error={max_error:.6f}, "
                f"mean_abs_logprob_error={mean_error:.6f}, "
                f"max_tolerance={max_tolerance}, mean_tolerance={mean_tolerance}. "
                "The endpoint checkpoint, causal shift, "
                "hidden-state layer, or frozen LM head is inconsistent."
            )
        _CROSS_TOKENIZER_TEACHER_VALIDATED = True
        logger.info(
            "Causal-hidden Teacher self-check passed: max_abs_logprob_error=%.6f, mean_abs_logprob_error=%.6f",
            max_error,
            mean_error,
        )
    return distribution


def compute_cross_tokenizer_opd_token_losses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    batch: RolloutBatch,
    teacher_payloads: list[TensorPayload],
    skipped_masks: list[torch.Tensor],
    return_diagnostics: bool = False,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], dict[str, list[torch.Tensor]]]:
    """Delegate Student-side scoring to the registered algorithm."""

    group = mpu.get_tensor_model_parallel_group()
    algorithm = _build_cross_tokenizer_scoring_algorithm(args)
    capabilities = algorithm.capabilities
    teacher_transport = capabilities.teacher_transport(args.opd_type)
    if capabilities.requires_loss_token_normalization:
        if not args.calculate_per_token_loss:
            raise ValueError(
                f"Cross-tokenizer algorithm {algorithm.name!r} requires token-normalized loss"
            )
    if capabilities.requires_full_response_loss_mask:
        if len(skipped_masks) != len(batch["loss_masks"]):
            raise ValueError(
                "Cross-tokenizer skipped-mask and Student loss-mask batch sizes differ"
            )
        for sample_index, (loss_mask, response_length, skipped_mask) in enumerate(
            zip(
                batch["loss_masks"],
                batch["response_lengths"],
                skipped_masks,
                strict=True,
            )
        ):
            mask = loss_mask.to(dtype=torch.bool).flatten()
            skipped = skipped_mask.to(dtype=torch.bool).flatten()
            if skipped.numel() != int(response_length):
                raise ValueError(
                    f"Cross-tokenizer skipped mask length {skipped.numel()} differs from "
                    f"Student response length {response_length} (sample {sample_index})"
                )
            # A fully skipped alignment is deliberately removed from the
            # policy batch (its framework loss mask is all zero) and carries
            # no Teacher target.  It must not trip the BPM/SimCT invariant
            # below, which applies only to samples that do have a target.
            if bool(skipped.numel()) and bool(skipped.all()):
                continue
            if mask.numel() != int(response_length) or not bool(mask.all()):
                raise ValueError(
                    f"Cross-tokenizer algorithm {algorithm.name!r} requires every realized Student response "
                    f"token in its framework loss mask (sample {sample_index})"
                )
    response_chunks = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
        # Keep the model facade at T=1. The opaque algorithm owns the meaning
        # of temperature after constructing its public-space scores.
        temperature=1.0,
    )
    cp_size = mpu.get_context_parallel_world_size()
    max_seq_lens = batch.get("max_seq_lens")
    teacher_hiddens = batch.get("teacher_cross_hidden_states")
    teacher_token_ids = batch.get("teacher_cross_response_token_ids")
    teacher_observed_log_probs = batch.get("teacher_cross_observed_log_probs")
    teacher_alignments = batch.get("teacher_cross_alignments")
    causal_hidden_mode = teacher_transport == "causal_hidden"
    step_active_token_counts = batch.get("step_active_token_counts")
    step_sample_counts = batch.get("step_sample_counts")
    if (step_active_token_counts is None) != (step_sample_counts is None):
        raise ValueError("Cross-tokenizer step normalization metadata is incomplete")
    if step_active_token_counts is not None and not (
        len(step_active_token_counts)
        == len(step_sample_counts)
        == len(batch["response_lengths"])
    ):
        raise ValueError("Cross-tokenizer step normalization metadata has a different batch size")
    if causal_hidden_mode:
        if not (
            teacher_hiddens is not None
            and teacher_token_ids is not None
            and teacher_observed_log_probs is not None
            and teacher_alignments is not None
            and len(teacher_hiddens)
            == len(teacher_token_ids)
            == len(teacher_observed_log_probs)
            == len(teacher_alignments)
            == len(batch["response_lengths"])
        ):
            raise ValueError("Causal-hidden Teacher transport fields have different batch sizes")
    losses: list[torch.Tensor] = []
    diagnostic_names = algorithm.training_diagnostic_names()
    diagnostics: dict[str, list[torch.Tensor]] = {name: [] for name in diagnostic_names}

    def append_empty_diagnostics(length: int, device: torch.device) -> None:
        for name in diagnostic_names:
            diagnostics[name].append(torch.zeros(length, dtype=torch.float32, device=device))

    def append_algorithm_diagnostics(
        values: dict[str, torch.Tensor], length: int
    ) -> None:
        if set(values) != set(diagnostic_names):
            raise ValueError(
                "Cross-tokenizer algorithm diagnostics do not match its declared names: "
                f"expected={diagnostic_names}, got={tuple(values)}"
            )
        for name in diagnostic_names:
            value = values[name]
            if value.ndim != 1 or value.numel() != length:
                raise ValueError(
                    f"Cross-tokenizer diagnostic {name!r} must have one value per local response token"
                )
            diagnostics[name].append(value.to(dtype=torch.float32))

    for sample_index, (
        (response_logits, response_tokens),
        teacher_payload,
        skipped_mask,
        total_length,
        response_length,
    ) in enumerate(zip(
        response_chunks,
        teacher_payloads,
        skipped_masks,
        batch["total_lengths"],
        batch["response_lengths"],
        strict=True,
    )):
        skipped_mask = skipped_mask.to(device=response_logits.device, dtype=torch.bool).flatten()
        if skipped_mask.numel() != response_length:
            raise ValueError("Cross-tokenizer skipped mask must match the full Student response length")
        if bool(skipped_mask.any()):
            if not bool(skipped_mask.all()):
                raise ValueError("Cross-tokenizer skipped mask must cover either the whole sample or none of it")
            losses.append(response_logits[:, 0] * 0.0)
            append_empty_diagnostics(response_logits.size(0), response_logits.device)
            continue
        student_context = StudentScoringContext(
            distribution=_MegatronVocabularyDistribution(response_logits, group),
            response_token_ids=response_tokens,
            temperature=args.cross_tokenizer_temperature,
            step_active_token_count=(
                int(step_active_token_counts[sample_index])
                if step_active_token_counts is not None
                else None
            ),
            step_sample_count=(
                int(step_sample_counts[sample_index])
                if step_sample_counts is not None
                else None
            ),
        )
        if causal_hidden_mode:
            if cp_size > 1:
                response_positions = _response_positions_for_cp(
                    total_length=total_length,
                    response_length=response_length,
                    args=args,
                    device=response_logits.device,
                    max_seq_len=max_seq_lens[sample_index] if max_seq_lens is not None else None,
                )
            else:
                response_positions = torch.arange(
                    response_length, device=response_logits.device, dtype=torch.long
                )
            if response_positions.numel() != response_logits.size(0):
                raise ValueError("Cross-tokenizer Student response ownership differs from local logits")
            alignment = AlignmentResult.from_dict(teacher_alignments[sample_index])
            teacher_distribution = _causal_hidden_teacher_distribution(
                teacher_hiddens[sample_index],
                teacher_token_ids[sample_index],
                teacher_observed_log_probs[sample_index],
                expected_rows=alignment.teacher_length,
            )
            token_losses = algorithm.compute_fused_token_losses(
                FusedScoringContext(
                    student=student_context,
                    teacher_distribution=teacher_distribution,
                    teacher_response_token_ids=teacher_token_ids[sample_index],
                    teacher_observed_log_probs=teacher_observed_log_probs[sample_index],
                    alignment=alignment,
                    local_response_positions=response_positions,
                    vocabulary_mapping=VocabularyMapping(
                        teacher_ids=torch.as_tensor(
                            args.cross_tokenizer_teacher_vocab_ids,
                            device=response_logits.device,
                            dtype=torch.long,
                        ),
                        student_ids=torch.as_tensor(
                            args.cross_tokenizer_student_vocab_ids,
                            device=response_logits.device,
                            dtype=torch.long,
                        ),
                    ),
                    gather_student_rows=lambda rows, positions=response_positions, length=response_length: (
                        _gather_context_parallel_response_rows(
                            rows,
                            local_positions=positions,
                            response_length=length,
                        )
                    ),
                    candidate_top_k=args.cross_tokenizer_top_k,
                    student_response_token_ids=batch["unconcat_tokens"][sample_index][
                        -int(response_length) :
                    ].to(device=response_logits.device, dtype=torch.long),
                )
            )
            if token_losses.ndim != 1 or token_losses.numel() != response_logits.size(0):
                raise ValueError(
                    "Cross-tokenizer algorithms must return one token loss per local Student response position"
                )
            losses.append(token_losses)
            append_algorithm_diagnostics(
                algorithm.pop_training_diagnostics(), response_logits.size(0)
            )
            continue
        if not isinstance(teacher_payload, dict) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in teacher_payload.items()
        ):
            raise TypeError("Cross-tokenizer Teacher payload must be dict[str, torch.Tensor]")
        moved_payload = {
            key: value.to(device=response_logits.device) for key, value in teacher_payload.items()
        }
        if cp_size > 1:
            if not capabilities.supports_context_parallel_for(args.opd_type):
                raise ValueError(
                    f"Cross-tokenizer algorithm {args.cross_tokenizer_algorithm!r} does not support "
                    "Context Parallel under its declared algorithm contract"
                )
            response_positions = _response_positions_for_cp(
                total_length=total_length,
                response_length=response_length,
                args=args,
                device=response_logits.device,
                max_seq_len=max_seq_lens[sample_index] if max_seq_lens is not None else None,
            )
            if response_positions.numel() != response_logits.size(0):
                raise ValueError(
                    "Cross-tokenizer CP response offsets do not match local logits rows: "
                    f"offsets={response_positions.numel()}, logits={response_logits.size(0)}"
                )
            assert capabilities.payload_position_field is not None
            moved_payload = localize_payload_by_student_positions(
                moved_payload,
                position_field=capabilities.payload_position_field,
                response_positions=response_positions,
                response_length=response_length,
            )
        token_losses = algorithm.compute_student_token_losses(
            student_context,
            moved_payload,
        )
        if token_losses.ndim != 1 or token_losses.numel() != response_logits.size(0):
            raise ValueError(
                "Cross-tokenizer algorithms must return one token loss per Student response position"
            )
        losses.append(token_losses)
        if diagnostic_names:
            append_algorithm_diagnostics(
                algorithm.pop_training_diagnostics(), response_logits.size(0)
            )
    if return_diagnostics:
        return losses, diagnostics
    return losses


def _cross_tokenizer_loss_anchor_mask(
    teacher_payloads: list[TensorPayload],
    skipped_masks: list[torch.Tensor],
    *,
    position_field: str | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Return response positions that receive an explicit cross-tokenizer loss.

    ``cross_tokenizer_skipped_token_ratio`` only describes samples rejected by
    the transport/alignment layer.  It intentionally does not include
    algorithm-level filtering such as BPM whitespace/spanning masks.  Both
    built-in policies expose the positions at which their sparse losses are
    anchored, so report that distinct quantity without changing skip
    semantics.  Returning ``None`` keeps third-party policies that do not
    expose anchors backward compatible.
    """

    if position_field is None:
        return None
    masks: list[torch.Tensor] = []
    for teacher_payload, skipped_mask in zip(teacher_payloads, skipped_masks, strict=True):
        skipped_mask = skipped_mask.to(device=device, dtype=torch.bool).flatten()
        if position_field not in teacher_payload:
            return None
        positions = teacher_payload[position_field]
        positions = positions.to(device=device, dtype=torch.long).flatten()
        if positions.numel() and bool(((positions < 0) | (positions >= skipped_mask.numel())).any()):
            raise ValueError("Cross-tokenizer Teacher payload contains an invalid loss-anchor position")
        mask = torch.zeros(skipped_mask.numel(), device=device, dtype=torch.float32)
        if not bool(skipped_mask.all()) and positions.numel():
            mask[positions] = 1.0
        masks.append(mask)
    return torch.cat(masks, dim=0)


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
    temperature: float | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing the default `rollout_temperature`.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.
        temperature: Optional scoring-only override. Cross-tokenizer KD passes
            ``1.0`` here, then applies its independent temperature after the
            algorithm constructs public-space scores.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    qkv_format = args.qkv_format

    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    score_temperature = args.rollout_temperature if temperature is None else float(temperature)
    if score_temperature <= 0:
        raise ValueError(f"Scoring temperature must be positive, got {score_temperature}")
    if score_temperature != 1.0:
        logits = logits.div(score_temperature)

    cp_size = mpu.get_context_parallel_world_size()
    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        if response_length <= 0:
            raise ValueError(f"Sample {i} has empty response; at least one response token is required")
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
            else:
                end += total_length
                start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = logits.size(0)
            cp_rank = mpu.get_context_parallel_rank()
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        yield logits_chunk, tokens_chunk


def _allgather_cp_redistribute(
    res: dict[str, list[torch.Tensor]],
    *,
    logits_local_len: int,
    args: Namespace,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> None:
    """Redistribute response tensors from allgather-CP layout to zigzag ring-attn layout.

    After allgather context parallelism, each rank holds a contiguous chunk of
    the global sequence.  This helper reconstructs per-sample full response
    tensors via a differentiable all-reduce and re-slices them into the zigzag
    CP pattern expected by downstream code.

    The *res* dict is modified **in-place**.

    Args:
        res: Dict mapping metric names to lists of per-sample tensors.
        logits_local_len: Local sequence length on this rank.
        args: Configuration (needs ``qkv_format``).
        total_lengths: Total sequence lengths (prompt + response) per sample.
        response_lengths: Response segment lengths per sample.
        max_seq_lens: Optional padded max sequence lengths per sample.
    """
    cp_group = mpu.get_context_parallel_group()
    cp_rank = mpu.get_context_parallel_rank()
    chunk_start = cp_rank * logits_local_len
    chunk_end = chunk_start + logits_local_len

    for key, values in res.items():
        # Skip keys where all values are None (e.g. entropy when not computed)
        if all(v is None for v in values):
            continue

        # Determine reference dtype/device from first non-None value
        ref_value = next(v for v in values if v is not None)
        ref_dtype = ref_value.dtype
        ref_device = ref_value.device

        # Reconstruct full response tensors with each rank's contiguous contribution
        full_resps = []
        seq_start = 0
        for value, total_length, response_length in zip(values, total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)

            if value is None or e <= s:
                # This rank has no response logprobs for this sample
                full_resp = torch.zeros(
                    response_length,
                    dtype=ref_dtype,
                    device=ref_device,
                    requires_grad=True,
                )
            else:
                resp_start = s - logit_global_start
                resp_end = e - logit_global_start
                full_resp = F.pad(value, (resp_start, response_length - resp_end))

            assert full_resp.size(0) == response_length, f"Expected {response_length}, got {full_resp.size(0)}"
            full_resps.append(full_resp)
            seq_start += total_length

        # Single differentiable all-reduce to gather full response from all CP ranks
        all_cat = torch.cat(full_resps, dim=0)
        all_cat = dist.nn.all_reduce(all_cat, group=cp_group)

        # Re-slice each sample into zigzag CP pattern
        new_values = []
        for idx, (full_resp, total_length, response_length) in enumerate(
            zip(all_cat.split(response_lengths, dim=0), total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
            new_values.append(
                slice_log_prob_with_cp(full_resp, total_length, response_length, args.qkv_format, max_seq_len)
            )

        res[key] = new_values


def _cp_zigzag_response_layouts(
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
) -> tuple[list[tuple[int, tuple, tuple, tuple, tuple[int, int]]], int, int]:
    """Return per-sample CP zigzag offsets plus padded/compact local lengths."""
    layouts = []
    padded_T = 0
    compact_T = 0

    for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=False)):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
        chunk_size_cp, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
            total_length, response_length, qkv_format, max_seq_len
        )
        token_lengths = (
            tokens_offset[0][1] - tokens_offset[0][0],
            tokens_offset[1][1] - tokens_offset[1][0],
        )
        layouts.append((chunk_size_cp, chunks_offset, logits_offset, tokens_offset, token_lengths))
        padded_T += 2 * chunk_size_cp
        compact_T += token_lengths[0] + token_lengths[1]

    return layouts, padded_T, compact_T


def _infer_cp_response_layout(
    values: list[torch.Tensor] | None,
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    allgather_cp: bool,
) -> str:
    """Infer whether per-sample response tensors are full zigzag or a single CP shard."""
    if not values or mpu.get_context_parallel_world_size() <= 1 or allgather_cp:
        return "default"

    layouts, _padded_T, _compact_T = _cp_zigzag_response_layouts(
        total_lengths, response_lengths, qkv_format, max_seq_lens
    )
    actual_lengths = [value.size(0) for value in values]
    single_lengths = [layout[4][0] for layout in layouts]
    if actual_lengths == single_lengths:
        return "single_shard"
    return "zigzag"


def _align_cp_response_values(
    values: list[torch.Tensor] | None,
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    allgather_cp: bool,
    target_layout: str,
    name: str,
) -> list[torch.Tensor] | None:
    """Align auxiliary response tensors to the student log-prob CP layout.

    Most rollout fields are already sliced before they reach this function,
    while cross-tokenizer masks intentionally stay in full Student-response
    coordinates.  Handle both inputs: preserve tensors that already match the
    local layout and slice full-response tensors to the one- or two-chunk CP
    ownership represented by the model logits.
    """
    if not values or target_layout == "default" or mpu.get_context_parallel_world_size() <= 1 or allgather_cp:
        return values

    layouts, _padded_T, _compact_T = _cp_zigzag_response_layouts(
        total_lengths, response_lengths, qkv_format, max_seq_lens
    )
    aligned: list[torch.Tensor] = []
    for sample_idx, (value, total_length, response_length, layout) in enumerate(
        zip(values, total_lengths, response_lengths, layouts, strict=False)
    ):
        _chunk_size_cp, _chunks_offset, _logits_offset, tokens_offset, token_lengths = layout
        target_len = token_lengths[0] if target_layout == "single_shard" else sum(token_lengths)
        zigzag_len = token_lengths[0] + token_lengths[1]
        if target_len == 0:
            aligned.append(value[:0])
            continue
        if value.size(0) == target_len:
            aligned.append(value)
            continue
        if target_layout == "single_shard" and value.size(0) == zigzag_len:
            aligned.append(value[:target_len])
            continue
        if value.size(0) == response_length:
            prompt_length = total_length - response_length
            halves = (0,) if target_layout == "single_shard" else (0, 1)
            parts = []
            for half in halves:
                start = tokens_offset[half][0] - prompt_length
                end = tokens_offset[half][1] - prompt_length
                if end > start:
                    parts.append(value[start:end])
            aligned.append(torch.cat(parts, dim=0) if parts else value[:0])
            continue
        raise ValueError(
            f"[CP response layout] cannot align {name} at sample {sample_idx}: "
            f"value={value.size(0)}, target={target_len}, target_layout={target_layout}, zigzag={zigzag_len}, "
            f"response_length={response_length}, qkv_format={qkv_format}, "
            f"max_seq_len={max_seq_lens[sample_idx] if max_seq_lens else None}"
        )
    return aligned


def _build_shifted_tokens(
    T: int,
    device: torch.device,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    allgather_cp: bool,
) -> torch.Tensor:
    """Build shifted target tokens for the full packed/padded logits."""
    cp_size = mpu.get_context_parallel_world_size()

    # --- zigzag CP: completely different layout ---
    if cp_size > 1 and not allgather_cp:
        layouts, padded_T, compact_T = _cp_zigzag_response_layouts(
            total_lengths, response_lengths, qkv_format, max_seq_lens
        )
        single_T = sum(layout[0] for layout in layouts)
        full_tokens = torch.zeros(T, dtype=torch.long, device=device)

        # Some model providers return only logits selected by loss_mask. In
        # that compact layout logits are concatenated as local response chunks
        # rather than placed in the padded zigzag sequence slots.
        if T == compact_T:
            pos = 0
            for tokens, (_chunk_size_cp, _chunks_offset, _logits_offset, tokens_offset, token_lengths) in zip(
                unconcat_tokens, layouts, strict=False
            ):
                for half, length in enumerate(token_lengths):
                    if length <= 0:
                        continue
                    src_start, src_end = tokens_offset[half]
                    full_tokens[pos : pos + length] = tokens[src_start:src_end]
                    pos += length
            return full_tokens

        # Qwen3-VL with BSHD context parallelism returns a single local CP shard
        # even though the input token stream is the usual two-chunk zigzag form.
        if T == single_T:
            pos = 0
            for sample_idx, (tokens, layout) in enumerate(zip(unconcat_tokens, layouts, strict=False)):
                chunk_size_cp, chunks_offset, logits_offset, tokens_offset, _token_lengths = layout
                lo = logits_offset[0][0] - chunks_offset[0][0]
                hi = logits_offset[0][1] - chunks_offset[0][0]
                src_start, src_end = tokens_offset[0]
                src = tokens[src_start:src_end]
                dst_start = pos + lo
                dst_end = pos + hi
                dst = full_tokens[dst_start:dst_end]
                if dst.numel() != src.numel():
                    raise RuntimeError(
                        "[CP shifted tokens] single-shard logits/token layout mismatch: "
                        f"T={T}, single_T={single_T}, compact_T={compact_T}, padded_T={padded_T}, "
                        f"sample={sample_idx}, dst=({dst_start},{dst_end})/{dst.numel()}, "
                        f"src=({src_start},{src_end})/{src.numel()}, "
                        f"total_length={total_lengths[sample_idx]}, response_length={response_lengths[sample_idx]}, "
                        f"qkv_format={qkv_format}, max_seq_len={max_seq_lens[sample_idx] if max_seq_lens else None}"
                    )
                full_tokens[dst_start:dst_end] = src
                pos += chunk_size_cp
            return full_tokens

        end = 0
        for sample_idx, (tokens, layout) in enumerate(zip(unconcat_tokens, layouts, strict=False)):
            chunk_size_cp, chunks_offset, logits_offset, tokens_offset, _token_lengths = layout
            for half, base in ((0, end), (1, end + chunk_size_cp)):
                lo = logits_offset[half][0] - chunks_offset[half][0]
                hi = logits_offset[half][1] - chunks_offset[half][0]
                src_start, src_end = tokens_offset[half]
                src = tokens[src_start:src_end]
                dst_start = base + lo
                dst_end = base + hi
                dst = full_tokens[dst_start:dst_end]
                if dst.numel() != src.numel():
                    raise RuntimeError(
                        "[CP shifted tokens] logits/token layout mismatch: "
                        f"T={T}, single_T={single_T}, compact_T={compact_T}, padded_T={padded_T}, "
                        f"sample={sample_idx}, half={half}, dst=({dst_start},{dst_end})/{dst.numel()}, "
                        f"src=({src_start},{src_end})/{src.numel()}, "
                        f"total_length={total_lengths[sample_idx]}, response_length={response_lengths[sample_idx]}, "
                        f"qkv_format={qkv_format}, max_seq_len={max_seq_lens[sample_idx] if max_seq_lens else None}"
                    )
                full_tokens[dst_start:dst_end] = src
            end += 2 * chunk_size_cp
        return full_tokens

    # --- cp1 and allgather-CP both build global shifted tokens the same way ---
    T_global = sum(total_lengths) if allgather_cp else T
    full_tokens = torch.zeros(T_global, dtype=torch.long, device=device)

    if qkv_format == "thd" or allgather_cp:
        offset = 0
        for tokens, total_length in zip(unconcat_tokens, total_lengths, strict=False):
            full_tokens[offset : offset + total_length - 1] = tokens[1:total_length]
            offset += total_length
    else:  # bshd, cp1
        for i, (tokens, total_length) in enumerate(zip(unconcat_tokens, total_lengths, strict=False)):
            seq_start = max_seq_lens[i] * i
            full_tokens[seq_start : seq_start + total_length - 1] = tokens[1:total_length]

    # allgather-CP: slice to local chunk
    if allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        chunk_start = cp_rank * T
        chunk_end = chunk_start + T
        if chunk_end <= T_global:
            return full_tokens[chunk_start:chunk_end].contiguous()
        local = torch.zeros(T, dtype=torch.long, device=device)
        valid = T_global - chunk_start
        if valid > 0:
            local[:valid] = full_tokens[chunk_start:]
        return local

    return full_tokens


def _extract_per_sample(
    log_prob_full: torch.Tensor,
    entropy_full: torch.Tensor | None,
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    allgather_cp: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Slice per-sample response log-probs/entropy from full-length 1-D tensors."""
    cp_size = mpu.get_context_parallel_world_size()
    log_probs_list: list[torch.Tensor] = []
    entropy_list: list[torch.Tensor] = []

    if cp_size > 1 and not allgather_cp:
        # zigzag CP
        layouts, _padded_T, compact_T = _cp_zigzag_response_layouts(
            total_lengths, response_lengths, qkv_format, max_seq_lens
        )
        single_T = sum(layout[0] for layout in layouts)
        if log_prob_full.size(0) == compact_T:
            pos = 0
            for _chunk_size_cp, _chunks_offset, _logits_offset, _tokens_offset, token_lengths in layouts:
                parts = []
                ent_parts = []
                for length in token_lengths:
                    if length > 0:
                        parts.append(log_prob_full[pos : pos + length])
                        if entropy_full is not None:
                            ent_parts.append(entropy_full[pos : pos + length])
                    pos += length
                if parts:
                    log_probs_list.append(torch.cat(parts, dim=0))
                    if entropy_full is not None:
                        entropy_list.append(torch.cat(ent_parts, dim=0))
                else:
                    log_probs_list.append(torch.zeros((0,), dtype=log_prob_full.dtype, device=log_prob_full.device))
                    if entropy_full is not None:
                        entropy_list.append(
                            torch.zeros((0,), dtype=entropy_full.dtype, device=entropy_full.device)
                        )
            return log_probs_list, entropy_list

        if log_prob_full.size(0) == single_T:
            pos = 0
            for chunk_size_cp, chunks_offset, logits_offset, _tokens_offset, _token_lengths in layouts:
                lo0 = logits_offset[0][0] - chunks_offset[0][0]
                hi0 = logits_offset[0][1] - chunks_offset[0][0]
                log_probs_list.append(log_prob_full[pos + lo0 : pos + hi0])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[pos + lo0 : pos + hi0])
                pos += chunk_size_cp
            return log_probs_list, entropy_list

        pos = 0
        for chunk_size_cp, chunks_offset, logits_offset, _tokens_offset, _token_lengths in layouts:
            lo0 = logits_offset[0][0] - chunks_offset[0][0]
            hi0 = logits_offset[0][1] - chunks_offset[0][0]
            lo1 = logits_offset[1][0] - chunks_offset[1][0]
            hi1 = logits_offset[1][1] - chunks_offset[1][0]

            lp = torch.cat(
                [
                    log_prob_full[pos + lo0 : pos + hi0],
                    log_prob_full[pos + chunk_size_cp + lo1 : pos + chunk_size_cp + hi1],
                ],
                dim=0,
            )
            log_probs_list.append(lp)
            if entropy_full is not None:
                ent = torch.cat(
                    [
                        entropy_full[pos + lo0 : pos + hi0],
                        entropy_full[pos + chunk_size_cp + lo1 : pos + chunk_size_cp + hi1],
                    ],
                    dim=0,
                )
                entropy_list.append(ent)
            pos += 2 * chunk_size_cp

    elif allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        local_len = log_prob_full.size(0)
        chunk_start = cp_rank * local_len
        chunk_end = chunk_start + local_len

        seq_start = 0
        for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                log_probs_list.append(torch.zeros((0,), dtype=log_prob_full.dtype, device=log_prob_full.device))
                if entropy_full is not None:
                    entropy_list.append(torch.zeros((0,), dtype=entropy_full.dtype, device=entropy_full.device))
            else:
                log_probs_list.append(log_prob_full[s - chunk_start : e - chunk_start])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[s - chunk_start : e - chunk_start])
            seq_start += total_length

    else:
        # cp1
        if qkv_format == "thd":
            offset = 0
            for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
                end = offset + total_length
                start = end - response_length
                log_probs_list.append(log_prob_full[start - 1 : end - 1])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[start - 1 : end - 1])
                offset += total_length
        else:  # bshd
            for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=False)):
                end = max_seq_lens[i] * i + total_length
                start = end - response_length
                log_probs_list.append(log_prob_full[start - 1 : end - 1])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[start - 1 : end - 1])

    return log_probs_list, entropy_list


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    Computes on the **full** logits ``[T, V]`` tensor at once (instead of
    per-sample slicing) so backward traverses ``[T, V]`` only once, then
    extracts per-sample response portions.

    When ``entropy_coef == 0``, entropy is computed under ``torch.no_grad()``
    to avoid retaining the computation graph and to skip cloning.
    """
    assert non_loss_data
    qkv_format = args.qkv_format

    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    # Apply rollout temperature scaling to logits to match rollout-time log-probs.
    rollout_temperature = getattr(args, "rollout_temperature", 1.0)
    if rollout_temperature != 1.0:
        logits = logits / rollout_temperature
    logits = logits.contiguous()
    T = logits.size(0)
    device = logits.device
    tp_group = mpu.get_tensor_model_parallel_group()
    chunk_size = args.log_probs_chunk_size

    # --- build full shifted-token target tensor ---
    full_tokens = _build_shifted_tokens(
        T, device, unconcat_tokens, total_lengths, response_lengths, qkv_format, max_seq_lens, args.allgather_cp
    )

    # --- compute on full [T,V] logits at once via calculate_log_probs_and_entropy ---
    log_prob_full, entropy_full = calculate_log_probs_and_entropy(
        logits,
        full_tokens,
        tp_group,
        with_entropy=with_entropy,
        chunk_size=chunk_size,
    )
    log_prob_full = log_prob_full.squeeze(-1)  # [T, 1] -> [T]

    # --- extract per-sample response portions ---
    log_probs_list, entropy_list = _extract_per_sample(
        log_prob_full,
        entropy_full,
        total_lengths,
        response_lengths,
        qkv_format,
        max_seq_lens,
        args.allgather_cp,
    )

    res = {"log_probs": log_probs_list}
    if with_entropy:
        res["entropy"] = entropy_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits_local_len=T,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return torch.empty((0,), device=device), res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits_local_len=logits.size(1),
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return torch.empty((0,), device=logits.device), res


def apply_opd_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:
    """Apply on-policy distillation KL penalty to advantages.

    Computes reverse KL (student_logp - teacher_logp) and adds weighted penalty
    to advantages in-place. This is orthogonal to the base advantage estimator.

    Args:
        args: Configuration containing `use_opd` and `opd_kl_coef`.
        rollout_data: Dict containing "teacher_log_probs".
        advantages: List of advantage tensors to modify in-place.
        student_log_probs: List of student log-probability tensors.

    References:
        https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py
    """

    if getattr(args, "use_cross_tokenizer", False) or getattr(args, "opd_loss_type", "sampled") != "sampled":
        return
    if student_log_probs is None:
        return

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")

    device = student_log_probs[0].device
    total_lengths = rollout_data.get("total_lengths")
    response_lengths = rollout_data.get("response_lengths")
    if total_lengths and response_lengths:
        target_layout = _infer_cp_response_layout(
            student_log_probs,
            total_lengths,
            response_lengths,
            args.qkv_format,
            rollout_data.get("max_seq_lens", None),
            getattr(args, "allgather_cp", False),
        )
        teacher_log_probs = _align_cp_response_values(
            teacher_log_probs,
            total_lengths,
            response_lengths,
            args.qkv_format,
            rollout_data.get("max_seq_lens", None),
            getattr(args, "allgather_cp", False),
            target_layout,
            "teacher_log_probs",
        )

    reverse_kls = []
    valid_masks = []
    coef = float(getattr(args, "opd_kl_coef", 1.0))
    for i, (adv, student_log_prob, teacher_log_prob) in enumerate(
        zip(advantages, student_log_probs, teacher_log_probs, strict=False)
    ):
        teacher_log_prob = teacher_log_prob.to(device=device)
        if teacher_log_prob.shape[0] != student_log_prob.shape[0]:
            raise ValueError(
                f"[OPD] teacher/student log_prob length mismatch at sample {i}: "
                f"teacher={teacher_log_prob.shape[0]}, student={student_log_prob.shape[0]}. "
                "Check response slicing and context-parallel logprob layout."
            )
        teacher_valid_mask = (teacher_log_prob != TEACHER_LOGP_FAILED_SENTINEL).float()
        reverse_kl = (student_log_prob - teacher_log_prob) * teacher_valid_mask
        advantages[i] = adv - coef * reverse_kl
        reverse_kls.append(reverse_kl)
        valid_masks.append(teacher_valid_mask)

    # Store reverse KL for logging
    rollout_data["opd_reverse_kl"] = reverse_kls
    rollout_data["opd_teacher_valid_mask"] = valid_masks


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    and "reinforce_plus_plus_baseline". When `args.normalize_advantages` is
    True, advantages are whitened across the data-parallel group using masked
    statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    If ``args.custom_advantage_function_path`` is set, it is called after KL computation
    and must populate ``rollout_data["advantages"]`` and
    ``rollout_data["returns"]``.

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    rollout_log_probs: list[torch.Tensor] | None = rollout_data.get("rollout_log_probs")
    log_probs: list[torch.Tensor] | None = (
        rollout_log_probs if args.use_rollout_logprobs else rollout_data.get("log_probs")
    )
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # return when not the last pp stage.
    if not mpu.is_pipeline_last_stage():
        return

    direct_cross_tokenizer_distillation = (
        getattr(args, "use_opd", False)
        and getattr(args, "use_cross_tokenizer", False)
        and args.advantage_estimator == "on_policy_distillation"
    )
    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs or rollout_log_probs or values
        if not xs and direct_cross_tokenizer_distillation:
            # Direct cross-tokenizer objectives add their differentiable loss
            # in policy_loss_function.  They only need shape-compatible zero
            # task returns here, so a separate old-policy forward is wasteful.
            xs = loss_masks
        if not xs:
            raise ValueError(
                "Advantage computation requires log_probs, rollout_log_probs, values, "
                "or direct cross-tokenizer response masks on the last pipeline stage."
            )
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]
    rollout_data["kl"] = kl

    if args.custom_advantage_function_path is not None:
        custom_adv_fn = load_function(args.custom_advantage_function_path)
        custom_adv_fn(args, rollout_data)
        advantages, returns = rollout_data["advantages"], rollout_data["returns"]

    elif args.advantage_estimator in ["grpo", "gspo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "ppo":
        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = mpu.get_context_parallel_rank()
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, args.lambd
        )

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    elif args.advantage_estimator == "on_policy_distillation":
        if not getattr(args, "use_opd", False):
            raise ValueError("advantage_estimator='on_policy_distillation' requires --use-opd.")
        if not log_probs and not getattr(args, "use_cross_tokenizer", False):
            raise ValueError(
                "advantage_estimator='on_policy_distillation' requires rollout/student log_probs. "
                "Use --use-rollout-logprobs for the SGLang OPD path."
            )
        opd_loss_type = getattr(args, "opd_loss_type", "sampled")
        if getattr(args, "use_cross_tokenizer", False):
            required = ("teacher_cross_payload", "teacher_cross_skipped_mask")
            missing = [key for key in required if not rollout_data.get(key)]
            if missing:
                raise ValueError(f"Cross-tokenizer OPD requires Teacher target fields: {missing}.")
        elif opd_loss_type == "sampled":
            if not rollout_data.get("teacher_log_probs"):
                raise ValueError("sampled OPD requires teacher_log_probs.")
        else:
            required = ("teacher_opd_ids", "teacher_opd_log_probs", "teacher_opd_mask")
            missing = [key for key in required if not rollout_data.get(key)]
            if missing:
                raise ValueError(f"{opd_loss_type} requires teacher Top-K payload fields: {missing}.")

        # Pure OPD uses zero task returns. Sampled OPD converts them to token
        # advantages below; Top-K modes add a direct differentiable loss.
        return_templates = log_probs or loss_masks
        returns = [
            torch.zeros_like(template, dtype=torch.float32, device=template.device)
            for template in return_templates
        ]
        advantages = [ret.clone() for ret in returns]

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    # Apply on-policy distillation to advantages. For on_policy_distillation,
    # this is the primary training signal; for other estimators it remains additive.
    if args.use_opd:
        apply_opd_kl_to_advantages(
            args=args,
            rollout_data=rollout_data,
            advantages=advantages,
            student_log_probs=log_probs,
        )
        if args.advantage_estimator == "on_policy_distillation":
            returns = advantages

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            advantage_layout = _infer_cp_response_layout(
                advantages,
                total_lengths,
                response_lengths,
                args.qkv_format,
                max_seq_lens,
                getattr(args, "allgather_cp", False),
            )
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len
                )

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if advantage_layout != "single_shard" and res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = mpu.get_data_parallel_group()

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    compute_entropy = abs(float(getattr(args, "entropy_coef", 0.0) or 0.0)) > 0.0
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=compute_entropy,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    response_layout = _infer_cp_response_layout(
        log_probs, total_lengths, response_lengths, args.qkv_format, max_seq_lens, args.allgather_cp
    )
    if not args.use_rollout_logprobs and not old_log_probs:
        old_log_probs = [log_prob.detach() for log_prob in log_probs]
    old_log_probs = _align_cp_response_values(
        old_log_probs,
        total_lengths,
        response_lengths,
        args.qkv_format,
        max_seq_lens,
        args.allgather_cp,
        response_layout,
        "old_log_probs",
    )
    advantages_list = _align_cp_response_values(
        batch["advantages"],
        total_lengths,
        response_lengths,
        args.qkv_format,
        max_seq_lens,
        args.allgather_cp,
        response_layout,
        "advantages",
    )
    teacher_log_probs_for_loss = _align_cp_response_values(
        batch.get("teacher_log_probs"),
        total_lengths,
        response_lengths,
        args.qkv_format,
        max_seq_lens,
        args.allgather_cp,
        response_layout,
        "teacher_log_probs",
    )
    teacher_opd_ids_for_loss = _align_cp_response_values(
        batch.get("teacher_opd_ids"),
        total_lengths,
        response_lengths,
        args.qkv_format,
        max_seq_lens,
        args.allgather_cp,
        response_layout,
        "teacher_opd_ids",
    )
    teacher_opd_log_probs_for_loss = _align_cp_response_values(
        batch.get("teacher_opd_log_probs"),
        total_lengths,
        response_lengths,
        args.qkv_format,
        max_seq_lens,
        args.allgather_cp,
        response_layout,
        "teacher_opd_log_probs",
    )
    teacher_opd_masks_for_loss = _align_cp_response_values(
        batch.get("teacher_opd_mask"),
        total_lengths,
        response_lengths,
        args.qkv_format,
        max_seq_lens,
        args.allgather_cp,
        response_layout,
        "teacher_opd_mask",
    )
    # Cross-tokenizer payloads deliberately remain in full Student-response
    # coordinates. BPM filters them to the rows owned by each CP rank only at
    # scoring time; this keeps the transport/plugin contract independent of
    # Megatron's zig-zag sequence layout.
    teacher_cross_payload_for_loss = batch.get("teacher_cross_payload")
    teacher_cross_skipped_masks_for_loss = batch.get("teacher_cross_skipped_mask")
    rollout_log_probs_for_loss = _align_cp_response_values(
        batch.get("rollout_log_probs"),
        total_lengths,
        response_lengths,
        args.qkv_format,
        max_seq_lens,
        args.allgather_cp,
        response_layout,
        "rollout_log_probs",
    )
    advantages = torch.cat(advantages_list, dim=0)
    train_log_probs_for_tis = batch.get("log_probs")
    if not train_log_probs_for_tis:
        train_log_probs_for_tis = [log_prob.detach() for log_prob in log_probs]
    train_log_probs_for_tis = _align_cp_response_values(
        train_log_probs_for_tis,
        total_lengths,
        response_lengths,
        args.qkv_format,
        max_seq_lens,
        args.allgather_cp,
        response_layout,
        "train_log_probs_for_tis",
    )

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=advantages_list,
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    opd_debug_metrics: dict[str, torch.Tensor] = {}
    if (
        getattr(args, "use_opd", False)
        and not getattr(args, "use_cross_tokenizer", False)
        and getattr(args, "opd_loss_type", "sampled") == "sampled"
    ):
        if not teacher_log_probs_for_loss:
            raise ValueError("OPD requires teacher_log_probs in batch, but it is missing.")
        teacher_log_probs_cat = torch.cat(teacher_log_probs_for_loss, dim=0).to(device=log_probs.device)
        if teacher_log_probs_cat.numel() != log_probs.numel():
            raise ValueError(
                "[OPD] teacher/student log_prob length mismatch in policy loss: "
                f"teacher={teacher_log_probs_cat.numel()}, student={log_probs.numel()}. "
                "Check response slicing and context-parallel logprob layout."
            )

        teacher_valid_mask = (teacher_log_probs_cat != TEACHER_LOGP_FAILED_SENTINEL).float()
        teacher_valid_ratio = sum_of_sample_mean(teacher_valid_mask)
        teacher_valid_ratio_for_div = teacher_valid_ratio.clamp_min(1e-12)
        report_normalizer = (
            sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])
            if args.calculate_per_token_loss
            else len(batch["response_lengths"])
        )

        def report_average_metric(value: torch.Tensor) -> torch.Tensor:
            return value * report_normalizer / mpu.get_context_parallel_world_size()

        rollout_log_probs_cat = (
            torch.cat(rollout_log_probs_for_loss, dim=0).to(device=log_probs.device)
            if rollout_log_probs_for_loss
            else None
        )
        teacher_minus_train = teacher_log_probs_cat - log_probs
        valid_teacher_minus_train = teacher_minus_train * teacher_valid_mask
        opd_debug_metrics["opd/teacher_valid_ratio"] = teacher_valid_ratio
        opd_debug_metrics["opd/train_logprob/mean"] = sum_of_sample_mean(log_probs)
        opd_debug_metrics["opd/teacher_logprob/mean"] = sum_of_sample_mean(teacher_log_probs_cat * teacher_valid_mask)
        opd_debug_metrics["opd/teacher_logprob/valid_mean"] = report_average_metric(
            opd_debug_metrics["opd/teacher_logprob/mean"] / teacher_valid_ratio_for_div
        )
        opd_debug_metrics["opd/teacher_minus_train_logprob/mean"] = sum_of_sample_mean(valid_teacher_minus_train)
        opd_debug_metrics["opd/teacher_minus_train_logprob/abs_mean"] = sum_of_sample_mean(
            teacher_minus_train.abs() * teacher_valid_mask
        )
        opd_debug_metrics["opd/teacher_minus_train_logprob/valid_mean"] = report_average_metric(
            opd_debug_metrics["opd/teacher_minus_train_logprob/mean"] / teacher_valid_ratio_for_div
        )
        opd_debug_metrics["opd/teacher_minus_train_logprob/valid_abs_mean"] = report_average_metric(
            opd_debug_metrics["opd/teacher_minus_train_logprob/abs_mean"] / teacher_valid_ratio_for_div
        )
        opd_debug_metrics["opd/advantage/mean"] = sum_of_sample_mean(advantages)
        opd_debug_metrics["opd/advantage/abs_mean"] = sum_of_sample_mean(advantages.abs())
        opd_debug_metrics["opd/advantage/positive_ratio"] = sum_of_sample_mean((advantages > 0).float())

        if rollout_log_probs_cat is not None and rollout_log_probs_cat.numel() == log_probs.numel():
            train_minus_rollout = log_probs - rollout_log_probs_cat
            teacher_minus_rollout = teacher_log_probs_cat - rollout_log_probs_cat
            opd_debug_metrics["opd/rollout_logprob/mean"] = sum_of_sample_mean(rollout_log_probs_cat)
            opd_debug_metrics["opd/train_minus_rollout_logprob/mean"] = sum_of_sample_mean(train_minus_rollout)
            opd_debug_metrics["opd/train_minus_rollout_logprob/abs_mean"] = sum_of_sample_mean(
                train_minus_rollout.abs()
            )
            opd_debug_metrics["opd/teacher_minus_rollout_logprob/mean"] = sum_of_sample_mean(
                teacher_minus_rollout * teacher_valid_mask
            )
            opd_debug_metrics["opd/teacher_minus_rollout_logprob/abs_mean"] = sum_of_sample_mean(
                teacher_minus_rollout.abs() * teacher_valid_mask
            )
            opd_debug_metrics["opd/teacher_minus_rollout_logprob/valid_mean"] = report_average_metric(
                opd_debug_metrics["opd/teacher_minus_rollout_logprob/mean"] / teacher_valid_ratio_for_div
            )
            opd_debug_metrics["opd/teacher_minus_rollout_logprob/valid_abs_mean"] = report_average_metric(
                opd_debug_metrics["opd/teacher_minus_rollout_logprob/abs_mean"] / teacher_valid_ratio_for_div
            )

        num_failed_tokens = (teacher_valid_mask == 0).sum().item()
        if num_failed_tokens > 0:
            logger.warning(
                "[policy_loss_function] %s/%s teacher tokens are sentinel (-100); "
                "OPD penalty for those tokens is disabled.",
                num_failed_tokens,
                teacher_valid_mask.numel(),
            )

        if getattr(args, "use_teacher_student_logprob_clip", False):
            clip_range = args.teacher_student_logprob_clip_range
            logprob_k1 = compute_approx_kl(log_probs, teacher_log_probs_cat, "k1")
            opd_clip_mask = (logprob_k1.abs() < clip_range).float() * teacher_valid_mask
            opd_debug_metrics["opd/teacher_student_clip_keep_ratio"] = sum_of_sample_mean(opd_clip_mask)
            opd_debug_metrics["opd/teacher_student_clip_drop_ratio"] = sum_of_sample_mean(1.0 - opd_clip_mask)
            opd_debug_metrics["opd/teacher_student_clip_range"] = sum_of_sample_mean(
                torch.full_like(log_probs, float(clip_range))
            )
            pg_loss = pg_loss * opd_clip_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": train_log_probs_for_tis,
            "rollout_log_probs": rollout_log_probs_for_loss,
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with
        # modified_response_masks for numerator correction (rejected tokens
        # zeroed in pg_loss). Denominators stay the precomputed per-rollout
        # totals from ``rollout_mask_sums`` (based on original loss_masks) —
        # same normalizer as the outer reducer, so pg_loss and the rest of the
        # reported metrics live in the same per-rollout-mean space.
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            batch["rollout_mask_sums"],
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # Entropy over full vocab logits is memory-heavy. Skip it when the
    # coefficient is zero so long VL sequences do not clone logits needlessly.
    if compute_entropy:
        entropy = log_probs_and_entropy["entropy"]
        entropy = torch.cat(entropy, dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
    else:
        entropy_loss = logits.new_zeros(())

    loss = pg_loss - args.entropy_coef * entropy_loss

    if getattr(args, "use_opd", False) and getattr(args, "use_cross_tokenizer", False):
        if not teacher_cross_payload_for_loss or not teacher_cross_skipped_masks_for_loss:
            raise ValueError("Cross-tokenizer OPD requires a complete Teacher target payload in the training batch.")
        cross_tokenizer_result = compute_cross_tokenizer_opd_token_losses(
            logits,
            args=args,
            batch=batch,
            teacher_payloads=teacher_cross_payload_for_loss,
            skipped_masks=teacher_cross_skipped_masks_for_loss,
            return_diagnostics=True,
        )
        cross_tokenizer_losses, cross_tokenizer_diagnostics = cross_tokenizer_result
        opd_distribution_loss = sum_of_sample_mean(torch.cat(cross_tokenizer_losses, dim=0))
        loss = loss + args.opd_kl_coef * opd_distribution_loss
        opd_debug_metrics["opd/distribution_loss"] = opd_distribution_loss
        # Training metrics are accumulated as numerators and divided by the
        # step's token/sample denominator in ``reduce_train_step_metrics``.
        # A naked scalar K would therefore be divided again (Top-128 appeared
        # as 1/16 in a 32K-token step). Represent constants on the response
        # layout and route them through the same reducer as other metrics.
        opd_debug_metrics["opd/cross_tokenizer_top_k"] = sum_of_sample_mean(
            torch.full_like(log_probs, float(args.cross_tokenizer_top_k))
        )
        metric_skipped_masks = _align_cp_response_values(
            teacher_cross_skipped_masks_for_loss,
            total_lengths,
            response_lengths,
            args.qkv_format,
            max_seq_lens,
            args.allgather_cp,
            response_layout,
            "teacher_cross_skipped_mask",
        )
        skipped_mask = torch.cat(metric_skipped_masks, dim=0).to(
            device=logits.device, dtype=torch.float32
        )
        if cross_tokenizer_diagnostics:
            diagnostic_values = {
                name: torch.cat(values, dim=0)
                for name, values in cross_tokenizer_diagnostics.items()
            }
            # These are numerator-like token sums.  The common train-step
            # reducer divides them by the full response-token denominator, so
            # root+child matches ``opd/distribution_loss`` in the logged space.
            opd_debug_metrics["opd/escd_root_loss_sum"] = sum_of_sample_mean(
                diagnostic_values["root_loss"]
            )
            opd_debug_metrics["opd/escd_child_loss_sum"] = sum_of_sample_mean(
                diagnostic_values["child_loss"]
            )
            opd_debug_metrics["opd/escd_compiled_child_row_ratio"] = sum_of_sample_mean(
                diagnostic_values["compiled_child_row"]
            )
            supported_event_mass = sum_of_sample_mean(
                diagnostic_values["supported_event_mass"]
            )
            nonshared_event_mass = sum_of_sample_mean(
                diagnostic_values["nonshared_event_mass"]
            )
            opd_debug_metrics["opd/escd_supported_event_mass_ratio"] = (
                supported_event_mass
                / nonshared_event_mass.clamp_min(1e-12)
                * sum_of_sample_mean(torch.ones_like(skipped_mask))
            )
            token_count = sum_of_sample_mean(torch.ones_like(skipped_mask))
            active_token_count = (token_count - sum_of_sample_mean(skipped_mask)).clamp_min(1.0)
            # Rescale the numerator before the common full-token reducer so
            # this metric reports the mean over non-skipped response tokens.
            opd_debug_metrics["opd/escd_active_token_renormalized_loss"] = (
                opd_distribution_loss * token_count / active_token_count
            )
        opd_debug_metrics["opd/cross_tokenizer_skipped_token_ratio"] = sum_of_sample_mean(skipped_mask)
        metric_algorithm = _build_cross_tokenizer_scoring_algorithm(args)
        loss_anchor_mask = _cross_tokenizer_loss_anchor_mask(
            teacher_cross_payload_for_loss,
            teacher_cross_skipped_masks_for_loss,
            position_field=metric_algorithm.capabilities.payload_position_field,
            device=logits.device,
        )
        diagnostic_masks: dict[str, torch.Tensor] = {}
        if batch.get("teacher_cross_alignments") is not None:
            diagnostic_masks = metric_algorithm.build_training_diagnostic_masks(
                [
                    AlignmentResult.from_dict(value)
                    for value in batch["teacher_cross_alignments"]
                ],
                teacher_cross_skipped_masks_for_loss,
                device=logits.device,
            )
            plugin_anchor = diagnostic_masks.pop(
                "cross_tokenizer_loss_anchor", None
            )
            if loss_anchor_mask is None:
                loss_anchor_mask = plugin_anchor
        if loss_anchor_mask is not None:
            anchor_masks = list(loss_anchor_mask.split(response_lengths, dim=0))
            anchor_masks = _align_cp_response_values(
                anchor_masks,
                total_lengths,
                response_lengths,
                args.qkv_format,
                max_seq_lens,
                args.allgather_cp,
                response_layout,
                "cross_tokenizer_loss_anchor_mask",
            )
            opd_debug_metrics["opd/cross_tokenizer_loss_anchor_ratio"] = sum_of_sample_mean(
                torch.cat(anchor_masks, dim=0)
            )
        for diagnostic_name, diagnostic_mask in diagnostic_masks.items():
            diagnostic_masks = list(diagnostic_mask.split(response_lengths, dim=0))
            diagnostic_masks = _align_cp_response_values(
                diagnostic_masks,
                total_lengths,
                response_lengths,
                args.qkv_format,
                max_seq_lens,
                args.allgather_cp,
                response_layout,
                f"{diagnostic_name}_mask",
            )
            opd_debug_metrics[f"opd/{diagnostic_name}_ratio"] = sum_of_sample_mean(
                torch.cat(diagnostic_masks, dim=0)
            )
    elif getattr(args, "use_opd", False) and getattr(args, "opd_loss_type", "sampled") != "sampled":
        if not teacher_opd_ids_for_loss or not teacher_opd_log_probs_for_loss or not teacher_opd_masks_for_loss:
            raise ValueError(f"{args.opd_loss_type} requires teacher Top-K payload in the training batch.")
        opd_token_losses = compute_topk_opd_token_losses(
            logits,
            args=args,
            batch=batch,
            teacher_ids=teacher_opd_ids_for_loss,
            teacher_log_probs=teacher_opd_log_probs_for_loss,
            teacher_masks=teacher_opd_masks_for_loss,
        )
        opd_distribution_loss = sum_of_sample_mean(torch.cat(opd_token_losses, dim=0))
        loss = loss + args.opd_kl_coef * opd_distribution_loss
        opd_debug_metrics["opd/distribution_loss"] = opd_distribution_loss
        opd_debug_metrics["opd/top_k"] = sum_of_sample_mean(
            torch.full_like(log_probs, float(args.opd_top_k))
        )

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if rollout_log_probs_for_loss:
        rollout_log_probs = torch.cat(rollout_log_probs_for_loss, dim=0).to(device=old_log_probs.device)
        if rollout_log_probs.numel() == old_log_probs.numel():
            train_rollout_logprob_abs_diff = sum_of_sample_mean((old_log_probs - rollout_log_probs).abs())
        else:
            logger.warning(
                "[policy_loss_function] skip train_rollout_logprob_abs_diff because aligned "
                "old/rollout logprob lengths differ: old=%s, rollout=%s",
                old_log_probs.numel(),
                rollout_log_probs.numel(),
            )

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()

    for metric_key, metric_value in opd_debug_metrics.items():
        reported_loss[metric_key] = metric_value.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # Add OPD metrics if available
    if "opd_reverse_kl" in batch:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    _, values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    step_global_batch_size: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        step_global_batch_size: Sample count for the current training step
            (total across DP). Replaces the legacy ``args.global_batch_size``
            fallback so the train side stops depending on "every DP rank holds
            the same N samples".
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        batch["rollout_mask_sums"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )

    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean, use_reentrant=False)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # With allgather-CP, some CP ranks may have no loss-contributing tokens (e.g., all
    # padding). Without this, gradient doesn't flow through their attention path, so
    # the CP gather's backward (reduce-scatter) is not called, deadlocking other CP
    # ranks that call it. Adding this zero loss forces autograd to traverse the full
    # graph on every rank without changing gradient values.
    if args.allgather_cp and mpu.get_context_parallel_world_size() > 1:
        loss = loss + 0 * logits.sum()

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    if not args.calculate_per_token_loss:
        loss = (
            loss
            * num_microbatches
            / step_global_batch_size
            * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    else:
        loss = loss * mpu.get_context_parallel_world_size()

    return (
        loss,
        (num_tokens if args.calculate_per_token_loss else torch.tensor(1, device=logits.device)),
        {
            "keys": list(log.keys()),
            # values[0] is the consumer's reporting denominator after
            # all-reduce. For per-token-loss it must equal step total tokens
            # (only known by summing per-mb num_tokens across mbs / DP). For
            # per-rollout-mean it is a constant — ``step_global_batch_size`` —
            # so we leave a 0 placeholder here and let ``train_one_step``
            # substitute the constant directly, instead of routing it through
            # per-mb fractions.
            "values": torch.tensor(
                [
                    num_tokens if args.calculate_per_token_loss else 0,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
