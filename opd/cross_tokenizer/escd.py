"""Event-Set Completion Distillation as a Slime cross-tokenizer plugin."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Sequence

import torch

from .base import (
    AlignmentResult,
    CrossTokenizerCapabilities,
    FusedScoringContext,
    StudentScoringContext,
    TeacherScoringContext,
    TensorPayload,
)
from .bpm import BPMCrossTokenizer
from .bpm_dense import bpm_projected_divergence_from_log_probs
from .escd_core import (
    ESCDChildEvent,
    StudentBytePrefixIndex,
    compile_child_events,
    project_root_topk,
)
from .registry import register_cross_tokenizer


_ESCD_STUDENT_INDEX_CACHE: dict[str, StudentBytePrefixIndex] = {}


def _bucket_width(width: int) -> int:
    if width <= 0:
        return 0
    return 1 << (int(width) - 1).bit_length()


@register_cross_tokenizer("escd")
class ESCDCrossTokenizer(BPMCrossTokenizer):
    """Zero-extra-forward ESCD with a matched BPM root and depth-one event sets.

    Inheriting ``BPMCrossTokenizer`` is an engineering reuse of its audited
    lossless byte artifact and response alignment.  The supervision objective
    is not BPM: only aligned root rows use the shared root projection, while a
    normally visited child row receives ``-m log sum_{u in C(r)} p(u)`` over the
    complete lossless success set.  No inverse behavior-probability weight is
    used and the child coefficient is fixed to one.
    """

    capabilities = CrossTokenizerCapabilities(
        fidelity="proposed-method",
        reference="Event-Set Completion Distillation (ESCD), natural visitation, depth one",
        candidate_top_k_mode="positive",
        default_candidate_top_k=128,
        teacher_backends=("sglang",),
        required_distribution_methods=("topk", "selected_log_probs", "projected_log_probs"),
        public_temperature_mode="one",
        supports_context_parallel=True,
        context_parallel_backends=("sglang",),
        teacher_transports=(("sglang", "causal_hidden"),),
        requires_loss_token_normalization=True,
        requires_full_response_loss_mask=True,
        # ESCD gathers dense Top-K rows across CP ranks.  Recomputing the loss
        # repeats those collectives during backward and can deadlock with CP
        # attention collectives on long sequences.  Keep the fix local to this
        # plugin; do not alter Slime's global recompute behavior.
        requires_loss_recompute_disabled=True,
        requires_chat_template=True,
        requires_thinking=True,
        external_reward_path=(
            "opd.cross_tokenizer.sglang_teacher.external_causal_hidden_reward"
        ),
        external_post_process_path=(
            "opd.cross_tokenizer.sglang_teacher.install_external_causal_hidden_payload"
        ),
    )

    _TRAINING_DIAGNOSTIC_NAMES = (
        "root_loss",
        "child_loss",
        "compiled_child_row",
        "supported_event_mass",
        "nonshared_event_mass",
    )

    def training_diagnostic_names(self) -> tuple[str, ...]:
        """Names of the per-response tensors emitted by fused ESCD scoring."""

        return self._TRAINING_DIAGNOSTIC_NAMES

    def _set_training_diagnostics(
        self,
        *,
        root_loss: torch.Tensor,
        child_loss: torch.Tensor,
        compiled_child_row: torch.Tensor,
        supported_event_mass: torch.Tensor,
        nonshared_event_mass: torch.Tensor,
    ) -> None:
        values = {
            "root_loss": root_loss,
            "child_loss": child_loss,
            "compiled_child_row": compiled_child_row,
            "supported_event_mass": supported_event_mass,
            "nonshared_event_mass": nonshared_event_mass,
        }
        expected = root_loss.numel()
        for name, value in values.items():
            if value.ndim != 1 or value.numel() != expected:
                raise ValueError(
                    f"ESCD training diagnostic {name!r} must have shape [{expected}], "
                    f"got {tuple(value.shape)}"
                )
        self._training_diagnostics = {
            name: value.detach().to(dtype=torch.float32)
            for name, value in values.items()
        }

    def pop_training_diagnostics(self) -> dict[str, torch.Tensor]:
        values = getattr(self, "_training_diagnostics", None)
        if values is None:
            return {}
        self._training_diagnostics = None
        return values

    def _escd_settings(self) -> tuple[bool, int, int, float]:
        mask_whitespace = bool(self.algorithm_kwargs.get("mask_whitespace", True))
        teacher_row_chunk_size = int(
            self.algorithm_kwargs.get("teacher_row_chunk_size", 8)
        )
        student_row_chunk_size = int(
            self.algorithm_kwargs.get("student_row_chunk_size", 32)
        )
        min_event_mass = float(self.algorithm_kwargs.get("min_event_mass", 0.0))
        child_loss_weight = float(self.algorithm_kwargs.get("child_loss_weight", 1.0))
        if teacher_row_chunk_size <= 0 or student_row_chunk_size <= 0:
            raise ValueError("ESCD row chunk sizes must be positive")
        if not math.isfinite(min_event_mass) or min_event_mass < 0.0:
            raise ValueError("ESCD min_event_mass must be finite and non-negative")
        if not math.isclose(child_loss_weight, 1.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                "The main ESCD method fixes child_loss_weight=1; use a separately named ablation plugin to tune it"
            )
        return mask_whitespace, teacher_row_chunk_size, student_row_chunk_size, min_event_mass

    def _escd_root_beta(self) -> float:
        """Return the generalized-D_beta coefficient for ESCD root rows.

        ``beta=0`` is the historical forward-KL root objective.  ``beta=0.5``
        is the symmetric Jensen-Shannon setting.  Child completion remains the
        ESCD event-completion objective rather than being relabeled as a
        distributional JSD term.
        """

        beta = float(self.algorithm_kwargs.get("beta", 0.0))
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError(f"ESCD root beta must be finite and in [0, 1], got {beta}")
        return beta

    def _student_index(self) -> StudentBytePrefixIndex:
        cached = getattr(self, "_escd_student_index", None)
        if isinstance(cached, StudentBytePrefixIndex):
            return cached
        artifact = self._artifact()
        # Slime reconstructs the lightweight plugin object for every loss call.
        # Reuse the immutable 248K-vocabulary prefix index within each worker so
        # the ESCD objective does not repeatedly pay the tokenizer scan cost.
        index = _ESCD_STUDENT_INDEX_CACHE.get(artifact.sha256)
        if index is None:
            index = StudentBytePrefixIndex(
                artifact.student_bytes,
                excluded_token_ids=artifact.student_stop_ids,
            )
            _ESCD_STUDENT_INDEX_CACHE[artifact.sha256] = index
        self._escd_student_index = index
        return index

    def build_teacher_payload(self, context: TeacherScoringContext) -> TensorPayload:
        del context
        raise NotImplementedError(
            "ESCD is intentionally implemented through the causal-hidden fused Slime transport"
        )

    def compute_student_token_losses(
        self,
        context: StudentScoringContext,
        teacher_payload: TensorPayload,
    ) -> torch.Tensor:
        del context, teacher_payload
        raise NotImplementedError(
            "ESCD is intentionally implemented through compute_fused_token_losses"
        )

    def build_training_diagnostic_masks(
        self,
        alignments: list[AlignmentResult],
        skipped_masks: list[torch.Tensor],
        *,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Expose structural root/child opportunities without changing Slime."""

        mask_whitespace, _, _, _ = self._escd_settings()
        names = (
            "cross_tokenizer_loss_anchor",
            "escd_root_candidate",
            "escd_child_state_candidate",
        )
        values: dict[str, list[torch.Tensor]] = {name: [] for name in names}
        for alignment, skipped in zip(alignments, skipped_masks, strict=True):
            skipped = skipped.to(device=device, dtype=torch.bool).flatten()
            masks = {
                name: torch.zeros(alignment.student_length, device=device, dtype=torch.float32)
                for name in names
            }
            if not bool(skipped.all()):
                by_position = {
                    int(row["student_position"]): row
                    for row in alignment.algorithm_metadata.get("bpm_rows", ())
                }
                for position, row in by_position.items():
                    root = str(row.get("kind")) in {"aligned", "stop"}
                    root = root and not (
                        mask_whitespace and bool(row.get("whitespace", False))
                    )
                    if root:
                        masks["escd_root_candidate"][position] = 1.0
                        masks["cross_tokenizer_loss_anchor"][position] = 1.0
                    previous = by_position.get(position - 1)
                    child = (
                        previous is not None
                        and str(previous.get("kind")) == "aligned"
                        and not (
                            mask_whitespace and bool(previous.get("whitespace", False))
                        )
                    )
                    if child:
                        masks["escd_child_state_candidate"][position] = 1.0
                        masks["cross_tokenizer_loss_anchor"][position] = 1.0
            for name in names:
                values[name].append(masks[name])
        return {name: torch.cat(parts, dim=0) for name, parts in values.items()}

    @staticmethod
    def _anchor_zero(context: FusedScoringContext, output: torch.Tensor) -> torch.Tensor:
        if context.student.distribution.length == 0:
            return output
        position = torch.zeros((1,), device=output.device, dtype=torch.long)
        token = context.student.response_token_ids[:1].reshape(1, 1)
        anchor = context.student.distribution.rows(position).selected_log_probs(
            token,
            vocab_size=None,
            temperature=1.0,
        ).sum() * 0.0
        return output + anchor

    def _score_root_targets(
        self,
        context: FusedScoringContext,
        records: Sequence[tuple[int, tuple[int, ...], tuple[float, ...], float]],
        *,
        student_row_chunk_size: int,
    ) -> torch.Tensor:
        artifact = self._artifact()
        device = context.student.response_token_ids.device
        output = torch.zeros(
            context.student.distribution.length, dtype=torch.float32, device=device
        )
        buckets: dict[int, list[tuple[int, tuple[int, ...], tuple[float, ...], float]]] = defaultdict(list)
        for record in records:
            if record[1]:
                buckets[_bucket_width(len(record[1]))].append(record)
        for width in sorted(buckets):
            bucket = buckets[width]
            for start in range(0, len(bucket), student_row_chunk_size):
                chunk = bucket[start : start + student_row_chunk_size]
                positions = torch.tensor(
                    [record[0] for record in chunk], device=device, dtype=torch.long
                )
                ids = torch.zeros((len(chunk), width), device=device, dtype=torch.long)
                mask = torch.zeros((len(chunk), width), device=device, dtype=torch.bool)
                target = torch.zeros(
                    (len(chunk), width + 1), device=device, dtype=torch.float32
                )
                for row_index, (_, support, probabilities, complement) in enumerate(chunk):
                    count = len(support)
                    support_set = set(support)
                    padding_id = next(
                        token_id
                        for token_id in range(len(artifact.student_bytes))
                        if token_id not in support_set
                    )
                    ids[row_index].fill_(padding_id)
                    ids[row_index, :count] = torch.tensor(
                        support, device=device, dtype=torch.long
                    )
                    mask[row_index, :count] = True
                    target[row_index, :count] = torch.tensor(
                        probabilities, device=device, dtype=torch.float32
                    )
                    target[row_index, -1] = float(complement)
                student_log_probs = context.student.distribution.rows(
                    positions
                ).projected_log_probs(
                    ids,
                    mask,
                    vocab_size=len(artifact.student_bytes),
                    temperature=1.0,
                ).float()
                losses = bpm_projected_divergence_from_log_probs(
                    target,
                    student_log_probs,
                    beta=self._escd_root_beta(),
                    skew=0.1,
                )
                output.index_add_(0, positions, losses)
        return output

    def _score_child_targets(
        self,
        context: FusedScoringContext,
        records: Sequence[tuple[int, tuple[ESCDChildEvent, ...]]],
        *,
        student_row_chunk_size: int,
    ) -> torch.Tensor:
        artifact = self._artifact()
        device = context.student.response_token_ids.device
        output = torch.zeros(
            context.student.distribution.length, dtype=torch.float32, device=device
        )
        prepared: list[
            tuple[int, tuple[int, ...], tuple[tuple[float, tuple[int, ...]], ...]]
        ] = []
        for local_position, events in records:
            union = tuple(
                sorted(
                    {
                        token_id
                        for event in events
                        for token_id in event.success_token_ids
                    }
                )
            )
            if not union:
                continue
            offsets = {token_id: index for index, token_id in enumerate(union)}
            event_specs = tuple(
                (
                    float(event.mass),
                    tuple(offsets[token_id] for token_id in event.success_token_ids),
                )
                for event in events
            )
            prepared.append((local_position, union, event_specs))

        buckets: dict[
            int,
            list[tuple[int, tuple[int, ...], tuple[tuple[float, tuple[int, ...]], ...]]],
        ] = defaultdict(list)
        for record in prepared:
            buckets[_bucket_width(len(record[1]))].append(record)
        for width in sorted(buckets):
            bucket = buckets[width]
            for start in range(0, len(bucket), student_row_chunk_size):
                chunk = bucket[start : start + student_row_chunk_size]
                positions = torch.tensor(
                    [record[0] for record in chunk], device=device, dtype=torch.long
                )
                ids = torch.zeros((len(chunk), width), device=device, dtype=torch.long)
                valid = torch.zeros((len(chunk), width), device=device, dtype=torch.bool)
                for row_index, (_, union, _) in enumerate(chunk):
                    ids[row_index, : len(union)] = torch.tensor(
                        union, device=device, dtype=torch.long
                    )
                    valid[row_index, : len(union)] = True
                log_probs = context.student.distribution.rows(positions).selected_log_probs(
                    ids,
                    vocab_size=len(artifact.student_bytes),
                    temperature=1.0,
                ).float()
                row_losses: list[torch.Tensor] = []
                for row_index, (_, _, event_specs) in enumerate(chunk):
                    row_loss = log_probs[row_index, 0] * 0.0
                    for mass, event_offsets in event_specs:
                        selected = torch.tensor(
                            event_offsets, device=device, dtype=torch.long
                        )
                        event_log_probability = torch.logsumexp(
                            log_probs[row_index].index_select(0, selected), dim=0
                        )
                        row_loss = row_loss - float(mass) * event_log_probability
                    row_losses.append(row_loss)
                losses = torch.stack(row_losses).float()
                if not bool(torch.isfinite(losses).all()):
                    raise FloatingPointError("ESCD produced a non-finite child loss")
                output.index_add_(0, positions, losses)
        return output

    def compute_fused_token_losses(self, context: FusedScoringContext) -> torch.Tensor:
        artifact = self._artifact()
        mask_whitespace, teacher_chunk, student_chunk, min_event_mass = self._escd_settings()
        candidate_top_k = int(context.candidate_top_k)
        if candidate_top_k <= 0:
            raise ValueError("ESCD requires a positive Teacher candidate_top_k")
        if not math.isclose(float(context.student.temperature), 1.0):
            raise ValueError("ESCD requires cross_tokenizer.temperature=1.0")
        if context.teacher_distribution.length != context.alignment.teacher_length:
            raise ValueError("ESCD Teacher distribution rows do not match the alignment")

        device = context.student.response_token_ids.device
        local_positions = context.local_response_positions.to(
            device=device, dtype=torch.long
        ).flatten()
        if local_positions.numel() != context.student.distribution.length:
            raise ValueError("ESCD local response positions do not match Student rows")
        global_to_local = {
            int(global_position): local_position
            for local_position, global_position in enumerate(local_positions.tolist())
        }
        rows_by_position = {
            int(row["student_position"]): dict(row)
            for row in context.alignment.algorithm_metadata.get("bpm_rows", ())
        }
        root_rows = [
            row
            for position, row in sorted(rows_by_position.items())
            if position in global_to_local
            and str(row.get("kind")) in {"aligned", "stop"}
            and not (mask_whitespace and bool(row.get("whitespace", False)))
        ]

        local_topk_ids = torch.zeros(
            (local_positions.numel(), candidate_top_k), device=device, dtype=torch.long
        )
        local_topk_probs = torch.zeros(
            (local_positions.numel(), candidate_top_k), device=device, dtype=torch.float32
        )
        for start in range(0, len(root_rows), teacher_chunk):
            chunk = root_rows[start : start + teacher_chunk]
            teacher_positions = torch.tensor(
                [int(row["teacher_position"]) for row in chunk],
                device=device,
                dtype=torch.long,
            )
            candidates = context.teacher_distribution.rows(teacher_positions).topk(
                candidate_top_k
            )
            if not bool(candidates.mask.all()):
                raise ValueError("ESCD causal-hidden Teacher Top-K contains masked candidates")
            probabilities = candidates.log_probs.detach().float().exp()
            probability_mass = probabilities.sum(dim=-1, keepdim=True)
            if bool((probability_mass > 1.0 + 2e-5).any()):
                raise FloatingPointError("ESCD Teacher Top-K mass exceeds one")
            # Keep the same numerically valid sub-probability measure for the
            # root projection and the child-event compiler.  bf16 log-prob
            # roundoff may otherwise make a Top-K row sum to 1+epsilon.
            probabilities = probabilities / probability_mass.clamp_min(1.0)
            local_rows = torch.tensor(
                [global_to_local[int(row["student_position"])] for row in chunk],
                device=device,
                dtype=torch.long,
            )
            local_topk_ids.index_copy_(0, local_rows, candidates.output_ids.detach())
            local_topk_probs.index_copy_(0, local_rows, probabilities)

        root_local_indices = torch.tensor(
            [global_to_local[int(row["student_position"])] for row in root_rows],
            device=device,
            dtype=torch.long,
        )
        root_ids_cpu = (
            local_topk_ids.index_select(0, root_local_indices).detach().cpu().tolist()
            if root_rows
            else []
        )
        root_probs_cpu = (
            local_topk_probs.index_select(0, root_local_indices).detach().cpu().tolist()
            if root_rows
            else []
        )
        root_targets: list[tuple[int, tuple[int, ...], tuple[float, ...], float]] = []
        for row, token_ids, probabilities in zip(
            root_rows, root_ids_cpu, root_probs_cpu, strict=True
        ):
            support, target_probs, complement = project_root_topk(
                teacher_token_ids=token_ids,
                teacher_probabilities=probabilities,
                artifact=artifact,
                # ``BPMCrossTokenizer.align`` stores the realized Student
                # stop on every content row, but the terminal ``stop`` row
                # itself only has ``student_token_id``.  Falling back to the
                # row token is essential: otherwise Teacher EOS mass is sent
                # to ESCD's complement instead of the actual Student EOS
                # coordinate, making ESCD's root target differ from BPM for
                # a purely bookkeeping reason.
                student_stop_token_id=(
                    int(row["student_stop_token_id"])
                    if "student_stop_token_id" in row
                    else int(row["student_token_id"])
                    if str(row.get("kind")) == "stop"
                    else None
                ),
                terminal_stop=str(row.get("kind")) == "stop",
            )
            root_targets.append(
                (
                    global_to_local[int(row["student_position"])],
                    support,
                    target_probs,
                    complement,
                )
            )
        root_output = self._score_root_targets(
            context, root_targets, student_row_chunk_size=student_chunk
        )

        if context.gather_student_rows is None:
            raise ValueError("ESCD requires the generic Context-Parallel row gather")
        # Root Top-K is computed exactly once on the CP rank owning that root.
        # Only compact IDs/probabilities are exchanged; no vocabulary logits or
        # hidden states are gathered and no second Teacher head evaluation is run.
        global_topk_ids = context.gather_student_rows(local_topk_ids.detach())
        global_topk_probs = context.gather_student_rows(local_topk_probs.detach())

        child_rows: list[tuple[int, int, int]] = []
        for global_position, local_position in sorted(global_to_local.items()):
            previous = rows_by_position.get(global_position - 1)
            if (
                previous is not None
                and str(previous.get("kind")) == "aligned"
                and not (mask_whitespace and bool(previous.get("whitespace", False)))
            ):
                child_rows.append(
                    (
                        local_position,
                        global_position - 1,
                        int(previous["student_token_id"]),
                    )
                )

        if child_rows:
            previous_positions = torch.tensor(
                [record[1] for record in child_rows], device=device, dtype=torch.long
            )
            child_ids_cpu = global_topk_ids.index_select(
                0, previous_positions
            ).detach().cpu().tolist()
            child_probs_cpu = global_topk_probs.index_select(
                0, previous_positions
            ).detach().cpu().tolist()
        else:
            child_ids_cpu = child_probs_cpu = []

        child_targets: list[tuple[int, tuple[ESCDChildEvent, ...]]] = []
        compiled_child_row = torch.zeros(
            context.student.distribution.length, dtype=torch.float32, device=device
        )
        supported_event_mass = torch.zeros_like(compiled_child_row)
        nonshared_event_mass = torch.zeros_like(compiled_child_row)
        student_index = self._student_index()
        for (local_position, _, first_token_id), token_ids, probabilities in zip(
            child_rows, child_ids_cpu, child_probs_cpu, strict=True
        ):
            compiled = compile_child_events(
                teacher_token_ids=token_ids,
                teacher_probabilities=probabilities,
                actual_first_token_id=first_token_id,
                artifact=artifact,
                student_index=student_index,
                min_event_mass=min_event_mass,
            )
            nonshared_event_mass[local_position] = float(compiled.nonshared_mass)
            supported_event_mass[local_position] = float(compiled.supported_mass)
            if compiled.events:
                compiled_child_row[local_position] = 1.0
                child_targets.append((local_position, compiled.events))
        child_output = self._score_child_targets(
            context, child_targets, student_row_chunk_size=student_chunk
        )
        output = root_output + child_output
        self._set_training_diagnostics(
            root_loss=root_output,
            child_loss=child_output,
            compiled_child_row=compiled_child_row,
            supported_event_mass=supported_event_mass,
            nonshared_event_mass=nonshared_event_mass,
        )
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("ESCD produced non-finite token losses")
        return self._anchor_zero(context, output)
