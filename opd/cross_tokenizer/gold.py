"""Paper-aligned GOLD hybrid cross-tokenizer distillation baseline."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from ._paper_utils import (
    aligned_segment_indices,
    export_tokenizer_runtime_state,
    merge_observed_probabilities,
    resolve_vocab_size,
)
from .base import (
    CrossTokenizerCapabilities,
    FusedScoringContext,
    SharedVocabulary,
    StudentScoringContext,
    TeacherScoringContext,
    TensorPayload,
)
from .registry import register_cross_tokenizer
from .simct import SimCTCrossTokenizer


def build_gold_shared_vocabulary(student_tokenizer: Any, teacher_tokenizer: Any) -> SharedVocabulary | None:
    """Match GOLD tokens by exact tokenizer vocabulary string.

    GOLD's public implementation does not apply SimCT's ``Ġ``/``▁``
    canonicalization and does not synthesize a semantic EOS pair.
    """

    student_vocab = {str(token): int(token_id) for token, token_id in student_tokenizer.get_vocab().items()}
    teacher_vocab = {str(token): int(token_id) for token, token_id in teacher_tokenizer.get_vocab().items()}
    tokens = sorted(set(student_vocab).intersection(teacher_vocab))
    if not tokens:
        return None
    return SharedVocabulary(
        tuple(student_vocab[token] for token in tokens),
        tuple(teacher_vocab[token] for token in tokens),
        tuple(tokens),
    )


@register_cross_tokenizer("gold")
class GOLDCrossTokenizer(SimCTCrossTokenizer):
    """GJSD on exact matches plus sorted ULD on unmatched vocabulary."""

    capabilities = CrossTokenizerCapabilities(
        fidelity="official-source",
        reference="huggingface/trl experimental/gold@6fae07f",
        candidate_top_k_mode="zero",
        default_candidate_top_k=0,
        teacher_backends=("megatron", "sglang"),
        required_distribution_methods=("full_probabilities", "selected_log_probs"),
        public_temperature_mode="one",
        # The payload path still requires CP=1 because an aligned group can
        # cross Student CP ranks.  The causal-hidden path gathers only the
        # realized Student continuation log-probabilities and evaluates each
        # dense group-start row on its owning rank, preserving exact GOLD.
        context_parallel_backends=("sglang",),
        teacher_transports=(("sglang", "causal_hidden"),),
        requires_loss_token_normalization=True,
        requires_full_response_loss_mask=True,
        requires_loss_recompute_disabled=True,
        requires_chat_template=True,
        external_reward_path=(
            "opd.cross_tokenizer.sglang_teacher.external_causal_hidden_reward"
        ),
        external_post_process_path=(
            "opd.cross_tokenizer.sglang_teacher.install_external_causal_hidden_payload"
        ),
    )

    def _settings(
        self,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
        bool,
        bool,
        float | None,
        float | None,
    ]:
        beta = float(self.algorithm_kwargs.get("beta", 0.5))
        student_temperature = float(self.algorithm_kwargs.get("student_temperature", 1.0))
        teacher_temperature = float(self.algorithm_kwargs.get("teacher_temperature", 1.0))
        distillation_weight = float(self.algorithm_kwargs.get("distillation_weight", 1.0))
        crossentropy_weight = float(self.algorithm_kwargs.get("crossentropy_weight", 0.0))
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError(f"GOLD beta must be finite and in [0, 1], got {beta}")
        for name, value in {
            "student_temperature": student_temperature,
            "teacher_temperature": teacher_temperature,
        }.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"GOLD {name} must be positive and finite, got {value}")
        for name, value in {
            "distillation_weight": distillation_weight,
            "crossentropy_weight": crossentropy_weight,
        }.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"GOLD {name} must be non-negative and finite, got {value}")
        strategy = str(self.algorithm_kwargs.get("token_merge_strategy", "observed")).strip().lower()
        if strategy != "observed":
            raise ValueError(
                "The integrated GOLD baseline currently implements the published/default 'observed' merge only; "
                "do not silently substitute the optional 'bayesian' ablation"
            )
        matched_weight = self.algorithm_kwargs.get("matched_weight")
        unmatched_weight = self.algorithm_kwargs.get("unmatched_weight")
        if (matched_weight is None) != (unmatched_weight is None):
            raise ValueError("GOLD matched_weight and unmatched_weight must be both set or both omitted")
        if matched_weight is not None:
            matched_weight = float(matched_weight)
            unmatched_weight = float(unmatched_weight)
            if not all(math.isfinite(value) and value >= 0 for value in (matched_weight, unmatched_weight)):
                raise ValueError("GOLD hybrid weights must be non-negative and finite")
        return (
            beta,
            student_temperature,
            teacher_temperature,
            distillation_weight,
            crossentropy_weight,
            bool(self.algorithm_kwargs.get("skip_student_eos", True)),
            bool(self.algorithm_kwargs.get("skip_teacher_eos", True)),
            matched_weight,
            unmatched_weight,
        )

    @staticmethod
    def _require_public_temperature_one(temperature: float) -> None:
        if float(temperature) != 1.0:
            raise ValueError(
                "GOLD temperatures belong in algorithm_kwargs.student_temperature/teacher_temperature; "
                "cross_tokenizer.temperature must remain 1.0"
            )

    def build_shared_vocabulary(self) -> SharedVocabulary | None:
        if self.teacher_tokenizer is None or self.student_tokenizer is None:
            raise RuntimeError("GOLD vocabulary construction requires both tokenizers")
        return build_gold_shared_vocabulary(self.student_tokenizer, self.teacher_tokenizer)

    def configure_model_contract(
        self,
        *,
        student_vocab_size: int,
        teacher_vocab_size: int,
        student_stop_ids: tuple[int, ...],
        teacher_stop_ids: tuple[int, ...],
    ) -> None:
        """Keep dense scoring on model vocabularies, not tokenizer estimates."""

        self.runtime_state = {
            **self.runtime_state,
            "student_vocab_size": int(student_vocab_size),
            "teacher_vocab_size": int(teacher_vocab_size),
            # Keep the complete semantic stop sets in the JSON runtime state.
            # GOLD's EOS-skip option must not interpret a max-length-truncated
            # content row as an EOS row.
            "student_stop_ids": [int(value) for value in student_stop_ids],
            "teacher_stop_ids": [int(value) for value in teacher_stop_ids],
        }

    def _drop_terminal_segment_if_eos(
        self,
        alignment: AlignmentResult,
        teacher_response_token_ids: torch.Tensor,
        student_response_token_ids: torch.Tensor | None = None,
    ) -> bool:
        """Return whether the final alignment group is an actual semantic stop.

        The upstream Trainer receives examples with an EOS in every supervised
        completion and therefore subtracts one row whenever ``skip_eos`` is
        enabled. Slime also trains responses that hit ``max_response_len``;
        those responses legitimately have no EOS. In that case dropping the
        last *aligned group* would remove content (and, for a span, more than
        one token). The guard preserves source behavior for EOS-terminated
        responses while keeping truncation from changing the target.
        """

        _, _, _, _, _, skip_student_eos, skip_teacher_eos, _, _ = self._settings()
        if not (skip_student_eos or skip_teacher_eos) or not alignment.segments:
            return False
        terminal = alignment.segments[-1]
        teacher_ids = teacher_response_token_ids.to(dtype=torch.long).flatten()
        teacher_stop_ids = self.runtime_state.get("teacher_stop_ids", ())
        if not teacher_stop_ids:
            eos = self.runtime_state.get("teacher_eos_token_id")
            teacher_stop_ids = () if eos is None else (eos,)
        teacher_is_stop = (
            skip_teacher_eos
            and teacher_ids.numel() >= alignment.teacher_length
            and terminal.teacher_end == alignment.teacher_length
            and terminal.teacher_end - terminal.teacher_start == 1
            and int(teacher_ids[alignment.teacher_length - 1])
            in {int(value) for value in teacher_stop_ids}
        )

        student_is_stop = False
        if skip_student_eos and student_response_token_ids is not None:
            student_ids = student_response_token_ids.to(dtype=torch.long).flatten()
            student_stop_ids = self.runtime_state.get("student_stop_ids", ())
            if not student_stop_ids:
                eos = self.runtime_state.get("student_eos_token_id")
                student_stop_ids = () if eos is None else (eos,)
            student_is_stop = (
                student_ids.numel() >= alignment.student_length
                and terminal.student_end == alignment.student_length
                and terminal.student_end - terminal.student_start == 1
                and int(student_ids[alignment.student_length - 1])
                in {int(value) for value in student_stop_ids}
            )
        return bool(teacher_is_stop or student_is_stop)

    def export_runtime_state(self) -> dict[str, Any]:
        if self.student_tokenizer is not None and self.teacher_tokenizer is not None:
            self.runtime_state = {
                **export_tokenizer_runtime_state(
                    self.student_tokenizer, self.teacher_tokenizer
                ),
                **self.runtime_state,
            }
        return super().export_runtime_state()

    def _dense_row_chunk_size(self) -> int:
        chunk_size = int(self.algorithm_kwargs.get("dense_row_chunk_size", 4))
        if chunk_size <= 0:
            raise ValueError("GOLD dense_row_chunk_size must be positive")
        return chunk_size

    @staticmethod
    def _hybrid_normalization_scale(
        *,
        student_active_tokens: int,
        aligned_groups: int,
        step_active_token_count: int | None,
        step_sample_count: int | None,
    ) -> float:
        """Make Slime's token reducer reproduce GOLD's batch reduction.

        The reference trainer first averages both hybrid arms over aligned
        groups inside each sample and then averages sample losses.  Slime's
        outer reducer divides by all active Student tokens in the optimizer
        step.  Therefore each aligned group receives ``T / (N * G_i)`` where
        ``T`` is the step token count, ``N`` the step sample count and ``G_i``
        this sample's aligned-group count.  Standalone/payload callers without
        step metadata retain the equivalent single-sample scale ``T_i / G_i``.
        """

        if aligned_groups <= 0:
            return 0.0
        if step_active_token_count is None and step_sample_count is None:
            return float(student_active_tokens) / float(aligned_groups)
        if step_active_token_count is None or step_sample_count is None:
            raise ValueError("GOLD step normalization metadata must be complete")
        if step_active_token_count <= 0 or step_sample_count <= 0:
            raise ValueError("GOLD step normalization metadata must be positive")
        return float(step_active_token_count) / float(
            step_sample_count * aligned_groups
        )

    @staticmethod
    def _generalized_jsd_rows(
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        *,
        beta: float,
    ) -> torch.Tensor:
        if student_probs.shape != teacher_probs.shape:
            raise ValueError("GOLD matched Student/Teacher tensors must have identical shapes")
        if student_probs.ndim != 2:
            raise ValueError("GOLD matched probabilities must be two-dimensional")
        if student_probs.size(-1) == 0:
            return student_probs.new_zeros((student_probs.size(0),))
        student_log = student_probs.clamp_min(1e-8).log()
        teacher_log = teacher_probs.detach().clamp_min(1e-8).log()
        if beta == 0.0:
            cells = F.kl_div(student_log, teacher_log, reduction="none", log_target=True)
        elif beta == 1.0:
            cells = F.kl_div(teacher_log, student_log, reduction="none", log_target=True)
        else:
            mixture_log = torch.logsumexp(
                torch.stack(
                    (
                        student_log + math.log1p(-beta),
                        teacher_log + math.log(beta),
                    )
                ),
                dim=0,
            )
            teacher_kl = F.kl_div(mixture_log, teacher_log, reduction="none", log_target=True)
            student_kl = F.kl_div(mixture_log, student_log, reduction="none", log_target=True)
            cells = beta * teacher_kl + (1.0 - beta) * student_kl
        return cells.sum(dim=-1)

    @staticmethod
    def _split_hybrid_probabilities(
        probabilities: torch.Tensor,
        matched_ids: torch.Tensor,
        *,
        padded_unmatched_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        matched_ids = matched_ids.to(device=probabilities.device, dtype=torch.long)
        if matched_ids.numel() and bool(
            ((matched_ids < 0) | (matched_ids >= probabilities.size(-1))).any()
        ):
            raise ValueError("GOLD matched token ID is outside the effective vocabulary")
        matched = probabilities.index_select(-1, matched_ids)
        unmatched_mask = torch.ones(probabilities.size(-1), dtype=torch.bool, device=probabilities.device)
        if matched_ids.numel():
            unmatched_mask[matched_ids] = False
        unmatched = probabilities[:, unmatched_mask]
        unmatched_sorted = unmatched.sort(dim=-1, descending=True).values
        if unmatched_sorted.size(-1) < padded_unmatched_size:
            unmatched_sorted = F.pad(
                unmatched_sorted, (0, padded_unmatched_size - unmatched_sorted.size(-1))
            )
        return matched, unmatched_sorted

    def build_teacher_payload(self, context: TeacherScoringContext) -> TensorPayload:
        self._require_public_temperature_one(context.temperature)
        if context.distribution.length != context.alignment.teacher_length:
            raise ValueError("Teacher distribution length does not match the GOLD alignment")
        (
            beta,
            _,
            teacher_temperature,
            _,
            _,
            skip_student_eos,
            skip_teacher_eos,
            fixed_matched_weight,
            fixed_unmatched_weight,
        ) = self._settings()
        del beta
        teacher_vocab_size = resolve_vocab_size(
            self.runtime_state, "teacher_vocab_size", self.teacher_tokenizer
        )
        student_vocab_size = resolve_vocab_size(
            self.runtime_state, "student_vocab_size", self.student_tokenizer
        )
        drop_terminal = self._drop_terminal_segment_if_eos(
            context.alignment,
            context.response_token_ids,
            context.student_response_token_ids,
        )
        teacher_starts, teacher_ends, student_starts, student_ends = aligned_segment_indices(
            context.alignment,
            context.student_loss_mask,
            device=context.response_token_ids.device,
            drop_terminal_segment=drop_terminal,
        )

        full_probs = context.distribution.full_probabilities(
            vocab_size=teacher_vocab_size,
            temperature=teacher_temperature,
        )
        teacher_aligned = merge_observed_probabilities(
            full_probs, teacher_starts, teacher_ends, context.response_token_ids
        )
        teacher_ids = context.vocabulary_mapping.teacher_ids.to(
            device=full_probs.device, dtype=torch.long
        )
        matched_count = int(teacher_ids.numel())
        student_unmatched_size = student_vocab_size - matched_count
        teacher_unmatched_size = teacher_vocab_size - matched_count
        padded_unmatched_size = max(student_unmatched_size, teacher_unmatched_size)
        teacher_matched, teacher_unmatched = self._split_hybrid_probabilities(
            teacher_aligned,
            teacher_ids,
            padded_unmatched_size=padded_unmatched_size,
        )
        if fixed_matched_weight is None:
            matched_weight = matched_count / max(teacher_vocab_size, 1)
            unmatched_weight = 1.0 - matched_weight
        else:
            matched_weight = fixed_matched_weight
            unmatched_weight = fixed_unmatched_weight
        active_denominator = int(context.student_loss_mask.to(dtype=torch.bool).sum())
        normalization_scale = self._hybrid_normalization_scale(
            student_active_tokens=active_denominator,
            aligned_groups=int(student_starts.numel()),
            step_active_token_count=None,
            step_sample_count=None,
        )
        def scalar(value: float) -> torch.Tensor:
            return torch.tensor(
                value, dtype=torch.float32, device=context.response_token_ids.device
            )
        return {
            "teacher_matched_probs": teacher_matched.detach(),
            "teacher_unmatched_sorted": teacher_unmatched.detach(),
            "student_starts": student_starts.detach(),
            "student_ends": student_ends.detach(),
            "student_matched_ids": context.vocabulary_mapping.student_ids.detach(),
            "matched_weight": scalar(matched_weight),
            "unmatched_weight": scalar(unmatched_weight),
            "normalization_scale": scalar(normalization_scale),
        }

    def compute_student_token_losses(
        self,
        context: StudentScoringContext,
        teacher_payload: TensorPayload,
    ) -> torch.Tensor:
        self._require_public_temperature_one(context.temperature)
        required = {
            "teacher_matched_probs",
            "teacher_unmatched_sorted",
            "student_starts",
            "student_ends",
            "student_matched_ids",
            "matched_weight",
            "unmatched_weight",
            "normalization_scale",
        }
        missing = required.difference(teacher_payload)
        if missing:
            raise ValueError(f"GOLD Teacher payload is missing fields: {sorted(missing)}")
        (
            beta,
            student_temperature,
            _,
            distillation_weight,
            crossentropy_weight,
            *_
        ) = self._settings()
        student_vocab_size = resolve_vocab_size(
            self.runtime_state, "student_vocab_size", self.student_tokenizer
        )

        actual_log_probs = context.distribution.selected_log_probs(
            context.response_token_ids.unsqueeze(-1), vocab_size=student_vocab_size
        ).squeeze(-1)
        token_losses = -actual_log_probs * crossentropy_weight
        starts = teacher_payload["student_starts"].to(device=token_losses.device, dtype=torch.long)
        ends = teacher_payload["student_ends"].to(device=token_losses.device, dtype=torch.long)
        if starts.numel() == 0:
            return token_losses
        full_probs = context.distribution.full_probabilities(
            vocab_size=student_vocab_size,
            temperature=student_temperature,
        )
        student_aligned = merge_observed_probabilities(
            full_probs, starts, ends, context.response_token_ids
        )
        matched_ids = teacher_payload["student_matched_ids"].to(
            device=token_losses.device, dtype=torch.long
        )
        teacher_unmatched = teacher_payload["teacher_unmatched_sorted"].to(
            device=token_losses.device, dtype=torch.float32
        )
        student_matched, student_unmatched = self._split_hybrid_probabilities(
            student_aligned,
            matched_ids,
            padded_unmatched_size=teacher_unmatched.size(-1),
        )
        teacher_matched = teacher_payload["teacher_matched_probs"].to(
            device=token_losses.device, dtype=torch.float32
        )
        if teacher_matched.shape != student_matched.shape:
            raise ValueError("GOLD matched Teacher/Student probability shapes differ")
        if teacher_unmatched.shape != student_unmatched.shape:
            raise ValueError("GOLD unmatched Teacher/Student probability shapes differ")
        matched_loss = self._generalized_jsd_rows(
            student_matched, teacher_matched, beta=beta
        )
        unmatched_loss = (
            (student_unmatched - teacher_unmatched.detach()).abs().sum(dim=-1)
            if student_unmatched.size(-1)
            else student_unmatched.new_zeros((student_unmatched.size(0),))
        )
        matched_weight = teacher_payload["matched_weight"].to(matched_loss)
        unmatched_weight = teacher_payload["unmatched_weight"].to(unmatched_loss)
        if context.step_active_token_count is None and context.step_sample_count is None:
            scale = teacher_payload["normalization_scale"].to(matched_loss)
        else:
            scale = matched_loss.new_tensor(
                self._hybrid_normalization_scale(
                    student_active_tokens=context.distribution.length,
                    aligned_groups=int(starts.numel()),
                    step_active_token_count=context.step_active_token_count,
                    step_sample_count=context.step_sample_count,
                )
            )
        row_losses = (
            matched_weight * matched_loss + unmatched_weight * unmatched_loss
        ) * scale * distillation_weight
        return token_losses.index_add(0, starts, row_losses.to(token_losses.dtype))

    def compute_fused_token_losses(self, context: FusedScoringContext) -> torch.Tensor:
        """Evaluate exact GOLD from causal Teacher hidden rows inside Slime.

        GOLD needs complete Teacher and Student vocabularies.  The fused path
        therefore reconstructs only a bounded number of aligned start rows at
        a time and never serializes a dense ``[response, vocabulary]`` payload
        through Ray or HTTP.
        """

        self._require_public_temperature_one(context.student.temperature)
        alignment = context.alignment
        if alignment.skipped:
            raise ValueError("A skipped GOLD alignment must be handled before fused scoring")
        if context.teacher_distribution.length != alignment.teacher_length:
            raise ValueError("Teacher distribution length does not match the GOLD alignment")
        device = context.student.response_token_ids.device
        teacher_response_ids = context.teacher_response_token_ids.to(
            device=device, dtype=torch.long
        ).flatten()
        if teacher_response_ids.numel() != alignment.teacher_length:
            raise ValueError("Teacher response token IDs do not match the GOLD alignment")

        local_positions = context.local_response_positions.to(
            device=device, dtype=torch.long
        ).flatten()
        if local_positions.numel() != context.student.distribution.length:
            raise ValueError("GOLD local response positions differ from Student distribution rows")
        if local_positions.numel() > 1 and bool(
            (local_positions[1:] <= local_positions[:-1]).any()
        ):
            raise ValueError("GOLD local response positions must be strictly increasing")
        if local_positions.numel() and bool(
            ((local_positions < 0) | (local_positions >= alignment.student_length)).any()
        ):
            raise ValueError("GOLD local response position falls outside the alignment")

        mapping = context.vocabulary_mapping
        if mapping is None:
            raise ValueError("Fused GOLD requires its exact shared-vocabulary mapping")
        teacher_matched_ids = mapping.teacher_ids.to(device=device, dtype=torch.long)
        student_matched_ids = mapping.student_ids.to(device=device, dtype=torch.long)
        if teacher_matched_ids.numel() != student_matched_ids.numel():
            raise ValueError("GOLD shared Teacher/Student vocabulary widths differ")

        (
            beta,
            student_temperature,
            teacher_temperature,
            distillation_weight,
            crossentropy_weight,
            _,
            _,
            fixed_matched_weight,
            fixed_unmatched_weight,
        ) = self._settings()
        student_vocab_size = resolve_vocab_size(
            self.runtime_state, "student_vocab_size", self.student_tokenizer
        )
        teacher_vocab_size = resolve_vocab_size(
            self.runtime_state, "teacher_vocab_size", self.teacher_tokenizer
        )
        full_mask = torch.ones(alignment.student_length, dtype=torch.bool, device=device)
        student_response_ids = context.student_response_token_ids
        if student_response_ids is None:
            if local_positions.numel() != alignment.student_length:
                raise ValueError(
                    "GOLD CP scoring requires the full Student response token IDs"
                )
            student_response_ids = context.student.response_token_ids
        student_response_ids = student_response_ids.to(device=device, dtype=torch.long).flatten()
        if student_response_ids.numel() != alignment.student_length:
            raise ValueError("Student response token IDs do not match the GOLD alignment")
        teacher_starts, teacher_ends, student_starts, student_ends = aligned_segment_indices(
            alignment,
            full_mask,
            device=device,
            drop_terminal_segment=self._drop_terminal_segment_if_eos(
                alignment, teacher_response_ids, student_response_ids
            ),
        )

        actual_log_probs = context.student.distribution.selected_log_probs(
            context.student.response_token_ids.unsqueeze(-1),
            vocab_size=student_vocab_size,
        ).squeeze(-1)
        token_losses = -actual_log_probs * crossentropy_weight

        local_student_observed = context.student.distribution.selected_log_probs(
            context.student.response_token_ids.unsqueeze(-1),
            vocab_size=student_vocab_size,
            temperature=student_temperature,
        ).squeeze(-1)
        if context.gather_student_rows is None:
            expected_positions = torch.arange(
                alignment.student_length, device=device, dtype=torch.long
            )
            if not torch.equal(local_positions, expected_positions):
                raise ValueError("GOLD needs a CP Student-row gather for non-local groups")
            student_observed = local_student_observed
        else:
            student_observed = context.gather_student_rows(local_student_observed)
        if student_observed.shape != (alignment.student_length,):
            raise ValueError("Gathered GOLD Student scores have the wrong global response length")
        total_group_count = int(student_starts.numel())
        if total_group_count == 0:
            # Every CP rank must retain a dependency on the gather, including
            # ranks that own no aligned start row for this packed response.
            return token_losses + student_observed.sum() * 0.0

        local_start_rows = torch.searchsorted(local_positions, student_starts)
        if local_positions.numel():
            owned = (local_start_rows < local_positions.numel()) & (
                local_positions.index_select(
                    0, local_start_rows.clamp(max=local_positions.numel() - 1)
                )
                == student_starts
            )
        else:
            owned = torch.zeros_like(student_starts, dtype=torch.bool)
        owned_indices = torch.nonzero(owned, as_tuple=False).flatten()
        if owned_indices.numel() == 0:
            return token_losses + student_observed.sum() * 0.0
        teacher_starts = teacher_starts.index_select(0, owned_indices)
        teacher_ends = teacher_ends.index_select(0, owned_indices)
        student_starts = student_starts.index_select(0, owned_indices)
        student_ends = student_ends.index_select(0, owned_indices)
        local_start_rows = local_start_rows.index_select(0, owned_indices)

        teacher_observed = context.teacher_distribution.selected_log_probs(
            teacher_response_ids.unsqueeze(-1),
            vocab_size=teacher_vocab_size,
            temperature=teacher_temperature,
        ).squeeze(-1).exp().clamp_min(1e-8).log()
        student_observed = student_observed.clamp_max(0.0)
        teacher_prefix = torch.cat(
            (teacher_observed.new_zeros(1), teacher_observed.cumsum(dim=0))
        )
        student_prefix = torch.cat(
            (student_observed.new_zeros(1), student_observed.cumsum(dim=0))
        )

        matched_count = int(teacher_matched_ids.numel())
        padded_unmatched_size = max(
            student_vocab_size - matched_count,
            teacher_vocab_size - matched_count,
        )
        if fixed_matched_weight is None:
            matched_weight = matched_count / max(teacher_vocab_size, 1)
            unmatched_weight = 1.0 - matched_weight
        else:
            matched_weight = fixed_matched_weight
            unmatched_weight = fixed_unmatched_weight
        normalization_scale = self._hybrid_normalization_scale(
            student_active_tokens=alignment.student_length,
            aligned_groups=total_group_count,
            step_active_token_count=context.student.step_active_token_count,
            step_sample_count=context.student.step_sample_count,
        )

        output = token_losses.float() + student_observed.sum() * 0.0
        chunk_size = self._dense_row_chunk_size()
        for offset in range(0, student_starts.numel(), chunk_size):
            selected = slice(offset, offset + chunk_size)
            current_teacher_starts = teacher_starts[selected]
            current_teacher_ends = teacher_ends[selected]
            current_student_starts = student_starts[selected]
            current_student_ends = student_ends[selected]

            teacher_aligned = context.teacher_distribution.rows(
                current_teacher_starts
            ).full_probabilities(
                vocab_size=teacher_vocab_size,
                temperature=teacher_temperature,
            )
            teacher_continuation = (
                teacher_prefix[current_teacher_ends]
                - teacher_prefix[current_teacher_starts + 1]
            ).exp()
            teacher_aligned = teacher_aligned * teacher_continuation.unsqueeze(-1)

            student_aligned = context.student.distribution.rows(
                local_start_rows[selected]
            ).full_probabilities(
                vocab_size=student_vocab_size,
                temperature=student_temperature,
            )
            student_continuation = (
                student_prefix[current_student_ends]
                - student_prefix[current_student_starts + 1]
            ).exp()
            student_aligned = student_aligned * student_continuation.unsqueeze(-1)

            teacher_matched, teacher_unmatched = self._split_hybrid_probabilities(
                teacher_aligned,
                teacher_matched_ids,
                padded_unmatched_size=padded_unmatched_size,
            )
            student_matched, student_unmatched = self._split_hybrid_probabilities(
                student_aligned,
                student_matched_ids,
                padded_unmatched_size=padded_unmatched_size,
            )
            matched_loss = self._generalized_jsd_rows(
                student_matched, teacher_matched, beta=beta
            )
            unmatched_loss = (
                (student_unmatched - teacher_unmatched.detach()).abs().sum(dim=-1)
                if student_unmatched.size(-1)
                else student_unmatched.new_zeros((student_unmatched.size(0),))
            )
            row_losses = (
                float(matched_weight) * matched_loss
                + float(unmatched_weight) * unmatched_loss
            ) * normalization_scale * distillation_weight
            if not bool(torch.isfinite(row_losses).all()):
                raise FloatingPointError("GOLD produced a non-finite fused loss")
            output = output.index_add(0, local_start_rows[selected], row_losses.float())
        return output.to(token_losses.dtype)
