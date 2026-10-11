"""X-Token projection-guided cross-tokenizer distillation (arXiv:2605.21699).

Independent implementation of Sreenivas et al., "X-Token: Projection-Guided
Cross-Tokenizer Knowledge Distillation".  The paper publishes no official
training code; this plugin keeps the published loss formulas exact while
reusing this repository's GOLD span alignment and observed chain-rule merge,
which the paper inherits from the same line of work.

Two loss modes, selected by ``algorithm_kwargs.loss_mode``:

* ``"hkl"`` — relax GOLD's 1-to-1 common set with the top-1 teacher token of
  the projection matrix W, then apply the unchanged GOLD hybrid loss over the
  expanded common set.  Used when the coverage audit keeps critical tokens
  inside the common set (the audited regime for Qwen3-32B and GLM-Z1-9B
  teachers into Qwen3.5-2B).
* ``"pkl"`` — partition-free: project the merged Student chunk distribution
  into Teacher vocabulary space with W and compute a direct KL against the
  merged Teacher distribution over the Teacher top-K support.

Deviations frozen for v1 (must be restated in every run manifest):

* W is frozen after the deterministic two-pass construction.  The paper's
  main P-KL result additionally learns W jointly (LR 1e-2); that refinement
  is v2 work.  Frozen-W P-KL is the paper's ablation arm.
* The paper's dynamic KD/CE rescaling ``sg(L_CE / L_KD) * L_KD + L_CE`` is a
  batch-level scalar ratio that requires a step-wide pre-aggregation pass the
  fused per-sample loss cannot express.  v1 keeps the static
  ``distillation_weight`` / ``crossentropy_weight`` combination shared with
  the GOLD baseline recipe (CE arm off by default in this repository's OPD
  recipes).
* Span alignment reuses GOLD's byte-boundary alignment instead of the
  paper's DP scoring alignment.  The paper's DP criticism targets TRL's
  incremental substring flush, which this byte alignment does not share.
"""

from __future__ import annotations

import base64
import math
import zlib

import numpy as np
import torch
from typing import Any

from ._paper_utils import (
    aligned_segment_indices,
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
    normalize_json_object,
)
from .gold import GOLDCrossTokenizer
from .registry import register_cross_tokenizer
from .xtoken_core import (
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_MAX_SPAN,
    DEFAULT_TOP_K,
    ProjectionMatrix,
    build_projection_matrix,
)

_LOSS_MODES = ("hkl", "pkl")
_DEFAULT_TEACHER_TOP_K = 8192
_LOG_FLOOR = 1e-8

# The scoring-side projection tensors depend only on the immutable JSON
# runtime state.  Cache them per process so the per-microbatch algorithm
# reconstruction does not rebuild a ~200k-entry index triple every step.
_PROJECTION_TENSOR_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


_PROJECTION_FORMAT = "b64z-v1"


def _pack_array(values: Any, dtype: Any) -> str:
    array = np.asarray(values, dtype=dtype)
    return base64.b64encode(zlib.compress(array.tobytes(), 9)).decode("ascii")


def _unpack_array(blob: Any, dtype: Any, count: int, *, name: str) -> np.ndarray:
    if not isinstance(blob, str):
        raise ValueError(f"X-Token projection state {name!r} must be a packed string")
    try:
        raw = zlib.decompress(base64.b64decode(blob))
    except Exception as error:
        raise ValueError(f"X-Token projection state {name!r} is not a valid packed array") from error
    array = np.frombuffer(raw, dtype=dtype)
    if array.size != count:
        raise ValueError(
            f"X-Token projection state {name!r} decodes to {array.size} elements, expected {count}"
        )
    return array


def _projection_cache_fingerprint(
    runtime_state: dict[str, Any],
    student_vocab_size: int,
    teacher_vocab_size: int,
    device: torch.device,
) -> tuple:
    teacher_ids = runtime_state["xtoken_w_teacher_ids"]
    weights = runtime_state["xtoken_w_weights"]
    return (
        str(device),
        int(student_vocab_size),
        int(teacher_vocab_size),
        int(runtime_state["xtoken_w_rows"]),
        int(runtime_state["xtoken_w_entries"]),
        teacher_ids[:24],
        teacher_ids[-24:],
        weights[:24],
        weights[-24:],
    )


@register_cross_tokenizer("xtoken")
class XTokenCrossTokenizer(GOLDCrossTokenizer):
    """X-Token H-KL / P-KL losses on GOLD's alignment and merge machinery."""

    capabilities = CrossTokenizerCapabilities(
        fidelity="paper-equation",
        reference=(
            "X-Token arXiv:2605.21699 (no official training code); frozen-W, "
            "static KD/CE weights, GOLD byte-boundary span alignment"
        ),
        candidate_top_k_mode="zero",
        default_candidate_top_k=0,
        teacher_backends=("megatron", "sglang"),
        required_distribution_methods=("full_probabilities", "selected_log_probs"),
        public_temperature_mode="one",
        # Same contract as GOLD: the payload path is CP=1 only; the
        # causal-hidden path gathers realized Student continuation scalars.
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

    def _loss_mode(self) -> str:
        mode = str(self.algorithm_kwargs.get("loss_mode", "hkl")).strip().lower()
        if mode not in _LOSS_MODES:
            raise ValueError(f"X-Token loss_mode must be one of {_LOSS_MODES}, got {mode!r}")
        return mode

    def _projection_settings(self) -> tuple[int, float, float, int, int]:
        top_k = int(self.algorithm_kwargs.get("projection_top_k", DEFAULT_TOP_K))
        beta = float(self.algorithm_kwargs.get("projection_beta", DEFAULT_BETA))
        gamma = float(self.algorithm_kwargs.get("projection_gamma", DEFAULT_GAMMA))
        max_span = self.algorithm_kwargs.get("projection_max_span", DEFAULT_MAX_SPAN)
        teacher_top_k = int(self.algorithm_kwargs.get("teacher_top_k", _DEFAULT_TEACHER_TOP_K))
        if top_k <= 0:
            raise ValueError(f"X-Token projection_top_k must be positive, got {top_k}")
        if not math.isfinite(beta) or not 0 < beta <= 1:
            raise ValueError(f"X-Token projection_beta must be in (0, 1], got {beta}")
        if not math.isfinite(gamma) or not 0 < gamma < 1:
            raise ValueError(f"X-Token projection_gamma must be in (0, 1), got {gamma}")
        if isinstance(max_span, bool) or int(max_span) != max_span or int(max_span) <= 0:
            raise ValueError(
                f"X-Token projection_max_span must be a positive integer, got {max_span!r}"
            )
        max_span = int(max_span)
        if teacher_top_k <= 0:
            raise ValueError(f"X-Token teacher_top_k must be positive, got {teacher_top_k}")
        return top_k, beta, gamma, max_span, teacher_top_k

    def _validate_mode_kwargs(self) -> None:
        """Reject knobs whose silent presence would change the declared loss."""

        mode = self._loss_mode()
        self._projection_settings()
        if mode == "pkl":
            if "beta" in self.algorithm_kwargs:
                raise ValueError(
                    "X-Token algorithm_kwargs.beta is GOLD's JSD knob and does not apply to P-KL; "
                    "use projection_beta for W construction"
                )
            for name in ("matched_weight", "unmatched_weight"):
                if name in self.algorithm_kwargs:
                    raise ValueError(
                        f"X-Token P-KL is partition-free; algorithm_kwargs.{name} only applies to H-KL"
                    )
        else:
            for name in ("matched_weight", "unmatched_weight"):
                if name not in self.algorithm_kwargs:
                    raise ValueError(
                        f"X-Token H-KL freezes lambda_KL=lambda_ULD=1; set algorithm_kwargs.{name} explicitly"
                    )
            if "teacher_top_k" in self.algorithm_kwargs:
                raise ValueError("X-Token teacher_top_k only applies to P-KL")

    def _projection_matrix(self) -> ProjectionMatrix:
        """Build (once) the deterministic two-pass projection matrix W."""

        cached = getattr(self, "_cached_projection_matrix", None)
        if cached is not None:
            return cached
        if self.student_tokenizer is None or self.teacher_tokenizer is None:
            raise RuntimeError(
                "X-Token projection matrix construction requires both tokenizers; "
                "GPU scoring must consume the exported runtime_state instead"
            )
        top_k, beta, gamma, max_span, _ = self._projection_settings()
        matrix = build_projection_matrix(
            self.student_tokenizer,
            self.teacher_tokenizer,
            top_k=top_k,
            beta=beta,
            gamma=gamma,
            max_span=max_span,
        )
        self._cached_projection_matrix = matrix
        return matrix

    # ------------------------------------------------------------------
    # H-KL: relaxed 1-to-1 common set handed to the unchanged GOLD loss.
    # ------------------------------------------------------------------

    def build_shared_vocabulary(self) -> SharedVocabulary | None:
        if self._loss_mode() == "pkl":
            # P-KL is partition-free; supervision flows through W only.
            return None
        matrix = self._projection_matrix()
        strict = super().build_shared_vocabulary()
        if strict is None:
            student_ids: list[int] = []
            teacher_ids: list[int] = []
            tokens: list[str] = []
        else:
            # H-KL extends GOLD's strict common set; it must never allow a
            # canonical spelling variant to steal the Teacher id of a native
            # raw-string match.  Seed the bijection with every GOLD pair first.
            student_ids = list(strict.student_ids)
            teacher_ids = list(strict.teacher_ids)
            tokens = [f"xtoken-strict:{token}" for token in strict.tokens]
        admitted = set(student_ids)
        used_teacher: set[int] = set(teacher_ids)
        # Next admit remaining canonical exact matches. Canonicalization can
        # collapse several spelling variants per side, so each Student token
        # greedily takes its lowest-id free exact partner.
        for student_id in sorted(matrix.exact_student_ids):
            if student_id in admitted:
                continue
            for teacher_id, _ in matrix.row(student_id):
                teacher_id = int(teacher_id)
                if teacher_id not in used_teacher:
                    admitted.add(int(student_id))
                    used_teacher.add(teacher_id)
                    student_ids.append(int(student_id))
                    teacher_ids.append(teacher_id)
                    tokens.append(f"xtoken-canonical:{student_id}")
                    break
        # Top-1 expansion: admit (s, argmax_t W[s, t]) when both sides are free.
        for student_id in sorted(matrix.rows):
            if student_id in admitted:
                continue
            teacher_id = matrix.top1(student_id)
            if teacher_id is None or teacher_id in used_teacher:
                continue
            admitted.add(student_id)
            used_teacher.add(teacher_id)
            student_ids.append(int(student_id))
            teacher_ids.append(int(teacher_id))
            tokens.append(f"xtoken-top1:{student_id}")
        if not tokens:
            raise ValueError("X-Token H-KL produced an empty expanded common set")
        return SharedVocabulary(
            tuple(student_ids),
            tuple(teacher_ids),
            tuple(tokens),
        )

    # ------------------------------------------------------------------
    # P-KL: runtime transport of W and projection-KL losses.
    # ------------------------------------------------------------------

    def export_runtime_state(self) -> dict[str, Any]:
        state = super().export_runtime_state()
        if self._loss_mode() != "pkl":
            return state
        matrix = self._projection_matrix()
        student_ids: list[int] = []
        counts: list[int] = []
        teacher_ids: list[int] = []
        weights: list[float] = []
        for student_id in sorted(matrix.rows):
            entries = matrix.row(student_id)
            if not entries:
                continue
            student_ids.append(int(student_id))
            counts.append(len(entries))
            for teacher_id, weight in entries:
                teacher_ids.append(int(teacher_id))
                weights.append(float(weight))
        # The CSR table runs ~400k entries for the production pairs; pack each
        # array as a compressed base64 string so the per-microbatch JSON
        # round-trip of runtime_state stays a string copy, not number parsing.
        state = {
            **state,
            "xtoken_w_format": _PROJECTION_FORMAT,
            "xtoken_w_rows": len(student_ids),
            "xtoken_w_entries": len(teacher_ids),
            "xtoken_w_student_ids": _pack_array(student_ids, np.uint32),
            "xtoken_w_counts": _pack_array(counts, np.uint8),
            "xtoken_w_teacher_ids": _pack_array(teacher_ids, np.uint32),
            "xtoken_w_weights": _pack_array(weights, np.float32),
        }
        return normalize_json_object(state, field_name="xtoken runtime_state")

    def _projection_tensors(
        self,
        *,
        device: torch.device,
        student_vocab_size: int,
        teacher_vocab_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand the CSR runtime state to per-entry (student, teacher, weight)."""

        required = (
            "xtoken_w_format",
            "xtoken_w_rows",
            "xtoken_w_entries",
            "xtoken_w_student_ids",
            "xtoken_w_counts",
            "xtoken_w_teacher_ids",
            "xtoken_w_weights",
        )
        for name in required:
            if name not in self.runtime_state:
                raise ValueError(
                    f"X-Token P-KL runtime_state is missing {name!r}; the matrix must be exported "
                    "from a tokenizer-bearing construction"
                )
        if self.runtime_state["xtoken_w_format"] != _PROJECTION_FORMAT:
            raise ValueError(
                f"X-Token projection state format must be {_PROJECTION_FORMAT!r}, "
                f"got {self.runtime_state['xtoken_w_format']!r}"
            )
        fingerprint = _projection_cache_fingerprint(
            self.runtime_state, student_vocab_size, teacher_vocab_size, device
        )
        cached = _PROJECTION_TENSOR_CACHE.get(fingerprint)
        if cached is not None:
            return cached
        row_count = int(self.runtime_state["xtoken_w_rows"])
        entry_count = int(self.runtime_state["xtoken_w_entries"])
        student_ids = _unpack_array(
            self.runtime_state["xtoken_w_student_ids"], np.uint32, row_count, name="xtoken_w_student_ids"
        )
        counts = _unpack_array(
            self.runtime_state["xtoken_w_counts"], np.uint8, row_count, name="xtoken_w_counts"
        )
        teacher_ids = _unpack_array(
            self.runtime_state["xtoken_w_teacher_ids"], np.uint32, entry_count, name="xtoken_w_teacher_ids"
        )
        weights = _unpack_array(
            self.runtime_state["xtoken_w_weights"], np.float32, entry_count, name="xtoken_w_weights"
        )
        if int(counts.astype(np.int64).sum()) != entry_count:
            raise ValueError("X-Token projection CSR counts do not sum to the entry count")
        entry_student = torch.repeat_interleave(
            torch.from_numpy(student_ids.astype(np.int64)),
            torch.from_numpy(counts.astype(np.int64)),
        )
        entry_teacher = torch.from_numpy(teacher_ids.astype(np.int64))
        entry_weight = torch.from_numpy(weights.astype(np.float32))
        if entry_student.numel():
            if int(entry_student.max()) >= student_vocab_size or int(entry_student.min()) < 0:
                raise ValueError("X-Token projection Student id outside the effective vocabulary")
            if int(entry_teacher.max()) >= teacher_vocab_size or int(entry_teacher.min()) < 0:
                raise ValueError("X-Token projection Teacher id outside the effective vocabulary")
            if not bool(torch.isfinite(entry_weight).all()):
                raise ValueError("X-Token projection weights must be finite")
        tensors = (
            entry_student.to(device),
            entry_teacher.to(device),
            entry_weight.to(device),
        )
        _PROJECTION_TENSOR_CACHE[fingerprint] = tensors
        return tensors

    def _pkl_projected_log_rows(
        self,
        student_aligned: torch.Tensor,
        projection: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        teacher_vocab_size: int,
    ) -> torch.Tensor:
        """Log of the W-projected, renormalized Student rows over V_T.

        ``student_aligned`` is graph-connected and uses the observed
        chain-rule merge (rows sum to <= 1).  The projection is renormalized
        over the full Teacher vocabulary, matching the paper's P-KL
        normalization of the truncated W.
        """

        if student_aligned.ndim != 2:
            raise ValueError("X-Token P-KL expects two-dimensional merged Student rows")
        entry_student, entry_teacher, entry_weight = projection
        gathered = student_aligned.index_select(1, entry_student) * entry_weight
        projected = student_aligned.new_zeros((student_aligned.size(0), teacher_vocab_size))
        projected = projected.scatter_add(
            1,
            entry_teacher.unsqueeze(0).expand(student_aligned.size(0), -1),
            gathered,
        )
        projected = projected / projected.sum(dim=-1, keepdim=True).clamp_min(_LOG_FLOOR)
        return projected.clamp_min(_LOG_FLOOR).log()

    def _pkl_kl_from_teacher_topk(
        self,
        projected_log: torch.Tensor,
        teacher_topk_probs: torch.Tensor,
        teacher_topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """KL(q_T || p~_S) with q_T the Teacher distribution renormed on top-K."""

        if projected_log.size(0) != teacher_topk_probs.size(0):
            raise ValueError("X-Token P-KL merged row counts differ")
        teacher_q = teacher_topk_probs / teacher_topk_probs.sum(dim=-1, keepdim=True).clamp_min(_LOG_FLOOR)
        return (
            teacher_q
            * (
                teacher_q.clamp_min(_LOG_FLOOR).log()
                - projected_log.gather(1, teacher_topk_ids.long())
            )
        ).sum(dim=-1)

    def build_teacher_payload(self, context: TeacherScoringContext) -> TensorPayload:
        if self._loss_mode() == "hkl":
            return super().build_teacher_payload(context)
        self._require_public_temperature_one(context.temperature)
        self._validate_mode_kwargs()
        if context.distribution.length != context.alignment.teacher_length:
            raise ValueError("Teacher distribution length does not match the X-Token alignment")
        (_, _, teacher_temperature, _, _, _, _, _, _) = self._settings()
        _, _, _, _, teacher_top_k = self._projection_settings()
        teacher_vocab_size = resolve_vocab_size(
            self.runtime_state, "teacher_vocab_size", self.teacher_tokenizer
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
        ).detach()
        k = min(teacher_top_k, teacher_vocab_size)
        if teacher_aligned.size(0):
            top_values, top_ids = teacher_aligned.topk(k, dim=-1)
        else:
            top_values = teacher_aligned.new_zeros((0, k))
            top_ids = torch.empty((0, k), dtype=torch.long, device=teacher_aligned.device)
        active_denominator = int(context.student_loss_mask.to(dtype=torch.bool).sum())
        normalization_scale = self._hybrid_normalization_scale(
            student_active_tokens=active_denominator,
            aligned_groups=int(student_starts.numel()),
            step_active_token_count=None,
            step_sample_count=None,
        )
        return {
            "teacher_topk_probs": top_values.detach(),
            "teacher_topk_ids": top_ids.detach(),
            "student_starts": student_starts.detach(),
            "student_ends": student_ends.detach(),
            "normalization_scale": torch.tensor(
                normalization_scale, dtype=torch.float32, device=context.response_token_ids.device
            ),
        }

    def compute_student_token_losses(
        self,
        context: StudentScoringContext,
        teacher_payload: TensorPayload,
    ) -> torch.Tensor:
        if self._loss_mode() == "hkl":
            return super().compute_student_token_losses(context, teacher_payload)
        self._require_public_temperature_one(context.temperature)
        self._validate_mode_kwargs()
        required = {
            "teacher_topk_probs",
            "teacher_topk_ids",
            "student_starts",
            "student_ends",
            "normalization_scale",
        }
        missing = required.difference(teacher_payload)
        if missing:
            raise ValueError(f"X-Token P-KL Teacher payload is missing fields: {sorted(missing)}")
        (_, student_temperature, _, distillation_weight, crossentropy_weight, *_) = self._settings()
        _, _, _, _, teacher_top_k = self._projection_settings()
        student_vocab_size = resolve_vocab_size(
            self.runtime_state, "student_vocab_size", self.student_tokenizer
        )
        teacher_vocab_size = resolve_vocab_size(
            self.runtime_state, "teacher_vocab_size", self.teacher_tokenizer
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
        projection = self._projection_tensors(
            device=token_losses.device,
            student_vocab_size=student_vocab_size,
            teacher_vocab_size=teacher_vocab_size,
        )
        projected_log = self._pkl_projected_log_rows(
            student_aligned,
            projection,
            teacher_vocab_size=teacher_vocab_size,
        )
        row_losses = self._pkl_kl_from_teacher_topk(
            projected_log,
            teacher_payload["teacher_topk_probs"].to(device=token_losses.device, dtype=torch.float32),
            teacher_payload["teacher_topk_ids"].to(device=token_losses.device, dtype=torch.long),
        )
        if context.step_active_token_count is None and context.step_sample_count is None:
            scale = teacher_payload["normalization_scale"].to(row_losses)
        else:
            scale = row_losses.new_tensor(
                self._hybrid_normalization_scale(
                    student_active_tokens=context.distribution.length,
                    aligned_groups=int(starts.numel()),
                    step_active_token_count=context.step_active_token_count,
                    step_sample_count=context.step_sample_count,
                )
            )
        row_losses = row_losses * scale * distillation_weight
        if not bool(torch.isfinite(row_losses).all()):
            raise FloatingPointError("X-Token P-KL produced a non-finite loss")
        return token_losses.index_add(0, starts, row_losses.to(token_losses.dtype))

    def compute_fused_token_losses(self, context: FusedScoringContext) -> torch.Tensor:
        if self._loss_mode() == "hkl":
            return super().compute_fused_token_losses(context)
        self._require_public_temperature_one(context.student.temperature)
        self._validate_mode_kwargs()
        alignment = context.alignment
        if alignment.skipped:
            raise ValueError("A skipped X-Token alignment must be handled before fused scoring")
        if context.teacher_distribution.length != alignment.teacher_length:
            raise ValueError("Teacher distribution length does not match the X-Token alignment")
        device = context.student.response_token_ids.device
        teacher_response_ids = context.teacher_response_token_ids.to(
            device=device, dtype=torch.long
        ).flatten()
        if teacher_response_ids.numel() != alignment.teacher_length:
            raise ValueError("Teacher response token IDs do not match the X-Token alignment")
        local_positions = context.local_response_positions.to(
            device=device, dtype=torch.long
        ).flatten()
        if local_positions.numel() != context.student.distribution.length:
            raise ValueError("X-Token local response positions differ from Student distribution rows")
        if local_positions.numel() > 1 and bool(
            (local_positions[1:] <= local_positions[:-1]).any()
        ):
            raise ValueError("X-Token local response positions must be strictly increasing")
        if local_positions.numel() and bool(
            ((local_positions < 0) | (local_positions >= alignment.student_length)).any()
        ):
            raise ValueError("X-Token local response position falls outside the alignment")

        (
            _,
            student_temperature,
            teacher_temperature,
            distillation_weight,
            crossentropy_weight,
            _,
            _,
            _,
            _,
        ) = self._settings()
        _, _, _, _, teacher_top_k = self._projection_settings()
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
                raise ValueError("X-Token CP scoring requires the full Student response token IDs")
            student_response_ids = context.student.response_token_ids
        student_response_ids = student_response_ids.to(device=device, dtype=torch.long).flatten()
        if student_response_ids.numel() != alignment.student_length:
            raise ValueError("Student response token IDs do not match the X-Token alignment")
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
                raise ValueError("X-Token needs a CP Student-row gather for non-local groups")
            student_observed = local_student_observed
        else:
            student_observed = context.gather_student_rows(local_student_observed)
        if student_observed.shape != (alignment.student_length,):
            raise ValueError("Gathered X-Token Student scores have the wrong global response length")
        total_group_count = int(student_starts.numel())
        if total_group_count == 0:
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
        ).squeeze(-1).exp().clamp_min(_LOG_FLOOR).log()
        student_observed = student_observed.clamp_max(0.0)
        teacher_prefix = torch.cat(
            (teacher_observed.new_zeros(1), teacher_observed.cumsum(dim=0))
        )
        student_prefix = torch.cat(
            (student_observed.new_zeros(1), student_observed.cumsum(dim=0))
        )
        normalization_scale = self._hybrid_normalization_scale(
            student_active_tokens=alignment.student_length,
            aligned_groups=total_group_count,
            step_active_token_count=context.student.step_active_token_count,
            step_sample_count=context.student.step_sample_count,
        )
        projection = self._projection_tensors(
            device=device,
            student_vocab_size=student_vocab_size,
            teacher_vocab_size=teacher_vocab_size,
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

            projected_log = self._pkl_projected_log_rows(
                student_aligned,
                projection,
                teacher_vocab_size=teacher_vocab_size,
            )
            teacher_topk_probs, teacher_topk_ids = teacher_aligned.detach().topk(
                min(teacher_top_k, teacher_vocab_size), dim=-1
            )
            row_losses = self._pkl_kl_from_teacher_topk(
                projected_log,
                teacher_topk_probs,
                teacher_topk_ids,
            )
            row_losses = row_losses * normalization_scale * distillation_weight
            if not bool(torch.isfinite(row_losses).all()):
                raise FloatingPointError("X-Token P-KL produced a non-finite fused loss")
            output = output.index_add(0, local_start_rows[selected], row_losses.float())
        return output.to(token_losses.dtype)
