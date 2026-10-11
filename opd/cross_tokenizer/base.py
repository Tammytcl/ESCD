"""Stable contracts shared by cross-tokenizer OPD algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import math
from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeAlias

import torch


JSONObject: TypeAlias = dict[str, Any]


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings, got {key!r}")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} must contain only JSON values, got {type(value).__name__}")


def normalize_json_object(value: Any, *, field_name: str) -> JSONObject:
    """Validate and deep-copy one JSON object with deterministic semantics."""

    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a JSON object")
    _validate_json_value(value, field_name)
    # The round-trip prevents callers from retaining mutable nested aliases and
    # proves that state passed from CPU tokenization to GPU scoring is serializable.
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True))


class CrossTokenizerAlignmentError(ValueError):
    """Raised when two token sequences cannot be aligned as the same text."""


@dataclass(frozen=True)
class AlignedSegment:
    """One minimal text unit represented in Teacher and Student token coordinates."""

    teacher_start: int
    teacher_end: int
    student_start: int
    student_end: int

    def __post_init__(self) -> None:
        if not (0 <= self.teacher_start < self.teacher_end):
            raise ValueError(f"Invalid Teacher segment: [{self.teacher_start}, {self.teacher_end})")
        if not (0 <= self.student_start < self.student_end):
            raise ValueError(f"Invalid Student segment: [{self.student_start}, {self.student_end})")

    @property
    def is_span(self) -> bool:
        return self.teacher_end - self.teacher_start > 1 or self.student_end - self.student_start > 1

    def to_tuple(self) -> tuple[int, int, int, int]:
        return self.teacher_start, self.teacher_end, self.student_start, self.student_end

    @classmethod
    def from_tuple(cls, values: tuple[int, int, int, int] | list[int]) -> "AlignedSegment":
        if len(values) != 4:
            raise ValueError(f"AlignedSegment requires four offsets, got {values!r}")
        return cls(*(int(value) for value in values))


@dataclass(frozen=True)
class AlignmentResult:
    """Serializable alignment result passed from CPU tokenization to GPU scoring."""

    segments: tuple[AlignedSegment, ...]
    teacher_length: int
    student_length: int
    skipped: bool = False
    reason: str | None = None
    algorithm_metadata: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.teacher_length < 0 or self.student_length < 0:
            raise ValueError("Alignment sequence lengths must be non-negative")
        previous_teacher_end = 0
        previous_student_end = 0
        for segment in self.segments:
            if segment.teacher_start != previous_teacher_end or segment.student_start != previous_student_end:
                raise ValueError("Alignment segments must be contiguous and ordered from zero")
            previous_teacher_end = segment.teacher_end
            previous_student_end = segment.student_end
        if self.skipped:
            if self.segments:
                raise ValueError("A skipped alignment cannot contain segments")
        elif previous_teacher_end != self.teacher_length or previous_student_end != self.student_length:
            raise ValueError("Alignment segments must consume both complete token sequences")
        object.__setattr__(
            self,
            "algorithm_metadata",
            normalize_json_object(self.algorithm_metadata, field_name="AlignmentResult.algorithm_metadata"),
        )

    @property
    def aligned_student_tokens(self) -> int:
        return sum(segment.student_end - segment.student_start for segment in self.segments)

    @property
    def coverage(self) -> float:
        if self.student_length == 0:
            return 1.0 if self.teacher_length == 0 and not self.skipped else 0.0
        return self.aligned_student_tokens / self.student_length

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [segment.to_tuple() for segment in self.segments],
            "teacher_length": self.teacher_length,
            "student_length": self.student_length,
            "skipped": self.skipped,
            "reason": self.reason,
            "algorithm_metadata": normalize_json_object(
                self.algorithm_metadata, field_name="AlignmentResult.algorithm_metadata"
            ),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AlignmentResult":
        return cls(
            segments=tuple(AlignedSegment.from_tuple(segment) for segment in value.get("segments", ())),
            teacher_length=int(value["teacher_length"]),
            student_length=int(value["student_length"]),
            skipped=bool(value.get("skipped", False)),
            reason=value.get("reason"),
            algorithm_metadata=(
                value["algorithm_metadata"] if value.get("algorithm_metadata") is not None else {}
            ),
        )

    @classmethod
    def skipped_result(
        cls,
        teacher_length: int,
        student_length: int,
        reason: str,
        *,
        algorithm_metadata: JSONObject | None = None,
    ) -> "AlignmentResult":
        return cls(
            (),
            teacher_length,
            student_length,
            skipped=True,
            reason=reason,
            algorithm_metadata=algorithm_metadata if algorithm_metadata is not None else {},
        )


@dataclass(frozen=True)
class SharedVocabulary:
    """Deterministic Teacher/Student token-ID pairs for equal surface tokens."""

    student_ids: tuple[int, ...]
    teacher_ids: tuple[int, ...]
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not (len(self.student_ids) == len(self.teacher_ids) == len(self.tokens)):
            raise ValueError("Shared vocabulary fields must have equal lengths")
        if len(set(self.student_ids)) != len(self.student_ids):
            raise ValueError("Shared vocabulary contains duplicate Student IDs")
        if len(set(self.teacher_ids)) != len(self.teacher_ids):
            raise ValueError("Shared vocabulary contains duplicate Teacher IDs")

    def __len__(self) -> int:
        return len(self.tokens)


TensorPayload: TypeAlias = dict[str, torch.Tensor]


CandidateTopKMode: TypeAlias = Literal["zero", "positive", "nonnegative"]
PublicTemperatureMode: TypeAlias = Literal["positive", "one"]
TeacherTransport: TypeAlias = Literal["payload", "causal_hidden"]


@dataclass(frozen=True)
class CrossTokenizerCapabilities:
    """Static algorithm contract consumed by launchers and model backends.

    Registration alone is not sufficient for a real plugin: the launcher must
    know whether a method needs the complete vocabulary, which Teacher
    transports preserve its objective, and whether its payload can be routed
    across Context Parallel ranks.  Keeping those facts on the algorithm
    prevents framework code from branching on names such as ``bpm``.
    """

    fidelity: str = "custom"
    reference: str = ""
    candidate_top_k_mode: CandidateTopKMode = "nonnegative"
    default_candidate_top_k: int = 128
    teacher_backends: tuple[str, ...] = ("megatron",)
    required_distribution_methods: tuple[str, ...] = ("selected_log_probs",)
    public_temperature_mode: PublicTemperatureMode = "positive"
    supports_context_parallel: bool = False
    context_parallel_backends: tuple[str, ...] = ()
    payload_position_field: str | None = None
    teacher_transports: tuple[tuple[str, TeacherTransport], ...] = ()
    requires_loss_token_normalization: bool = False
    requires_full_response_loss_mask: bool = False
    # Some plugins execute distributed row-collection operations inside their
    # loss function.  Re-running that function under Megatron's activation
    # checkpoint can issue a second set of collectives in a different order
    # from attention backward.  Such plugins must opt out at the launcher
    # boundary; Slime's native training path is otherwise untouched.
    requires_loss_recompute_disabled: bool = False
    requires_chat_template: bool = False
    requires_thinking: bool = False
    external_reward_path: str | None = None
    external_post_process_path: str | None = None

    def __post_init__(self) -> None:
        if self.candidate_top_k_mode not in {"zero", "positive", "nonnegative"}:
            raise ValueError(f"Unknown candidate_top_k mode: {self.candidate_top_k_mode!r}")
        if self.public_temperature_mode not in {"positive", "one"}:
            raise ValueError(f"Unknown public temperature mode: {self.public_temperature_mode!r}")
        if self.default_candidate_top_k < 0:
            raise ValueError("default_candidate_top_k must be non-negative")
        if not self.teacher_backends:
            raise ValueError("A cross-tokenizer algorithm must support at least one Teacher backend")
        transports = dict(self.teacher_transports)
        if len(transports) != len(self.teacher_transports):
            raise ValueError("Cross-tokenizer Teacher transport backends must be unique")
        unknown_backends = set(transports).difference(self.teacher_backends)
        if unknown_backends:
            raise ValueError(
                "Cross-tokenizer Teacher transports reference unsupported backends: "
                f"{sorted(unknown_backends)}"
            )
        if any(value not in {"payload", "causal_hidden"} for value in transports.values()):
            raise ValueError("Unknown cross-tokenizer Teacher transport")
        if any(value == "causal_hidden" for value in transports.values()):
            if not self.external_reward_path or not self.external_post_process_path:
                raise ValueError(
                    "causal_hidden plugins must declare their external reward and post-process hooks"
                )
        unknown_cp_backends = set(self.context_parallel_backends).difference(self.teacher_backends)
        if unknown_cp_backends:
            raise ValueError(
                "Cross-tokenizer Context Parallel backends are unsupported Teacher backends: "
                f"{sorted(unknown_cp_backends)}"
            )
        cp_backends = (
            set(self.teacher_backends)
            if self.supports_context_parallel
            else set(self.context_parallel_backends)
        )
        payload_cp_backends = {
            backend for backend in cp_backends if self.teacher_transport(backend) == "payload"
        }
        if payload_cp_backends and not self.payload_position_field:
            raise ValueError(
                "Context-parallel payload algorithms must declare the field containing global Student rows"
            )

    def teacher_transport(self, teacher_backend: str) -> TeacherTransport:
        """Return the backend-to-plugin transport without inspecting its name."""

        return dict(self.teacher_transports).get(teacher_backend, "payload")

    def supports_context_parallel_for(self, teacher_backend: str) -> bool:
        """Return CP support for one transport without branching on an algorithm name."""

        return self.supports_context_parallel or teacher_backend in self.context_parallel_backends

    def validate_runtime_config(
        self,
        *,
        candidate_top_k: int,
        public_temperature: float,
        teacher_backend: str,
    ) -> None:
        """Reject a launch that would silently change the declared objective."""

        if candidate_top_k < 0:
            raise ValueError("cross_tokenizer.candidate_top_k must be non-negative")
        if self.candidate_top_k_mode == "zero" and candidate_top_k != 0:
            raise ValueError(
                "this algorithm is full-vocabulary and requires cross_tokenizer.candidate_top_k=0"
            )
        if self.candidate_top_k_mode == "positive" and candidate_top_k <= 0:
            raise ValueError("this algorithm requires a positive cross_tokenizer.candidate_top_k")
        if not math.isfinite(public_temperature) or public_temperature <= 0:
            raise ValueError("cross_tokenizer.temperature must be a positive finite number")
        if self.public_temperature_mode == "one" and public_temperature != 1.0:
            raise ValueError(
                "this algorithm owns temperature in algorithm_kwargs; cross_tokenizer.temperature must be 1.0"
            )
        if teacher_backend not in self.teacher_backends:
            supported = ", ".join(self.teacher_backends)
            raise ValueError(
                f"this algorithm does not support Teacher backend {teacher_backend!r}; supported: {supported}"
            )


def localize_payload_by_student_positions(
    teacher_payload: TensorPayload,
    *,
    position_field: str,
    response_positions: torch.Tensor,
    response_length: int,
) -> TensorPayload:
    """Route a row-oriented opaque payload to one Context Parallel owner.

    Any tensor whose leading dimension equals the number of supervised rows is
    filtered consistently. Scalar metadata is retained. The algorithm remains
    responsible for declaring that its rows are independent under CP.
    """

    if position_field not in teacher_payload:
        raise ValueError(f"Teacher payload is missing CP position field {position_field!r}")
    positions = teacher_payload[position_field].to(
        device=response_positions.device, dtype=torch.long
    ).flatten()
    if positions.numel() and bool(((positions < 0) | (positions >= int(response_length))).any()):
        raise ValueError("Cross-tokenizer payload contains an invalid global Student row")
    if response_positions.ndim != 1:
        raise ValueError("CP response positions must be one-dimensional")
    if response_positions.numel() > 1 and bool(
        (response_positions[1:] < response_positions[:-1]).any()
    ):
        raise ValueError("CP response positions must be sorted")

    local_rows = torch.searchsorted(response_positions, positions).clamp(
        max=response_positions.numel()
    )
    if response_positions.numel():
        owned = (local_rows < response_positions.numel()) & (
            response_positions.index_select(
                0, local_rows.clamp(max=response_positions.numel() - 1)
            )
            == positions
        )
    else:
        owned = torch.zeros_like(positions, dtype=torch.bool)
    owned_indices = torch.nonzero(owned, as_tuple=False).flatten()

    localized: TensorPayload = {}
    row_count = positions.numel()
    for key, value in teacher_payload.items():
        moved = value.to(device=response_positions.device)
        if moved.ndim > 0 and moved.size(0) == row_count:
            moved = moved.index_select(0, owned_indices)
        localized[key] = moved
    localized[position_field] = local_rows.index_select(0, owned_indices)
    return localized


@dataclass(frozen=True)
class CandidateScores:
    """Candidate IDs and scores returned by a vocabulary-distribution backend."""

    output_ids: torch.Tensor
    log_probs: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class VocabularyMapping:
    """Teacher token IDs paired with their Student-side public-space IDs."""

    teacher_ids: torch.Tensor
    student_ids: torch.Tensor


class VocabularyDistribution(Protocol):
    """Model-backend facade used by tokenizer algorithms.

    The implementation owns vocabulary parallelism and collectives. Algorithms
    only request rows, selected log-probabilities, or candidate sets and never
    depend on Megatron internals.
    """

    @property
    def length(self) -> int:
        """Number of next-token prediction rows in this distribution."""

    def rows(self, row_indices: torch.Tensor) -> "VocabularyDistribution":
        """Return a view containing the selected prediction rows."""

    def selected_log_probs(
        self,
        token_ids: torch.Tensor,
        *,
        vocab_size: int | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return log-probabilities for supplied IDs over the effective vocabulary."""

    def selected_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return unnormalized model logits for supplied global token IDs.

        This is intentionally separate from ``selected_log_probs``.  The
        released SimCT implementation averages raw logits for its virtual span
        coordinate; replacing them with normalized log-probabilities changes
        the relative span score when the span covers more than one row.
        """

    def full_probabilities(
        self,
        *,
        vocab_size: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return dense probabilities over the unpadded effective vocabulary.

        This deliberately exists alongside ``topk`` rather than being
        implemented through it.  Paper-faithful ULD and GOLD compare complete
        distributions; truncating them changes the published objectives.
        Backends own any vocabulary-parallel gather and its autograd contract.
        """

    def projected_log_probs(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        vocab_size: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return explicit-cell log-probabilities plus one exact complement.

        The final coordinate contains all effective-vocabulary outcomes absent
        from the explicit support. Backends must avoid the unstable
        ``log(1 - sum(exp(log_p)))`` construction.
        """

    def topk(self, k: int) -> CandidateScores:
        """Return global-vocabulary Top-K candidates."""

    def mapped_topk(
        self,
        source_ids: torch.Tensor,
        output_ids: torch.Tensor,
        *,
        k: int,
        vocab_size: int | None = None,
        temperature: float = 1.0,
    ) -> CandidateScores:
        """Select among source IDs and express the result with output IDs."""


@dataclass(frozen=True)
class TeacherScoringContext:
    """Complete model-agnostic input for one Teacher response."""

    distribution: VocabularyDistribution
    response_token_ids: torch.Tensor
    alignment: AlignmentResult
    vocabulary_mapping: VocabularyMapping
    candidate_top_k: int
    student_loss_mask: torch.Tensor
    temperature: float = 1.0
    # Native Megatron Teacher scoring needs the realized Student terminal ID
    # when the optional source-compatible ``skip_student_eos`` switch is on.
    # It is optional to keep third-party/plugin callers source-compatible.
    student_response_token_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class StudentScoringContext:
    """Complete model-agnostic input for one Student response."""

    distribution: VocabularyDistribution
    response_token_ids: torch.Tensor
    temperature: float = 1.0
    # Optional reducer metadata for objectives whose components use different
    # batch normalizers.  ULD uses it to combine token-mean CE with a
    # sample-mean rank loss under Slime's token-normalized outer reducer.
    step_active_token_count: int | None = None
    step_sample_count: int | None = None


@dataclass(frozen=True)
class FusedScoringContext:
    """Backend-neutral input for a causal-hidden Teacher transport."""

    student: StudentScoringContext
    teacher_distribution: VocabularyDistribution
    teacher_response_token_ids: torch.Tensor
    teacher_observed_log_probs: torch.Tensor | None
    alignment: AlignmentResult
    local_response_positions: torch.Tensor
    vocabulary_mapping: VocabularyMapping | None = None
    gather_student_rows: Callable[[torch.Tensor], torch.Tensor] | None = None
    candidate_top_k: int = 0
    # Full, unsliced Student response IDs. Under Context Parallelism
    # ``student.response_token_ids`` contains only locally owned rows, which is
    # insufficient for terminal-EOS decisions made by a plugin.
    student_response_token_ids: torch.Tensor | None = None


class CrossTokenizerAlgorithm(ABC):
    """Pluggable alignment and public-supervision-space algorithm."""

    name: str
    capabilities = CrossTokenizerCapabilities()

    def __init__(
        self,
        student_tokenizer: Any | None = None,
        teacher_tokenizer: Any | None = None,
        *,
        algorithm_kwargs: JSONObject | None = None,
        runtime_state: JSONObject | None = None,
    ) -> None:
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.algorithm_kwargs = normalize_json_object(
            algorithm_kwargs, field_name="cross-tokenizer algorithm_kwargs"
        )
        self.runtime_state = normalize_json_object(runtime_state, field_name="cross-tokenizer runtime_state")

    def export_runtime_state(self) -> JSONObject:
        """Return tokenizer-derived JSON state needed by both GPU scoring phases.

        Tokenizer objects are deliberately CPU-only. Algorithms whose scoring
        hooks need tokenizer-derived constants must export those constants here
        and consume ``self.runtime_state`` when reconstructed for GPU scoring.
        """

        return normalize_json_object(self.runtime_state, field_name="cross-tokenizer runtime_state")

    def configure_model_contract(
        self,
        *,
        student_vocab_size: int,
        teacher_vocab_size: int,
        student_stop_ids: tuple[int, ...],
        teacher_stop_ids: tuple[int, ...],
    ) -> None:
        """Attach model-declared vocabulary and stop metadata before setup."""

        del student_vocab_size, teacher_vocab_size, student_stop_ids, teacher_stop_ids

    def compute_fused_token_losses(self, context: FusedScoringContext) -> torch.Tensor:
        """Score one response for a ``causal_hidden`` Teacher transport."""

        del context
        raise NotImplementedError(
            f"Cross-tokenizer algorithm {self.name!r} does not implement causal-hidden scoring"
        )

    def build_training_diagnostic_masks(
        self,
        alignments: list[AlignmentResult],
        skipped_masks: list[torch.Tensor],
        *,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Return optional full-response masks for framework metric reduction."""

        del alignments, skipped_masks, device
        return {}

    def training_diagnostic_names(self) -> tuple[str, ...]:
        """Return per-response tensors emitted by fused training scoring.

        The Slime loss wrapper uses this optional interface to collect
        algorithm-specific diagnostics without changing the public token-loss
        return type.  Implementations should make the tensors available via
        :meth:`pop_training_diagnostics` immediately after one fused scoring
        call.
        """

        return ()

    def pop_training_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return and clear diagnostics from the most recent fused scoring call."""

        return {}

    @abstractmethod
    def build_shared_vocabulary(self) -> SharedVocabulary | None:
        """Return a one-to-one candidate mapping, or ``None`` for span-only algorithms."""

    @abstractmethod
    def align(self, teacher_token_ids: list[int], student_token_ids: list[int]) -> AlignmentResult:
        """Align two tokenizations; data failures must raise CrossTokenizerAlignmentError."""

    @abstractmethod
    def build_teacher_payload(self, context: TeacherScoringContext) -> TensorPayload:
        """Build an algorithm-private, detached payload for one response."""

    @abstractmethod
    def compute_student_token_losses(
        self,
        context: StudentScoringContext,
        teacher_payload: TensorPayload,
    ) -> torch.Tensor:
        """Return graph-connected losses in Student response coordinates."""
