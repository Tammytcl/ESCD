"""BPM cross-tokenizer plugin for online and offline OPD experiments.

The routing core follows Byte-Prefix Marginalization (BPM): Teacher tokens
are routed to the longest Student-token byte prefix and all unmatched/tail
probability is retained in one complement cell.  ``candidate_top_k=0`` uses
the complete effective Teacher vocabulary and is the paper-aligned setting;
positive values are an explicitly labelled source-Top-K approximation.
"""

from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any, Mapping

import torch

from .base import (
    AlignedSegment,
    AlignmentResult,
    CrossTokenizerAlgorithm,
    CrossTokenizerCapabilities,
    CrossTokenizerAlignmentError,
    FusedScoringContext,
    SharedVocabulary,
    StudentScoringContext,
    TeacherScoringContext,
    TensorPayload,
    normalize_json_object,
)
from .bpm_core import (
    BPMRoutingArtifact,
    INCOMPATIBLE_ROUTE,
    RESIDUAL_ROUTE,
    _student_trie,
    build_routing_artifact,
    routes_for_bpm_row,
    spanning_chain_spec,
    spanning_correction_updates,
)
from .registry import register_cross_tokenizer
from .bpm_dense import bpm_projected_divergence_from_log_probs
from ._paper_utils import tokenizer_vocab_size


class BPMZeroCompatibleTeacherMassError(ValueError):
    """Raised when source-TopK cannot define an interior conditional target."""


def _offsets(pieces: list[bytes]) -> list[int]:
    values: list[int] = []
    total = 0
    for piece in pieces:
        if not piece:
            raise CrossTokenizerAlignmentError("BPM response contains an empty token byte string")
        total += len(piece)
        values.append(total)
    return values


def _common_boundary_segments(
    teacher_offsets: list[int], student_offsets: list[int]
) -> list[AlignedSegment]:
    teacher_index = student_index = 0
    teacher_start = student_start = 0
    segments: list[AlignedSegment] = []
    while teacher_index < len(teacher_offsets) and student_index < len(student_offsets):
        teacher_end = teacher_offsets[teacher_index]
        student_end = student_offsets[student_index]
        if teacher_end == student_end:
            teacher_index += 1
            student_index += 1
            segments.append(AlignedSegment(teacher_start, teacher_index, student_start, student_index))
            teacher_start = teacher_index
            student_start = student_index
        elif teacher_end < student_end:
            teacher_index += 1
        else:
            student_index += 1
    if teacher_index != len(teacher_offsets) or student_index != len(student_offsets):
        raise CrossTokenizerAlignmentError("BPM byte alignment left response tokens unconsumed")
    return segments


@register_cross_tokenizer("bpm")
class BPMCrossTokenizer(CrossTokenizerAlgorithm):
    """Mass-preserving byte-prefix projection with an explicit complement."""

    capabilities = CrossTokenizerCapabilities(
        fidelity="legacy-mixed",
        reference="BPM arXiv:2607.22334 plus source-TopK engineering path",
        candidate_top_k_mode="nonnegative",
        default_candidate_top_k=128,
        teacher_backends=("megatron", "sglang"),
        required_distribution_methods=(
            "topk",
            "selected_log_probs",
            "projected_log_probs",
        ),
        public_temperature_mode="one",
        supports_context_parallel=True,
        payload_position_field="student_positions",
    )

    def _default_spanning_policy(self) -> str:
        return "mask"

    def _validate_scoring_context(self, context: TeacherScoringContext, spanning_policy: str) -> None:
        del context, spanning_policy

    def _artifact(self) -> BPMRoutingArtifact:
        cached = getattr(self, "_bpm_artifact", None)
        if isinstance(cached, BPMRoutingArtifact):
            return cached
        if self.runtime_state:
            payload = self.runtime_state.get("routing_artifact")
            if not isinstance(payload, dict):
                raise ValueError("BPM runtime_state is missing routing_artifact")
            artifact = BPMRoutingArtifact.from_dict(payload)
            self._bpm_artifact = artifact
            return artifact
        if self.student_tokenizer is None or self.teacher_tokenizer is None:
            raise RuntimeError("BPM requires tokenizer-derived runtime state during GPU scoring")
        artifact = build_routing_artifact(
            self.student_tokenizer,
            self.teacher_tokenizer,
            student_vocab_size=self.algorithm_kwargs.get("student_vocab_size"),
            teacher_vocab_size=self.algorithm_kwargs.get("teacher_vocab_size"),
            additional_student_stop_ids=self.algorithm_kwargs.get(
                "additional_student_stop_ids", ()
            ),
            additional_teacher_stop_ids=self.algorithm_kwargs.get(
                "additional_teacher_stop_ids", ()
            ),
        )
        self.runtime_state = {"routing_artifact": artifact.to_dict()}
        self._bpm_artifact = artifact
        return artifact

    def _student_vocab_size(self) -> int:
        """Return the Student softmax width needed by the GPU projector.

        The external SGLang Teacher path deliberately reconstructs the BPM
        adapter without tokenizers: the Teacher has already built the routing
        target and sends only Student token ids/probabilities.  The projector
        still needs the Student vocabulary width, but it does *not* need the
        CPU byte trie.  Keeping this scalar in JSON state/kwargs avoids
        rebuilding a large tokenizer artifact on every Megatron rank.
        """

        cached = getattr(self, "_bpm_artifact", None)
        if isinstance(cached, BPMRoutingArtifact):
            return len(cached.student_bytes)
        for source in (self.runtime_state, self.algorithm_kwargs):
            value = source.get("student_vocab_size")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return int(value)
        # Preserve the old, actionable failure for an in-process adapter that
        # was reconstructed without either tokenizers or exported state.
        return len(self._artifact().student_bytes)

    def _settings(self) -> tuple[float, float, bool, str]:
        beta = float(self.algorithm_kwargs.get("beta", 0.5))
        skew = float(self.algorithm_kwargs.get("rkl_skew", 0.1))
        mask_whitespace = bool(self.algorithm_kwargs.get("mask_whitespace", True))
        spanning_policy = str(
            self.algorithm_kwargs.get("spanning_policy", self._default_spanning_policy())
        ).strip().lower()
        if not 0.0 <= beta <= 1.0 or not math.isfinite(beta):
            raise ValueError("BPM beta must be finite and in [0, 1]")
        if not 0.0 < skew < 1.0 or not math.isfinite(skew):
            raise ValueError("BPM rkl_skew must be finite and in (0, 1)")
        if spanning_policy not in {"mask", "head", "exact"}:
            raise ValueError("BPM spanning_policy must be 'mask', 'head', or 'exact'")
        return beta, skew, mask_whitespace, spanning_policy

    def build_shared_vocabulary(self) -> SharedVocabulary | None:
        # BPM is not a one-to-one shared-vocabulary method.  This call is also
        # the CPU preparation hook used to export the routing table.
        self._artifact()
        return None

    def configure_model_contract(
        self,
        *,
        student_vocab_size: int,
        teacher_vocab_size: int,
        student_stop_ids: tuple[int, ...],
        teacher_stop_ids: tuple[int, ...],
    ) -> None:
        configured = dict(self.algorithm_kwargs)
        # HF configs may report TP-only output-head padding rows (Qwen3.6:
        # 248320 rows versus 248077 tokenizer IDs).  BPM normalization and
        # its complement cell must use the effective tokenizer vocabulary.
        effective_student_vocab_size = (
            tokenizer_vocab_size(self.student_tokenizer)
            if self.student_tokenizer is not None
            else int(student_vocab_size)
        )
        effective_teacher_vocab_size = (
            tokenizer_vocab_size(self.teacher_tokenizer)
            if self.teacher_tokenizer is not None
            else int(teacher_vocab_size)
        )
        configured["student_vocab_size"] = effective_student_vocab_size
        configured["teacher_vocab_size"] = effective_teacher_vocab_size
        configured.setdefault("additional_student_stop_ids", list(student_stop_ids))
        configured.setdefault("additional_teacher_stop_ids", list(teacher_stop_ids))
        self.algorithm_kwargs = normalize_json_object(
            configured,
            field_name="BPM algorithm kwargs with model vocabulary and stop metadata",
        )

    def build_training_diagnostic_masks(
        self,
        alignments: list[AlignmentResult],
        skipped_masks: list[torch.Tensor],
        *,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Expose paper row coverage without teaching the framework BPM fields."""

        _, _, mask_whitespace, _ = self._settings()
        names = (
            "cross_tokenizer_loss_anchor",
            "bpm_whitespace_masked",
            "bpm_excluded",
            "bpm_aligned",
            "bpm_interior",
            "bpm_spanning",
            "bpm_stop",
        )
        per_name: dict[str, list[torch.Tensor]] = {name: [] for name in names}
        for alignment, skipped_mask in zip(alignments, skipped_masks, strict=True):
            skipped_mask = skipped_mask.to(device=device, dtype=torch.bool).flatten()
            if alignment.student_length != skipped_mask.numel():
                raise ValueError("BPM diagnostics received an alignment/mask length mismatch")
            masks = {
                name: torch.zeros(skipped_mask.numel(), device=device, dtype=torch.float32)
                for name in names
            }
            if not bool(skipped_mask.all()):
                for raw_position in alignment.algorithm_metadata.get(
                    "bpm_excluded_positions", ()
                ):
                    position = int(raw_position)
                    if not 0 <= position < skipped_mask.numel():
                        raise ValueError("BPM diagnostics contain an invalid excluded position")
                    masks["bpm_excluded"][position] = 1.0
                for row in alignment.algorithm_metadata.get("bpm_rows", ()):
                    position = int(row["student_position"])
                    if not 0 <= position < skipped_mask.numel():
                        raise ValueError("BPM diagnostics contain an invalid row position")
                    kind = str(row["kind"])
                    metric_name = f"bpm_{kind}"
                    if metric_name not in masks:
                        raise ValueError(f"BPM diagnostics contain an invalid row kind: {kind!r}")
                    masks[metric_name][position] = 1.0
                    if mask_whitespace and bool(row.get("whitespace", False)):
                        masks["bpm_whitespace_masked"][position] = 1.0
                    else:
                        masks["cross_tokenizer_loss_anchor"][position] = 1.0
            for name in names:
                per_name[name].append(masks[name])
        return {name: torch.cat(values, dim=0) for name, values in per_name.items()}

    def export_runtime_state(self) -> dict[str, Any]:
        self._artifact()
        return super().export_runtime_state()

    def align(self, teacher_token_ids: list[int], student_token_ids: list[int]) -> AlignmentResult:
        artifact = self._artifact()
        teacher_ids = [int(value) for value in teacher_token_ids]
        student_ids = [int(value) for value in student_token_ids]
        if not teacher_ids and not student_ids:
            return AlignmentResult((), 0, 0, algorithm_metadata={"bpm_rows": []})
        if not teacher_ids or not student_ids:
            raise CrossTokenizerAlignmentError("One BPM response tokenization is empty")

        teacher_stops = set(artifact.teacher_stop_ids)
        student_stops = set(artifact.student_stop_ids)
        teacher_has_stop = teacher_ids[-1] in teacher_stops
        student_has_stop = student_ids[-1] in student_stops
        if teacher_has_stop != student_has_stop:
            raise CrossTokenizerAlignmentError("Teacher and Student BPM responses disagree on terminal stop")
        teacher_content = teacher_ids[:-1] if teacher_has_stop else teacher_ids
        student_content = student_ids[:-1] if student_has_stop else student_ids
        # A stop token is a structural boundary, not ordinary answer bytes.
        # If it occurs before the terminal position, the dense causal route
        # builder intentionally excludes every route crossing that boundary;
        # allowing the sample through therefore reaches the loss with zero
        # compatible Teacher mass (the crash seen in rollout 67).  Fail closed
        # here so the configured alignment_error_policy=skip can mark the
        # entire sample as fully skipped and replenish just that group.
        internal_teacher_stops = [token_id for token_id in teacher_content if token_id in teacher_stops]
        if internal_teacher_stops:
            raise CrossTokenizerAlignmentError(
                "BPM response contains internal Teacher stop token(s): "
                f"{internal_teacher_stops[:8]}"
            )
        try:
            teacher_pieces = [artifact.teacher_bytes[token_id] for token_id in teacher_content]
            student_pieces = [artifact.student_bytes[token_id] for token_id in student_content]
        except IndexError as error:
            raise CrossTokenizerAlignmentError("BPM response token ID is outside its tokenizer vocabulary") from error
        if any(piece is None for piece in teacher_pieces + student_pieces):
            raise CrossTokenizerAlignmentError("BPM response contains a token without lossless bytes")
        teacher_bytes = [piece for piece in teacher_pieces if piece is not None]
        student_bytes = [piece for piece in student_pieces if piece is not None]
        if b"".join(teacher_bytes) != b"".join(student_bytes):
            raise CrossTokenizerAlignmentError("BPM Teacher and Student response bytes differ")

        teacher_ends = _offsets(teacher_bytes) if teacher_bytes else []
        student_ends = _offsets(student_bytes) if student_bytes else []
        segments = _common_boundary_segments(teacher_ends, student_ends) if teacher_ends else []
        sync_offsets = {0, *(set(teacher_ends) & set(student_ends))}
        rows: list[dict[str, Any]] = []
        excluded_positions: list[int] = []
        teacher_index = 0
        student_start_byte = 0
        for student_position, (student_id, piece) in enumerate(
            zip(student_content, student_bytes, strict=True)
        ):
            while teacher_index < len(teacher_ends) and student_start_byte >= teacher_ends[teacher_index]:
                teacher_index += 1
            if teacher_index >= len(teacher_ends):
                raise CrossTokenizerAlignmentError("BPM could not locate the causal Teacher token")
            teacher_piece_start = teacher_ends[teacher_index] - len(teacher_bytes[teacher_index])
            prefix_size = student_start_byte - teacher_piece_start
            prefix = teacher_bytes[teacher_index][:prefix_size]
            student_end_byte = student_start_byte + len(piece)
            kind = (
                "spanning"
                if student_end_byte > teacher_ends[teacher_index]
                else "interior"
                if prefix
                else "aligned"
            )
            # In an M:N chunk, a token that starts inside one Teacher token,
            # crosses a Teacher boundary, and still ends before the next sync
            # point has no exact realized-path target in the paper. It remains
            # in Z but contributes no numerator loss.
            if (
                kind == "spanning"
                and student_start_byte not in sync_offsets
                and student_end_byte not in sync_offsets
            ):
                excluded_positions.append(student_position)
                student_start_byte = student_end_byte
                continue
            row: dict[str, Any] = {
                "student_position": student_position,
                "student_token_id": student_id,
                "teacher_position": teacher_index,
                "teacher_target_position": teacher_index,
                "prefix_hex": prefix.hex(),
                "kind": kind,
                "whitespace": bool(piece) and all(byte in (9, 32) for byte in piece),
            }
            if student_has_stop:
                row["student_stop_token_id"] = student_ids[-1]
            rows.append(row)
            student_start_byte = student_end_byte

        if teacher_has_stop:
            segments.append(
                AlignedSegment(
                    len(teacher_content), len(teacher_ids), len(student_content), len(student_ids)
                )
            )
            rows.append(
                {
                    "student_position": len(student_content),
                    "student_token_id": student_ids[-1],
                    "teacher_position": len(teacher_content),
                    "teacher_target_position": len(teacher_content),
                    "prefix_hex": "",
                    "kind": "stop",
                    "whitespace": False,
                }
            )
        return AlignmentResult(
            tuple(segments),
            len(teacher_ids),
            len(student_ids),
            algorithm_metadata={
                "bpm_rows": rows,
                "bpm_excluded_positions": excluded_positions,
            },
        )

    @staticmethod
    def _candidate_route(
        artifact: BPMRoutingArtifact,
        row: Mapping[str, Any],
        teacher_id: int,
    ) -> int:
        if not 0 <= teacher_id < len(artifact.teacher_bytes):
            return RESIDUAL_ROUTE
        kind = str(row["kind"])
        if kind == "stop":
            return int(row["student_token_id"]) if teacher_id in set(artifact.teacher_stop_ids) else RESIDUAL_ROUTE
        prefix = bytes.fromhex(str(row.get("prefix_hex", "")))
        if prefix:
            value = artifact.teacher_bytes[teacher_id]
            if teacher_id in set(artifact.teacher_stop_ids) or value is None or not value.startswith(prefix):
                return INCOMPATIBLE_ROUTE
            return _student_trie(artifact).longest_prefix(value[len(prefix) :])
        if teacher_id in set(artifact.teacher_stop_ids) and "student_stop_token_id" in row:
            return int(row["student_stop_token_id"])
        return int(artifact.teacher_to_student[teacher_id])

    def build_teacher_payload(self, context: TeacherScoringContext) -> TensorPayload:
        artifact = self._artifact()
        _, _, mask_whitespace, spanning_policy = self._settings()
        self._validate_scoring_context(context, spanning_policy)
        if context.distribution.length != context.alignment.teacher_length:
            raise ValueError("Teacher distribution length does not match BPM alignment")
        if not math.isclose(float(context.temperature), 1.0):
            raise ValueError("BPM currently requires cross_tokenizer.temperature=1.0")
        mask = context.student_loss_mask.detach().to(dtype=torch.bool).flatten().cpu()
        if mask.numel() != context.alignment.student_length:
            raise ValueError("Student loss mask does not match BPM alignment")
        raw_rows = context.alignment.algorithm_metadata.get("bpm_rows", [])
        active_rows = [
            row
            for row in raw_rows
            if bool(mask[int(row["student_position"])])
            and not (mask_whitespace and bool(row.get("whitespace", False)))
            and not (str(row.get("kind")) == "spanning" and spanning_policy == "mask")
        ]
        device = context.response_token_ids.device
        if not active_rows:
            empty_long = torch.empty((0,), dtype=torch.long, device=device)
            return {
                "target_ids": torch.empty((0, 0), dtype=torch.long, device=device),
                "target_probs": torch.empty((0, 0), dtype=torch.float32, device=device),
                "target_mask": torch.empty((0, 0), dtype=torch.bool, device=device),
                "target_complement": torch.empty((0,), dtype=torch.float32, device=device),
                "student_positions": empty_long,
                "row_kinds": empty_long,
                "force_fkl": torch.empty((0,), dtype=torch.bool, device=device),
            }

        if context.candidate_top_k == 0 and spanning_policy == "exact":
            return self._build_exact_teacher_payload(context, active_rows)

        teacher_positions = torch.tensor(
            [int(row["teacher_position"]) for row in active_rows], dtype=torch.long, device=device
        )
        requested_k = int(context.candidate_top_k)
        source_k = (
            len(artifact.teacher_to_student)
            if requested_k == 0
            else min(requested_k, len(artifact.teacher_to_student))
        )
        if source_k <= 0:
            raise ValueError("BPM candidate_top_k must be non-negative")
        source = context.distribution.rows(teacher_positions).topk(source_k)
        source_ids = source.output_ids.detach().cpu()
        source_probs = source.log_probs.detach().float().exp().cpu()
        source_mask = source.mask.detach().to(dtype=torch.bool).cpu()

        row_ids: list[list[int]] = []
        row_probs: list[list[float]] = []
        complements: list[float] = []
        for row_index, row in enumerate(active_rows):
            selected_mass = float(source_probs[row_index][source_mask[row_index]].sum().item())
            tail_mass = max(0.0, 1.0 - selected_mass)
            mapped: dict[int, float] = {}
            residual = tail_mass
            compatible_mass = tail_mass
            has_prefix = bool(row.get("prefix_hex"))
            for teacher_id, probability, valid in zip(
                source_ids[row_index].tolist(),
                source_probs[row_index].tolist(),
                source_mask[row_index].tolist(),
                strict=True,
            ):
                if not valid:
                    continue
                route = self._candidate_route(artifact, row, int(teacher_id))
                if route == INCOMPATIBLE_ROUTE:
                    continue
                compatible_mass += float(probability) if has_prefix else 0.0
                if route == RESIDUAL_ROUTE:
                    residual += float(probability)
                else:
                    mapped[route] = mapped.get(route, 0.0) + float(probability)
            if has_prefix:
                # With full vocabulary this is exact conditional
                # marginalization.  Under source Top-K, unseen tail is
                # conservatively treated as compatible complement mass.
                if compatible_mass <= 0:
                    raise BPMZeroCompatibleTeacherMassError(
                        "BPM interior row has zero compatible Teacher mass"
                    )
                mapped = {token_id: value / compatible_mass for token_id, value in mapped.items()}
                residual /= compatible_mass
            total = sum(mapped.values()) + residual
            if not math.isfinite(total) or total <= 0:
                raise ValueError("BPM projected target has invalid probability mass")
            mapped = {token_id: value / total for token_id, value in mapped.items() if value > 0}
            residual = max(0.0, residual / total)
            ordered = sorted(mapped.items(), key=lambda item: (-item[1], item[0]))
            if not ordered:
                # Preserve a two-cell differentiable distribution even when
                # every Teacher outcome landed in the complement.
                ordered = [(int(row["student_token_id"]), 0.0)]
            row_ids.append([token_id for token_id, _ in ordered])
            row_probs.append([value for _, value in ordered])
            complements.append(residual)

        width = max(len(values) for values in row_ids)
        ids_tensor = torch.zeros((len(active_rows), width), dtype=torch.long, device=device)
        probs_tensor = torch.zeros((len(active_rows), width), dtype=torch.float32, device=device)
        target_mask = torch.zeros((len(active_rows), width), dtype=torch.bool, device=device)
        for index, (ids, probabilities) in enumerate(zip(row_ids, row_probs, strict=True)):
            ids_tensor[index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            probs_tensor[index, : len(probabilities)] = torch.tensor(
                probabilities, dtype=torch.float32, device=device
            )
            target_mask[index, : len(ids)] = True
        kind_codes = {"aligned": 0, "interior": 1, "spanning": 2, "stop": 3}
        return {
            "target_ids": ids_tensor.detach(),
            "target_probs": probs_tensor.detach(),
            "target_mask": target_mask.detach(),
            "target_complement": torch.tensor(complements, dtype=torch.float32, device=device).detach(),
            "student_positions": torch.tensor(
                [int(row["student_position"]) for row in active_rows], dtype=torch.long, device=device
            ),
            "row_kinds": torch.tensor(
                [kind_codes[str(row["kind"])] for row in active_rows], dtype=torch.long, device=device
            ),
            "force_fkl": torch.tensor(
                [str(row["kind"]) == "stop" for row in active_rows],
                dtype=torch.bool,
                device=device,
            ),
        }

    @staticmethod
    def _route_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
        kind = str(row.get("kind", ""))
        if kind == "stop":
            return ("stop", int(row["student_token_id"]))
        return (
            "conditional" if row.get("prefix_hex") else "aligned",
            str(row.get("prefix_hex", "")),
            int(row.get("student_stop_token_id", -1)),
        )

    def _build_exact_teacher_payload(
        self,
        context: TeacherScoringContext,
        active_rows: list[Mapping[str, Any]],
    ) -> TensorPayload:
        """Evaluate BPM's complete Teacher-vocabulary target, including Eq. 6.

        The implementation intentionally requests exact probabilities through
        the model-agnostic distribution facade.  It never converts full-vocab
        BPM into source Top-K.  Teacher values are detached and projected one
        row at a time to bound temporary memory; production backends may batch
        equal route signatures without changing this contract.
        """

        artifact = self._artifact()
        device = context.response_token_ids.device
        teacher_vocab_size = len(artifact.teacher_bytes)
        teacher_ids = [int(value) for value in context.response_token_ids.detach().cpu().tolist()]
        route_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        # Exact BPM needs several neighboring Teacher rows for Equation 6, but
        # retaining every [V_T] row would add O(response_length * V_T) memory
        # on top of the projected payload.  A tiny LRU preserves exact values
        # while bounding this temporary cache independently of sequence length.
        probability_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        probability_cache_rows = 4

        def full_row(position: int) -> torch.Tensor:
            position = int(position)
            cached = probability_cache.get(position)
            if cached is None:
                row_index = torch.tensor([position], dtype=torch.long, device=device)
                cached = context.distribution.rows(row_index).full_probabilities(
                    vocab_size=teacher_vocab_size,
                    temperature=1.0,
                )[0].detach().float()
                probability_cache[position] = cached
                if len(probability_cache) > probability_cache_rows:
                    probability_cache.popitem(last=False)
            else:
                probability_cache.move_to_end(position)
            return cached

        def route_tensor(row: Mapping[str, Any]) -> torch.Tensor:
            signature = self._route_signature(row)
            cached = route_cache.get(signature)
            if cached is None:
                cached = torch.tensor(
                    routes_for_bpm_row(artifact, row),
                    dtype=torch.long,
                    device=device,
                )
                route_cache[signature] = cached
            return cached

        def project(row: Mapping[str, Any]) -> tuple[dict[int, float], float, torch.Tensor]:
            position = int(row.get("teacher_target_position", row["teacher_position"]))
            probabilities = full_row(position)
            routes = route_tensor(row)
            compatible = routes != INCOMPATIBLE_ROUTE
            denominator = probabilities[compatible].sum() if bool(row.get("prefix_hex")) else probabilities.sum()
            denominator_value = float(denominator.item())
            if not math.isfinite(denominator_value) or denominator_value <= 0:
                raise BPMZeroCompatibleTeacherMassError(
                    "BPM exact conditional row has zero compatible Teacher mass"
                )
            normalized = probabilities / denominator
            explicit = routes >= 0
            if bool(explicit.any()):
                ids, inverse = torch.unique(routes[explicit], sorted=True, return_inverse=True)
                values = normalized.new_zeros((ids.numel(),))
                values.scatter_add_(0, inverse, normalized[explicit])
                mapped = {
                    int(token_id): float(probability)
                    for token_id, probability in zip(ids.tolist(), values.tolist(), strict=True)
                    if probability > 0
                }
            else:
                mapped = {}
            residual = float(normalized[routes == RESIDUAL_ROUTE].sum().item())
            return mapped, residual, routes

        row_ids: list[list[int]] = []
        row_probs: list[list[float]] = []
        complements: list[float] = []
        force_fkl: list[bool] = []
        for row in active_rows:
            mapped, residual, routes = project(row)
            if str(row.get("kind")) == "spanning":
                prepared = dict(row)
                prepared["teacher_target_position"] = int(
                    row.get("teacher_target_position", row["teacher_position"])
                )
                spec = spanning_chain_spec(artifact, prepared, teacher_ids)

                def token_probability(position: int, token_id: int) -> float:
                    return float(full_row(position)[int(token_id)].item())

                def prefix_mass(position: int, prefix: bytes) -> float:
                    conditional = {"kind": "interior", "prefix_hex": prefix.hex()}
                    compatible_routes = route_tensor(conditional)
                    return float(
                        full_row(position)[compatible_routes != INCOMPATIBLE_ROUTE].sum().item()
                    )

                target_position = int(prepared["teacher_target_position"])
                first_teacher_id = teacher_ids[target_position]
                correction_ids, correction_values, residual_delta = spanning_correction_updates(
                    spec,
                    token_prob=token_probability,
                    prefix_mass=prefix_mass,
                    base_route=int(routes[first_teacher_id].item()),
                )
                for token_id, delta in zip(correction_ids, correction_values, strict=True):
                    mapped[int(token_id)] = mapped.get(int(token_id), 0.0) + float(delta)
                residual += float(residual_delta)

            tolerance = 2e-5
            if residual < -tolerance or any(value < -tolerance for value in mapped.values()):
                raise FloatingPointError("BPM exact correction made a target cell negative")
            residual = max(0.0, residual)
            mapped = {token_id: max(0.0, value) for token_id, value in mapped.items() if value > 0}
            total = sum(mapped.values()) + residual
            if not math.isfinite(total) or total <= 0:
                raise FloatingPointError("BPM exact target has invalid total mass")
            mapped = {token_id: value / total for token_id, value in mapped.items()}
            residual /= total
            ordered = sorted(mapped.items(), key=lambda item: item[0])
            if not ordered:
                ordered = [(int(row["student_token_id"]), 0.0)]
            row_ids.append([token_id for token_id, _ in ordered])
            row_probs.append([value for _, value in ordered])
            complements.append(residual)
            positive_cells = sum(value > 1e-8 for _, value in ordered) + int(residual > 1e-8)
            force_fkl.append(str(row.get("kind")) == "stop" or positive_cells == 1)

        width = max(len(values) for values in row_ids)
        ids_tensor = torch.zeros((len(active_rows), width), dtype=torch.long, device=device)
        probs_tensor = torch.zeros((len(active_rows), width), dtype=torch.float32, device=device)
        target_mask = torch.zeros((len(active_rows), width), dtype=torch.bool, device=device)
        for index, (ids, probabilities) in enumerate(zip(row_ids, row_probs, strict=True)):
            ids_tensor[index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            probs_tensor[index, : len(probabilities)] = torch.tensor(
                probabilities, dtype=torch.float32, device=device
            )
            target_mask[index, : len(ids)] = True
        kind_codes = {"aligned": 0, "interior": 1, "spanning": 2, "stop": 3}
        return {
            "target_ids": ids_tensor.detach(),
            "target_probs": probs_tensor.detach(),
            "target_mask": target_mask.detach(),
            "target_complement": torch.tensor(
                complements, dtype=torch.float32, device=device
            ).detach(),
            "student_positions": torch.tensor(
                [int(row["student_position"]) for row in active_rows],
                dtype=torch.long,
                device=device,
            ),
            "row_kinds": torch.tensor(
                [kind_codes[str(row["kind"])] for row in active_rows],
                dtype=torch.long,
                device=device,
            ),
            "force_fkl": torch.tensor(force_fkl, dtype=torch.bool, device=device),
        }

    @staticmethod
    def _kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(left.dtype).tiny
        return torch.where(
            left > 0,
            left * (left.clamp_min(eps).log() - right.clamp_min(eps).log()),
            torch.zeros_like(left),
        ).sum(dim=-1)

    def compute_student_token_losses(
        self,
        context: StudentScoringContext,
        teacher_payload: TensorPayload,
    ) -> torch.Tensor:
        beta, skew, _, _ = self._settings()
        if not math.isclose(float(context.temperature), 1.0):
            raise ValueError("BPM currently requires cross_tokenizer.temperature=1.0")
        required = {
            "target_ids", "target_probs", "target_mask", "target_complement",
            "student_positions", "row_kinds",
        }
        missing = required.difference(teacher_payload)
        if missing:
            raise ValueError(f"BPM Teacher payload is missing fields: {sorted(missing)}")
        device = context.response_token_ids.device
        positions = teacher_payload["student_positions"].to(
            device=device, dtype=torch.long
        )
        if positions.numel() == 0:
            # Keep an all-filtered sample connected to the model graph without
            # normalizing the entire response vocabulary merely to manufacture
            # a zero.  One row is enough for a differentiable scalar anchor.
            if context.distribution.length == 0:
                return torch.empty((0,), dtype=torch.float32, device=device)
            first_position = torch.zeros((1,), dtype=torch.long, device=device)
            first_token = context.response_token_ids[:1].reshape(1, 1)
            anchor = context.distribution.rows(first_position).selected_log_probs(
                first_token,
                vocab_size=self._student_vocab_size(),
                temperature=1.0,
            ).sum() * 0.0
            return anchor.expand(context.distribution.length)
        ids = teacher_payload["target_ids"].to(device=device, dtype=torch.long)
        target_probs = teacher_payload["target_probs"].to(
            device=device, dtype=torch.float32
        )
        target_mask = teacher_payload["target_mask"].to(device=device, dtype=torch.bool)
        target_complement = teacher_payload["target_complement"].to(
            device=device, dtype=torch.float32
        )
        row_kinds = teacher_payload["row_kinds"].to(device=device, dtype=torch.long)
        if bool(((positions < 0) | (positions >= context.distribution.length)).any()):
            raise ValueError("BPM payload contains an invalid Student row")

        student_log_probs = context.distribution.rows(positions).projected_log_probs(
            ids,
            target_mask,
            vocab_size=self._student_vocab_size(),
            temperature=1.0,
        ).float()
        target_explicit = target_probs.masked_fill(~target_mask, 0.0)
        target = torch.cat((target_explicit, target_complement.unsqueeze(-1)), dim=-1)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(target.dtype).tiny)
        force_fkl = teacher_payload.get("force_fkl", row_kinds == 3).to(
            device=device, dtype=torch.bool
        )
        if force_fkl.shape != row_kinds.shape:
            raise ValueError("BPM force_fkl rows do not match the target payload")
        losses = bpm_projected_divergence_from_log_probs(
            target,
            student_log_probs,
            beta=beta,
            skew=skew,
            force_fkl=force_fkl,
        )
        if not bool(torch.isfinite(losses).all()):
            raise FloatingPointError("BPM produced a non-finite Student token loss")
        token_losses = losses.new_zeros((context.distribution.length,))
        return token_losses.index_add(0, positions, losses)


@register_cross_tokenizer("bpm_paper")
class BPMPaperCrossTokenizer(BPMCrossTokenizer):
    """Strict full-vocabulary BPM contract from arXiv:2607.22334v1."""

    capabilities = CrossTokenizerCapabilities(
        fidelity="paper-equation",
        reference="BPM arXiv:2607.22334v1 Eqs. 3-10 and Algorithm 1",
        candidate_top_k_mode="zero",
        default_candidate_top_k=0,
        teacher_backends=("megatron", "sglang"),
        required_distribution_methods=(
            "full_probabilities",
            "selected_log_probs",
            "projected_log_probs",
        ),
        public_temperature_mode="one",
        supports_context_parallel=True,
        payload_position_field="student_positions",
        teacher_transports=(("sglang", "causal_hidden"),),
        requires_loss_token_normalization=True,
        requires_full_response_loss_mask=True,
        requires_loss_recompute_disabled=True,
        requires_chat_template=True,
        requires_thinking=True,
        external_reward_path=(
            "opd.cross_tokenizer.sglang_teacher.external_bpm_paper_hidden_reward"
        ),
        external_post_process_path=(
            "opd.cross_tokenizer.sglang_teacher.install_external_bpm_paper_hidden_payload"
        ),
    )

    def compute_fused_token_losses(self, context: FusedScoringContext) -> torch.Tensor:
        from .bpm_dense import compute_dense_bpm_paper_losses

        return compute_dense_bpm_paper_losses(self, context)

    def _default_spanning_policy(self) -> str:
        return "exact"

    def _validate_scoring_context(self, context: TeacherScoringContext, spanning_policy: str) -> None:
        if context.candidate_top_k != 0:
            raise ValueError("bpm_paper requires the complete Teacher vocabulary (candidate_top_k=0)")
        if spanning_policy != "exact":
            raise ValueError("bpm_paper requires spanning_policy='exact'")
