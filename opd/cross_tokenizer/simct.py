"""SimCT-style minimal-span alignment and reverse-KL supervision."""

from __future__ import annotations

import math
from typing import Any

import torch

from ._paper_utils import export_tokenizer_runtime_state, resolve_vocab_size
from .base import (
    AlignedSegment,
    AlignmentResult,
    CandidateScores,
    CrossTokenizerAlgorithm,
    CrossTokenizerCapabilities,
    CrossTokenizerAlignmentError,
    FusedScoringContext,
    SharedVocabulary,
    StudentScoringContext,
    TeacherScoringContext,
    TensorPayload,
)
from .registry import register_cross_tokenizer
from .tokenizer_bytes import semantic_stop_ids


def _canonical_token(token: str) -> str:
    return token.replace("Ġ", "▁")


def _canonical_vocab(tokenizer: Any) -> dict[str, int]:
    canonical: dict[str, int] = {}
    entries = sorted(
        ((_canonical_token(str(token)), int(token_id)) for token, token_id in tokenizer.get_vocab().items()),
        key=lambda item: (item[0], item[1]),
    )
    for token, token_id in entries:
        canonical.setdefault(token, token_id)
    return canonical


def build_shared_vocabulary(student_tokenizer: Any, teacher_tokenizer: Any) -> SharedVocabulary:
    """Build a deterministic one-to-one token mapping, including semantic EOS."""

    student_vocab = _canonical_vocab(student_tokenizer)
    teacher_vocab = _canonical_vocab(teacher_tokenizer)
    pairs = [
        (token, student_vocab[token], teacher_vocab[token])
        for token in sorted(set(student_vocab).intersection(teacher_vocab))
    ]

    student_eos = getattr(student_tokenizer, "eos_token_id", None)
    teacher_eos = getattr(teacher_tokenizer, "eos_token_id", None)
    if student_eos is not None and teacher_eos is not None:
        student_eos = int(student_eos)
        teacher_eos = int(teacher_eos)
        pairs = [pair for pair in pairs if pair[1] != student_eos and pair[2] != teacher_eos]
        pairs.append(("<semantic-eos>", student_eos, teacher_eos))

    # Canonicalization can collapse multiple spellings. Keep the first stable
    # one-to-one pair instead of allowing ambiguous IDs into TP lookup tables.
    tokens: list[str] = []
    student_ids: list[int] = []
    teacher_ids: list[int] = []
    seen_student: set[int] = set()
    seen_teacher: set[int] = set()
    for token, student_id, teacher_id in pairs:
        if student_id in seen_student or teacher_id in seen_teacher:
            continue
        tokens.append(token)
        student_ids.append(student_id)
        teacher_ids.append(teacher_id)
        seen_student.add(student_id)
        seen_teacher.add(teacher_id)

    if not tokens:
        raise ValueError("Teacher and Student tokenizers have no shared token mapping")
    return SharedVocabulary(tuple(student_ids), tuple(teacher_ids), tuple(tokens))


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    try:
        return tokenizer.decode([int(token_id) for token_id in token_ids], **kwargs)
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces")
        return tokenizer.decode([int(token_id) for token_id in token_ids], **kwargs)


def _gpt2_byte_decoder() -> dict[str, int]:
    """Return the inverse alphabet used by GPT-2/ByteLevel tokenizers."""

    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = list(byte_values)
    extra_index = 0
    for byte_value in range(256):
        if byte_value in byte_values:
            continue
        byte_values.append(byte_value)
        unicode_values.append(256 + extra_index)
        extra_index += 1
    return {chr(unicode_value): byte_value for byte_value, unicode_value in zip(byte_values, unicode_values)}


_BYTE_DECODER = _gpt2_byte_decoder()


def _bytelevel_boundaries(tokenizer: Any, token_ids: list[int]) -> tuple[tuple[int, ...], bytes] | None:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    decoder = getattr(backend, "decoder", None)
    if type(decoder).__name__ != "ByteLevel":
        return None
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert):
        return None

    token_strings = convert(token_ids)
    if isinstance(token_strings, str):
        token_strings = [token_strings]
    if len(token_strings) != len(token_ids):
        raise CrossTokenizerAlignmentError("ByteLevel tokenizer returned the wrong number of token strings")

    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    added_ids = {int(token_id) for token_id in (get_added_vocab() or {}).values()} if callable(get_added_vocab) else set()
    pieces: list[bytes] = []
    offsets: list[int] = []
    total = 0
    for token_id, token in zip(token_ids, token_strings, strict=True):
        if int(token_id) in added_ids:
            piece = _decode(tokenizer, [int(token_id)]).encode("utf-8")
        else:
            try:
                piece = bytes(_BYTE_DECODER[character] for character in str(token))
            except KeyError as error:
                raise CrossTokenizerAlignmentError(
                    f"Unsupported ByteLevel token character {error.args[0]!r} in token {token!r}"
                ) from error
        if not piece:
            raise CrossTokenizerAlignmentError(f"Token ID {token_id} decodes to an empty byte sequence")
        pieces.append(piece)
        total += len(piece)
        offsets.append(total)

    raw_bytes = b"".join(pieces)
    decoded_bytes = _decode(tokenizer, token_ids).encode("utf-8")
    terminal_incomplete = False
    if raw_bytes != decoded_bytes:
        # A max-length rollout can stop after the first one or two bytes of a
        # multi-byte UTF-8 character.  Hugging Face's text decoder replaces
        # that incomplete suffix with U+FFFD, so its re-encoded bytes cannot
        # equal the lossless ByteLevel bytes even though the token sequence is
        # valid and the two tokenizers can still be aligned by byte offsets.
        # Accept only this precise terminal condition; an invalid byte in the
        # middle of a response remains an alignment error.
        try:
            raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            incomplete_terminal = (
                error.reason == "unexpected end of data"
                and error.end == len(raw_bytes)
            )
            if not incomplete_terminal:
                raise CrossTokenizerAlignmentError(
                    "ByteLevel token bytes do not round-trip through the tokenizer decoder; "
                    "the response contains an invalid UTF-8 sequence"
                ) from error
            terminal_incomplete = True
        else:
            raise CrossTokenizerAlignmentError(
                "ByteLevel token bytes do not round-trip through the tokenizer decoder; "
                "the tokenizer changed a complete response byte sequence"
            )
    if terminal_incomplete:
        # ``decode(errors='replace')`` exposes the incomplete terminal bytes as
        # one U+FFFD character.  The Teacher response is reconstructed from
        # that decoded text, so it contains the replacement bytes rather than
        # the raw prefix.  Use the same alignment coordinate on the Student
        # side and move only the final boundary; invalid bytes in the middle
        # still raise above.  The original token IDs/bytes remain unchanged in
        # the rollout and are not treated as a completed Unicode character.
        raw_bytes = decoded_bytes
        offsets[-1] = len(raw_bytes)
    return tuple(offsets), raw_bytes


def _fast_reencoding_boundaries(tokenizer: Any, token_ids: list[int]) -> tuple[tuple[int, ...], bytes] | None:
    if not getattr(tokenizer, "is_fast", False) or not callable(tokenizer):
        return None
    text = _decode(tokenizer, token_ids)
    try:
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except (NotImplementedError, TypeError, ValueError):
        return None
    reencoded_ids = encoding.get("input_ids")
    offsets = encoding.get("offset_mapping")
    if reencoded_ids is None or offsets is None:
        return None
    if reencoded_ids and isinstance(reencoded_ids[0], list):
        reencoded_ids = reencoded_ids[0]
        offsets = offsets[0]
    if [int(token_id) for token_id in reencoded_ids] != token_ids:
        return None

    byte_offsets: list[int] = []
    previous = 0
    for _, character_end in offsets:
        byte_end = len(text[: int(character_end)].encode("utf-8"))
        if byte_end <= previous:
            return None
        byte_offsets.append(byte_end)
        previous = byte_end
    raw_bytes = text.encode("utf-8")
    if byte_offsets and byte_offsets[-1] != len(raw_bytes):
        return None
    return tuple(byte_offsets), raw_bytes


def _decoded_piece_boundaries(tokenizer: Any, token_ids: list[int]) -> tuple[tuple[int, ...], bytes]:
    pieces = [_decode(tokenizer, [token_id]).encode("utf-8") for token_id in token_ids]
    if any(not piece for piece in pieces):
        raise CrossTokenizerAlignmentError("A response token decodes to an empty byte sequence")
    raw_bytes = b"".join(pieces)
    if raw_bytes != _decode(tokenizer, token_ids).encode("utf-8"):
        raise CrossTokenizerAlignmentError(
            "Tokenizer does not expose reliable per-token byte boundaries; "
            "use a fast tokenizer or register support for its decoder"
        )
    offsets: list[int] = []
    total = 0
    for piece in pieces:
        total += len(piece)
        offsets.append(total)
    return tuple(offsets), raw_bytes


def _token_byte_boundaries(tokenizer: Any, token_ids: list[int]) -> tuple[tuple[int, ...], bytes]:
    if not token_ids:
        return (), b""
    bytelevel = _bytelevel_boundaries(tokenizer, token_ids)
    if bytelevel is not None:
        return bytelevel
    fast = _fast_reencoding_boundaries(tokenizer, token_ids)
    if fast is not None:
        return fast
    return _decoded_piece_boundaries(tokenizer, token_ids)


@register_cross_tokenizer("simct")
class SimCTCrossTokenizer(CrossTokenizerAlgorithm):
    """Paper-equation SimCT with mean-log-probability span scoring.

    The released repository instead averages raw logits.  That runnable-code
    contract is exposed separately as ``simct_official`` so experiments never
    change meaning when the paper/code discrepancy is resolved.
    """

    capabilities = CrossTokenizerCapabilities(
        fidelity="paper-equation",
        reference="SimCT arXiv:2605.07711v2 Eqs. 6-9 and Algorithm 1",
        candidate_top_k_mode="zero",
        default_candidate_top_k=0,
        teacher_backends=("megatron", "sglang"),
        required_distribution_methods=("selected_log_probs",),
        public_temperature_mode="one",
        context_parallel_backends=("sglang",),
        teacher_transports=(("sglang", "causal_hidden"),),
        requires_loss_token_normalization=True,
        requires_full_response_loss_mask=True,
        requires_chat_template=True,
        external_reward_path=(
            "opd.cross_tokenizer.sglang_teacher.external_causal_hidden_reward"
        ),
        external_post_process_path=(
            "opd.cross_tokenizer.sglang_teacher.install_external_causal_hidden_payload"
        ),
    )

    def _score_space(self) -> str:
        return "log_probability"

    def _validate_algorithm_settings(self) -> int:
        """Validate implementation knobs without silently changing SimCT."""

        legacy_threshold = float(self.algorithm_kwargs.get("gh_mask_threshold", 0.0))
        if not math.isfinite(legacy_threshold) or legacy_threshold != 0.0:
            raise ValueError(
                "gh_mask_threshold is not part of SimCT arXiv v2 or the pinned official "
                "span_ctkd.py; omit it or set it to 0"
            )
        chunk_size = int(self.algorithm_kwargs.get("dense_row_chunk_size", 8))
        if chunk_size <= 0:
            raise ValueError("SimCT dense_row_chunk_size must be positive")
        return chunk_size

    def _selected_scores(
        self,
        distribution: Any,
        token_ids: torch.Tensor,
        *,
        vocab_size: int,
    ) -> torch.Tensor:
        if self._score_space() == "raw_logit":
            return distribution.selected_logits(token_ids)
        return distribution.selected_log_probs(token_ids, vocab_size=vocab_size)

    def _candidate_top_k(self, context: TeacherScoringContext) -> int:
        # SimCT defines its public space over the complete shared vocabulary.
        # A positive K is an engineering approximation registered separately
        # as ``simct_topk`` so it cannot be mistaken for the paper baseline.
        del context
        return 0

    def build_shared_vocabulary(self) -> SharedVocabulary:
        if self.teacher_tokenizer is None or self.student_tokenizer is None:
            raise RuntimeError("SimCT shared-vocabulary construction requires both tokenizers")
        return build_shared_vocabulary(self.student_tokenizer, self.teacher_tokenizer)

    def export_runtime_state(self) -> dict[str, Any]:
        if self.student_tokenizer is not None and self.teacher_tokenizer is not None:
            self.runtime_state = export_tokenizer_runtime_state(
                self.student_tokenizer, self.teacher_tokenizer
            )
        return super().export_runtime_state()

    def align(self, teacher_token_ids: list[int], student_token_ids: list[int]) -> AlignmentResult:
        if self.teacher_tokenizer is None or self.student_tokenizer is None:
            raise RuntimeError("SimCT alignment requires both Teacher and Student tokenizers")
        teacher_token_ids = [int(token_id) for token_id in teacher_token_ids]
        student_token_ids = [int(token_id) for token_id in student_token_ids]
        if not teacher_token_ids and not student_token_ids:
            return AlignmentResult((), 0, 0)
        if not teacher_token_ids or not student_token_ids:
            raise CrossTokenizerAlignmentError(
                f"One response tokenization is empty: teacher={len(teacher_token_ids)}, "
                f"student={len(student_token_ids)}"
            )

        teacher_stops = set(semantic_stop_ids(self.teacher_tokenizer))
        student_stops = set(semantic_stop_ids(self.student_tokenizer))
        teacher_has_eos = bool(teacher_token_ids) and teacher_token_ids[-1] in teacher_stops
        student_has_eos = bool(student_token_ids) and student_token_ids[-1] in student_stops
        if teacher_has_eos != student_has_eos:
            raise CrossTokenizerAlignmentError("Teacher and Student responses disagree on terminal semantic EOS")

        teacher_content_ids = teacher_token_ids[:-1] if teacher_has_eos else teacher_token_ids
        student_content_ids = student_token_ids[:-1] if student_has_eos else student_token_ids
        if not teacher_content_ids and not student_content_ids:
            segments: list[AlignedSegment] = []
        elif not teacher_content_ids or not student_content_ids:
            raise CrossTokenizerAlignmentError(
                "One response content tokenization is empty: "
                f"teacher={len(teacher_content_ids)}, student={len(student_content_ids)}"
            )
        else:
            teacher_offsets, teacher_bytes = _token_byte_boundaries(self.teacher_tokenizer, teacher_content_ids)
            student_offsets, student_bytes = _token_byte_boundaries(self.student_tokenizer, student_content_ids)
            if teacher_bytes != student_bytes:
                raise CrossTokenizerAlignmentError(
                    "Decoded response byte streams diverged: "
                    f"teacher={teacher_bytes[-80:]!r}, student={student_bytes[-80:]!r}"
                )

            teacher_index = 0
            student_index = 0
            teacher_start = 0
            student_start = 0
            segments = []
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
                raise CrossTokenizerAlignmentError("Unconsumed response tokens after byte-boundary alignment")

        if teacher_has_eos:
            segments.append(
                AlignedSegment(
                    len(teacher_content_ids),
                    len(teacher_token_ids),
                    len(student_content_ids),
                    len(student_token_ids),
                )
            )
        return AlignmentResult(tuple(segments), len(teacher_token_ids), len(student_token_ids))

    def build_teacher_payload(self, context: TeacherScoringContext) -> TensorPayload:
        self._validate_algorithm_settings()
        alignment = context.alignment
        if context.distribution.length != alignment.teacher_length:
            raise ValueError("Teacher distribution length does not match the alignment")
        student_loss_mask = context.student_loss_mask.to(
            device=context.response_token_ids.device, dtype=torch.bool
        ).flatten()
        if student_loss_mask.numel() != alignment.student_length:
            raise ValueError("Student loss-mask length does not match the alignment")

        device = context.response_token_ids.device
        teacher_starts_all = torch.tensor(
            [segment.teacher_start for segment in alignment.segments], dtype=torch.long, device=device
        )
        teacher_ends_all = torch.tensor(
            [segment.teacher_end for segment in alignment.segments], dtype=torch.long, device=device
        )
        student_starts_all = torch.tensor(
            [segment.student_start for segment in alignment.segments], dtype=torch.long, device=device
        )
        student_ends_all = torch.tensor(
            [segment.student_end for segment in alignment.segments], dtype=torch.long, device=device
        )
        span_mask_all = torch.tensor(
            [segment.is_span for segment in alignment.segments], dtype=torch.bool, device=device
        )

        if alignment.skipped or not alignment.segments:
            valid_indices = torch.empty((0,), dtype=torch.long, device=device)
        else:
            mask_prefix = torch.cat(
                (torch.zeros(1, dtype=torch.long, device=device), student_loss_mask.long().cumsum(dim=0))
            )
            valid_counts = mask_prefix[student_ends_all] - mask_prefix[student_starts_all]
            segment_lengths = student_ends_all - student_starts_all
            valid_indices = (valid_counts == segment_lengths).nonzero(as_tuple=False).flatten()

        teacher_starts = teacher_starts_all.index_select(0, valid_indices)
        teacher_ends = teacher_ends_all.index_select(0, valid_indices)
        student_starts = student_starts_all.index_select(0, valid_indices)
        student_ends = student_ends_all.index_select(0, valid_indices)
        span_mask = span_mask_all.index_select(0, valid_indices)

        if teacher_starts.numel() == 0:
            shared_width = int(context.vocabulary_mapping.teacher_ids.numel())
            return {
                "candidate_ids": torch.empty(
                    (0, shared_width), dtype=torch.long, device=device
                ),
                "teacher_scores": torch.empty(
                    (0, shared_width + 1), dtype=torch.float32, device=device
                ),
                "candidate_mask": torch.empty(
                    (0, shared_width + 1), dtype=torch.bool, device=device
                ),
                "student_starts": student_starts.detach(),
                "student_ends": student_ends.detach(),
            }

        teacher_vocab_size = resolve_vocab_size(
            self.runtime_state, "teacher_vocab_size", self.teacher_tokenizer
        )
        candidate_k = self._candidate_top_k(context)
        if self._score_space() == "raw_logit":
            if candidate_k != 0:
                raise ValueError("Official SimCT raw-logit scoring requires the complete shared vocabulary")
            teacher_candidate_ids = context.vocabulary_mapping.teacher_ids.to(device=device, dtype=torch.long)
            student_candidate_ids = context.vocabulary_mapping.student_ids.to(device=device, dtype=torch.long)
            expanded_teacher_ids = teacher_candidate_ids.unsqueeze(0).expand(teacher_starts.numel(), -1)
            raw_scores = context.distribution.rows(teacher_starts).selected_logits(expanded_teacher_ids)
            output_ids = student_candidate_ids.unsqueeze(0).expand_as(expanded_teacher_ids)
            candidate_scores = CandidateScores(
                output_ids,
                raw_scores,
                torch.ones_like(output_ids, dtype=torch.bool),
            )
        else:
            candidate_scores = context.distribution.rows(teacher_starts).mapped_topk(
                context.vocabulary_mapping.teacher_ids,
                context.vocabulary_mapping.student_ids,
                k=candidate_k,
                vocab_size=teacher_vocab_size,
            )
        teacher_token_scores = self._selected_scores(
            context.distribution,
            context.response_token_ids.unsqueeze(-1),
            vocab_size=teacher_vocab_size,
        ).squeeze(-1)
        score_prefix = torch.cat(
            (teacher_token_scores.new_zeros(1), teacher_token_scores.cumsum(dim=0))
        )
        span_scores = (score_prefix[teacher_ends] - score_prefix[teacher_starts]) / (
            teacher_ends - teacher_starts
        ).clamp_min(1)
        teacher_scores = torch.cat((candidate_scores.log_probs, span_scores.unsqueeze(-1)), dim=-1)
        candidate_mask = torch.cat((candidate_scores.mask, span_mask.unsqueeze(-1)), dim=-1)

        candidate_output_ids = candidate_scores.output_ids
        if candidate_mask.numel() and not bool(candidate_mask.any(dim=-1).all()):
            raise ValueError("Every SimCT segment must contain at least one public-space candidate")
        return {
            "candidate_ids": candidate_output_ids.detach(),
            "teacher_scores": teacher_scores.detach(),
            "candidate_mask": candidate_mask.detach(),
            "student_starts": student_starts.detach(),
            "student_ends": student_ends.detach(),
        }

    def compute_fused_token_losses(self, context: FusedScoringContext) -> torch.Tensor:
        """Evaluate exact SimCT from causal Teacher hidden rows inside Slime.

        Only realized-token scalar scores are gathered across Context Parallel
        ranks. Shared-vocabulary rows remain owned by their local CP rank and
        are evaluated in bounded chunks.
        """

        chunk_size = self._validate_algorithm_settings()
        alignment = context.alignment
        if alignment.skipped:
            raise ValueError("A skipped SimCT alignment must be handled before fused scoring")
        if context.teacher_distribution.length != alignment.teacher_length:
            raise ValueError("Teacher distribution length does not match the SimCT alignment")
        device = context.student.response_token_ids.device
        teacher_response_ids = context.teacher_response_token_ids.to(
            device=device, dtype=torch.long
        ).flatten()
        if teacher_response_ids.numel() != alignment.teacher_length:
            raise ValueError("Teacher response token IDs do not match the SimCT alignment")
        mapping = context.vocabulary_mapping
        if mapping is None or mapping.teacher_ids.numel() == 0:
            raise ValueError("Fused SimCT requires the complete shared vocabulary mapping")
        if mapping.teacher_ids.numel() != mapping.student_ids.numel():
            raise ValueError("SimCT shared Teacher/Student vocabulary widths differ")

        local_positions = context.local_response_positions.to(
            device=device, dtype=torch.long
        ).flatten()
        if local_positions.numel() != context.student.distribution.length:
            raise ValueError("Local SimCT response positions differ from Student distribution rows")
        if local_positions.numel() > 1 and bool(
            (local_positions[1:] <= local_positions[:-1]).any()
        ):
            raise ValueError("Local SimCT response positions must be strictly increasing")
        if local_positions.numel() and bool(
            ((local_positions < 0) | (local_positions >= alignment.student_length)).any()
        ):
            raise ValueError("Local SimCT response position falls outside the alignment")

        student_vocab_size = resolve_vocab_size(
            self.runtime_state, "student_vocab_size", self.student_tokenizer
        )
        teacher_vocab_size = resolve_vocab_size(
            self.runtime_state, "teacher_vocab_size", self.teacher_tokenizer
        )
        local_observed_scores = self._selected_scores(
            context.student.distribution,
            context.student.response_token_ids.unsqueeze(-1),
            vocab_size=student_vocab_size,
        ).squeeze(-1)
        if context.gather_student_rows is None:
            expected_positions = torch.arange(
                alignment.student_length, device=device, dtype=torch.long
            )
            if not torch.equal(local_positions, expected_positions):
                raise ValueError("SimCT needs a CP Student-row gather for non-local MAUs")
            global_student_observed = local_observed_scores
        else:
            global_student_observed = context.gather_student_rows(local_observed_scores)
        if global_student_observed.shape != (alignment.student_length,):
            raise ValueError("Gathered SimCT Student scores have the wrong global response length")

        teacher_observed_scores = self._selected_scores(
            context.teacher_distribution,
            teacher_response_ids.unsqueeze(-1),
            vocab_size=teacher_vocab_size,
        ).squeeze(-1)
        student_prefix = torch.cat(
            (global_student_observed.new_zeros(1), global_student_observed.cumsum(dim=0))
        )
        teacher_prefix = torch.cat(
            (teacher_observed_scores.new_zeros(1), teacher_observed_scores.cumsum(dim=0))
        )

        segments = alignment.segments
        student_starts_all = torch.tensor(
            [segment.student_start for segment in segments], device=device, dtype=torch.long
        )
        if local_positions.numel():
            local_start_rows = torch.searchsorted(local_positions, student_starts_all)
            owned = (local_start_rows < local_positions.numel()) & (
                local_positions.index_select(
                    0, local_start_rows.clamp(max=local_positions.numel() - 1)
                )
                == student_starts_all
            )
        else:
            local_start_rows = torch.zeros_like(student_starts_all)
            owned = torch.zeros_like(student_starts_all, dtype=torch.bool)
        owned_segment_indices = torch.nonzero(owned, as_tuple=False).flatten()
        # Every CP rank participated in ``gather_student_rows`` above, so every
        # rank must also traverse its autograd node.  A short/packed response
        # can leave one CP rank with no MAU start rows; returning a purely local
        # zero in that case would make the other ranks wait forever in the
        # gather's backward all-reduce.  Keep an explicit zero-valued dependency
        # on the gathered scalar rows without changing the SimCT objective.
        output = (
            local_observed_scores.float() * 0.0
            + global_student_observed.float().sum() * 0.0
        )
        if owned_segment_indices.numel() == 0:
            return output

        teacher_ids = mapping.teacher_ids.to(device=device, dtype=torch.long)
        student_ids = mapping.student_ids.to(device=device, dtype=torch.long)
        for offset in range(0, owned_segment_indices.numel(), chunk_size):
            segment_indices = owned_segment_indices[offset : offset + chunk_size]
            selected_segments = [segments[int(index)] for index in segment_indices.tolist()]
            teacher_starts = torch.tensor(
                [segment.teacher_start for segment in selected_segments],
                device=device,
                dtype=torch.long,
            )
            teacher_ends = torch.tensor(
                [segment.teacher_end for segment in selected_segments],
                device=device,
                dtype=torch.long,
            )
            student_starts = torch.tensor(
                [segment.student_start for segment in selected_segments],
                device=device,
                dtype=torch.long,
            )
            student_ends = torch.tensor(
                [segment.student_end for segment in selected_segments],
                device=device,
                dtype=torch.long,
            )
            span_mask = torch.tensor(
                [segment.is_span for segment in selected_segments],
                device=device,
                dtype=torch.bool,
            )
            local_starts = local_start_rows.index_select(0, segment_indices)

            expanded_teacher_ids = teacher_ids.unsqueeze(0).expand(teacher_starts.numel(), -1)
            expanded_student_ids = student_ids.unsqueeze(0).expand(student_starts.numel(), -1)
            teacher_shared = self._selected_scores(
                context.teacher_distribution.rows(teacher_starts),
                expanded_teacher_ids,
                vocab_size=teacher_vocab_size,
            )
            student_shared = self._selected_scores(
                context.student.distribution.rows(local_starts),
                expanded_student_ids,
                vocab_size=student_vocab_size,
            )
            teacher_spans = (teacher_prefix[teacher_ends] - teacher_prefix[teacher_starts]) / (
                teacher_ends - teacher_starts
            )
            student_spans = (student_prefix[student_ends] - student_prefix[student_starts]) / (
                student_ends - student_starts
            )
            teacher_public = torch.cat((teacher_shared, teacher_spans.unsqueeze(-1)), dim=-1)
            student_public = torch.cat((student_shared, student_spans.unsqueeze(-1)), dim=-1)
            public_mask = torch.cat(
                (torch.ones_like(teacher_shared, dtype=torch.bool), span_mask.unsqueeze(-1)),
                dim=-1,
            )
            losses = self._reverse_kl(
                student_public,
                teacher_public,
                public_mask,
                temperature=context.student.temperature,
            )
            if not bool(torch.isfinite(losses).all()):
                raise FloatingPointError("SimCT produced a non-finite reverse-KL loss")
            output = output.index_add(0, local_starts, losses.float())
        return output

    @staticmethod
    def _reverse_kl(
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if student_scores.shape != teacher_scores.shape or candidate_mask.shape != student_scores.shape:
            raise ValueError("Student scores, Teacher scores and candidate mask must have identical shapes")
        if not bool(candidate_mask.any(dim=-1).all()):
            raise ValueError("Each SimCT loss row must contain at least one valid candidate")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"SimCT public-space temperature must be positive and finite, got {temperature}")
        student_scores = student_scores.float().div(temperature).masked_fill(~candidate_mask, float("-inf"))
        teacher_scores = (
            teacher_scores.detach().float().div(temperature).masked_fill(~candidate_mask, float("-inf"))
        )
        student_log_probs = torch.log_softmax(student_scores, dim=-1)
        teacher_log_probs = torch.log_softmax(teacher_scores, dim=-1)
        student_probs = student_log_probs.exp().masked_fill(~candidate_mask, 0.0)
        log_ratio = (student_log_probs - teacher_log_probs).masked_fill(~candidate_mask, 0.0)
        return (student_probs * log_ratio).sum(dim=-1)

    def compute_student_token_losses(
        self,
        context: StudentScoringContext,
        teacher_payload: TensorPayload,
    ) -> torch.Tensor:
        required = {"candidate_ids", "teacher_scores", "candidate_mask", "student_starts", "student_ends"}
        missing = required.difference(teacher_payload)
        if missing:
            raise ValueError(f"SimCT Teacher payload is missing fields: {sorted(missing)}")

        student_vocab_size = resolve_vocab_size(
            self.runtime_state, "student_vocab_size", self.student_tokenizer
        )
        actual_student_scores = self._selected_scores(
            context.distribution,
            context.response_token_ids.unsqueeze(-1),
            vocab_size=student_vocab_size,
        ).squeeze(-1)
        if actual_student_scores.numel() != context.distribution.length:
            raise ValueError("Student token scores do not match the response distribution length")

        ids = teacher_payload["candidate_ids"].to(device=actual_student_scores.device, dtype=torch.long)
        teacher_scores = teacher_payload["teacher_scores"].to(
            device=actual_student_scores.device, dtype=actual_student_scores.dtype
        )
        candidate_mask = teacher_payload["candidate_mask"].to(
            device=actual_student_scores.device, dtype=torch.bool
        )
        starts = teacher_payload["student_starts"].to(device=actual_student_scores.device, dtype=torch.long)
        ends = teacher_payload["student_ends"].to(device=actual_student_scores.device, dtype=torch.long)
        if not (ids.ndim == 2 and teacher_scores.ndim == 2 and candidate_mask.ndim == 2):
            raise ValueError("SimCT candidate payload tensors must be two-dimensional")
        if not (ids.size(0) == teacher_scores.size(0) == candidate_mask.size(0) == starts.numel() == ends.numel()):
            raise ValueError("SimCT Teacher payload row counts do not match")
        if starts.numel() == 0:
            return actual_student_scores * 0.0
        invalid_bounds = (starts < 0) | (starts >= ends) | (ends > context.distribution.length)
        if bool(invalid_bounds.any()):
            raise ValueError("SimCT Teacher payload contains an invalid Student span")

        shared_student_scores = self._selected_scores(
            context.distribution.rows(starts),
            ids,
            vocab_size=student_vocab_size,
        )
        score_prefix = torch.cat(
            (actual_student_scores.new_zeros(1), actual_student_scores.cumsum(dim=0))
        )
        span_scores = (score_prefix[ends] - score_prefix[starts]) / (ends - starts)
        student_scores = torch.cat((shared_student_scores, span_scores.unsqueeze(-1)), dim=-1)
        segment_losses = self._reverse_kl(
            student_scores,
            teacher_scores,
            candidate_mask,
            temperature=context.temperature,
        )
        return (actual_student_scores * 0.0).index_add(
            0, starts, segment_losses.to(actual_student_scores.dtype)
        )


@register_cross_tokenizer("simct_official")
class SimCTOfficialCrossTokenizer(SimCTCrossTokenizer):
    """Full-vocabulary SimCT matching the released ``span_ctkd.py`` code."""

    capabilities = CrossTokenizerCapabilities(
        fidelity="official-source",
        reference="sunjie279/SimCT-@4dae6fc:kdflow/algorithms/span_ctkd.py",
        candidate_top_k_mode="zero",
        default_candidate_top_k=0,
        teacher_backends=("megatron", "sglang"),
        required_distribution_methods=("selected_logits", "selected_log_probs"),
        public_temperature_mode="one",
        context_parallel_backends=("sglang",),
        teacher_transports=(("sglang", "causal_hidden"),),
        requires_loss_token_normalization=True,
        requires_full_response_loss_mask=True,
        requires_chat_template=True,
        external_reward_path=(
            "opd.cross_tokenizer.sglang_teacher.external_causal_hidden_reward"
        ),
        external_post_process_path=(
            "opd.cross_tokenizer.sglang_teacher.install_external_causal_hidden_payload"
        ),
    )

    def _score_space(self) -> str:
        return "raw_logit"

@register_cross_tokenizer("simct_topk")
class SimCTTopKCrossTokenizer(SimCTCrossTokenizer):
    """Explicit non-paper SimCT approximation retaining only Teacher Top-K shared tokens."""

    capabilities = CrossTokenizerCapabilities(
        fidelity="engineering-approximation",
        reference="SimCT shared-vocabulary source-TopK ablation",
        candidate_top_k_mode="positive",
        default_candidate_top_k=128,
        teacher_backends=("megatron", "sglang"),
        required_distribution_methods=("selected_log_probs",),
    )

    def _candidate_top_k(self, context: TeacherScoringContext) -> int:
        if context.candidate_top_k <= 0:
            raise ValueError("simct_topk requires a positive candidate_top_k")
        self._validate_algorithm_settings()
        return context.candidate_top_k
