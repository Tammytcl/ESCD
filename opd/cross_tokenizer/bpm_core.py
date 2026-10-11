"""Byte-Prefix Marginalization primitives.

This module contains the tokenizer-pair precomputation and CPU reference
projection used by both the teacher runtime and offline policy comparison.
It deliberately has no Transformers or CUDA dependency.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .tokenizer_bytes import (
    semantic_stop_ids as _stop_ids,
    tokenizer_byte_table,
)


RESIDUAL_ROUTE = -1
INCOMPATIBLE_ROUTE = -2
BPM_ARTIFACT_VERSION = 1
BPM_PAYLOAD_VERSION = 2


class _ByteTrie:
    def __init__(self) -> None:
        self.children: dict[int, _ByteTrie] = {}
        self.token_id: int | None = None

    def add(self, value: bytes, token_id: int) -> None:
        node = self
        for byte in value:
            node = node.children.setdefault(byte, _ByteTrie())
        if node.token_id is None or token_id < node.token_id:
            node.token_id = int(token_id)

    def longest_prefix(self, value: bytes) -> int:
        node = self
        selected = node.token_id
        for byte in value:
            node = node.children.get(byte)
            if node is None:
                break
            if node.token_id is not None:
                selected = node.token_id
        return RESIDUAL_ROUTE if selected is None else selected


_STUDENT_TRIE_CACHE: dict[str, _ByteTrie] = {}


def _student_trie(artifact: "BPMRoutingArtifact") -> _ByteTrie:
    trie = _STUDENT_TRIE_CACHE.get(artifact.sha256)
    if trie is not None:
        return trie
    trie = _ByteTrie()
    stop_ids = set(artifact.student_stop_ids)
    for token_id, value in enumerate(artifact.student_bytes):
        if token_id not in stop_ids and value:
            trie.add(value, token_id)
    _STUDENT_TRIE_CACHE[artifact.sha256] = trie
    return trie


def compile_longest_prefix_routes(
    teacher_bytes: Sequence[bytes | None],
    student_bytes: Sequence[bytes | None],
    *,
    excluded_student_ids: Iterable[int] = (),
    excluded_teacher_ids: Iterable[int] = (),
) -> tuple[int, ...]:
    """Map every teacher token to its longest Student byte-prefix token."""

    excluded_student = {int(value) for value in excluded_student_ids}
    excluded_teacher = {int(value) for value in excluded_teacher_ids}
    trie = _ByteTrie()
    for student_id, value in enumerate(student_bytes):
        if student_id not in excluded_student and value:
            trie.add(value, student_id)
    return tuple(
        RESIDUAL_ROUTE
        if teacher_id in excluded_teacher or not value
        else trie.longest_prefix(value)
        for teacher_id, value in enumerate(teacher_bytes)
    )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class BPMRoutingArtifact:
    teacher_to_student: tuple[int, ...]
    teacher_bytes: tuple[bytes | None, ...]
    student_bytes: tuple[bytes | None, ...]
    teacher_stop_ids: tuple[int, ...]
    student_stop_ids: tuple[int, ...]
    student_byte_ranks: tuple[int, ...]
    sha256: str
    schema_version: int = BPM_ARTIFACT_VERSION

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "teacher_to_student": list(self.teacher_to_student),
            "student_byte_ranks": list(self.student_byte_ranks),
            "student_effective_vocab_size": len(self.student_bytes),
            "teacher_effective_vocab_size": len(self.teacher_bytes),
            "teacher_bytes_hex": [None if value is None else value.hex() for value in self.teacher_bytes],
            "student_bytes_hex": [None if value is None else value.hex() for value in self.student_bytes],
            "teacher_stop_ids": list(self.teacher_stop_ids),
            "student_stop_ids": list(self.student_stop_ids),
        }
        if include_sha256:
            payload["sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BPMRoutingArtifact":
        artifact = cls(
            teacher_to_student=tuple(int(value) for value in payload["teacher_to_student"]),
            teacher_bytes=tuple(
                None if value is None else bytes.fromhex(str(value))
                for value in payload["teacher_bytes_hex"]
            ),
            student_bytes=tuple(
                None if value is None else bytes.fromhex(str(value))
                for value in payload["student_bytes_hex"]
            ),
            teacher_stop_ids=tuple(int(value) for value in payload.get("teacher_stop_ids", ())),
            student_stop_ids=tuple(int(value) for value in payload.get("student_stop_ids", ())),
            student_byte_ranks=tuple(int(value) for value in payload["student_byte_ranks"]),
            sha256=str(payload.get("sha256") or ""),
            schema_version=int(payload.get("schema_version", -1)),
        )
        if artifact.schema_version != BPM_ARTIFACT_VERSION:
            raise ValueError(f"unsupported BPM artifact version: {artifact.schema_version!r}")
        if int(payload.get("student_effective_vocab_size", -1)) != len(artifact.student_bytes):
            raise ValueError("BPM Student effective vocabulary size is inconsistent")
        if int(payload.get("teacher_effective_vocab_size", -1)) != len(artifact.teacher_bytes):
            raise ValueError("BPM Teacher effective vocabulary size is inconsistent")
        expected = hashlib.sha256(_canonical_json(artifact.to_dict(include_sha256=False))).hexdigest()
        if artifact.sha256 != expected:
            raise ValueError(
                f"BPM routing SHA mismatch: stored={artifact.sha256}, computed={expected}"
            )
        return artifact


def build_routing_artifact(
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    *,
    student_vocab_size: int | None = None,
    teacher_vocab_size: int | None = None,
    additional_student_stop_ids: Iterable[int] = (),
    additional_teacher_stop_ids: Iterable[int] = (),
) -> BPMRoutingArtifact:
    student_table = tokenizer_byte_table(student_tokenizer, vocab_size=student_vocab_size)
    teacher_table = tokenizer_byte_table(teacher_tokenizer, vocab_size=teacher_vocab_size)
    # The paper collects stop tokens from three sources: tokenizer EOS,
    # generation-config EOS and the chat-template turn terminator.  _stop_ids
    # covers the tokenizer/template-visible part; runtimes that also own model
    # paths pass generation-config IDs through these additive arguments.
    student_stops = tuple(
        sorted(
            set(_stop_ids(student_tokenizer))
            | {
                int(value)
                for value in additional_student_stop_ids
                if 0 <= int(value) < len(student_table)
            }
        )
    )
    teacher_stops = tuple(
        sorted(
            set(_stop_ids(teacher_tokenizer))
            | {
                int(value)
                for value in additional_teacher_stop_ids
                if 0 <= int(value) < len(teacher_table)
            }
        )
    )
    routes = compile_longest_prefix_routes(
        teacher_table,
        student_table,
        excluded_student_ids=student_stops,
        excluded_teacher_ids=teacher_stops,
    )
    byte_order = sorted(
        range(len(student_table)),
        key=lambda token_id: (
            student_table[token_id] is None,
            student_table[token_id] or b"",
            token_id,
        ),
    )
    byte_ranks = [0] * len(student_table)
    for rank, token_id in enumerate(byte_order):
        byte_ranks[token_id] = rank
    provisional = BPMRoutingArtifact(
        routes,
        teacher_table,
        student_table,
        teacher_stops,
        student_stops,
        tuple(byte_ranks),
        "",
    )
    digest = hashlib.sha256(_canonical_json(provisional.to_dict(include_sha256=False))).hexdigest()
    return BPMRoutingArtifact(
        routes,
        teacher_table,
        student_table,
        teacher_stops,
        student_stops,
        tuple(byte_ranks),
        digest,
    )


def _strip_terminal_stop(
    ids: Sequence[int], table: Sequence[bytes | None], stop_ids: set[int]
) -> tuple[list[int], int | None]:
    values = [int(value) for value in ids]
    stop = values.pop() if values and values[-1] in stop_ids else None
    if any(value < 0 or value >= len(table) or table[value] is None for value in values):
        raise ValueError("BPM assistant span contains an ID without lossless token bytes")
    return values, stop


def build_bpm_rows(
    *,
    student_response_ids: Sequence[int],
    teacher_input_ids: Sequence[int],
    assistant_spans: Sequence[Mapping[str, Any]],
    artifact: BPMRoutingArtifact,
) -> tuple[list[dict[str, Any]], list[list[int]], dict[str, int | float]]:
    """Build schema-v2 causal BPM rows from Kimi-native assistant spans."""

    rows: list[dict[str, Any]] = []
    ranges: list[list[int]] = []
    student_stops = set(artifact.student_stop_ids)
    teacher_stops = set(artifact.teacher_stop_ids)
    kind_counts = {
        "aligned": 0,
        "interior": 0,
        "spanning": 0,
        "excluded": 0,
        "stop": 0,
    }
    whitespace = 0
    for raw_span in assistant_spans:
        span_row_start = len(rows)
        ss, se = int(raw_span["student_start"]), int(raw_span["student_end"])
        ts, te = int(raw_span["teacher_start"]), int(raw_span["teacher_end"])
        if not (0 <= ss < se <= len(student_response_ids)):
            raise ValueError("BPM Student assistant span is outside the response")
        if not (0 < ts < te <= len(teacher_input_ids)):
            raise ValueError("BPM Teacher assistant span is outside the causal context")
        ranges.append([ts, te])
        s_ids, s_stop = _strip_terminal_stop(
            student_response_ids[ss:se], artifact.student_bytes, student_stops
        )
        t_ids, t_stop = _strip_terminal_stop(
            teacher_input_ids[ts:te], artifact.teacher_bytes, teacher_stops
        )
        # Only the final stop in an assistant span is a semantic template
        # terminator.  An internal Teacher stop means that a literal special
        # string from content was tokenized as control text.  Sending such a
        # row to the SGLang scorer is unsafe: its spanning-chain compiler
        # intentionally stops at semantic Teacher stops and cannot represent a
        # Student token crossing that boundary.  Fail closed before making the
        # request; the reward hook turns this into a zero-loss no-op sample.
        internal_teacher_stops = [token_id for token_id in t_ids if token_id in teacher_stops]
        if internal_teacher_stops:
            raise ValueError(
                "BPM assistant span contains internal Teacher stop token(s): "
                f"{internal_teacher_stops[:8]}"
            )
        s_pieces = [artifact.student_bytes[token_id] or b"" for token_id in s_ids]
        t_pieces = [artifact.teacher_bytes[token_id] or b"" for token_id in t_ids]
        if b"".join(s_pieces) != b"".join(t_pieces):
            raise ValueError("BPM Student and Teacher assistant bytes differ")

        teacher_offsets: list[tuple[int, int]] = []
        offset = 0
        for piece in t_pieces:
            teacher_offsets.append((offset, offset + len(piece)))
            offset += len(piece)
        teacher_ends = {end for _, end in teacher_offsets}
        student_ends: set[int] = set()
        offset = 0
        for piece in s_pieces:
            offset += len(piece)
            student_ends.add(offset)
        sync_offsets = {0, *(teacher_ends & student_ends)}
        student_offset = 0
        teacher_index = 0
        for local_student, (student_id, piece) in enumerate(zip(s_ids, s_pieces, strict=True)):
            while teacher_index < len(teacher_offsets) and student_offset >= teacher_offsets[teacher_index][1]:
                teacher_index += 1
            if teacher_index >= len(teacher_offsets):
                raise ValueError("BPM could not locate the causal Teacher token")
            teacher_start, teacher_end = teacher_offsets[teacher_index]
            prefix = t_pieces[teacher_index][: student_offset - teacher_start]
            if student_offset + len(piece) > teacher_end:
                kind = "spanning"
            elif prefix:
                kind = "interior"
            else:
                kind = "aligned"
            student_end = student_offset + len(piece)
            if (
                kind == "spanning"
                and student_offset not in sync_offsets
                and student_end not in sync_offsets
            ):
                kind_counts["excluded"] += 1
                student_offset = student_end
                continue
            is_whitespace = bool(piece) and all(byte in (9, 32) for byte in piece)
            rows.append(
                {
                    "student_position": ss + local_student,
                    "student_token_id": student_id,
                    "teacher_target_position": ts + teacher_index,
                    "prefix_hex": prefix.hex(),
                    "kind": kind,
                    "whitespace": is_whitespace,
                }
            )
            kind_counts[kind] += 1
            whitespace += int(is_whitespace)
            student_offset = student_end
        if (s_stop is None) != (t_stop is None):
            raise ValueError("BPM Student/Teacher stop-token structure differs")
        if s_stop is not None and t_stop is not None:
            # At a shared byte boundary the paper extends phi by mapping every
            # Teacher stop token to the stop token actually emitted by this
            # Student response.  Keep that response-local identity on all
            # content rows in the span; the Teacher scorer consumes it for
            # aligned rows, while interior rows continue to condition only on
            # Teacher tokens compatible with their already-emitted prefix.
            for row in rows[span_row_start:]:
                row["student_stop_token_id"] = s_stop
            rows.append(
                {
                    "student_position": se - 1,
                    "student_token_id": s_stop,
                    "teacher_target_position": te - 1,
                    "prefix_hex": "",
                    "kind": "stop",
                    "whitespace": False,
                }
            )
            kind_counts["stop"] += 1
    stats: dict[str, int | float] = {
        "teacher_token_count": len(teacher_input_ids),
        "bpm_rows": len(rows),
        "whitespace_rows": whitespace,
        **{f"{kind}_rows": count for kind, count in kind_counts.items()},
    }
    return rows, ranges, stats


def routes_for_bpm_row(
    artifact: BPMRoutingArtifact,
    row: Mapping[str, Any],
) -> tuple[int, ...]:
    """Return the exact Teacher-vocabulary route used by one BPM row.

    This is the dependency-free oracle for the SGLang Teacher implementation.
    ``INCOMPATIBLE_ROUTE`` denotes Teacher outcomes removed by an interior
    prefix condition; ``RESIDUAL_ROUTE`` remains an explicit probability cell.
    """

    kind = str(row.get("kind", ""))
    prefix = bytes.fromhex(str(row.get("prefix_hex", "")))
    teacher_stops = set(artifact.teacher_stop_ids)
    if kind == "stop":
        student_stop = int(row["student_token_id"])
        if student_stop not in set(artifact.student_stop_ids):
            raise ValueError("BPM stop row contains an unknown Student stop token")
        return tuple(
            student_stop if token_id in teacher_stops else RESIDUAL_ROUTE
            for token_id in range(len(artifact.teacher_bytes))
        )
    if kind not in {"aligned", "interior", "spanning"}:
        raise ValueError(f"unsupported BPM row kind: {kind!r}")
    if prefix:
        trie = _student_trie(artifact)
        return tuple(
            INCOMPATIBLE_ROUTE
            if token_id in teacher_stops or value is None or not value.startswith(prefix)
            else trie.longest_prefix(value[len(prefix) :])
            for token_id, value in enumerate(artifact.teacher_bytes)
        )

    realized_stop = int(row.get("student_stop_token_id", -1))
    if realized_stop >= 0:
        if realized_stop not in set(artifact.student_stop_ids):
            raise ValueError("BPM row contains an unknown realized Student stop token")
        return tuple(
            realized_stop if token_id in teacher_stops else route
            for token_id, route in enumerate(artifact.teacher_to_student)
        )
    return artifact.teacher_to_student


def spanning_chain_spec(
    artifact: BPMRoutingArtifact,
    row: Mapping[str, Any],
    input_ids: Sequence[int],
) -> dict[str, Any]:
    """Compile the paper's Equation 6 queries for one spanning row."""

    if row.get("kind") != "spanning":
        raise ValueError("BPM chain specification requires a spanning row")
    target_position = int(row["teacher_target_position"])
    if not 0 <= target_position < len(input_ids):
        raise ValueError("BPM spanning target position is outside the Teacher input")
    prefix = bytes.fromhex(str(row.get("prefix_hex", "")))
    teacher_stops = set(artifact.teacher_stop_ids)
    student_stops = set(artifact.student_stop_ids)
    first_teacher_id = int(input_ids[target_position])
    first_piece = artifact.teacher_bytes[first_teacher_id]
    if first_teacher_id in teacher_stops or first_piece is None or not first_piece.startswith(prefix):
        raise ValueError("BPM spanning row is incompatible with its realized Teacher token")

    max_student_bytes = max(
        len(value)
        for token_id, value in enumerate(artifact.student_bytes)
        if token_id not in student_stops and value
    )
    teacher_ids: list[int] = []
    teacher_pieces: list[bytes] = []
    stream_size = 0
    for raw_teacher_id in input_ids[target_position:]:
        teacher_id = int(raw_teacher_id)
        value = artifact.teacher_bytes[teacher_id]
        if teacher_id in teacher_stops or value is None:
            break
        teacher_ids.append(teacher_id)
        teacher_pieces.append(value)
        stream_size += len(value)
        if stream_size - len(prefix) >= max_student_bytes:
            break
    if not teacher_pieces:
        raise ValueError("BPM spanning row has no realized Teacher byte stream")

    suffix = b"".join(teacher_pieces)[len(prefix) :]
    canonical_student_ids: dict[bytes, int] = {}
    for student_id, student_piece in enumerate(artifact.student_bytes):
        if student_id not in student_stops and student_piece:
            canonical_student_ids.setdefault(student_piece, student_id)
    candidate_ids: list[int] = []
    first_remaining = len(teacher_pieces[0]) - len(prefix)
    # Walk the trie once and retain terminal tokens extending beyond the first
    # Teacher token. _ByteTrie intentionally hides its nodes, so reproduce the
    # bounded prefix lookup over Student tokens here; this is preparation-time
    # code and is cached by the caller per spanning row.
    for student_piece, student_id in canonical_student_ids.items():
        if (
            len(student_piece) > first_remaining
            and suffix.startswith(student_piece)
        ):
            candidate_ids.append(student_id)
    realized_student_id = int(row["student_token_id"])
    if realized_student_id not in candidate_ids:
        # A tokenizer may expose aliases (different ids with identical byte
        # pieces).  The canonical map above keeps one id per byte string, so
        # the realized id can be absent even when its bytes are valid.  Check
        # the realized bytes before rejecting the row and preserve that id for
        # downstream routing/scoring.
        realized_piece = (
            artifact.student_bytes[realized_student_id]
            if 0 <= realized_student_id < len(artifact.student_bytes)
            else None
        )
        if (
            realized_student_id in student_stops
            or not realized_piece
            or len(realized_piece) <= first_remaining
            or not suffix.startswith(realized_piece)
        ):
            raise ValueError(
                "realized spanning Student token is not a chain candidate: "
                f"student_id={realized_student_id} "
                f"prefix_bytes={len(prefix)} "
                f"first_remaining={first_remaining} "
                f"realized_piece_bytes={len(realized_piece) if realized_piece else 0}"
            )
        candidate_ids.append(realized_student_id)

    teacher_ends: list[int] = []
    total = 0
    for value in teacher_pieces:
        total += len(value)
        teacher_ends.append(total)
    candidates: list[dict[str, Any]] = []
    for student_id in candidate_ids:
        student_piece = artifact.student_bytes[student_id]
        assert student_piece is not None
        combined_end = len(prefix) + len(student_piece)
        fully_covered = bisect_left(teacher_ends, combined_end)
        factor_positions = list(range(target_position, target_position + fully_covered))
        factor_token_ids = teacher_ids[:fully_covered]
        residual_start = teacher_ends[fully_covered - 1] if fully_covered else 0
        residual_size = combined_end - residual_start
        residual_prefix = (
            teacher_pieces[fully_covered][:residual_size]
            if residual_size and fully_covered < len(teacher_pieces)
            else b""
        )
        candidates.append(
            {
                "student_token_id": student_id,
                "student_bytes": student_piece,
                "factor_positions": factor_positions,
                "factor_token_ids": factor_token_ids,
                "residual_position": target_position + fully_covered if residual_prefix else None,
                "residual_prefix": residual_prefix,
            }
        )
    candidates.sort(key=lambda item: (len(item["student_bytes"]), item["student_bytes"]))
    return {
        "target_position": target_position,
        "condition_prefix": prefix,
        "candidates": candidates,
    }


def spanning_correction_updates(
    spec: Mapping[str, Any],
    *,
    token_prob: Any,
    prefix_mass: Any,
    base_route: int,
) -> tuple[list[int], list[float], float]:
    """Evaluate Equation 6 and return mass-neutral sparse updates."""

    condition_prefix = bytes(spec["condition_prefix"])
    condition_mass = (
        float(prefix_mass(int(spec["target_position"]), condition_prefix))
        if condition_prefix
        else 1.0
    )
    if not condition_mass > 0:
        raise RuntimeError("BPM spanning condition has zero Teacher prefix mass")
    chain_values: list[float] = []
    for candidate in spec["candidates"]:
        value = 1.0
        for position, token_id in zip(
            candidate["factor_positions"], candidate["factor_token_ids"], strict=True
        ):
            value *= float(token_prob(int(position), int(token_id)))
        value /= condition_mass
        if candidate["residual_prefix"]:
            value *= float(
                prefix_mass(
                    int(candidate["residual_position"]),
                    bytes(candidate["residual_prefix"]),
                )
            )
        chain_values.append(value)
    if any(not (value >= 0.0 and math.isfinite(value)) for value in chain_values):
        raise RuntimeError("BPM spanning chain produced an invalid probability")
    # These are nested events, so their exact masses are non-increasing.  Small
    # independent reduction errors must not create a negative difference cell;
    # snap only roundoff-sized overshoots to the shallower mass and retain the
    # assertion for a material violation of the Equation 6 construction.
    for index in range(1, len(chain_values)):
        shallower = chain_values[index - 1]
        deeper = chain_values[index]
        if deeper <= shallower:
            continue
        tolerance = max(1e-6, 2e-5 * max(abs(shallower), abs(deeper)))
        if deeper - shallower > tolerance:
            raise RuntimeError(
                "BPM nested spanning chain masses are not monotone: "
                f"shallower={shallower:.9g}, deeper={deeper:.9g}, "
                f"overshoot={deeper - shallower:.9g}"
            )
        chain_values[index] = shallower

    ids = [int(candidate["student_token_id"]) for candidate in spec["candidates"]]
    deltas = [
        chain_values[index] - (chain_values[index + 1] if index + 1 < len(chain_values) else 0.0)
        for index in range(len(chain_values))
    ]
    residual_delta = 0.0
    if base_route >= 0:
        ids.append(int(base_route))
        deltas.append(-chain_values[0])
    elif base_route == RESIDUAL_ROUTE:
        residual_delta = -chain_values[0]
    else:
        raise RuntimeError("BPM spanning head routed to an incompatible Teacher cell")
    if abs(sum(deltas) + residual_delta) > 1e-5:
        raise RuntimeError("BPM spanning correction is not mass-neutral")
    return ids, deltas, residual_delta


def project_teacher_probabilities(
    teacher_probs: Sequence[float],
    teacher_to_student: Sequence[int],
    *,
    student_vocab_size: int,
) -> tuple[list[float], float]:
    """CPU oracle for the mass-preserving BPM scatter."""

    if len(teacher_probs) != len(teacher_to_student):
        raise ValueError("teacher probabilities and routes must have equal length")
    projected = [0.0] * int(student_vocab_size)
    residual = 0.0
    for probability, route in zip(teacher_probs, teacher_to_student, strict=True):
        probability = float(probability)
        if not math.isfinite(probability) or probability < 0:
            raise ValueError("teacher probabilities must be finite and non-negative")
        route = int(route)
        if route == RESIDUAL_ROUTE:
            residual += probability
        elif 0 <= route < student_vocab_size:
            projected[route] += probability
        else:
            raise ValueError(f"BPM route {route} is outside the Student vocabulary")
    input_mass = sum(float(value) for value in teacher_probs)
    output_mass = sum(projected) + residual
    if not math.isclose(input_mass, output_mass, rel_tol=1e-7, abs_tol=1e-7):
        raise AssertionError(f"BPM projection lost mass: input={input_mass}, output={output_mass}")
    return projected, residual


def compact_projected_distribution(
    projected: Sequence[float],
    residual: float,
    *,
    top_k: int,
) -> dict[str, Any]:
    """Select Top-K only after full BPM projection and add one complement."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    selected = sorted(
        ((float(probability), token_id) for token_id, probability in enumerate(projected)),
        key=lambda item: (-item[0], item[1]),
    )[: min(top_k, len(projected))]
    ids = [token_id for _, token_id in selected]
    probs = [probability for probability, _ in selected]
    total = sum(float(value) for value in projected) + float(residual)
    complement = max(0.0, total - sum(probs))
    if not math.isclose(sum(probs) + complement, total, rel_tol=1e-7, abs_tol=1e-7):
        raise AssertionError("BPM compact distribution is not mass preserving")
    return {
        "target_ids": ids,
        "target_probs": probs,
        "complement_prob": complement,
        "unmatched_prob": float(residual),
    }


def generalized_jsd(
    target: Sequence[float],
    student: Sequence[float],
    *,
    beta: float = 0.5,
    skew: float = 0.1,
) -> float:
    """Dependency-free BPM divergence oracle, including paper endpoints."""

    if len(target) != len(student) or not target:
        raise ValueError("target and student distributions must have equal non-zero length")
    t = [float(value) for value in target]
    p = [float(value) for value in student]
    if any(value < 0 or not math.isfinite(value) for value in (*t, *p)):
        raise ValueError("distributions must be finite and non-negative")
    if not math.isclose(sum(t), 1.0, abs_tol=1e-6) or not math.isclose(sum(p), 1.0, abs_tol=1e-6):
        raise ValueError("distributions must sum to one")

    def kl(left: Sequence[float], right: Sequence[float]) -> float:
        value = 0.0
        for lhs, rhs in zip(left, right, strict=True):
            if lhs == 0:
                continue
            if rhs <= 0:
                return math.inf
            value += lhs * math.log(lhs / rhs)
        return value

    if beta == 0:
        return kl(t, p)
    if beta == 1:
        mixture = [(1.0 - skew) * lhs + skew * rhs for lhs, rhs in zip(t, p, strict=True)]
        return kl(p, mixture)
    if not 0 < beta < 1:
        raise ValueError("beta must be in [0, 1]")
    mixture = [beta * lhs + (1.0 - beta) * rhs for lhs, rhs in zip(t, p, strict=True)]
    return beta * kl(t, mixture) + (1.0 - beta) * kl(p, mixture)
