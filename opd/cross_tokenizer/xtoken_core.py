"""X-Token projection matrix construction and coverage audit (CPU, one-time).

Implements the tokenizer-level machinery of Sreenivas et al., "X-Token:
Projection-Guided Cross-Tokenizer Knowledge Distillation" (arXiv 2605.21699):

1. canonicalization rules (Appendix, "Canonicalization rules");
2. the two-pass sparse projection matrix W (exact-match pass, then the
   multi-token rule with exponential decay beta*gamma^i, row normalization,
   top-4 truncation);
3. the per-category coverage audit used to select P-KL vs H-KL.

This module is pure CPU and tokenizer-agnostic (get_vocab / decode / encode).
It does not implement the training losses; see ``xtoken.py`` for those.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "DEFAULT_BETA",
    "DEFAULT_GAMMA",
    "DEFAULT_MAX_SPAN",
    "DEFAULT_TOP_K",
    "ProjectionMatrix",
    "canonicalize_token_text",
    "build_projection_matrix",
    "coverage_audit",
]

DEFAULT_BETA = 0.9
DEFAULT_GAMMA = 0.1
DEFAULT_MAX_SPAN = 4
DEFAULT_TOP_K = 4

_BYTE_FALLBACK_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def canonicalize_token_text(text: str) -> str:
    """Apply X-Token's canonicalization rules to one decoded token string.

    Rules (applied in order, idempotent):
      1. space prefix unification: Ġ / _ / ␣ -> a single leading space;
      2. newline unification: Ċ / escaped "\\n" / literal newline -> "\\n";
      3. byte-fallback tokens: "<0xHH>" -> the literal byte character;
      4. leading whitespace+punctuation pairs: "Ġ," / "Ġ." / ... normalize
         to the punctuation alone;
      5. special tokens are NOT mapped here; callers exclude them via
         explicit special-token handling.
    """

    if not text:
        return text
    match = _BYTE_FALLBACK_RE.match(text)
    if match:
        return chr(int(match.group(1), 16))
    out = text
    # Newline unification first so a leading-space rule cannot hide it.
    out = out.replace("Ċ", "\n")
    # Space prefix unification (leading position only, one marker).
    for marker in ("Ġ", "▁", "_", "␣"):
        if out.startswith(marker):
            out = " " + out[len(marker):]
            break
    # Escaped newline unification.
    out = out.replace("\\n", "\n")
    # Leading whitespace+punctuation pairs collapse to the punctuation alone
    # when the token is exactly whitespace plus one punctuation character.
    if len(out) == 2 and out[0] == " " and out[1] in ",.:;!?)]}":
        out = out[1]
    return out


def _decode_one(tokenizer: Any, token_id: int) -> str:
    kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    try:
        return tokenizer.decode([int(token_id)], **kwargs)
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces")
        try:
            return tokenizer.decode([int(token_id)], **kwargs)
        except TypeError:
            return tokenizer.decode([int(token_id)])


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    return [int(value) for value in ids]


def _special_token_ids(tokenizer: Any) -> set[int]:
    special: set[int] = set()
    getter = getattr(tokenizer, "get_added_vocabulary", None)
    if callable(getter):
        try:
            special.update(int(v) for v in getter().values())
        except Exception:  # pragma: no cover - tokenizer-specific failure
            pass
    for name in ("all_special_ids",):
        value = getattr(tokenizer, name, None)
        if value:
            special.update(int(v) for v in value)
    return special


@dataclass(frozen=True)
class ProjectionMatrix:
    """Sparse top-K projection matrix W in R^{|V_S| x |V_T|}.

    ``rows[s]`` is a tuple of ``(teacher_id, weight)`` pairs sorted by
    descending weight, truncated to ``top_k`` entries.  Pass-1 exact matches
    carry weight 1.0 before truncation; multi-token rows are row-normalized.
    """

    student_vocab_size: int
    teacher_vocab_size: int
    top_k: int
    beta: float
    gamma: float
    rows: dict[int, tuple[tuple[int, float], ...]] = field(default_factory=dict)
    exact_student_ids: frozenset[int] = frozenset()
    exact_teacher_ids: frozenset[int] = frozenset()

    def row(self, student_id: int) -> tuple[tuple[int, float], ...]:
        return self.rows.get(int(student_id), ())

    def top1(self, student_id: int) -> int | None:
        entries = self.row(student_id)
        return entries[0][0] if entries else None

    def dense_row(self, student_id: int) -> dict[int, float]:
        return dict(self.row(student_id))


def build_projection_matrix(
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    *,
    top_k: int = DEFAULT_TOP_K,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    max_span: int = DEFAULT_MAX_SPAN,
) -> ProjectionMatrix:
    """Construct the sparse top-K projection matrix W (two passes).

    Pass 1 (exact match): canonicalized string-identical (s, t) pairs get
    W[s, t] = 1.  Pass 2 (multi-token rule): each remaining student token is
    decoded and re-tokenized under the teacher tokenizer, yielding
    (tau_0, ..., tau_l) with weights beta*gamma**i, row-normalized.  Mappings
    longer than ``max_span`` are discarded, as in the paper's Appendix
    algorithm. Every row is finally truncated to its top-``top_k`` entries
    (post-truncation rows may sum to less than 1, matching the paper).
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if isinstance(max_span, bool) or int(max_span) != max_span or int(max_span) <= 0:
        raise ValueError("max_span must be a positive integer")
    max_span = int(max_span)
    if not (0.0 < gamma < 1.0) or not (0.0 < beta <= 1.0):
        raise ValueError("beta/gamma must satisfy 0 < gamma < 1 and 0 < beta <= 1")

    student_vocab = {str(t): int(i) for t, i in student_tokenizer.get_vocab().items()}
    teacher_vocab = {str(t): int(i) for t, i in teacher_tokenizer.get_vocab().items()}
    student_special = _special_token_ids(student_tokenizer)
    teacher_special = _special_token_ids(teacher_tokenizer)

    # Pass 1: exact canonical match.  Hash on canonicalized decoded strings.
    # The paper's double loop admits every (s, t) pair whose decoded strings
    # match after canonicalization, so several spelling variants of one
    # canonical string all enter the row at weight 1.0 (implemented as a
    # hashmap from the canonical string to the id list on each side).
    teacher_canonical: dict[str, list[int]] = {}
    for token, teacher_id in teacher_vocab.items():
        if teacher_id in teacher_special:
            continue
        canonical = canonicalize_token_text(_decode_one(teacher_tokenizer, teacher_id))
        teacher_canonical.setdefault(canonical, []).append(teacher_id)

    # Special tokens bypass canonicalization and pair by raw vocabulary
    # string only; this is the paper's explicit special-token mapping in its
    # unambiguous form (chat-template and stop tokens share literal strings
    # across these tokenizer families).
    teacher_special_raw: dict[str, int] = {}
    for token, teacher_id in teacher_vocab.items():
        if teacher_id not in teacher_special:
            continue
        if token not in teacher_special_raw or teacher_id < teacher_special_raw[token]:
            teacher_special_raw[token] = teacher_id

    rows: dict[int, tuple[tuple[int, float], ...]] = {}
    exact_student: set[int] = set()
    exact_teacher: set[int] = set()
    for token, student_id in student_vocab.items():
        if student_id in student_special:
            teacher_id = teacher_special_raw.get(token)
            if teacher_id is not None:
                rows[student_id] = ((teacher_id, 1.0),)
                exact_student.add(student_id)
                exact_teacher.add(teacher_id)
                continue
            # The other tokenizer may not flag the same control string as
            # special; fall through to the canonical pass before giving up.
        canonical = canonicalize_token_text(_decode_one(student_tokenizer, student_id))
        teacher_ids = teacher_canonical.get(canonical)
        if teacher_ids is None and student_id not in student_special:
            # Symmetric fallback: a Student-side regular token may raw-match a
            # Teacher-flagged special string.
            teacher_id = teacher_special_raw.get(token)
            teacher_ids = None if teacher_id is None else [teacher_id]
        if teacher_ids:
            entries = tuple(
                (teacher_id, 1.0 / len(teacher_ids))
                for teacher_id in sorted(teacher_ids)
            )
            rows[student_id] = entries[:top_k]
            exact_student.add(student_id)
            exact_teacher.update(teacher_ids)

    # Pass 2: multi-token rule for student tokens without an exact match.
    for token, student_id in student_vocab.items():
        if student_id in student_special or student_id in exact_student:
            continue
        text = _decode_one(student_tokenizer, student_id)
        if not text:
            continue
        teacher_ids = [
            teacher_id
            for teacher_id in _encode_text(teacher_tokenizer, text)
            if teacher_id not in teacher_special
        ]
        if not teacher_ids or len(teacher_ids) > max_span:
            continue
        weights = [beta * (gamma **i) for i in range(len(teacher_ids))]
        total = sum(weights)
        entries = sorted(
            ((teacher_id, weight / total) for teacher_id, weight in zip(teacher_ids, weights)),
            key=lambda item: (-item[1], item[0]),
        )
        # Merge duplicate teacher ids (re-tokenization can repeat ids) by
        # summing their weights before truncation.
        merged: dict[int, float] = {}
        for teacher_id, weight in entries:
            merged[teacher_id] = merged.get(teacher_id, 0.0) + weight
        rows[student_id] = tuple(
            sorted(merged.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        )

    # Truncate exact rows as well (they are already single-entry) and apply
    # the global top-K truncation uniformly.
    rows = {sid: tuple(entries[:top_k]) for sid, entries in rows.items() if entries}

    return ProjectionMatrix(
        student_vocab_size=len(student_vocab),
        teacher_vocab_size=len(teacher_vocab),
        top_k=top_k,
        beta=beta,
        gamma=gamma,
        rows=rows,
        exact_student_ids=frozenset(exact_student),
        exact_teacher_ids=frozenset(exact_teacher),
    )


_TOKEN_CATEGORY_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("digit_1", re.compile(r"^\s?\d$")),
    ("digit_2", re.compile(r"^\s?\d{2}$")),
    ("digit_3plus", re.compile(r"^\s?\d{3,}$")),
    ("alpha", re.compile(r"^\s?[A-Za-z']+$")),
    ("punct", re.compile(r"^\s?[.,:;!?()\[\]{}<>'\"`~@#$%^&*+=/\\|-]+$")),
)
_MULTIBYTE_RE = re.compile(r"[^\x00-\x7F]")


def categorize_token_text(text: str) -> str:
    if _MULTIBYTE_RE.search(text):
        return "multibyte"
    for name, pattern in _TOKEN_CATEGORY_RES:
        if pattern.match(text):
            return name
    return "other"


def coverage_audit(
    student_tokenizer: Any,
    common_student_ids: Iterable[int],
) -> dict[str, dict[str, int | float]]:
    """Per-category retention of the student vocabulary in the common set.

    Mirrors X-Token's coverage analysis used to select P-KL vs H-KL:
    categories are digits by length, alphabetic, punctuation, multi-byte /
    non-ASCII, and other.  Returns ``{category: {total, covered, ratio}}``.
    """

    common = {int(value) for value in common_student_ids}
    special = _special_token_ids(student_tokenizer)
    stats: dict[str, list[int]] = {}
    for token, student_id in student_tokenizer.get_vocab().items():
        student_id = int(student_id)
        if student_id in special:
            continue
        category = categorize_token_text(canonicalize_token_text(_decode_one(student_tokenizer, student_id)))
        total, covered = stats.get(category, [0, 0])
        stats[category] = [total + 1, covered + (1 if student_id in common else 0)]
    return {
        category: {
            "total": total,
            "covered": covered,
            "ratio": (covered / total) if total else 0.0,
        }
        for category, (total, covered) in sorted(stats.items())
    }
