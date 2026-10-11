"""Pure byte-event primitives for Event-Set Completion Distillation (ESCD).

The production plugin keeps tokenizer work off the model hot path.  This file
contains the deterministic CPU part of the method: a prefix-free quotient of
non-shared Teacher Top-K tokens, detection of a realized Student prefix, and
construction of the complete one-token success set for the remaining bytes.
It deliberately contains no Slime, Transformers, CUDA, or distributed code.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .bpm_core import BPMRoutingArtifact, RESIDUAL_ROUTE


@dataclass(frozen=True)
class ESCDChildEvent:
    """One naturally visited depth-one event-completion target."""

    mass: float
    target_bytes: bytes
    remaining_bytes: bytes
    source_token_ids: tuple[int, ...]
    success_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ESCDChildCompilation:
    """Child events plus mass accounting for one realized root action."""

    events: tuple[ESCDChildEvent, ...]
    nonshared_mass: float
    hit_mass: float
    supported_mass: float
    deeper_only_mass: float
    invalid_mass: float


@dataclass(frozen=True)
class FullVocabularyESCDPlan:
    """Tokenizer-only plan for the exact full-vocabulary ESCD ablation.

    ``teacher_component_ids`` maps each Teacher vocabulary coordinate to one
    prefix-quotient component, or ``-1`` when the coordinate is shared,
    special, or losslessly undecodable.  The plan is independent of model
    probabilities and is therefore compiled once per tokenizer pair.
    """

    component_targets: tuple[bytes, ...]
    teacher_component_ids: tuple[int, ...]
    sorted_component_targets: tuple[bytes, ...]
    sorted_component_ids: tuple[int, ...]

    def components_after(self, first_piece: bytes) -> tuple[int, ...]:
        """Components strictly extending one realized Student root token."""

        if not first_piece:
            return ()
        start = bisect_left(self.sorted_component_targets, first_piece)
        upper = _prefix_upper_bound(first_piece)
        end = (
            len(self.sorted_component_targets)
            if upper is None
            else bisect_left(self.sorted_component_targets, upper)
        )
        return tuple(
            self.sorted_component_ids[index]
            for index in range(start, end)
            if len(self.sorted_component_targets[index]) > len(first_piece)
            and self.sorted_component_targets[index].startswith(first_piece)
        )


def compile_full_vocabulary_plan(
    *,
    artifact: BPMRoutingArtifact,
    student_index: "StudentBytePrefixIndex",
) -> FullVocabularyESCDPlan:
    """Compile the full non-shared Teacher prefix quotient in linear byte time.

    The Top-K reference implementation constructs a pairwise prefix graph,
    which is appropriate for K=128 but quadratic in K.  For the full
    vocabulary, processing unique surfaces from shortest to longest lets a
    trie assign every surface to its shortest selected prefix.  This produces
    exactly the same connected components without an O(|V|^2) scan.
    """

    exact_student_surfaces = student_index.exact_surfaces
    teacher_stops = set(artifact.teacher_stop_ids)
    surface_to_token_ids: dict[bytes, list[int]] = {}
    for token_id, piece in enumerate(artifact.teacher_bytes):
        if token_id in teacher_stops or not piece:
            continue
        value = bytes(piece)
        if value in exact_student_surfaces:
            continue
        surface_to_token_ids.setdefault(value, []).append(token_id)

    # Each trie node is [children, component_id].  A terminal component ID is
    # inherited by every longer selected surface below it.
    trie: dict[int | None, object] = {None: None}
    component_targets: list[bytes] = []
    surface_component: dict[bytes, int] = {}
    for surface in sorted(surface_to_token_ids, key=lambda value: (len(value), value)):
        node = trie
        inherited: int | None = None
        for byte in surface:
            terminal = node.get(None)
            if terminal is not None and inherited is None:
                inherited = int(terminal)
            child = node.get(byte)
            if child is None:
                child = {None: None}
                node[byte] = child
            node = child  # type: ignore[assignment]
        terminal = node.get(None)
        if terminal is not None and inherited is None:
            inherited = int(terminal)
        if inherited is None:
            inherited = len(component_targets)
            component_targets.append(surface)
        node[None] = inherited
        surface_component[surface] = inherited

    teacher_component_ids = [-1] * len(artifact.teacher_bytes)
    for surface, token_ids in surface_to_token_ids.items():
        component_id = surface_component[surface]
        for token_id in token_ids:
            teacher_component_ids[token_id] = component_id

    sorted_entries = sorted(
        (target, component_id)
        for component_id, target in enumerate(component_targets)
    )
    return FullVocabularyESCDPlan(
        component_targets=tuple(component_targets),
        teacher_component_ids=tuple(teacher_component_ids),
        sorted_component_targets=tuple(target for target, _ in sorted_entries),
        sorted_component_ids=tuple(component_id for _, component_id in sorted_entries),
    )


def prefix_components(pieces: Sequence[bytes]) -> tuple[tuple[int, ...], ...]:
    """Connected components under lossless byte-prefix comparability.

    A component is represented by its shortest byte target.  Components are
    pairwise prefix-incomparable, so the resulting event ledger does not charge
    overlapping Teacher prefix cylinders as independent categorical events.
    """

    if any(not isinstance(piece, bytes) or not piece for piece in pieces):
        raise ValueError("ESCD prefix quotient requires non-empty byte strings")
    parents = list(range(len(pieces)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left, left_piece in enumerate(pieces):
        for right in range(left + 1, len(pieces)):
            right_piece = pieces[right]
            if left_piece.startswith(right_piece) or right_piece.startswith(left_piece):
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(pieces)):
        groups.setdefault(find(index), []).append(index)
    return tuple(
        tuple(values)
        for _, values in sorted(groups.items(), key=lambda item: min(item[1]))
    )


def _prefix_upper_bound(prefix: bytes) -> bytes | None:
    value = bytearray(prefix)
    for index in range(len(value) - 1, -1, -1):
        if value[index] != 0xFF:
            value[index] += 1
            return bytes(value[: index + 1])
    return None


class StudentBytePrefixIndex:
    """Lossless Student byte surfaces indexed for exact prefix queries."""

    def __init__(
        self,
        token_bytes: Sequence[bytes | None],
        *,
        excluded_token_ids: Iterable[int] = (),
    ) -> None:
        excluded = {int(value) for value in excluded_token_ids}
        entries = sorted(
            (bytes(piece), int(token_id))
            for token_id, piece in enumerate(token_bytes)
            if token_id not in excluded and piece
        )
        self._pieces = tuple(piece for piece, _ in entries)
        self._token_ids = tuple(token_id for _, token_id in entries)
        by_surface: dict[bytes, list[int]] = {}
        for piece, token_id in entries:
            by_surface.setdefault(piece, []).append(token_id)
        self._by_surface = {
            piece: tuple(token_ids) for piece, token_ids in by_surface.items()
        }
        self._exact_surfaces = frozenset(self._by_surface)

    @property
    def exact_surfaces(self) -> frozenset[bytes]:
        return self._exact_surfaces

    def completion_token_ids(self, prefix: bytes) -> tuple[int, ...]:
        """Return every legal token whose bytes complete or cross ``prefix``."""

        if not prefix:
            raise ValueError("ESCD completion prefix must be non-empty")
        start = bisect_left(self._pieces, prefix)
        upper = _prefix_upper_bound(prefix)
        end = len(self._pieces) if upper is None else bisect_left(self._pieces, upper)
        return tuple(
            self._token_ids[index]
            for index in range(start, end)
            if self._pieces[index].startswith(prefix)
        )

    def has_strict_continuation(self, target: bytes) -> bool:
        """Whether a legal token advances into ``target`` without completing it."""

        return any(target[:length] in self._by_surface for length in range(1, len(target)))


def _component_target(pieces: Sequence[bytes]) -> bytes:
    target = min((bytes(value) for value in pieces), key=lambda value: (len(value), value))
    if any(not value.startswith(target) for value in pieces):
        raise ValueError("ESCD prefix component has no unique shortest target")
    return target


def compile_child_events(
    *,
    teacher_token_ids: Sequence[int],
    teacher_probabilities: Sequence[float],
    actual_first_token_id: int,
    artifact: BPMRoutingArtifact,
    student_index: StudentBytePrefixIndex,
    min_event_mass: float = 0.0,
) -> ESCDChildCompilation:
    """Compile the depth-one ESCD target for one normally visited child state.

    Exact-shared Teacher surfaces are removed before the prefix quotient.  The
    realized first Student token is used only as an on-policy reach condition;
    no inverse-propensity factor is introduced.  Event masses therefore remain
    the Teacher Top-K masses observed at the root state.
    """

    if len(teacher_token_ids) != len(teacher_probabilities):
        raise ValueError("ESCD Teacher IDs and probabilities must have equal lengths")
    if not math.isfinite(float(min_event_mass)) or float(min_event_mass) < 0.0:
        raise ValueError("ESCD min_event_mass must be finite and non-negative")
    actual_first_token_id = int(actual_first_token_id)
    if not 0 <= actual_first_token_id < len(artifact.student_bytes):
        raise ValueError("ESCD realized Student token is outside its vocabulary")
    first_piece = artifact.student_bytes[actual_first_token_id]
    if actual_first_token_id in set(artifact.student_stop_ids) or not first_piece:
        return ESCDChildCompilation((), 0.0, 0.0, 0.0, 0.0, 0.0)

    exact_student_surfaces = student_index.exact_surfaces
    teacher_stops = set(artifact.teacher_stop_ids)
    selected: list[tuple[int, float, bytes]] = []
    invalid_mass = 0.0
    seen_teacher_ids: set[int] = set()
    for raw_token_id, raw_probability in zip(
        teacher_token_ids, teacher_probabilities, strict=True
    ):
        token_id = int(raw_token_id)
        probability = float(raw_probability)
        if token_id in seen_teacher_ids:
            raise ValueError("ESCD Teacher Top-K contains duplicate token IDs")
        seen_teacher_ids.add(token_id)
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError("ESCD Teacher Top-K contains invalid probability mass")
        piece = (
            artifact.teacher_bytes[token_id]
            if 0 <= token_id < len(artifact.teacher_bytes)
            else None
        )
        if token_id in teacher_stops or not piece:
            invalid_mass += probability
            continue
        value = bytes(piece)
        if value in exact_student_surfaces:
            continue
        if probability > 0.0:
            selected.append((token_id, probability, value))

    nonshared_mass = math.fsum(probability for _, probability, _ in selected)
    components = prefix_components([value for _, _, value in selected]) if selected else ()
    events: list[ESCDChildEvent] = []
    hit_mass = 0.0
    supported_mass = 0.0
    deeper_only_mass = 0.0
    for component in components:
        values = [selected[index] for index in component]
        target = _component_target([value for _, _, value in values])
        mass = math.fsum(probability for _, probability, _ in values)
        if mass < float(min_event_mass):
            continue
        # Equal means the event was already completed by the root action;
        # incompatibility means this rollout never entered the event branch.
        if not target.startswith(first_piece) or len(target) <= len(first_piece):
            continue
        hit_mass += mass
        remaining = target[len(first_piece) :]
        success_ids = student_index.completion_token_ids(remaining)
        if success_ids:
            events.append(
                ESCDChildEvent(
                    mass=mass,
                    target_bytes=target,
                    remaining_bytes=remaining,
                    source_token_ids=tuple(sorted(token_id for token_id, _, _ in values)),
                    success_token_ids=success_ids,
                )
            )
            supported_mass += mass
        elif student_index.has_strict_continuation(remaining):
            deeper_only_mass += mass

    return ESCDChildCompilation(
        events=tuple(events),
        nonshared_mass=nonshared_mass,
        hit_mass=hit_mass,
        supported_mass=supported_mass,
        deeper_only_mass=deeper_only_mass,
        invalid_mass=invalid_mass,
    )


def project_root_topk(
    *,
    teacher_token_ids: Sequence[int],
    teacher_probabilities: Sequence[float],
    artifact: BPMRoutingArtifact,
    student_stop_token_id: int | None = None,
    terminal_stop: bool = False,
) -> tuple[tuple[int, ...], tuple[float, ...], float]:
    """Project Teacher Top-K mass to BPM's verified root cells plus complement."""

    if len(teacher_token_ids) != len(teacher_probabilities):
        raise ValueError("ESCD root IDs and probabilities must have equal lengths")
    teacher_stops = set(artifact.teacher_stop_ids)
    mapped: dict[int, float] = {}
    seen: set[int] = set()
    selected_mass = 0.0
    for raw_token_id, raw_probability in zip(
        teacher_token_ids, teacher_probabilities, strict=True
    ):
        token_id = int(raw_token_id)
        probability = float(raw_probability)
        if token_id in seen:
            raise ValueError("ESCD root Top-K contains duplicate token IDs")
        seen.add(token_id)
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError("ESCD root Top-K contains invalid probability mass")
        selected_mass += probability
        if token_id in teacher_stops and student_stop_token_id is not None:
            route = int(student_stop_token_id)
        elif terminal_stop:
            route = RESIDUAL_ROUTE
        elif 0 <= token_id < len(artifact.teacher_to_student):
            route = int(artifact.teacher_to_student[token_id])
        else:
            route = RESIDUAL_ROUTE
        if route >= 0:
            mapped[route] = mapped.get(route, 0.0) + probability

    if selected_mass > 1.0 + 2e-5:
        raise FloatingPointError("ESCD Teacher Top-K mass exceeds one")
    # The causal-hidden head is evaluated in bf16 and returned as log
    # probabilities.  Exponentiating Top-K values in float32 can make their
    # sum exceed one by a few ulps.  That is representation error rather than
    # probability mass, so renormalize only this tiny overshoot.  A material
    # overshoot was rejected above and still fails loudly.
    normalization = max(1.0, selected_mass)
    ordered = sorted(
        (token_id, probability / normalization)
        for token_id, probability in mapped.items()
    )
    explicit_mass = math.fsum(value for _, value in ordered)
    complement = max(0.0, 1.0 - explicit_mass)
    total = explicit_mass + complement
    if not math.isclose(total, 1.0, rel_tol=2e-6, abs_tol=2e-6):
        raise FloatingPointError("ESCD root projection does not preserve unit mass")
    return (
        tuple(token_id for token_id, _ in ordered),
        tuple(probability for _, probability in ordered),
        complement,
    )
