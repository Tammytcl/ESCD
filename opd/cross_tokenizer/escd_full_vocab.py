"""Exact full-Teacher-vocabulary ablation of ESCD.

This plugin changes only ESCD's candidate support.  Rollout sampling, the
Student OPD loss, byte alignment, natural child visitation, and the unit child
coefficient are inherited from :mod:`opd.cross_tokenizer.escd`.
"""

from __future__ import annotations

import math

import torch

from .base import CrossTokenizerCapabilities, FusedScoringContext
from .bpm_dense import bpm_projected_divergence_from_log_probs
from .escd import ESCDCrossTokenizer
from .escd_core import (
    ESCDChildEvent,
    FullVocabularyESCDPlan,
    compile_full_vocabulary_plan,
)
from .registry import register_cross_tokenizer


_FULL_PLAN_CACHE: dict[str, FullVocabularyESCDPlan] = {}


@register_cross_tokenizer("escd_full_vocab")
class ESCDFullVocabularyCrossTokenizer(ESCDCrossTokenizer):
    """ESCD ablation whose Teacher candidate set is the complete vocabulary."""

    capabilities = CrossTokenizerCapabilities(
        fidelity="proposed-method-ablation",
        reference="ESCD full-Teacher-vocabulary candidate-support ablation",
        candidate_top_k_mode="zero",
        default_candidate_top_k=0,
        teacher_backends=("sglang",),
        required_distribution_methods=(
            "full_probabilities",
            "selected_log_probs",
            "projected_log_probs",
        ),
        public_temperature_mode="one",
        supports_context_parallel=True,
        context_parallel_backends=("sglang",),
        teacher_transports=(("sglang", "causal_hidden"),),
        requires_loss_token_normalization=True,
        requires_full_response_loss_mask=True,
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

    def _full_plan(self) -> FullVocabularyESCDPlan:
        artifact = self._artifact()
        cached = _FULL_PLAN_CACHE.get(artifact.sha256)
        if cached is None:
            cached = compile_full_vocabulary_plan(
                artifact=artifact,
                student_index=self._student_index(),
            )
            _FULL_PLAN_CACHE[artifact.sha256] = cached
        return cached

    def _root_projection_plan(
        self, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
        artifact = self._artifact()
        support = tuple(
            sorted(
                {int(route) for route in artifact.teacher_to_student if int(route) >= 0}
                | {int(token_id) for token_id in artifact.student_stop_ids}
            )
        )
        slot_by_token = {token_id: slot for slot, token_id in enumerate(support)}
        complement_slot = len(support)
        route_slots = torch.full(
            (len(artifact.teacher_bytes),),
            complement_slot,
            device=device,
            dtype=torch.long,
        )
        for teacher_id, route in enumerate(artifact.teacher_to_student):
            if int(route) >= 0:
                route_slots[teacher_id] = slot_by_token[int(route)]
        support_ids = torch.tensor(support, device=device, dtype=torch.long)
        return support_ids, route_slots, slot_by_token

    def compute_fused_token_losses(self, context: FusedScoringContext) -> torch.Tensor:
        artifact = self._artifact()
        mask_whitespace, teacher_chunk, student_chunk, min_event_mass = self._escd_settings()
        if int(context.candidate_top_k) != 0:
            raise ValueError("escd_full_vocab requires candidate_top_k=0")
        if not math.isclose(float(context.student.temperature), 1.0):
            raise ValueError("escd_full_vocab requires cross_tokenizer.temperature=1.0")
        if context.teacher_distribution.length != context.alignment.teacher_length:
            raise ValueError("Full-vocabulary ESCD Teacher rows do not match the alignment")
        if context.gather_student_rows is None:
            raise ValueError("Full-vocabulary ESCD requires the generic CP row gather")

        device = context.student.response_token_ids.device
        local_positions = context.local_response_positions.to(
            device=device, dtype=torch.long
        ).flatten()
        if local_positions.numel() != context.student.distribution.length:
            raise ValueError("Full-vocabulary ESCD local positions do not match Student rows")
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
        root_output = torch.zeros(
            context.student.distribution.length, dtype=torch.float32, device=device
        )
        root_nonshared_event_mass = torch.zeros_like(root_output)
        compiled_child_row = torch.zeros_like(root_output)
        supported_event_mass = torch.zeros_like(root_output)
        nonshared_event_mass = torch.zeros_like(root_output)

        full_plan = self._full_plan()
        support_ids, route_slots, root_slot_by_token = self._root_projection_plan(
            device=device
        )
        component_count = len(full_plan.component_targets)
        discard_component = component_count
        component_slots = torch.tensor(
            [
                discard_component if component_id < 0 else int(component_id)
                for component_id in full_plan.teacher_component_ids
            ],
            device=device,
            dtype=torch.long,
        )
        teacher_stop_ids = tuple(int(value) for value in artifact.teacher_stop_ids)
        local_events: dict[int, tuple[tuple[int, float], ...]] = {}

        for start in range(0, len(root_rows), teacher_chunk):
            chunk = root_rows[start : start + teacher_chunk]
            teacher_positions = torch.tensor(
                [int(row["teacher_position"]) for row in chunk],
                device=device,
                dtype=torch.long,
            )
            teacher_probs = context.teacher_distribution.rows(
                teacher_positions
            ).full_probabilities(
                vocab_size=len(artifact.teacher_bytes), temperature=1.0
            ).detach().float()
            if teacher_probs.shape != (len(chunk), len(artifact.teacher_bytes)):
                raise ValueError("Full-vocabulary ESCD received an invalid Teacher probability shape")
            probability_sums = teacher_probs.sum(dim=-1)
            if not bool(torch.isfinite(teacher_probs).all()) or not bool(
                torch.allclose(
                    probability_sums,
                    torch.ones_like(probability_sums),
                    atol=3e-5,
                    rtol=3e-5,
                )
            ):
                raise FloatingPointError("Full-vocabulary ESCD Teacher probabilities are invalid")
            # The distributed softmax is normalized mathematically, but its
            # gathered FP32 row can differ from one by a few ulps.  Normalize
            # once here so every downstream partition is a strict probability
            # distribution rather than relying on cancellation between cells.
            teacher_probs = teacher_probs / probability_sums.unsqueeze(-1)

            stop_columns = (
                torch.tensor(teacher_stop_ids, device=device, dtype=torch.long)
                if teacher_stop_ids
                else None
            )
            stop_mass = (
                teacher_probs.index_select(1, stop_columns).sum(dim=-1)
                if stop_columns is not None
                else teacher_probs.new_zeros((len(chunk),))
            )
            # Stop coordinates are routed per realized row below.  Exclude
            # them before the static root scatter instead of first accumulating
            # them into the complement and subtracting their separately summed
            # mass.  With multiple model-declared stop IDs, those two FP32
            # reductions can round differently and create a tiny negative
            # complement (the production GLM-Z1 failure at step 17).
            root_teacher_probs = teacher_probs
            if stop_columns is not None:
                root_teacher_probs = teacher_probs.clone()
                root_teacher_probs.index_fill_(1, stop_columns, 0.0)

            root_target = torch.zeros(
                (len(chunk), support_ids.numel() + 1),
                device=device,
                dtype=torch.float32,
            )
            root_target.scatter_add_(
                1,
                route_slots.unsqueeze(0).expand(len(chunk), -1),
                root_teacher_probs,
            )
            if teacher_stop_ids:
                for row_index, row in enumerate(chunk):
                    # A terminal row is a binary EOS-vs-non-EOS event.  It is
                    # not the ordinary longest-prefix root projection.
                    if str(row.get("kind")) == "stop":
                        root_target[row_index].zero_()
                        student_stop = int(row["student_token_id"])
                        root_target[row_index, root_slot_by_token[student_stop]] = stop_mass[
                            row_index
                        ]
                        # Sum the non-stop coordinates directly.  ``1-stop``
                        # is numerically unsafe when several stop IDs jointly
                        # carry essentially all Teacher mass.
                        root_target[row_index, -1] = root_teacher_probs[row_index].sum()
                        continue
                    student_stop = (
                        int(row["student_stop_token_id"])
                        if "student_stop_token_id" in row
                        else None
                    )
                    if student_stop is None or student_stop not in root_slot_by_token:
                        root_target[row_index, -1] += stop_mass[row_index]
                    else:
                        root_target[row_index, root_slot_by_token[student_stop]] += stop_mass[
                            row_index
                        ]

            root_target_sums = root_target.sum(dim=-1, keepdim=True)
            if not bool(torch.isfinite(root_target).all()) or bool(
                (root_target < 0).any()
            ) or bool((root_target_sums <= 0).any()):
                raise FloatingPointError(
                    "Full-vocabulary ESCD root projection produced an invalid probability row"
                )
            root_target = root_target / root_target_sums

            local_rows = torch.tensor(
                [global_to_local[int(row["student_position"])] for row in chunk],
                device=device,
                dtype=torch.long,
            )
            projected_ids = support_ids.unsqueeze(0).expand(len(chunk), -1)
            projected_mask = torch.ones_like(projected_ids, dtype=torch.bool)
            student_log_probs = context.student.distribution.rows(
                local_rows
            ).projected_log_probs(
                projected_ids,
                projected_mask,
                vocab_size=len(artifact.student_bytes),
                temperature=1.0,
            ).float()
            # Terminal supervision has only two cells: realized Student EOS
            # and every non-EOS token.  The static full-root support contains
            # all ordinary mapped roots, so fold those cells back into the
            # terminal complement before evaluating its FKL.
            for row_index, row in enumerate(chunk):
                if str(row.get("kind")) != "stop":
                    continue
                stop_slot = root_slot_by_token[int(row["student_token_id"])]
                non_stop_columns = [
                    column
                    for column in range(support_ids.numel())
                    if column != stop_slot
                ]
                complement_columns = torch.tensor(
                    non_stop_columns + [support_ids.numel()],
                    device=device,
                    dtype=torch.long,
                )
                student_log_probs[row_index, -1] = torch.logsumexp(
                    student_log_probs[row_index].index_select(
                        0, complement_columns
                    ),
                    dim=0,
                )
            root_losses = bpm_projected_divergence_from_log_probs(
                root_target,
                student_log_probs,
                beta=self._escd_root_beta(),
                skew=0.1,
            )
            root_output.index_add_(0, local_rows, root_losses)

            component_mass = torch.zeros(
                (len(chunk), component_count + 1),
                device=device,
                dtype=torch.float32,
            )
            component_mass.scatter_add_(
                1,
                component_slots.unsqueeze(0).expand(len(chunk), -1),
                teacher_probs,
            )
            for row_index, row in enumerate(chunk):
                local_position = global_to_local[int(row["student_position"])]
                root_nonshared_event_mass[local_position] = component_mass[
                    row_index, :component_count
                ].sum()
                first_token_id = int(row["student_token_id"])
                first_piece = (
                    artifact.student_bytes[first_token_id]
                    if 0 <= first_token_id < len(artifact.student_bytes)
                    else None
                )
                component_ids = (
                    full_plan.components_after(bytes(first_piece)) if first_piece else ()
                )
                if component_ids:
                    ids_tensor = torch.tensor(
                        component_ids, device=device, dtype=torch.long
                    )
                    masses = component_mass[row_index].index_select(0, ids_tensor)
                    local_events[local_position] = tuple(
                        (component_id, float(mass))
                        for component_id, mass in zip(
                            component_ids, masses.detach().cpu().tolist(), strict=True
                        )
                        if float(mass) >= min_event_mass and float(mass) > 0.0
                    )

        # Exchange only the event IDs and masses that can follow the actually
        # realized root token.  In contrast, gathering [response, |V_T|]
        # probabilities would require tens of GiB on a 27K-token trajectory.
        local_counts = torch.zeros(
            (local_positions.numel(), 1), device=device, dtype=torch.long
        )
        for local_position, events in local_events.items():
            local_counts[local_position, 0] = len(events)
        global_counts = context.gather_student_rows(local_counts.detach()).flatten()
        event_width = int(global_counts.max().item()) if global_counts.numel() else 0
        if event_width:
            local_component_ids = torch.full(
                (local_positions.numel(), event_width),
                -1,
                device=device,
                dtype=torch.long,
            )
            # Piggyback the diagnostic denominator on the existing compact
            # component-mass exchange.  Keeping it in a reserved final column
            # avoids introducing another CP collective between the loss
            # forward and backward paths.
            local_component_masses = torch.zeros(
                (local_positions.numel(), event_width + 1),
                device=device,
                dtype=torch.float32,
            )
            for local_position, events in local_events.items():
                count = len(events)
                if count:
                    local_component_ids[local_position, :count] = torch.tensor(
                        [component_id for component_id, _ in events],
                        device=device,
                        dtype=torch.long,
                    )
                    local_component_masses[local_position, :count] = torch.tensor(
                        [mass for _, mass in events],
                        device=device,
                        dtype=torch.float32,
                    )
            local_component_masses[:, event_width] = root_nonshared_event_mass
            global_component_ids = context.gather_student_rows(
                local_component_ids.detach()
            )
            global_component_masses_with_diagnostics = context.gather_student_rows(
                local_component_masses.detach()
            )
            global_component_masses = global_component_masses_with_diagnostics[
                :, :event_width
            ]
            global_root_nonshared_event_mass = global_component_masses_with_diagnostics[
                :, event_width
            ]
        else:
            global_component_ids = global_component_masses = None
            global_root_nonshared_event_mass = None

        student_index = self._student_index()
        child_targets: list[tuple[int, tuple[ESCDChildEvent, ...]]] = []
        for global_position, local_position in sorted(global_to_local.items()):
            previous = rows_by_position.get(global_position - 1)
            if (
                previous is None
                or str(previous.get("kind")) != "aligned"
                or (mask_whitespace and bool(previous.get("whitespace", False)))
            ):
                continue
            count = int(global_counts[global_position - 1].item())
            if count <= 0:
                continue
            assert global_component_ids is not None
            assert global_component_masses is not None
            assert global_root_nonshared_event_mass is not None
            nonshared_event_mass[local_position] = global_root_nonshared_event_mass[
                global_position - 1
            ]
            first_token_id = int(previous["student_token_id"])
            first_piece = artifact.student_bytes[first_token_id]
            if not first_piece:
                continue
            events: list[ESCDChildEvent] = []
            for component_id, mass in zip(
                global_component_ids[global_position - 1, :count].detach().cpu().tolist(),
                global_component_masses[global_position - 1, :count].detach().cpu().tolist(),
                strict=True,
            ):
                target = full_plan.component_targets[int(component_id)]
                remaining = target[len(first_piece) :]
                success_ids = student_index.completion_token_ids(remaining)
                if success_ids:
                    events.append(
                        ESCDChildEvent(
                            mass=float(mass),
                            target_bytes=target,
                            remaining_bytes=remaining,
                            source_token_ids=(),
                            success_token_ids=success_ids,
                        )
                    )
            if events:
                compiled_child_row[local_position] = 1.0
                supported_event_mass[local_position] = sum(
                    event.mass for event in events
                )
                child_targets.append((local_position, tuple(events)))

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
            raise FloatingPointError("Full-vocabulary ESCD produced non-finite token losses")
        return self._anchor_zero(context, output)
