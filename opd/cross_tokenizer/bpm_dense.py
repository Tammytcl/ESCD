"""Memory-bounded dense BPM projection primitives for Slime.

These functions operate on complete Teacher-vocabulary probability rows.  A
caller may process a small row chunk at a time; no function stores a
``[response_length, vocabulary]`` tensor.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .base import FusedScoringContext
from .bpm_core import (
    INCOMPATIBLE_ROUTE,
    RESIDUAL_ROUTE,
    routes_for_bpm_row,
    spanning_chain_spec,
    spanning_correction_updates,
)


@dataclass(frozen=True)
class DenseBPMRoutePlan:
    routes: torch.Tensor
    support_ids: torch.Tensor
    compatible_teacher_ids: torch.Tensor
    mapped_teacher_ids: torch.Tensor
    mapped_cells: torch.Tensor
    residual_teacher_ids: torch.Tensor


def compile_dense_route_plan(
    routes: Sequence[int] | torch.Tensor,
    *,
    device: torch.device,
) -> DenseBPMRoutePlan:
    route_tensor = torch.as_tensor(routes, dtype=torch.long, device=device)
    compatible_teacher_ids = torch.nonzero(
        route_tensor != INCOMPATIBLE_ROUTE, as_tuple=False
    ).flatten()
    mapped_teacher_ids = torch.nonzero(route_tensor >= 0, as_tuple=False).flatten()
    residual_teacher_ids = torch.nonzero(
        route_tensor == RESIDUAL_ROUTE, as_tuple=False
    ).flatten()
    mapped_routes = route_tensor.index_select(0, mapped_teacher_ids)
    support_ids = torch.unique(mapped_routes, sorted=True)
    mapped_cells = torch.searchsorted(support_ids, mapped_routes)
    return DenseBPMRoutePlan(
        routes=route_tensor,
        support_ids=support_ids,
        compatible_teacher_ids=compatible_teacher_ids,
        mapped_teacher_ids=mapped_teacher_ids,
        mapped_cells=mapped_cells,
        residual_teacher_ids=residual_teacher_ids,
    )


def project_dense_teacher_rows(
    teacher_probabilities: torch.Tensor,
    plan: DenseBPMRoutePlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter complete Teacher rows into Student cells plus complement."""

    if teacher_probabilities.ndim != 2:
        raise ValueError("BPM Teacher probabilities must have shape [rows, vocabulary]")
    if teacher_probabilities.size(1) != plan.routes.numel():
        raise ValueError("BPM Teacher probability width differs from the routing table")
    compatible_mass = teacher_probabilities.index_select(
        1, plan.compatible_teacher_ids
    ).sum(dim=-1)
    if not bool(torch.isfinite(compatible_mass).all()) or bool((compatible_mass <= 0).any()):
        raise FloatingPointError("BPM row has zero or non-finite compatible Teacher mass")
    mapped = teacher_probabilities.index_select(1, plan.mapped_teacher_ids)
    mapped = mapped / compatible_mass.unsqueeze(-1)
    target = teacher_probabilities.new_zeros(
        (teacher_probabilities.size(0), plan.support_ids.numel())
    )
    if mapped.numel():
        target.scatter_add_(
            1,
            plan.mapped_cells.unsqueeze(0).expand(teacher_probabilities.size(0), -1),
            mapped,
        )
    residual = (
        teacher_probabilities.index_select(1, plan.residual_teacher_ids).sum(dim=-1)
        / compatible_mass
        if plan.residual_teacher_ids.numel()
        else compatible_mass.new_zeros(compatible_mass.shape)
    )
    total = target.sum(dim=-1) + residual
    if not bool(torch.allclose(total, torch.ones_like(total), atol=2e-5, rtol=2e-5)):
        raise FloatingPointError("BPM dense projection did not preserve probability mass")
    return target, residual


def bpm_projected_divergence(
    target: torch.Tensor,
    student: torch.Tensor,
    *,
    beta: float,
    skew: float,
    force_fkl: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paper D_beta including its normalized endpoint definitions."""

    def kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        tiny = torch.finfo(left.dtype).tiny
        return torch.where(
            left > 0,
            left * (left.clamp_min(tiny).log() - right.clamp_min(tiny).log()),
            torch.zeros_like(left),
        ).sum(dim=-1)

    if beta == 0.0:
        losses = kl(target, student)
    elif beta == 1.0:
        mixture = (1.0 - skew) * target + skew * student
        losses = kl(student, mixture)
    else:
        mixture = beta * target + (1.0 - beta) * student
        losses = beta * kl(target, mixture) + (1.0 - beta) * kl(student, mixture)
    if force_fkl is not None and bool(force_fkl.any()):
        losses = torch.where(force_fkl, kl(target, student), losses)
    return losses


def bpm_projected_divergence_from_log_probs(
    target: torch.Tensor,
    student_log_probs: torch.Tensor,
    *,
    beta: float,
    skew: float,
    force_fkl: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate paper ``D_beta`` while keeping every FKL row in log space.

    Explicit Student cells can be arbitrarily unlikely and the complement can
    carry almost all or almost none of the distribution. Consuming their
    backend-computed log-probabilities avoids both ``exp -> log`` underflow and
    cancellation in ``log(1 - sum(p_explicit))``. Non-FKL arms still use the
    probability-domain paper definition after a normalized exponentiation.
    """

    if target.shape != student_log_probs.shape:
        raise ValueError("BPM target and Student log-probabilities must have equal shapes")
    if target.ndim != 2:
        raise ValueError("BPM projected distributions must be rank two")
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("BPM beta must be in [0, 1]")

    active = target > 0
    active_student_log_probs = student_log_probs.masked_select(active)
    if not bool(torch.isfinite(active_student_log_probs).all()):
        raise FloatingPointError(
            "BPM Student log-probabilities are non-finite on positive target cells"
        )
    safe_student_log_probs = student_log_probs.masked_fill(~active, 0.0)
    fkl = (
        torch.special.xlogy(target, target)
        - target * safe_student_log_probs
    ).sum(dim=-1)

    if beta == 0.0:
        losses = fkl
    else:
        student = student_log_probs.exp()
        student = student / student.sum(dim=-1, keepdim=True)
        losses = bpm_projected_divergence(
            target,
            student,
            beta=beta,
            skew=skew,
        )
    if force_fkl is not None:
        if force_fkl.shape != target.shape[:-1]:
            raise ValueError("BPM force_fkl mask does not match projected rows")
        if bool(force_fkl.any()):
            losses = torch.where(force_fkl, fkl, losses)
    if not bool(torch.isfinite(losses).all()):
        raise FloatingPointError("BPM produced a non-finite projected divergence")
    return losses


def compute_dense_bpm_paper_losses(
    algorithm: Any,
    context: FusedScoringContext,
) -> torch.Tensor:
    """Compute exact full-vocabulary BPM through the generic plugin contract."""

    artifact = algorithm._artifact()
    beta, skew, mask_whitespace, spanning_policy = algorithm._settings()
    if spanning_policy != "exact":
        raise ValueError("Paper BPM causal-hidden scoring requires exact spanning correction")
    if context.teacher_distribution.length != context.alignment.teacher_length:
        raise ValueError("Teacher distribution rows do not match the BPM alignment")
    teacher_ids_tensor = context.teacher_response_token_ids.to(dtype=torch.long).flatten()
    if teacher_ids_tensor.numel() != context.alignment.teacher_length:
        raise ValueError("Teacher response token IDs do not match the BPM alignment")
    teacher_ids = teacher_ids_tensor.detach().cpu().tolist()

    local_positions = context.local_response_positions.to(
        device=context.student.response_token_ids.device, dtype=torch.long
    ).flatten()
    position_to_local = {
        int(global_position): local_position
        for local_position, global_position in enumerate(local_positions.tolist())
    }
    rows = [
        dict(row)
        for row in context.alignment.algorithm_metadata.get("bpm_rows", ())
        if int(row["student_position"]) in position_to_local
        and not (mask_whitespace and bool(row.get("whitespace", False)))
    ]
    output = torch.zeros(
        (context.student.distribution.length,),
        dtype=torch.float32,
        device=context.student.response_token_ids.device,
    )
    if not rows:
        if context.student.distribution.length == 0:
            return output
        anchor_position = torch.zeros((1,), device=output.device, dtype=torch.long)
        anchor_token = context.student.response_token_ids[:1].reshape(1, 1)
        anchor = context.student.distribution.rows(anchor_position).selected_log_probs(
            anchor_token,
            vocab_size=len(artifact.student_bytes),
            temperature=1.0,
        ).sum() * 0.0
        return output + anchor

    route_plan_cache: dict[tuple[Any, ...], DenseBPMRoutePlan] = {}
    probability_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
    probability_cache_rows = 4

    def full_rows(positions: torch.Tensor) -> torch.Tensor:
        return context.teacher_distribution.rows(positions).full_probabilities(
            vocab_size=len(artifact.teacher_bytes),
            temperature=1.0,
        ).float()

    def full_row(position: int) -> torch.Tensor:
        position = int(position)
        cached = probability_cache.get(position)
        if cached is None:
            cached = full_rows(
                torch.tensor([position], device=output.device, dtype=torch.long)
            )[0]
            probability_cache[position] = cached
            if len(probability_cache) > probability_cache_rows:
                probability_cache.popitem(last=False)
        else:
            probability_cache.move_to_end(position)
        return cached

    def route_plan(row: dict[str, Any]) -> DenseBPMRoutePlan:
        signature = algorithm._route_signature(row)
        plan = route_plan_cache.get(signature)
        if plan is None:
            plan = compile_dense_route_plan(
                routes_for_bpm_row(artifact, row), device=output.device
            )
            route_plan_cache[signature] = plan
        return plan

    def student_log_projection(
        positions: torch.Tensor,
        support_ids: torch.Tensor,
    ) -> torch.Tensor:
        support_mask = torch.ones_like(support_ids, dtype=torch.bool)
        return context.student.distribution.rows(positions).projected_log_probs(
            support_ids,
            support_mask,
            vocab_size=len(artifact.student_bytes),
            temperature=1.0,
        ).float()

    def projected_loss(
        row_group: list[dict[str, Any]],
        *,
        force_fkl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = row_group[0]
        plan = route_plan(row)
        teacher_positions = torch.tensor(
            [int(item["teacher_position"]) for item in row_group],
            device=output.device,
            dtype=torch.long,
        )
        target_explicit, target_residual = project_dense_teacher_rows(
            full_rows(teacher_positions), plan
        )
        target = torch.cat(
            (target_explicit, target_residual.unsqueeze(-1)), dim=-1
        )
        student_positions = torch.tensor(
            [position_to_local[int(item["student_position"])] for item in row_group],
            device=output.device,
            dtype=torch.long,
        )
        point_mass = (target > 1e-8).sum(dim=-1) == 1
        if force_fkl:
            point_mass = torch.ones_like(point_mass)
        return student_positions, bpm_projected_divergence_from_log_probs(
            target,
            student_log_projection(student_positions, plan.support_ids),
            beta=beta,
            skew=skew,
            force_fkl=point_mass,
        )

    ordinary: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    spanning: list[dict[str, Any]] = []
    stops: list[dict[str, Any]] = []
    for row in rows:
        kind = str(row["kind"])
        if kind == "spanning":
            spanning.append(row)
        elif kind == "stop":
            stops.append(row)
        else:
            ordinary.setdefault(algorithm._route_signature(row), []).append(row)

    chunk_size = int(algorithm.algorithm_kwargs.get("dense_row_chunk_size", 32))
    if chunk_size <= 0:
        raise ValueError("BPM dense_row_chunk_size must be positive")
    for group_rows in ordinary.values():
        for start in range(0, len(group_rows), chunk_size):
            positions, losses = projected_loss(group_rows[start : start + chunk_size])
            output.index_add_(0, positions, losses)

    for row in spanning:
        plan = route_plan(row)
        teacher_position = int(row["teacher_position"])
        base_target, base_residual = project_dense_teacher_rows(
            full_row(teacher_position).unsqueeze(0), plan
        )
        spec = spanning_chain_spec(artifact, row, teacher_ids)

        def token_probability(position: int, token_id: int) -> float:
            return float(full_row(position)[int(token_id)].item())

        def prefix_mass(position: int, prefix: bytes) -> float:
            prefix_routes = torch.tensor(
                routes_for_bpm_row(
                    artifact, {"kind": "interior", "prefix_hex": prefix.hex()}
                ),
                device=output.device,
            )
            return float(
                full_row(position)[prefix_routes != INCOMPATIBLE_ROUTE].sum().item()
            )

        first_teacher_id = teacher_ids[teacher_position]
        correction_ids, correction_values, residual_delta = spanning_correction_updates(
            spec,
            token_prob=token_probability,
            prefix_mass=prefix_mass,
            base_route=int(plan.routes[first_teacher_id].item()),
        )
        correction_ids_tensor = torch.tensor(
            correction_ids, device=output.device, dtype=torch.long
        )
        support_ids = torch.unique(
            torch.cat((plan.support_ids, correction_ids_tensor)), sorted=True
        )
        target_explicit = base_target.new_zeros((1, support_ids.numel()))
        target_explicit.scatter_add_(
            1,
            torch.searchsorted(support_ids, plan.support_ids).unsqueeze(0),
            base_target,
        )
        target_explicit.index_add_(
            1,
            torch.searchsorted(support_ids, correction_ids_tensor),
            torch.tensor(
                correction_values,
                device=output.device,
                dtype=target_explicit.dtype,
            ).unsqueeze(0),
        )
        residual = (base_residual + float(residual_delta)).clamp_min(0.0)
        target_explicit = target_explicit.clamp_min(0.0)
        target = torch.cat((target_explicit, residual.unsqueeze(-1)), dim=-1)
        target = target / target.sum(dim=-1, keepdim=True)
        student_position = torch.tensor(
            [position_to_local[int(row["student_position"])]],
            device=output.device,
            dtype=torch.long,
        )
        loss = bpm_projected_divergence_from_log_probs(
            target,
            student_log_projection(student_position, support_ids),
            beta=beta,
            skew=skew,
            force_fkl=(target > 1e-8).sum(dim=-1) == 1,
        )
        output.index_add_(0, student_position, loss)

    for row in stops:
        positions, losses = projected_loss([row], force_fkl=True)
        output.index_add_(0, positions, losses)
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError("Dense BPM produced non-finite token losses")
    return output
