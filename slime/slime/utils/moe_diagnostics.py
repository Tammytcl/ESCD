"""Optional training-time MoE routing diagnostics for the Slime loop.

Megatron's stock ``track_moe_metrics`` is wired into its pretraining loop,
whereas Slime drives Megatron through its own Ray actor loop.  This module
keeps the instrumentation opt-in and records only small ``[layers, experts]``
count tensors during a train step.  The counts are reduced across the actor
world and returned as scalar metrics which Slime forwards to W&B.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_INSTALLED = False
_ACTIVE = False
_COUNTS: dict[int, torch.Tensor] = {}
_ROUTER_PARAMETER_RE = re.compile(r"(^|\.)mlp\.router\.weight$")


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def enabled(args: Any | None = None) -> bool:
    """Whether training-time routing diagnostics are enabled."""

    if _truthy(os.environ.get("MOE_LOAD_DIAGNOSTICS", "0")):
        return True
    return bool(getattr(args, "moe_load_diagnostics", False)) if args is not None else False


def install(args: Any) -> None:
    """Install a lightweight ``TopKRouter.routing`` recorder.

    This function is used through Slime's ``--custom-megatron-init-path`` and
    therefore runs after distributed groups are initialized but before model
    construction.  It intentionally does not add a forward hook to every
    module, which would retain activations and increase memory use.
    """

    global _INSTALLED
    if _INSTALLED or not enabled(args):
        return

    from megatron.core.transformer.moe.router import TopKRouter

    original_routing = TopKRouter.routing

    def routing_with_diagnostics(self, logits, padding_mask=None):
        probs, routing_map = original_routing(self, logits, padding_mask=padding_mask)
        if _ACTIVE and self.training and torch.is_grad_enabled():
            _record(self.layer_number, routing_map)
        return probs, routing_map

    # Mark the class so repeated actor/model initialization cannot wrap it
    # more than once in a process.
    routing_with_diagnostics.__name__ = "routing_with_moe_diagnostics"
    routing_with_diagnostics.__wrapped__ = original_routing
    TopKRouter.routing = routing_with_diagnostics
    _INSTALLED = True
    logger.info(
        "Enabled training MoE routing diagnostics (interval=%s, per_layer=%s)",
        os.environ.get("MOE_DIAGNOSTICS_LOG_INTERVAL", "10"),
        os.environ.get("MOE_DIAGNOSTICS_PER_LAYER", "1"),
    )


def _record(layer_number: int | None, routing_map: torch.Tensor) -> None:
    if layer_number is None or routing_map is None or routing_map.numel() == 0:
        return
    if routing_map.ndim != 2:
        return
    counts = routing_map.detach().to(dtype=torch.float32).sum(dim=0)
    previous = _COUNTS.get(int(layer_number))
    _COUNTS[int(layer_number)] = counts if previous is None else previous + counts


def begin_step(args: Any | None = None) -> None:
    """Start collecting routing counts for one Slime optimizer step."""

    global _ACTIVE, _COUNTS
    if not enabled(args):
        return
    _COUNTS = {}
    _ACTIVE = True

    # The auxiliary-loss tracker is also populated by Router.forward.  Clear
    # stale values from a preceding forward-only/ref pass before this step.
    try:
        from megatron.core.transformer.moe.moe_utils import clear_aux_losses_tracker

        clear_aux_losses_tracker()
    except (ImportError, AttributeError):
        pass


def _all_reduce(value: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(value, op=op)
    return value


def _router_parameter_metrics(model: Any | None, device: torch.device) -> dict[str, float]:
    """Return runtime evidence for the student router freeze contract.

    ``freeze_params_name_list`` is applied after model construction and the
    parameter names are Megatron names, not HF names.  A config/dry-run check
    therefore cannot prove that the intended parameters were actually frozen.
    This helper counts the matching parameters and their gradients immediately
    before ``optimizer.step``.  Every distributed rank participates in the
    same small all-reduce, including pipeline stages that own no router
    shard.
    """

    # [all, trainable, frozen, grad_sq, grad_numel, nonfinite_grad_numel,
    #  tensor_count, frozen_tensor_count]
    stats = torch.zeros(8, dtype=torch.float64, device=device)
    if model is not None:
        chunks = model if isinstance(model, (list, tuple)) else (model,)
        for chunk in chunks:
            named_parameters = getattr(chunk, "named_parameters", None)
            if named_parameters is None:
                continue
            for name, parameter in named_parameters():
                if not _ROUTER_PARAMETER_RE.search(name):
                    continue
                parameter_count = float(parameter.numel())
                stats[0] += parameter_count
                stats[6] += 1.0
                if parameter.requires_grad:
                    stats[1] += parameter_count
                else:
                    stats[2] += parameter_count
                    stats[7] += 1.0
                # Megatron's distributed optimizer normally stores the
                # reduced gradient in ``main_grad``; plain torch modules use
                # ``grad``.  Prefer main_grad so a frozen-router check does
                # not falsely report zero merely because this is the
                # Megatron path.
                gradient = getattr(parameter, "main_grad", None)
                if gradient is None:
                    gradient = parameter.grad
                if gradient is None:
                    continue
                grad = gradient.detach().float()
                finite = torch.isfinite(grad)
                stats[3] += torch.where(finite, grad, torch.zeros_like(grad)).square().sum().double()
                stats[4] += float(grad.numel())
                stats[5] += (~finite).sum().double()

    _all_reduce(stats)
    metrics = {
        "moe/router_param_count": float(stats[0].item()),
        "moe/router_trainable_param_count": float(stats[1].item()),
        "moe/router_frozen_param_count": float(stats[2].item()),
        "moe/router_tensor_count": float(stats[6].item()),
        "moe/router_frozen_tensor_count": float(stats[7].item()),
        "moe/router_grad_norm": float(torch.sqrt(stats[3].clamp_min(0.0)).item()),
        "moe/router_grad_param_fraction": float(
            (stats[4] / stats[1].clamp_min(1.0)).item()
        ),
        "moe/router_grad_nonfinite_fraction": float(
            (stats[5] / stats[4].clamp_min(1.0)).item()
        ),
    }
    return metrics


def _gini(fractions: torch.Tensor) -> torch.Tensor:
    """Gini coefficient of one normalized expert-load vector."""

    sorted_values = torch.sort(fractions).values
    experts = sorted_values.numel()
    if experts == 0:
        return sorted_values.new_zeros(())
    weights = torch.arange(1, experts + 1, device=sorted_values.device, dtype=sorted_values.dtype)
    total = sorted_values.sum().clamp_min(torch.finfo(sorted_values.dtype).eps)
    return (2 * (weights * sorted_values).sum() - (experts + 1) * total) / (experts * total)


def end_step(args: Any, iteration: int, model: Any | None = None) -> dict[str, float]:
    """Reduce counts and create scalar W&B metrics for the completed step.

    All actor ranks execute the same world-size all-reduce.  The returned
    dictionary is identical on every rank; Slime only emits it on its normal
    logging rank alongside the loss metrics.
    """

    global _ACTIVE, _COUNTS
    if not enabled(args):
        return {}
    _ACTIVE = False

    try:
        num_layers = int(getattr(args, "num_layers"))
        raw_experts = getattr(args, "num_experts")
        num_experts = int(raw_experts[0] if isinstance(raw_experts, (list, tuple)) else raw_experts)
    except (TypeError, ValueError, AttributeError):
        _COUNTS = {}
        return {}

    if num_layers <= 0 or num_experts <= 0:
        _COUNTS = {}
        return {}

    device = (
        next(iter(_COUNTS.values())).device
        if _COUNTS
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    counts = torch.zeros((num_layers, num_experts), dtype=torch.float32, device=device)
    for layer_number, layer_counts in _COUNTS.items():
        index = int(layer_number) - 1
        if 0 <= index < num_layers:
            width = min(num_experts, int(layer_counts.numel()))
            counts[index, :width] += layer_counts[:width]

    # Keep a pre-reduction copy as well.  The aggregate statistics below are
    # useful for the model as a whole, but an all-reduce can hide a hot expert
    # on one rank behind balanced traffic on another.  We therefore export a
    # worst-rank summary in addition to the global summary.
    local_totals = counts.sum(dim=1)
    local_valid = local_totals > 0
    local_fractions = counts / local_totals.clamp_min(1.0).unsqueeze(1)
    local_means = counts.sum(dim=1) / num_experts
    local_std = torch.sqrt(((counts - local_means.unsqueeze(1)) ** 2).mean(dim=1))
    local_cv = local_std / local_means.clamp_min(1.0)
    local_load_ratio = local_fractions * num_experts
    local_max_ratio = local_load_ratio.max(dim=1).values
    local_min_ratio = local_load_ratio.min(dim=1).values
    local_zero_fraction = (counts == 0).float().mean(dim=1)
    _all_reduce(counts)

    totals = counts.sum(dim=1)
    valid = totals > 0
    metrics: dict[str, float] = {}
    if not bool(valid.any()):
        _COUNTS = {}
        return metrics

    # Keep architecture/coverage metadata beside the route scalars.  The
    # human-readable DLC sink clips long dictionaries, so downstream audits
    # must use the structured W&B event record (or another structured sink) to
    # read these fields reliably.  ``valid`` is computed after the actor-world
    # reduction and therefore measures global, not rank-local, layer coverage.
    valid_layer_count = int(valid.sum().item())
    metrics.update(
        {
            "moe/diagnostic_num_layers": float(num_layers),
            "moe/diagnostic_num_experts": float(num_experts),
            "moe/diagnostic_valid_layers": float(valid_layer_count),
            "moe/diagnostic_layer_coverage_fraction": float(valid_layer_count / num_layers),
            "moe/diagnostic_counts_global_reduced": 1.0,
        }
    )

    # This is intentionally computed before optimizer.step.  It proves the
    # freeze/gradient contract without changing the objective or retaining
    # any activation.  A frozen router should report a non-zero frozen count
    # and zero router_grad_norm; route histograms may still drift through
    # changing hidden states.
    metrics.update(_router_parameter_metrics(model, device=device))

    # Reduce only extrema for rank-local diagnostics.  This preserves the
    # signal of the most overloaded rank without emitting one W&B series per
    # distributed rank.  Ranks with no local MoE layer contribute neutral
    # extrema, so pipeline stages without an MoE block do not skew the result.
    if bool(local_valid.any()):
        local_extrema_max = torch.stack(
            [
                local_cv[local_valid].max(),
                local_max_ratio[local_valid].max(),
                local_zero_fraction[local_valid].max(),
            ]
        )
        local_extrema_min = local_min_ratio[local_valid].min().reshape(1)
    else:
        local_extrema_max = counts.new_zeros(3)
        local_extrema_min = counts.new_full((1,), float("inf"))
    _all_reduce(local_extrema_max, op=dist.ReduceOp.MAX)
    _all_reduce(local_extrema_min, op=dist.ReduceOp.MIN)

    fractions = counts / totals.clamp_min(1.0).unsqueeze(1)
    expected = 1.0 / num_experts
    load_ratio = fractions / expected
    means = counts.sum(dim=1) / num_experts
    std = torch.sqrt(((counts - means.unsqueeze(1)) ** 2).mean(dim=1))
    cv = std / means.clamp_min(1.0)
    max_ratio = load_ratio.max(dim=1).values
    min_ratio = load_ratio.min(dim=1).values
    zero_fraction = (counts == 0).float().mean(dim=1)
    gini = torch.stack([_gini(row) for row in fractions])

    valid_cv = cv[valid]
    valid_max = max_ratio[valid]
    valid_min = min_ratio[valid]
    valid_zero = zero_fraction[valid]
    valid_gini = gini[valid]
    worst = int(torch.argmax(torch.where(valid, cv, cv.new_full(cv.shape, -1))).item())
    worst_top_expert = int(torch.argmax(fractions[worst]).item())
    worst_bottom_expert = int(torch.argmin(fractions[worst]).item())

    metrics.update(
        {
            "moe/load_cv_mean": float(valid_cv.mean().item()),
            "moe/load_cv_max": float(valid_cv.max().item()),
            "moe/load_gini_mean": float(valid_gini.mean().item()),
            "moe/load_gini_max": float(valid_gini.max().item()),
            "moe/load_max_over_mean": float(valid_max.max().item()),
            "moe/load_min_over_mean": float(valid_min.min().item()),
            "moe/zero_expert_fraction_mean": float(valid_zero.mean().item()),
            "moe/local_load_cv_max": float(local_extrema_max[0].item()),
            "moe/local_load_max_over_mean_max": float(local_extrema_max[1].item()),
            "moe/local_load_min_over_mean_min": float(local_extrema_min[0].item()),
            "moe/local_zero_expert_fraction_max": float(local_extrema_max[2].item()),
            "moe/worst_layer": float(worst + 1),
            "moe/worst_layer_cv": float(cv[worst].item()),
            "moe/worst_layer_max_over_mean": float(max_ratio[worst].item()),
            "moe/worst_layer_min_over_mean": float(min_ratio[worst].item()),
            "moe/worst_layer_zero_expert_fraction": float(zero_fraction[worst].item()),
            "moe/worst_layer_top_expert": float(worst_top_expert),
            "moe/worst_layer_bottom_expert": float(worst_bottom_expert),
            "moe/worst_layer_top_expert_fraction": float(fractions[worst, worst_top_expert].item()),
            "moe/worst_layer_bottom_expert_fraction": float(fractions[worst, worst_bottom_expert].item()),
        }
    )

    # The stock Megatron tracker stores the unscaled load-balancing objective
    # (aux_loss / coefficient).  Expose it here because Slime's custom loop
    # does not call Megatron's pretraining ``training_log`` helper.
    try:
        from megatron.core.transformer.moe.moe_utils import get_moe_layer_wise_logging_tracker

        tracker = get_moe_layer_wise_logging_tracker()
        coeff = getattr(args, "moe_aux_loss_coeff", 0.0)
        if isinstance(coeff, (list, tuple)):
            coeff = coeff[0] if coeff else 0.0
        coeff = float(coeff)
        route_types = getattr(args, "moe_router_load_balancing_type", "")
        if isinstance(route_types, str):
            route_types = [route_types]
        expected_names = []
        if "aux_loss" in route_types:
            expected_names.append("load_balancing_loss")
        if "seq_aux_loss" in route_types:
            expected_names.append("seq_load_balancing_loss")
        if "global_aux_loss" in route_types:
            expected_names.append("global_load_balancing_loss")
        if getattr(args, "moe_z_loss_coeff", None) is not None:
            expected_names.append("z_loss")
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        # Use a fixed name list on every rank.  PP stages may own no MoE layer
        # (and consequently have no local tracker entry); allocating zeros
        # here prevents a conditional all-reduce deadlock.
        for name in expected_names:
            entry = tracker.get(name, {})
            values = entry.get("values")
            aux_sum = (
                values.detach().float().sum()
                if values is not None
                else counts.new_zeros(())
            )
            _all_reduce(aux_sum)
            # Every PP/DP replica contributes its owned layers; normalize by
            # the number of replica copies and layers for a stable scalar.
            metrics[f"moe/{name}_unscaled"] = float((aux_sum / max(1, world * num_layers)).item())
            if coeff > 0:
                metrics[f"moe/{name}_weighted"] = float(
                    (aux_sum * coeff / max(1, world * num_layers)).item()
                )
    except (ImportError, AttributeError, TypeError, ValueError):
        # Diagnostics must never make training fail if a different Megatron
        # image omits the optional aux-loss tracker.
        pass

    try:
        from megatron.core.transformer.moe.moe_utils import clear_aux_losses_tracker

        clear_aux_losses_tracker()
    except (ImportError, AttributeError):
        pass

    try:
        interval = max(1, int(os.environ.get("MOE_DIAGNOSTICS_LOG_INTERVAL", "10")))
    except ValueError:
        interval = 10
    if _truthy(os.environ.get("MOE_DIAGNOSTICS_PER_LAYER", "1")) and iteration % interval == 0:
        for layer in torch.nonzero(valid, as_tuple=False).flatten().tolist():
            metrics[f"moe/layer_{layer + 1:02d}/cv"] = float(cv[layer].item())
            metrics[f"moe/layer_{layer + 1:02d}/gini"] = float(gini[layer].item())
            metrics[f"moe/layer_{layer + 1:02d}/max_over_mean"] = float(max_ratio[layer].item())
            metrics[f"moe/layer_{layer + 1:02d}/min_over_mean"] = float(min_ratio[layer].item())
            metrics[f"moe/layer_{layer + 1:02d}/zero_expert_fraction"] = float(zero_fraction[layer].item())

    _COUNTS = {}
    return metrics
