import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = ["check_reward_nonzero_std", "check_not_truncated", "check_fully_skipped_alignment"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_not_truncated(args, samples, **kwargs):
    """Legacy filter that rejects truncation and fully-skipped alignments.

    The active long-run config uses :func:`check_fully_skipped_alignment`
    instead, so this stricter behavior remains available to older manifests
    without forcing the current run to replenish every truncated group.
    """

    # The normal path passes ``list[Sample]``.  Flatten one nested level so
    # custom fan-out rollout functions remain compatible with this filter.
    flattened = []
    for sample in samples:
        if isinstance(sample, (list, tuple)):
            flattened.extend(sample)
        else:
            flattened.append(sample)

    truncated = any(sample.status == Sample.Status.TRUNCATED for sample in flattened)
    if truncated:
        return DynamicFilterOutput(keep=False, reason="truncated")

    for sample in flattened:
        reward = getattr(sample, "reward", None)
        if not isinstance(reward, dict):
            continue
        skipped_mask = reward.get("teacher_cross_skipped_mask")
        if skipped_mask is None:
            continue
        # The external reward hook normally returns a Python list.  Accept a
        # tensor/array as well without requiring a specific transport type.
        try:
            values = [bool(value) for value in skipped_mask]
        except TypeError:
            values = [bool(skipped_mask)]
        if values and all(values):
            return DynamicFilterOutput(
                keep=False,
                reason="cross_tokenizer_target_fully_skipped",
            )

    return DynamicFilterOutput(keep=True, reason=None)


def check_fully_skipped_alignment(args, samples, **kwargs):
    """Reject only groups with no usable cross-tokenizer target.

    This is the filter used by the current BPM run.  ``TRUNCATED`` samples
    are deliberately *not* rejected: they remain in the training batch and
    are counted by the rollout truncation metrics.  Only an all-true
    ``teacher_cross_skipped_mask`` has no valid BPM/ESCD supervision and is
    replenished.
    """

    flattened = []
    for sample in samples:
        if isinstance(sample, (list, tuple)):
            flattened.extend(sample)
        else:
            flattened.append(sample)

    for sample in flattened:
        reward = getattr(sample, "reward", None)
        if not isinstance(reward, dict):
            continue
        skipped_mask = reward.get("teacher_cross_skipped_mask")
        if skipped_mask is None:
            continue
        try:
            values = [bool(value) for value in skipped_mask]
        except TypeError:
            values = [bool(skipped_mask)]
        if values and all(values):
            return DynamicFilterOutput(
                keep=False,
                reason="cross_tokenizer_target_fully_skipped",
            )

    return DynamicFilterOutput(keep=True, reason=None)
