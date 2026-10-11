from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import torch
import torch.distributed as dist

from slime.utils.distributed_utils import get_gloo_group

from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import UpdateWeightFromDistributed


class UpdateWeightFromDistributedBridge(UpdateWeightFromDistributed):
    """
    Bridge exporter + distributed NCCL transport.

    The normal distributed updater uses the raw MCore->HF name mapper. Bridge-built
    models need Megatron Bridge for correct HF names, but non-colocated rollout
    engines must still receive weights through NCCL instead of CUDA IPC.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        super().__init__(
            args=args,
            model=model,
            weights_getter=weights_getter,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self.weights_getter = weights_getter
        if self._hf_weight_iterator is None:
            self._hf_weight_iterator = HfWeightIteratorBase.create(
                args=args,
                model=model,
                model_name=model_name,
                quantization_config=quantization_config,
            )

    def _send_weights(self, pbar) -> None:
        megatron_local_weights = self._get_snapshot_weights()
        for hf_chunk in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            if self._is_pp_src_rank:
                self._on_chunk(hf_chunk)
                self._update_bucket_weights_from_distributed(hf_chunk, pbar=pbar)
            # Keep all PP sources on the same snapshot bucket.  The SGLang
            # broadcast lock is shared across PP stages, while the next
            # iterator bucket requires PP collectives on those same sources.
            dist.barrier(group=get_gloo_group())

        dist.barrier(group=get_gloo_group())
