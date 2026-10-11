import json
import logging
import os
import random
from argparse import Namespace
from contextlib import nullcontext

import numpy as np
import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from opd.cross_tokenizer import (
    SharedVocabulary,
    build_cross_tokenizer,
    cross_tokenizer_capabilities,
    normalize_json_object,
)
from opd.cross_tokenizer.runtime import ensure_student_terminal_eos, prepare_teacher_rollout_data
from opd.cross_tokenizer._paper_utils import tokenizer_vocab_size
from opd.cross_tokenizer.tokenizer_bytes import model_stop_ids, preferred_model_stop_id
from opd.cross_tokenizer.teacher_head import load_frozen_lm_head_shard
from slime.ray.train_actor import TrainRayActor
from slime.utils import train_dump_utils
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.logging_utils import init_tracking
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.misc import Box
from slime.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from slime.utils.routing_replay import RoutingReplay, prepare_bshd_routing_replay
from slime.utils.timer import Timer, inverse_timer, timer, with_defer
from slime.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from .checkpoint import load_checkpoint
from .cp_utils import slice_log_prob_with_cp, slice_with_cp
from .data import DataIterator, get_data_iterator, log_perf_data, log_rollout_data
from .initialize import init, is_megatron_main_rank
from .loss import (
    compute_advantages_and_returns,
    get_cross_tokenizer_teacher_data,
    get_log_probs_and_entropy,
    get_opd_student_topk,
    get_opd_teacher_data,
    get_values,
    set_cross_tokenizer_teacher_head_shard,
)
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_distributed import UpdateWeightFromDistributed
from .update_weight.update_weight_from_distributed_bridge import UpdateWeightFromDistributedBridge
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        if args.debug_rollout_only:
            self.args = args
            return 0

        monkey_patch_torch_dist()
        super().init(args, role, with_ref, with_opd_teacher)

        init(args)

        if is_megatron_main_rank():
            init_tracking(args, primary=False, role=role)

        self.prof = TrainProfiler(args)

        self.teacher_tokenizer = None
        self.teacher_hf_config = None
        self.cross_tokenizer_capabilities = (
            cross_tokenizer_capabilities(args.cross_tokenizer_algorithm)
            if getattr(args, "use_cross_tokenizer", False)
            else None
        )
        teacher_transport = (
            self.cross_tokenizer_capabilities.teacher_transport(args.opd_type)
            if self.cross_tokenizer_capabilities is not None
            else "payload"
        )
        prepare_local_cross_tokenizer = (
            getattr(args, "use_cross_tokenizer", False)
            and (args.opd_type == "megatron" or teacher_transport == "causal_hidden")
        )
        # read config and tokenizers serialized to prevent concurrent writing bug.
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
                if prepare_local_cross_tokenizer:
                    self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                        args.opd_teacher_tokenizer, trust_remote_code=True
                    )
                    self.teacher_hf_config = AutoConfig.from_pretrained(
                        args.opd_teacher_model_path or args.opd_teacher_tokenizer,
                        trust_remote_code=True,
                    )
            dist.barrier(group=get_gloo_group())

        dist.barrier(group=get_gloo_group())

        self.cross_tokenizer_algorithm = None
        self.cross_tokenizer_vocab_limit = None
        self.cross_tokenizer_teacher_vocab_limit = None
        self.cross_tokenizer_student_stop_ids: tuple[int, ...] = ()
        self.cross_tokenizer_teacher_stop_ids: tuple[int, ...] = ()
        self.cross_tokenizer_teacher_bridge_stop_id: int | None = None
        if getattr(args, "use_cross_tokenizer", False):
            opaque_external_teacher = args.opd_type == "sglang" and teacher_transport == "payload"
            self.cross_tokenizer_student_stop_ids = model_stop_ids(
                args.hf_checkpoint, self.tokenizer
            )
            if not self.cross_tokenizer_student_stop_ids:
                raise ValueError("Cross-tokenizer OPD requires a declared semantic Student stop token")
            if not opaque_external_teacher:
                if self.teacher_tokenizer is None:
                    raise ValueError("In-process cross-tokenizer OPD requires a Teacher tokenizer")
                self.cross_tokenizer_teacher_stop_ids = model_stop_ids(
                    args.opd_teacher_tokenizer, self.teacher_tokenizer
                )
                if not self.cross_tokenizer_teacher_stop_ids:
                    raise ValueError("Cross-tokenizer OPD requires a declared semantic Teacher stop token")
                self.cross_tokenizer_teacher_bridge_stop_id = preferred_model_stop_id(
                    args.opd_teacher_model_path or args.opd_teacher_tokenizer,
                    self.teacher_tokenizer,
                    declared_stop_ids=self.cross_tokenizer_teacher_stop_ids,
                )
            algorithm_kwargs = dict(args.cross_tokenizer_algorithm_kwargs)
            # An external SGLang Teacher owns the byte-routing table and
            # sends an already projected BPM payload.  GPU loss reconstruction
            # must therefore not serialize that large CPU artifact, but BPM
            # still needs the Student softmax width for its complement cell.
            # Carry this scalar in the JSON algorithm kwargs on every rank.
            if opaque_external_teacher:
                # Use the tokenizer's effective vocabulary rather than the HF
                # output-head width.  Some models pad the latter for tensor
                # parallelism (Qwen3.6 has 248320 rows but 248077 IDs); the
                # padded rows must be excluded from BPM normalization.
                algorithm_kwargs["student_vocab_size"] = tokenizer_vocab_size(self.tokenizer)
                args.cross_tokenizer_algorithm_kwargs = normalize_json_object(
                    algorithm_kwargs,
                    field_name="cross-tokenizer algorithm kwargs with external Teacher contract",
                )
            self.cross_tokenizer_algorithm = build_cross_tokenizer(
                args.cross_tokenizer_algorithm,
                student_tokenizer=None if opaque_external_teacher else self.tokenizer,
                teacher_tokenizer=None if opaque_external_teacher else self.teacher_tokenizer,
                algorithm_kwargs=algorithm_kwargs,
            )
            if not opaque_external_teacher:
                self.cross_tokenizer_algorithm.configure_model_contract(
                    student_vocab_size=tokenizer_vocab_size(self.tokenizer),
                    teacher_vocab_size=tokenizer_vocab_size(self.teacher_tokenizer),
                    student_stop_ids=self.cross_tokenizer_student_stop_ids,
                    teacher_stop_ids=self.cross_tokenizer_teacher_stop_ids,
                )
                algorithm_kwargs = dict(self.cross_tokenizer_algorithm.algorithm_kwargs)
                args.cross_tokenizer_algorithm_kwargs = normalize_json_object(
                    algorithm_kwargs,
                    field_name="cross-tokenizer algorithm kwargs with model contract",
                )
            # The external Teacher already constructs the opaque target.  The
            # Student-side loss only needs algorithm settings, so avoid
            # rebuilding/serializing the large BPM tokenizer routing table on
            # every Megatron rank.
            shared_vocabulary = (
                None if opaque_external_teacher else self.cross_tokenizer_algorithm.build_shared_vocabulary()
            )
            if shared_vocabulary is not None and not isinstance(shared_vocabulary, SharedVocabulary):
                raise TypeError(
                    "Cross-tokenizer build_shared_vocabulary() must return SharedVocabulary or None"
                )
            runtime_state = normalize_json_object(
                {} if opaque_external_teacher else self.cross_tokenizer_algorithm.export_runtime_state(),
                field_name=f"{args.cross_tokenizer_algorithm} runtime_state",
            )
            # Keep only pure JSON on the Megatron Namespace. The explicit
            # round-trip proves that GPU scoring never depends on live tokenizer
            # objects or CPU-only algorithm instance state.
            args.cross_tokenizer_runtime_state = json.loads(
                json.dumps(runtime_state, ensure_ascii=False, allow_nan=False, sort_keys=True)
            )
            self.cross_tokenizer_vocab_limit = int(
                getattr(args, "padded_vocab_size", None) or self.hf_config.vocab_size
            )
            student_vocab_limit = min(
                self.cross_tokenizer_vocab_limit,
                int(runtime_state.get("student_vocab_size", self.cross_tokenizer_vocab_limit)),
            )
            teacher_vocab_limit = int(
                runtime_state.get(
                    "teacher_vocab_size",
                    max(shared_vocabulary.teacher_ids, default=-1) + 1
                    if shared_vocabulary is not None
                    else self.cross_tokenizer_vocab_limit,
                )
            )
            self.cross_tokenizer_teacher_vocab_limit = teacher_vocab_limit
            if shared_vocabulary is None:
                valid_pairs: list[tuple[int, int]] = []
            else:
                valid_pairs = [
                    (student_id, teacher_id)
                    for student_id, teacher_id in zip(
                        shared_vocabulary.student_ids, shared_vocabulary.teacher_ids, strict=True
                    )
                    if 0 <= student_id < student_vocab_limit
                    and 0 <= teacher_id < teacher_vocab_limit
                ]
                if not valid_pairs:
                    raise ValueError("No shared Teacher/Student tokens fit inside the model vocabulary")
                dropped_pairs = len(shared_vocabulary) - len(valid_pairs)
                if dropped_pairs:
                    logger.warning(
                        "Dropped %d shared token mappings outside effective Student/Teacher vocabularies %d/%d",
                        dropped_pairs,
                        student_vocab_limit,
                        teacher_vocab_limit,
                    )
            args.cross_tokenizer_student_vocab_ids = [pair[0] for pair in valid_pairs]
            args.cross_tokenizer_teacher_vocab_ids = [pair[1] for pair in valid_pairs]
            logger.info(
                "Cross-tokenizer OPD enabled: algorithm=%s, shared_vocab=%d, candidate_top_k=%d, temperature=%g",
                args.cross_tokenizer_algorithm,
                len(valid_pairs),
                args.cross_tokenizer_top_k,
                args.cross_tokenizer_temperature,
            )

        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id = initialize_model_and_optimizer(
            args, role
        )
        if (
            role == "actor"
            and getattr(args, "use_cross_tokenizer", False)
            and teacher_transport == "causal_hidden"
            and mpu.is_pipeline_last_stage(ignore_virtual=True)
        ):
            teacher_head, teacher_vocab_size = load_frozen_lm_head_shard(
                args.opd_teacher_model_path,
                tensor_parallel_rank=mpu.get_tensor_model_parallel_rank(),
                tensor_parallel_size=mpu.get_tensor_model_parallel_world_size(),
                device=torch.device("cuda", torch.cuda.current_device()),
            )
            teacher_text_config = getattr(
                self.teacher_hf_config, "text_config", self.teacher_hf_config
            )
            expected_teacher_vocab_size = int(teacher_text_config.vocab_size)
            expected_teacher_hidden_size = int(teacher_text_config.hidden_size)
            if teacher_vocab_size != expected_teacher_vocab_size:
                raise ValueError(
                    "Frozen cross-tokenizer Teacher LM-head vocabulary differs from Teacher config: "
                    f"head={teacher_vocab_size}, config={expected_teacher_vocab_size}"
                )
            if teacher_head.size(1) != expected_teacher_hidden_size:
                raise ValueError(
                    "Frozen cross-tokenizer Teacher LM-head hidden width differs from Teacher config: "
                    f"head={teacher_head.size(1)}, config={expected_teacher_hidden_size}"
                )
            expected_local_rows = (
                expected_teacher_vocab_size
                + mpu.get_tensor_model_parallel_world_size()
                - 1
            ) // mpu.get_tensor_model_parallel_world_size()
            if teacher_head.size(0) != expected_local_rows:
                raise ValueError(
                    "Frozen cross-tokenizer Teacher LM-head TP shard has an invalid row count: "
                    f"head={teacher_head.size(0)}, expected={expected_local_rows}"
                )
            set_cross_tokenizer_teacher_head_shard(
                teacher_head, vocab_size=teacher_vocab_size
            )
            self._cross_tokenizer_teacher_head = teacher_head

        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
        if vpp_size > 1:
            from megatron.core.utils import get_model_config

            microbatch_group_size_per_vp_stage = get_model_config(self.model[0]).microbatch_group_size_per_vp_stage
        else:
            microbatch_group_size_per_vp_stage = 1
        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
            "cp_size": mpu.get_context_parallel_world_size(),
            "vpp_size": vpp_size,
            "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
        }

        start_rollout_id = loaded_rollout_id + 1

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return start_rollout_id

        # A checkpoint conversion only needs the loaded DDP model.  Avoid a
        # redundant multi-gigabyte CPU backup before the Bridge HF exporter is
        # invoked by save_model().
        if self.args.export_hf_only:
            return start_rollout_id

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=args.megatron_to_hf_mode == "raw",
                translate_gpu_to_cpu=not self.args.enable_weights_backuper,
            ),
            single_tag=None if args.enable_weights_backuper else "actor",
        )
        self._active_model_tag: str | None = "actor"
        self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        # Load teacher model for Megatron-based on-policy distillation
        if with_opd_teacher:
            self.load_other_checkpoint("teacher", args.opd_teacher_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            # Prefer HF config vocab_size (which may include model-native padding)
            # over tokenizer vocab_size (which may be smaller, e.g. GPT-OSS).
            hf_vocab = getattr(self.hf_config, "vocab_size", None)
            self.args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size

        # Bridge-built models need the Bridge HF exporter; the raw distributed
        # updater uses a direct MCore mapper. For remote rollout engines, keep
        # NCCL transport and only swap the exporter.
        if self.args.colocate:
            update_weight_cls = UpdateWeightFromTensor
        elif self.args.megatron_to_hf_mode == "bridge":
            update_weight_cls = UpdateWeightFromDistributedBridge
        elif self.args.update_weight_mode == "delta":
            # Lazy import: the delta module pulls DeltaEncoding/DeltaParam/DeltaSpec from
            # sglang, which only exist on newer images. Importing eagerly would break old
            # images even when delta mode is unused.
            from .update_weight.update_weight_from_distributed_delta import UpdateWeightFromDistributedDelta

            update_weight_cls = UpdateWeightFromDistributedDelta
        else:
            update_weight_cls = UpdateWeightFromDistributed
        logger.info(
            "Using %s for weight updates (megatron_to_hf_mode=%s, colocate=%s)",
            update_weight_cls.__name__,
            self.args.megatron_to_hf_mode,
            self.args.colocate,
        )
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )

        # empty cache after initialization
        clear_memory()

        if self.args.offload_train:
            # recover to actor in the end.
            self._switch_model("actor")
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        if (
            self.role == "actor"
            and self.args.use_critic
            and not self.args.colocate
            and hasattr(self.weight_updater, "disconnect_rollout_engines")
        ):
            self.weight_updater.disconnect_rollout_engines()
        destroy_process_groups()

        torch_memory_saver.pause()

        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        torch_memory_saver.resume()

        clear_memory()
        reload_process_groups()
        print_memory("after wake_up model")

    def _get_rollout_data(self, rollout_data_ref: Box) -> RolloutBatch:
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will receive the data.
        rollout_data = process_rollout_data(
            self.args,
            rollout_data_ref,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
        )
        if getattr(self.args, "use_cross_tokenizer", False):
            rollout_data = ensure_student_terminal_eos(
                rollout_data,
                student_tokenizer=self.tokenizer,
                max_sequence_length=self.args.seq_length,
            )
        # TODO: this is ugly, move to somewhere else?
        # move tokens to GPU in advance
        rollout_data["tokens"] = [
            torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device()) for t in rollout_data["loss_masks"]
        ]
        if "rollout_mask_sums" in rollout_data:
            # Promote precomputed per-rollout mask totals to GPU tensors here
            # (matching loss_masks) so the loss reducer can just divide.
            rollout_data["rollout_mask_sums"] = torch.tensor(
                rollout_data["rollout_mask_sums"], dtype=torch.float32, device=torch.cuda.current_device()
            )
        if "multimodal_train_inputs" in rollout_data:
            # Move multimodal training tensors to GPU in advance
            rollout_data["multimodal_train_inputs"] = [
                (
                    {
                        key: (
                            torch.from_numpy(v.copy()).to(device=torch.cuda.current_device())
                            if isinstance(v, np.ndarray)
                            else v.to(device=torch.cuda.current_device())
                        )
                        for key, v in mm_dict.items()
                    }
                    if mm_dict is not None
                    else None
                )
                for mm_dict in rollout_data["multimodal_train_inputs"]
            ]

        self._set_bshd_max_seq_lens(rollout_data)

        for key in ["rollout_log_probs", "teacher_log_probs"]:
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                torch.tensor(
                    slice_log_prob_with_cp(
                        log_prob,
                        total_length,
                        response_length,
                        self.args.qkv_format,
                        rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                    ),
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
                for i, (log_prob, total_length, response_length) in enumerate(
                    zip(
                        rollout_data[key],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                )
            ]
        if "rollout_routed_experts" in rollout_data:
            rollout_data["rollout_routed_experts"] = [
                torch.from_numpy(r) for r in rollout_data["rollout_routed_experts"]
            ]
        return rollout_data

    def _set_bshd_max_seq_lens(self, rollout_data: RolloutBatch) -> None:
        if self.args.qkv_format != "bshd":
            rollout_data.pop("max_seq_lens", None)
            return
        pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
        max_seq_lens = [
            (length + pad_size - 1) // pad_size * pad_size for length in rollout_data["total_lengths"]
        ]
        for micro_batch in rollout_data.get("micro_batch_indices", []):
            if not micro_batch:
                continue
            mb_max_seq_len = max(max_seq_lens[i] for i in micro_batch)
            for i in micro_batch:
                max_seq_lens[i] = mb_max_seq_len
        rollout_data["max_seq_lens"] = max_seq_lens

    def _prepare_cross_tokenizer_teacher_data(self, rollout_data: RolloutBatch) -> RolloutBatch:
        if self.cross_tokenizer_algorithm is None or self.teacher_tokenizer is None:
            raise RuntimeError("Cross-tokenizer runtime was not initialized")
        teacher_data = prepare_teacher_rollout_data(
            rollout_data,
            student_tokenizer=self.tokenizer,
            teacher_tokenizer=self.teacher_tokenizer,
            algorithm=self.cross_tokenizer_algorithm,
            alignment_error_policy=self.args.cross_tokenizer_alignment_error_policy,
            teacher_vocab_limit=self.cross_tokenizer_teacher_vocab_limit,
            max_sequence_length=self.args.seq_length,
            apply_teacher_chat_template=self.args.apply_chat_template,
            teacher_chat_template_kwargs=self.args.apply_chat_template_kwargs,
            student_stop_ids=self.cross_tokenizer_student_stop_ids,
            teacher_stop_ids=self.cross_tokenizer_teacher_stop_ids,
            teacher_bridge_stop_id=self.cross_tokenizer_teacher_bridge_stop_id,
        )
        self._set_bshd_max_seq_lens(teacher_data)
        return teacher_data

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
            )

        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        from slime.utils.routing_replay import RoutingReplay

        if not RoutingReplay.all_routing_replays:
            raise RuntimeError(
                "R3 requested but no Slime routing hooks were registered. "
                "Check the Megatron router/moe_utils patches in the runtime image."
            )

        for iterator in data_iterator:
            iterator.reset()

        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        def pad_func(experts, pad):
            _, num_layers, topk = experts.shape
            pad = (
                torch.arange(
                    pad * num_layers * topk,
                    device=experts.device,
                    dtype=experts.dtype,
                ).reshape((pad, num_layers, topk))
                % self.args.num_experts
            )
            return torch.cat([experts, pad], dim=0)

        for _ in range(sum(num_microbatches)):
            replay_keys = ["rollout_routed_experts", "tokens"]
            if self.args.qkv_format == "bshd":
                replay_keys.append("max_seq_lens")
            batch = data_iterator[0].get_next(replay_keys)
            rollout_routed_experts = batch["rollout_routed_experts"]
            tokens = batch["tokens"]
            assert len(rollout_routed_experts) == len(tokens)
            for a, b in zip(rollout_routed_experts, tokens, strict=False):
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            if self.args.qkv_format == "bshd":
                rollout_routed_experts = prepare_bshd_routing_replay(
                    rollout_routed_experts, [len(t) for t in tokens], max(batch["max_seq_lens"]),
                    cp_size=mpu.get_context_parallel_world_size(), cp_rank=mpu.get_context_parallel_rank(),
                    tp_size=tp_size, tp_rank=tp_rank, sequence_parallel=self.args.sequence_parallel,
                    num_experts=self.args.num_experts,
                )
            else:
                # THD packs each sequence after its own zig-zag CP split.
                rollout_routed_experts = [pad_func(r, 1) for r in rollout_routed_experts]
                rollout_routed_experts = [slice_with_cp(r, pad_func) for r in rollout_routed_experts]
                rollout_routed_experts = torch.cat(rollout_routed_experts, dim=0)
                pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
                pad = (pad_size - rollout_routed_experts.size(0) % pad_size) % pad_size
                if pad != 0:
                    rollout_routed_experts = pad_func(rollout_routed_experts, pad)

                if self.args.sequence_parallel:
                    seqlen = rollout_routed_experts.size(0)
                    assert seqlen % tp_size == 0
                    start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                    rollout_routed_experts = rollout_routed_experts[start:end]

            routing_replay_offset = 0
            for vp_stage, model in enumerate(self.model):
                config = model.module.config
                num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    # skip dense layer
                    if isinstance(config.moe_layer_freq, int):
                        if layer_id % config.moe_layer_freq != 0:
                            continue
                    elif isinstance(config.moe_layer_freq, list):
                        assert len(config.moe_layer_freq) == config.num_layers
                        if config.moe_layer_freq[layer_id] == 0:
                            continue
                    layer_routed_experts = rollout_routed_experts[:, layer_id]
                    RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
                    routing_replay_offset += 1
            assert routing_replay_offset == len(RoutingReplay.all_routing_replays)

        del rollout_data["rollout_routed_experts"]
        logger.info(
            "[R3] Installed rollout routes for %s local MoE layers, %s microbatches; "
            "Student expert IDs are replayed and router scores remain differentiable.",
            len(RoutingReplay.all_routing_replays), sum(num_microbatches),
        )

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def compute_opd_student_topk(
        self, data_iterator: list[DataIterator], num_microbatches: list[int]
    ) -> dict[str, list[torch.Tensor]]:
        with timer("opd_student_topk"):
            return forward_only(
                get_opd_student_topk,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix="opd_student_",
            )

    def compute_opd_teacher_data(
        self, data_iterator: list[DataIterator], num_microbatches: list[int]
    ) -> dict[str, list[torch.Tensor]]:
        with timer("teacher_opd_distribution"):
            return forward_only(
                get_opd_teacher_data,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix="teacher_",
            )

    def compute_cross_tokenizer_teacher_data(
        self, data_iterator: list[DataIterator], num_microbatches: list[int]
    ) -> dict[str, list[torch.Tensor]]:
        with timer("teacher_cross_tokenizer_distribution"):
            return forward_only(
                get_cross_tokenizer_teacher_data,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix="teacher_",
            )

    def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):
        if self.args.debug_rollout_only:
            return None

        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        if self.role == "critic":
            result = self.train_critic(rollout_id, rollout_data)
        else:
            self.train_actor(rollout_id, rollout_data, external_data=external_data)
            result = None

        if self.args.offload_train:
            self.sleep()

        return result

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch):
        """Train critic and return CPU values (used as old-values for the next actor train)."""
        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        # Compute current critic values (used as old_values for value loss and for actor advantages).
        rollout_data.update(forward_only(get_values, self.args, self.model, data_iterator, num_microbatches))

        compute_advantages_and_returns(self.args, rollout_data)

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
            global_batch_sizes,
        )

        if mpu.is_pipeline_last_stage() and "values" in rollout_data:
            from slime.backends.megatron_utils.data import tensors_to_cpu

            return {"values": tensors_to_cpu(rollout_data["values"])}
        return {}

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch, external_data=None) -> None:
        # Create data iterator for log_probs and train.
        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )

                # Top-K union needs the current student support before the teacher forward.
                if (
                    "teacher" in self.weights_backuper.backup_tags
                    and not self.args.use_cross_tokenizer
                    and self.args.opd_loss_type == "union_kl"
                ):
                    self._switch_model("actor")
                    rollout_data.update(self.compute_opd_student_topk(data_iterator, num_microbatches))

                # Forward the teacher and cache exactly the payload required by the selected OPD loss.
                if "teacher" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("teacher")
                    if self.args.use_cross_tokenizer:
                        cross_teacher_rollout_data = self._prepare_cross_tokenizer_teacher_data(rollout_data)
                        cross_teacher_data_iterator = get_data_iterator(cross_teacher_rollout_data)
                        teacher_data = self.compute_cross_tokenizer_teacher_data(
                            cross_teacher_data_iterator, num_microbatches
                        )
                    elif self.args.opd_loss_type == "sampled":
                        teacher_data = self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="teacher_",
                        )
                    else:
                        teacher_data = self.compute_opd_teacher_data(data_iterator, num_microbatches)
                    rollout_data.update(teacher_data)
                    rollout_data.pop("opd_student_topk_ids", None)

                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                direct_cross_tokenizer_distillation = (
                    self.args.use_opd
                    and self.args.use_cross_tokenizer
                    and self.args.advantage_estimator == "on_policy_distillation"
                    and self.args.kl_coef == 0
                    and not self.args.use_rollout_logprobs
                    and not self.args.get_mismatch_metrics
                    and not self.args.use_critic
                    and not self.args.keep_old_actor
                    and not self.args.use_routing_replay
                )
                can_reuse_log_probs_in_loss = (
                    len(num_microbatches) == 1
                    and self.args.loss_type == "policy_loss"
                    and self.args.kl_coef == 0
                    and not self.args.use_rollout_logprobs
                    and not self.args.get_mismatch_metrics
                    and not self.args.use_critic
                    and not self.args.keep_old_actor
                    and not self.args.use_opd
                    and not self.args.use_routing_replay
                    and self.args.advantage_estimator != "gspo"
                )
                if (
                    not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics
                ) and not can_reuse_log_probs_in_loss and not direct_cross_tokenizer_distillation:
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                        )
                    )
                    if self.args.use_rollout_routing_replay:
                        RoutingReplay.clear_all_forward()

                if self.args.use_critic:
                    if external_data is not None and mpu.is_pipeline_last_stage():
                        values = external_data.get("values")
                        if values is not None:
                            from slime.backends.megatron_utils.data import tensors_to_gpu

                            rollout_data["values"] = tensors_to_gpu(values)
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args, rollout_id, rollout_data)

            log_rollout_data(
                rollout_id,
                self.args,
                rollout_data,
            )

            # Model-parallel groups overlap TP/PP/CP/EP axes, while the
            # preceding forward and logging work can finish at different times
            # on otherwise independent groups. Qwen3.6's interleaved GDN/MoE
            # path makes that skew large enough for an early rank to enter the
            # backward schedule while a peer is still issuing forward
            # collectives, which presents as mismatched NCCL sequence numbers.
            # Fence the complete actor world at the phase boundary.
            with timer("actor_train_phase_fence"):
                dist.barrier(group=get_gloo_group())

            # Train
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                    global_batch_sizes,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        # update the cpu actor weight to the latest model
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args, extra_metrics=self.weight_updater.pop_metrics())

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving.
        if self.args.offload_train:
            self.wake_up()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        if not self.args.export_hf_only:
            save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if not self.args.export_hf_only and force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from slime.backends.megatron_utils.model import save_hf_model

            save_hf_model(self.args, rollout_id, self.model)

        if self.args.offload_train:
            self.sleep()

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_updatable_engines_and_lock.remote()
        )

        reconnect_rollout_engines = self.args.offload_train and self.args.use_critic and not self.args.colocate

        if reconnect_rollout_engines:
            self.wake_up()
        elif self.args.offload_train:
            reload_process_groups()

        if num_new_engines > 0 or reconnect_rollout_engines:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0:
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if reconnect_rollout_engines:
            self.sleep()
        elif self.args.offload_train:
            destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        old_ckpt_step = None
        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step
        elif model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if old_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag
