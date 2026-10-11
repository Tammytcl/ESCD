#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from opd.cross_tokenizer import cross_tokenizer_capabilities
from opd.routing_replay import validate_cross_tokenizer_r3
from opd.rollout_horizon import normalize_horizon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile a cross-tokenizer OPD training config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--format", choices=("shell", "json"), default="shell")
    return parser.parse_args()


def require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping or mapping[key] in (None, ""):
        raise ValueError(f"Missing required field: {section}.{key}")
    return mapping[key]


def truthy(value: Any) -> str:
    return "1" if bool(value) else "0"


def parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(f"{field} must be a boolean value, got {value!r}")


def parse_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{field} must be an integer")
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{field} must be an integer") from error


def parse_json_object(value: Any, field: str) -> dict[str, Any]:
    if value in (None, ""):
        parsed: Any = {}
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{field} must be valid JSON") from error
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise ValueError(f"{field} must be a JSON object")

    def validate(item: Any, path: str) -> None:
        if item is None or isinstance(item, (bool, int, str)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{path} must not contain non-finite numbers")
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                validate(child, f"{path}[{index}]")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError(f"{path} keys must be strings")
                validate(child, f"{path}.{key}")
            return
        raise ValueError(f"{path} must contain only JSON values")

    validate(parsed, field)
    return json.loads(json.dumps(parsed, ensure_ascii=False, allow_nan=False, sort_keys=True))


def remap_path_prefix(value: str, source_prefix: str, target_prefix: str) -> str:
    """Map a mounted filesystem prefix without rewriting lookalike paths."""
    source = source_prefix.rstrip("/")
    target = target_prefix.rstrip("/")
    if not source or not target or source == target:
        return value
    if value == source:
        return target
    if value.startswith(source + "/"):
        return target + value[len(source) :]
    return value


def compile_config(config: dict[str, Any], environment: dict[str, str] | None = None) -> dict[str, str]:
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")

    run = config.get("run") or {}
    data = config.get("data") or {}
    loss = config.get("loss") or {}
    sampling = config.get("sampling") or {}
    parallel = config.get("parallel") or {}
    optimizer = config.get("optimizer") or {}
    sglang = config.get("sglang") or {}
    logging = config.get("logging") or {}
    reward = config.get("reward") or {}
    evaluation = config.get("evaluation") or {}
    cross_tokenizer = config.get("cross_tokenizer") or {}
    environment = environment or {}

    evaluation_mode = str(evaluation.get("mode", "disabled")).strip().lower()
    if evaluation_mode != "disabled":
        raise ValueError("Public training configs require evaluation.mode: disabled; evaluate saved checkpoints separately")
    if config.get("skill") is not None:
        raise ValueError("Skill experiments are not part of the training release")
    evaluation_interval_raw = evaluation.get("interval")
    evaluation_interval = ""
    if evaluation_interval_raw not in (None, ""):
        parsed_evaluation_interval = parse_integer(
            evaluation_interval_raw, "evaluation.interval"
        )
        if parsed_evaluation_interval <= 0:
            raise ValueError("evaluation.interval must be positive")
        evaluation_interval = str(parsed_evaluation_interval)

    opd_type = str(run.get("teacher_backend", "megatron")).strip().lower()
    if opd_type not in {"megatron", "sglang"}:
        raise ValueError("run.teacher_backend must be 'megatron' or 'sglang'")

    # ``teacher_backend`` describes the implementation used by Slime, while
    # ``teacher_deployment`` describes where that implementation runs.  Keep
    # the two axes separate so a manifest can move from an in-process
    # Megatron teacher to an external SGLang service without changing the
    # cross-tokenizer plugin contract.  The default preserves all historical
    # manifests: Megatron is in-process and SGLang is an HTTP endpoint.
    teacher_deployment = str(
        run.get(
            "teacher_deployment",
            "megatron_inprocess" if opd_type == "megatron" else "external_http",
        )
    ).strip().lower()
    deployment_aliases = {
        "inprocess": "megatron_inprocess",
        "megatron": "megatron_inprocess",
        "external": "external_http",
        "http": "external_http",
        "sglang": "external_http",
        "no_kv": "sglang_no_kv",
        "kimi_no_kv": "sglang_no_kv",
        "kimi_no_kv_sglang": "sglang_no_kv",
    }
    teacher_deployment = deployment_aliases.get(teacher_deployment, teacher_deployment)
    supported_deployments = {
        "megatron_inprocess",
        "external_http",
        "sglang_no_kv",
    }
    if teacher_deployment not in supported_deployments:
        raise ValueError(
            "run.teacher_deployment must be one of "
            f"{sorted(supported_deployments)}, got {teacher_deployment!r}"
        )
    if teacher_deployment == "megatron_inprocess" and opd_type != "megatron":
        raise ValueError(
            "run.teacher_deployment=megatron_inprocess requires "
            "run.teacher_backend=megatron"
        )
    if teacher_deployment == "sglang_no_kv" and opd_type != "sglang":
        raise ValueError(
            "run.teacher_deployment=sglang_no_kv requires "
            "run.teacher_backend=sglang"
        )

    megatron_to_hf_mode = str(run.get("megatron_to_hf_mode", "bridge")).strip().lower()
    if megatron_to_hf_mode not in {"bridge", "raw"}:
        raise ValueError("run.megatron_to_hf_mode must be 'bridge' or 'raw'")

    chat_kwargs = data.get("apply_chat_template_kwargs") or {}
    opd_loss_type = str(loss.get("opd_loss_type", "sampled"))
    supported_opd_loss_types = {"sampled", "union_kl", "teacher_topk_gkl"}
    if opd_loss_type not in supported_opd_loss_types:
        raise ValueError(
            f"loss.opd_loss_type must be one of {sorted(supported_opd_loss_types)}, got {opd_loss_type!r}"
        )
    opd_top_k = int(loss.get("opd_top_k", 20))
    if opd_top_k <= 0:
        raise ValueError("loss.opd_top_k must be positive")
    kl_loss_coef_raw = environment.get("KL_LOSS_COEF", loss.get("kl_loss_coef", 0.0))
    try:
        kl_loss_coef = float(kl_loss_coef_raw)
    except (TypeError, ValueError) as error:
        raise ValueError("loss.kl_loss_coef must be a finite non-negative number") from error
    if not math.isfinite(kl_loss_coef) or kl_loss_coef < 0:
        raise ValueError("loss.kl_loss_coef must be a finite non-negative number")
    use_kl_loss = parse_bool(
        environment.get("USE_KL_LOSS", loss.get("use_kl_loss", False)),
        "loss.use_kl_loss",
    ) or kl_loss_coef > 0
    kl_loss_type = str(environment.get("KL_LOSS_TYPE", loss.get("kl_loss_type", "k1"))).strip().lower()
    if kl_loss_type not in {"k1", "k2", "k3", "low_var_kl"}:
        raise ValueError("loss.kl_loss_type must be one of ['k1', 'k2', 'k3', 'low_var_kl']")
    use_unbiased_kl = parse_bool(
        environment.get("USE_UNBIASED_KL", loss.get("use_unbiased_kl", False)),
        "loss.use_unbiased_kl",
    )

    cross_tokenizer_enabled = parse_bool(
        environment.get("USE_CROSS_TOKENIZER", cross_tokenizer.get("enabled", False)),
        "cross_tokenizer.enabled",
    )
    cross_tokenizer_algorithm = str(
        environment.get("CROSS_TOKENIZER_ALGORITHM", cross_tokenizer.get("algorithm", "simct"))
    ).strip().lower()
    teacher_tokenizer = str(
        environment.get("OPD_TEACHER_TOKENIZER", cross_tokenizer.get("teacher_tokenizer") or "")
    ).strip()
    teacher_model_path = str(
        environment.get("OPD_TEACHER_MODEL_PATH", run.get("teacher_model_path") or "")
    ).strip()
    use_rollout_routing_replay = parse_bool(
        parallel.get("use_rollout_routing_replay", False),
        "parallel.use_rollout_routing_replay",
    )
    use_routing_replay = parse_bool(
        parallel.get("use_routing_replay", False), "parallel.use_routing_replay"
    ) or use_rollout_routing_replay
    if cross_tokenizer_enabled and use_rollout_routing_replay:
        validate_cross_tokenizer_r3(
            algorithm=cross_tokenizer_algorithm,
            teacher_backend=opd_type,
            teacher_deployment=teacher_deployment,
            post_process_path=str(reward.get("post_process_path") or ""),
        )
    try:
        algorithm_capabilities = cross_tokenizer_capabilities(cross_tokenizer_algorithm)
    except ValueError:
        # Installed entry-point plugins may only be present in the worker
        # image. Preserve that extension path while built-ins use their frozen
        # contracts below.
        algorithm_capabilities = None
    default_cross_tokenizer_top_k = (
        algorithm_capabilities.default_candidate_top_k
        if cross_tokenizer_enabled and algorithm_capabilities is not None
        else 128
    )
    cross_tokenizer_top_k_raw = environment.get(
        "CROSS_TOKENIZER_TOP_K",
        cross_tokenizer.get("candidate_top_k", default_cross_tokenizer_top_k),
    )
    cross_tokenizer_temperature_raw = environment.get(
        "CROSS_TOKENIZER_TEMPERATURE", cross_tokenizer.get("temperature", 1.0)
    )
    cross_tokenizer_algorithm_kwargs_raw = environment.get(
        "CROSS_TOKENIZER_ALGORITHM_KWARGS", cross_tokenizer.get("algorithm_kwargs", {})
    )
    alignment_error_policy = str(
        environment.get(
            "CROSS_TOKENIZER_ALIGNMENT_ERROR_POLICY",
            cross_tokenizer.get("alignment_error_policy", "error"),
        )
    ).strip()
    if cross_tokenizer_enabled:
        if not cross_tokenizer_algorithm:
            raise ValueError("cross_tokenizer.algorithm cannot be empty")
        cross_tokenizer_algorithm_kwargs = parse_json_object(
            cross_tokenizer_algorithm_kwargs_raw, "cross_tokenizer.algorithm_kwargs"
        )
        cross_tokenizer_top_k = parse_integer(
            cross_tokenizer_top_k_raw, "cross_tokenizer.candidate_top_k"
        )
        try:
            cross_tokenizer_temperature = float(cross_tokenizer_temperature_raw)
        except (TypeError, ValueError) as error:
            raise ValueError("cross_tokenizer.temperature must be a number") from error
        if algorithm_capabilities is not None:
            try:
                algorithm_capabilities.validate_runtime_config(
                    candidate_top_k=cross_tokenizer_top_k,
                    public_temperature=cross_tokenizer_temperature,
                    teacher_backend=opd_type,
                )
            except ValueError as error:
                raise ValueError(
                    f"cross_tokenizer.algorithm={cross_tokenizer_algorithm!r}: {error}"
                ) from error
        else:
            if cross_tokenizer_top_k < 0:
                raise ValueError("cross_tokenizer.candidate_top_k must be non-negative")
            if not math.isfinite(cross_tokenizer_temperature) or cross_tokenizer_temperature <= 0:
                raise ValueError("cross_tokenizer.temperature must be a positive finite number")
        if alignment_error_policy not in {"error", "skip"}:
            raise ValueError("cross_tokenizer.alignment_error_policy must be 'error' or 'skip'")
        if not teacher_tokenizer:
            raise ValueError("cross_tokenizer.teacher_tokenizer is required when cross-tokenizer mode is enabled")
        if opd_loss_type != "sampled":
            raise ValueError("loss.opd_loss_type must be 'sampled' when cross-tokenizer mode is enabled")
        if (
            algorithm_capabilities is not None
            and algorithm_capabilities.requires_loss_token_normalization
            and not parse_bool(
                loss.get("normalize_by_loss_tokens", True),
                "loss.normalize_by_loss_tokens",
            )
        ):
            raise ValueError(
                f"cross_tokenizer.algorithm={cross_tokenizer_algorithm!r} requires "
                "loss.normalize_by_loss_tokens=true"
            )
        if algorithm_capabilities is not None:
            recompute_loss_function = parse_bool(
                parallel.get("recompute_loss_function", False),
                "parallel.recompute_loss_function",
            )
            if algorithm_capabilities.requires_loss_recompute_disabled and recompute_loss_function:
                raise ValueError(
                    f"cross_tokenizer.algorithm={cross_tokenizer_algorithm!r} executes distributed "
                    "loss-side row gathers and requires parallel.recompute_loss_function=false; "
                    "enabling loss recomputation can reorder duplicate collectives and deadlock"
                )
            if algorithm_capabilities.requires_chat_template and not parse_bool(
                data.get("apply_chat_template", True),
                "data.apply_chat_template",
            ):
                raise ValueError(
                    f"cross_tokenizer.algorithm={cross_tokenizer_algorithm!r} requires "
                    "data.apply_chat_template=true"
                )
            if algorithm_capabilities.requires_thinking and not parse_bool(
                chat_kwargs.get("enable_thinking", False),
                "data.apply_chat_template_kwargs.enable_thinking",
            ):
                raise ValueError(
                    f"cross_tokenizer.algorithm={cross_tokenizer_algorithm!r} requires "
                    "data.apply_chat_template_kwargs.enable_thinking=true"
                )
        if (
            opd_type == "megatron"
            and str(run.get("model_config", "")) != str(run.get("teacher_model_config", ""))
        ):
            raise ValueError(
                "Cross-tokenizer OPD cannot use Slime's in-process Megatron weight-switch backend for heterogeneous "
                "Teacher/Student architectures: it loads both checkpoints into one model instance. "
                "Use a plugin/backend transport that supports an external Teacher."
            )
        teacher_transport = (
            algorithm_capabilities.teacher_transport(opd_type)
            if algorithm_capabilities is not None
            else "payload"
        )
        if teacher_transport == "causal_hidden":
            if not teacher_model_path:
                raise ValueError(
                    "run.teacher_model_path is required by the selected causal-hidden Teacher transport"
                )
            assert algorithm_capabilities is not None
            expected_reward = algorithm_capabilities.external_reward_path
            expected_post = algorithm_capabilities.external_post_process_path
            if str(reward.get("custom_rm_path") or "") != expected_reward:
                raise ValueError(
                    f"causal-hidden cross-tokenizer transport requires reward.custom_rm_path={expected_reward}"
                )
            if str(reward.get("post_process_path") or "") != expected_post:
                raise ValueError(
                    f"causal-hidden cross-tokenizer transport requires reward.post_process_path={expected_post}"
                )
        context_parallel_size = parse_integer(
            parallel.get("context_parallel_size", 1), "parallel.context_parallel_size"
        )
        if (
            context_parallel_size > 1
            and algorithm_capabilities is not None
            and not algorithm_capabilities.supports_context_parallel_for(opd_type)
        ):
            raise ValueError(
                f"cross_tokenizer.algorithm={cross_tokenizer_algorithm!r} does not support "
                "parallel.context_parallel_size > 1"
            )
    else:
        # Disabled cross-tokenizer settings are intentionally inert. This
        # preserves the native OPD path even when stale experiment exports are
        # present in the shell.
        cross_tokenizer_algorithm = cross_tokenizer_algorithm or "simct"
        alignment_error_policy = (
            alignment_error_policy if alignment_error_policy in {"error", "skip"} else "error"
        )
        try:
            cross_tokenizer_top_k = max(
                parse_integer(cross_tokenizer_top_k_raw, "cross_tokenizer.candidate_top_k"), 0
            )
        except ValueError:
            cross_tokenizer_top_k = 128
        try:
            cross_tokenizer_temperature = float(cross_tokenizer_temperature_raw)
            if not math.isfinite(cross_tokenizer_temperature) or cross_tokenizer_temperature <= 0:
                cross_tokenizer_temperature = 1.0
        except (TypeError, ValueError):
            cross_tokenizer_temperature = 1.0
        try:
            cross_tokenizer_algorithm_kwargs = parse_json_object(
                cross_tokenizer_algorithm_kwargs_raw, "cross_tokenizer.algorithm_kwargs"
            )
        except ValueError:
            cross_tokenizer_algorithm_kwargs = {}

    teacher_load = str(run.get("teacher_load") or "")
    if opd_type == "megatron" and not teacher_load:
        raise ValueError("Missing required field: run.teacher_load")

    # Mirror Megatron's ConstantNumMicroBatchesCalculator constraints here so
    # an invalid cluster topology fails during the no-cost config compile instead
    # of after every GPU process has initialized.  Context parallel ranks are
    # part of one model replica, so DP = world / (TP * PP * CP).
    actor_num_nodes = parse_integer(
        require(parallel, "actor_num_nodes", "parallel"),
        "parallel.actor_num_nodes",
    )
    actor_num_gpus_per_node = parse_integer(
        parallel.get("actor_num_gpus_per_node", 8),
        "parallel.actor_num_gpus_per_node",
    )
    tensor_parallel_size = parse_integer(
        parallel.get("tensor_parallel_size", 1),
        "parallel.tensor_parallel_size",
    )
    pipeline_parallel_size = parse_integer(
        parallel.get("pipeline_parallel_size", 1),
        "parallel.pipeline_parallel_size",
    )
    context_parallel_size = parse_integer(
        parallel.get("context_parallel_size", 1),
        "parallel.context_parallel_size",
    )
    expert_parallel_size = parse_integer(
        parallel.get("expert_parallel_size", 1),
        "parallel.expert_parallel_size",
    )
    # Megatron defaults expert tensor parallelism to dense TP when the CLI
    # option is omitted.  Keep that backward-compatible default, but compile
    # the value explicitly so MoE jobs never acquire a different expert
    # topology merely because a launcher forgot the flag.
    expert_tensor_parallel_size = parse_integer(
        parallel.get("expert_tensor_parallel_size", tensor_parallel_size),
        "parallel.expert_tensor_parallel_size",
    )
    qwen_gdn_backend = str(parallel.get("qwen_gdn_backend", "fla")).strip().lower()
    if qwen_gdn_backend not in {"fla", "flashqla"}:
        raise ValueError("parallel.qwen_gdn_backend must be 'fla' or 'flashqla'")
    micro_batch_size = parse_integer(
        parallel.get("micro_batch_size", 1),
        "parallel.micro_batch_size",
    )
    distributed_timeout_minutes = parse_integer(
        parallel.get("distributed_timeout_minutes", 10),
        "parallel.distributed_timeout_minutes",
    )
    train_memory_margin_bytes = parse_integer(
        parallel.get("train_memory_margin_bytes", 1024**3),
        "parallel.train_memory_margin_bytes",
    )
    global_batch_size = parse_integer(
        require(sampling, "global_batch_size", "sampling"),
        "sampling.global_batch_size",
    )
    topology_values = {
        "parallel.actor_num_nodes": actor_num_nodes,
        "parallel.actor_num_gpus_per_node": actor_num_gpus_per_node,
        "parallel.tensor_parallel_size": tensor_parallel_size,
        "parallel.pipeline_parallel_size": pipeline_parallel_size,
        "parallel.context_parallel_size": context_parallel_size,
        "parallel.expert_parallel_size": expert_parallel_size,
        "parallel.expert_tensor_parallel_size": expert_tensor_parallel_size,
        "parallel.micro_batch_size": micro_batch_size,
        "parallel.distributed_timeout_minutes": distributed_timeout_minutes,
        "parallel.train_memory_margin_bytes": train_memory_margin_bytes,
        "sampling.global_batch_size": global_batch_size,
    }
    for field, value in topology_values.items():
        if field == "parallel.train_memory_margin_bytes":
            if value < 0:
                raise ValueError(f"{field} must be non-negative, got {value}")
        elif value <= 0:
            raise ValueError(f"{field} must be positive, got {value}")
    actor_world_size = actor_num_nodes * actor_num_gpus_per_node
    model_parallel_size = (
        tensor_parallel_size * pipeline_parallel_size * context_parallel_size
    )
    if actor_world_size % model_parallel_size != 0:
        raise ValueError(
            "parallel actor world size must be divisible by tensor_parallel_size * "
            "pipeline_parallel_size * context_parallel_size: "
            f"world={actor_world_size}, model_parallel={model_parallel_size}"
        )
    data_parallel_size = actor_world_size // model_parallel_size
    # Megatron constructs expert groups without a CP axis.  Expert TP is an
    # independent axis and must not be inferred from dense TP here: the
    # Qwen3.6 multi-node gate deliberately uses TP=2 and ETP=1.
    expert_model_parallel_size = (
        expert_tensor_parallel_size * expert_parallel_size * pipeline_parallel_size
    )
    if actor_world_size % expert_model_parallel_size != 0:
        raise ValueError(
            "parallel actor world size must be divisible by expert_tensor_parallel_size * "
            "expert_parallel_size * pipeline_parallel_size: "
            f"world={actor_world_size}, tensor_parallel_size={tensor_parallel_size}, "
            f"expert_tensor_parallel_size={expert_tensor_parallel_size}, "
            f"pipeline_parallel_size={pipeline_parallel_size}, "
            f"context_parallel_size={context_parallel_size}, "
            f"expert_parallel_size={expert_parallel_size}, "
            f"expert_model_parallel={expert_model_parallel_size}"
        )
    batch_divisor = micro_batch_size * data_parallel_size
    if global_batch_size % batch_divisor != 0:
        raise ValueError(
            "sampling.global_batch_size must be divisible by parallel.micro_batch_size * "
            "derived data_parallel_size: "
            f"global_batch_size={global_batch_size}, micro_batch_size={micro_batch_size}, "
            f"data_parallel_size={data_parallel_size}, required_divisor={batch_divisor}"
        )

    env = {
        "JOB_NAME": str(require(run, "name", "run")),
        "SEED": str(run.get("seed", 1234)),
        "MODEL_CONFIG": str(require(run, "model_config", "run")),
        # Optional regexes for parameters that should remain frozen during
        # Megatron training.  Keep this as a list so a config can freeze a
        # Router without freezing the whole MoE block.
        "FREEZE_PARAMS_NAME_LIST": json.dumps(
            [str(pattern) for pattern in (run.get("freeze_params_name_list") or [])],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "TEACHER_MODEL_CONFIG": str(require(run, "teacher_model_config", "run")),
        "OPD_TYPE": opd_type,
        "OPD_TEACHER_DEPLOYMENT": teacher_deployment,
        "OPD_TEACHER_TRANSPORT": (
            "inprocess" if teacher_deployment == "megatron_inprocess" else "http"
        ),
        "MEGATRON_TO_HF_MODE": megatron_to_hf_mode,
        "MODEL_PATH": str(require(run, "model_path", "run")),
        "LOAD_PATH": str(require(run, "load_path", "run")),
        "REF_LOAD": str(run.get("ref_load") or run["load_path"]),
        "OPD_TEACHER_LOAD": teacher_load,
        "OPD_TEACHER_ENDPOINT_SUFFIX": str(
            run.get("teacher_endpoint_suffix", "generate")
        ),
        "OPD_BPM_ROUTING_SHA256": str(run.get("bpm_routing_sha256", "")),
        "SAVE_DIR": str(require(run, "save_dir", "run")),
        "SAVE_ROLLOUT_TEXT_DIR": str(run.get("rollout_text_dir") or ""),
        "SAVE_ROLLOUT_DIAGNOSTICS_DIR": str(
            environment.get("SAVE_ROLLOUT_DIAGNOSTICS_DIR", run.get("rollout_diagnostics_dir") or "")
        ),
        "SAVE_INTERVAL": str(run.get("save_interval", 10)),
        "TRAIN_DATA": str(require(data, "path", "data")),
        "RESET_ROLLOUT_DATA_STATE_ON_LOAD": truthy(data.get("reset_state_on_load", False)),
        "INPUT_KEY": str(data.get("input_key", "prompt")),
        # Empty means slime_opd_job.sh omits --label-key for prompt-only data.
        "LABEL_KEY": str(data.get("label_key") or ""),
        "METADATA_KEY": str(data.get("metadata_key", "metadata")),
        "APPLY_CHAT_TEMPLATE": truthy(data.get("apply_chat_template", True)),
        "APPLY_CHAT_TEMPLATE_KWARGS": json.dumps(chat_kwargs, separators=(",", ":")),
        "OPD_KL_COEF": str(loss.get("opd_kl_coef", 1.0)),
        "OPD_LOSS_TYPE": opd_loss_type,
        "OPD_TOP_K": str(opd_top_k),
        # Stable-OPD's reference divergence uses Slime's generic KL-loss
        # path.  Keep it opt-in so the historical OPD contract is unchanged;
        # a positive coefficient automatically enables the corresponding
        # reference forward pass in the launcher.
        "USE_KL_LOSS": "1" if use_kl_loss else "0",
        "KL_LOSS_COEF": str(kl_loss_coef),
        "KL_LOSS_TYPE": kl_loss_type,
        "USE_UNBIASED_KL": "1" if use_unbiased_kl else "0",
        "USE_CROSS_TOKENIZER": truthy(cross_tokenizer_enabled),
        "CROSS_TOKENIZER_ALGORITHM": cross_tokenizer_algorithm,
        "OPD_TEACHER_TOKENIZER": teacher_tokenizer,
        "OPD_TEACHER_MODEL_PATH": teacher_model_path or teacher_tokenizer,
        "CROSS_TOKENIZER_TOP_K": str(cross_tokenizer_top_k),
        "CROSS_TOKENIZER_TEMPERATURE": str(cross_tokenizer_temperature),
        "CROSS_TOKENIZER_ALGORITHM_KWARGS": json.dumps(
            cross_tokenizer_algorithm_kwargs,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "CROSS_TOKENIZER_ALIGNMENT_ERROR_POLICY": alignment_error_policy,
        "CALCULATE_PER_TOKEN_LOSS": truthy(loss.get("normalize_by_loss_tokens", True)),
        "USE_TEACHER_STUDENT_LOGPROB_CLIP": truthy(loss.get("use_teacher_student_logprob_clip", False)),
        "TEACHER_STUDENT_LOGPROB_CLIP_RANGE": str(loss.get("teacher_student_logprob_clip_range", 10.0)),
        # Keep ``num_rollout`` and ``num_epoch`` mutually exclusive at the
        # shell boundary.  Slime derives the rollout count from the loaded
        # global dataset when ``num_epoch`` is present, so do not force a
        # synthetic fixed default in that mode.
        "NUM_ROLLOUT": str(sampling.get("num_rollout", "")),
        "NUM_EPOCH": str(sampling.get("num_epoch", "")),
        "ROLLOUT_BATCH_SIZE": str(require(sampling, "rollout_batch_size", "sampling")),
        "OVER_SAMPLING_BATCH_SIZE": str(
            sampling.get("over_sampling_batch_size", require(sampling, "rollout_batch_size", "sampling"))
        ),
        "N_SAMPLES_PER_PROMPT": str(sampling.get("samples_per_prompt", 1)),
        "GLOBAL_BATCH_SIZE": str(global_batch_size),
        "ROLLOUT_MAX_PROMPT_LEN": str(sampling.get("max_prompt_len", 2048)),
        "ROLLOUT_MAX_RESPONSE_LEN": str(require(sampling, "max_response_len", "sampling")),
        "ROLLOUT_HORIZON": json.dumps(
            normalize_horizon(sampling.get("horizon"), sampling.get("max_response_len")),
            sort_keys=True,
        ),
        "DYNAMIC_SAMPLING_FILTER_PATH": str(sampling.get("dynamic_sampling_filter_path") or ""),
        "ROLLOUT_TEMPERATURE": str(sampling.get("temperature", 1.0)),
        "ROLLOUT_TOP_P": str(sampling.get("top_p", 1.0)),
        "ROLLOUT_TOP_K": str(sampling.get("top_k", -1)),
        "ROLLOUT_SEED": str(sampling.get("seed", 42)),
        "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": truthy(
            sampling.get("deterministic_inference", False)
        ),
        "ACTOR_NUM_NODES": str(actor_num_nodes),
        "ACTOR_NUM_GPUS_PER_NODE": str(actor_num_gpus_per_node),
        "ROLLOUT_NUM_GPUS": str(require(parallel, "rollout_num_gpus", "parallel")),
        "ROLLOUT_NUM_GPUS_PER_ENGINE": str(parallel.get("rollout_num_gpus_per_engine", 8)),
        "TP_SIZE": str(tensor_parallel_size),
        "PP_SIZE": str(pipeline_parallel_size),
        "CP_SIZE": str(context_parallel_size),
        "EP_SIZE": str(expert_parallel_size),
        "ETP_SIZE": str(expert_tensor_parallel_size),
        "QWEN_GDN_BACKEND": qwen_gdn_backend,
        "MICRO_BATCH_SIZE": str(micro_batch_size),
        "DISTRIBUTED_TIMEOUT_MINUTES": str(distributed_timeout_minutes),
        "TRAIN_MEMORY_MARGIN_BYTES": str(train_memory_margin_bytes),
        "SEQ_LENGTH": str(require(parallel, "sequence_length", "parallel")),
        "QKV_FORMAT": str(parallel.get("qkv_format", "bshd")),
        "COLOCATE": truthy(parallel.get("colocate", False)),
        "OFFLOAD_TRAIN": truthy(parallel.get("offload_train", False)),
        "OFFLOAD_ROLLOUT": truthy(parallel.get("offload_rollout", False)),
        "USE_DYNAMIC_BATCH_SIZE": truthy(
            parallel.get("use_dynamic_batch_size", False)
        ),
        "MAX_TOKENS_PER_GPU": str(parallel.get("max_tokens_per_gpu") or ""),
        "LOG_PROBS_MAX_TOKENS_PER_GPU": str(
            parallel.get("log_probs_max_tokens_per_gpu") or ""
        ),
        "LOG_PROBS_CHUNK_SIZE": str(parallel.get("log_probs_chunk_size", 1024)),
        "RECOMPUTE_NUM_LAYERS": str(parallel.get("recompute_num_layers", 1)),
        "RECOMPUTE_LOSS_FUNCTION": truthy(parallel.get("recompute_loss_function", False)),
        "USE_ROUTING_REPLAY": truthy(use_routing_replay),
        "USE_ROLLOUT_ROUTING_REPLAY": truthy(use_rollout_routing_replay),
        # Slime's custom Megatron loop needs an explicit opt-in for routing
        # diagnostics.  Keep the default off so historical runs do not pay
        # the extra small all-reduce or W&B metric cost.
        "MOE_LOAD_DIAGNOSTICS": truthy(parallel.get("moe_load_diagnostics", False)),
        "MOE_DIAGNOSTICS_LOG_INTERVAL": str(
            parallel.get("moe_diagnostics_log_interval", 10)
        ),
        "MOE_DIAGNOSTICS_PER_LAYER": truthy(
            parallel.get("moe_diagnostics_per_layer", True)
        ),
        "LR": str(optimizer.get("learning_rate", 1.0e-6)),
        "WEIGHT_DECAY": str(optimizer.get("weight_decay", 0.1)),
        "OPTIMIZER_CPU_OFFLOAD": truthy(optimizer.get("cpu_offload", True)),
        "OVERLAP_CPU_OPTIMIZER": truthy(
            optimizer.get("overlap_cpu_optimizer", True)
        ),
        "USE_PRECISION_AWARE_OPTIMIZER": truthy(
            optimizer.get("precision_aware", True)
        ),
        "OVERRIDE_OPT_PARAM_SCHEDULER": truthy(optimizer.get("override_scheduler", False)),
        "SGLANG_SERVER_CONCURRENCY": str(sglang.get("server_concurrency", 512)),
        "SGLANG_MAX_RUNNING_REQUESTS": str(sglang.get("max_running_requests", 4)),
        "SGLANG_CHUNKED_PREFILL_SIZE": str(sglang.get("chunked_prefill_size", 4096)),
        "SGLANG_CONTEXT_LENGTH": str(sglang.get("context_length") or ""),
        "SGLANG_PIPELINE_PARALLEL_SIZE": str(
            sglang.get("pipeline_parallel_size", 1)
        ),
        "SGLANG_EXPERT_PARALLEL_SIZE": str(
            sglang.get("expert_parallel_size", 1)
        ),
        "SGLANG_ATTENTION_BACKEND": str(sglang.get("attention_backend") or ""),
        "SGLANG_CUDA_GRAPH_BS": " ".join(
            str(value) for value in (sglang.get("cuda_graph_bs") or [])
        ),
        "SGLANG_DISABLE_RADIX_CACHE": truthy(
            sglang.get("disable_radix_cache", False)
        ),
        "SGLANG_DISABLE_OVERLAP_SCHEDULE": truthy(
            sglang.get("disable_overlap_schedule", False)
        ),
        "SGLANG_DISABLE_CUDA_GRAPH": truthy(sglang.get("disable_cuda_graph", True)),
        # SGLang currently enables the experimental piecewise graph path by
        # default even when full CUDA graphs are disabled.  On the Qwen3 MoE
        # rollout TP/PP shape this can make ranks enter different barriers
        # during bootstrap; keep the two switches independent and explicit.
        "SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH": truthy(
            sglang.get("disable_piecewise_cuda_graph", True)
        ),
        "SGLANG_MEM_FRACTION_STATIC": str(sglang.get("memory_fraction_static", 0.7)),
        "CUSTOM_RM_PATH": str(reward.get("custom_rm_path") or "opd.reward.zero_reward"),
        "CUSTOM_REWARD_POST_PROCESS_PATH": str(reward.get("post_process_path") or ""),
        "REWARD_KEY": str(reward.get("key") or ""),
        "EVAL_REWARD_KEY": str(evaluation.get("reward_key") or reward.get("key") or ""),
        # Only inline mode reaches Slime's synchronous eval loop. Platform
        # modes are consumed outside the optimizer process.
        "EVAL_INTERVAL": evaluation_interval if evaluation_mode == "inline" else "",
        "EVAL_CONFIG": str(evaluation.get("config") or ""),
        "SKIP_EVAL_BEFORE_TRAIN": truthy(evaluation.get("skip_before_train", True)),
        "LOG_PASSRATE": truthy(
            evaluation_mode == "inline" and evaluation.get("log_passrate", False)
        ),
        "ENABLE_WANDB": truthy(logging.get("enable_wandb", False)),
        "WANDB_PROJECT": str(logging.get("project", "cross-tokenizer-opd-research")),
        "WANDB_ENTITY": str(logging.get("entity") or ""),
    }

    env["OPD_SKILL_CONFIG_JSON"] = ""
    env["DATA_SOURCE_PATH"] = ""

    # Optional explicit mapping between host and container path prefixes.
    path_prefix_from = environment.get("OPD_PATH_PREFIX_FROM", "")
    path_prefix_to = environment.get("OPD_PATH_PREFIX_TO", "")
    if path_prefix_from and path_prefix_to:
        for key in (
            "MODEL_PATH",
            "LOAD_PATH",
            "REF_LOAD",
            "OPD_TEACHER_LOAD",
            "SAVE_DIR",
            "SAVE_ROLLOUT_TEXT_DIR",
            "SAVE_ROLLOUT_DIAGNOSTICS_DIR",
            "TRAIN_DATA",
            "OPD_TEACHER_TOKENIZER",
            "OPD_TEACHER_MODEL_PATH",
            "EVAL_CONFIG",
        ):
            env[key] = remap_path_prefix(env[key], path_prefix_from, path_prefix_to)
    return env


def main() -> int:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    env = compile_config(config, environment=dict(os.environ))
    if args.format == "json":
        print(json.dumps(env, indent=2, sort_keys=True))
    else:
        for key, value in env.items():
            print(f"export {key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
