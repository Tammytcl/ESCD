#!/usr/bin/env bash
set -euo pipefail

# Keep line-oriented training diagnostics visible in cluster.  TorchMemorySaver
# (used by the offloaded Megatron actor) rejects PyTorch's expandable-segment
# allocator, so do not install that default in the student process.  Callers
# may still opt into a different allocator explicitly for non-TMS jobs.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export SLIME_PLAIN_PROGRESS_LOG="${SLIME_PLAIN_PROGRESS_LOG:-1}"
export NCCL_DEBUG="${OPD_NCCL_DEBUG:-WARN}"
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF
else
  unset PYTORCH_CUDA_ALLOC_CONF
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SLIME_ROOT="${SLIME_ROOT:-${PROJECT_ROOT}/slime}"
DRY_RUN="${DRY_RUN:-${RAY_JOB_DRY_RUN:-0}}"
EMPTY_JSON_OBJECT='{}'
if [[ -n "${OPD_CONFIG:-}" ]]; then
  COMPILED_OPD_ENV="$(python3 "${PROJECT_ROOT}/scripts/training/compile_config.py" --config "${OPD_CONFIG}" --format shell)"
  eval "${COMPILED_OPD_ENV}"
fi
# Submission wrappers can request a bounded canary without cloning the full
# formal YAML.  Keep these overrides explicit and validate them before Ray
# starts so a typo cannot turn into an accidental long run.
if [[ -n "${NUM_ROLLOUT_OVERRIDE:-}" ]]; then
  [[ "${NUM_ROLLOUT_OVERRIDE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "NUM_ROLLOUT_OVERRIDE must be a positive integer" >&2
    exit 2
  }
  NUM_ROLLOUT="${NUM_ROLLOUT_OVERRIDE}"
  NUM_EPOCH=""
fi
if [[ -n "${SAVE_INTERVAL_OVERRIDE:-}" ]]; then
  [[ "${SAVE_INTERVAL_OVERRIDE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "SAVE_INTERVAL_OVERRIDE must be a positive integer" >&2
    exit 2
  }
  SAVE_INTERVAL="${SAVE_INTERVAL_OVERRIDE}"
fi
if [[ -n "${ROLLOUT_MAX_RESPONSE_LEN_OVERRIDE:-}" ]]; then
  [[ "${ROLLOUT_MAX_RESPONSE_LEN_OVERRIDE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ROLLOUT_MAX_RESPONSE_LEN_OVERRIDE must be a positive integer" >&2
    exit 2
  }
  ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN_OVERRIDE}"
fi
JOB_NAME="${JOB_NAME_OVERRIDE:-${JOB_NAME:-escd-training}}"
SAVE_DIR="${SAVE_DIR_OVERRIDE:-${SAVE_DIR:-${PROJECT_ROOT}/outputs/checkpoints/${JOB_NAME}}}"
SAVE_ROLLOUT_TEXT_DIR="${SAVE_ROLLOUT_TEXT_DIR_OVERRIDE:-${SAVE_ROLLOUT_TEXT_DIR:-}}"
SAVE_ROLLOUT_DIAGNOSTICS_DIR="${SAVE_ROLLOUT_DIAGNOSTICS_DIR_OVERRIDE:-${SAVE_ROLLOUT_DIAGNOSTICS_DIR:-}}"
MODEL_PATH="${MODEL_PATH_OVERRIDE:-${MODEL_PATH}}"
LOAD_PATH="${LOAD_PATH_OVERRIDE:-${LOAD_PATH}}"
REF_LOAD="${REF_LOAD_OVERRIDE:-${REF_LOAD:-${LOAD_PATH}}}"
TRAIN_DATA="${TRAIN_DATA_OVERRIDE:-${TRAIN_DATA}}"

# Megatron distributed checkpoints are loaded from their root directory; the
# root contains latest_checkpointed_iteration.txt, which selects the iter_*
# subdirectory.  A direct iter_* path looks non-empty and can therefore pass
# the old checks, but Megatron then cannot resolve its metadata and silently
# starts from random weights.  Release checkpoints are a valid exception:
# Megatron writes ``release`` plus a ``release/`` directory for pretrained
# initialization (there is no numbered iteration yet).  Fail fast only for
# roots that are neither a valid numbered checkpoint nor a valid release.
if [[ "${LOAD_PATH:-}" =~ /iter_[0-9]{7}$ ]]; then
  echo "ERROR: LOAD_PATH must be a checkpoint root with latest_checkpointed_iteration.txt, not an iter_* directory: ${LOAD_PATH}" >&2
  echo "       Create a checkpoint view (tracker + iter symlink) for a pinned iteration." >&2
  exit 2
fi
if [[ -n "${LOAD_PATH:-}" && -f "${LOAD_PATH}/latest_checkpointed_iteration.txt" ]]; then
  tracked_iteration="$(tr -d '[:space:]' <"${LOAD_PATH}/latest_checkpointed_iteration.txt")"
  if [[ "${tracked_iteration}" == "release" ]]; then
    if [[ ! -d "${LOAD_PATH}/release" ]]; then
      echo "ERROR: release checkpoint tracker exists but ${LOAD_PATH}/release is missing" >&2
      exit 2
    fi
  elif [[ ! "${tracked_iteration}" =~ ^[0-9]+$ || ! -d "${LOAD_PATH}/iter_$(printf '%07d' "${tracked_iteration}")" ]]; then
    echo "ERROR: checkpoint root has an invalid latest iteration: ${LOAD_PATH}/latest_checkpointed_iteration.txt" >&2
    exit 2
  fi
fi
if [[ -n "${SKIP_EVAL_BEFORE_TRAIN_OVERRIDE:-}" ]]; then
  SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN_OVERRIDE}"
fi

MODEL_CONFIG="${MODEL_CONFIG:-qwen3.5-27B}"
MODEL_CONFIG_PATH="${SLIME_ROOT}/scripts/models/${MODEL_CONFIG}.sh"
if [[ ! -f "${MODEL_CONFIG_PATH}" ]]; then
  echo "Missing slime model config: ${MODEL_CONFIG_PATH}" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "${MODEL_CONFIG_PATH}"

is_truthy() {
  local value="${1:-}"
  [[ "${value}" == "1" || "${value,,}" == "true" || "${value,,}" == "yes" ]]
}

# Preserve the paper's rollout cap exactly.  Cross-tokenizer training must not
# append EOS after generation: a length-truncated response has no realized
# stop and therefore no BPM stop bridge.
EFFECTIVE_ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN}"
if is_truthy "${USE_CROSS_TOKENIZER:-0}"; then
  if [[ ! "${ROLLOUT_MAX_RESPONSE_LEN}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ROLLOUT_MAX_RESPONSE_LEN must be a positive integer when USE_CROSS_TOKENIZER=1" >&2
    exit 2
  fi
fi

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_PATH}"
  --load "${LOAD_PATH}"
  --ref-load "${REF_LOAD:-${LOAD_PATH}}"
  --save "${SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL:-10}"
  --seed "${SEED:-1234}"
  --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE:-bridge}"
)
if [[ -n "${SAVE_HF:-}" ]]; then
  CKPT_ARGS+=(--save-hf "${SAVE_HF}")
fi

ROLLOUT_ARGS=(
  --prompt-data "${TRAIN_DATA}"
  --input-key "${INPUT_KEY:-prompt}"
  --metadata-key "${METADATA_KEY:-metadata}"
  --rollout-shuffle
  --custom-rm-path "${CUSTOM_RM_PATH:-opd.reward.zero_reward}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-1}"
  --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN:-2048}"
  --rollout-max-response-len "${EFFECTIVE_ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
  --rollout-top-p "${ROLLOUT_TOP_P:-1.0}"
  --rollout-top-k "${ROLLOUT_TOP_K:--1}"
  --rollout-seed "${ROLLOUT_SEED:-42}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)
if [[ -n "${LABEL_KEY:-}" ]]; then
  ROLLOUT_ARGS+=(--label-key "${LABEL_KEY}")
fi
if [[ -n "${ROLLOUT_HORIZON:-}" ]]; then
  ROLLOUT_ARGS+=(--rollout-horizon "${ROLLOUT_HORIZON}")
fi
if [[ -n "${DATA_SOURCE_PATH:-}" ]]; then
  ROLLOUT_ARGS+=(--data-source-path "${DATA_SOURCE_PATH}")
fi
# ``num_epoch`` is resolved by Slime after it constructs the global dataset,
# which is the only reliable way to consume a complete dataset when its size
# or batching changes.  Do not pass a fixed --num-rollout alongside it.
if [[ -n "${NUM_EPOCH:-}" ]]; then
  ROLLOUT_ARGS+=(--num-epoch "${NUM_EPOCH}")
else
  ROLLOUT_ARGS+=(--num-rollout "${NUM_ROLLOUT:-1000}")
fi
if [[ -n "${OVER_SAMPLING_BATCH_SIZE:-}" ]]; then
  ROLLOUT_ARGS+=(--over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}")
fi
if [[ -n "${DYNAMIC_SAMPLING_FILTER_PATH:-}" ]]; then
  ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}")
fi
if [[ -n "${CUSTOM_REWARD_POST_PROCESS_PATH:-}" ]]; then
  ROLLOUT_ARGS+=(--custom-reward-post-process-path "${CUSTOM_REWARD_POST_PROCESS_PATH}")
fi
if [[ -n "${REWARD_KEY:-}" ]]; then
  ROLLOUT_ARGS+=(--reward-key "${REWARD_KEY}")
fi

if is_truthy "${APPLY_CHAT_TEMPLATE:-1}"; then
  ROLLOUT_ARGS+=(--apply-chat-template)
fi
if [[ -n "${APPLY_CHAT_TEMPLATE_KWARGS:-}" ]]; then
  ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}")
fi

ALGO_ARGS=(
  --advantage-estimator on_policy_distillation
  --use-opd
  --opd-type "${OPD_TYPE:-megatron}"
  --opd-kl-coef "${OPD_KL_COEF:-1.0}"
  --opd-loss-type "${OPD_LOSS_TYPE:-sampled}"
  --opd-top-k "${OPD_TOP_K:-20}"
  --kl-loss-coef "${KL_LOSS_COEF:-0.0}"
  --kl-coef 0.0
  --entropy-coef 0.0
)
if is_truthy "${USE_KL_LOSS:-0}"; then
  ALGO_ARGS+=(--use-kl-loss --kl-loss-type "${KL_LOSS_TYPE:-k1}")
  if is_truthy "${USE_UNBIASED_KL:-0}"; then
    ALGO_ARGS+=(--use-unbiased-kl)
  fi
fi
if [[ "${OPD_TYPE:-megatron}" == "megatron" ]]; then
  ALGO_ARGS+=(--opd-teacher-load "${OPD_TEACHER_LOAD}")
fi
if is_truthy "${USE_CROSS_TOKENIZER:-0}"; then
  if [[ -z "${OPD_TEACHER_TOKENIZER:-}" ]]; then
    echo "OPD_TEACHER_TOKENIZER is required when USE_CROSS_TOKENIZER=1" >&2
    exit 2
  fi
  ALGO_ARGS+=(
    --use-cross-tokenizer
    --cross-tokenizer-algorithm "${CROSS_TOKENIZER_ALGORITHM:-simct}"
    --cross-tokenizer-algorithm-kwargs "${CROSS_TOKENIZER_ALGORITHM_KWARGS:-${EMPTY_JSON_OBJECT}}"
    --opd-teacher-tokenizer "${OPD_TEACHER_TOKENIZER}"
    --opd-teacher-model-path "${OPD_TEACHER_MODEL_PATH:-${OPD_TEACHER_TOKENIZER}}"
    --cross-tokenizer-top-k "${CROSS_TOKENIZER_TOP_K:-0}"
    --cross-tokenizer-temperature "${CROSS_TOKENIZER_TEMPERATURE:-1.0}"
    --cross-tokenizer-alignment-error-policy "${CROSS_TOKENIZER_ALIGNMENT_ERROR_POLICY:-error}"
  )
elif [[ "${OPD_LOSS_TYPE:-sampled}" == "sampled" ]] && is_truthy "${USE_TEACHER_STUDENT_LOGPROB_CLIP:-0}"; then
  ALGO_ARGS+=(--use-teacher-student-logprob-clip --teacher-student-logprob-clip-range "${TEACHER_STUDENT_LOGPROB_CLIP_RANGE:-10.0}")
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR:-1e-6}"
  --lr-decay-style constant
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --adam-beta1 0.9
  --adam-beta2 0.98
)
if is_truthy "${OPTIMIZER_CPU_OFFLOAD:-1}"; then
  OPTIMIZER_ARGS+=(--optimizer-cpu-offload)
fi
if is_truthy "${OVERLAP_CPU_OPTIMIZER:-1}"; then
  OPTIMIZER_ARGS+=(--overlap-cpu-optimizer-d2h-h2d)
fi
if is_truthy "${USE_PRECISION_AWARE_OPTIMIZER:-1}"; then
  OPTIMIZER_ARGS+=(--use-precision-aware-optimizer)
fi
if is_truthy "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}"; then
  OPTIMIZER_ARGS+=(--override-opt-param-scheduler)
fi

PERF_ARGS=(
  --train-backend megatron
  --qwen-gdn-backend "${QWEN_GDN_BACKEND:-fla}"
  --tensor-model-parallel-size "${TP_SIZE:-4}"
  --pipeline-model-parallel-size "${PP_SIZE:-4}"
  --context-parallel-size "${CP_SIZE:-1}"
  --expert-model-parallel-size "${EP_SIZE:-1}"
  --expert-tensor-parallel-size "${ETP_SIZE:-${TP_SIZE:-4}}"
  --sequence-parallel
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}"
  --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES:-10}"
  --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES:-1073741824}"
  --seq-length "${SEQ_LENGTH}"
  --qkv-format "${QKV_FORMAT:-bshd}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-1024}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)
if is_truthy "${MOE_LOAD_DIAGNOSTICS:-0}"; then
  # The hook is installed before model construction by Slime's Megatron
  # initializer.  Propagate its switches through ``train_env_vars`` because
  # Ray actor environments do not inherit arbitrary driver variables.
  PERF_ARGS+=(--custom-megatron-init-path slime.utils.moe_diagnostics.install)
  MOE_TRAIN_ENV_VARS_JSON="$(python3 - "${TRAIN_ENV_VARS_JSON:-{}}" <<'PY'
import json
import os
import sys

try:
    values = json.loads(sys.argv[1])
except (TypeError, json.JSONDecodeError):
    values = {}
if not isinstance(values, dict):
    values = {}
values.update(
    {
        "MOE_LOAD_DIAGNOSTICS": "1",
        "MOE_DIAGNOSTICS_LOG_INTERVAL": os.environ.get("MOE_DIAGNOSTICS_LOG_INTERVAL", "10"),
        "MOE_DIAGNOSTICS_PER_LAYER": os.environ.get("MOE_DIAGNOSTICS_PER_LAYER", "1"),
    }
)
print(json.dumps(values, separators=(",", ":")))
PY
)"
else
  MOE_TRAIN_ENV_VARS_JSON="${TRAIN_ENV_VARS_JSON:-{}}"
fi
if is_truthy "${USE_DYNAMIC_BATCH_SIZE:-0}"; then
  [[ -n "${MAX_TOKENS_PER_GPU:-}" ]] || {
    echo "MAX_TOKENS_PER_GPU is required when USE_DYNAMIC_BATCH_SIZE=1" >&2
    exit 2
  }
  PERF_ARGS+=(--use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}")
  if [[ -n "${LOG_PROBS_MAX_TOKENS_PER_GPU:-}" ]]; then
    PERF_ARGS+=(--log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU}")
  fi
fi
if is_truthy "${CALCULATE_PER_TOKEN_LOSS:-1}"; then
  PERF_ARGS+=(--calculate-per-token-loss)
fi
if is_truthy "${RECOMPUTE_LOSS_FUNCTION:-0}"; then
  PERF_ARGS+=(--recompute-loss-function)
fi
if is_truthy "${USE_ROLLOUT_ROUTING_REPLAY:-0}"; then
  PERF_ARGS+=(--use-rollout-routing-replay)
elif is_truthy "${USE_ROUTING_REPLAY:-0}"; then
  PERF_ARGS+=(--use-routing-replay)
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-512}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.70}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-4}"
  --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE:-4096}"
)
if [[ -n "${SGLANG_CONTEXT_LENGTH:-}" ]]; then
  SGLANG_ARGS+=(--sglang-context-length "${SGLANG_CONTEXT_LENGTH}")
fi
if [[ "${SGLANG_PIPELINE_PARALLEL_SIZE:-1}" != "1" ]]; then
  SGLANG_ARGS+=(--sglang-pipeline-parallel-size "${SGLANG_PIPELINE_PARALLEL_SIZE}")
fi
if [[ "${SGLANG_EXPERT_PARALLEL_SIZE:-1}" != "1" ]]; then
  SGLANG_ARGS+=(--sglang-expert-parallel-size "${SGLANG_EXPERT_PARALLEL_SIZE}")
fi
if [[ -n "${SGLANG_ATTENTION_BACKEND:-}" ]]; then
  SGLANG_ARGS+=(--sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}")
fi
if [[ -n "${SGLANG_CUDA_GRAPH_BS:-}" ]]; then
  read -r -a SGLANG_CUDA_GRAPH_BS_VALUES <<<"${SGLANG_CUDA_GRAPH_BS}"
  SGLANG_ARGS+=(--sglang-cuda-graph-bs "${SGLANG_CUDA_GRAPH_BS_VALUES[@]}")
fi
if is_truthy "${SGLANG_DISABLE_RADIX_CACHE:-0}"; then
  SGLANG_ARGS+=(--sglang-disable-radix-cache)
fi
if is_truthy "${SGLANG_DISABLE_OVERLAP_SCHEDULE:-0}"; then
  SGLANG_ARGS+=(--sglang-disable-overlap-schedule)
fi
if is_truthy "${SGLANG_DISABLE_CUDA_GRAPH:-1}"; then
  SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi
if is_truthy "${SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH:-1}"; then
  SGLANG_ARGS+=(--sglang-disable-piecewise-cuda-graph)
fi
if is_truthy "${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-0}"; then
  SGLANG_ARGS+=(--sglang-enable-deterministic-inference)
fi

CLUSTER_ARGS=(
  --actor-num-nodes "${ACTOR_NUM_NODES}"
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE:-8}"
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
)
if is_truthy "${COLOCATE:-0}"; then
  CLUSTER_ARGS+=(--colocate)
else
  if is_truthy "${OFFLOAD_TRAIN:-0}"; then
    CLUSTER_ARGS+=(--offload-train)
  fi
  if is_truthy "${OFFLOAD_ROLLOUT:-0}"; then
    CLUSTER_ARGS+=(--offload-rollout)
  fi
fi

WANDB_ARGS=()
if is_truthy "${ENABLE_WANDB:-0}"; then
  WANDB_GROUP="${WANDB_GROUP_OVERRIDE:-${JOB_NAME:-escd-training}}"
  if (( ${#WANDB_GROUP} > 128 )); then
    echo "ERROR: W&B group is ${#WANDB_GROUP} characters (maximum 128); set WANDB_GROUP_OVERRIDE to a shorter value" >&2
    exit 2
  fi
  WANDB_ARGS+=(--use-wandb --wandb-project "${WANDB_PROJECT:-cross-tokenizer-opd-research}" --wandb-group "${WANDB_GROUP}")
fi

EVAL_ARGS=()
if [[ -n "${EVAL_INTERVAL:-}" ]]; then
  [[ -n "${EVAL_CONFIG:-}" ]] || { echo "EVAL_CONFIG is required when EVAL_INTERVAL is set" >&2; exit 2; }
  EVAL_ARGS+=(--eval-interval "${EVAL_INTERVAL}" --eval-config "${EVAL_CONFIG}")
  if is_truthy "${SKIP_EVAL_BEFORE_TRAIN:-1}"; then
    EVAL_ARGS+=(--skip-eval-before-train)
  fi
  if [[ -n "${EVAL_REWARD_KEY:-}" ]]; then
    EVAL_ARGS+=(--eval-reward-key "${EVAL_REWARD_KEY}")
  fi
fi
if is_truthy "${LOG_PASSRATE:-0}"; then
  EVAL_ARGS+=(--log-passrate)
fi

DEBUG_ARGS=()
if [[ -n "${SAVE_ROLLOUT_TEXT_DIR:-}" ]]; then
  DEBUG_ARGS+=(--save-rollout-text-dir "${SAVE_ROLLOUT_TEXT_DIR}")
fi
if [[ -n "${SAVE_ROLLOUT_DIAGNOSTICS_DIR:-}" ]]; then
  DEBUG_ARGS+=(--save-rollout-diagnostics-dir "${SAVE_ROLLOUT_DIAGNOSTICS_DIR}")
fi
if [[ -n "${SAVE_DEBUG_ROLLOUT_DATA:-}" ]]; then
  DEBUG_ARGS+=(--save-debug-rollout-data "${SAVE_DEBUG_ROLLOUT_DATA}")
fi
if is_truthy "${DEBUG_ROLLOUT_ONLY:-0}"; then
  DEBUG_ARGS+=(--debug-rollout-only)
fi
if is_truthy "${EXPORT_HF_ONLY:-0}"; then
  DEBUG_ARGS+=(
    --export-hf-only
    --debug-train-only
    --no-load-optim
    --no-load-rng
    --exit-on-missing-checkpoint
  )
fi

COMMAND=(python3 train.py "${CLUSTER_ARGS[@]}" "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${ALGO_ARGS[@]}" "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${WANDB_ARGS[@]}" "${EVAL_ARGS[@]}" "${DEBUG_ARGS[@]}")
if [[ -n "${FREEZE_PARAMS_NAME_LIST:-}" && "${FREEZE_PARAMS_NAME_LIST}" != "[]" ]]; then
  # The compiler emits a JSON list so regexes containing spaces or shell
  # metacharacters remain intact when passed through the scheduler/Ray launcher.
  mapfile -t FREEZE_PARAMS < <(python3 - "${FREEZE_PARAMS_NAME_LIST}" <<'PY'
import json
import sys

patterns = json.loads(sys.argv[1])
if not isinstance(patterns, list):
    raise SystemExit("FREEZE_PARAMS_NAME_LIST must be a JSON list")
for pattern in patterns:
    print(str(pattern))
PY
  )
  if [[ "${#FREEZE_PARAMS[@]}" -gt 0 ]]; then
    COMMAND+=(--freeze-params-name-list "${FREEZE_PARAMS[@]}")
  fi
fi
if is_truthy "${MOE_LOAD_DIAGNOSTICS:-0}" || [[ -n "${TRAIN_ENV_VARS_JSON:-}" ]]; then
  COMMAND+=(--train-env-vars "${MOE_TRAIN_ENV_VARS_JSON}")
fi
RAY_COMMAND=("${COMMAND[@]}")
if is_truthy "${DRY_RUN}"; then
  printf 'cd %q && ' "${SLIME_ROOT}"
  printf '%q ' "${RAY_COMMAND[@]}"
  echo
  exit 0
fi

export PROJECT_ROOT SLIME_ROOT
RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json, os

env_vars = {
    "PYTHONPATH": ":".join((os.environ["PROJECT_ROOT"], os.environ["SLIME_ROOT"], os.environ.get("MEGATRON_ROOT", "/opt/Megatron-LM"), os.environ.get("PYTHONPATH", ""))),
    "PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED", "1"),
    "SLIME_PLAIN_PROGRESS_LOG": os.environ.get("SLIME_PLAIN_PROGRESS_LOG", "1"),
    "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "MASTER_ADDR": os.environ.get("MASTER_ADDR", "127.0.0.1"),
    "OPD_TEACHER_DEPLOYMENT": os.environ.get(
        "OPD_TEACHER_DEPLOYMENT", "megatron_inprocess"
    ),
    "OPD_TEACHER_TRANSPORT": os.environ.get(
        "OPD_TEACHER_TRANSPORT", "inprocess"
    ),
    "OPD_TEACHER_URLS": os.environ.get("OPD_TEACHER_URLS", ""),
    "OPD_TEACHER_ENDPOINT_SUFFIX": os.environ.get(
        "OPD_TEACHER_ENDPOINT_SUFFIX", "generate"
    ),
    "OPD_BPM_ROUTING_SHA256": os.environ.get("OPD_BPM_ROUTING_SHA256", ""),
    "OPD_TEACHER_TIMEOUT_SECONDS": os.environ.get("OPD_TEACHER_TIMEOUT_SECONDS", "1800"),
    "OPD_TEACHER_RETRIES": os.environ.get("OPD_TEACHER_RETRIES", "3"),
    "OPD_TEACHER_CONCURRENCY_PER_ENDPOINT": os.environ.get("OPD_TEACHER_CONCURRENCY_PER_ENDPOINT", "4"),
    "OPD_TEACHER_MAX_INPUT_TOKENS": os.environ.get("OPD_TEACHER_MAX_INPUT_TOKENS", ""),
    "OPD_BPM_BINARY_HIDDEN_DIR": os.environ.get("OPD_BPM_BINARY_HIDDEN_DIR", ""),
    "OPD_BPM_BINARY_HIDDEN_TIMEOUT_SECONDS": os.environ.get("OPD_BPM_BINARY_HIDDEN_TIMEOUT_SECONDS", "60"),
    "OPD_BPM_JSON_HIDDEN_MAX_TOKENS": os.environ.get("OPD_BPM_JSON_HIDDEN_MAX_TOKENS", "2048"),
    "OPD_CROSS_TOKENIZER_BINARY_HIDDEN_TIMEOUT_SECONDS": os.environ.get(
        "OPD_CROSS_TOKENIZER_BINARY_HIDDEN_TIMEOUT_SECONDS",
        os.environ.get("OPD_BPM_BINARY_HIDDEN_TIMEOUT_SECONDS", "60"),
    ),
    "OPD_CROSS_TOKENIZER_JSON_HIDDEN_MAX_TOKENS": os.environ.get(
        "OPD_CROSS_TOKENIZER_JSON_HIDDEN_MAX_TOKENS",
        os.environ.get("OPD_BPM_JSON_HIDDEN_MAX_TOKENS", "2048"),
    ),
    "OPD_BPM_HEAD_LOGPROB_ATOL": os.environ.get("OPD_BPM_HEAD_LOGPROB_ATOL", "0.25"),
    "OPD_BPM_HEAD_LOGPROB_MEAN_ATOL": os.environ.get("OPD_BPM_HEAD_LOGPROB_MEAN_ATOL", "0.06"),
    "OPD_CROSS_TOKENIZER_HEAD_LOGPROB_ATOL": os.environ.get(
        "OPD_CROSS_TOKENIZER_HEAD_LOGPROB_ATOL",
        os.environ.get("OPD_BPM_HEAD_LOGPROB_ATOL", "0.25"),
    ),
    "OPD_CROSS_TOKENIZER_HEAD_LOGPROB_MEAN_ATOL": os.environ.get(
        "OPD_CROSS_TOKENIZER_HEAD_LOGPROB_MEAN_ATOL",
        os.environ.get("OPD_BPM_HEAD_LOGPROB_MEAN_ATOL", "0.06"),
    ),
    "OPD_DISABLE_MOE_PERMUTE_FUSION": os.environ.get("OPD_DISABLE_MOE_PERMUTE_FUSION", "0"),
    # Keep phase-level train diagnostics available inside Ray actors when a
    # canary explicitly enables them; default remains quiet for formal runs.
    "OPD_TRAIN_PHASE_LOG": os.environ.get("OPD_TRAIN_PHASE_LOG", "0"),
    "MOE_LOAD_DIAGNOSTICS": os.environ.get("MOE_LOAD_DIAGNOSTICS", "0"),
    "MOE_DIAGNOSTICS_LOG_INTERVAL": os.environ.get("MOE_DIAGNOSTICS_LOG_INTERVAL", "10"),
    "MOE_DIAGNOSTICS_PER_LAYER": os.environ.get("MOE_DIAGNOSTICS_PER_LAYER", "1"),
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800"),
    "TORCH_NCCL_DUMP_ON_TIMEOUT": os.environ.get("TORCH_NCCL_DUMP_ON_TIMEOUT", "0"),
    "TORCH_NCCL_TRACE_BUFFER_SIZE": os.environ.get("TORCH_NCCL_TRACE_BUFFER_SIZE", "0"),
    "RESET_ROLLOUT_DATA_STATE_ON_LOAD": os.environ.get("RESET_ROLLOUT_DATA_STATE_ON_LOAD", "0"),
    "OPD_SKILL_CONFIG_JSON": os.environ.get("OPD_SKILL_CONFIG_JSON", ""),
    "OPD_EVALPLUS_JAIL": os.environ.get("OPD_EVALPLUS_JAIL", ""),
    "OPD_EVALPLUS_CONCURRENCY": os.environ.get("OPD_EVALPLUS_CONCURRENCY", "32"),
}
# W&B credentials are intentionally not included here.  The config path is
# safe to propagate to Ray actors; each actor loads the key locally in
# wandb_utils._get_wandb_login_key().  Do not put empty optional settings in
# Ray's runtime environment: W&B 0.27 treats WANDB_MODE="" as an invalid
# enum value and aborts secondary actor construction.
for name in (
    "NVTE_TORCH_COMPILE",
    "TORCHDYNAMO_DISABLE",
    "CUDA_MODULE_LOADING",
    "CUDA_MODULE_DATA_LOADING",
    "WANDB_LOCAL_CONFIG",
    "WANDB_MODE",
    "WANDB_ENTITY",
    "WANDB_TEAM",
    "WANDB_PROJECT",
):
    value = os.environ.get(name, "")
    if value:
        env_vars[name] = value
print(json.dumps({"env_vars": env_vars}))
PY
)"
RAY_JOB_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT:-8265}"
RAY_SUBMISSION_ID="${SLIME_RAY_SUBMISSION_ID:-${JOB_NAME_OVERRIDE:-slime-opd}}"
RAY_SUBMISSION_ID="${RAY_SUBMISSION_ID//[^A-Za-z0-9_.-]/-}"

# Do not trust the exit status of the log-streaming `ray job submit` client as
# the training result.  In this scheduler/Ray combination the stream can return 0
# while the remote job is still RUNNING, which used to make the outer cluster
# entrypoint tear down the Ray cluster and report a false success.  Submit
# asynchronously, stream logs in the background, and make the remote state the
# sole source of truth.
RAY_LOG_PID=""
cleanup_background_processes() {
  if [[ -n "${RAY_LOG_PID}" ]]; then
    kill "${RAY_LOG_PID}" >/dev/null 2>&1 || true
    wait "${RAY_LOG_PID}" >/dev/null 2>&1 || true
  fi

}
trap cleanup_background_processes EXIT INT TERM

ray job submit \
  --address="${RAY_JOB_ADDRESS}" \
  --submission-id="${RAY_SUBMISSION_ID}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  --no-wait \
  -- bash -lc "$(printf 'cd %q && ' "${SLIME_ROOT}")$(printf '%q ' "${RAY_COMMAND[@]}")"

ray job logs \
  --address="${RAY_JOB_ADDRESS}" \
  --follow \
  --log-style=record \
  --log-color=false \
  "${RAY_SUBMISSION_ID}" &
RAY_LOG_PID=$!

status_query_failures=0
while true; do
  status_rc=0
  status_output="$(ray job status --address="${RAY_JOB_ADDRESS}" --log-style=record "${RAY_SUBMISSION_ID}" 2>&1)" || status_rc=$?
  ray_state="$(sed -n "s/.*Status for job '[^']*': \([A-Z_]*\).*/\1/p" <<<"${status_output}" | tail -n 1)"
  if [[ -z "${ray_state}" ]]; then
    if grep -Eqi "Job '[^']+' succeeded" <<<"${status_output}"; then
      ray_state=SUCCEEDED
    elif grep -Eqi "Job '[^']+' failed" <<<"${status_output}"; then
      ray_state=FAILED
    elif grep -Eqi "Job '[^']+' (was )?stopped" <<<"${status_output}"; then
      ray_state=STOPPED
    fi
  fi
  if (( status_rc != 0 )) && [[ -z "${ray_state}" ]]; then
    status_query_failures=$((status_query_failures + 1))
    echo "[WARN] Ray job status query failed (${status_query_failures}/10): ${status_output}" >&2
    if (( status_query_failures >= 10 )); then
      exit 1
    fi
    sleep 15
    continue
  fi
  status_query_failures=0
  echo "[ray-job-monitor] submission=${RAY_SUBMISSION_ID} state=${ray_state:-UNKNOWN}"
  case "${ray_state}" in
    SUCCEEDED)
      exit 0
      ;;
    FAILED|STOPPED)
      ray job logs --address="${RAY_JOB_ADDRESS}" "${RAY_SUBMISSION_ID}" || true
      exit 1
      ;;
    PENDING|RUNNING)
      sleep 15
      ;;
    *)
      echo "[WARN] Unrecognized Ray job state: ${status_output}" >&2
      sleep 15
      ;;
  esac
done
