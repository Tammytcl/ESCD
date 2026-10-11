#!/usr/bin/env bash
set -euo pipefail

# cluster entrypoint for direct-driver slime OPD jobs.
# Rank 0 starts the Ray head and runs TRAIN_SCRIPT; other ranks join the Ray
# cluster, then exit after rank 0 writes DONE_FILE.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/training/train.sh}"

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_DIST_INIT_BARRIER_TIMEOUT="${TORCH_DIST_INIT_BARRIER_TIMEOUT:-1800}"


NODE_RANK="${NODE_RANK:-0}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${WORKER_GPU:-${GPUS_PER_NODE:-8}}}"
MASTER_ADDR="${MASTER_ADDR:-${HEAD_ADDR:-}}"
if [[ -z "${MASTER_ADDR}" ]]; then
  if [[ "${NODE_RANK}" == "0" ]]; then
    MASTER_ADDR="$(hostname -I | awk '{print $1}')"
  else
    echo "[ERROR] MASTER_ADDR is empty on non-master rank ${NODE_RANK}" >&2
    exit 1
  fi
fi

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_NODE_MANAGER_PORT="${RAY_NODE_MANAGER_PORT:-6380}"
RAY_OBJECT_MANAGER_PORT="${RAY_OBJECT_MANAGER_PORT:-6381}"
RAY_RUNTIME_ENV_AGENT_PORT="${RAY_RUNTIME_ENV_AGENT_PORT:-6382}"
RAY_DASHBOARD_AGENT_PORT="${RAY_DASHBOARD_AGENT_PORT:-6383}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-20000}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-25000}"

OPD_RUN_ID="${OPD_RUN_ID:?Set a unique OPD_RUN_ID shared by all student nodes}"
DONE_DIR="${OPD_RAY_DONE_DIR:-${ROOT_DIR}/outputs/ray_done}"
DONE_FILE="${DONE_DIR}/${OPD_RUN_ID}.done"

mkdir -p "${DONE_DIR}"
cd "${ROOT_DIR}"

ray_common_args=(
  --node-manager-port="${RAY_NODE_MANAGER_PORT}"
  --object-manager-port="${RAY_OBJECT_MANAGER_PORT}"
  --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}"
  --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_PORT}"
  --min-worker-port="${RAY_MIN_WORKER_PORT}"
  --max-worker-port="${RAY_MAX_WORKER_PORT}"
  --num-gpus="${NPROC_PER_NODE}"
)

debug_log() {
  local debug_flag="${OPD_ENTRYPOINT_DEBUG:-0}"
  if [[ "${debug_flag}" == "1" || "${debug_flag,,}" == "true" ]]; then
    echo "$@"
  fi
}

debug_dump_env() {
  debug_log "========== cluster RAY DIRECT ENTRYPOINT DEBUG =========="
  debug_log "HOSTNAME=$(hostname)"
  debug_log "HOST_IPS=$(hostname -I 2>/dev/null || true)"
  debug_log "NODE_RANK=${NODE_RANK}"
  debug_log "NNODES=${NNODES}"
  debug_log "WORLD_SIZE=${WORLD_SIZE:-<unset>}"
  debug_log "RANK=${RANK:-<unset>}"
  debug_log "NPROC_PER_NODE=${NPROC_PER_NODE}"
  debug_log "WORKER_GPU=${WORKER_GPU:-<unset>}"
  debug_log "MASTER_ADDR=${MASTER_ADDR}"
  debug_log "MASTER_PORT=${MASTER_PORT:-<unset>}"
  debug_log "RAY_PORT=${RAY_PORT}"
  debug_log "DONE_FILE=${DONE_FILE}"
  debug_log "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
  debug_log "PATH_RAY=$(command -v ray 2>/dev/null || echo '<not found>')"
  debug_log "PATH_PYTHON3=$(command -v python3 2>/dev/null || echo '<not found>')"
  debug_log "====================================================="
}

wait_for_nodes() {
  local expected_nodes="${EXPECTED_RAY_NODES:-${NNODES}}"
  local wait_timeout="${RAY_WAIT_TIMEOUT_SECONDS:-7200}"
  local wait_interval="${RAY_WAIT_INTERVAL_SECONDS:-5}"
  local check_timeout="${RAY_NODE_CHECK_TIMEOUT_SECONDS:-20}"
  local elapsed=0
  local current_nodes=0

  echo "[INFO] Waiting for ${expected_nodes} Ray nodes at ${MASTER_ADDR}:${RAY_PORT}"
  while true; do
    current_nodes="$(timeout "${check_timeout}" python3 - "${MASTER_ADDR}:${RAY_PORT}" <<'PY' 2>/dev/null || echo 0
import sys
import ray

ray.init(address=sys.argv[1], ignore_reinit_error=True)
print(sum(1 for node in ray.nodes() if node.get("Alive")))
ray.shutdown()
PY
)"
    echo "[INFO] Ray nodes ready: ${current_nodes}/${expected_nodes} (${elapsed}s)"
    if [[ "${current_nodes}" -ge "${expected_nodes}" ]]; then
      break
    fi
    if [[ "${elapsed}" -ge "${wait_timeout}" ]]; then
      echo "[ERROR] Timed out waiting for Ray nodes: ${current_nodes}/${expected_nodes}." >&2
      return 1
    fi
    sleep "${wait_interval}"
    elapsed=$((elapsed + wait_interval))
  done
}

if [[ "${NODE_RANK}" == "0" ]]; then
  debug_dump_env
  rm -f "${DONE_FILE}"
  ray stop --force >/dev/null 2>&1 || true
  ray start --head \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    "${ray_common_args[@]}"

  wait_for_nodes

  export NNODES="${NNODES}"
  export NGPUS_PER_NODE="${NPROC_PER_NODE}"
  export RAY_INIT_ADDRESS="${RAY_INIT_ADDRESS:-auto}"

  status=0
  bash "${TRAIN_SCRIPT}" || status=$?
  echo "${status}" >"${DONE_FILE}"
  ray stop --force >/dev/null 2>&1 || true
  exit "${status}"
fi

debug_dump_env
sleep "${RAY_WORKER_STARTUP_SLEEP:-15}"
ray stop --force >/dev/null 2>&1 || true
ray start \
  --address="${MASTER_ADDR}:${RAY_PORT}" \
  "${ray_common_args[@]}"

while [[ ! -f "${DONE_FILE}" ]]; do
  sleep "${OPD_DONE_POLL_SECONDS:-30}"
done

status="$(cat "${DONE_FILE}" 2>/dev/null || echo 1)"
ray stop --force >/dev/null 2>&1 || true
exit "${status}"
