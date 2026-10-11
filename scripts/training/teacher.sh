#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
NODE_RANK="${NODE_RANK:-0}"
COORD_DIR="${OPD_TEACHER_COORD_DIR:?OPD_TEACHER_COORD_DIR is required}"
MODEL_PATH="${OPD_TEACHER_MODEL_PATH:?Set OPD_TEACHER_MODEL_PATH}"
PORT="${OPD_TEACHER_PORT:-30000}"
TP_SIZE="${OPD_TEACHER_TP_SIZE:-8}"
CONTEXT_LENGTH="${OPD_TEACHER_CONTEXT_LENGTH:-65536}"
LOG_DIR="${COORD_DIR}/logs"
URL_FILE="${COORD_DIR}/teacher-${NODE_RANK}.url"
# Publish both endpoint names for launcher compatibility.
ENDPOINT_FILE="${COORD_DIR}/endpoint_${NODE_RANK}.txt"
DONE_FILE="${COORD_DIR}/student.done"
BINARY_HIDDEN_DIR="${OPD_BPM_BINARY_HIDDEN_DIR:-${COORD_DIR}/hidden}"

mkdir -p "${LOG_DIR}" "${BINARY_HIDDEN_DIR}"
rm -f "${URL_FILE}" "${ENDPOINT_FILE}"
HOST_IP="$(hostname -I | awk '{print $1}')"
LOG_FILE="${LOG_DIR}/teacher-${NODE_RANK}.log"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPD_BPM_BINARY_HIDDEN_DIR="${BINARY_HIDDEN_DIR}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_PLUGINS="${SGLANG_PLUGINS:+${SGLANG_PLUGINS},}opd_bpm_binary_hidden"
python3 - <<'PY'
from importlib.metadata import entry_points

matches = [
    ep for ep in entry_points(group="sglang.srt.plugins")
    if ep.name == "opd_bpm_binary_hidden"
]
if len(matches) != 1:
    raise SystemExit(f"expected exactly one opd_bpm_binary_hidden entry point, got {matches}")
PY

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${OPD_TEACHER_MEM_FRACTION_STATIC:-0.85}" \
  --max-running-requests "${OPD_TEACHER_MAX_RUNNING_REQUESTS:-4}" \
  --chunked-prefill-size "${OPD_TEACHER_CHUNKED_PREFILL_SIZE:-4096}" \
  --enable-return-hidden-states \
  --disable-radix-cache \
  --disable-cuda-graph \
  >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + ${OPD_TEACHER_STARTUP_TIMEOUT_SECONDS:-3600}))
while (( SECONDS < deadline )); do
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "ERROR: Teacher ${NODE_RANK} exited during startup; see ${LOG_FILE}" >&2
    exit 1
  fi
  if curl -fsS --max-time 5 "http://${HOST_IP}:${PORT}/health" >/dev/null 2>&1; then
    for endpoint_file in "${URL_FILE}" "${ENDPOINT_FILE}"; do
      tmp_file="${endpoint_file}.tmp.$$"
      printf 'http://%s:%s/generate\n' "${HOST_IP}" "${PORT}" >"${tmp_file}"
      mv "${tmp_file}" "${endpoint_file}"
    done
    echo "[teacher] rank=${NODE_RANK} ready at http://${HOST_IP}:${PORT}"
    break
  fi
  sleep 5
done
[[ -f "${URL_FILE}" ]] || { echo "ERROR: Teacher ${NODE_RANK} startup timed out" >&2; exit 1; }

while [[ ! -f "${DONE_FILE}" ]]; do
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "ERROR: Teacher ${NODE_RANK} exited before Student completion; see ${LOG_FILE}" >&2
    exit 1
  fi
  sleep 30
done
echo "[teacher] Student completion observed; stopping rank=${NODE_RANK}"
