#!/usr/bin/env bash
set -euo pipefail

NODE_RANK="${NODE_RANK:-0}"
COORD_DIR="${OPD_TEACHER_COORD_DIR:?OPD_TEACHER_COORD_DIR is required}"
TEACHER_COUNT="${OPD_TEACHER_COUNT:-2}"
DONE_FILE="${COORD_DIR}/student.done"
deadline=$((SECONDS + ${OPD_TEACHER_WAIT_TIMEOUT_SECONDS:-5400}))
next_wait_log=0
ready=0

urls=()
while (( SECONDS < deadline )); do
  urls=()
  ready=1
  for ((index=0; index<TEACHER_COUNT; index++)); do
    url_file="${COORD_DIR}/teacher-${index}.url"
    if [[ ! -s "${url_file}" ]]; then
      ready=0
      wait_reason="missing Teacher ${index} URL file"
      break
    fi
    url="$(<"${url_file}")"
    # Older launchers publish a service root; shared launchers publish the
    # generation endpoint. Health checks and OPD_TEACHER_URLS use the root
    # (the Python Teacher client appends its selected endpoint exactly once).
    url="${url%/}"
    url="${url%/generate}"
    [[ "${url}" =~ ^http://[0-9.]+:[0-9]+$ ]] || {
      ready=0
      wait_reason="invalid Teacher ${index} URL; expected http://IPv4:port[/generate]"
      break
    }
    # A URL file can outlive its Teacher (e.g. a reused coordination
    # directory after a failed Student wrote student.done and the Teacher
    # exited).  Require a live /health so a stale file fails fast here
    # instead of after a full rollout.
    curl -fsS --max-time 5 "${url}/health" >/dev/null 2>&1 || {
      ready=0
      wait_reason="Teacher ${index} health check not ready at ${url}/health"
      break
    }
    urls+=("${url}")
  done
  (( ready == 1 )) && break
  if (( SECONDS >= next_wait_log )); then
    echo "[student] waiting for external Teachers: ${wait_reason}"
    next_wait_log=$((SECONDS + 60))
  fi
  sleep 10
done
[[ "${ready}" -eq 1 && "${#urls[@]}" -eq "${TEACHER_COUNT}" ]] || { echo "ERROR: external Teachers did not become ready" >&2; exit 1; }

export OPD_TEACHER_URLS="$(IFS=,; printf '%s' "${urls[*]}")"
echo "[student] external Teachers ready: ${OPD_TEACHER_URLS}"
export ROOT_DIR="${PROJECT_ROOT:?Set PROJECT_ROOT}"
export TRAIN_SCRIPT="scripts/training/train.sh"

status=0
bash "${ROOT_DIR}/scripts/training/ray_entrypoint.sh" || status=$?
if [[ "${NODE_RANK}" == "0" ]]; then
  mkdir -p "${COORD_DIR}"
  printf '%s\n' "${status}" >"${DONE_FILE}"
fi
exit "${status}"
