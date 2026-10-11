#!/usr/bin/env bash

# The local Qwen3.6-35B-A3B checkpoint advertises
# Qwen3_5MoeForConditionalGeneration. Reuse the audited architecture spec and
# keep the public checkpoint name explicit in experiment manifests.
[[ -n "${SLIME_ROOT:-}" ]] || {
  echo "ERROR: SLIME_ROOT must be set before sourcing this model profile." >&2
  return 1
}

# shellcheck source=/dev/null
source "${SLIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"
