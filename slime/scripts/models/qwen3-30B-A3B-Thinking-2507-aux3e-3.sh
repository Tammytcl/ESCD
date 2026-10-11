#!/usr/bin/env bash
set -euo pipefail

# Qwen3-30B-A3B-Thinking-2507 with the stronger auxiliary MoE
# load-balancing coefficient used by the router-trainable OPD ablation.
# The canonical alias is kept as the single source of topology/RoPE settings;
# only --moe-aux-loss-coeff is overridden here.
source "$(dirname -- "${BASH_SOURCE[0]}")/qwen3-30B-A3B-Thinking-2507.sh"

for i in "${!MODEL_ARGS[@]}"; do
  if [[ "${MODEL_ARGS[$i]}" == "--moe-aux-loss-coeff" ]]; then
    MODEL_ARGS[$((i + 1))]="3e-3"
  fi
done
