#!/usr/bin/env bash
set -euo pipefail

# Qwen3-30B-A3B-Thinking-2507 with a small auxiliary MoE load-balancing
# coefficient. Keep the exact released topology/RoPE settings from the
# canonical alias and override only the coefficient for this ablation.
source "$(dirname -- "${BASH_SOURCE[0]}")/qwen3-30B-A3B-Thinking-2507.sh"

for i in "${!MODEL_ARGS[@]}"; do
  if [[ "${MODEL_ARGS[$i]}" == "--moe-aux-loss-coeff" ]]; then
    MODEL_ARGS[$((i + 1))]="1e-3"
  fi
done

