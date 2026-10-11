#!/usr/bin/env bash
set -euo pipefail

# Qwen3-30B-A3B-Thinking-2507 uses the Qwen3 MoE topology with the 10M
# rotary base used by the released checkpoint.  Keep this as a distinct model
# name: silently falling back to qwen3-30B-A3B changes RoPE and makes a raw
# Megatron release load look healthy while producing incorrect logits.
MODEL_ARGS_ROTARY_BASE=10000000
source "$(dirname -- "${BASH_SOURCE[0]}")/qwen3-30B-A3B.sh"
