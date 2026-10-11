# SmolLM3 differs from Llama: every fourth layer has NO positional rotation.
MODEL_ARGS=(
  --num-layers 36 --hidden-size 2048 --ffn-hidden-size 11008
  --num-attention-heads 16 --group-query-attention --num-query-groups 4
  --kv-channels 128 --swiglu --disable-bias-linear
  --normalization RMSNorm --norm-epsilon 1e-6
  --use-rotary-position-embeddings --rotary-base 5000000 --no-rope-freq 4
  --max-position-embeddings 65536 --vocab-size 128256
)
# Deliberately DO NOT set --untie-embeddings-and-output-weights.
