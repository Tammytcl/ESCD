"""Load a frozen Hugging Face LM-head shard for Slime dense Teacher scoring."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


_HEAD_KEYS = (
    "lm_head.weight",
    "model.lm_head.weight",
    "language_model.lm_head.weight",
    "model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
)


def _weight_map(root: Path) -> dict[str, str]:
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if indexes:
        payload = json.loads(indexes[0].read_text(encoding="utf-8"))
        mapping = payload.get("weight_map")
        if isinstance(mapping, dict):
            return {str(key): str(value) for key, value in mapping.items()}
    files = sorted(root.glob("*.safetensors"))
    if len(files) == 1:
        with safe_open(str(files[0]), framework="pt", device="cpu") as handle:
            return {str(key): files[0].name for key in handle.keys()}
    raise FileNotFoundError(f"Cannot resolve a safetensors weight map under {root}")


def load_frozen_lm_head_shard(
    model_path: str | Path,
    *,
    tensor_parallel_rank: int,
    tensor_parallel_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Load only this TP rank's output rows, without constructing the model."""

    root = Path(model_path)
    mapping = _weight_map(root)
    key = next((candidate for candidate in _HEAD_KEYS if candidate in mapping), None)
    if key is None:
        raise KeyError(
            f"No supported LM-head/ tied-embedding key found under {root}; checked {_HEAD_KEYS}"
        )
    if tensor_parallel_size <= 0 or not 0 <= tensor_parallel_rank < tensor_parallel_size:
        raise ValueError("Invalid tensor-parallel rank/size for Teacher LM head")
    path = root / mapping[key]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        weight_slice = handle.get_slice(key)
        shape = tuple(weight_slice.get_shape())
        if len(shape) != 2:
            raise ValueError(f"Teacher LM head must be rank two, got {shape}")
        vocab_size = int(shape[0])
        rows_per_rank = math.ceil(vocab_size / tensor_parallel_size)
        start = tensor_parallel_rank * rows_per_rank
        end = min(start + rows_per_rank, vocab_size)
        local = weight_slice[start:end, :]
    if local.size(0) < rows_per_rank:
        local = torch.nn.functional.pad(local, (0, 0, 0, rows_per_rank - local.size(0)))
    return local.to(device=device, non_blocking=False).detach(), vocab_size
