import re

from .deepseekv3 import convert_deepseekv3_to_hf
from .glm4 import convert_glm4_to_hf
from .glm4moe import convert_glm4moe_to_hf
from .gpt_oss import convert_gpt_oss_to_hf
from .llama import convert_llama_to_hf
from .mimo import convert_mimo_to_hf
from .processors import quantize_params, remove_padding
from .qwen2 import convert_qwen2_to_hf
from .qwen3_5 import convert_qwen3_5_to_hf
from .qwen3_next import convert_qwen3_next_to_hf
from .qwen3_vl import convert_qwen3vl_to_hf
from .qwen3moe import convert_qwen3moe_to_hf


_QWEN_VL_VISUAL_BLOCK_RE = re.compile(r"^(?:model\.)?visual\.blocks\.(\d+)\.(.+)$")
_QWEN_VL_VISUAL_DECODER_RE = re.compile(r"^(?:model\.)?visual\.decoder\.layers\.(\d+)\.(.+)$")
_QWEN_VL_DECODER_TO_SGLANG_BLOCK_ALIASES = {
    "self_attention.linear_qkv.weight": "attn.qkv.weight",
    "self_attention.linear_qkv.bias": "attn.qkv.bias",
    "self_attention.linear_proj.weight": "attn.proj.weight",
    "self_attention.linear_proj.bias": "attn.proj.bias",
    "self_attention.linear_qkv.layer_norm_weight": "norm1.weight",
    "self_attention.linear_qkv.layer_norm_bias": "norm1.bias",
    "mlp.linear_fc1.layer_norm_weight": "norm2.weight",
    "mlp.linear_fc1.layer_norm_bias": "norm2.bias",
    "mlp.linear_fc1.weight": "mlp.linear_fc1.weight",
    "mlp.linear_fc1.bias": "mlp.linear_fc1.bias",
    "mlp.linear_fc2.weight": "mlp.linear_fc2.weight",
    "mlp.linear_fc2.bias": "mlp.linear_fc2.bias",
}
_QWEN_VL_SGLANG_VISUAL_MERGER_ALIASES = {
    "patch_norm.weight": "norm.weight",
    "patch_norm.bias": "norm.bias",
}


def _is_incompatible_sglang_visual_name(name):
    if name.startswith("model.visual."):
        name = "visual." + name[len("model.visual.") :]
    return name.startswith("visual.decoder.layers.") or name.startswith("visual.merger.patch_norm.")


def _append_sglang_visual_alias(aliases, alias):
    if alias and not _is_incompatible_sglang_visual_name(alias):
        aliases.append(alias)


def _sglang_visual_aliases(name):
    aliases = []
    if name.startswith("model.visual."):
        _append_sglang_visual_alias(aliases, "visual." + name[len("model.visual.") :])

    match = _QWEN_VL_VISUAL_DECODER_RE.match(name)
    if match is not None:
        layer_idx, suffix = match.groups()
        mapped_suffix = _QWEN_VL_DECODER_TO_SGLANG_BLOCK_ALIASES.get(suffix)
        if mapped_suffix is not None:
            _append_sglang_visual_alias(aliases, f"visual.blocks.{layer_idx}.{mapped_suffix}")

    for prefix in ("model.visual.merger.", "visual.merger."):
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            mapped_suffix = _QWEN_VL_SGLANG_VISUAL_MERGER_ALIASES.get(suffix)
            if mapped_suffix is not None:
                _append_sglang_visual_alias(aliases, f"visual.merger.{mapped_suffix}")

    return aliases


def add_sglang_visual_aliases(converted_named_tensors):
    """Normalize Qwen VL visual weight names for the SGLang loader."""
    aliased_named_tensors = []
    seen_names = set()
    for name, tensor in converted_named_tensors:
        if not _is_incompatible_sglang_visual_name(name):
            aliased_named_tensors.append((name, tensor))
            seen_names.add(name)

        for alias in _sglang_visual_aliases(name):
            if alias not in seen_names:
                aliased_named_tensors.append((alias, tensor))
                seen_names.add(alias)

    return aliased_named_tensors


# TODO unify w/ `convert_to_hf`
def postprocess_hf_param(args, megatron_param_name, hf_param_name, param):
    param = remove_padding(megatron_param_name, param, args.vocab_size)
    # TODO support quant
    return param


# TODO optimize code details
def convert_to_hf(args, model_name, name, param, quantization_config=None):
    param = remove_padding(name, param, args.vocab_size)

    converted_named_tensors = _convert_to_hf_core(args, model_name, name, param)
    converted_named_tensors = add_sglang_visual_aliases(converted_named_tensors)

    return quantize_params(args, name, converted_named_tensors, quantization_config)


# TODO optimize
_cached_tensors = {}


# TODO optimize code details
def _convert_to_hf_core(args, model_name, name, param):
    if "smollm3" in model_name:
        return convert_llama_to_hf(args, name, param)
    if "glm4moelite" in model_name or "deepseekv3" in model_name:
        converted_named_tensors = convert_deepseekv3_to_hf(args, name, param)
    elif "glm4moe" in model_name:
        converted_named_tensors = convert_glm4moe_to_hf(args, name, param)
    elif "glm4" in model_name:
        converted_named_tensors = convert_glm4_to_hf(args, name, param)
    elif "gpt_oss" in model_name or "gpt-oss" in model_name or "gptoss" in model_name:
        converted_named_tensors = convert_gpt_oss_to_hf(args, name, param)
    elif "qwen3moe" in model_name:
        converted_named_tensors = convert_qwen3moe_to_hf(args, name, param)
    elif "qwen3next" in model_name:
        converted_named_tensors = convert_qwen3_next_to_hf(args, name, param)
    elif "qwen3_5" in model_name:
        converted_named_tensors = convert_qwen3_5_to_hf(args, name, param)
    elif "qwen3vl" in model_name:
        converted_named_tensors = convert_qwen3vl_to_hf(args, name, param)
    elif "qwen2" in model_name or "qwen3" in model_name:
        converted_named_tensors = convert_qwen2_to_hf(args, name, param)
    elif "llama" in model_name:
        converted_named_tensors = convert_llama_to_hf(args, name, param)
    elif "mimo" in model_name:
        converted_named_tensors = convert_mimo_to_hf(args, name, param)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # to compatible with sglang implementation
    if args.q_lora_rank is not None:
        old_converted_named_tensors = converted_named_tensors
        converted_named_tensors = []
        for converted_name, converted_param in old_converted_named_tensors:
            if "q_a_proj" in converted_name:
                pair_name = converted_name.replace("q_a_proj", "kv_a_proj_with_mqa")
                if pair_name in _cached_tensors:
                    converted_named_tensors += [
                        (converted_name, converted_param),
                        (pair_name, _cached_tensors[pair_name]),
                    ]
                    del _cached_tensors[pair_name]
                else:
                    _cached_tensors[converted_name] = converted_param
            elif "kv_a_proj_with_mqa" in converted_name:
                pair_name = converted_name.replace("kv_a_proj_with_mqa", "q_a_proj")
                if pair_name in _cached_tensors:
                    converted_named_tensors += [
                        (converted_name, converted_param),
                        (pair_name, _cached_tensors[pair_name]),
                    ]
                    del _cached_tensors[pair_name]
                else:
                    _cached_tensors[converted_name] = converted_param
            else:
                converted_named_tensors.append((converted_name, converted_param))
    return converted_named_tensors
