"""Compatibility mappings for Qwen3.5-VL MTP in the Slime model provider.

The Megatron Bridge shipped in the current Slime image calls the inner MTP
block ``mtp_model_layer``.  Slime's Qwen3.5 model spec (and the checkpoint
loader used by our OPD jobs) exposes the same block as ``transformer_layer``.
The two names describe the same module; without these aliases all MTP MoE
parameters become unmapped during Megatron -> HF weight synchronization.

This is deliberately a small compatibility layer around the upstream bridge:
the regular decoder mappings and the original ``mtp_model_layer`` mappings are
left untouched.  We only add the equivalent ``transformer_layer`` patterns.
"""

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge


def _transformer_layer_mtp_mappings():
    """Return aliases for the MTP names emitted by Slime's Qwen3.5 spec."""

    direct = {
        "language_model.mtp.layers.0.transformer_layer.mlp.router.weight": "mtp.layers.0.mlp.gate.weight",
        "language_model.mtp.layers.0.transformer_layer.pre_mlp_layernorm.weight": "mtp.layers.0.post_attention_layernorm.weight",
        "language_model.mtp.layers.0.transformer_layer.self_attention.linear_qkv.layer_norm_weight": "mtp.layers.0.input_layernorm.weight",
        "language_model.mtp.layers.0.transformer_layer.self_attention.q_layernorm.weight": "mtp.layers.0.self_attn.q_norm.weight",
        "language_model.mtp.layers.0.transformer_layer.self_attention.k_layernorm.weight": "mtp.layers.0.self_attn.k_norm.weight",
        "language_model.mtp.layers.0.transformer_layer.self_attention.linear_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
    }
    mappings = [AutoMapping(megatron_param=k, hf_param=v) for k, v in direct.items()]
    mappings.extend(
        [
            QKVMapping(
                megatron_param="language_model.mtp.layers.*.transformer_layer.self_attention.linear_qkv.weight",
                q="mtp.layers.*.self_attn.q_proj.weight",
                k="mtp.layers.*.self_attn.k_proj.weight",
                v="mtp.layers.*.self_attn.v_proj.weight",
            ),
            # Qwen3.6 stores every MTP expert in two fused rank-3 tensors,
            # exactly like its main decoder: [experts, gate+up, hidden] and
            # [experts, hidden, intermediate].  Grouped mappings gather/split
            # the per-expert Megatron parameters without inventing 256 HF keys.
            FusedGatedExpertMapping(
                megatron_param="language_model.mtp.layers.*.transformer_layer.mlp.experts.linear_fc1.weight*",
                hf_param="mtp.layers.*.mlp.experts.gate_up_proj",
            ),
            FusedExpertMapping(
                megatron_param="language_model.mtp.layers.*.transformer_layer.mlp.experts.linear_fc2.weight*",
                hf_param="mtp.layers.*.mlp.experts.down_proj",
                transpose_on_export=True,
            ),
            GatedMLPMapping(
                megatron_param="language_model.mtp.layers.*.transformer_layer.mlp.shared_experts.linear_fc1.weight",
                gate="mtp.layers.*.mlp.shared_expert.gate_proj.weight",
                up="mtp.layers.*.mlp.shared_expert.up_proj.weight",
            ),
            AutoMapping(
                megatron_param="language_model.mtp.layers.*.transformer_layer.mlp.shared_experts.linear_fc2.weight",
                hf_param="mtp.layers.*.mlp.shared_expert.down_proj.weight",
            ),
            ReplicatedMapping(
                megatron_param="language_model.mtp.layers.0.transformer_layer.mlp.shared_experts.gate_weight",
                hf_param="mtp.layers.0.mlp.shared_expert_gate.weight",
            ),
        ]
    )
    return mappings


if not getattr(Qwen35VLMoEBridge, "_slime_transformer_layer_mtp_patched", False):
    _upstream_mapping_registry = Qwen35VLMoEBridge.mapping_registry

    def _mapping_registry_with_slime_mtp_aliases(self):
        upstream = _upstream_mapping_registry(self)
        return MegatronMappingRegistry(*upstream.mappings, *_transformer_layer_mtp_mappings())

    Qwen35VLMoEBridge.mapping_registry = _mapping_registry_with_slime_mtp_aliases
    Qwen35VLMoEBridge._slime_transformer_layer_mtp_patched = True
