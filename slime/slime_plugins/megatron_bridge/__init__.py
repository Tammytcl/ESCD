import warnings

try:
    import slime_plugins.megatron_bridge.glm4v_moe  # noqa: F401  # register GLM-4.6V bridge
except ModuleNotFoundError as exc:
    if exc.name != "megatron.bridge.models.qwen.qwen_provider":
        raise
    warnings.warn(
        "Skipping optional GLM-4.6V Megatron Bridge registration because this "
        "Megatron Bridge install does not provide megatron.bridge.models.qwen.qwen_provider.",
        RuntimeWarning,
        stacklevel=2,
    )

# Qwen3.5's MTP block is named ``transformer_layer`` by Slime's model spec,
# while the image's upstream Qwen3.5 bridge registers ``mtp_model_layer``.
# Importing this compatibility module adds the equivalent mappings before any
# checkpoint/update-weight conversion task is constructed.
try:
    import slime_plugins.megatron_bridge.qwen35_mtp  # noqa: F401
except ModuleNotFoundError as exc:
    # Keep non-Qwen jobs usable with older Megatron Bridge images.
    if not str(exc.name).startswith("megatron.bridge.models.qwen_vl"):
        raise
