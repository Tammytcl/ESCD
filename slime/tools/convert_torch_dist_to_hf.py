#!/usr/bin/env python3
"""Convert a Slime torch.distributed checkpoint into HF safetensors.

This lives in the main research checkout so conversion is independent of the
training/deployment checkout.  ``--origin-hf-dir`` is optional, but when it is
provided it supplies tokenizer/config assets and anchors the model name for
checkpoints whose ``common.pt`` was created inside a DLC mount.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import time

import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp
from transformers import AutoConfig
from typing_extensions import override

from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf, remove_padding


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        # Megatron stores argparse/model helper objects in common.pt.  Their
        # classes are not needed for conversion and may not be importable in a
        # lightweight offline worker image.
        if mod_name.startswith(("megatron", "glm")):
            class DummyClass:
                def __init__(self, *args, **kwargs):
                    pass

            return DummyClass
        return super().find_class(mod_name, name)


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        for key, value in metadata.state_dict_metadata.items():
            if "optimizer" in key or "_state" in key:
                continue
            if isinstance(value, dist_cp.metadata.TensorStorageMetadata):
                value = torch.empty(value.size, dtype=value.properties.dtype)
            state_dict[key] = value
        super().set_up_planner(state_dict, metadata, is_coordinator)


def get_expert_param(args, name, param):
    if ".experts." not in name:
        yield name, param
        return
    match = re.search(r"mlp.experts\.(.+)\.weight(\d+)", name)
    if not match:
        assert param.shape[0] == args.num_experts
        for expert_id in range(args.num_experts):
            expert_name = name.replace(".experts.experts.", ".experts.") + str(expert_id)
            yield expert_name, param[expert_id]
    else:
        yield name, param


def get_layer_param(args, name, param):
    if ".layers." not in name:
        yield from get_expert_param(args, name, param)
        return
    match = re.search(r"\.layers\.(\d+)\.", name)
    if match:
        yield from get_expert_param(args, name, param)
        return
    assert param.shape[0] == args.num_layers
    for layer_id in range(args.num_layers):
        layer_name = name.replace(".layers.", f".layers.{layer_id}.")
        yield from get_expert_param(args, layer_name, param[layer_id])


def get_named_params(args, state_dict, model_name):
    for name, param in state_dict.items():
        # Qwen3.5/3.6 grouped-GEMM checkpoints already use the fused rank-3
        # expert layout expected by the official HF model.
        if (
            "qwen3_5" in model_name
            and ".experts.experts.linear_fc" in name
            and name.endswith(".weight")
            and param.ndim == 3
        ):
            name = name.replace(".experts.experts.", ".experts.").removesuffix(".weight")
            yield f"module.module.{name}", param
            continue
        yield from get_layer_param(args, f"module.module.{name}", param)


def save_tensors(args, model_name, state_dict, output_dir, chunk_size, vocab_size, origin_hf_dir):
    os.makedirs(output_dir, exist_ok=True)
    chunks: list[dict[str, torch.Tensor]] = [{}]
    converted_names: set[str] = set()
    current_size = 0
    total_size = 0
    for name, param in get_named_params(args, state_dict, model_name):
        if vocab_size:
            param = remove_padding(name, param, vocab_size)
        for converted_name, converted_param in convert_to_hf(args, model_name, name, param):
            converted_names.add(converted_name)
            size = converted_param.numel() * converted_param.element_size()
            if chunks[-1] and current_size + size > chunk_size:
                chunks.append({})
                current_size = 0
            chunks[-1][converted_name] = converted_param
            current_size += size
            total_size += size

    if origin_hf_dir:
        for filename in os.listdir(origin_hf_dir):
            if not filename.endswith(".safetensors"):
                continue
            with safetensors.safe_open(os.path.join(origin_hf_dir, filename), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    tensor = handle.get_tensor(name)
                    size = tensor.numel() * tensor.element_size()
                    if name not in converted_names:
                        if chunks[-1] and current_size + size > chunk_size:
                            chunks.append({})
                            current_size = 0
                        chunks[-1][name] = tensor
                        converted_names.add(name)
                        current_size += size
                    total_size += size

    weight_map = {}
    for index, tensors in enumerate(chunks):
        filename = f"model-{index + 1:05d}-of-{len(chunks):05d}.safetensors"
        weight_map.update({name: filename for name in tensors})
        safetensors.torch.save_file(tensors, os.path.join(output_dir, filename))
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as handle:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, handle, indent=2)


def copy_assets(origin_hf_dir, output_dir):
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        source = os.path.join(origin_hf_dir, filename)
        if os.path.isfile(source):
            shutil.copy(source, os.path.join(output_dir, filename))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--origin-hf-dir")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-a", "--add-missing-from-origin-hf", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=5 * 1024**3)
    parser.add_argument("--vocab-size", type=int)
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"output directory exists: {args.output_dir}; use --force")
    if args.model_name is None and args.origin_hf_dir is None:
        raise ValueError("--model-name or --origin-hf-dir is required")
    if args.model_name is None:
        args.model_name = type(AutoConfig.from_pretrained(args.origin_hf_dir, trust_remote_code=True)).__name__.lower()

    print(f"loading model from {args.input_dir}")
    started = time.time()
    megatron_args = torch.load(os.path.join(args.input_dir, "common.pt"), weights_only=False)["args"]
    if args.origin_hf_dir:
        megatron_args.hf_checkpoint = args.origin_hf_dir
        megatron_args.tokenizer_model = args.origin_hf_dir
    state_dict = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(args.input_dir),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(f"model loaded in {time.time() - started:.2f} sec")
    save_tensors(
        megatron_args,
        args.model_name,
        state_dict,
        args.output_dir,
        args.chunk_size,
        args.vocab_size,
        args.origin_hf_dir if args.add_missing_from_origin_hf else None,
    )
    if args.origin_hf_dir:
        copy_assets(args.origin_hf_dir, args.output_dir)


if __name__ == "__main__":
    main()
