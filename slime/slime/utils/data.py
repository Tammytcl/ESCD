import itertools
import json
import logging
import os
import random
import re
from collections import defaultdict

import numpy as np
import ray
import yaml

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from slime.utils.types import MultimodalTypes, Sample

from .timer import Timer

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


def read_file(path):
    path, row_slice = _parse_generalized_path(path)
    reader = None

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error at line {line_num}: {e}")
                        continue

        reader = jsonl_reader(path)

    elif path.endswith(".parquet"):
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")

        def parquet_reader(p):
            pf = pq.ParquetFile(p)

            for batch in pf.iter_batches():
                yield from batch.to_pylist()

        reader = parquet_reader(path)

    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:

        logger.info("read_file path=%s applying slice row_slice=%s", path, row_slice)
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader


_YAML_ENTRY_METADATA_KEYS = (
    "task_id",
    "teacher_id",
    "env_id",
    "loss_policy",
    "loss_weight",
    "sample_weight",
    "sample_ratio",
    "task_type",
    "modalities",
    "eval_group",
)


def read_yaml_config(
    yaml_path: str,
    *,
    input_key: str = "text",
    label_key: str | None = None,
    metadata_key: str = "metadata",
):
    """Read a MultiTeacher-OPD YAML manifest and yield normalized records."""
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    data_entries = config.get("data") or config.get("train_data")
    if not isinstance(data_entries, list) or not data_entries:
        raise ValueError(f"YAML config '{yaml_path}' must contain a non-empty 'data' or 'train_data' list.")

    for entry in data_entries:
        if not isinstance(entry, dict):
            raise ValueError(f"YAML data entry must be a mapping, got {entry!r}")

        file_path = entry.get("path")
        if not file_path:
            raise ValueError(f"YAML entry missing 'path': {entry}")

        file_path = _resolve_yaml_entry_path(yaml_path, str(file_path))
        teacher_override = entry.get("teacher") or entry.get("teacher_id")
        src_input_key = entry.get("input_key")
        src_label_key = entry.get("label_key")
        src_metadata_key = entry.get("metadata_key")
        entry_metadata = {key: entry[key] for key in _YAML_ENTRY_METADATA_KEYS if key in entry}
        logger.info(
            "YAML config: loading '%s'%s",
            file_path,
            f" with teacher override='{teacher_override}'" if teacher_override else "",
        )

        count = 0
        for record in read_file(file_path):
            if teacher_override is not None:
                record["teacher_model"] = teacher_override
                entry_metadata.setdefault("teacher_id", teacher_override)
            if src_input_key and src_input_key != input_key and src_input_key in record:
                record[input_key] = record.pop(src_input_key)
            if label_key and src_label_key and src_label_key != label_key and src_label_key in record:
                record[label_key] = record.pop(src_label_key)
            if src_metadata_key and src_metadata_key != metadata_key and src_metadata_key in record:
                merged_metadata = _coerce_metadata(record.get(src_metadata_key))
                merged_metadata.update(_coerce_metadata(record.get(metadata_key)))
                record[metadata_key] = merged_metadata
            if entry_metadata:
                metadata = _coerce_metadata(record.get(metadata_key))
                for key, value in entry_metadata.items():
                    metadata.setdefault(key, value)
                record[metadata_key] = metadata
            count += 1
            yield record
        logger.info("  -> %s: %d samples", file_path, count)


def _coerce_metadata(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _resolve_yaml_entry_path(yaml_path: str, file_path: str) -> str:
    real_path, row_slice = _parse_generalized_path(file_path)
    suffix = ""
    if row_slice is not None:
        start = "" if row_slice.start is None else row_slice.start
        stop = "" if row_slice.stop is None else row_slice.stop
        suffix = f"@[{start}:{stop}]"

    if os.path.isabs(real_path) or os.path.exists(real_path):
        return file_path

    yaml_dir = os.path.abspath(os.path.dirname(yaml_path))
    for base in [yaml_dir, *list(_iter_parent_dirs(yaml_dir))]:
        candidate = os.path.join(base, real_path)
        if os.path.exists(candidate):
            return candidate + suffix
    return os.path.join(yaml_dir, real_path) + suffix


def _iter_parent_dirs(path: str):
    current = os.path.abspath(path)
    while True:
        parent = os.path.dirname(current)
        if parent == current:
            return
        yield parent
        current = parent


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def filter_long_prompt(origin_samples: list[Sample], tokenizer, processor, max_length: int | None) -> list[Sample]:
    if max_length is None:
        return origin_samples

    if not isinstance(origin_samples[0].prompt, str):
        logger.warning(
            "Skipping max_length check for list prompt. Set apply_chat_template=True to enable length filtering."
        )
        return origin_samples

    if processor:
        # Use processor only for samples with actual multimodal content; use batched tokenizer for text-only.
        text_only = []
        multimodal = []
        for sample in origin_samples:
            if sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
                multimodal.append(sample)
            else:
                text_only.append(sample)
        filtered_samples = []
        if text_only:
            prompts = [s.prompt for s in text_only]
            input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
            for sample, input_ids in zip(text_only, input_ids_list, strict=True):
                if len(input_ids) <= max_length:
                    filtered_samples.append(sample)
        if multimodal:
            from slime.utils.processing_utils import process_vision_info

            for sample in multimodal:
                # Dataset.__init__ stores processed images/videos before applying the
                # chat template. When apply_chat_template=True, sample.prompt is the
                # rendered string, so re-parsing it as message dicts would fail.
                multimodal_inputs = (
                    sample.multimodal_inputs
                    if isinstance(sample.prompt, str)
                    else process_vision_info(sample.prompt, processor)
                )
                multimodal_inputs = {k: v for k, v in (multimodal_inputs or {}).items() if v is not None}
                processor_output = processor(text=sample.prompt, **multimodal_inputs)
                input_ids = processor_output["input_ids"][0]
                if len(input_ids) <= max_length:
                    filtered_samples.append(sample)
    else:
        prompts = [sample.prompt for sample in origin_samples]
        input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        filtered_samples = [
            sample
            for sample, input_ids in zip(origin_samples, input_ids_list, strict=True)
            if len(input_ids) <= max_length
        ]

    logger.info(f"Filtered {len(origin_samples) - len(filtered_samples)} samples longer than max_length={max_length}.")

    return filtered_samples


def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        # If prompt is a string and we don't apply chat template, return the prompt as is.
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]

    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodal_data = data.get(data_key)
                if multimodal_data is not None:
                    multimodals[mt.placeholder] = (mt, list(multimodal_data))

        if not multimodals:
            return prompt

        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in prompt:
            if isinstance(message["content"], str):
                content_list = []
                for segment in re.split(pattern, message["content"]):
                    if not segment:
                        continue
                    if segment in multimodals:
                        mt, content = multimodals[segment]
                        assert len(content) > 0, (
                            f"Not enough {mt.name} data: more '{mt.placeholder}' placeholders in prompt "
                            f"than {mt.name}s provided in data"
                        )
                        content_list.append({"type": mt.name, mt.name: content.pop(0)})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list

            elif isinstance(message["content"], list):
                # TODO: handle more general cases. where message['content'] is a dict and contains multiple types of content.
                # e.g.
                #  "content": [
                #     {
                #         "type": "image",
                #         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                #     },
                #     {"type": "text", "text": "Describe this image."},
                # ],
                logger.warning("message['content'] is a list of dicts, no processing will be done.")
                continue
            else:
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                )

        for placeholder, (mt, remaining) in multimodals.items():
            assert len(remaining) == 0, (
                f"Multimodal data count mismatch: {len(remaining)} more {mt.name}(s)"
                f"than '{placeholder}' placeholders in prompt"
            )

    return prompt


def _get_teacher_name(data: dict) -> str | None:
    for key in ("teacher_model", "teacher_model_name", "teacher"):
        if data.get(key) is not None:
            return data[key]
    return None


def _get_sample_ratio(sample: Sample) -> float | None:
    metadata = sample.metadata or {}
    value = metadata.get("sample_ratio")
    if value in (None, ""):
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid sample_ratio=%r for teacher=%s", value, sample.teacher_model_name)
        return None
    if ratio <= 0:
        logger.warning("Ignoring non-positive sample_ratio=%r for teacher=%s", value, sample.teacher_model_name)
        return None
    return ratio


def _sample_ratio_group(sample: Sample) -> str:
    metadata = sample.metadata or {}
    return str(metadata.get("teacher_id") or sample.teacher_model_name or metadata.get("task_id") or "default")


def _build_ratio_groups(samples: list[Sample]) -> tuple[dict[str, list[Sample]], dict[str, float]] | None:
    groups: dict[str, list[Sample]] = defaultdict(list)
    ratios: dict[str, float] = {}
    saw_ratio = False
    for sample in samples:
        key = _sample_ratio_group(sample)
        groups[key].append(sample)
        ratio = _get_sample_ratio(sample)
        if ratio is not None:
            saw_ratio = True
            if key in ratios and ratios[key] != ratio:
                logger.warning(
                    "Teacher %s has multiple sample_ratio values (%s, %s); using the first one.",
                    key,
                    ratios[key],
                    ratio,
                )
            else:
                ratios[key] = ratio

    if not saw_ratio or len(groups) <= 1:
        return None
    for key in groups:
        ratios.setdefault(key, 1.0)
    return dict(groups), ratios


def _largest_remainder_counts(keys: list[str], ratios: dict[str, float], total: int) -> dict[str, int]:
    ratio_sum = sum(ratios[key] for key in keys)
    raw_counts = {key: (ratios[key] / ratio_sum) * total for key in keys}
    counts = {key: int(raw_counts[key]) for key in keys}
    remainder = total - sum(counts.values())
    by_fraction = sorted(keys, key=lambda key: (raw_counts[key] - counts[key], ratios[key], key), reverse=True)
    for key in by_fraction[:remainder]:
        counts[key] += 1
    return counts


def _interleave_ratio_samples(
    groups: dict[str, list[Sample]],
    ratios: dict[str, float],
    *,
    seed: int,
    epoch_id: int,
) -> list[Sample]:
    rng = random.Random(seed + epoch_id)
    keys = sorted(groups)
    total = sum(len(items) for items in groups.values())
    counts = _largest_remainder_counts(keys, ratios, total)
    shuffled_groups: dict[str, list[Sample]] = {}
    for key in keys:
        items = list(groups[key])
        rng.shuffle(items)
        shuffled_groups[key] = items

    schedule = []
    for key in keys:
        count = counts[key]
        if count <= 0:
            continue
        for item_idx in range(count):
            schedule.append(((item_idx + 0.5) / count, rng.random(), key))
    schedule.sort()

    cursors = {key: 0 for key in keys}
    interleaved = []
    for _, _, key in schedule:
        items = shuffled_groups[key]
        if not items:
            continue
        cursor = cursors[key]
        interleaved.append(items[cursor % len(items)])
        cursors[key] = cursor + 1
    return interleaved


class Dataset:
    def __init__(
        self,
        path,
        tokenizer,
        processor,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
    ):
        origin_samples = []
        data_reader = (
            read_yaml_config(path, input_key=prompt_key, label_key=label_key, metadata_key=metadata_key)
            if isinstance(path, str) and path.endswith((".yaml", ".yml"))
            else read_file(path)
        )
        empty_teacher_num = 0
        for data in data_reader:
            # Both chat templates and multimodal inputs require conversation format (list of message dicts)
            as_conversation = apply_chat_template or (multimodal_keys is not None)
            prompt = _build_messages(data, prompt_key, as_conversation, multimodal_keys)

            metadata = _coerce_metadata(data.get(metadata_key))
            tools = None
            if tool_key is not None and tool_key in data:
                tools = data[tool_key]
                if isinstance(tools, str):
                    tools = json.loads(tools)
                elif isinstance(tools, np.ndarray):
                    tools = tools.tolist()
                assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                metadata["tools"] = tools

            if apply_chat_template:
                output_prompt = tokenizer.apply_chat_template(
                    prompt,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(apply_chat_template_kwargs or {}),
                )
            else:
                output_prompt = prompt

            if processor:
                from slime.utils.processing_utils import process_vision_info

                assert isinstance(
                    prompt, list
                ), f"prompt must be a list when processor is not None, got {type(prompt)} instead"
                multimodal_inputs = process_vision_info(prompt, processor)
            else:
                multimodal_inputs = None

            teacher_name = _get_teacher_name(data)
            if teacher_name is None:
                teacher_name = "default"
                empty_teacher_num += 1

            origin_samples.append(
                Sample(
                    prompt=output_prompt,
                    raw_prompt=prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=metadata,
                    multimodal_inputs=multimodal_inputs,
                    teacher_model_name=teacher_name,
                )
            )

        if empty_teacher_num:
            logger.warning("%d samples have no teacher field; using teacher_model_name='default'", empty_teacher_num)

        if max_length is not None:
            self.origin_samples = filter_long_prompt(origin_samples, tokenizer, processor, max_length)
        else:
            self.origin_samples = origin_samples

        self.epoch_id = -1
        self.seed = seed
        self._ratio_sampling = _build_ratio_groups(self.origin_samples)
        if self._ratio_sampling is not None:
            groups, ratios = self._ratio_sampling
            logger.info(
                "Using sample_ratio interleaving: %s",
                {
                    key: {"num_samples": len(groups[key]), "sample_ratio": ratios[key]}
                    for key in sorted(groups)
                },
            )
            self.samples = _interleave_ratio_samples(groups, ratios, seed=self.seed, epoch_id=0)
        else:
            self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        if self._ratio_sampling is not None:
            groups, ratios = self._ratio_sampling
            self.samples = _interleave_ratio_samples(groups, ratios, seed=self.seed, epoch_id=new_epoch_id)
        else:
            rng = random.Random(self.seed + new_epoch_id)
            permutation = list(range(len(self.origin_samples)))
            rng.shuffle(permutation)
            self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
