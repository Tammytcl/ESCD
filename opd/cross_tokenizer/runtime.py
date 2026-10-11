"""CPU-side retokenization helpers for the Megatron weight-switching runtime."""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any

import torch

from .base import AlignmentResult, CrossTokenizerAlgorithm, CrossTokenizerAlignmentError
from .tokenizer_bytes import semantic_stop_ids

logger = logging.getLogger(__name__)


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    try:
        return tokenizer.decode(list(token_ids), **kwargs)
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces")
        return tokenizer.decode(list(token_ids), **kwargs)


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]


def ensure_student_terminal_eos(
    rollout_data: dict[str, Any],
    *,
    student_tokenizer: Any,
    max_sequence_length: int | None = None,
) -> dict[str, Any]:
    """Validate Student trajectories without fabricating a terminal token.

    The historical name is retained for callers outside this repository.  BPM
    only creates a stop bridge when the Student *actually emitted* a semantic
    stop token.  Appending EOS to a length-truncated rollout changes both the
    response and the paper objective, so this helper is now validation-only.
    """

    if not semantic_stop_ids(student_tokenizer):
        raise ValueError("Cross-tokenizer OPD requires at least one semantic Student stop token")

    normalized = dict(rollout_data)
    normalized_tokens: list[Any] = []
    normalized_response_lengths: list[int] = []
    normalized_total_lengths: list[int] = []
    normalized_loss_masks: list[Any] = []
    for sample_index, (tokens, response_length, total_length, loss_mask) in enumerate(
        zip(
            rollout_data["tokens"],
            rollout_data["response_lengths"],
            rollout_data["total_lengths"],
            rollout_data["loss_masks"],
            strict=True,
        )
    ):
        token_ids = [int(token_id) for token_id in (tokens.detach().cpu().tolist() if isinstance(tokens, torch.Tensor) else tokens)]
        response_length = int(response_length)
        total_length = int(total_length)
        if total_length != len(token_ids):
            raise ValueError(
                f"Sample {sample_index} total_length={total_length} does not match token count {len(token_ids)}"
            )
        if not (0 <= response_length <= total_length):
            raise ValueError(f"Sample {sample_index} has invalid response_length={response_length}")
        mask_values = [int(value) for value in (loss_mask.detach().cpu().tolist() if isinstance(loss_mask, torch.Tensor) else loss_mask)]
        if len(mask_values) != response_length:
            raise ValueError(
                f"Sample {sample_index} loss-mask length {len(mask_values)} does not match response length {response_length}"
            )

        if max_sequence_length is not None and total_length > max_sequence_length:
            raise ValueError(
                f"Sample {sample_index} Student sequence length {total_length} exceeds configured "
                f"seq_length {max_sequence_length}"
            )

        if isinstance(tokens, torch.Tensor):
            normalized_tokens.append(torch.tensor(token_ids, dtype=tokens.dtype, device=tokens.device))
        else:
            normalized_tokens.append(token_ids)
        if isinstance(loss_mask, torch.Tensor):
            normalized_loss_masks.append(
                torch.tensor(mask_values, dtype=loss_mask.dtype, device=loss_mask.device)
            )
        else:
            normalized_loss_masks.append(mask_values)
        normalized_response_lengths.append(response_length)
        normalized_total_lengths.append(total_length)

    normalized["tokens"] = normalized_tokens
    normalized["response_lengths"] = normalized_response_lengths
    normalized["total_lengths"] = normalized_total_lengths
    normalized["loss_masks"] = normalized_loss_masks
    return normalized


def _joint_encode_with_boundary(tokenizer: Any, prompt: str, response: str) -> tuple[list[int], list[int]] | None:
    if not getattr(tokenizer, "is_fast", False) or not callable(tokenizer):
        return None
    full_text = prompt + response
    try:
        encoding = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    except (NotImplementedError, TypeError, ValueError):
        return None
    token_ids = encoding.get("input_ids")
    offsets = encoding.get("offset_mapping")
    if token_ids is None or offsets is None:
        return None
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
        offsets = offsets[0]
    boundary = len(prompt)
    split = 0
    for token_index, (start, end) in enumerate(offsets):
        start = int(start)
        end = int(end)
        if start < boundary < end:
            return None
        if end <= boundary:
            split = token_index + 1
    prompt_ids = [int(token_id) for token_id in token_ids[:split]]
    response_ids = [int(token_id) for token_id in token_ids[split:]]
    if not prompt_ids:
        return None
    if _decode(tokenizer, prompt_ids) + _decode(tokenizer, response_ids) != full_text:
        return None
    return prompt_ids, response_ids


def _encode_prompt_and_response(tokenizer: Any, prompt: str, response: str) -> tuple[list[int], list[int]]:
    joint = _joint_encode_with_boundary(tokenizer, prompt, response)
    if joint is not None:
        return joint

    prompt_ids = _encode(tokenizer, prompt)
    response_ids = _encode(tokenizer, response)
    if not prompt_ids:
        raise CrossTokenizerAlignmentError("Teacher prompt tokenization is empty")
    if _decode(tokenizer, prompt_ids + response_ids) != prompt + response:
        raise CrossTokenizerAlignmentError(
            "Teacher tokenizer cannot preserve the prompt/response text at an exact token boundary"
        )
    return prompt_ids, response_ids


def _teacher_prompt_text(
    *,
    raw_prompt: Any,
    raw_tools: Any,
    student_prompt_ids: list[int],
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    apply_teacher_chat_template: bool,
    teacher_chat_template_kwargs: dict[str, Any],
) -> str:
    if apply_teacher_chat_template and isinstance(raw_prompt, list):
        apply_chat_template = getattr(teacher_tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            kwargs = dict(teacher_chat_template_kwargs)
            if raw_tools is not None:
                kwargs["tools"] = raw_tools
            try:
                return apply_chat_template(
                    raw_prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
            except (TypeError, ValueError) as error:
                logger.warning("Teacher chat template was unavailable; using the decoded Student prompt: %s", error)
    if isinstance(raw_prompt, str) and not apply_teacher_chat_template:
        return raw_prompt
    return _decode(student_tokenizer, student_prompt_ids)


def _safe_skipped_teacher_sequence(
    teacher_tokenizer: Any, prompt_text: str, teacher_stop_id: int
) -> tuple[list[int], list[int]]:
    teacher_eos = int(teacher_stop_id)
    prompt_ids = _encode(teacher_tokenizer, prompt_text)
    if not prompt_ids:
        bos_id = getattr(teacher_tokenizer, "bos_token_id", None)
        prompt_ids = [int(bos_id) if bos_id is not None else teacher_eos]
    return prompt_ids, [teacher_eos]


def prepare_teacher_rollout_data(
    rollout_data: dict[str, Any],
    *,
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    algorithm: CrossTokenizerAlgorithm,
    alignment_error_policy: str,
    teacher_vocab_limit: int | None = None,
    max_sequence_length: int | None = None,
    apply_teacher_chat_template: bool = False,
    teacher_chat_template_kwargs: dict[str, Any] | None = None,
    student_stop_ids: Sequence[int] | None = None,
    teacher_stop_ids: Sequence[int] | None = None,
    teacher_bridge_stop_id: int | None = None,
) -> dict[str, Any]:
    """Retokenize Student trajectories and attach serializable alignment metadata."""

    if alignment_error_policy not in {"error", "skip"}:
        raise ValueError(f"Unsupported alignment_error_policy: {alignment_error_policy!r}")
    effective_student_stops = {
        int(value) for value in (student_stop_ids or semantic_stop_ids(student_tokenizer))
    }
    effective_teacher_stops = {
        int(value) for value in (teacher_stop_ids or semantic_stop_ids(teacher_tokenizer))
    }
    if not effective_student_stops:
        raise ValueError("Cross-tokenizer OPD requires at least one semantic Student stop token")
    if not effective_teacher_stops:
        raise ValueError("Cross-tokenizer OPD requires at least one semantic Teacher stop token")
    if teacher_bridge_stop_id is not None:
        teacher_bridge_stop = int(teacher_bridge_stop_id)
        if teacher_bridge_stop not in effective_teacher_stops:
            raise ValueError("Preferred Teacher bridge stop is absent from the declared stop set")
    else:
        declared_teacher_eos = getattr(teacher_tokenizer, "eos_token_id", None)
        if isinstance(declared_teacher_eos, (list, tuple, set)):
            declared_teacher_eos = next(
                (int(value) for value in declared_teacher_eos if int(value) in effective_teacher_stops),
                None,
            )
        teacher_bridge_stop = (
            int(declared_teacher_eos)
            if declared_teacher_eos is not None
            and int(declared_teacher_eos) in effective_teacher_stops
            else min(effective_teacher_stops)
        )

    rollout_data = ensure_student_terminal_eos(
        rollout_data,
        student_tokenizer=student_tokenizer,
        max_sequence_length=max_sequence_length,
    )

    teacher_data = dict(rollout_data)
    teacher_tokens: list[torch.Tensor] = []
    teacher_response_lengths: list[int] = []
    teacher_total_lengths: list[int] = []
    teacher_loss_masks: list[torch.Tensor] = []
    student_loss_masks: list[torch.Tensor] = []
    student_response_token_ids_for_teacher: list[torch.Tensor] = []
    alignments: list[dict[str, Any]] = []
    raw_prompts = rollout_data.get("raw_prompts")
    raw_tools = rollout_data.get("raw_tools")
    sample_count = len(rollout_data["tokens"])
    if raw_prompts is not None and len(raw_prompts) != sample_count:
        raise ValueError("raw_prompts length does not match the rollout batch")
    if raw_tools is not None and len(raw_tools) != sample_count:
        raise ValueError("raw_tools length does not match the rollout batch")

    for sample_index, (tokens, response_length) in enumerate(
        zip(
            rollout_data["tokens"],
            rollout_data["response_lengths"],
            strict=True,
        )
    ):
        student_ids = [int(token_id) for token_id in tokens.detach().cpu().tolist()]
        prompt_length = len(student_ids) - int(response_length)
        if prompt_length < 0:
            raise ValueError(f"Sample {sample_index} has response_length larger than its token sequence")
        student_prompt_ids = student_ids[:prompt_length]
        student_response_ids = student_ids[prompt_length:]
        student_response_token_ids_for_teacher.append(
            torch.tensor(student_response_ids, dtype=torch.long, device=tokens.device)
        )
        student_has_stop = bool(student_response_ids) and student_response_ids[-1] in effective_student_stops
        student_response_content_ids = (
            student_response_ids[:-1] if student_has_stop else student_response_ids
        )

        raw_prompt = raw_prompts[sample_index] if raw_prompts is not None else None
        raw_tool = raw_tools[sample_index] if raw_tools is not None else None
        teacher_prompt_text = _teacher_prompt_text(
            raw_prompt=raw_prompt,
            raw_tools=raw_tool,
            student_prompt_ids=student_prompt_ids,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            apply_teacher_chat_template=apply_teacher_chat_template,
            teacher_chat_template_kwargs=teacher_chat_template_kwargs or {},
        )
        response_text = _decode(student_tokenizer, student_response_content_ids)
        teacher_prompt_ids: list[int] | None = None
        teacher_response_ids: list[int] | None = None
        try:
            teacher_prompt_ids, teacher_response_content_ids = _encode_prompt_and_response(
                teacher_tokenizer, teacher_prompt_text, response_text
            )
            teacher_response_ids = teacher_response_content_ids + (
                [teacher_bridge_stop] if student_has_stop else []
            )
            if not teacher_response_ids:
                raise CrossTokenizerAlignmentError("Teacher response tokenization is empty")
            alignment = algorithm.align(teacher_response_ids, student_response_ids)
        except CrossTokenizerAlignmentError as error:
            if alignment_error_policy == "error":
                raise CrossTokenizerAlignmentError(
                    f"Cross-tokenizer alignment failed for sample {sample_index}: {error}"
                ) from error
            if teacher_prompt_ids is None or teacher_response_ids is None:
                teacher_prompt_ids, teacher_response_ids = _safe_skipped_teacher_sequence(
                    teacher_tokenizer, teacher_prompt_text, teacher_bridge_stop
                )
            alignment = AlignmentResult.skipped_result(
                len(teacher_response_ids), len(student_response_ids), str(error)
            )
        if not isinstance(alignment, AlignmentResult):
            raise TypeError(
                f"Cross-tokenizer algorithm returned {type(alignment).__name__} for sample {sample_index}; "
                "expected AlignmentResult"
            )
        expected_lengths = (len(teacher_response_ids), len(student_response_ids))
        actual_lengths = (alignment.teacher_length, alignment.student_length)
        if actual_lengths != expected_lengths:
            raise ValueError(
                f"Cross-tokenizer alignment lengths for sample {sample_index} are {actual_lengths}, "
                f"expected {expected_lengths}"
            )
        combined_teacher_ids = teacher_prompt_ids + teacher_response_ids

        if teacher_vocab_limit is not None and combined_teacher_ids:
            smallest_id = min(combined_teacher_ids)
            largest_id = max(combined_teacher_ids)
            if smallest_id < 0 or largest_id >= teacher_vocab_limit:
                raise ValueError(
                    f"Sample {sample_index} Teacher token ID range [{smallest_id}, {largest_id}] falls outside "
                    f"model vocabulary [0, {teacher_vocab_limit})"
                )
        if max_sequence_length is not None and len(combined_teacher_ids) > max_sequence_length:
            raise ValueError(
                f"Sample {sample_index} Teacher sequence length {len(combined_teacher_ids)} exceeds "
                f"configured seq_length {max_sequence_length}"
            )

        # The forward-only Teacher path may use this mask to compact logits.
        # Keep every Teacher response position so alignment/scoring shapes stay
        # valid; the original Student mask still controls the final loss.
        mapped_teacher_mask = [1] * len(teacher_response_ids)

        device = tokens.device
        teacher_tokens.append(torch.tensor(combined_teacher_ids, dtype=torch.long, device=device))
        teacher_response_lengths.append(len(teacher_response_ids))
        teacher_total_lengths.append(len(combined_teacher_ids))
        teacher_loss_masks.append(torch.tensor(mapped_teacher_mask, dtype=torch.int, device=device))
        original_student_mask = rollout_data["loss_masks"][sample_index]
        student_loss_masks.append(
            original_student_mask.to(device=device, dtype=torch.bool)
            if isinstance(original_student_mask, torch.Tensor)
            else torch.tensor(original_student_mask, dtype=torch.bool, device=device)
        )
        alignments.append(alignment.to_dict())

    teacher_data["tokens"] = teacher_tokens
    teacher_data["response_lengths"] = teacher_response_lengths
    teacher_data["total_lengths"] = teacher_total_lengths
    teacher_data["loss_masks"] = teacher_loss_masks
    teacher_data["cross_tokenizer_alignments"] = alignments
    teacher_data["cross_tokenizer_student_loss_masks"] = student_loss_masks
    # Preserve the realized Student response IDs alongside the retokenized
    # Teacher batch.  ULD's optional skip_student_eos switch must inspect the
    # actual terminal Student ID; the Teacher forward cannot infer it from
    # positional alignment alone (and a max-length rollout may have no EOS).
    teacher_data["cross_tokenizer_student_response_token_ids"] = student_response_token_ids_for_teacher
    teacher_data.pop("max_seq_lens", None)
    teacher_data.pop("opd_student_topk_ids", None)
    return teacher_data
