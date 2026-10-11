"""External SGLang Teacher adapter for heterogeneous cross-tokenizer OPD.

The legacy hooks in this module transport source-TopK input log-probabilities.
Causal-hidden plugins instead transport the Teacher's final hidden rows; Slime
applies a frozen sharded Teacher LM head and delegates exact scoring to the
registered cross-tokenizer algorithm.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import os
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import aiohttp
import torch
from transformers import AutoConfig, AutoTokenizer

from opd.reward import score_answer

from .base import (
    AlignmentResult,
    CandidateScores,
    CrossTokenizerAlignmentError,
    TeacherScoringContext,
    VocabularyMapping,
)
from .bpm import BPMZeroCompatibleTeacherMassError
from .registry import build_cross_tokenizer, cross_tokenizer_capabilities
from .runtime import _decode, _encode_prompt_and_response
from .sglang_binary_hidden import FORMAT_VERSION, binary_hidden_directory, sidecar_path
from .tokenizer_bytes import model_stop_ids, preferred_model_stop_id
from ._paper_utils import tokenizer_vocab_size


_RUNTIME: "_TeacherRuntime | None" = None
_SEMAPHORES: dict[str, asyncio.Semaphore] = {}
_VERIFIED_DENSE_TEACHERS: set[tuple[str, str]] = set()
_LOGGER = logging.getLogger(__name__)


def _positive_int(value: Any, *, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _teacher_urls() -> tuple[str, ...]:
    raw = os.environ.get("OPD_TEACHER_URLS", "")
    urls = tuple(item.strip().rstrip("/") for item in raw.split(",") if item.strip())
    if not urls:
        raise RuntimeError("OPD_TEACHER_URLS must contain at least one external SGLang Teacher URL")
    suffix = os.environ.get("OPD_TEACHER_ENDPOINT_SUFFIX", "generate").strip().strip("/")
    if suffix not in {"generate", "opd/score"}:
        raise ValueError("OPD_TEACHER_ENDPOINT_SUFFIX must be 'generate' or 'opd/score'")
    return tuple(
        url if url.endswith(f"/{suffix}") else f"{url}/{suffix}" for url in urls
    )


def _teacher_endpoint_suffix() -> str:
    suffix = os.environ.get("OPD_TEACHER_ENDPOINT_SUFFIX", "generate").strip().strip("/")
    if suffix not in {"generate", "opd/score"}:
        raise ValueError("OPD_TEACHER_ENDPOINT_SUFFIX must be 'generate' or 'opd/score'")
    return suffix


def _build_external_bpm_server_result(
    *,
    runtime: _TeacherRuntime,
    result: dict[str, Any],
    alignment: AlignmentResult,
    task_reward: float,
    teacher_url: str,
    student_response_length: int,
    teacher_response_length: int,
    candidate_top_k: int,
    reward_fields: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Translate the Kimi no-KV ``/opd/score`` schema to Slime's payload.

    The server has already performed BPM routing on its frozen Kimi LM head.
    This adapter only validates mass/row identity and converts JSON arrays to
    the generic plugin payload; no Slime loss or model code is bypassed.
    """

    del runtime
    rows = list(alignment.algorithm_metadata.get("bpm_rows", ()))
    ids = result.get("target_ids")
    probs = result.get("target_probs")
    complements = result.get("complement_probs")
    if not all(isinstance(value, list) for value in (ids, probs, complements)):
        raise ValueError("Kimi BPM response is missing target arrays")
    if not (len(ids) == len(probs) == len(complements) == len(rows)):
        raise ValueError("Kimi BPM response row count differs from the alignment")
    active: list[int] = []
    for index, row in enumerate(rows):
        if bool(row.get("whitespace", False)):
            continue
        if str(row.get("kind")) == "spanning":
            continue
        row_ids = [int(value) for value in ids[index]]
        row_probs = [float(value) for value in probs[index]]
        complement = float(complements[index])
        if len(row_ids) != candidate_top_k or len(row_probs) != candidate_top_k:
            raise ValueError("Kimi BPM response Top-K width differs from the request")
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("Kimi BPM response contains duplicate Student target IDs")
        if any((not math.isfinite(value) or value < 0.0) for value in (*row_probs, complement)):
            raise ValueError("Kimi BPM response contains invalid probabilities")
        if not math.isclose(sum(row_probs) + complement, 1.0, abs_tol=3e-4):
            raise ValueError("Kimi BPM response is not mass preserving")
        active.append(index)
    payload = {
        "target_ids": [ids[index] for index in active],
        "target_probs": [probs[index] for index in active],
        "target_mask": [[True] * candidate_top_k for _ in active],
        "target_complement": [complements[index] for index in active],
        "student_positions": [int(rows[index]["student_position"]) for index in active],
        "row_kinds": [
            {"aligned": 0, "interior": 1, "spanning": 2, "stop": 3}[str(rows[index]["kind"])]
            for index in active
        ],
    }
    return {
        **(reward_fields or {"task_reward": task_reward}),
        "teacher_url": teacher_url.rsplit("/opd/score", 1)[0],
        "teacher_cross_payload": payload,
        "teacher_cross_skipped_mask": [False] * student_response_length,
        "student_response_length_with_eos": student_response_length,
        "teacher_response_length_with_eos": teacher_response_length,
        "cross_tokenizer_algorithm": "bpm",
    }


def _select_teacher_url(sample: Any) -> str:
    urls = _teacher_urls()
    sample_id = getattr(sample, "index", None)
    if sample_id is None:
        sample_id = getattr(sample, "group_index", 0) or 0
    return urls[int(sample_id) % len(urls)]


def _canonical_model_path(value: str) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


async def _verify_dense_teacher_model(
    session: aiohttp.ClientSession,
    *,
    teacher_url: str,
    expected_model_path: str,
) -> None:
    """Fail before training if a causal-hidden endpoint serves another checkpoint."""

    expected = _canonical_model_path(expected_model_path)
    base_url = teacher_url
    for endpoint_suffix in ("/opd/score", "/generate"):
        if base_url.endswith(endpoint_suffix):
            base_url = base_url[: -len(endpoint_suffix)]
            break
    key = (base_url, expected)
    if key in _VERIFIED_DENSE_TEACHERS:
        return
    async with session.get(f"{base_url}/model_info") as response:
        response.raise_for_status()
        model_info = await response.json()
    served_model_path = model_info.get("model_path") if isinstance(model_info, dict) else None
    if not isinstance(served_model_path, str) or not served_model_path.strip():
        raise ValueError(f"Causal-hidden Teacher {base_url} omitted model_path from /model_info")
    served = _canonical_model_path(served_model_path)
    if served != expected:
        raise RuntimeError(
            "Causal-hidden Teacher endpoint serves the wrong checkpoint: "
            f"endpoint={base_url}, served={served}, expected={expected}"
        )
    _VERIFIED_DENSE_TEACHERS.add(key)


def _reward_breakdown(sample: Any) -> dict[str, float]:
    """Return stable math/code/aggregate reward fields for one rollout.

    The mixed BPM corpus labels code rows with ``domain=code`` and
    ``rm_type=code_sandbox``.  The code sandbox scorer is intentionally not
    run by this adapter yet, so those rows expose an explicit zero code
    reward.  Math rows retain the existing answer verifier unchanged.
    """

    metadata = getattr(sample, "metadata", None) or {}
    label = getattr(sample, "label", None)
    family = str(metadata.get("reward_family") or "math_verify")
    label_domain = ""
    if isinstance(label, dict):
        family = str(label.get("reward_family") or family)
        label_domain = str(label.get("domain") or "").lower()
        label = label.get("ground_truth", label.get("answer", label.get("label")))
    domain = str(metadata.get("domain") or "").lower()
    rm_type = str(metadata.get("rm_type") or "").lower()
    is_code = label_domain == "code" or domain == "code" or "code" in rm_type or "sandbox" in rm_type
    reward_math = 0.0 if is_code else float(
        score_answer(getattr(sample, "response", ""), label, family=family)
    )
    reward_code = 0.0
    return {
        "reward_math": reward_math,
        "reward_code": reward_code,
        "reward": reward_math + reward_code,
    }


def _task_reward(sample: Any) -> float:
    return float(_reward_breakdown(sample)["reward"])


def _reward_fields(sample: Any, *, task_reward: float | None = None) -> dict[str, float]:
    fields = _reward_breakdown(sample)
    if task_reward is not None:
        fields["reward"] = float(task_reward)
    fields["task_reward"] = float(fields["reward"])
    return fields


def _extract_sglang_prefill_hidden(
    raw_hidden: Any,
    *,
    input_length: int,
    prompt_length: int,
    response_length: int,
) -> torch.Tensor:
    """Extract causal response rows from SGLang's first hidden-state chunk.

    With one generated throw-away token, SGLang returns a one-element list;
    that first element contains all prefill rows. Row ``k`` predicts token
    ``k+1``, so response supervision starts at the final prompt row.
    """

    if input_length <= 0 or prompt_length <= 0:
        raise ValueError("Dense BPM requires a non-empty Teacher prompt and input")
    if response_length < 0 or prompt_length + response_length != input_length:
        raise ValueError("Dense BPM Teacher prompt/response lengths are inconsistent")
    hidden = torch.as_tensor(raw_hidden, dtype=torch.bfloat16)
    while hidden.ndim > 2 and hidden.size(0) == 1:
        hidden = hidden.squeeze(0)
    if hidden.ndim != 2 or hidden.size(0) != input_length:
        raise ValueError(
            "Teacher prefill hidden-state shape differs from its input: "
            f"hidden={tuple(hidden.shape)}, tokens={input_length}"
        )
    causal_start = prompt_length - 1
    causal_end = input_length - 1
    response_hidden = hidden[causal_start:causal_end]
    if response_hidden.size(0) != response_length:
        raise ValueError("Teacher causal hidden rows do not match its response tokenization")
    return response_hidden.contiguous()


def _load_binary_response_hidden(
    directory: Path,
    *,
    rid: str,
    input_length: int,
    prompt_length: int,
    response_length: int,
    timeout_seconds: float,
) -> torch.Tensor:
    """Load and validate one atomic SGLang hidden-state sidecar.

    Only the causal rows needed for the Teacher response are concatenated;
    prompt-only rows are validated but do not survive this boundary.
    """

    path = sidecar_path(directory, rid)
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            # Keep the first actionable failure self-contained.  A missing
            # sidecar can mean a PP-stage write never happened, a writer rank
            # died, or CPFS is still exposing a temporary file; the directory
            # census makes those cases distinguishable in a Ray traceback.
            directory_exists = directory.is_dir()
            complete: list[str] = []
            temporary: list[str] = []
            try:
                if directory_exists:
                    entries = sorted(directory.iterdir(), key=lambda item: item.name)
                    complete = [item.name for item in entries if item.suffix == ".pt"]
                    temporary = [item.name for item in entries if item.suffix == ".tmp"]
            except OSError as error:
                raise TimeoutError(
                    "Timed out waiting for BPM Teacher hidden sidecar: "
                    f"{path} (directory_exists={directory_exists}; "
                    f"directory_scan_error={error!r})"
                ) from error
            preview = ",".join((*complete[:4], *temporary[:4])) or "none"
            raise TimeoutError(
                "Timed out waiting for BPM Teacher hidden sidecar: "
                f"{path} (directory_exists={directory_exists}; "
                f"complete_sidecars={len(complete)}; temporary_sidecars={len(temporary)}; "
                f"entries={preview})"
            )
        time.sleep(0.05)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    finally:
        # This is a single-use file produced for our UUID under the run-scoped
        # transport directory.  Removing it bounds CPFS usage during training.
        path.unlink(missing_ok=True)
    if not isinstance(payload, dict):
        raise ValueError("BPM Teacher hidden sidecar is not a dictionary")
    if int(payload.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("BPM Teacher hidden sidecar has an unsupported format version")
    if payload.get("rid") != rid:
        raise ValueError("BPM Teacher hidden sidecar request ID does not match the HTTP request")
    if int(payload.get("input_length", -1)) != input_length:
        raise ValueError("BPM Teacher hidden sidecar input length differs from the request")
    chunks = payload.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("BPM Teacher hidden sidecar contains no chunks")
    hidden_size = int(payload.get("hidden_size", -1))
    if hidden_size <= 0:
        raise ValueError("BPM Teacher hidden sidecar has an invalid hidden width")

    causal_start = prompt_length - 1
    causal_end = input_length - 1
    if prompt_length <= 0 or response_length < 0 or causal_end - causal_start != response_length:
        raise ValueError("Dense BPM Teacher prompt/response lengths are inconsistent")
    selected: list[torch.Tensor] = []
    row_offset = 0
    for chunk in chunks:
        if (
            not isinstance(chunk, torch.Tensor)
            or chunk.ndim != 2
            or chunk.device.type != "cpu"
            or chunk.size(1) != hidden_size
        ):
            raise ValueError("BPM Teacher hidden sidecar contains an inconsistent chunk")
        next_offset = row_offset + int(chunk.size(0))
        overlap_start = max(causal_start, row_offset)
        overlap_end = min(causal_end, next_offset)
        if overlap_start < overlap_end:
            selected.append(chunk[overlap_start - row_offset : overlap_end - row_offset])
        row_offset = next_offset
    if row_offset != input_length:
        raise ValueError(
            "BPM Teacher hidden sidecar rows differ from the request: "
            f"hidden={row_offset}, tokens={input_length}"
        )
    if response_length == 0:
        return torch.empty((0, hidden_size), dtype=torch.bfloat16)
    if not selected:
        raise ValueError("BPM Teacher hidden sidecar omitted all causal response rows")
    response_hidden = torch.cat(selected, dim=0).to(dtype=torch.bfloat16).contiguous()
    if response_hidden.size(0) != response_length:
        raise ValueError("BPM Teacher binary causal rows do not match its response tokenization")
    return response_hidden


def _parse_top_entry(entry: Any) -> tuple[int, float]:
    if isinstance(entry, dict):
        token_id = entry.get("token_id", entry.get("id"))
        log_prob = entry.get("logprob", entry.get("log_prob"))
    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
        log_prob, token_id = entry[0], entry[1]
    else:
        raise ValueError(f"Unsupported SGLang Top-K entry: {entry!r}")
    token_id = int(token_id)
    log_prob = float(log_prob)
    if token_id < 0 or not math.isfinite(log_prob):
        raise ValueError(f"Invalid SGLang Top-K entry: token_id={token_id}, log_prob={log_prob}")
    return token_id, log_prob


class _SparseTopKDistribution:
    """VocabularyDistribution backed by normalized SGLang Top-K rows."""

    def __init__(self, token_ids: torch.Tensor, log_probs: torch.Tensor, mask: torch.Tensor) -> None:
        if token_ids.ndim != 2 or log_probs.shape != token_ids.shape or mask.shape != token_ids.shape:
            raise ValueError("Sparse Teacher tensors must have equal [rows, K] shapes")
        self._token_ids = token_ids
        self._log_probs = log_probs
        self._mask = mask

    @classmethod
    def from_sglang(
        cls,
        rows: Sequence[Any],
        *,
        expected_k: int,
        forced_tokens: Sequence[tuple[int, float] | None] | None = None,
    ) -> "_SparseTopKDistribution":
        if forced_tokens is not None and len(forced_tokens) != len(rows):
            raise ValueError("Forced Teacher tokens must match the number of sparse rows")
        parsed_rows: list[list[tuple[int, float]]] = []
        for row_index, row in enumerate(rows):
            if row is None:
                raise ValueError(f"SGLang omitted Top-K values for Teacher response row {row_index}")
            parsed: list[tuple[int, float]] = []
            seen: set[int] = set()
            for entry in row:
                token_id, log_prob = _parse_top_entry(entry)
                if token_id in seen:
                    continue
                seen.add(token_id)
                parsed.append((token_id, log_prob))
            if len(parsed) < expected_k:
                raise ValueError(
                    f"SGLang returned only {len(parsed)} unique candidates for row {row_index}; "
                    f"requested source Top-{expected_k}"
                )
            parsed = parsed[:expected_k]
            forced = None if forced_tokens is None else forced_tokens[row_index]
            if forced is not None:
                forced_id, forced_log_prob = int(forced[0]), float(forced[1])
                if forced_id < 0 or not math.isfinite(forced_log_prob):
                    raise ValueError(
                        f"Invalid forced Teacher token at row {row_index}: "
                        f"token_id={forced_id}, log_prob={forced_log_prob}"
                    )
                parsed = [item for item in parsed if item[0] != forced_id]
                parsed.append((forced_id, forced_log_prob))
                parsed.sort(key=lambda item: (-item[1], item[0]))
                parsed = parsed[:expected_k]
                # A forced stop may sit below the source Top-K threshold.  The
                # paper's stop bridge nevertheless requires its exact mass, so
                # replace the lowest ordinary candidate rather than dropping it.
                if all(token_id != forced_id for token_id, _ in parsed):
                    parsed[-1] = (forced_id, forced_log_prob)
                    parsed.sort(key=lambda item: (-item[1], item[0]))
            parsed_rows.append(parsed)
        return cls(
            torch.tensor([[item[0] for item in row] for row in parsed_rows], dtype=torch.long),
            torch.tensor([[item[1] for item in row] for row in parsed_rows], dtype=torch.float32),
            torch.ones((len(parsed_rows), expected_k), dtype=torch.bool),
        )

    @property
    def length(self) -> int:
        return int(self._token_ids.size(0))

    def rows(self, row_indices: torch.Tensor) -> "_SparseTopKDistribution":
        indices = row_indices.detach().cpu().to(dtype=torch.long)
        return _SparseTopKDistribution(
            self._token_ids.index_select(0, indices),
            self._log_probs.index_select(0, indices),
            self._mask.index_select(0, indices),
        )

    def topk(self, k: int) -> CandidateScores:
        if not 0 < k <= self._token_ids.size(1):
            raise ValueError(f"Sparse Teacher only contains Top-{self._token_ids.size(1)}, requested {k}")
        return CandidateScores(self._token_ids[:, :k], self._log_probs[:, :k], self._mask[:, :k])

    def selected_log_probs(
        self,
        token_ids: torch.Tensor,
        *,
        vocab_size: int | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        del vocab_size
        if float(temperature) != 1.0:
            raise ValueError("Sparse external Teacher scores only support temperature=1.0")
        requested = token_ids.detach().cpu().to(dtype=torch.long)
        if requested.ndim == 1:
            requested = requested.unsqueeze(-1)
        if requested.size(0) != self.length:
            raise ValueError("selected_log_probs row count differs from the sparse Teacher distribution")
        matches = requested.unsqueeze(-1) == self._token_ids.unsqueeze(1)
        valid = matches & self._mask.unsqueeze(1)
        if not bool(valid.any(dim=-1).all()):
            raise ValueError("Requested token is outside the cached external Teacher Top-K support")
        offsets = valid.to(dtype=torch.int64).argmax(dim=-1)
        return self._log_probs.unsqueeze(1).expand(-1, requested.size(1), -1).gather(
            -1, offsets.unsqueeze(-1)
        ).squeeze(-1)

    def full_probabilities(
        self,
        *,
        vocab_size: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        del vocab_size, temperature
        raise ValueError("Sparse external Teacher does not contain a full vocabulary distribution")

    def selected_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        del token_ids
        raise ValueError("SGLang Top-K transport exposes normalized log-probabilities, not raw logits")

    def mapped_topk(
        self,
        source_ids: torch.Tensor,
        output_ids: torch.Tensor,
        *,
        k: int,
        vocab_size: int | None = None,
        temperature: float = 1.0,
    ) -> CandidateScores:
        del vocab_size
        if float(temperature) != 1.0:
            raise ValueError("Sparse external Teacher scores only support temperature=1.0")
        source_ids = source_ids.detach().cpu().to(dtype=torch.long)
        output_ids = output_ids.detach().cpu().to(dtype=torch.long)
        if source_ids.ndim != 1 or output_ids.shape != source_ids.shape:
            raise ValueError("mapped_topk mappings must be equal one-dimensional tensors")
        lookup = {int(source): int(target) for source, target in zip(source_ids, output_ids, strict=True)}
        rows: list[list[tuple[int, float]]] = []
        for ids, scores, mask in zip(self._token_ids, self._log_probs, self._mask, strict=True):
            mapped = [
                (lookup[int(token_id)], float(score))
                for token_id, score, valid in zip(ids, scores, mask, strict=True)
                if bool(valid) and int(token_id) in lookup
            ]
            mapped.sort(key=lambda item: item[1], reverse=True)
            rows.append(mapped[:k] if k else mapped)
        width = max((len(row) for row in rows), default=0)
        ids_out = torch.zeros((self.length, width), dtype=torch.long)
        scores_out = torch.full((self.length, width), float("-inf"), dtype=torch.float32)
        mask_out = torch.zeros((self.length, width), dtype=torch.bool)
        for row_index, row in enumerate(rows):
            for column, (token_id, score) in enumerate(row):
                ids_out[row_index, column] = token_id
                scores_out[row_index, column] = score
                mask_out[row_index, column] = True
        return CandidateScores(ids_out, scores_out, mask_out)


class _SelectedTokenDistribution:
    """Dense rows over an explicitly selected Teacher token set.

    SGLang's ``token_ids_logprob`` API avoids sorting the entire vocabulary and
    returns exactly the paper SimCT shared-token support. Observed continuation
    scores are kept separately because a span token need not itself be shared.
    """

    def __init__(
        self,
        selected_ids: torch.Tensor,
        selected_log_probs: torch.Tensor,
        observed_ids: torch.Tensor,
        observed_log_probs: torch.Tensor,
    ) -> None:
        if selected_ids.ndim != 1 or selected_log_probs.ndim != 2:
            raise ValueError("Selected Teacher tensors must be [V] and [rows, V]")
        if selected_log_probs.size(1) != selected_ids.numel():
            raise ValueError("Selected Teacher ID and score widths differ")
        if observed_ids.ndim != 1 or observed_log_probs.shape != observed_ids.shape:
            raise ValueError("Observed Teacher tensors must have equal [rows] shapes")
        if selected_log_probs.size(0) != observed_ids.numel():
            raise ValueError("Selected and observed Teacher row counts differ")
        self._selected_ids = selected_ids
        self._selected_log_probs = selected_log_probs
        self._observed_ids = observed_ids
        self._observed_log_probs = observed_log_probs

    @property
    def length(self) -> int:
        return int(self._selected_log_probs.size(0))

    def rows(self, row_indices: torch.Tensor) -> "_SelectedTokenDistribution":
        indices = row_indices.detach().cpu().to(dtype=torch.long)
        return _SelectedTokenDistribution(
            self._selected_ids,
            self._selected_log_probs.index_select(0, indices),
            self._observed_ids.index_select(0, indices),
            self._observed_log_probs.index_select(0, indices),
        )

    def selected_log_probs(
        self,
        token_ids: torch.Tensor,
        *,
        vocab_size: int | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        del vocab_size
        if float(temperature) != 1.0:
            raise ValueError("Selected external Teacher scores only support temperature=1.0")
        requested = token_ids.detach().cpu().to(dtype=torch.long)
        if requested.ndim == 1:
            requested = requested.unsqueeze(-1)
        if requested.size(0) != self.length:
            raise ValueError("selected_log_probs row count differs from the Teacher distribution")

        output = torch.empty(requested.shape, dtype=self._selected_log_probs.dtype)
        observed_match = requested == self._observed_ids.unsqueeze(-1)
        if bool(observed_match.any()):
            output[observed_match] = self._observed_log_probs.unsqueeze(-1).expand_as(requested)[observed_match]
        unresolved = ~observed_match
        if bool(unresolved.any()):
            matches = requested.unsqueeze(-1) == self._selected_ids.view(1, 1, -1)
            if not bool((matches.any(dim=-1) | observed_match).all()):
                raise ValueError("Requested token is outside selected and observed Teacher support")
            offsets = matches.to(dtype=torch.int64).argmax(dim=-1)
            selected = self._selected_log_probs.unsqueeze(1).expand(-1, requested.size(1), -1).gather(
                -1, offsets.unsqueeze(-1)
            ).squeeze(-1)
            output[unresolved] = selected[unresolved]
        return output

    def full_probabilities(
        self,
        *,
        vocab_size: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        del vocab_size, temperature
        raise ValueError("Selected external Teacher does not contain non-shared vocabulary probabilities")

    def selected_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        del token_ids
        raise ValueError("SGLang selected-token transport exposes normalized log-probabilities, not raw logits")

    def topk(self, k: int) -> CandidateScores:
        if not 0 < k <= self._selected_ids.numel():
            raise ValueError(f"Invalid selected Teacher Top-K: {k}")
        scores, offsets = self._selected_log_probs.topk(k, dim=-1)
        ids = self._selected_ids[offsets]
        return CandidateScores(ids, scores, torch.ones_like(ids, dtype=torch.bool))

    def mapped_topk(
        self,
        source_ids: torch.Tensor,
        output_ids: torch.Tensor,
        *,
        k: int,
        vocab_size: int | None = None,
        temperature: float = 1.0,
    ) -> CandidateScores:
        del vocab_size
        if float(temperature) != 1.0:
            raise ValueError("Selected external Teacher scores only support temperature=1.0")
        source_ids = source_ids.detach().cpu().to(dtype=torch.long)
        output_ids = output_ids.detach().cpu().to(dtype=torch.long)
        if not torch.equal(source_ids, self._selected_ids):
            raise ValueError("Selected Teacher token order differs from the SimCT shared vocabulary")
        scores = self._selected_log_probs
        mapped_ids = output_ids.unsqueeze(0).expand(self.length, -1)
        if k:
            scores, offsets = scores.topk(k, dim=-1)
            mapped_ids = mapped_ids.gather(-1, offsets)
        return CandidateScores(
            mapped_ids,
            scores,
            torch.ones_like(mapped_ids, dtype=torch.bool),
        )


@dataclass
class _TeacherRuntime:
    student_tokenizer: Any
    teacher_tokenizer: Any
    algorithm: Any
    vocabulary_mapping: VocabularyMapping
    student_stop_ids: tuple[int, ...]
    teacher_stop_ids: tuple[int, ...]
    student_bridge_stop_id: int
    teacher_bridge_stop_id: int
    teacher_hidden_size: int | None = None


def _runtime(args: Any) -> _TeacherRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        student = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        teacher = AutoTokenizer.from_pretrained(args.opd_teacher_tokenizer, trust_remote_code=True)
        student_stops = model_stop_ids(args.hf_checkpoint, student)
        teacher_stops = model_stop_ids(args.opd_teacher_tokenizer, teacher)
        # ULD's released data path appends the Student tokenizer EOS to each
        # Teacher-generated answer.  Prefer that ID over a model config's
        # fallback EOS (Qwen3.5 declares <|endoftext|> in config.json while its
        # chat tokenizer terminates assistant turns with <|im_end|>).
        tokenizer_eos = getattr(student, "eos_token_id", None)
        if isinstance(tokenizer_eos, int) and tokenizer_eos in set(student_stops):
            student_bridge_stop_id = tokenizer_eos
        else:
            student_bridge_stop_id = preferred_model_stop_id(
                args.hf_checkpoint,
                student,
                declared_stop_ids=student_stops,
            )
        teacher_bridge_stop_id = preferred_model_stop_id(
            getattr(args, "opd_teacher_model_path", None) or args.opd_teacher_tokenizer,
            teacher,
            declared_stop_ids=teacher_stops,
        )
        algorithm_kwargs = dict(args.cross_tokenizer_algorithm_kwargs)
        teacher_model_path = getattr(args, "opd_teacher_model_path", None) or args.opd_teacher_tokenizer
        student_config = AutoConfig.from_pretrained(
            args.hf_checkpoint, trust_remote_code=True
        )
        teacher_config = AutoConfig.from_pretrained(
            teacher_model_path, trust_remote_code=True
        )
        student_text_config = getattr(student_config, "text_config", student_config)
        teacher_text_config = getattr(teacher_config, "text_config", teacher_config)
        algorithm = build_cross_tokenizer(
            args.cross_tokenizer_algorithm,
            student_tokenizer=student,
            teacher_tokenizer=teacher,
            algorithm_kwargs=algorithm_kwargs,
        )
        # Model vocabulary widths and generation-config stop IDs are part of
        # the generic plugin contract.  Calling the hook for every algorithm
        # avoids name-based framework branches and lets new byte algorithms
        # reuse the same causal-hidden transport without Slime changes.
        algorithm.configure_model_contract(
            # The HF model config may include rows added only to make the
            # output projection TP-divisible (Qwen3.6: 248320 rows vs.
            # 248077 tokenizer IDs).  Those padding rows must not enter the
            # BPM softmax/complement cell or the routing artifact.
            student_vocab_size=tokenizer_vocab_size(student),
            teacher_vocab_size=tokenizer_vocab_size(teacher),
            student_stop_ids=student_stops,
            teacher_stop_ids=teacher_stops,
        )
        shared = algorithm.build_shared_vocabulary()
        if shared is None:
            mapping = VocabularyMapping(
                teacher_ids=torch.empty((0,), dtype=torch.long),
                student_ids=torch.empty((0,), dtype=torch.long),
            )
        else:
            mapping = VocabularyMapping(
                teacher_ids=torch.tensor(shared.teacher_ids, dtype=torch.long),
                student_ids=torch.tensor(shared.student_ids, dtype=torch.long),
            )
        if algorithm.capabilities.teacher_transport("sglang") == "causal_hidden":
            teacher_text_config = getattr(teacher_config, "text_config", teacher_config)
            teacher_hidden_size = int(teacher_text_config.hidden_size)
        else:
            teacher_hidden_size = None
        _RUNTIME = _TeacherRuntime(
            student,
            teacher,
            algorithm,
            mapping,
            student_stops,
            teacher_stops,
            student_bridge_stop_id,
            teacher_bridge_stop_id,
            teacher_hidden_size,
        )
    return _RUNTIME


def _normalized_student_sequences(sample: Any, runtime: _TeacherRuntime):
    response_length = int(getattr(sample, "response_length", 0) or 0)
    tokens = [int(value) for value in (getattr(sample, "tokens", None) or [])]
    if response_length <= 0 or response_length > len(tokens):
        raise ValueError("Cross-tokenizer Teacher scoring requires a non-empty valid Student response")
    prompt_ids = tokens[:-response_length]
    student_response_ids = tokens[-response_length:]
    loss_mask = list(getattr(sample, "loss_mask", None) or [1] * response_length)
    if len(loss_mask) != response_length:
        raise ValueError("Student loss mask length differs from response_length")
    return prompt_ids, student_response_ids, loss_mask


def _teacher_prompt_text(args: Any, sample: Any, runtime: _TeacherRuntime) -> str:
    """Render the original chat with the frozen Teacher's template."""

    raw_prompt = getattr(sample, "raw_prompt", None)
    metadata = getattr(sample, "metadata", None) or {}
    if raw_prompt is None:
        raw_prompt = metadata.get("raw_prompt")
    if not isinstance(raw_prompt, list):
        raise ValueError(
            "The causal-hidden cross-tokenizer transport requires raw chat messages so the trajectory "
            "can be rerendered with the Teacher chat template"
        )
    template_kwargs = dict(getattr(args, "apply_chat_template_kwargs", {}) or {})
    raw_tools = getattr(sample, "raw_tools", None)
    if raw_tools is None:
        raw_tools = metadata.get("raw_tools", metadata.get("tools"))
    if raw_tools is not None:
        template_kwargs["tools"] = raw_tools
    # Ordinary distillation does not depend on the optional skill experiment.
    # Still validate its frozen contract when either configuration or sample
    # metadata explicitly requests it.
    suffix = ""
    if os.environ.get("OPD_SKILL_CONFIG_JSON") or metadata.get("opd_skill_runtime_v1") is not None:
        from opd.skill_runtime import teacher_prompt_suffix

        suffix = teacher_prompt_suffix(sample)

    return runtime.teacher_tokenizer.apply_chat_template(
        raw_prompt,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    ) + suffix


def _normalized_sequences(args: Any, sample: Any, runtime: _TeacherRuntime):
    prompt_ids, student_response_ids, loss_mask = _normalized_student_sequences(sample, runtime)
    student_has_stop = bool(student_response_ids) and (
        student_response_ids[-1] in set(runtime.student_stop_ids)
    )
    response_content_ids = student_response_ids[:-1] if student_has_stop else student_response_ids

    teacher_prompt_text = _teacher_prompt_text(args, sample, runtime)
    response_text = _decode(runtime.student_tokenizer, response_content_ids)
    teacher_prompt_ids, teacher_response_content_ids = _encode_prompt_and_response(
        runtime.teacher_tokenizer, teacher_prompt_text, response_text
    )
    teacher_stop = runtime.teacher_bridge_stop_id
    teacher_response_ids = teacher_response_content_ids + ([teacher_stop] if student_has_stop else [])
    if not teacher_response_ids:
        raise CrossTokenizerAlignmentError("Teacher response tokenization is empty")
    max_teacher_input_tokens = int(os.environ.get("OPD_TEACHER_MAX_INPUT_TOKENS", "0") or "0")
    teacher_input_length = len(teacher_prompt_ids) + len(teacher_response_ids)
    if max_teacher_input_tokens > 0 and teacher_input_length > max_teacher_input_tokens:
        # The external SGLang Teacher rejects inputs beyond its context
        # budget.  Report the overflow as a skippable alignment error so the
        # configured alignment_error_policy decides whether the run survives
        # trajectories the Teacher cannot score.
        raise CrossTokenizerAlignmentError(
            f"Teacher input length {teacher_input_length} exceeds the external Teacher context "
            f"budget OPD_TEACHER_MAX_INPUT_TOKENS={max_teacher_input_tokens}"
        )
    alignment = runtime.algorithm.align(teacher_response_ids, student_response_ids)
    return teacher_prompt_ids, teacher_response_ids, student_response_ids, loss_mask, alignment


def _sglang_output_token_ids(result: dict[str, Any]) -> list[int]:
    """Read generated token IDs without re-tokenizing SGLang's output text."""

    output_ids = result.get("output_ids")
    if isinstance(output_ids, list) and all(isinstance(value, int) for value in output_ids):
        parsed = [int(value) for value in output_ids]
    else:
        rows = result.get("meta_info", {}).get("output_token_logprobs")
        if not isinstance(rows, list):
            raise ValueError("SGLang Teacher generation omitted output token IDs")
        parsed = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or len(row) < 2 or row[1] is None:
                raise ValueError(
                    f"Invalid SGLang output_token_logprobs row {row_index}: {row!r}"
                )
            parsed.append(int(row[1]))
    rows = result.get("meta_info", {}).get("output_token_logprobs")
    if isinstance(rows, list) and len(rows) == len(parsed):
        sampled_ids = [int(row[1]) for row in rows]
        if sampled_ids != parsed:
            raise ValueError("SGLang Teacher output_ids disagree with output_token_logprobs")
    if not parsed:
        raise CrossTokenizerAlignmentError("ULD Teacher generated an empty answer")
    return parsed


def _sglang_finish_reason(result: dict[str, Any]) -> str:
    reason = result.get("meta_info", {}).get("finish_reason")
    if isinstance(reason, dict):
        return str(reason.get("type") or reason.get("reason") or "unknown").lower()
    return str(reason or "unknown").lower()


def _encode_response(tokenizer: Any, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().reshape(-1).tolist()
    return [int(value) for value in token_ids]


def _uld_teacher_answer_trajectory(
    args: Any,
    sample: Any,
    runtime: _TeacherRuntime,
    *,
    teacher_output_ids: Sequence[int],
    finish_reason: str,
) -> tuple[Any, str, bool]:
    """Build ULD's supervised Student trajectory from one Teacher answer.

    The released ULD recipe first generates an answer with the Teacher and
    then tokenizes that same answer for both models.  The prior Slime adapter
    instead retained the Student rollout, which made CE self-training and left
    the rank-only ULD term with no token-identity supervision.
    """

    student_prompt_ids, _, _ = _normalized_student_sequences(sample, runtime)
    raw_teacher_ids = [int(value) for value in teacher_output_ids]
    teacher_has_terminal_id = bool(raw_teacher_ids) and (
        raw_teacher_ids[-1] in set(runtime.teacher_stop_ids)
    )
    teacher_completed = finish_reason == "stop" or teacher_has_terminal_id
    teacher_content_ids = raw_teacher_ids[:-1] if teacher_has_terminal_id else raw_teacher_ids
    teacher_text = _decode(runtime.teacher_tokenizer, teacher_content_ids)
    student_content_ids = _encode_response(runtime.student_tokenizer, teacher_text)

    max_response_tokens = _positive_int(
        getattr(args, "rollout_max_response_len"),
        name="rollout_max_response_len",
    )
    sequence_length = int(getattr(args, "seq_length", 0) or 0)
    if sequence_length > 0:
        max_response_tokens = min(
            max_response_tokens,
            sequence_length - len(student_prompt_ids),
        )
    if max_response_tokens <= 0:
        raise CrossTokenizerAlignmentError(
            "ULD Teacher answer has no room after the Student prompt"
        )

    # A terminal Student EOS is only real when the complete Teacher answer and
    # the EOS both fit.  If re-tokenization expands past the Student budget,
    # preserve the maximum amount of content and mark the target truncated.
    student_completed = teacher_completed and len(student_content_ids) + 1 <= max_response_tokens
    if len(student_content_ids) > max_response_tokens:
        student_content_ids = student_content_ids[:max_response_tokens]
        student_completed = False
    student_response_ids = student_content_ids + (
        [runtime.student_bridge_stop_id] if student_completed else []
    )
    if not student_response_ids:
        raise CrossTokenizerAlignmentError("ULD Teacher answer is empty after Student tokenization")

    # Decode after any Student-side truncation.  The generic causal-hidden path
    # re-tokenizes this canonical text for the Teacher, guaranteeing that both
    # distributions describe the exact text used as the Student CE target.
    canonical_text = _decode(runtime.student_tokenizer, student_content_ids)
    target = copy.copy(sample)
    target.tokens = student_prompt_ids + student_response_ids
    target.response_length = len(student_response_ids)
    target.response = canonical_text
    target.loss_mask = [1] * len(student_response_ids)
    target.rollout_log_probs = None
    target.rollout_routed_experts = None
    return target, canonical_text, student_completed


def _observed_log_prob(entry: Any, *, expected_token_id: int, row_index: int) -> float:
    """Parse one SGLang input-token log-probability and verify its token ID."""

    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        raise ValueError(f"Invalid SGLang input_token_logprobs row {row_index}: {entry!r}")
    log_prob, token_id = entry[0], entry[1]
    if log_prob is None or token_id is None:
        raise ValueError(f"SGLang omitted an observed log-probability at row {row_index}")
    if int(token_id) != int(expected_token_id):
        raise ValueError(
            f"SGLang observed token mismatch at row {row_index}: "
            f"expected={expected_token_id}, returned={token_id}"
        )
    value = float(log_prob)
    if not math.isfinite(value):
        raise ValueError(f"SGLang returned a non-finite observed log-probability at row {row_index}")
    return value


def _selected_log_prob_rows(
    rows: Sequence[Any],
    *,
    expected_token_ids: torch.Tensor,
) -> torch.Tensor:
    expected = expected_token_ids.detach().cpu().to(dtype=torch.long).tolist()
    values: list[list[float]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(expected):
            raise ValueError(
                f"SGLang returned {0 if not isinstance(row, list) else len(row)} selected scores "
                f"for row {row_index}; expected {len(expected)}"
            )
        parsed = [_parse_top_entry(entry) for entry in row]
        returned_ids = [token_id for token_id, _ in parsed]
        if returned_ids != expected:
            raise ValueError(f"SGLang selected-token order mismatch at row {row_index}")
        values.append([log_prob for _, log_prob in parsed])
    return torch.tensor(values, dtype=torch.float32)


_PAYLOAD_DTYPES_BY_ALGORITHM = {
    "bpm": {
        "target_ids": torch.long,
        "target_probs": torch.float32,
        "target_mask": torch.bool,
        "target_complement": torch.float32,
        "student_positions": torch.long,
        "row_kinds": torch.long,
    },
    "simct_topk": {
        "candidate_ids": torch.long,
        "teacher_scores": torch.float32,
        "candidate_mask": torch.bool,
        "student_starts": torch.long,
        "student_ends": torch.long,
    },
    "simct": {
        "candidate_ids": torch.long,
        "teacher_scores": torch.float32,
        "candidate_mask": torch.bool,
        "student_starts": torch.long,
        "student_ends": torch.long,
    },
}


def _empty_cross_payload(algorithm_name: str) -> dict[str, list[Any]]:
    try:
        fields = _PAYLOAD_DTYPES_BY_ALGORITHM[algorithm_name]
    except KeyError as error:
        raise ValueError(f"External Teacher has no payload schema for {algorithm_name!r}") from error
    return {field: [] for field in fields}


def _empty_bpm_payload() -> dict[str, list[Any]]:
    """Backward-compatible BPM empty payload helper."""

    return _empty_cross_payload("bpm")


def _build_external_result(
    *,
    runtime: _TeacherRuntime,
    context: TeacherScoringContext,
    task_reward: float,
    teacher_url: str,
    student_response_length: int,
    teacher_response_length: int,
    algorithm_name: str,
    reward_fields: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Serialize one external cross-tokenizer target."""

    try:
        payload = runtime.algorithm.build_teacher_payload(context)
    except BPMZeroCompatibleTeacherMassError as error:
        if algorithm_name != "bpm":
            raise
        return {
            **(reward_fields or {"task_reward": task_reward}),
            "teacher_url": teacher_url.rsplit("/generate", 1)[0],
            "teacher_cross_payload": _empty_cross_payload(algorithm_name),
            "teacher_cross_skipped_mask": [True] * student_response_length,
            "student_response_length_with_eos": student_response_length,
            "teacher_response_length_with_eos": teacher_response_length,
            "bpm_target_skipped": True,
            "bpm_target_error": str(error),
        }
    return {
        **(reward_fields or {"task_reward": task_reward}),
        "teacher_url": teacher_url.rsplit("/generate", 1)[0],
        "teacher_cross_payload": {key: value.detach().cpu().tolist() for key, value in payload.items()},
        "teacher_cross_skipped_mask": [False] * student_response_length,
        "student_response_length_with_eos": student_response_length,
        "teacher_response_length_with_eos": teacher_response_length,
        "cross_tokenizer_algorithm": algorithm_name,
    }


def _build_external_bpm_result(
    *,
    runtime: _TeacherRuntime,
    context: TeacherScoringContext,
    task_reward: float,
    teacher_url: str,
    student_response_length: int,
    teacher_response_length: int,
) -> dict[str, Any]:
    return _build_external_result(
        runtime=runtime,
        context=context,
        task_reward=task_reward,
        teacher_url=teacher_url,
        student_response_length=student_response_length,
        teacher_response_length=teacher_response_length,
        algorithm_name="bpm",
    )


async def external_cross_tokenizer_reward(args: Any, sample: Any, **_: Any) -> dict[str, Any]:
    """Return task reward plus a sparse cross-tokenizer payload."""

    task_reward = _task_reward(sample)
    reward_fields = _reward_fields(sample, task_reward=task_reward)
    metadata = getattr(sample, "metadata", None) or {}
    if bool(metadata.get("opd_skip_teacher", False)):
        return {**reward_fields, "evaluation_only": True}
    algorithm_name = str(args.cross_tokenizer_algorithm).lower()
    if algorithm_name not in _PAYLOAD_DTYPES_BY_ALGORITHM:
        raise ValueError(
            "The external Teacher supports bpm, simct_topk, and dense selected-token simct"
        )

    runtime = _runtime(args)
    try:
        teacher_prompt_ids, teacher_response_ids, student_response_ids, loss_mask, alignment = _normalized_sequences(
            args, sample, runtime
        )
    except CrossTokenizerAlignmentError as error:
        policy = str(getattr(args, "cross_tokenizer_alignment_error_policy", "error")).lower()
        if policy != "skip":
            raise
        _, student_response_ids, _ = _normalized_student_sequences(sample, runtime)
        return {
            **reward_fields,
            "teacher_cross_payload": _empty_cross_payload(algorithm_name),
            "teacher_cross_skipped_mask": [True] * len(student_response_ids),
            "student_response_length_with_eos": len(student_response_ids),
            "teacher_response_length_with_eos": 0,
            "alignment_skipped": True,
            "alignment_error": str(error),
        }
    candidate_top_k = int(args.cross_tokenizer_top_k)
    if algorithm_name == "simct":
        if candidate_top_k != 0:
            raise ValueError("Paper SimCT requires cross_tokenizer_top_k=0")
    else:
        candidate_top_k = _positive_int(candidate_top_k, name="cross_tokenizer_top_k")
    teacher_input_ids = teacher_prompt_ids + teacher_response_ids
    is_server_bpm = algorithm_name == "bpm" and _teacher_endpoint_suffix() == "opd/score"
    if is_server_bpm:
        rows = list(alignment.algorithm_metadata.get("bpm_rows", ()))
        if not rows:
            raise CrossTokenizerAlignmentError("Kimi BPM server request contains no aligned Student rows")
        prompt_offset = len(teacher_prompt_ids)
        server_rows = []
        for row in rows:
            server_row = dict(row)
            server_row["teacher_target_position"] = (
                int(server_row["teacher_target_position"]) + prompt_offset
            )
            server_rows.append(server_row)
        request = {
            "schema_version": 2,
            "routing_sha256": os.environ.get("OPD_BPM_ROUTING_SHA256", ""),
            "input_ids": teacher_input_ids,
            "assistant_target_ranges": [[len(teacher_prompt_ids), len(teacher_input_ids)]],
            "bpm_rows": server_rows,
            "top_k": candidate_top_k,
        }
    else:
        request = {
            "input_ids": teacher_input_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
            "return_logprob": True,
            "logprob_start_len": 0,
            "return_text_in_logprobs": False,
        }
    if algorithm_name == "simct" and not is_server_bpm:
        request["top_logprobs_num"] = 0
        request["token_ids_logprob"] = runtime.vocabulary_mapping.teacher_ids.tolist()
    elif not is_server_bpm:
        request["top_logprobs_num"] = candidate_top_k
    teacher_url = _select_teacher_url(sample)
    timeout = aiohttp.ClientTimeout(total=float(os.environ.get("OPD_TEACHER_TIMEOUT_SECONDS", "1800")))
    retries = int(os.environ.get("OPD_TEACHER_RETRIES", "3"))
    concurrency = int(os.environ.get("OPD_TEACHER_CONCURRENCY_PER_ENDPOINT", "4"))
    semaphore = _SEMAPHORES.setdefault(teacher_url, asyncio.Semaphore(concurrency))
    result: dict[str, Any] | None = None
    last_error: Exception | None = None
    async with semaphore:
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    if not is_server_bpm:
                        await _verify_dense_teacher_model(
                            session,
                            teacher_url=teacher_url,
                            expected_model_path=str(args.opd_teacher_model_path),
                        )
                    async with session.post(teacher_url, json=request) as response:
                        response.raise_for_status()
                        result = await response.json()
                break
            except Exception as error:
                last_error = error
                if attempt + 1 < retries:
                    await asyncio.sleep(min(2**attempt, 8) + random.random())
    if result is None:
        raise RuntimeError(f"External Teacher failed after {retries} attempts: {last_error}") from last_error

    if is_server_bpm:
        return _build_external_bpm_server_result(
            runtime=runtime,
            result=result,
            alignment=alignment,
            task_reward=task_reward,
            teacher_url=teacher_url,
            student_response_length=len(student_response_ids),
            teacher_response_length=len(teacher_response_ids),
            candidate_top_k=candidate_top_k,
            reward_fields=reward_fields,
        )

    observed_rows = result.get("meta_info", {}).get("input_token_logprobs")
    if not isinstance(observed_rows, list) or len(observed_rows) != len(teacher_input_ids):
        raise ValueError("External Teacher returned an invalid input_token_logprobs array")
    response_observed_rows = observed_rows[-len(teacher_response_ids) :]
    observed_log_probs = torch.tensor(
        [
            _observed_log_prob(entry, expected_token_id=token_id, row_index=row_index)
            for row_index, (entry, token_id) in enumerate(
                zip(response_observed_rows, teacher_response_ids, strict=True)
            )
        ],
        dtype=torch.float32,
    )
    if algorithm_name == "simct":
        selected_rows = result.get("meta_info", {}).get("input_token_ids_logprobs")
        if not isinstance(selected_rows, list) or len(selected_rows) != len(teacher_input_ids):
            raise ValueError("External Teacher returned an invalid input_token_ids_logprobs array")
        response_selected_rows = selected_rows[-len(teacher_response_ids) :]
        distribution = _SelectedTokenDistribution(
            runtime.vocabulary_mapping.teacher_ids,
            _selected_log_prob_rows(
                response_selected_rows,
                expected_token_ids=runtime.vocabulary_mapping.teacher_ids,
            ),
            torch.tensor(teacher_response_ids, dtype=torch.long),
            observed_log_probs,
        )
    else:
        top_rows = result.get("meta_info", {}).get("input_top_logprobs")
        if not isinstance(top_rows, list) or len(top_rows) != len(teacher_input_ids):
            raise ValueError("External Teacher returned an invalid input_top_logprobs array")
        response_top_rows = top_rows[-len(teacher_response_ids) :]
        forced_tokens: list[tuple[int, float] | None] = [None] * len(teacher_response_ids)
        forced_rows = (
            range(len(teacher_response_ids))
            if algorithm_name == "simct_topk"
            else [len(teacher_response_ids) - 1]
        )
        # BPM needs the exact terminal stop mass. SimCT additionally needs every
        # observed continuation score to construct its virtual span candidate,
        # even when that continuation falls below the requested source Top-K.
        for row_index in forced_rows:
            forced_tokens[row_index] = (
                int(teacher_response_ids[row_index]),
                float(observed_log_probs[row_index]),
            )
        distribution = _SparseTopKDistribution.from_sglang(
            response_top_rows,
            expected_k=candidate_top_k,
            forced_tokens=forced_tokens,
        )
    return _build_external_result(
        runtime=runtime,
        context=TeacherScoringContext(
            distribution=distribution,
            response_token_ids=torch.tensor(teacher_response_ids, dtype=torch.long),
            alignment=alignment,
            vocabulary_mapping=runtime.vocabulary_mapping,
            candidate_top_k=candidate_top_k,
            student_loss_mask=torch.tensor(loss_mask, dtype=torch.bool),
            temperature=float(args.cross_tokenizer_temperature),
            student_response_token_ids=torch.tensor(
                student_response_ids, dtype=torch.long
            ),
        ),
        task_reward=task_reward,
        teacher_url=teacher_url,
        student_response_length=len(student_response_ids),
        teacher_response_length=len(teacher_response_ids),
        algorithm_name=algorithm_name,
        reward_fields=reward_fields,
    )


async def _generate_uld_teacher_answer(
    args: Any,
    sample: Any,
    runtime: _TeacherRuntime,
) -> dict[str, Any]:
    """Greedily generate the fixed answer corpus required by ULD."""

    if _teacher_endpoint_suffix() != "generate":
        raise ValueError("ULD Teacher-answer supervision requires the SGLang /generate endpoint")
    teacher_prompt_text = _teacher_prompt_text(args, sample, runtime)
    teacher_prompt_ids = _encode_response(runtime.teacher_tokenizer, teacher_prompt_text)
    if not teacher_prompt_ids:
        raise ValueError("ULD Teacher chat template produced an empty prompt")

    algorithm_kwargs = dict(getattr(args, "cross_tokenizer_algorithm_kwargs", {}) or {})
    max_new_tokens = _positive_int(
        algorithm_kwargs.get(
            "teacher_generation_max_new_tokens",
            getattr(args, "rollout_max_response_len"),
        ),
        name="teacher_generation_max_new_tokens",
    )
    max_teacher_input_tokens = int(os.environ.get("OPD_TEACHER_MAX_INPUT_TOKENS", "0") or "0")
    if max_teacher_input_tokens > 0:
        max_new_tokens = min(max_new_tokens, max_teacher_input_tokens - len(teacher_prompt_ids))
    if max_new_tokens <= 0:
        raise CrossTokenizerAlignmentError(
            "ULD Teacher prompt leaves no generation room in OPD_TEACHER_MAX_INPUT_TOKENS"
        )

    request = {
        "input_ids": teacher_prompt_ids,
        # The released ULD answer generator uses greedy decoding.  Sampling a
        # Student trajectory here would recreate the incorrect online target.
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            # Do not rely on generation_config.json being present in an HF
            # export (the evaluated Qwen3.5 checkpoint did not carry one).
            "stop_token_ids": list(runtime.teacher_stop_ids),
            "skip_special_tokens": False,
        },
        # The generation result is only used for output token IDs and the
        # finish reason.  ULD performs a separate causal-hidden forward pass
        # below to obtain the Teacher log-probs/hidden rows, so asking SGLang
        # to materialize per-token logprobs here just adds serialization and
        # memory overhead for a 27k-token response.
        "return_logprob": False,
        "return_text_in_logprobs": False,
    }
    teacher_url = _select_teacher_url(sample)
    timeout = aiohttp.ClientTimeout(
        total=float(os.environ.get("OPD_TEACHER_TIMEOUT_SECONDS", "1800"))
    )
    retries = _positive_int(
        os.environ.get("OPD_TEACHER_RETRIES", "3"), name="OPD_TEACHER_RETRIES"
    )
    concurrency = _positive_int(
        os.environ.get("OPD_TEACHER_CONCURRENCY_PER_ENDPOINT", "4"),
        name="OPD_TEACHER_CONCURRENCY_PER_ENDPOINT",
    )
    semaphore = _SEMAPHORES.setdefault(teacher_url, asyncio.Semaphore(concurrency))
    result: dict[str, Any] | None = None
    last_error: Exception | None = None
    async with semaphore:
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    await _verify_dense_teacher_model(
                        session,
                        teacher_url=teacher_url,
                        expected_model_path=str(args.opd_teacher_model_path),
                    )
                    async with session.post(teacher_url, json=request) as response:
                        response.raise_for_status()
                        result = await response.json()
                break
            except Exception as error:
                last_error = error
                if attempt + 1 < retries:
                    await asyncio.sleep(min(2**attempt, 8) + random.random())
    if result is None:
        raise RuntimeError(
            f"ULD Teacher generation failed after {retries} attempts: {last_error}"
        ) from last_error
    return result


async def external_causal_hidden_reward(args: Any, sample: Any, **_: Any) -> dict[str, Any]:
    """Collect exact Teacher hidden rows for a causal-hidden Slime plugin.

    The external service only executes the frozen Teacher forward.  Tokenizer
    Alignment, the frozen LM-head shard and the algorithm-specific loss remain
    in this repository's Slime runtime.
    """

    algorithm_name = str(args.cross_tokenizer_algorithm).lower()
    capabilities = cross_tokenizer_capabilities(algorithm_name)
    if capabilities.teacher_transport("sglang") != "causal_hidden":
        raise ValueError(
            f"Cross-tokenizer algorithm {algorithm_name!r} does not declare causal-hidden SGLang transport"
        )
    task_reward = _task_reward(sample)
    reward_fields = _reward_fields(sample, task_reward=task_reward)
    runtime = _runtime(args)
    try:
        teacher_prompt_ids, teacher_response_ids, student_response_ids, loss_mask, alignment = _normalized_sequences(
            args, sample, runtime
        )
    except CrossTokenizerAlignmentError as error:
        policy = str(getattr(args, "cross_tokenizer_alignment_error_policy", "error")).lower()
        if policy != "skip":
            raise
        _, student_response_ids, _ = _normalized_student_sequences(sample, runtime)
        return {
            **reward_fields,
            "teacher_cross_hidden_states": [],
            "teacher_cross_response_token_ids": [],
            "teacher_cross_observed_log_probs": [],
            "teacher_cross_alignment": AlignmentResult.skipped_result(
                0, len(student_response_ids), str(error)
            ).to_dict(),
            "teacher_cross_skipped_mask": [True] * len(student_response_ids),
            "alignment_error": str(error),
        }
    if capabilities.requires_full_response_loss_mask and not all(bool(value) for value in loss_mask):
        raise ValueError(
            f"{algorithm_name} requires every realized Student response token; transport loss_mask contains zero"
        )
    binary_directory = binary_hidden_directory()
    if binary_directory is None:
        json_hidden_limit = int(
            os.environ.get(
                "OPD_CROSS_TOKENIZER_JSON_HIDDEN_MAX_TOKENS",
                os.environ.get("OPD_BPM_JSON_HIDDEN_MAX_TOKENS", "2048"),
            )
        )
        if json_hidden_limit <= 0:
            raise ValueError("The causal-hidden JSON token limit must be positive")
        if len(teacher_response_ids) > json_hidden_limit:
            raise RuntimeError(
                "The standard SGLang hidden-state endpoint serializes hidden rows as JSON and is restricted "
                f"to structural canaries ({json_hidden_limit} Teacher tokens); got {len(teacher_response_ids)}. "
                "Configure OPD_BPM_BINARY_HIDDEN_DIR on both Teacher and Student for binary transport."
            )
    teacher_input_ids = teacher_prompt_ids + teacher_response_ids
    request_base = {
        "input_ids": teacher_input_ids,
        # SGLang publishes the prefill hidden-state chunk when it executes the
        # first decode step. With max_new_tokens=0 the request can finish before
        # that chunk is emitted. Generate exactly one throw-away token and use
        # only the returned prefill rows for Teacher scoring.
        "sampling_params": {"temperature": 0, "max_new_tokens": 1, "skip_special_tokens": False},
        "return_hidden_states": True,
        "return_logprob": True,
        "logprob_start_len": 0,
        "return_text_in_logprobs": False,
    }
    teacher_url = _select_teacher_url(sample)
    timeout = aiohttp.ClientTimeout(total=float(os.environ.get("OPD_TEACHER_TIMEOUT_SECONDS", "1800")))
    retries = _positive_int(
        os.environ.get("OPD_TEACHER_RETRIES", "3"), name="OPD_TEACHER_RETRIES"
    )
    concurrency = _positive_int(
        os.environ.get("OPD_TEACHER_CONCURRENCY_PER_ENDPOINT", "1"),
        name="OPD_TEACHER_CONCURRENCY_PER_ENDPOINT",
    )
    semaphore = _SEMAPHORES.setdefault(teacher_url, asyncio.Semaphore(concurrency))
    # Treat the HTTP response and binary sidecar as one transaction.  In PP
    # mode SGLang can answer HTTP before the final stage commits the sidecar;
    # retrying only HTTP leaves the code waiting forever on an orphan UUID.
    result: dict[str, Any] | None = None
    request_rid: str | None = None
    response_hidden: torch.Tensor | None = None
    last_error: Exception | None = None
    sidecar_timeout = float(
        os.environ.get(
            "OPD_CROSS_TOKENIZER_BINARY_HIDDEN_TIMEOUT_SECONDS",
            os.environ.get("OPD_BPM_BINARY_HIDDEN_TIMEOUT_SECONDS", "60"),
        )
    )
    if binary_directory is not None and sidecar_timeout <= 0:
        raise ValueError("The causal-hidden binary sidecar timeout must be positive")
    async with semaphore:
        for attempt in range(retries):
            request = dict(request_base)
            attempt_rid: str | None = None
            if binary_directory is not None:
                attempt_rid = uuid.uuid4().hex
                request["rid"] = attempt_rid
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    await _verify_dense_teacher_model(
                        session,
                        teacher_url=teacher_url,
                        expected_model_path=str(args.opd_teacher_model_path),
                    )
                    async with session.post(teacher_url, json=request) as response:
                        response.raise_for_status()
                        result = await response.json()
                raw_hidden = result.get("meta_info", {}).get("hidden_states")
                if binary_directory is not None:
                    if attempt_rid is None:
                        raise RuntimeError("Causal-hidden binary request completed without a request ID")
                    if raw_hidden is not None:
                        raise RuntimeError(
                            "Teacher returned JSON hidden states even though binary transport was requested; "
                            "the opd_bpm_binary_hidden SGLang plugin is not active"
                        )
                    response_hidden = await asyncio.to_thread(
                        _load_binary_response_hidden,
                        binary_directory,
                        rid=attempt_rid,
                        input_length=len(teacher_input_ids),
                        prompt_length=len(teacher_prompt_ids),
                        response_length=len(teacher_response_ids),
                        timeout_seconds=sidecar_timeout,
                    )
                else:
                    if raw_hidden is None:
                        raise ValueError(
                            "Teacher omitted hidden_states; launch SGLang with --enable-return-hidden-states"
                        )
                    response_hidden = _extract_sglang_prefill_hidden(
                        raw_hidden,
                        input_length=len(teacher_input_ids),
                        prompt_length=len(teacher_prompt_ids),
                        response_length=len(teacher_response_ids),
                    )
                request_rid = attempt_rid
                break
            except Exception as error:
                last_error = error
                if binary_directory is not None and attempt_rid is not None:
                    # The request UUID uniquely identifies this attempt.  A
                    # completed-but-unread sidecar is safe to discard here.
                    sidecar_path(binary_directory, attempt_rid).unlink(missing_ok=True)
                _LOGGER.warning(
                    "Causal-hidden Teacher attempt %d/%d failed (url=%s, rid=%s): %s",
                    attempt + 1, retries, teacher_url, attempt_rid, error,
                )
                if attempt + 1 < retries:
                    await asyncio.sleep(min(2**attempt, 8) + random.random())
    if result is None or response_hidden is None:
        raise RuntimeError(f"Causal-hidden Teacher failed after {retries} attempts: {last_error}") from last_error
    if (
        runtime.teacher_hidden_size is None
        or response_hidden.size(1) != runtime.teacher_hidden_size
    ):
        raise ValueError(
            "Causal-hidden Teacher width differs from the configured checkpoint: "
            f"hidden={response_hidden.size(1)}, config={runtime.teacher_hidden_size}"
        )
    observed_rows = result.get("meta_info", {}).get("input_token_logprobs")
    if not isinstance(observed_rows, list) or len(observed_rows) != len(teacher_input_ids):
        raise ValueError("Causal-hidden Teacher returned an invalid input_token_logprobs array")
    response_observed_rows = observed_rows[-len(teacher_response_ids) :]
    observed_log_probs = [
        _observed_log_prob(entry, expected_token_id=token_id, row_index=row_index)
        for row_index, (entry, token_id) in enumerate(
            zip(response_observed_rows, teacher_response_ids, strict=True)
        )
    ]
    return {
        **reward_fields,
        "teacher_url": teacher_url.rsplit("/generate", 1)[0],
        # Keep tensors binary through Slime/Ray.  Converting this back to a
        # Python list would reintroduce the memory blow-up avoided above.
        "teacher_cross_hidden_states": response_hidden,
        "teacher_cross_response_token_ids": teacher_response_ids,
        "teacher_cross_observed_log_probs": observed_log_probs,
        "teacher_cross_alignment": alignment.to_dict(),
        "teacher_cross_skipped_mask": [False] * len(student_response_ids),
    }


async def external_uld_teacher_answer_reward(
    args: Any,
    sample: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate a Teacher answer, then score that fixed answer for ULD.

    Student rollout generation still supplies prompt scheduling in Slime, but
    its sampled continuation is deliberately discarded before training.  CE
    and the ULD rank loss therefore share the Teacher-generated corpus used by
    the paper and released implementation.
    """

    if str(args.cross_tokenizer_algorithm).lower() != "uld":
        raise ValueError("The ULD Teacher-answer reward hook requires algorithm='uld'")
    # Keep Slime's scalar metric tied to the original Student rollout.  ULD's
    # advantage path is zero, but replacing this value with the Teacher answer
    # would make rollout dashboards report Teacher quality as Student quality.
    student_task_reward = _task_reward(sample)
    student_reward_fields = _reward_fields(sample, task_reward=student_task_reward)
    runtime = _runtime(args)
    generated = await _generate_uld_teacher_answer(args, sample, runtime)
    target, response_text, completed = _uld_teacher_answer_trajectory(
        args,
        sample,
        runtime,
        teacher_output_ids=_sglang_output_token_ids(generated),
        finish_reason=_sglang_finish_reason(generated),
    )
    result = await external_causal_hidden_reward(args, target, **kwargs)
    result["uld_teacher_task_reward"] = result.get("task_reward", 0.0)
    result.update(student_reward_fields)
    result.update(
        {
            "uld_student_response_token_ids": target.tokens[-target.response_length :],
            "uld_response_text": response_text,
            "uld_response_completed": completed,
            "uld_protocol": "teacher_answer_ce_v1",
        }
    )
    return result


async def external_bpm_paper_hidden_reward(args: Any, sample: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible BPM entry point for existing experiment manifests."""

    if str(args.cross_tokenizer_algorithm).lower() != "bpm_paper":
        raise ValueError("The BPM hidden-state reward hook requires bpm_paper")
    return await external_causal_hidden_reward(args, sample, **kwargs)


def _install_external_payload(
    args: Any,
    samples: list[Any],
    *,
    algorithm_name: str,
):
    """Move external payloads onto Samples and exclude fully skipped samples."""

    del args
    try:
        payload_dtypes = _PAYLOAD_DTYPES_BY_ALGORITHM[algorithm_name]
    except KeyError as error:
        raise ValueError(f"Unsupported external payload algorithm: {algorithm_name}") from error
    raw_rewards: list[float] = []
    training_rewards: list[float] = []
    for sample in samples:
        result = getattr(sample, "reward", None)
        if not isinstance(result, dict):
            raise ValueError("External cross-tokenizer reward result must be a dictionary")
        task_reward = float(result.get("task_reward", 0.0))
        reward_math = float(result.get("reward_math", task_reward))
        reward_code = float(result.get("reward_code", 0.0))
        aggregate_reward = float(result.get("reward", reward_math + reward_code))
        raw_payload = result.get("teacher_cross_payload")
        skipped = result.get("teacher_cross_skipped_mask")
        if raw_payload is None or skipped is None:
            raise ValueError("External Teacher result is missing its training payload")
        missing = set(payload_dtypes).difference(raw_payload)
        if missing:
            raise ValueError(f"External {algorithm_name} payload is missing fields: {sorted(missing)}")
        sample.teacher_cross_payload = {
            key: torch.tensor(raw_payload[key], dtype=dtype) for key, dtype in payload_dtypes.items()
        }
        sample.teacher_cross_skipped_mask = torch.tensor(skipped, dtype=torch.bool)
        expected_length = int(result.get("student_response_length_with_eos", -1))
        if sample.teacher_cross_skipped_mask.numel() != expected_length:
            raise ValueError("External skipped mask length differs from the normalized Student response")
        if getattr(sample, "metadata", None) is None:
            sample.metadata = {}
        sample.metadata["raw_reward"] = task_reward
        sample.metadata["reward_math"] = reward_math
        sample.metadata["reward_code"] = reward_code
        sample.metadata["reward"] = aggregate_reward
        sample.metadata["opd_teacher_url"] = result.get("teacher_url")
        sample.metadata["opd_teacher_response_length"] = int(
            result.get("teacher_response_length_with_eos", 0)
        )
        sample.metadata["opd_alignment_skipped"] = bool(result.get("alignment_skipped", False))
        if result.get("alignment_error"):
            sample.metadata["opd_alignment_error"] = str(result["alignment_error"])
        sample.metadata["opd_bpm_target_skipped"] = bool(result.get("bpm_target_skipped", False))
        if result.get("bpm_target_error"):
            sample.metadata["opd_bpm_target_error"] = str(result["bpm_target_error"])
        fully_skipped = bool(sample.teacher_cross_skipped_mask.numel()) and bool(
            sample.teacher_cross_skipped_mask.all()
        )
        if fully_skipped:
            # A cross-tokenizer alignment skip has no valid distribution
            # target.  Leaving the scalar task reward and Student response
            # mask active would silently turn the sample into ordinary policy
            # gradient training, so bad/overlong trajectories could reinforce
            # themselves even though ESCD was skipped.  ``remove_sample`` is
            # consumed by Slime's rollout converter and zeros the complete
            # response loss mask.  Keep the raw reward only for pass-rate and
            # rollout diagnostics.
            sample.remove_sample = True
            sample.metadata["opd_training_excluded"] = True
            sample.metadata["opd_training_exclusion_reason"] = (
                "cross_tokenizer_target_fully_skipped"
            )
        # Preserve Slime's historical scalar ``Sample.reward`` contract;
        # ``reward_math``/``reward_code``/``reward`` live in the serialized
        # external result and sample metadata for inspection and metrics.
        sample.reward = task_reward
        raw_rewards.append(task_reward)
        training_rewards.append(0.0 if fully_skipped else task_reward)
    return raw_rewards, training_rewards


def install_external_cross_tokenizer_payload(args: Any, samples: list[Any], **_: Any):
    algorithm_name = str(args.cross_tokenizer_algorithm).lower()
    return _install_external_payload(args, samples, algorithm_name=algorithm_name)


def install_external_causal_hidden_payload(args: Any, samples: list[Any], **_: Any):
    algorithm_name = str(args.cross_tokenizer_algorithm).lower()
    if cross_tokenizer_capabilities(algorithm_name).teacher_transport("sglang") != "causal_hidden":
        raise ValueError(f"Cross-tokenizer algorithm {algorithm_name!r} is not a causal-hidden plugin")
    raw_rewards: list[float] = []
    training_rewards: list[float] = []
    for sample in samples:
        result = getattr(sample, "reward", None)
        if not isinstance(result, dict):
            raise ValueError("Causal-hidden Teacher result must be a dictionary")
        task_reward = float(result.get("task_reward", 0.0))
        reward_math = float(result.get("reward_math", task_reward))
        reward_code = float(result.get("reward_code", 0.0))
        aggregate_reward = float(result.get("reward", reward_math + reward_code))
        sample.teacher_cross_hidden_states = torch.as_tensor(
            result["teacher_cross_hidden_states"], dtype=torch.bfloat16
        )
        sample.teacher_cross_response_token_ids = torch.tensor(
            result["teacher_cross_response_token_ids"], dtype=torch.long
        )
        sample.teacher_cross_observed_log_probs = torch.tensor(
            result["teacher_cross_observed_log_probs"], dtype=torch.float32
        )
        sample.teacher_cross_alignment = result["teacher_cross_alignment"]
        sample.teacher_cross_skipped_mask = torch.tensor(
            result["teacher_cross_skipped_mask"], dtype=torch.bool
        )
        alignment = AlignmentResult.from_dict(sample.teacher_cross_alignment)
        if alignment.student_length != sample.teacher_cross_skipped_mask.numel():
            raise ValueError("Causal-hidden alignment differs from the Student skipped-mask length")
        skipped = bool(sample.teacher_cross_skipped_mask.all())
        if skipped:
            if (
                sample.teacher_cross_hidden_states.numel()
                or sample.teacher_cross_response_token_ids.numel()
                or sample.teacher_cross_observed_log_probs.numel()
            ):
                raise ValueError("A skipped causal-hidden sample must not carry Teacher tensors")
        elif (
            sample.teacher_cross_hidden_states.ndim != 2
            or sample.teacher_cross_response_token_ids.ndim != 1
            or sample.teacher_cross_observed_log_probs.ndim != 1
            or sample.teacher_cross_hidden_states.size(0) != alignment.teacher_length
            or sample.teacher_cross_response_token_ids.numel() != alignment.teacher_length
            or sample.teacher_cross_observed_log_probs.numel() != alignment.teacher_length
        ):
            raise ValueError("Causal-hidden Teacher tensors differ from the alignment length")
        if sample.teacher_cross_skipped_mask.numel() != int(sample.response_length):
            raise ValueError("Causal-hidden skipped mask differs from the realized Student response")
        sample.reward = task_reward
        if getattr(sample, "metadata", None) is None:
            sample.metadata = {}
        sample.metadata["raw_reward"] = task_reward
        sample.metadata["reward_math"] = reward_math
        sample.metadata["reward_code"] = reward_code
        sample.metadata["reward"] = aggregate_reward
        sample.metadata["opd_teacher_url"] = result.get("teacher_url")
        if result.get("alignment_error"):
            sample.metadata["opd_alignment_error"] = str(result["alignment_error"])
        if skipped:
            # See the sparse-payload path above.  A causal-hidden alignment
            # skip must remove both the PG and direct ESCD contributions; a
            # zero Teacher tensor alone only removes the latter.
            sample.remove_sample = True
            sample.metadata["opd_training_excluded"] = True
            sample.metadata["opd_training_exclusion_reason"] = (
                "cross_tokenizer_target_fully_skipped"
            )
        raw_rewards.append(task_reward)
        training_rewards.append(0.0 if skipped else task_reward)
    return raw_rewards, training_rewards


def install_external_uld_teacher_answer_payload(
    args: Any,
    samples: list[Any],
    **kwargs: Any,
):
    """Replace Student rollouts with Teacher-generated ULD CE targets."""

    if str(args.cross_tokenizer_algorithm).lower() != "uld":
        raise ValueError("The ULD Teacher-answer payload installer requires algorithm='uld'")
    for sample in samples:
        result = getattr(sample, "reward", None)
        if not isinstance(result, dict):
            raise ValueError("ULD Teacher result must be a dictionary")
        if result.get("uld_protocol") != "teacher_answer_ce_v1":
            raise ValueError("ULD Teacher result is missing the Teacher-answer protocol marker")
        response_ids = result.get("uld_student_response_token_ids")
        if not isinstance(response_ids, list) or not response_ids:
            raise ValueError("ULD Teacher result is missing its Student-tokenized answer")
        response_ids = [int(value) for value in response_ids]
        old_response_length = int(getattr(sample, "response_length", 0) or 0)
        old_tokens = [int(value) for value in (getattr(sample, "tokens", None) or [])]
        if old_response_length <= 0 or old_response_length > len(old_tokens):
            raise ValueError("ULD cannot replace an invalid Student rollout trajectory")
        prompt_ids = old_tokens[:-old_response_length]
        sample.tokens = prompt_ids + response_ids
        sample.response_length = len(response_ids)
        sample.response = str(result.get("uld_response_text", ""))
        sample.loss_mask = [1] * len(response_ids)
        sample.rollout_log_probs = None
        sample.rollout_routed_experts = None
        status_type = getattr(sample, "Status", None)
        if status_type is not None:
            sample.status = (
                status_type.COMPLETED
                if bool(result.get("uld_response_completed"))
                else status_type.TRUNCATED
            )
        if getattr(sample, "metadata", None) is None:
            sample.metadata = {}
        sample.metadata["opd_uld_protocol"] = "teacher_answer_ce_v1"
        sample.metadata["opd_uld_discarded_student_response_length"] = old_response_length
        sample.metadata["opd_uld_teacher_answer_completed"] = bool(
            result.get("uld_response_completed")
        )
        if "uld_teacher_task_reward" in result:
            sample.metadata["opd_uld_teacher_task_reward"] = float(
                result["uld_teacher_task_reward"]
            )
    return install_external_causal_hidden_payload(args, samples, **kwargs)


def install_external_bpm_paper_hidden_payload(args: Any, samples: list[Any], **kwargs: Any):
    """Backward-compatible BPM entry point for existing experiment manifests."""

    if str(args.cross_tokenizer_algorithm).lower() != "bpm_paper":
        raise ValueError("The BPM hidden payload installer requires bpm_paper")
    return install_external_causal_hidden_payload(args, samples, **kwargs)


async def external_bpm_reward(args: Any, sample: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible name used by existing BPM configs."""

    return await external_cross_tokenizer_reward(args, sample, **kwargs)


def install_external_bpm_payload(args: Any, samples: list[Any], **_: Any):
    """Backward-compatible name used by existing BPM configs and tests."""

    return _install_external_payload(args, samples, algorithm_name="bpm")
