"""Binary hidden-state sidecar for the external BPM Teacher.

SGLang's standard ``return_hidden_states`` response converts every hidden row
to nested JSON lists.  That is useful for short diagnostics but is not viable
for the paper's 27,648-token response cap.  This module is an out-of-tree
SGLang plugin: it captures the already-computed CPU hidden tensor at the
Scheduler boundary, accumulates chunked-prefill rows, and atomically writes a
``torch.save`` sidecar on the shared filesystem.  The ordinary HTTP response
still carries the much smaller observed-token log-probabilities.

The plugin is deliberately transport-only.  It does not implement BPM,
tokenizer alignment, the Teacher LM head, or a training loop; those remain in
the repository's Slime runtime.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable

import torch


FORMAT_VERSION = 1
PLUGIN_NAME = "opd_bpm_binary_hidden"
_RID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_CHUNKS_ATTR = "_opd_bpm_binary_hidden_chunks"
_ROWS_ATTR = "_opd_bpm_binary_hidden_rows"
_PP_PATCHED_ATTR = "_opd_bpm_binary_hidden_pp_patched"


def binary_hidden_directory() -> Path | None:
    """Return the configured sidecar directory, or ``None`` when disabled."""

    value = os.environ.get("OPD_BPM_BINARY_HIDDEN_DIR", "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve(strict=False)


def sidecar_path(directory: Path, rid: str) -> Path:
    """Resolve a request sidecar without allowing request-id path traversal."""

    if not _RID_PATTERN.fullmatch(str(rid)):
        raise ValueError(
            "BPM binary hidden transport requires a 32-character lowercase hexadecimal rid"
        )
    path = directory / f"{rid}.pt"
    if path.parent != directory:
        raise ValueError("BPM hidden sidecar escaped its configured directory")
    return path


def _write_sidecar(req: Any, directory: Path) -> Path:
    chunks = getattr(req, _CHUNKS_ATTR, None)
    rows = int(getattr(req, _ROWS_ATTR, 0))
    if not isinstance(chunks, list) or not chunks:
        raise RuntimeError(
            f"BPM Teacher request {req.rid} has no captured hidden chunks"
        )
    expected_rows = len(req.origin_input_ids)
    if rows != expected_rows:
        raise RuntimeError(
            "BPM Teacher hidden chunks do not cover the full uncached input: "
            f"rid={req.rid}, captured={rows}, expected={expected_rows}. "
            "Launch the dedicated Teacher with --disable-radix-cache."
        )
    hidden_size = int(chunks[0].size(1))
    if any(
        not isinstance(chunk, torch.Tensor)
        or chunk.ndim != 2
        or chunk.size(1) != hidden_size
        or chunk.device.type != "cpu"
        for chunk in chunks
    ):
        raise RuntimeError(
            f"BPM Teacher request {req.rid} has inconsistent hidden chunks"
        )

    directory.mkdir(parents=True, exist_ok=True)
    destination = sidecar_path(directory, str(req.rid))
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite BPM hidden sidecar: {destination}"
        )
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    payload = {
        "format_version": FORMAT_VERSION,
        "rid": str(req.rid),
        "input_length": expected_rows,
        "hidden_size": hidden_size,
        "chunks": chunks,
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        # A failed write must not be mistaken for a complete response.  This
        # only targets the uniquely named temporary file created above.
        temporary.unlink(missing_ok=True)
    delattr(req, _CHUNKS_ATTR)
    delattr(req, _ROWS_ATTR)
    return destination


def capture_prefill_hidden_sidecar(
    original: Callable[..., Any],
    scheduler: Any,
    batch: Any,
    result: Any,
) -> Any:
    """SGLang AROUND hook for ``process_batch_result_prefill``.

    ``logits_output.hidden_states`` is flattened across the requests in the
    prefill batch.  ``extend_input_len_per_req`` is the authoritative row
    partition, including intermediate chunked-prefill calls.  Every shape is
    checked before the stock output processor is allowed to continue.
    """

    directory = binary_hidden_directory()
    if directory is None:
        return original(scheduler, batch, result)

    requested = [
        bool(getattr(req, "return_hidden_states", False)) for req in batch.reqs
    ]
    if not any(requested):
        return original(scheduler, batch, result)

    def call_without_json_hidden() -> Any:
        """Run SGLang's output processor without serializing hidden rows.

        In PP mode the final-stage hidden rows are captured by
        ``_pp_prepare_tensor_dict`` before the proxy is sent to PP0.  The
        output processor can still run on PP0 (and, depending on the
        scheduler version, on the final stage) after that capture.  Leaving
        ``return_hidden_states`` set in this path makes SGLang serialize the
        full hidden tensor into JSON, defeating the binary sidecar and
        causing the student to reject an otherwise valid response.
        """
        for req, enabled in zip(batch.reqs, requested, strict=True):
            if enabled:
                req.return_hidden_states = False
        try:
            return original(scheduler, batch, result)
        finally:
            for req, enabled in zip(batch.reqs, requested, strict=True):
                if enabled:
                    req.return_hidden_states = True

    # With pipeline parallelism every PP stage enters the output processor,
    # but only the last stage owns the final logits/hidden states.  Earlier
    # stages legitimately expose ``hidden_states=None``.  Let the stock
    # processor handle those stages and only capture the user-facing output
    # on the last PP rank.  Without this guard a PP0 rank raises below and
    # tears down the whole multi-node teacher even though the final rank is
    # healthy.
    pp_group = getattr(scheduler, "pp_group", None)
    if pp_group is not None and not bool(
        getattr(pp_group, "is_last_rank", True)
    ):
        return call_without_json_hidden()
    # In PP mode the final-stage tensor is converted to a compact proxy and
    # sent to PP0 before this hook runs; ``get_logprob_from_pp_outputs``
    # intentionally drops hidden_states from that proxy.  The PP wrapper
    # installed below captures the tensor before it is serialized, so this
    # output hook must not treat the missing field as a fatal error.
    if pp_group is not None and int(getattr(scheduler, "pp_size", 1)) > 1:
        return call_without_json_hidden()
    # Tensor-parallel Scheduler ranks all execute output processing, while only
    # attention TP rank zero owns the user-facing stream.  Capture/write once;
    # non-writer ranks still suppress their otherwise wasted list conversion.
    if int(getattr(scheduler, "attn_tp_rank", 0)) != 0:
        for req, enabled in zip(batch.reqs, requested, strict=True):
            if enabled:
                req.return_hidden_states = False
        try:
            return original(scheduler, batch, result)
        finally:
            for req, enabled in zip(batch.reqs, requested, strict=True):
                if enabled:
                    req.return_hidden_states = True
    if result.copy_done is not None:
        result.copy_done.synchronize()
    logits_output = result.logits_output
    hidden = None if logits_output is None else logits_output.hidden_states
    lengths = result.extend_input_len_per_req
    if hidden is None:
        # In PP mode the final-stage hidden rows are captured by
        # ``_pp_prepare_tensor_dict`` before the proxy is sent to PP0.  PP0
        # then runs this stock output hook once more with a proxy that has no
        # hidden_states field.  Treat that second invocation as transport
        # bookkeeping instead of tearing down the whole teacher.  The final
        # stage remains strict below, so a genuinely missing sidecar still
        # surfaces as an error in the caller.
        pp_group = getattr(scheduler, "pp_group", None)
        if pp_group is not None and int(getattr(scheduler, "pp_size", 1)) > 1:
            return call_without_json_hidden()
        raise RuntimeError("SGLang omitted hidden_states for a BPM binary request")
    if lengths is None or len(lengths) != len(batch.reqs):
        raise RuntimeError(
            "SGLang omitted extend_input_len_per_req; BPM binary requests must set return_logprob"
        )
    if hidden.ndim != 2:
        raise RuntimeError(
            f"Expected flattened [tokens, hidden] states, got {tuple(hidden.shape)}"
        )
    normalized_lengths = [int(value) for value in lengths]
    if any(value < 0 for value in normalized_lengths):
        raise RuntimeError("SGLang returned a negative prefill row count")
    expected_batch_rows = sum(normalized_lengths)
    if hidden.size(0) != expected_batch_rows:
        raise RuntimeError(
            "SGLang hidden rows differ from the prefill partitions: "
            f"hidden={hidden.size(0)}, partitions={normalized_lengths}"
        )

    offset = 0
    for req, enabled, length in zip(
        batch.reqs, requested, normalized_lengths, strict=True
    ):
        next_offset = offset + length
        if enabled:
            if getattr(req, "is_retracted", False):
                raise RuntimeError(
                    f"BPM Teacher request {req.rid} was retracted during prefill"
                )
            # The async D2H copy was synchronized above.  Clone each view so
            # retaining one chunk cannot retain the full flattened batch.
            chunk = (
                hidden[offset:next_offset]
                .detach()
                .to(dtype=torch.bfloat16)
                .clone()
                .contiguous()
            )
            if chunk.device.type != "cpu":
                chunk = chunk.cpu()
            chunks = getattr(req, _CHUNKS_ATTR, None)
            if chunks is None:
                chunks = []
                setattr(req, _CHUNKS_ATTR, chunks)
                setattr(req, _ROWS_ATTR, 0)
            chunks.append(chunk)
            setattr(req, _ROWS_ATTR, int(getattr(req, _ROWS_ATTR)) + length)
            # Most SGLang versions set ``is_chunked <= 0`` on the final/full
            # prefill pass.  Some PP/chunked-prefill paths leave that flag
            # positive on the final chunk, however.  The captured row count
            # is authoritative: once it covers the complete uncached input,
            # commit the sidecar regardless of the advisory flag.  Retain
            # the old branch for an early ``is_chunked <= 0`` so a genuinely
            # incomplete capture still fails loudly in ``_write_sidecar``.
            captured_rows = int(getattr(req, _ROWS_ATTR, 0))
            expected_rows = len(req.origin_input_ids)
            if captured_rows == expected_rows or int(getattr(req, "is_chunked", 0)) <= 0:
                _write_sidecar(req, directory)
        offset = next_offset

    # Suppress the stock nested-list conversion and JSON field for this one
    # output-processing call.  Restore the request flags afterwards so an
    # intermediate chunk still asks the model worker for the next chunk.
    return call_without_json_hidden()


def register_sglang_plugin() -> None:
    """Register the binary transport through SGLang's official hook API."""

    from sglang.srt.plugins.hook_registry import HookType, plugin_hook

    plugin_hook(
        "sglang.srt.managers.scheduler_output_processor_mixin."
        "SchedulerOutputProcessorMixin.process_batch_result_prefill",
        type=HookType.AROUND,
    )(capture_prefill_hidden_sidecar)

    # Pipeline parallelism normally drops final hidden states while sending
    # the last-stage output back to PP0.  Capture them at that boundary so
    # the external Teacher can keep PP4 (required by the Qwen3.5 vision
    # tower) without sending a second, unbounded JSON response.
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    if getattr(SchedulerPPMixin, _PP_PATCHED_ATTR, False):
        return

    original_prepare = SchedulerPPMixin._pp_prepare_tensor_dict

    def prepare_with_binary_hidden(self, result, batch):
        directory = binary_hidden_directory()
        pp_group = getattr(self, "pp_group", None)
        is_last_pp = pp_group is not None and bool(
            getattr(pp_group, "is_last_rank", False)
        )
        is_writer_tp = int(getattr(self, "attn_tp_rank", 0)) == 0
        requested = [
            bool(getattr(req, "return_hidden_states", False))
            for req in getattr(batch, "reqs", [])
        ]
        if directory is not None and is_last_pp and is_writer_tp and any(requested):
            logits_output = getattr(result, "logits_output", None)
            hidden = None if logits_output is None else logits_output.hidden_states
            lengths = getattr(result, "extend_input_len_per_req", None)
            if hidden is None:
                raise RuntimeError(
                    "SGLang omitted hidden_states on the final PP stage for a BPM request"
                )
            if lengths is None or len(lengths) != len(batch.reqs):
                raise RuntimeError(
                    "SGLang omitted PP prefill partitions for a BPM binary request"
                )
            if hidden.ndim != 2:
                raise RuntimeError(
                    f"Expected PP hidden states [tokens, hidden], got {tuple(hidden.shape)}"
                )
            normalized_lengths = [int(value) for value in lengths]
            if any(value < 0 for value in normalized_lengths):
                raise RuntimeError("SGLang returned a negative PP prefill row count")
            if hidden.size(0) != sum(normalized_lengths):
                raise RuntimeError(
                    "PP hidden rows differ from the prefill partitions: "
                    f"hidden={hidden.size(0)}, partitions={normalized_lengths}"
                )
            offset = 0
            for req, enabled, length in zip(
                batch.reqs, requested, normalized_lengths, strict=True
            ):
                next_offset = offset + length
                if enabled:
                    if getattr(req, "is_retracted", False):
                        raise RuntimeError(
                            f"BPM Teacher request {req.rid} was retracted during PP prefill"
                        )
                    chunk = (
                        hidden[offset:next_offset]
                        .detach()
                        .to(dtype=torch.bfloat16)
                        .cpu()
                        .clone()
                        .contiguous()
                    )
                    chunks = getattr(req, _CHUNKS_ATTR, None)
                    if chunks is None:
                        chunks = []
                        setattr(req, _CHUNKS_ATTR, chunks)
                        setattr(req, _ROWS_ATTR, 0)
                    chunks.append(chunk)
                    setattr(req, _ROWS_ATTR, int(getattr(req, _ROWS_ATTR)) + length)
                    captured_rows = int(getattr(req, _ROWS_ATTR, 0))
                    expected_rows = len(req.origin_input_ids)
                    if captured_rows == expected_rows or int(getattr(req, "is_chunked", 0)) <= 0:
                        _write_sidecar(req, directory)
                offset = next_offset
        return original_prepare(self, result, batch)

    SchedulerPPMixin._pp_prepare_tensor_dict = prepare_with_binary_hidden
    setattr(SchedulerPPMixin, _PP_PATCHED_ATTR, True)
