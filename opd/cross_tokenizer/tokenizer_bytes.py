"""Algorithm-neutral lossless tokenizer-byte utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _gpt2_byte_decoder() -> dict[str, int]:
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = list(values)
    extra = 0
    for value in range(256):
        if value in values:
            continue
        values.append(value)
        unicode_values.append(256 + extra)
        extra += 1
    return {
        chr(unicode_value): byte_value
        for byte_value, unicode_value in zip(values, unicode_values, strict=True)
    }


_BYTE_DECODER = _gpt2_byte_decoder()


def _decode_one(tokenizer: Any, token_id: int) -> str:
    kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    try:
        return tokenizer.decode([int(token_id)], **kwargs)
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces")
        return tokenizer.decode([int(token_id)], **kwargs)


def token_bytes(tokenizer: Any, token_id: int) -> bytes:
    """Return the lossless byte representation of one tokenizer token.

    Tiktoken's byte API is preferred. Hugging Face ByteLevel token strings are
    decoded through the GPT-2 byte alphabet. A text fallback is allowed only
    when it does not contain U+FFFD, because that character commonly means the
    token is merely part of a multi-byte UTF-8 code point.
    """

    token_id = int(token_id)
    model = getattr(tokenizer, "model", None)
    decode_single = getattr(model, "decode_single_token_bytes", None)
    if not callable(decode_single):
        backend = getattr(tokenizer, "_tokenizer", None)
        decode_single = getattr(getattr(backend, "model", None), "decode_single_token_bytes", None)
    if callable(decode_single):
        try:
            return bytes(decode_single(token_id))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"token {token_id} cannot be decoded to bytes") from error

    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    added_ids = (
        {int(value) for value in (get_added_vocab() or {}).values()}
        if callable(get_added_vocab)
        else set()
    )
    backend = getattr(tokenizer, "backend_tokenizer", None)
    decoder = getattr(backend, "decoder", None)
    if callable(convert) and type(decoder).__name__ == "ByteLevel" and token_id not in added_ids:
        token = convert(token_id)
        if isinstance(token, (list, tuple)):
            if len(token) != 1:
                raise ValueError(f"tokenizer returned multiple strings for token {token_id}")
            token = token[0]
        try:
            return bytes(_BYTE_DECODER[character] for character in str(token))
        except KeyError as error:
            raise ValueError(f"ByteLevel token {token_id} contains an unknown byte symbol") from error

    text = _decode_one(tokenizer, token_id)
    if "\ufffd" in text:
        raise ValueError(
            f"token {token_id} decodes through U+FFFD; lossless token bytes are unavailable"
        )
    return text.encode("utf-8")


def tokenizer_byte_table(
    tokenizer: Any, *, vocab_size: int | None = None
) -> tuple[bytes | None, ...]:
    """Build an ID-indexed byte table, leaving tokenizer holes as ``None``."""

    vocab = {str(token): int(token_id) for token, token_id in tokenizer.get_vocab().items()}
    valid_ids = set(vocab.values())
    valid_ids.update(int(value) for value in (getattr(tokenizer, "all_special_ids", ()) or ()))
    if vocab_size is None:
        vocab_size = max(valid_ids, default=-1) + 1
    if vocab_size < 0:
        raise ValueError("vocab_size must be non-negative")
    table: list[bytes | None] = [None] * int(vocab_size)
    for token_id in sorted(valid_ids):
        if 0 <= token_id < vocab_size:
            table[token_id] = token_bytes(tokenizer, token_id)
    return tuple(table)


def semantic_stop_ids(tokenizer: Any) -> tuple[int, ...]:
    """Return semantic generation stops, without excluding every special token."""

    values: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, (list, tuple, set)):
            values.update(int(value) for value in eos)
        else:
            values.add(int(eos))
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert):
        for token_id in (getattr(tokenizer, "all_special_ids", ()) or ()):
            token = str(convert(int(token_id))).lower()
            if any(marker in token for marker in ("eos", "im_end", "endoftext", "end_of_text")):
                values.add(int(token_id))
    return tuple(sorted(values))


def _declared_model_eos_ids(model_path: str | Path) -> tuple[int, ...]:
    """Return configured EOS IDs in generation-precedence order."""

    ordered: list[int] = []
    root = Path(model_path)
    for name in ("generation_config.json", "config.json"):
        path = root / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        candidates = [payload.get("eos_token_id")]
        text_config = payload.get("text_config")
        if isinstance(text_config, dict):
            candidates.append(text_config.get("eos_token_id"))
        for candidate in candidates:
            if candidate is None:
                continue
            values = candidate if isinstance(candidate, (list, tuple, set)) else (candidate,)
            for value in values:
                token_id = int(value)
                if token_id >= 0 and token_id not in ordered:
                    ordered.append(token_id)
    return tuple(ordered)


def model_stop_ids(model_path: str | Path, tokenizer: Any) -> tuple[int, ...]:
    """Return the complete declared stop set used by BPM.

    BPM's paper contract takes stops from the tokenizer, the model generation
    configuration and the chat-template-visible turn terminator.  The latter
    is already covered by :func:`semantic_stop_ids` when it is registered as a
    special token.  Read the two local model configuration files as well so a
    secondary generation EOS cannot silently be treated as content.

    This function intentionally performs no Hub lookup.  Training launchers
    must point at the exact local model/tokenizer revision they are using.
    """

    values = set(semantic_stop_ids(tokenizer))
    values.update(_declared_model_eos_ids(model_path))
    return tuple(sorted(value for value in values if value >= 0))


def preferred_model_stop_id(
    model_path: str | Path,
    tokenizer: Any,
    *,
    declared_stop_ids: tuple[int, ...] | None = None,
) -> int:
    """Choose the stop used to bridge a realized terminal across tokenizers.

    Generation configuration is authoritative for actual model termination.
    Tokenizer ``eos_token_id`` is only a fallback; Kimi-K2.7, for example,
    exposes ``[EOS]`` there while generation ends turns with ``<|im_end|>``.
    """

    allowed = set(declared_stop_ids or model_stop_ids(model_path, tokenizer))
    if not allowed:
        raise ValueError("Model does not declare a semantic stop token")
    for token_id in _declared_model_eos_ids(model_path):
        if token_id in allowed:
            return token_id
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    tokenizer_candidates = (
        tokenizer_eos
        if isinstance(tokenizer_eos, (list, tuple, set))
        else (() if tokenizer_eos is None else (tokenizer_eos,))
    )
    for value in tokenizer_candidates:
        token_id = int(value)
        if token_id in allowed:
            return token_id
    return min(allowed)
