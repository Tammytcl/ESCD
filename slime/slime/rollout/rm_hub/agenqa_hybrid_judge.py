import asyncio
import json
import logging
import os
import re
from typing import Any

import aiohttp

from slime.utils.types import Sample

from .deepscaler import get_deepscaler_rule_based_reward

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_PATH = "/chat/completions"
VALID_JUDGE_LABELS = {"T", "F", "T(-)", "F(-)", "C"}
CORRECT_JUDGE_LABELS = {"T", "T(-)"}

_session: aiohttp.ClientSession | None = None
_tokenizers: dict[str, Any] = {}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer env {name}={value!r}; using default {default}")
        return default


def _get_session(timeout_s: int) -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        connector = aiohttp.TCPConnector(limit=_env_int("JUDGE_HTTP_LIMIT", 64))
        _session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    return _session


def _env_csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in os.getenv(name, default).split(",") if item.strip())


def _metadata_text(sample: Sample, args) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    values: list[str] = []
    for key in ("rm_mode", "domain", "subject", "category", "dataset", "source", "task_type"):
        value = metadata.get(key)
        if value is not None:
            values.append(str(value))
    prompt_data = getattr(args, "prompt_data", None)
    if prompt_data is not None:
        values.append(str(prompt_data))
    return " ".join(values).lower()


def _reward_mode(args, sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    explicit_mode = metadata.get("rm_mode") or metadata.get("reward_mode")
    if explicit_mode in {"rule_first", "llm_direct"}:
        return str(explicit_mode)

    text = _metadata_text(sample, args)
    llm_direct_domains = _env_csv("JUDGE_LLM_DIRECT_DOMAINS", "science,scientific,biology,chemistry")
    rule_first_domains = _env_csv("JUDGE_RULE_FIRST_DOMAINS", "math,physic,physics")

    if any(domain in text for domain in llm_direct_domains):
        return "llm_direct"
    if any(domain in text for domain in rule_first_domains):
        return "rule_first"
    return os.getenv("JUDGE_DEFAULT_MODE", "rule_first")


def _prompt_text(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    for key in ("question", "raw_question", "problem", "prompt_text"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value

    prompt = sample.prompt
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(message["content"])
        return "\n".join(parts)
    return str(prompt or "")


def _get_tokenizer(args):
    checkpoint = getattr(args, "hf_checkpoint", None)
    if not checkpoint:
        return None
    checkpoint = str(checkpoint)
    if checkpoint not in _tokenizers:
        try:
            from transformers import AutoTokenizer

            _tokenizers[checkpoint] = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        except Exception as exc:
            logger.warning(f"Failed to load tokenizer for judge truncation from {checkpoint}: {exc}")
            _tokenizers[checkpoint] = None
    return _tokenizers[checkpoint]


def _judge_model_output(args, sample: Sample) -> tuple[str, bool]:
    response = str(sample.response or "")
    max_tokens = _env_int("JUDGE_MAX_OUTPUT_TOKENS", 2048)
    if max_tokens <= 0:
        return response, False

    if sample.tokens and sample.response_length:
        response_token_count = min(int(sample.response_length), len(sample.tokens))
        response_tokens = sample.tokens[-response_token_count:]
        tail_tokens = response_tokens[-max_tokens:]
        tokenizer = _get_tokenizer(args)
        if tokenizer is not None:
            try:
                text = tokenizer.decode(tail_tokens, skip_special_tokens=False)
                if text.strip():
                    return text, len(response_tokens) > len(tail_tokens)
            except Exception as exc:
                logger.warning(f"Failed to decode judge output tail, falling back to char truncation: {exc}")

    max_chars = _env_int("JUDGE_MAX_OUTPUT_CHARS", max_tokens * 4)
    return response[-max_chars:], len(response) > max_chars


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    candidates = [text]
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if match:
        candidates.insert(0, match.group(1))
    for candidate in candidates:
        start = candidate.find("{")
        if start < 0:
            continue
        for end in range(len(candidate), start, -1):
            if candidate[end - 1] != "}":
                continue
            try:
                obj = json.loads(candidate[start:end])
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _build_judge_prompt(question: str, reference_answer: str, model_output: str) -> str:
    return (
        "You are a strict grader for math, physics, and scientific QA.\n"
        "Grade semantic correctness. Do not require exact formatting when the answer is otherwise unambiguous.\n\n"
        "Task:\n"
        "1) Extract the final answer from the model output.\n"
        "2) Compare it with the reference answer.\n"
        "3) Assign exactly one label:\n"
        "   - T: clearly correct or equivalent.\n"
        "   - F: clearly incorrect.\n"
        "   - T(-): probably correct, but the problem/reference is ambiguous or incomplete.\n"
        "   - F(-): reasoning is mostly right but final answer violates required units/constants/conventions/format.\n"
        "   - C: no usable output, timeout, refusal, or answer cannot be determined.\n\n"
        "Return ONLY a JSON object with this schema:\n"
        "{\n"
        '  "final_answer_excerpt": "<short excerpt>",\n'
        '  "answer_extracted": "<answer>",\n'
        '  "label": "T|F|T(-)|F(-)|C",\n'
        '  "analysis_zh": "<brief Chinese explanation, 1-3 sentences>"\n'
        "}\n\n"
        f"[QUESTION]\n{question}\n\n"
        f"[REFERENCE ANSWER]\n{reference_answer}\n\n"
        f"[MODEL OUTPUT]\n{model_output}\n"
    )


def _completion_token_param(model: str) -> str:
    configured = os.getenv("JUDGE_COMPLETION_TOKEN_PARAM")
    if configured:
        return configured
    return "max_completion_tokens" if model.lower().startswith("o") else "max_tokens"


async def _call_llm_judge(prompt: str) -> dict[str, Any]:
    api_base = os.getenv("JUDGE_BASE_URL") or os.getenv("JUDGE_API_BASE")
    api_key = os.getenv("JUDGE_API_KEY")
    model = os.getenv("JUDGE_MODEL")
    if not api_base or not model:
        raise RuntimeError("Set JUDGE_BASE_URL and JUDGE_MODEL for your own judge provider.")
    if not api_key:
        raise RuntimeError("Missing judge API key. Set JUDGE_API_KEY.")

    max_tokens = _env_int("JUDGE_MAX_TOKENS", 1024)
    timeout_s = _env_int("JUDGE_TIMEOUT", 120)
    retries = _env_int("JUDGE_RETRIES", 3)
    url = api_base.rstrip("/") + CHAT_COMPLETIONS_PATH
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        _completion_token_param(model): max_tokens,
    }

    last_error = None
    session = _get_session(timeout_s)
    for attempt in range(1, retries + 1):
        try:
            async with session.post(url, headers=headers, json=payload, allow_redirects=False) as resp:
                data = await resp.json(content_type=None)
                if resp.status == 200:
                    choices = data.get("choices") or []
                    message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
                    return {
                        "ok": True,
                        "text": message.get("content", "") if isinstance(message, dict) else "",
                        "http_status": resp.status,
                        "attempts": attempt,
                    }
                last_error = f"Judge API returned HTTP {resp.status}"
        except Exception as exc:
            last_error = f"Judge request failed: {type(exc).__name__}"

        if attempt < retries:
            await asyncio.sleep(min(2**attempt, 20))

    return {"ok": False, "text": "", "error": last_error, "attempts": retries}


async def _llm_judge_reward(args, sample: Sample, *, fallback_from_rule: bool) -> dict[str, Any]:
    response = str(sample.response or "")
    if not response.strip():
        return {
            "score": 0.0,
            "judge_label": "C",
            "judge_source": "llm",
            "judge_error": "empty_model_output",
            "fallback_from_rule": fallback_from_rule,
        }

    judge_model_output, judge_input_truncated = _judge_model_output(args, sample)
    prompt = _build_judge_prompt(
        question=_prompt_text(sample),
        reference_answer=str(sample.label or ""),
        model_output=judge_model_output,
    )
    raw = await _call_llm_judge(prompt)
    obj = _extract_json_object(raw.get("text") or "")
    if not obj:
        return {
            "score": 0.0,
            "judge_label": "C",
            "judge_source": "llm",
            "judge_error": raw.get("error") or "parse_failed",
            "fallback_from_rule": fallback_from_rule,
            "judge_input_truncated": judge_input_truncated,
        }

    label = obj.get("label")
    if label not in VALID_JUDGE_LABELS:
        label = "F"

    return {
        "score": 1.0 if label in CORRECT_JUDGE_LABELS else 0.0,
        "judge_label": label,
        "judge_source": "llm",
        "answer_extracted": str(obj.get("answer_extracted") or ""),
        "judge_analysis_zh": str(obj.get("analysis_zh") or ""),
        "fallback_from_rule": fallback_from_rule,
        "judge_input_truncated": judge_input_truncated,
    }


async def _single_custom_rm(args, sample: Sample) -> dict[str, Any]:
    mode = _reward_mode(args, sample)
    if mode == "llm_direct":
        return await _llm_judge_reward(args, sample, fallback_from_rule=False)

    rule_score = float(get_deepscaler_rule_based_reward(str(sample.response or ""), str(sample.label or "")))
    if rule_score > 0:
        return {
            "score": rule_score,
            "judge_label": "T",
            "judge_source": "deepscaler",
            "fallback_from_rule": False,
        }

    reward = await _llm_judge_reward(args, sample, fallback_from_rule=True)
    reward["rule_score"] = rule_score
    reward["reward_mode"] = mode
    return reward


async def custom_rm(args, sample: Sample | list[Sample]):
    if isinstance(sample, list):
        return await asyncio.gather(*[_single_custom_rm(args, item) for item in sample])
    return await _single_custom_rm(args, sample)
