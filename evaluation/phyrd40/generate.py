"""Generate one PHYRD-40 response per problem from a user-configured Chat API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
from pathlib import Path
from typing import Any

from .api_client import load_endpoint, post_json, safe_error
from .rubric_judge import extract_text, nonempty_string, read_jsonl, sha256_text


def load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("generation config must be an object")
    config = load_endpoint(raw, "MODEL")
    for name, default in (("temperature", 1.0), ("top_p", 0.95)):
        number = raw.get(name, default)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number")
        if not 0 <= number <= (2 if name == "temperature" else 1):
            raise ValueError(f"{name} is out of range")
        config[name] = number
    config["max_output_tokens"] = raw.get("max_output_tokens", 256000)
    if (isinstance(config["max_output_tokens"], bool)
            or not isinstance(config["max_output_tokens"], int) or config["max_output_tokens"] < 1):
        raise ValueError("max_output_tokens must be a positive integer")
    config["token_limit_field"] = raw.get("token_limit_field", "max_tokens")
    if config["token_limit_field"] not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("invalid token_limit_field")
    extra = raw.get("extra_body", {})
    # Explicit opt-in to server-specific sampling and native thinking-template settings.
    if not isinstance(extra, dict) or set(extra) - {"top_k", "chat_template_kwargs", "seed"}:
        raise ValueError("extra_body only supports top_k, chat_template_kwargs and seed")
    config["extra_body"] = extra
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=40)
    args = parser.parse_args()
    config = load_config(args.config)
    prompts = list(read_jsonl(args.prompts))
    ids = [nonempty_string(row.get("id"), "prompt id") for row in prompts]
    if len(prompts) != args.expected_count or len(set(ids)) != len(ids):
        raise ValueError("prompt count or unique IDs do not match the requested evaluation")
    for row in prompts:
        if set(row) != {"id", "messages"}:
            raise ValueError("generation rows may contain only id and messages")
        messages = row["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("each prompt must contain chat messages")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in {"system", "user"}):
                raise ValueError("invalid generation message")
            nonempty_string(message["content"], "message content")

    def fingerprint(row: dict[str, Any]) -> str:
        public_config = {key: value for key, value in config.items() if key != "api_key"}
        return sha256_text(json.dumps({"prompt": row, "config": public_config}, sort_keys=True))

    completed = {}
    if args.output.is_file():
        for row in read_jsonl(args.output):
            if row.get("id") in completed:
                raise ValueError("duplicate IDs in cached predictions")
            completed[row.get("id")] = row
        if set(completed) - set(ids):
            raise ValueError("cached predictions contain IDs from another dataset")
    for row in prompts:
        saved = completed.get(row["id"])
        if saved and (saved.get("fingerprint") != fingerprint(row) or not saved.get("response")):
            raise ValueError("prediction cache differs from this run; choose a fresh output file")

    def generate(row: dict[str, Any]) -> dict[str, Any]:
        payload = {"model": config["model"], "messages": row["messages"], "n": 1,
                   "temperature": config["temperature"], "top_p": config["top_p"],
                   config["token_limit_field"]: config["max_output_tokens"], **config["extra_body"]}
        for attempt in range(1, config["max_retries"] + 1):
            try:
                response = post_json(config, config["base_url"] + "/chat/completions", payload)
                choices = response.get("choices")
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ValueError("expected one generated response")
                answer = nonempty_string(extract_text(response), "generated response")
                return {"id": row["id"], "response": answer, "model": config["model"],
                        "finish_reason": choices[0].get("finish_reason"),
                        "fingerprint": fingerprint(row)}
            except Exception as exc:
                if attempt == config["max_retries"]:
                    raise RuntimeError(safe_error(exc)) from None
                time.sleep(min(2 ** attempt, 30))
        raise AssertionError("unreachable")

    pending = [row for row in prompts if row["id"] not in completed]
    failed = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["concurrency"]) as executor:
            futures = {executor.submit(generate, row): row["id"] for row in pending}
            for future in concurrent.futures.as_completed(futures):
                problem_id = futures[future]
                try:
                    record = future.result()
                except Exception:
                    failed += 1
                    print(f"Generation failed for {problem_id}; rerun to retry.", flush=True)
                    continue
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
    print(f"Generation complete: {len(prompts) - failed}/{len(prompts)}; failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
