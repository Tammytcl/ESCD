"""Environment-based API configuration and HTTP transport, without platform dependencies."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class ApiError(RuntimeError):
    """A transport error whose message never includes credentials or provider URLs."""

    def __init__(self, message: str, status: int | None = None):
        self.status = status
        super().__init__(message)


def safe_error(exc: BaseException) -> str:
    # Third-party error messages may echo headers, URLs or tokens. Do not persist them.
    if isinstance(exc, ApiError):
        return f"API request failed (HTTP {exc.status})" if exc.status else "API transport or response failed"
    return f"{type(exc).__name__}: request or response validation failed"


def load_endpoint(raw: dict[str, Any], prefix: str) -> dict[str, Any]:
    if "api_key" in raw:
        raise ValueError("inline api_key is not supported; use api_key_env")
    config: dict[str, Any] = {}
    for field in ("base_url", "model", "api_key"):
        env_name = raw.get(field + "_env", prefix + "_" + field.upper())
        if not isinstance(env_name, str) or not env_name.strip():
            raise ValueError(f"{field}_env must name an environment variable")
        value = os.environ.get(env_name) or raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"set {env_name} before running")
        config[field] = value.strip()
    parts = urllib.parse.urlsplit(config["base_url"])
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("base_url must be an HTTP(S) API root")
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise ValueError("base_url must not contain credentials, query parameters or fragments")
    config["base_url"] = config["base_url"].rstrip("/")
    for name, default, lower, upper in (
        ("concurrency", 4, 1, 32),
        ("timeout_seconds", 1800, 1, 172800),
        ("max_retries", 5, 1, 20),
    ):
        value = raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        config[name] = value
    return config


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward an Authorization header to a redirected host.
        return None


def post_json(config: dict[str, Any], url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + config["api_key"], "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.build_opener(_NoRedirect()).open(
            request, timeout=config["timeout_seconds"]
        ) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        exc.close()
        raise ApiError("API returned an HTTP error", exc.code) from None
    except (OSError, urllib.error.URLError, ValueError, UnicodeError):
        raise ApiError("API transport or response failed") from None
    if not isinstance(value, dict):
        raise ApiError("API response must be a JSON object")
    return value


def validate_schema(value: Any, schema: dict[str, Any], name: str = "judge JSON") -> None:
    """Validate the JSON Schema subset used by the rubric, including fallback responses."""
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be an object")
        if set(schema.get("required", [])) - set(value):
            raise ValueError(f"{name} has missing fields")
        if schema.get("additionalProperties") is False and set(value) - set(schema["properties"]):
            raise ValueError(f"{name} has unsupported fields")
        for key, child in schema["properties"].items():
            if key in value:
                validate_schema(value[key], child, name + "." + key)
    elif kind == "array":
        if not isinstance(value, list) or len(value) < schema.get("minItems", 0):
            raise ValueError(f"{name} must be a valid list")
        for child in value:
            validate_schema(child, schema["items"], name + "[]")
    elif kind == "string":
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
    elif kind in {"number", "integer"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite {kind}")
        if kind == "integer" and value != int(value):
            raise ValueError(f"{name} must be an integer")
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError(f"{name} is out of range")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{name} has an invalid enum")
