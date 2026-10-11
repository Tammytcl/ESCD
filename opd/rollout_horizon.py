"""POPD/TOPD response budgets, independent of teacher/student token alignment.

The schedule is stateless: k is the number of optimizer updates before the
current rollout, derived from the checkpoint-restored rollout id. Budgets are
in student tokens; neither the prompt nor a synthetic EOS is included.
"""

import json
import math
from fractions import Fraction


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"sampling.horizon.{name} must be a positive integer")
    return value


def _fraction_tokens(value, max_response_len):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("sampling.horizon.fraction must be a finite number strictly between 0 and 1")
    # Use the decimal representation so e.g. 0.29 * 100 gives 29, not 28
    # after binary floating-point rounding. Fractional tokens round down.
    tokens = int(Fraction(str(value)) * max_response_len)
    if tokens < 1:
        raise ValueError("sampling.horizon.fraction produces fewer than one response token")
    return tokens


def normalize_horizon(config, max_response_len):
    if isinstance(config, str):
        config = json.loads(config)
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("sampling.horizon must be an object")
    mode = config.get("mode", "full")
    fields = {
        "full": {"mode"},
        "popd": {"mode", "initial_tokens", "increment_tokens", "increment_every", "max_tokens"},
        "topd": {"mode", "fixed_tokens", "fraction"},
    }
    if not isinstance(mode, str) or mode not in fields:
        raise ValueError("sampling.horizon.mode must be full, popd, or topd")
    unknown = set(config) - fields[mode]
    if unknown:
        raise ValueError(f"Unexpected sampling.horizon fields for {mode}: {sorted(unknown)}")
    result = {"mode": mode}
    if mode == "full":
        return result
    _positive_int(max_response_len, "max_response_len")
    if mode == "topd":
        if ("fixed_tokens" in config) == ("fraction" in config):
            raise ValueError("TOPD requires exactly one of sampling.horizon.fixed_tokens or fraction")
        if "fraction" in config:
            _fraction_tokens(config["fraction"], max_response_len)
            result["fraction"] = config["fraction"]
        else:
            result["fixed_tokens"] = _positive_int(config["fixed_tokens"], "fixed_tokens")
            if result["fixed_tokens"] > max_response_len:
                raise ValueError("sampling.horizon.fixed_tokens exceeds the response budget")
        return result
    for key in ("initial_tokens", "increment_tokens", "increment_every"):
        if key == "initial_tokens" and config.get(key) is None:
            raise ValueError(
                "sampling.horizon.initial_tokens (H_0) is required: POPD paper 2605.31490v2 "
                "does not specify it; supply an explicitly sourced value, not an implicit 512 default"
            )
        result[key] = _positive_int(config.get(key), key)
    cap = max_response_len
    if "max_tokens" in config:
        cap = _positive_int(config["max_tokens"], "max_tokens")
        if cap > max_response_len:
            raise ValueError("sampling.horizon.max_tokens exceeds sampling.max_response_len")
        result["max_tokens"] = cap
    initial = result["initial_tokens"]
    if initial > cap:
        raise ValueError("sampling.horizon initial/fixed tokens exceed the response budget")
    return result


def response_horizon(config, max_response_len, step):
    """Resolve an absolute, zero-based optimizer step (including after resume)."""
    config = normalize_horizon(config, max_response_len)
    if type(step) is not int or step < 0:
        raise ValueError("horizon step must be a nonnegative integer")
    if config["mode"] == "full":
        return max_response_len
    if config["mode"] == "topd":
        if "fraction" in config:
            return _fraction_tokens(config["fraction"], max_response_len)
        return config["fixed_tokens"]
    return min(
        config.get("max_tokens", max_response_len),
        config["initial_tokens"] + config["increment_tokens"] * (step // config["increment_every"]),
    )


def validate_horizon_args(args):
    config = normalize_horizon(getattr(args, "rollout_horizon", None), args.rollout_max_response_len)
    if config["mode"] == "full":
        return config
    # These alternate generators can resume old prefixes or ignore per-request
    # sampling params. Fail explicitly instead of silently running full OPD.
    if getattr(args, "partial_rollout", False):
        raise ValueError("POPD/TOPD requires fresh rollouts; disable --partial-rollout")
    if getattr(args, "rollout_function_path", "slime.rollout.sglang_rollout.generate_rollout") != (
        "slime.rollout.sglang_rollout.generate_rollout"
    ):
        raise ValueError("POPD/TOPD requires slime.rollout.sglang_rollout.generate_rollout")
    if getattr(args, "custom_generate_function_path", None) is not None:
        raise ValueError("POPD/TOPD currently requires the direct SGLang generator")
    batch = args.rollout_batch_size * args.n_samples_per_prompt
    if not args.global_batch_size or batch % args.global_batch_size:
        raise ValueError("POPD/TOPD requires rollout samples divisible by global_batch_size")
    return config


def horizon_for_rollout(args, rollout_id):
    config = validate_horizon_args(args)
    if config["mode"] == "full":
        return None
    if type(rollout_id) is not int or rollout_id < 0:
        raise ValueError("rollout_id must be a nonnegative integer")
    steps_per_rollout = args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    step = rollout_id * steps_per_rollout
    return {
        "mode": config["mode"],
        "rollout_id": rollout_id,
        "step": step,
        "max_new_tokens": response_horizon(config, args.rollout_max_response_len, step),
    }
