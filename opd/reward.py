from __future__ import annotations

import re
from typing import Any


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).casefold()


def _numeric(text: Any) -> float | None:
    matches = re.findall(r"[-+]?\d*\.?\d+", str(text or ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def _extract_last_boxed(text: Any) -> str | None:
    raw = str(text or "")
    marker = r"\boxed{"
    start = raw.rfind(marker)
    if start < 0:
        return None
    index = start + len(marker)
    depth = 1
    chars: list[str] = []
    while index < len(raw) and depth:
        char = raw[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        chars.append(char)
        index += 1
    return "".join(chars).strip() or None


def _math_candidate(text: Any) -> str:
    raw_text = str(text or "").strip()
    suffix = raw_text.rsplit("</think>", 1)[-1].strip() if "</think>" in raw_text else raw_text
    # BPM asks for a final ``Answer: ...`` line.  Prefer that explicit contract
    # over an incidental boxed expression that may occur earlier in the work.
    answer_lines: list[str] = []
    for line in suffix.splitlines():
        # Models commonly render the requested contract as ``**Answer:** 116``
        # or ``### Answer: 116``.  Remove only Markdown decoration around the
        # label; do not strip punctuation from the answer itself.
        cleaned = re.sub(r"^\s*(?:#{1,6}\s*|>\s*|[-+]\s+)", "", line).strip()
        cleaned = cleaned.replace("**", "").replace("__", "")
        cleaned = re.sub(r"^\*\s+", "", cleaned).strip()
        match = re.fullmatch(r"(?:final\s+)?answer\s*:\s*(.+?)\s*", cleaned, flags=re.IGNORECASE)
        if match:
            answer_lines.append(match.group(1))
    if answer_lines:
        raw = answer_lines[-1]
    else:
        raw = _extract_last_boxed(suffix) or _extract_last_boxed(raw_text) or suffix or raw_text
    for pattern in (
        r"(?:the\s+)?final answer is[:\s]*([^\n]+)",
        r"(?:the\s+)?answer is[:\s]*([^\n]+)",
        r"(?:the\s+)?final answer[:\s]*([^\n]+)",
        r"therefore[,:\s]+([^\n]+)",
    ):
        matches = re.findall(pattern, raw, flags=re.IGNORECASE)
        if matches:
            raw = matches[-1]
    raw = raw.strip()
    raw = re.sub(r"^(?:is|:|：)\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<\|im_end\|>\s*$", "", raw)
    return raw.strip().strip(".$")


def _math_parse_variants(text: Any) -> list[str]:
    raw = str(text or "").strip()
    candidate = _math_candidate(raw)
    variants: list[str] = []
    base_items = [candidate]
    boxed = _extract_last_boxed(raw)
    if boxed:
        base_items.append(boxed)
    if len(raw) <= 512:
        base_items.append(raw)
    for item in base_items:
        item = str(item or "").strip()
        if not item:
            continue
        variants.extend([item, f"${item}$", rf"\boxed{{{item}}}", f"$\\boxed{{{item}}}$"])
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _math_norm(text: Any) -> str:
    raw = _math_candidate(text)
    raw = raw.replace("\\left", "").replace("\\right", "")
    raw = raw.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    raw = raw.replace("\\mathrm", "")
    raw = raw.replace("\\mathbf", "")
    raw = raw.replace("\\boldsymbol", "")
    raw = raw.replace("$", "")
    raw = re.sub(r"\\boxed\{(.+)\}", r"\1", raw)
    raw = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", raw)
    raw = re.sub(r"\\frac\s+([A-Za-z0-9\\]+)\s+([A-Za-z0-9\\]+)", r"(\1)/(\2)", raw)
    raw = raw.replace("\\pi", "pi").replace("\\rho", "rho")
    raw = raw.replace("\\Omega", "Omega").replace("\\omega", "omega")
    raw = raw.replace("\\gamma", "gamma")
    raw = raw.replace("{", "").replace("}", "")
    raw = re.sub(r"\s+", "", raw)
    return raw.casefold()


def _simple_math_number(text: Any) -> float | None:
    raw = _math_candidate(text)
    raw = raw.replace("$", "").replace(",", "").strip()
    raw = re.sub(r"<\|im_end\|>\s*$", "", raw).strip()
    raw = raw.strip(".")
    frac = re.fullmatch(r"\\frac\{([-+]?\d+(?:\.\d+)?)\}\{([-+]?\d+(?:\.\d+)?)\}", raw)
    if frac:
        denominator = float(frac.group(2))
        return None if denominator == 0 else float(frac.group(1)) / denominator
    slash_frac = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)", raw)
    if slash_frac:
        denominator = float(slash_frac.group(2))
        return None if denominator == 0 else float(slash_frac.group(1)) / denominator
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", raw):
        return float(raw)
    return None


def _math_fast_score(prediction: Any, target: Any) -> bool:
    pred_num = _simple_math_number(prediction)
    target_num = _simple_math_number(target)
    if pred_num is not None and target_num is not None and abs(pred_num - target_num) <= 1e-4:
        return True
    return _math_norm(prediction) == _math_norm(target)


def _safe_for_symbolic_verify(*texts: Any) -> bool:
    joined = " ".join(str(text or "") for text in texts)
    if len(joined) > 320:
        return False
    if re.search(r"\\?(sin|cos|tan|sec|csc|cot|arcsin|arccos|arctan)\b", joined, flags=re.IGNORECASE):
        return False
    if sum(joined.count(item) for item in ("+", "-", "*", "/", "^", "=", "\\frac")) > 24:
        return False
    return True


def _math_parse_safe_text(text: Any) -> bool:
    raw = str(text or "")
    if re.search(r"\\begin|\\end|\\overline|\\mathbf|\\boldsymbol|\\mathrm|\\text|\\left\\{|\\right\\}", raw):
        return False
    if raw.count("=") > 1 or raw.count(",") > 4:
        return False
    return True


def _sympy_equivalent(target_items: list[Any], prediction_items: list[Any]) -> bool:
    try:
        import sympy as sp

        target_exprs = [item for item in target_items if isinstance(item, sp.Basic)]
        prediction_exprs = [item for item in prediction_items if isinstance(item, sp.Basic)]
        for target in target_exprs:
            for prediction in prediction_exprs:
                if sp.simplify(prediction - target) == 0:
                    return True
                if sp.simplify(sp.powdenest(prediction, force=True) - sp.powdenest(target, force=True)) == 0:
                    return True
    except Exception:
        return False
    return False


def _math_verify_score(prediction: Any, target: Any) -> bool:
    candidates = _math_parse_variants(prediction)
    target_candidates = _math_parse_variants(target)
    if _math_fast_score(prediction, target):
        return True
    if not any(_safe_for_symbolic_verify(candidate, target_text) for candidate in candidates for target_text in target_candidates):
        return False
    try:
        from math_verify import parse, verify

        parsed_targets = [
            parse(target_text)
            for target_text in target_candidates
            if _safe_for_symbolic_verify(target_text) and _math_parse_safe_text(target_text)
        ]
        parsed_targets = [item for item in parsed_targets if item]
        for candidate in [
            item
            for item in candidates
            if _safe_for_symbolic_verify(item) and _math_parse_safe_text(item)
        ]:
            parsed = parse(candidate)
            if not parsed:
                continue
            for gold in parsed_targets:
                if verify(gold, parsed) or _sympy_equivalent(gold, parsed):
                    return True
    except Exception:
        pass
    return _math_fast_score(prediction, target)


def score_answer(prediction: str, ground_truth: Any, family: str = "exact") -> float:
    family = (family or "exact").strip().lower()
    targets = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    if family in {"math_fast", "math_verify_fast", "boxed_math_fast", "latex_math_fast"}:
        return 1.0 if any(_math_fast_score(prediction, target) for target in targets) else 0.0
    if family in {"math", "math_verify", "math_answer", "boxed_math", "latex_math"}:
        return 1.0 if any(_math_verify_score(prediction, target) for target in targets) else 0.0
    if family in {"numeric", "numeric_exact"}:
        pred_num = _numeric(prediction)
        return 1.0 if pred_num is not None and any(_numeric(target) == pred_num for target in targets) else 0.0
    if family in {"contains", "ocr_inclusion"}:
        pred = _norm(prediction)
        return 1.0 if any(_norm(target) and _norm(target) in pred for target in targets) else 0.0
    return 1.0 if any(_norm(prediction) == _norm(target) for target in targets) else 0.0


async def zero_reward(_args: Any, _sample: Any, **_kwargs: Any) -> float:
    """Return zero because pure OPD uses only teacher-student distribution loss."""
    return 0.0
