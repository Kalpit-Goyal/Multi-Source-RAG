"""Token usage extraction and USD cost estimation.

Groq prices change. The defaults below are per 1,000,000 tokens and can be
overridden with ``MODEL_PRICING_JSON`` (same shape as ``DEFAULT_PRICING``) or by
editing this table. Costs are estimates for local reporting, not billing data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

USD_PER_MILLION = 1_000_000.0

#: USD per 1M input/output tokens. Verify before quoting a cost in a resume.
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11, "output": 0.34},
    "meta-llama/llama-4-maverick-17b-128e-instruct": {"input": 0.20, "output": 0.60},
}


def _load_pricing() -> dict[str, dict[str, float]]:
    pricing = {model: dict(rates) for model, rates in DEFAULT_PRICING.items()}
    raw = os.getenv("MODEL_PRICING_JSON", "").strip()
    if not raw:
        return pricing
    try:
        override = json.loads(raw)
    except json.JSONDecodeError:
        return pricing
    if not isinstance(override, dict):
        return pricing
    for model, rates in override.items():
        if not isinstance(rates, dict):
            continue
        try:
            pricing[str(model)] = {
                "input": float(rates["input"]),
                "output": float(rates["output"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return pricing


PRICING = _load_pricing()


def _rates(model: str | None) -> dict[str, float]:
    """Return rates for a model, tolerating provider prefixes and fine-tunes."""
    name = (model or "").strip()
    if name in PRICING:
        return PRICING[name]
    for candidate, rates in PRICING.items():
        if name.endswith(candidate) or candidate.endswith(name):
            return rates
    return {"input": 0.0, "output": 0.0}


def estimate_cost(model: str | None, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for one or more LLM calls."""
    rates = _rates(model)
    cost = (
        max(0, int(input_tokens or 0)) * rates["input"]
        + max(0, int(output_tokens or 0)) * rates["output"]
    ) / USD_PER_MILLION
    return round(cost, 10)


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def usage_from_response(response: Any) -> dict[str, int]:
    """Extract token usage from a LangChain AIMessage or ChatResult."""
    usage: Mapping[str, Any] = {}
    metadata = getattr(response, "usage_metadata", None)
    if isinstance(metadata, Mapping):
        usage = metadata

    if not usage:
        response_metadata = getattr(response, "response_metadata", None) or {}
        if isinstance(response_metadata, Mapping):
            candidate = (
                response_metadata.get("token_usage")
                or response_metadata.get("usage")
                or {}
            )
            if isinstance(candidate, Mapping):
                usage = candidate

    input_tokens = _as_int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("input_token_count")
    )
    output_tokens = _as_int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("output_token_count")
    )
    total_tokens = _as_int(usage.get("total_tokens")) or input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def estimate_tokens(text: str) -> int:
    """Cheap fallback when a provider omits usage (roughly 4 chars/token)."""
    return max(1, len(text or "") // 4)


def estimate_usage(prompt_text: str, completion_text: str) -> dict[str, int]:
    input_tokens = estimate_tokens(prompt_text)
    output_tokens = estimate_tokens(completion_text)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def merge_usage(
    current: Mapping[str, Any] | None,
    delta: Mapping[str, Any] | None,
) -> dict[str, int]:
    """Add two usage dictionaries key by key."""
    merged = {
        "input_tokens": _as_int((current or {}).get("input_tokens")),
        "output_tokens": _as_int((current or {}).get("output_tokens")),
        "total_tokens": _as_int((current or {}).get("total_tokens")),
    }
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        merged[key] += _as_int((delta or {}).get(key))
    merged["total_tokens"] = max(
        merged["total_tokens"],
        merged["input_tokens"] + merged["output_tokens"],
    )
    return merged


def format_cost(cost: float | None) -> str:
    """Human-readable USD amount that stays useful for sub-cent costs."""
    if cost is None:
        return "n/a"
    if cost == 0:
        return "$0.00"
    if cost < 0.01:
        return f"${cost:.6f}"
    return f"${cost:.4f}"


def model_name() -> str:
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


@dataclass(frozen=True)
class UsageCost:
    usage: dict[str, int]
    cost_usd: float

    def as_dict(self) -> dict[str, Any]:
        return {"usage": dict(self.usage), "cost_usd": self.cost_usd}
