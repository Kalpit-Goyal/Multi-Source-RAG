"""Retry, exponential backoff, and rate-limit handling for LLM calls.

LangChain's ChatGroq has its own retries, but they are opaque and do not
surface the server's ``Retry-After`` hint. This module adds explicit retries
with jitter for rate limits, timeouts, connection errors, and 5xx responses,
plus an optional minimum interval between calls to avoid bursting a free-tier
quota.
"""

from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from typing import Any

LOGGER = logging.getLogger(__name__)

RETRY_ATTEMPTS = max(1, int(os.getenv("LLM_RETRY_ATTEMPTS", "4")))
RETRY_BASE_DELAY = max(0.0, float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0")))
RETRY_MAX_DELAY = max(RETRY_BASE_DELAY, float(os.getenv("LLM_RETRY_MAX_DELAY", "30")))
RETRY_JITTER = max(0.0, float(os.getenv("LLM_RETRY_JITTER", "0.3")))
LLM_TIMEOUT_S = max(1.0, float(os.getenv("LLM_TIMEOUT_S", "60")))

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_TEXT = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "429",
    "timeout",
    "timed out",
    "connection error",
    "connection reset",
    "temporarily unavailable",
    "overloaded",
    "service unavailable",
    "internal server error",
    "502",
    "503",
    "504",
)
_RETRY_AFTER_PATTERN = re.compile(
    r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|s|seconds?|minutes?)?",
    re.IGNORECASE,
)


class RateLimiter:
    """A process-wide minimum interval between outbound LLM requests."""

    def __init__(self, min_interval: float = 0.0):
        self.min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_allowed = now + self.min_interval


LLM_RATE_LIMITER = RateLimiter(float(os.getenv("LLM_MIN_INTERVAL_S", "0")))


def status_code(error: BaseException) -> int | None:
    """Best-effort HTTP status extraction from provider SDK exceptions."""
    for value in (
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    match = re.search(r"\b(4\d\d|5\d\d)\b", str(error))
    return int(match.group(1)) if match else None


def is_retryable(error: BaseException) -> bool:
    """Return True for transient quota, network, timeout, and server errors."""
    code = status_code(error)
    if code in {400, 401, 403, 404, 405, 413, 422}:
        return False
    if code in _RETRYABLE_STATUS:
        return True

    name = type(error).__name__.lower()
    if any(
        marker in name
        for marker in (
            "ratelimit",
            "rate_limit",
            "timeout",
            "connection",
            "internalserver",
            "serviceunavailable",
            "overloaded",
        )
    ):
        return True

    message = str(error).lower()
    return any(marker in message for marker in _RETRYABLE_TEXT)


def retry_after_seconds(error: BaseException) -> float | None:
    """Read ``Retry-After`` or a Groq "try again in 6.9s" message."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        for header in ("retry-after", "x-ratelimit-reset-requests"):
            raw = headers.get(header) if hasattr(headers, "get") else None
            if raw is None:
                continue
            text = str(raw).strip()
            try:
                return max(0.0, float(text))
            except ValueError:
                pass
            # Some headers use Go-style durations such as "1m30s".
            match = re.fullmatch(
                r"(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?",
                text,
                re.IGNORECASE,
            )
            if match and any(match.groups()):
                minutes = float(match.group(1) or 0)
                seconds = float(match.group(2) or 0)
                return minutes * 60 + seconds

    match = _RETRY_AFTER_PATTERN.search(str(error))
    if not match:
        return None
    value = float(match.group(1))
    unit = (match.group(2) or "s").lower()
    if unit == "ms":
        return value / 1000.0
    if unit.startswith("minute"):
        return value * 60.0
    return value


def backoff_delay(attempt: int, error: BaseException) -> float:
    """Exponential backoff with jitter, honoring ``Retry-After`` when present."""
    exponential = RETRY_BASE_DELAY * (2 ** max(0, attempt - 1))
    hinted = retry_after_seconds(error)
    base = max(exponential, hinted) if hinted is not None else exponential
    base = min(RETRY_MAX_DELAY, base)
    jitter = random.uniform(0, RETRY_JITTER * base) if RETRY_JITTER else 0.0
    return min(RETRY_MAX_DELAY, base + jitter)


def invoke_with_retry(
    runnable: Any,
    payload: Any,
    *,
    description: str = "LLM call",
    attempts: int | None = None,
) -> Any:
    """Invoke a LangChain runnable (or callable) with explicit retries."""
    total_attempts = max(1, int(attempts or RETRY_ATTEMPTS))
    last_error: BaseException | None = None

    for attempt in range(1, total_attempts + 1):
        LLM_RATE_LIMITER.wait()
        try:
            if hasattr(runnable, "invoke"):
                return runnable.invoke(payload)
            return runnable(payload)
        except Exception as error:
            last_error = error
            if attempt >= total_attempts or not is_retryable(error):
                raise
            delay = backoff_delay(attempt, error)
            LOGGER.warning(
                "%s failed (%s: %s); retrying in %.1fs (attempt %d/%d).",
                description,
                type(error).__name__,
                error,
                delay,
                attempt + 1,
                total_attempts,
            )
            time.sleep(delay)

    assert last_error is not None
    raise last_error


def friendly_error(error: BaseException) -> str:
    """Turn a provider exception into a short message suitable for a UI."""
    code = status_code(error)
    if code == 429:
        return "The model provider is rate-limiting requests. Wait a moment and try again."
    if code in {500, 502, 503, 504}:
        return "The model provider is temporarily unavailable. Try again shortly."
    if "timeout" in type(error).__name__.lower() or "timeout" in str(error).lower():
        return "The model provider timed out. Try a shorter question or retry."
    return f"{type(error).__name__}: {error}"
