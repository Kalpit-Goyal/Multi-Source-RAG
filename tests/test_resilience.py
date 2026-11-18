from types import SimpleNamespace

import pytest

import resilience


class RateLimitError(Exception):
    status_code = 429

    def __init__(self, message="rate limit", retry_after=None):
        super().__init__(message)
        headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
        self.response = SimpleNamespace(headers=headers)


class FatalError(Exception):
    pass


class BadRequestError(Exception):
    status_code = 400


class Flaky:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        if self.calls <= self.failures:
            raise RateLimitError()
        return f"ok:{payload}"


def test_status_code_from_exception():
    assert resilience.status_code(RateLimitError()) == 429


def test_retryable_classification():
    assert resilience.is_retryable(RateLimitError())
    assert not resilience.is_retryable(FatalError("bad input"))
    assert not resilience.is_retryable(BadRequestError("bad request"))


def test_retry_after_header_is_honored():
    assert resilience.retry_after_seconds(RateLimitError(retry_after=7)) == 7.0


def test_retry_after_message_is_parsed():
    assert resilience.retry_after_seconds(
        Exception("Please try again in 2.5s")
    ) == 2.5


def test_invoke_with_retry_retries_transient_errors(monkeypatch):
    monkeypatch.setattr(resilience, "RETRY_BASE_DELAY", 0.0)
    monkeypatch.setattr(resilience, "RETRY_MAX_DELAY", 0.0)
    monkeypatch.setattr(resilience, "RETRY_JITTER", 0.0)
    monkeypatch.setattr(resilience.time, "sleep", lambda _seconds: None)

    flaky = Flaky(failures=2)
    assert resilience.invoke_with_retry(flaky, "payload", attempts=3) == "ok:payload"
    assert flaky.calls == 3


def test_invoke_with_retry_does_not_retry_fatal_errors(monkeypatch):
    monkeypatch.setattr(resilience.time, "sleep", lambda _seconds: None)

    class FatalRunnable:
        calls = 0

        def invoke(self, payload):
            self.calls += 1
            raise FatalError("bad input")

    runnable = FatalRunnable()
    with pytest.raises(FatalError):
        resilience.invoke_with_retry(runnable, "payload", attempts=3)
    assert runnable.calls == 1


def test_friendly_error_explains_rate_limits():
    assert "rate-limiting" in resilience.friendly_error(RateLimitError())
