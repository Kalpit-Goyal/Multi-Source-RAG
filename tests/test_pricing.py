from types import SimpleNamespace

from pricing import (
    estimate_cost,
    format_cost,
    merge_usage,
    usage_from_response,
)


def test_usage_from_langchain_metadata():
    response = SimpleNamespace(
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
        }
    )
    assert usage_from_response(response) == {
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
    }


def test_usage_falls_back_to_response_metadata():
    response = SimpleNamespace(
        usage_metadata=None,
        response_metadata={
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5}
        },
    )
    assert usage_from_response(response)["total_tokens"] == 15


def test_cost_uses_per_million_rates():
    assert estimate_cost("openai/gpt-oss-120b", 1_000_000, 1_000_000) == 0.75


def test_cost_is_zero_for_unknown_model():
    assert estimate_cost("not-a-real-model", 1_000_000, 1_000_000) == 0.0


def test_merge_usage_adds_calls():
    merged = merge_usage(
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        {"input_tokens": 3, "output_tokens": 2},
    )
    assert merged == {"input_tokens": 13, "output_tokens": 7, "total_tokens": 20}


def test_format_cost_keeps_sub_cent_precision():
    assert format_cost(0.0000123).startswith("$0.000012")
    assert format_cost(0.25) == "$0.2500"
    assert format_cost(None) == "n/a"
