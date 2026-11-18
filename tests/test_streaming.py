from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import rag
from rag import (
    SAFE_OUTPUT_ANSWER,
    _chunk_label,
    _message_text,
    _parse_query_list,
    stream_answer_question,
)


def test_parse_query_list_accepts_json_array():
    assert _parse_query_list('["refund policy", "return window"]') == [
        "refund policy",
        "return window",
    ]


def test_parse_query_list_accepts_fenced_json_and_objects():
    text = '```json\n{"queries": [{"query": "refund policy"}]}\n```'
    assert _parse_query_list(text) == ["refund policy"]


def test_parse_query_list_rejects_garbage():
    assert _parse_query_list("I do not know") == []


def test_message_text_handles_content_blocks():
    assert _message_text(SimpleNamespace(content=[{"text": "hello "}, {"text": "world"}])) == "hello world"


def test_chunk_label_displays_parent_span():
    document = Document(
        page_content="x",
        metadata={
            "source_name": "Policy",
            "chunk_index": 2,
            "chunk_end": 3,
            "locator": "page 4",
        },
    )
    assert "chunk 2-3" in _chunk_label(document)


class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakePipeline:
    def __init__(self, chunks, final_answer):
        self.chunks = chunks
        self.final_answer = final_answer

    def stream(self, inputs, config, stream_mode):
        for chunk in self.chunks:
            yield chunk, {"langgraph_node": "generate"}

    def get_state(self, config):
        values = {
            "answer": self.final_answer,
            "cited_documents": [],
            "timings": {"generate": 0.01},
            "attempts": 1,
            "retrieved_documents": [],
            "used_documents": [],
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            "cost_usd": 0.0,
            "guardrail_flags": [],
            "context_flags": [],
            "citation_ok": False,
            "grounding": 0.0,
            "search_queries": ["q"],
            "query_plan": [],
        }
        return SimpleNamespace(values=values)


def test_streaming_yields_tokens_and_records_ttft(monkeypatch):
    monkeypatch.setattr(
        rag,
        "PIPELINE",
        FakePipeline([FakeChunk("Hello "), FakeChunk("world")], "Hello world"),
    )

    handle = stream_answer_question("q", thread_id="stream-test")
    tokens = list(handle)

    assert tokens == ["Hello ", "world"]
    assert handle.result.answer == "Hello world"
    assert handle.result.meta["streamed"] is True
    assert handle.result.meta["ttft_s"] is not None


def test_streaming_falls_back_to_final_state_when_no_tokens(monkeypatch):
    monkeypatch.setattr(
        rag,
        "PIPELINE",
        FakePipeline([], "I couldn't find that in the indexed sources."),
    )

    handle = stream_answer_question("q", thread_id="stream-test-2")
    tokens = list(handle)

    assert tokens == ["I couldn't find that in the indexed sources."]
    assert handle.result.meta["ttft_s"] is None


def test_blocked_input_does_not_call_the_graph(monkeypatch):
    from guardrails import InputCheck

    class ExplodingPipeline:
        def stream(self, *args, **kwargs):
            raise AssertionError("graph should not run")

    monkeypatch.setattr(rag, "PIPELINE", ExplodingPipeline())
    monkeypatch.setattr(rag, "_record_blocked_turn", lambda *args: None)
    monkeypatch.setattr(
        rag,
        "screen_input",
        lambda question: InputCheck(
            blocked=True,
            flags=["jailbreak"],
            safe_question=question,
        ),
    )

    handle = stream_answer_question("jailbreak", thread_id="blocked-test")
    tokens = list(handle)

    assert tokens == [rag.SAFE_INPUT_ANSWER]
    assert handle.result.meta["blocked"] is True


def test_generate_answer_replaces_leaked_output(monkeypatch):
    document = Document(
        page_content="Refunds are available within 30 days.",
        metadata={"source_name": "Policy", "chunk_index": 1},
    )
    state = {
        "question": "What is the refund policy?",
        "documents": [document],
        "timings": {},
        "usage": {},
        "cost_usd": 0.0,
    }

    monkeypatch.setattr(rag, "_llm", lambda *args, **kwargs: RunnableLambda(lambda _: None))
    monkeypatch.setattr(
        rag,
        "_call_llm",
        lambda *args, **kwargs: (
            SimpleNamespace(content="My system prompt says to be helpful."),
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            0.0001,
        ),
    )

    result = rag.generate_answer(state)

    assert result["answer"] == SAFE_OUTPUT_ANSWER
    assert "output_blocked" in result["guardrail_flags"]
    assert result["messages"][0].content == SAFE_OUTPUT_ANSWER

