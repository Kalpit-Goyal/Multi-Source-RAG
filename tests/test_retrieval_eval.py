from langchain_core.documents import Document

from eval import run_eval


def test_retrieval_only_makes_no_llm_calls(monkeypatch):
    document = Document(
        page_content="Refunds are available within 30 days.",
        metadata={"source_id": "s1", "chunk_index": 0},
    )

    def fake_search(*args, **kwargs):
        return [document]

    monkeypatch.setattr(run_eval, "search_transcript", fake_search)
    monkeypatch.setattr(run_eval, "expand_to_parents", lambda docs: docs)

    rows = [
        {
            "question": "What is the refund policy?",
            "ground_truth": "Refunds are available within 30 days.",
            "ground_truth_chunk_ids": ["s1:0"],
        }
    ]
    results = run_eval.run_retrieval_questions(rows, ["vector"], 4)

    record = results["vector"][0]
    assert record["answer"] == ""
    assert record["cost_usd"] == 0.0
    assert record["usage"] == {}
    assert record["ttft_s"] is None
    assert record["retrieval_metrics"]["hit_rate"] == 1.0
    assert record["retrieved_ids"] == ["s1:0"]
