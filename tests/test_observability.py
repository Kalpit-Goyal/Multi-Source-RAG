from observability import run_config


def test_run_config_without_tracing_is_minimal(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

    config = run_config(thread_id="chat-1", mode="hybrid", k=4)

    assert config == {"configurable": {"thread_id": "chat-1"}}


def test_run_config_with_tracing_adds_metadata(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    config = run_config(
        thread_id="chat-1",
        mode="hybrid_rerank",
        k=4,
        source_ids=["file:a", "file:b"],
        question="hello",
    )

    assert config["configurable"]["thread_id"] == "chat-1"
    assert config["run_name"] == "rag-hybrid_rerank"
    assert config["metadata"]["retrieval_mode"] == "hybrid_rerank"
    assert config["metadata"]["source_count"] == 2
    assert config["metadata"]["scoped"] is True
    assert config["tags"] == ["rag", "hybrid_rerank", "k=4"]
