from eval.ragas_resume import (
    _append_jsonl,
    _coverage,
    cache_key,
    load_cache,
)


def test_cache_key_is_stable_and_scope_specific():
    base = cache_key("vector", "context_recall", "What  is\nRAG?", "judge-a")
    assert base == cache_key("vector", "context_recall", "What is RAG?", "judge-a")
    assert base != cache_key("hybrid", "context_recall", "What is RAG?", "judge-a")
    assert base != cache_key("vector", "context_precision", "What is RAG?", "judge-a")
    assert base != cache_key("vector", "context_recall", "What is RAG?", "judge-b")


def test_load_cache_skips_corrupt_and_unscored_lines(tmp_path):
    path = tmp_path / "ragas_cache.jsonl"
    _append_jsonl(
        path,
        {"key": "good", "mode": "vector", "metric": "context_recall", "score": 0.5},
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
        handle.write('{"key": "missing-score"}\n')

    cache = load_cache(path)
    assert set(cache) == {"good"}
    assert cache["good"]["score"] == 0.5


def test_load_cache_missing_file_is_empty(tmp_path):
    assert load_cache(tmp_path / "nope.jsonl") == {}


def test_coverage_counts_only_cached_scores(tmp_path):
    path = tmp_path / "ragas_cache.jsonl"
    records = [
        {"question": "q1"},
        {"question": "q2"},
    ]
    _append_jsonl(
        path,
        {
            "key": cache_key("vector", "context_precision", "q1", "judge"),
            "score": 1.0,
        },
    )
    cache = load_cache(path)
    coverage = _coverage(
        records, ["context_precision", "context_recall"], cache, "vector", "judge"
    )

    assert coverage["context_precision"] == {"scored": 1, "total": 2}
    assert coverage["context_recall"] == {"scored": 0, "total": 2}
