import pytest

from vector_store import search_transcript


def test_empty_source_scope_returns_nothing():
    assert search_transcript("anything", source_ids=[], mode="hybrid") == []


def test_unknown_retrieval_mode_is_rejected():
    with pytest.raises(ValueError):
        search_transcript("anything", mode="bm25-only")
