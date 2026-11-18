from langchain_core.documents import Document

from vector_store import _reciprocal_rank_fusion


def _doc(chunk: int) -> Document:
    return Document(
        page_content=f"chunk {chunk}",
        metadata={"source_id": "file:test", "chunk_index": chunk},
    )


def test_rrf_rewards_chunks_found_by_both_arms():
    both = _doc(0)
    first_only = [_doc(1), both]
    second_only = [both, _doc(2)]

    fused = _reciprocal_rank_fusion([first_only, second_only], k=3)

    assert fused[0].metadata["chunk_index"] == 0
    assert {document.metadata["chunk_index"] for document in fused} == {0, 1, 2}


def test_rrf_preserves_rank_order_within_one_list():
    ranked = [_doc(index) for index in range(5)]
    fused = _reciprocal_rank_fusion([ranked], k=3)
    assert [document.metadata["chunk_index"] for document in fused] == [0, 1, 2]


def test_rrf_handles_empty_lists():
    assert _reciprocal_rank_fusion([[], []], k=5) == []
