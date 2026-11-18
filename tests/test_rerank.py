from langchain_core.documents import Document

from rerank import rerank_documents


class FakeCrossEncoder:
    """Longer chunks score higher, which makes the expected order obvious."""

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        return [float(len(text)) for _query, text in pairs]


def test_rerank_orders_by_cross_encoder_score():
    documents = [
        Document(page_content="short", metadata={"chunk_index": 0}),
        Document(page_content="a much longer chunk of text", metadata={"chunk_index": 1}),
        Document(page_content="medium length", metadata={"chunk_index": 2}),
    ]

    ranked = rerank_documents("query", documents, k=2, model=FakeCrossEncoder())

    assert [document.metadata["chunk_index"] for document in ranked] == [1, 2]
    assert ranked[0].metadata["rerank_score"] > ranked[1].metadata["rerank_score"]


def test_rerank_handles_empty_input():
    assert rerank_documents("query", [], k=3, model=FakeCrossEncoder()) == []


def test_rerank_returns_at_most_k():
    documents = [
        Document(page_content="x" * index, metadata={"chunk_index": index})
        for index in range(1, 6)
    ]
    ranked = rerank_documents("query", documents, k=3, model=FakeCrossEncoder())
    assert len(ranked) == 3
