"""Cross-encoder reranking of fused retrieval candidates.

Bi-encoder vector search is fast but scores query and chunk independently.
A cross-encoder reads the pair together and is markedly better at deciding
whether a chunk actually answers the question. It is too slow to run over the
whole corpus, so it only reorders a small fused candidate pool.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


DEFAULT_RERANK_MODEL = os.getenv(
    "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "16"))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load the cross-encoder once and reuse it (first call downloads weights)."""
    return CrossEncoder(
        DEFAULT_RERANK_MODEL,
        device=os.getenv("RERANK_DEVICE", "cpu"),
        max_length=RERANK_MAX_LENGTH,
    )


def rerank_documents(
    query: str,
    documents: list[Document],
    k: int,
    model: CrossEncoder | None = None,
) -> list[Document]:
    """Score every ``(query, chunk)`` pair and return the best ``k``.

    ``rerank_score`` is copied into metadata so callers can inspect the
    ordering without re-running the model.
    """
    if not documents or k <= 0:
        return []

    encoder = model or get_reranker()
    pairs = [(query, document.page_content) for document in documents]
    scores = encoder.predict(
        pairs,
        batch_size=RERANK_BATCH_SIZE,
        show_progress_bar=False,
    )

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [
        Document(
            page_content=document.page_content,
            metadata={
                **document.metadata,
                "rerank_score": round(float(score), 6),
            },
        )
        for document, score in ranked[:k]
    ]
