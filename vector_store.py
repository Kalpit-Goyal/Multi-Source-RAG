from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from lexical import LexicalIndex, build_index, chunk_key
from loaders import LoadedSource
from rerank import rerank_documents


DEFAULT_DB_DIR = Path(__file__).parent / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "youtube_transcripts"

#: Retrieval modes exposed to the UI, CLI, and evaluation harness.
RETRIEVAL_MODES = ("vector", "hybrid", "hybrid_rerank")

#: Candidates pulled from each arm before fusion.
VECTOR_POOL = 20
LEXICAL_POOL = 20
#: Fused candidates handed to the cross-encoder.
RERANK_POOL = int(os.getenv("RERANK_POOL", "20"))
#: Standard reciprocal-rank-fusion constant; dampens the gap between adjacent ranks.
RRF_K = 60

#: Rebuilt lazily from Chroma, keyed by persist directory.
_LEXICAL_CACHE: dict[str, tuple[tuple[int, int, int], LexicalIndex]] = {}


@dataclass(frozen=True)
class SourceInfo:
    source_id: str
    name: str
    type: str
    chunks: int


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Load the local embedding model once and reuse it across queries."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(persist_directory: str | Path = DEFAULT_DB_DIR) -> Chroma:
    """Return the persistent Chroma collection used by the application."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_directory),
    )


def _fingerprint(
    persist_directory: str | Path,
    chunk_count: int,
) -> tuple[int, int, int]:
    """Cheap signal that the Chroma database changed, including from another process."""
    try:
        stat = (Path(persist_directory) / "chroma.sqlite3").stat()
    except OSError:
        return 0, 0, chunk_count
    return stat.st_mtime_ns, stat.st_size, chunk_count


def _invalidate_lexical_index(persist_directory: str | Path) -> None:
    """Drop the cached BM25 index after a write to this persist directory."""
    _LEXICAL_CACHE.pop(str(Path(persist_directory)), None)


def _lexical_index(persist_directory: str | Path = DEFAULT_DB_DIR) -> LexicalIndex:
    """Return the cached BM25 index, rebuilding it when the corpus changed."""
    key = str(Path(persist_directory))
    vector_store = get_vector_store(persist_directory)
    collection = getattr(vector_store, "_collection", None)
    count = collection.count() if collection is not None else -1
    fingerprint = _fingerprint(persist_directory, count)

    cached = _LEXICAL_CACHE.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]

    stored = vector_store.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=content or "", metadata=metadata or {})
        for content, metadata in zip(
            stored.get("documents") or [],
            stored.get("metadatas") or [],
        )
    ]
    index = build_index(documents)
    _LEXICAL_CACHE[key] = (fingerprint, index)
    return index


def index_source(
    source: LoadedSource,
    chunks: list[Document],
    persist_directory: str | Path = DEFAULT_DB_DIR,
) -> int:
    """Replace any existing chunks for a source with the newly chunked text."""
    vector_store = get_vector_store(persist_directory)
    vector_store.delete(where={"source_id": source.source_id})

    documents = [
        Document(
            page_content=chunk.page_content,
            metadata={
                **chunk.metadata,
                "source_id": source.source_id,
                "source_name": source.source_name,
                "source_type": source.source_type,
            },
        )
        for chunk in chunks
    ]
    ids = [
        f"{source.source_id}:{document.metadata['chunk_index']}"
        for document in documents
    ]
    if documents:
        vector_store.add_documents(documents=documents, ids=ids)
    _invalidate_lexical_index(persist_directory)
    return len(documents)


def list_sources(
    persist_directory: str | Path = DEFAULT_DB_DIR,
) -> list[SourceInfo]:
    """Return every indexed source, derived from chunk metadata."""
    stored = get_vector_store(persist_directory).get(include=["metadatas"])
    counts: dict[str, list] = {}
    for metadata in stored.get("metadatas") or []:
        source_id = metadata.get("source_id")
        if not source_id:
            continue
        if source_id in counts:
            counts[source_id][2] += 1
        else:
            counts[source_id] = [
                metadata.get("source_name", source_id),
                metadata.get("source_type", "unknown"),
                1,
            ]

    return sorted(
        (
            SourceInfo(source_id, name, source_type, chunks)
            for source_id, (name, source_type, chunks) in counts.items()
        ),
        key=lambda source: (source.type, source.name.lower()),
    )


def delete_source(
    source_id: str,
    persist_directory: str | Path = DEFAULT_DB_DIR,
) -> None:
    """Remove every chunk that belongs to one source."""
    get_vector_store(persist_directory).delete(where={"source_id": source_id})
    _invalidate_lexical_index(persist_directory)


def delete_all_sources(
    persist_directory: str | Path = DEFAULT_DB_DIR,
) -> None:
    """Remove every chunk from the knowledge base."""
    vector_store = get_vector_store(persist_directory)
    stored = vector_store.get(include=[])
    ids = stored.get("ids") or []
    if ids:
        vector_store.delete(ids=ids)
    _invalidate_lexical_index(persist_directory)


def _build_where(
    source_ids: list[str] | None,
    source_types: list[str] | None,
) -> dict | None:
    """Build a Chroma metadata filter from source ids and source types."""
    clauses: list[dict] = []
    if source_ids is not None:
        clauses.append({"source_id": {"$in": list(source_ids)}})
    if source_types is not None:
        clauses.append({"source_type": {"$in": list(source_types)}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _metadata_predicate(
    source_ids: list[str] | None,
    source_types: list[str] | None,
):
    """Apply the same filters to the in-memory BM25 index."""
    if source_ids is None and source_types is None:
        return None
    allowed_ids = set(source_ids) if source_ids is not None else None
    allowed_types = set(source_types) if source_types is not None else None

    def predicate(document: Document) -> bool:
        metadata = document.metadata or {}
        if allowed_ids is not None and metadata.get("source_id") not in allowed_ids:
            return False
        if allowed_types is not None and metadata.get("source_type") not in allowed_types:
            return False
        return True

    return predicate


def expand_to_parents(documents: list[Document]) -> list[Document]:
    """Collapse child hits into their larger parent-window context.

    Chunks indexed before parent metadata existed pass through unchanged.
    Child ids are retained in ``child_chunk_ids`` so retrieval metrics can
    still compare against a gold child-chunk id.
    """
    expanded: list[Document] = []
    seen: set[str] = set()
    for document in documents:
        metadata = document.metadata or {}
        parent_id = metadata.get("parent_id")
        parent_text = metadata.get("parent_text")
        if parent_id and parent_text:
            key = f"parent:{parent_id}"
            if key in seen:
                continue
            seen.add(key)
            start = metadata.get("parent_start", metadata.get("chunk_index"))
            end = metadata.get("parent_end", start)
            child_ids: list[str] = []
            source_id = metadata.get("source_id")
            if source_id is not None and start is not None and end is not None:
                child_ids = [
                    f"{source_id}:{index}"
                    for index in range(int(start), int(end) + 1)
                ]
            parent_metadata = {
                **metadata,
                "chunk_index": start,
                "chunk_end": end,
                "child_chunk_index": metadata.get("chunk_index"),
                "child_chunk_ids": child_ids,
            }
            parent_metadata.pop("parent_text", None)
            expanded.append(
                Document(page_content=parent_text, metadata=parent_metadata)
            )
            continue

        key = chunk_key(document)
        if key in seen:
            continue
        seen.add(key)
        expanded.append(document)
    return expanded


def _reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int,
) -> list[Document]:
    """Merge ranked lists by summing ``1 / (RRF_K + rank)`` per chunk."""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for hits in ranked_lists:
        for rank, document in enumerate(hits, start=1):
            key = chunk_key(document)
            documents.setdefault(key, document)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)

    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [documents[key] for key in ordered[:k]]


def search_transcript(
    question: str,
    persist_directory: str | Path = DEFAULT_DB_DIR,
    source_ids: list[str] | None = None,
    k: int = 4,
    mode: str = "hybrid_rerank",
    source_types: list[str] | None = None,
) -> list[Document]:
    """Retrieve the most relevant chunks, optionally scoped by metadata.

    ``vector`` uses dense similarity only. ``hybrid`` fuses dense similarity
    with BM25 using reciprocal rank fusion. ``hybrid_rerank`` does the same,
    then reorders the fused candidate pool with a cross-encoder and keeps the
    best ``k``. All three share the same candidate pool so evaluation compares
    like with like.

    ``source_ids`` and ``source_types`` are metadata filters. An explicit
    empty list means "match nothing", not "no filter".
    """
    if mode not in RETRIEVAL_MODES:
        raise ValueError(
            f"Unknown retrieval mode: {mode!r}. Choose from {RETRIEVAL_MODES}."
        )
    if source_ids is not None and len(source_ids) == 0:
        return []
    if source_types is not None and len(source_types) == 0:
        return []

    vector_store = get_vector_store(persist_directory)
    search_filter = _build_where(source_ids, source_types)
    predicate = _metadata_predicate(source_ids, source_types)
    pool = max(k * 3, VECTOR_POOL)

    ranked = [
        [
            document
            for document, _distance in vector_store.similarity_search_with_score(
                question,
                k=pool,
                filter=search_filter,
            )
        ]
    ]
    if mode in {"hybrid", "hybrid_rerank"}:
        ranked.append(
            _lexical_index(persist_directory).search(
                question,
                k=max(k * 3, LEXICAL_POOL),
                allowed=set(source_ids) if source_ids is not None else None,
                predicate=predicate,
            )
        )

    fused = _reciprocal_rank_fusion(
        ranked,
        max(k, RERANK_POOL) if mode == "hybrid_rerank" else k,
    )
    if mode == "hybrid_rerank":
        return rerank_documents(question, fused, k)
    return fused


def search_multi_query(
    queries: list[str],
    persist_directory: str | Path = DEFAULT_DB_DIR,
    source_ids: list[str] | None = None,
    k: int = 4,
    mode: str = "hybrid_rerank",
    source_types: list[str] | None = None,
) -> list[Document]:
    """Retrieve for several paraphrases/sub-questions and fuse the results.

    For ``hybrid_rerank`` each query first uses hybrid retrieval, the ranked
    lists are combined with RRF, and the cross-encoder reranks the fused pool
    once. Reranking every sub-query separately would multiply the most
    expensive step and make scores incomparable.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for query in queries or []:
        value = " ".join(str(query or "").split())
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    if not cleaned:
        return []
    if len(cleaned) == 1:
        return search_transcript(
            cleaned[0],
            persist_directory=persist_directory,
            source_ids=source_ids,
            k=k,
            mode=mode,
            source_types=source_types,
        )

    base_mode = "hybrid" if mode == "hybrid_rerank" else mode
    candidate_k = max(k * 3, RERANK_POOL) if mode == "hybrid_rerank" else k
    ranked_lists = [
        search_transcript(
            query,
            persist_directory=persist_directory,
            source_ids=source_ids,
            k=candidate_k,
            mode=base_mode,
            source_types=source_types,
        )
        for query in cleaned
    ]
    fused = _reciprocal_rank_fusion(
        ranked_lists,
        max(k, RERANK_POOL) if mode == "hybrid_rerank" else k,
    )
    if mode == "hybrid_rerank":
        return rerank_documents(cleaned[0], fused, k)
    return fused
