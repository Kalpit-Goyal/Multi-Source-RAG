from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from loaders import LoadedSource


DEFAULT_DB_DIR = Path(__file__).parent / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "youtube_transcripts"


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


def delete_all_sources(
    persist_directory: str | Path = DEFAULT_DB_DIR,
) -> None:
    """Remove every chunk from the knowledge base."""
    vector_store = get_vector_store(persist_directory)
    stored = vector_store.get(include=[])
    ids = stored.get("ids") or []
    if ids:
        vector_store.delete(ids=ids)


def search_transcript(
    question: str,
    persist_directory: str | Path = DEFAULT_DB_DIR,
    source_ids: list[str] | None = None,
    k: int = 4,
) -> list[Document]:
    """Retrieve the most relevant chunks, optionally scoped to some sources."""
    vector_store = get_vector_store(persist_directory)
    search_filter = (
        {"source_id": {"$in": list(source_ids)}} if source_ids else None
    )
    return vector_store.similarity_search(question, k=k, filter=search_filter)
