from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


DEFAULT_DB_DIR = Path(__file__).parent / "chroma_db"


def get_vector_store(persist_directory: str | Path = DEFAULT_DB_DIR) -> Chroma:
    """Return the persistent Chroma collection used by the application."""
    return Chroma(
        collection_name="youtube_transcripts",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=str(persist_directory),
    )


def store_transcript(
    video_id: str,
    chunks: list[str],
    persist_directory: str | Path = DEFAULT_DB_DIR,
) -> int:
    """Store transcript chunks and return the number of chunks written."""
    vector_store = get_vector_store(persist_directory)
    ids = [f"{video_id}-{index}" for index in range(len(chunks))]
    existing_ids = set(vector_store.get(ids=ids).get("ids", []))

    if existing_ids:
        vector_store.delete(ids=list(existing_ids))

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "video_id": video_id,
                "chunk_index": index,
            },
        )
        for index, chunk in enumerate(chunks)
    ]
    vector_store.add_documents(documents=documents, ids=ids)
    return len(documents)


def search_transcript(
    question: str,
    persist_directory: str | Path = DEFAULT_DB_DIR,
    video_id: str | None = None,
    k: int = 4,
) -> list[Document]:
    """Retrieve the most relevant transcript chunks for a question."""
    vector_store = get_vector_store(persist_directory)
    if video_id:
        return vector_store.similarity_search(
            question,
            k=k,
            filter={"video_id": video_id},
        )
    return vector_store.similarity_search(question, k=k)