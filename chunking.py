from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


#: Child chunks are the unit indexed and retrieved. Keeping the original size
#: preserves existing chunk ids when a source is re-indexed.
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", "? ", " ", ""],
)

#: Number of adjacent child chunks grouped into one parent context window.
PARENT_WINDOW = max(1, int(os.getenv("PARENT_WINDOW", "2")))


def create_chunks(text: str) -> list[str]:
    """Split a block of text into embedding-sized chunks."""
    return SPLITTER.split_text(text)


def chunk_documents(documents: list[Document], source_id: str) -> list[Document]:
    """Split documents into child chunks and attach parent-window metadata.

    Retrieval still ranks the small child chunks, but generation can expand a
    hit into a larger parent window. The parent text is stored on each child
    (denormalized) so no second collection or extra round trip is needed.
    ``chunk_index`` remains the global child index, so existing eval ids stay
    valid when the same source is re-indexed.
    """
    chunks: list[Document] = []
    for document in documents:
        pieces = SPLITTER.split_text(document.page_content)
        if not pieces:
            continue
        base = len(chunks)
        for offset, piece in enumerate(pieces):
            metadata = {
                **document.metadata,
                "source_id": source_id,
                "chunk_index": base + offset,
            }
            chunks.append(Document(page_content=piece, metadata=metadata))

        for offset in range(len(pieces)):
            window_start = (offset // PARENT_WINDOW) * PARENT_WINDOW
            window_end = min(window_start + PARENT_WINDOW, len(pieces))
            parent_text = "\n\n".join(pieces[window_start:window_end])
            parent_index = base + window_start
            chunks[base + offset].metadata.update(
                {
                    "parent_id": f"{source_id}:parent:{parent_index}",
                    "parent_index": parent_index,
                    "parent_start": base + window_start,
                    "parent_end": base + window_end - 1,
                    "parent_text": parent_text,
                }
            )
    return chunks
