from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", "? ", " ", ""],
)


def create_chunks(text: str) -> list[str]:
    """Split a block of text into embedding-sized chunks."""
    return SPLITTER.split_text(text)


def chunk_documents(documents: list[Document], source_id: str) -> list[Document]:
    """Split per-page/sheet documents while keeping their locator metadata."""
    chunks: list[Document] = []
    for document in documents:
        for piece in SPLITTER.split_text(document.page_content):
            metadata = {
                **document.metadata,
                "source_id": source_id,
                "chunk_index": len(chunks),
            }
            chunks.append(Document(page_content=piece, metadata=metadata))
    return chunks
