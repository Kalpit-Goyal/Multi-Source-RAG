from langchain_core.documents import Document

from lexical import chunk_key, tokenize


def test_tokenize_lowercases_and_keeps_codes():
    assert tokenize("Error X-200 at 1,234 ms") == [
        "error",
        "x",
        "200",
        "at",
        "1234",
        "ms",
        "x200",
    ]


def test_chunk_key_prefers_metadata_identity():
    document = Document(
        page_content="anything",
        metadata={"source_id": "file:abc", "chunk_index": 7},
    )
    assert chunk_key(document) == "file:abc:7"


def test_chunk_key_falls_back_to_content_hash():
    first = Document(page_content="same text", metadata={})
    second = Document(page_content="same text", metadata={})
    third = Document(page_content="different text", metadata={})
    assert chunk_key(first) == chunk_key(second)
    assert chunk_key(first) != chunk_key(third)
