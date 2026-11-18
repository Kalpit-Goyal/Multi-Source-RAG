from langchain_core.documents import Document

from chunking import chunk_documents
from lexical import chunk_key
from vector_store import _build_where, _metadata_predicate, expand_to_parents


def test_chunking_attaches_parent_window_metadata():
    text = "\n\n".join(
        f"Paragraph {index} " + ("word " * 400) for index in range(3)
    )
    chunks = chunk_documents(
        [Document(page_content=text, metadata={"locator": "page 1"})],
        "file:test",
    )

    assert len(chunks) >= 2
    for chunk in chunks:
        metadata = chunk.metadata
        assert metadata["parent_id"].startswith("file:test:parent:")
        assert metadata["parent_text"]
        assert metadata["parent_start"] <= metadata["chunk_index"]
        assert metadata["chunk_index"] <= metadata["parent_end"]


def test_expand_to_parents_collapses_hits_and_keeps_child_ids():
    first = Document(
        page_content="child zero",
        metadata={
            "source_id": "file:test",
            "source_name": "Test",
            "chunk_index": 0,
            "parent_id": "file:test:parent:0",
            "parent_start": 0,
            "parent_end": 1,
            "parent_text": "child zero\n\nchild one",
        },
    )
    second = Document(
        page_content="child one",
        metadata={
            "source_id": "file:test",
            "source_name": "Test",
            "chunk_index": 1,
            "parent_id": "file:test:parent:0",
            "parent_start": 0,
            "parent_end": 1,
            "parent_text": "child zero\n\nchild one",
        },
    )

    expanded = expand_to_parents([first, second])

    assert len(expanded) == 1
    parent = expanded[0]
    assert parent.page_content == "child zero\n\nchild one"
    assert parent.metadata["chunk_index"] == 0
    assert parent.metadata["chunk_end"] == 1
    assert parent.metadata["child_chunk_ids"] == ["file:test:0", "file:test:1"]
    assert "parent_text" not in parent.metadata


def test_old_chunks_pass_through_parent_expansion():
    old = Document(
        page_content="legacy",
        metadata={"source_id": "file:old", "chunk_index": 2},
    )
    assert expand_to_parents([old]) == [old]


def test_metadata_where_combines_source_and_type():
    where = _build_where(["file:a"], ["pdf", "website"])
    assert where == {
        "$and": [
            {"source_id": {"$in": ["file:a"]}},
            {"source_type": {"$in": ["pdf", "website"]}},
        ]
    }


def test_metadata_predicate_applies_both_filters():
    predicate = _metadata_predicate(["file:a"], ["pdf"])
    assert predicate(
        Document(page_content="x", metadata={"source_id": "file:a", "source_type": "pdf"})
    )
    assert not predicate(
        Document(page_content="x", metadata={"source_id": "file:b", "source_type": "pdf"})
    )
    assert not predicate(
        Document(page_content="x", metadata={"source_id": "file:a", "source_type": "text"})
    )


def test_chunk_key_unchanged_for_parent_children():
    child = Document(
        page_content="x",
        metadata={"source_id": "file:a", "chunk_index": 3},
    )
    assert chunk_key(child) == "file:a:3"
