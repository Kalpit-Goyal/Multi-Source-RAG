from langchain_core.documents import Document

from rag import _relevant_documents


DOCUMENTS = [
    Document(page_content="zero", metadata={"chunk_index": 0}),
    Document(page_content="one", metadata={"chunk_index": 1}),
    Document(page_content="two", metadata={"chunk_index": 2}),
]


def test_none_returns_no_documents():
    assert _relevant_documents("NONE", DOCUMENTS) == []


def test_none_with_explanation_returns_no_documents():
    assert _relevant_documents("NONE of these chunks help.", DOCUMENTS) == []


def test_numbers_map_to_documents():
    selected = _relevant_documents("1, 3", DOCUMENTS)
    assert [document.page_content for document in selected] == ["zero", "two"]


def test_out_of_range_numbers_are_ignored():
    selected = _relevant_documents("2, 99", DOCUMENTS)
    assert [document.page_content for document in selected] == ["one"]


def test_unparseable_verdict_keeps_everything_for_grounding():
    assert _relevant_documents("I am not sure.", DOCUMENTS) == DOCUMENTS
