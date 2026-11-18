import pytest

from loaders import _html_documents, _normalize_url


def test_normalize_url_strips_tracking_and_fragment():
    normalized = _normalize_url(
        "https://www.Example.com/article?utm_source=news&id=7#section"
    )
    assert normalized == "https://example.com/article?id=7"


def test_normalize_url_keeps_meaningful_query():
    normalized = _normalize_url("https://example.com/search?q=rag&page=2")
    assert normalized == "https://example.com/search?page=2&q=rag"


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/private",
        "http://127.0.0.1/private",
        "http://192.168.1.10/private",
        "ftp://example.com/file",
    ],
)
def test_normalize_url_rejects_private_or_unsupported(url):
    with pytest.raises(ValueError):
        _normalize_url(url)


def test_html_extraction_drops_scripts_and_navigation():
    payload = b"""
    <html>
      <head><title>Example Article</title></head>
      <body>
        <nav>Navigation links that should not be indexed.</nav>
        <script>alert('do not index this')</script>
        <article>
          <h1>Heading</h1>
          <p>This paragraph is long enough to be kept by the extractor.</p>
          <p>This paragraph is long enough to be kept by the extractor.</p>
          <ul><li>This list item is also long enough to be kept.</li></ul>
        </article>
      </body>
    </html>
    """
    documents = _html_documents(payload, "https://example.com/article", "utf-8")

    assert len(documents) == 1
    text = documents[0].page_content
    assert "long enough to be kept" in text
    assert "do not index this" not in text
    assert "Navigation links" not in text
    assert text.count("This paragraph is long enough") == 1
    assert documents[0].metadata["title"] == "Example Article"
