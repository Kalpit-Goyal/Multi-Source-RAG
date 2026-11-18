"""Extract text from websites, YouTube videos, documents, sheets, and images."""

from __future__ import annotations

import csv
import hashlib
import io
import ipaddress
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from transcript import get_transcript, get_video_id


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SHEET_SUFFIXES = {".xlsx", ".xlsm"}
TEXT_SUFFIXES = {".txt", ".md"}
SUPPORTED_SUFFIXES = (
    {".pdf", ".csv", ".docx"} | IMAGE_SUFFIXES | SHEET_SUFFIXES | TEXT_SUFFIXES
)
ROWS_PER_BLOCK = 25
BLOCK_CHARS = 1500

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}
MAX_WEBSITE_BYTES = int(os.getenv("MAX_WEBSITE_BYTES", "5000000"))
MAX_WEBSITE_CHARS = int(os.getenv("MAX_WEBSITE_CHARS", "400000"))
TRACKING_PARAMS = {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid"}
USER_AGENT = os.getenv(
    "WEBSITE_USER_AGENT",
    "Mozilla/5.0 (compatible; MultiSourceRAG/1.0; +https://example.com/bot)",
)


@dataclass
class LoadedSource:
    source_id: str
    source_name: str
    source_type: str
    documents: list[Document] = field(default_factory=list)


def _source_type(suffix: str) -> str:
    if suffix == ".pdf":
        return "pdf"
    if suffix in SHEET_SUFFIXES:
        return "spreadsheet"
    if suffix == ".csv":
        return "csv"
    if suffix == ".docx":
        return "document"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return "text"


def load_youtube(url: str) -> LoadedSource:
    """Fetch a YouTube transcript and wrap it as a source."""
    video_id = get_video_id(url)
    text = get_transcript(video_id)
    if not text.strip():
        raise ValueError(f"No transcript found for {url}")
    return LoadedSource(
        source_id=f"yt:{video_id}",
        source_name=f"YouTube: {video_id}",
        source_type="youtube",
        documents=[Document(page_content=text, metadata={"locator": "transcript"})],
    )


def is_youtube_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in YOUTUBE_HOSTS


def load_url(url: str) -> LoadedSource:
    """Load a YouTube transcript or a public web page."""
    return load_youtube(url) if is_youtube_url(url) else load_website(url)


def _normalize_url(url: str) -> str:
    """Canonicalize a URL so the same page is not indexed twice."""
    candidate = (url or "").strip()
    if not candidate:
        raise ValueError("Empty URL")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme or 'missing'}")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError(f"Invalid URL: {url}")

    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise ValueError(f"Refusing to fetch a private host: {host}")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
    ):
        raise ValueError(f"Refusing to fetch a private address: {host}")

    if host.startswith("www."):
        host = host[4:]
    port = parsed.port
    if port and not (
        (parsed.scheme == "http" and port == 80)
        or (parsed.scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"

    query = sorted(
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in TRACKING_PARAMS
    )
    path = parsed.path or "/"
    if len(path) > 1:
        path = path.rstrip("/")
    return urlunparse(
        (
            parsed.scheme.lower(),
            host,
            path,
            "",
            urlencode(query),
            "",
        )
    )


def _fetch(url: str) -> tuple[bytes, str, str, str | None]:
    """Download a page with a size cap. Returns bytes, content type, final URL, encoding."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
    }
    with requests.get(
        url,
        headers=headers,
        timeout=20,
        allow_redirects=True,
        stream=True,
    ) as response:
        response.raise_for_status()
        content_type = (
            response.headers.get("content-type", "").split(";")[0].strip().lower()
        )
        payload = bytearray()
        for piece in response.iter_content(chunk_size=65536):
            if not piece:
                continue
            payload.extend(piece)
            if len(payload) > MAX_WEBSITE_BYTES:
                raise ValueError(
                    f"Website is larger than {MAX_WEBSITE_BYTES:,} bytes: {url}"
                )
        return (
            bytes(payload),
            content_type,
            response.url,
            response.encoding,
        )


def _html_documents(payload: bytes, url: str, encoding: str | None) -> list[Document]:
    """Extract readable prose from an HTML page, dropping navigation and scripts."""
    if not encoding:
        try:
            from charset_normalizer import from_bytes

            best = from_bytes(payload).best()
            encoding = best.encoding if best else "utf-8"
        except Exception:
            encoding = "utf-8"
    html = payload.decode(encoding or "utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "form",
            "nav",
            "footer",
            "header",
            "aside",
            "iframe",
            "button",
        ]
    ):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = " ".join(soup.title.string.split())
    if not title:
        heading = soup.find("h1")
        title = " ".join(heading.get_text(" ", strip=True).split()) if heading else ""

    root = soup.find("article") or soup.find("main") or soup.body or soup
    blocks: list[str] = []
    seen: set[str] = set()
    for element in root.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "pre", "blockquote", "td", "th"]
    ):
        text = " ".join(element.get_text(" ", strip=True).split())
        if len(text) < 25:
            continue
        key = text.lower()
        if key in seen:
            continue
        # Nested markup (a <p> inside <li>) would otherwise repeat the text.
        if blocks and key in blocks[-1].lower():
            continue
        seen.add(key)
        blocks.append(text)

    if not blocks:
        fallback = " ".join(root.get_text(" ", strip=True).split())
        if fallback:
            blocks = [fallback]

    text = "\n\n".join(blocks).strip()
    if len(text) > MAX_WEBSITE_CHARS:
        text = text[:MAX_WEBSITE_CHARS].rsplit(" ", 1)[0]
    if not text:
        return []

    host = urlparse(url).netloc
    return [
        Document(
            page_content=text,
            metadata={
                "locator": url,
                "title": title or host,
                "url": url,
                "host": host,
            },
        )
    ]


def load_website(url: str) -> LoadedSource:
    """Fetch a public web page (or a PDF served over HTTP) as a source."""
    normalized = _normalize_url(url)
    # Fetch the original URL (some sites only serve on ``www``) but key the
    # source by its canonical form so duplicates collapse.
    payload, content_type, final_url, encoding = _fetch(url.strip())
    # Re-validate after redirects so a public URL cannot bounce to a private host.
    _normalize_url(final_url)

    if content_type == "application/pdf" or final_url.lower().endswith(".pdf"):
        documents = _pdf_documents(payload)
        source_type = "pdf"
    else:
        html_types = {
            "text/html",
            "application/xhtml+xml",
            "text/plain",
            "",
        }
        if content_type not in html_types:
            raise ValueError(f"Unsupported website content type: {content_type}")
        documents = _html_documents(payload, final_url, encoding)
        source_type = "website"

    if not documents:
        raise ValueError(f"No readable text found at {url}")

    title = (documents[0].metadata.get("title") or "").strip()
    host = urlparse(final_url).netloc
    source_name = title or host
    if len(source_name) > 90:
        source_name = source_name[:87].rstrip() + "..."

    return LoadedSource(
        source_id="web:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16],
        source_name=source_name,
        source_type=source_type,
        documents=documents,
    )


def load_file(filename: str, data: bytes) -> LoadedSource:
    """Extract text from an uploaded file."""
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

    if suffix == ".pdf":
        documents = _pdf_documents(data)
    elif suffix in SHEET_SUFFIXES:
        documents = _excel_documents(data)
    elif suffix == ".csv":
        documents = _csv_documents(data)
    elif suffix == ".docx":
        documents = _docx_documents(data)
    elif suffix in IMAGE_SUFFIXES:
        documents = _image_documents(data)
    else:
        documents = _text_documents(data)

    if not documents:
        raise ValueError(f"No readable text found in {name}")

    return LoadedSource(
        source_id="file:" + hashlib.sha256(data).hexdigest()[:16],
        source_name=name,
        source_type=_source_type(suffix),
        documents=documents,
    )


def _pdf_documents(data: bytes) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    documents = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(
                Document(page_content=text, metadata={"locator": f"page {number}"})
            )
    return documents


def _table_documents(
    header: list | None,
    rows,
    locator: str,
) -> list[Document]:
    """Turn spreadsheet/CSV rows into text blocks that embed well."""
    if header is None:
        return []

    columns = [
        str(value).strip() if value is not None and str(value).strip() else f"column {index + 1}"
        for index, value in enumerate(header)
    ]

    documents: list[Document] = []
    block: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        cells = [
            f"{columns[index]}: {value}"
            for index, value in enumerate(row)
            if index < len(columns) and value is not None and str(value).strip()
        ]
        if not cells:
            continue
        block.append(f"Row {row_number}: " + " | ".join(cells))
        if len(block) >= ROWS_PER_BLOCK or sum(map(len, block)) > BLOCK_CHARS:
            documents.append(
                Document(page_content="\n".join(block), metadata={"locator": locator})
            )
            block = []

    if block:
        documents.append(
            Document(page_content="\n".join(block), metadata={"locator": locator})
        )
    return documents


def _excel_documents(data: bytes) -> list[Document]:
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    documents: list[Document] = []
    try:
        for sheet in workbook.worksheets:
            rows = sheet.iter_rows(values_only=True)
            header = next(rows, None)
            documents.extend(
                _table_documents(header, rows, f"sheet {sheet.title}")
            )
    finally:
        workbook.close()
    return documents


def _csv_documents(data: bytes) -> list[Document]:
    text = _decode(data)
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    return _table_documents(header, reader, "rows")


def _docx_documents(data: bytes) -> list[Document]:
    from docx import Document as DocxDocument

    document = DocxDocument(io.BytesIO(data))
    documents: list[Document] = []

    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    if paragraphs:
        documents.append(
            Document(
                page_content="\n\n".join(paragraphs),
                metadata={"locator": "text"},
            )
        )

    for index, table in enumerate(document.tables, start=1):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if not rows:
            continue
        documents.extend(
            _table_documents(rows[0], rows[1:], f"table {index}")
        )
    return documents


def _text_documents(data: bytes) -> list[Document]:
    text = _decode(data).strip()
    return [Document(page_content=text, metadata={"locator": "text"})] if text else []


@lru_cache(maxsize=1)
def _ocr_engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _image_documents(data: bytes) -> list[Document]:
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGB")
    result, _ = _ocr_engine()(np.array(image))
    text = "\n".join(line[1] for line in (result or [])).strip()
    if not text:
        return []
    return [Document(page_content=text, metadata={"locator": "OCR text"})]


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")
