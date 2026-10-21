"""Extract text from YouTube videos, documents, spreadsheets, and images."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

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
