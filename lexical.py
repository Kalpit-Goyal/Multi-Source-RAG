"""Lexical (BM25) retrieval over the same chunks stored in Chroma."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


#: ``\w+`` keeps numbers, model codes, and acronyms as single tokens.
TOKEN_PATTERN = re.compile(r"\w+")
#: ``1,234`` should match a query for ``1234``; stripping the separator keeps them equal.
DIGIT_GROUP_COMMA = re.compile(r"(?<=\d),(?=\d{3}\b)")
#: Hyphenated codes such as ``X-200`` also get a joined token (``x200``) so both spellings match.
COMPOUND_PATTERN = re.compile(r"[a-z0-9]*[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)+")


def tokenize(text: str) -> list[str]:
    """Split text into the lowercase tokens BM25 matches on."""
    normalized = DIGIT_GROUP_COMMA.sub("", (text or "").lower())
    tokens = TOKEN_PATTERN.findall(normalized)
    tokens.extend(
        re.sub(r"[-_.]", "", compound)
        for compound in COMPOUND_PATTERN.findall(normalized)
    )
    return tokens


def chunk_key(document: Document) -> str:
    """Return the id shared by the vector and lexical arms during fusion."""
    metadata = document.metadata or {}
    source_id = metadata.get("source_id")
    chunk_index = metadata.get("chunk_index")
    if source_id is not None and chunk_index is not None:
        return f"{source_id}:{chunk_index}"
    digest = hashlib.sha1(
        document.page_content.encode("utf-8", "ignore")
    ).hexdigest()
    return f"content:{digest}"


@dataclass
class LexicalIndex:
    """A BM25 index over the corpus, with optional source scoping."""

    documents: list[Document]
    engine: BM25Okapi | None

    def search(
        self,
        query: str,
        k: int,
        allowed: set[str] | None = None,
        predicate: Callable[[Document], bool] | None = None,
    ) -> list[Document]:
        """Return up to ``k`` positive-scoring chunks, best first.

        ``allowed`` scopes by source id. ``predicate`` applies arbitrary
        metadata filters (for example source type) without rebuilding BM25.
        """
        if self.engine is None or k <= 0:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.engine.get_scores(tokens)
        hits: list[Document] = []
        for position in scores.argsort()[::-1]:
            if len(hits) >= k or scores[position] <= 0:
                break
            document = self.documents[int(position)]
            if (
                allowed is not None
                and document.metadata.get("source_id") not in allowed
            ):
                continue
            if predicate is not None and not predicate(document):
                continue
            hits.append(document)
        return hits


def build_index(documents: list[Document]) -> LexicalIndex:
    """Build a BM25 index; an empty or untokenizable corpus yields no engine."""
    if not documents:
        return LexicalIndex([], None)

    corpus = [tokenize(document.page_content) for document in documents]
    if not any(corpus):
        # Every chunk was punctuation/whitespace; BM25 would divide by zero.
        return LexicalIndex(list(documents), None)

    return LexicalIndex(list(documents), BM25Okapi(corpus))
