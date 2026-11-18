"""Prepare a public QA dataset for a full RAGAS run.

The importer indexes the dataset's own context passages into a separate Chroma
directory, then writes ``eval/dataset.public.jsonl`` with gold child-chunk ids.
That gives you a reproducible corpus instead of mixing benchmark questions with
unrelated local documents.

Sources:
  * ``amnesty``  -- explodinggradients/amnesty_qa (english_v2); human reference
  * ``rag12000`` -- neural-bridge/rag-dataset-12000; one context per question
  * ``ragbench`` -- rungalileo/ragbench (choose a config); model-generated
                   reference responses, so treat answer correctness cautiously

Usage:
    python eval/import_public_dataset.py --source amnesty --limit 50 --reset
    python eval/run_eval.py --dataset eval/dataset.public.jsonl \
        --persist-directory chroma_db_public

Notes:
  * ``--reset`` clears the target Chroma directory first. Use a dedicated
    directory such as ``chroma_db_public``; do not point it at your app corpus.
  * IR metrics are only meaningful when the dataset provides gold context ids.
    Amnesty and rag12000 do; ragbench's sentence ids are not chunk ids, so only
    RAGAS metrics are meaningful there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document  # noqa: E402

from chunking import chunk_documents  # noqa: E402
from loaders import LoadedSource  # noqa: E402
from vector_store import DEFAULT_DB_DIR, delete_all_sources, index_source  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)


class CorpusIndexer:
    """Index unique context strings and remember their child chunk ids."""

    def __init__(self, persist_directory: Path, source_type: str, reset: bool):
        self.persist_directory = persist_directory
        self.source_type = source_type
        self.by_text: dict[str, dict] = {}
        if reset:
            delete_all_sources(persist_directory)
            print(f"Cleared {persist_directory}")

    def add(self, text: str, label: str) -> dict:
        value = " ".join((text or "").split())
        if not value:
            return {"source_id": None, "chunk_ids": []}
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
        if digest in self.by_text:
            return self.by_text[digest]

        source_id = f"public:{digest[:16]}"
        loaded = LoadedSource(
            source_id=source_id,
            source_name=f"{self.source_type}: {label}"[:120],
            source_type=self.source_type,
            documents=[
                Document(
                    page_content=value,
                    metadata={"locator": label, "public_dataset": self.source_type},
                )
            ],
        )
        chunks = chunk_documents(loaded.documents, source_id)
        index_source(loaded, chunks, persist_directory=self.persist_directory)
        record = {
            "source_id": source_id,
            "chunk_ids": [
                f"{source_id}:{chunk.metadata['chunk_index']}" for chunk in chunks
            ],
        }
        self.by_text[digest] = record
        return record


def _unique(values) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _load_amnesty(limit: int, indexer: CorpusIndexer) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("explodinggradients/amnesty_qa", "english_v2", split="eval")
    count = min(limit, len(dataset))
    rows: list[dict] = []
    for index in range(count):
        item = dataset[index]
        contexts = [str(context) for context in item.get("contexts") or []]
        indexed = [
            indexer.add(context, f"context {index + 1}.{offset + 1}")
            for offset, context in enumerate(contexts)
        ]
        gold = _unique(
            chunk_id for record in indexed for chunk_id in record["chunk_ids"]
        )
        rows.append(
            {
                "question": str(item["question"]),
                "ground_truth": str(item["ground_truth"]),
                "ground_truth_chunk_ids": gold,
                "source_ids": None,
                "source_types": [indexer.source_type],
            }
        )
    return rows


def _load_rag12000(limit: int, indexer: CorpusIndexer, split: str) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("neural-bridge/rag-dataset-12000", split=split)
    count = min(limit, len(dataset))
    rows: list[dict] = []
    for index in range(count):
        item = dataset[index]
        record = indexer.add(str(item["context"]), f"context {index + 1}")
        rows.append(
            {
                "question": str(item["question"]),
                "ground_truth": str(item["answer"]),
                "ground_truth_chunk_ids": record["chunk_ids"],
                "source_ids": None,
                "source_types": [indexer.source_type],
            }
        )
    return rows


def _load_ragbench(
    limit: int,
    indexer: CorpusIndexer,
    split: str,
    config: str,
) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("rungalileo/ragbench", config, split=split)
    count = min(limit, len(dataset))
    rows: list[dict] = []
    for index in range(count):
        item = dataset[index]
        documents = [str(document) for document in item.get("documents") or []]
        indexed = [
            indexer.add(document, f"document {index + 1}.{offset + 1}")
            for offset, document in enumerate(documents)
        ]
        gold = _unique(
            chunk_id for record in indexed for chunk_id in record["chunk_ids"]
        )
        reference = str(item.get("response") or item.get("answer") or "").strip()
        if not reference:
            continue
        rows.append(
            {
                "question": str(item["question"]),
                "ground_truth": reference,
                "ground_truth_chunk_ids": gold,
                "source_ids": None,
                "source_types": [indexer.source_type],
                "reference_kind": "dataset_response",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a public QA corpus for RAGAS.")
    parser.add_argument(
        "--source",
        choices=("amnesty", "rag12000", "ragbench"),
        default="amnesty",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--split", default=None)
    parser.add_argument("--config", default="hotpotqa", help="RAGBench config.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "eval" / "dataset.public.jsonl"),
    )
    parser.add_argument(
        "--persist-directory",
        default=str(PROJECT_ROOT / "chroma_db_public"),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the target Chroma directory before indexing.",
    )
    args = parser.parse_args()

    persist_directory = Path(args.persist_directory)
    if persist_directory.resolve() == Path(DEFAULT_DB_DIR).resolve() and args.reset:
        raise SystemExit(
            "Refusing to reset the app's default chroma_db. Use a dedicated directory."
        )

    source_type = args.source
    indexer = CorpusIndexer(persist_directory, source_type, args.reset)

    if args.source == "amnesty":
        rows = _load_amnesty(args.limit, indexer)
    elif args.source == "rag12000":
        rows = _load_rag12000(args.limit, indexer, args.split or "test")
    else:
        rows = _load_ragbench(
            args.limit,
            indexer,
            args.split or "test",
            args.config,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Indexed {len(indexer.by_text)} unique contexts from {args.source}.")
    print(f"Wrote {len(rows)} questions to {output}")
    print(
        "\nRun the evaluation with:\n"
        f"  python eval/run_eval.py --dataset {output} "
        f"--persist-directory {persist_directory}"
    )
    if source_type == "ragbench":
        print(
            "\nNote: RAGBench references are model-generated responses. Use RAGAS "
            "for answer quality, not IR metrics (its sentence ids are not chunk ids)."
        )


if __name__ == "__main__":
    main()
