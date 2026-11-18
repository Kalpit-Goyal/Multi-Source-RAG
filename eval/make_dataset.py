"""Draft an evaluation dataset from indexed chunks using the Groq model.

This is a *starter* generator: every question and reference answer must be
reviewed before it is trusted as ground truth. It intentionally skips contact
details when writing questions.

Usage:
    python eval/make_dataset.py --limit 20
    python eval/make_dataset.py --limit 20 --output eval/dataset.draft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq  # noqa: E402

from vector_store import get_vector_store  # noqa: E402


PROMPT = """You write evaluation questions for a retrieval-augmented chatbot.
Read the chunk below and write ONE specific question that is answerable only
from this chunk, plus a short reference answer.

Rules:
- Do not ask for email addresses, phone numbers, home addresses, or other contact details.
- The question must not mention the chunk number or the source file name.
- The reference answer must be supported by the chunk.
- Reply with JSON only, no markdown: {{"question": "...", "answer": "..."}}

Chunk:
{chunk}
"""


def parse_json(text: str) -> dict | None:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    question = str(value.get("question") or "").strip()
    answer = str(value.get("answer") or "").strip()
    if not question or not answer:
        return None
    return {"question": question, "answer": answer}


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft eval questions from indexed chunks.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "eval" / "dataset.draft.jsonl"))
    parser.add_argument(
        "--model",
        default=os.getenv("RAGAS_JUDGE_MODEL") or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    )
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    stored = get_vector_store().get(include=["documents", "metadatas"])
    rows = list(zip(stored.get("documents") or [], stored.get("metadatas") or []))
    if not rows:
        raise SystemExit("The knowledge base is empty. Index a source first.")

    llm = ChatGroq(model=args.model, temperature=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for text, metadata in rows:
            if written >= args.limit:
                break
            source_id = metadata.get("source_id")
            chunk_index = metadata.get("chunk_index")
            if not source_id or chunk_index is None or not (text or "").strip():
                continue

            response = llm.invoke(PROMPT.format(chunk=text[:4000]))
            parsed = parse_json(str(response.content))
            if parsed is None or parsed["question"] in seen:
                continue
            seen.add(parsed["question"])
            row = {
                "question": parsed["question"],
                "ground_truth": parsed["answer"],
                "ground_truth_chunk_ids": [f"{source_id}:{chunk_index}"],
                "source_ids": [source_id],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            print(f"[{written}/{args.limit}] {parsed['question']}")
            if args.sleep:
                time.sleep(args.sleep)

    print(f"\nWrote {written} draft rows to {output}")
    print("Review and edit them before using the file with run_eval.py.")


if __name__ == "__main__":
    main()
