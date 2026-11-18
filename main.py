import argparse
import sys
from pathlib import Path

from chunking import chunk_documents
from loaders import load_file, load_url
from pricing import format_cost
from rag import (
    DEFAULT_PARENT_EXPANSION,
    answer_question,
    delete_thread,
    stream_answer_question,
)
from resilience import friendly_error
from vector_store import RETRIEVAL_MODES, index_source


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def index_target(target: str) -> None:
    """Index a website, YouTube URL, or local file path."""
    if target.startswith(("http://", "https://")):
        loaded = load_url(target)
    else:
        path = Path(target)
        if not path.is_file():
            raise FileNotFoundError(target)
        loaded = load_file(path.name, path.read_bytes())

    chunks = chunk_documents(loaded.documents, loaded.source_id)
    count = index_source(loaded, chunks)
    print(f"Indexed {loaded.source_name}: {count} chunks.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat across indexed sources.")
    parser.add_argument(
        "sources",
        nargs="+",
        help="Website/YouTube URLs and/or file paths (PDF, Excel, CSV, Word, text, images).",
    )
    parser.add_argument("--question", help="Ask one question and exit")
    parser.add_argument(
        "--thread",
        default="cli",
        help="Conversation id for persistent memory (default: cli).",
    )
    parser.add_argument(
        "--mode",
        choices=RETRIEVAL_MODES,
        default="hybrid_rerank",
        help="Retrieval mode (default: hybrid_rerank).",
    )
    parser.add_argument("--k", type=int, default=4, help="Chunks to retrieve (default: 4).")
    parser.add_argument(
        "--source-type",
        nargs="+",
        help="Only retrieve from these source types (pdf, website, youtube, ...).",
    )
    parser.add_argument(
        "--multi-query",
        action="store_true",
        help="Generate paraphrases/sub-questions and fuse their retrieval results.",
    )
    parent_group = parser.add_mutually_exclusive_group()
    parent_group.add_argument(
        "--parent-expansion",
        dest="parent_expansion",
        action="store_true",
        default=None,
        help="Send parent windows to the generator (default comes from .env).",
    )
    parent_group.add_argument(
        "--no-parent-expansion",
        dest="parent_expansion",
        action="store_false",
        help="Send child chunks to the generator instead of parent windows.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream answer tokens and print time-to-first-token.",
    )
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Delete this thread's checkpoint before starting.",
    )
    args = parser.parse_args()
    parent_expansion = (
        DEFAULT_PARENT_EXPANSION
        if args.parent_expansion is None
        else args.parent_expansion
    )

    for target in args.sources:
        index_target(target)

    if args.reset_memory:
        delete_thread(args.thread)
        print(f"Cleared memory for thread {args.thread!r}.")

    def ask(question: str) -> None:
        try:
            if args.stream:
                handle = stream_answer_question(
                    question,
                    thread_id=args.thread,
                    mode=args.mode,
                    k=args.k,
                    source_types=args.source_type,
                    multi_query=args.multi_query,
                    parent_expansion=parent_expansion,
                )
                for token in handle:
                    print(token, end="", flush=True)
                print()
                meta = handle.result.meta
                answer = handle.result.answer
                if "output_blocked" in (meta.get("guardrail_flags") or []):
                    print(
                        "[The streamed text above was blocked by the output guardrail. "
                        f"Safe answer: {answer}]"
                    )
            else:
                answer, _, meta = answer_question(
                    question,
                    thread_id=args.thread,
                    mode=args.mode,
                    k=args.k,
                    source_types=args.source_type,
                    multi_query=args.multi_query,
                    parent_expansion=parent_expansion,
                    with_meta=True,
                )
                print(f"\n{answer}")
        except Exception as error:
            print(f"\n[failed] {friendly_error(error)}")
            return

        usage = meta.get("usage") or {}
        parts = [
            args.mode,
            f"k={args.k}",
            f"{meta['timings'].get('total', 0):.2f}s total",
            f"{meta.get('attempts', 0)} retrieval attempt(s)",
            f"{usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out",
            f"{format_cost(meta.get('cost_usd'))} estimated",
        ]
        if meta.get("ttft_s") is not None:
            parts.insert(3, f"TTFT {meta['ttft_s']:.2f}s")
        if args.multi_query:
            parts.append("multi-query")
        if parent_expansion:
            parts.append("parent windows")
        print(f"[{', '.join(parts)}]")
        if meta.get("guardrail_flags"):
            print(f"[guardrail: {', '.join(meta['guardrail_flags'])}]")
        if not answer:
            print("[empty answer]")

    if args.question:
        ask(args.question)
        return

    print(
        "Sources indexed. Ask questions, or type 'exit' to quit.\n"
        f"Memory is persisted on thread {args.thread!r}."
    )
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        ask(question)


if __name__ == "__main__":
    main()
