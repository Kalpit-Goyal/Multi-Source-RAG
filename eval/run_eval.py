"""Evaluate the RAG pipeline over vector, hybrid, and hybrid+rerank modes.

Produces:
  * per-question answers, retrieved chunks, IR metrics, timings, TTFT, usage,
    guardrail flags, and estimated USD cost (JSONL)
  * RAGAS scores per mode (CSV)
  * an aggregate summary (JSON + CSV)

Usage:
    python eval/run_eval.py --limit 10
    python eval/run_eval.py --modes hybrid vector --no-ragas
    python eval/run_eval.py --dataset eval/dataset.jsonl --k 4
    python eval/run_eval.py --stream --multi-query --source-type pdf
    python eval/run_eval.py --retrieval-only
    python eval/run_eval.py --from-records eval/results/<run>/per_question.jsonl \
        --modes hybrid_rerank --metrics faithfulness context_recall
    python eval/run_eval.py --from-records eval/results/<run>/per_question.jsonl \
        --modes vector hybrid_rerank --metrics context_precision context_recall \
        --resume --output-dir eval/results/<run>/ragas-resume

RAGAS needs a judge model. It defaults to GROQ_MODEL and uses the same Groq
key as the app. On a free tier, run --retrieval-only for zero-Groq IR metrics,
then --no-ragas for a small end-to-end sample, then --from-records with a
limited --metrics list. Set RAGAS_JUDGE_RETRIES=0 to fail fast on quota errors.

``--resume`` scores one question/metric at a time and appends each success to
``<output-dir>/ragas_cache.jsonl`` before moving on. If the quota runs out,
re-run the *same* command later: cached scores are reused and only missing ones
are requested. ``<output-dir>/ragas_failures.jsonl`` records what failed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

# Must run before ragas is imported: ragas 0.4.3 imports a module that
# langchain-community 0.4 removed. See eval/_ragas_compat.py.
from eval import _ragas_compat  # noqa: E402,F401

from rag import (  # noqa: E402
    DEFAULT_PARENT_EXPANSION,
    answer_question,
    delete_thread,
    stream_answer_question,
)
from vector_store import (  # noqa: E402
    DEFAULT_DB_DIR,
    RETRIEVAL_MODES,
    expand_to_parents,
    get_embeddings,
    search_transcript,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULT_RAGAS_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "semantic_similarity",
)


def load_dataset(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}") from error
            missing = {"question", "ground_truth"} - set(row)
            if missing:
                raise SystemExit(
                    f"{path}:{line_number}: missing keys {sorted(missing)}"
                )
            row.setdefault("ground_truth_chunk_ids", [])
            row.setdefault("source_ids", None)
            row.setdefault("source_types", None)
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    return rows


def _mean(values) -> float | None:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return sum(clean) / len(clean) if clean else None


def _percentile(values, percentile: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1 - weight) + clean[upper] * weight


def ranking_metrics(retrieved: list[str], gold: list[str], k: int) -> dict:
    """Hit rate, recall, precision, MRR, and NDCG with binary relevance."""
    gold_set = set(gold)
    top = retrieved[:k]
    if not gold_set:
        return {
            "hit_rate": None,
            "recall": None,
            "precision": None,
            "mrr": None,
            "ndcg": None,
        }

    hits = [1 if chunk_id in gold_set else 0 for chunk_id in top]
    hit_rate = 1.0 if any(hits) else 0.0
    recall = len(set(top) & gold_set) / len(gold_set)
    precision = (sum(hits) / len(top)) if top else 0.0

    mrr = 0.0
    for rank, relevant in enumerate(hits, start=1):
        if relevant:
            mrr = 1.0 / rank
            break

    dcg = sum(
        relevant / math.log2(rank + 1)
        for rank, relevant in enumerate(hits, start=1)
    )
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        "hit_rate": round(hit_rate, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "mrr": round(mrr, 4),
        "ndcg": round(ndcg, 4),
    }


def _chunk_ids(documents) -> list[str]:
    """Return child chunk ids, expanding parent windows when present."""
    from lexical import chunk_key

    ids: list[str] = []
    for document in documents:
        child_ids = (document.metadata or {}).get("child_chunk_ids")
        if child_ids:
            ids.extend(str(item) for item in child_ids)
        else:
            ids.append(chunk_key(document))
    return list(dict.fromkeys(ids))


def _effective_source_types(row_types, cli_types):
    """Intersect a per-row filter with the CLI filter; empty means no matches."""
    if row_types and cli_types:
        return [value for value in row_types if value in set(cli_types)]
    return row_types or cli_types


def _contexts(documents) -> list[str]:
    return [document.page_content for document in documents]


def run_questions(
    rows: list[dict],
    modes: list[str],
    k: int,
    sleep_seconds: float,
    *,
    source_types: list[str] | None = None,
    multi_query: bool = False,
    parent_expansion: bool = True,
    stream: bool = False,
    persist_directory=DEFAULT_DB_DIR,
) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {mode: [] for mode in modes}
    total = len(rows) * len(modes)
    completed = 0

    # Interleave modes per question and rotate their order. Running one mode
    # for the whole set lets Groq rate limits bias whichever mode runs last.
    for index, row in enumerate(rows):
        offset = index % len(modes)
        order = modes[offset:] + modes[:offset]
        row_types = _effective_source_types(row.get("source_types"), source_types)
        for mode in order:
            thread_id = f"eval-{mode}-{index}-{int(time.time() * 1000)}"
            try:
                if stream:
                    handle = stream_answer_question(
                        row["question"],
                        thread_id=thread_id,
                        persist_directory=persist_directory,
                        source_ids=row.get("source_ids") or None,
                        source_types=row_types,
                        k=k,
                        mode=mode,
                        multi_query=multi_query,
                        parent_expansion=parent_expansion,
                    )
                    for _token in handle:
                        pass
                    answer = handle.result.answer
                    documents = handle.result.documents
                    meta = handle.result.meta
                else:
                    answer, documents, meta = answer_question(
                        row["question"],
                        thread_id=thread_id,
                        persist_directory=persist_directory,
                        source_ids=row.get("source_ids") or None,
                        source_types=row_types,
                        k=k,
                        mode=mode,
                        multi_query=multi_query,
                        parent_expansion=parent_expansion,
                        with_meta=True,
                    )
            finally:
                delete_thread(thread_id)

            retrieved = meta.get("retrieved_documents", [])
            used = meta.get("used_documents", documents)
            gold = row.get("ground_truth_chunk_ids") or []
            retrieved_ids = _chunk_ids(retrieved)
            used_ids = _chunk_ids(used)

            record = {
                "mode": mode,
                "model": meta.get("model") or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                "k": meta.get("k", k),
                "source_ids": meta.get("source_ids", row.get("source_ids") or None),
                "source_types": meta.get("source_types", row_types),
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "gold_chunk_ids": gold,
                "answer": answer,
                "retrieved_ids": retrieved_ids,
                "used_ids": used_ids,
                "contexts": _contexts(used),
                "retrieved_contexts": _contexts(retrieved),
                "latency_s": meta["timings"].get("total"),
                "ttft_s": meta.get("ttft_s"),
                "timings": meta["timings"],
                "attempts": meta["attempts"],
                "usage": meta.get("usage") or {},
                "cost_usd": meta.get("cost_usd"),
                "guardrail_flags": meta.get("guardrail_flags") or [],
                "context_flags": meta.get("context_flags") or [],
                "blocked": bool(meta.get("blocked")),
                "multi_query": bool(meta.get("multi_query")),
                "parent_expansion": bool(meta.get("parent_expansion")),
                "query_plan": meta.get("query_plan") or [],
                "retrieval_metrics": ranking_metrics(retrieved_ids, gold, k),
                "used_metrics": ranking_metrics(used_ids, gold, k),
            }
            results[mode].append(record)

            completed += 1
            ttft = (
                f", ttft={record['ttft_s']:.2f}s"
                if record["ttft_s"] is not None
                else ""
            )
            print(
                f"[{completed}/{total}] {mode} q{index + 1}: "
                f"{record['latency_s']:.2f}s{ttft}, "
                f"hit={record['used_metrics']['hit_rate']}, "
                f"recall={record['used_metrics']['recall']}, "
                f"cost=${(record['cost_usd'] or 0):.6f}"
            )
            if sleep_seconds:
                time.sleep(sleep_seconds)

    return results


def run_retrieval_questions(
    rows: list[dict],
    modes: list[str],
    k: int,
    *,
    source_types: list[str] | None = None,
    parent_expansion: bool = True,
    persist_directory=DEFAULT_DB_DIR,
) -> dict[str, list[dict]]:
    """Evaluate retrieval only, with no LLM or RAGAS calls.

    This is the zero-quota path: it runs the same dense/BM25/reranker
    retrieval and parent expansion as the graph, but skips condense, expand,
    grade, refine, and generate.
    """
    results: dict[str, list[dict]] = {mode: [] for mode in modes}
    total = len(rows) * len(modes)
    completed = 0

    for index, row in enumerate(rows):
        row_types = _effective_source_types(row.get("source_types"), source_types)
        source_ids = row.get("source_ids") or None
        gold = row.get("ground_truth_chunk_ids") or []
        for mode in modes:
            started = time.perf_counter()
            hits = search_transcript(
                row["question"],
                persist_directory=persist_directory,
                source_ids=source_ids,
                k=k,
                mode=mode,
                source_types=row_types,
            )
            used = expand_to_parents(hits) if parent_expansion else hits
            elapsed = round(time.perf_counter() - started, 4)
            retrieved_ids = _chunk_ids(hits)
            used_ids = _chunk_ids(used)
            record = {
                "mode": mode,
                "model": "retrieval-only",
                "k": k,
                "source_ids": source_ids,
                "source_types": row_types,
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "gold_chunk_ids": gold,
                "answer": "",
                "retrieved_ids": retrieved_ids,
                "used_ids": used_ids,
                "contexts": _contexts(used),
                "retrieved_contexts": _contexts(hits),
                "latency_s": elapsed,
                "ttft_s": None,
                "timings": {"retrieve": elapsed, "total": elapsed},
                "attempts": 1,
                "usage": {},
                "cost_usd": 0.0,
                "guardrail_flags": [],
                "context_flags": [],
                "blocked": False,
                "multi_query": False,
                "parent_expansion": bool(parent_expansion),
                "query_plan": [],
                "retrieval_metrics": ranking_metrics(retrieved_ids, gold, k),
                "used_metrics": ranking_metrics(used_ids, gold, k),
            }
            results[mode].append(record)
            completed += 1
            print(
                f"[{completed}/{total}] {mode} q{index + 1}: "
                f"{elapsed:.3f}s, hit={record['retrieval_metrics']['hit_rate']}, "
                f"recall={record['retrieval_metrics']['recall']}, "
                f"used_hit={record['used_metrics']['hit_rate']}"
            )

    return results


def run_ragas(
    records: list[dict],
    judge_model: str,
    workers: int,
    timeout: int,
    strict: bool,
    metric_names: list[str] | None = None,
):
    """Score one mode with RAGAS. Returns a DataFrame or None."""
    try:
        from langchain_groq import ChatGroq
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerCorrectness,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
            SemanticSimilarity,
        )
        from ragas.run_config import RunConfig
    except Exception as error:  # pragma: no cover - optional dependency issues
        print(f"RAGAS import failed, skipping: {error}")
        return None

    samples = [
        SingleTurnSample(
            user_input=record["question"],
            retrieved_contexts=record["contexts"] or ["No context was retrieved."],
            response=record["answer"] or "(empty answer)",
            reference=record["ground_truth"],
        )
        for record in records
    ]
    dataset = EvaluationDataset(samples=samples)

    builders = {
        "faithfulness": Faithfulness,
        # Groq rejects n > 1, which response relevancy requests by default.
        "answer_relevancy": lambda: ResponseRelevancy(strictness=1),
        "context_precision": LLMContextPrecisionWithReference,
        "context_recall": LLMContextRecall,
        "answer_correctness": AnswerCorrectness,
        "semantic_similarity": SemanticSimilarity,
    }
    selected = list(metric_names or DEFAULT_RAGAS_METRICS)
    unknown = [name for name in selected if name not in builders]
    if unknown:
        print(f"Unknown RAGAS metrics, skipping: {', '.join(unknown)}")
        selected = [name for name in selected if name in builders]
    if not selected:
        print("No RAGAS metrics selected, skipping.")
        return None
    metrics = [builders[name]() for name in selected]

    judge_retries = max(0, int(os.getenv("RAGAS_JUDGE_RETRIES", "4")))
    llm = LangchainLLMWrapper(
        ChatGroq(
            model=judge_model,
            temperature=0,
            max_retries=judge_retries,
            timeout=timeout,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    print(
        f"  RAGAS judge: {judge_model} ({len(samples)} samples; "
        f"metrics: {', '.join(selected)})"
    )
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=strict,
        show_progress=True,
        run_config=RunConfig(max_workers=workers, timeout=timeout),
    )
    frame = result.to_pandas()
    frame.insert(0, "question", [record["question"] for record in records])
    return frame


def run_ragas_resumable(
    records: list[dict],
    mode: str,
    judge_model: str,
    metric_names: list[str],
    timeout: int,
    cache_path: Path,
    failures_path: Path,
    max_failures: int,
    sleep: float,
    strict: bool,
):
    """Score one mode with an append-only cache so runs can span days."""
    from eval.ragas_resume import score_mode

    rows, coverage = score_mode(
        records,
        mode,
        metric_names,
        judge_model,
        timeout,
        max(0, int(os.getenv("RAGAS_JUDGE_RETRIES", "4"))),
        cache_path,
        failures_path,
        max_consecutive_failures=max_failures,
        sleep_s=sleep,
        strict=strict,
    )

    import pandas as pd

    frame = pd.DataFrame(rows, columns=list(metric_names)).astype(float)
    frame.insert(0, "question", [record["question"] for record in records])
    frame.attrs["coverage"] = coverage
    return frame


def _numeric_summary(frame) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    if frame is None:
        return summary
    for column in frame.columns:
        if column in {"question"}:
            continue
        values = frame[column]
        if values.dtype.kind not in {"f", "i"}:
            continue
        summary[column] = _mean(values)
    return summary


def _sanitize(value):
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval modes with IR metrics and RAGAS.")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "eval" / "dataset.jsonl"))
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=RETRIEVAL_MODES,
        default=list(RETRIEVAL_MODES),
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions.")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N questions; useful for resuming a free-tier run.",
    )
    parser.add_argument("--no-ragas", action="store_true", help="Skip judge-based metrics.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=DEFAULT_RAGAS_METRICS,
        default=list(DEFAULT_RAGAS_METRICS),
        help=(
            "RAGAS metrics to compute (default: all). Fewer metrics save free-tier "
            "judge quota; e.g. --metrics faithfulness context_recall."
        ),
    )
    parser.add_argument("--strict", action="store_true", help="Raise RAGAS metric errors.")
    parser.add_argument(
        "--judge-model",
        default=os.getenv("RAGAS_JUDGE_MODEL") or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between questions.")
    parser.add_argument(
        "--source-type",
        nargs="+",
        help="Only retrieve from these source types (pdf, website, youtube, ...).",
    )
    parser.add_argument(
        "--multi-query",
        action="store_true",
        help="Use paraphrases/sub-questions and fuse their retrieval results.",
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
        help="Consume the streaming path and record TTFT (slower for large batches).",
    )
    parser.add_argument(
        "--persist-directory",
        default=str(DEFAULT_DB_DIR),
        help="Chroma directory to evaluate (default: chroma_db).",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--from-records",
        help=(
            "Reuse a per_question.jsonl from a previous run and only compute "
            "RAGAS/summary. Skips retrieval, generation, and warm-up."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Score RAGAS one question/metric at a time and append each score to "
            "<output-dir>/ragas_cache.jsonl immediately, so a later run continues "
            "instead of restarting. Recommended for free-tier judge quotas."
        ),
    )
    parser.add_argument(
        "--cache-path",
        help=(
            "Override the RAGAS resume cache path "
            "(default: <output-dir>/ragas_cache.jsonl)."
        ),
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help=(
            "With --resume, stop a mode after N consecutive judge failures "
            "(default: 3). Cached scores are kept."
        ),
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "Run dense/BM25/reranker retrieval and IR metrics only. Makes no "
            "LLM or RAGAS calls, so it is safe on a free tier."
        ),
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    parent_expansion = (
        DEFAULT_PARENT_EXPANSION
        if args.parent_expansion is None
        else args.parent_expansion
    )
    if args.retrieval_only and args.from_records:
        raise SystemExit("--retrieval-only cannot be combined with --from-records.")
    if args.retrieval_only and args.multi_query:
        raise SystemExit("--retrieval-only cannot use --multi-query (that needs an LLM).")

    started = datetime.now().strftime("%Y%m%d-%H%M%S")
    reused = bool(args.from_records)

    if reused:
        records_path = Path(args.from_records)
        if not records_path.is_file():
            raise SystemExit(f"Records not found: {records_path}")
        requested = set(args.modes)
        results: dict[str, list[dict]] = {}
        with records_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                mode = record.get("mode")
                if mode in requested:
                    results.setdefault(mode, []).append(record)
        if not results:
            raise SystemExit(
                f"No records for modes {sorted(requested)} in {records_path}"
            )
        modes = list(results)
        rows = []
        n_questions = len({record["question"] for record in next(iter(results.values()))})
        dataset_label = f"reused:{records_path}"
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else PROJECT_ROOT / "eval" / "results" / f"ragas-{started}"
        )
        first = next(iter(results.values()))[0] if next(iter(results.values())) else {}
    else:
        dataset_path = Path(args.dataset)
        if not dataset_path.is_file():
            raise SystemExit(
                f"Dataset not found: {dataset_path}\n"
                "Create eval/dataset.jsonl or run: python eval/make_dataset.py"
            )
        rows = load_dataset(dataset_path)
        if args.offset < 0:
            raise SystemExit("--offset must be zero or greater.")
        if args.offset:
            rows = rows[args.offset :]
        if args.limit:
            rows = rows[: args.limit]

        if not args.skip_warmup:
            print("Warming up embedding, BM25, and reranker models...")
            for mode in args.modes:
                search_transcript(
                    "warm up",
                    persist_directory=args.persist_directory,
                    k=args.k,
                    mode=mode,
                    source_types=args.source_type,
                )

        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else PROJECT_ROOT / "eval" / "results" / started
        )
        if args.retrieval_only:
            results = run_retrieval_questions(
                rows,
                args.modes,
                args.k,
                source_types=args.source_type,
                parent_expansion=parent_expansion,
                persist_directory=args.persist_directory,
            )
        else:
            results = run_questions(
                rows,
                args.modes,
                args.k,
                args.sleep,
                source_types=args.source_type,
                multi_query=args.multi_query,
                parent_expansion=parent_expansion,
                stream=args.stream,
                persist_directory=args.persist_directory,
            )
        modes = list(results)
        n_questions = len(rows)
        dataset_label = str(dataset_path)
        first = {}

    output_dir.mkdir(parents=True, exist_ok=True)

    per_question_path = output_dir / "per_question.jsonl"
    with per_question_path.open("w", encoding="utf-8") as handle:
        for records in results.values():
            for record in records:
                handle.write(json.dumps(_sanitize(record), ensure_ascii=False) + "\n")

    default_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    summary: dict = {
        "dataset": dataset_label,
        "records_source": str(args.from_records) if reused else None,
        "k": first.get("k", args.k) if reused else args.k,
        "modes": modes,
        "n_questions": n_questions,
        "generated_at": started,
        "model": (
            "retrieval-only"
            if args.retrieval_only
            else (first.get("model") or default_model) if reused else default_model
        ),
        "judge_model": args.judge_model,
        "source_types": (
            first.get("source_types", args.source_type) if reused else args.source_type
        ),
        "multi_query": (
            False
            if args.retrieval_only
            else first.get("multi_query", args.multi_query) if reused else args.multi_query
        ),
        "parent_expansion": (
            first.get("parent_expansion", parent_expansion)
            if reused
            else parent_expansion
        ),
        "stream": (
            False
            if args.retrieval_only
            else any(record.get("ttft_s") is not None for record in next(iter(results.values())))
            if reused
            else args.stream
        ),
        "persist_directory": (
            "reused-records" if reused else str(Path(args.persist_directory))
        ),
        "per_mode": {},
    }

    for mode, records in results.items():
        latencies = [record["latency_s"] for record in records]
        ttfts = [
            record["ttft_s"]
            for record in records
            if record.get("ttft_s") is not None
        ]
        costs = [
            record["cost_usd"]
            for record in records
            if record.get("cost_usd") is not None
        ]
        usage_totals = {
            key: sum(int((record.get("usage") or {}).get(key, 0)) for record in records)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        ragas_frame = None
        ragas_coverage = None
        if not args.no_ragas and not args.retrieval_only:
            print(f"\n=== RAGAS: {mode} ===")
            if args.resume:
                cache_path = (
                    Path(args.cache_path)
                    if args.cache_path
                    else output_dir / "ragas_cache.jsonl"
                )
                ragas_frame = run_ragas_resumable(
                    records,
                    mode,
                    args.judge_model,
                    args.metrics,
                    args.timeout,
                    cache_path,
                    cache_path.parent / "ragas_failures.jsonl",
                    args.max_failures,
                    args.sleep,
                    args.strict,
                )
                ragas_coverage = ragas_frame.attrs.get("coverage")
            else:
                ragas_frame = run_ragas(
                    records,
                    args.judge_model,
                    args.workers,
                    args.timeout,
                    args.strict,
                    args.metrics,
                )
            if ragas_frame is not None:
                ragas_frame.to_csv(output_dir / f"ragas_{mode}.csv", index=False)

        retrieval_metrics = {
            key: _mean(record["retrieval_metrics"][key] for record in records)
            for key in ("hit_rate", "recall", "precision", "mrr", "ndcg")
        }
        used_metrics = {
            key: _mean(record["used_metrics"][key] for record in records)
            for key in ("hit_rate", "recall", "precision", "mrr", "ndcg")
        }
        node_names = ("condense", "expand", "retrieve", "grade", "refine", "generate")
        node_latency = {
            node: _mean(record["timings"].get(node) for record in records)
            for node in node_names
        }

        summary["per_mode"][mode] = {
            "n": len(records),
            "retrieval_ir": retrieval_metrics,
            "used_ir": used_metrics,
            "latency_s": {
                "mean": _mean(latencies),
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "p99": _percentile(latencies, 0.99),
            },
            "ttft_s": {
                "mean": _mean(ttfts),
                "p50": _percentile(ttfts, 0.50),
                "p95": _percentile(ttfts, 0.95),
                "p99": _percentile(ttfts, 0.99),
            },
            "cost_usd": {
                "total": round(sum(costs), 8) if costs else None,
                "mean": _mean(costs),
            },
            "usage": usage_totals,
            "guardrail_blocks": sum(
                1
                for record in records
                if record.get("blocked")
                or "output_blocked" in (record.get("guardrail_flags") or [])
            ),
            "node_latency_s": node_latency,
            "avg_attempts": _mean(record["attempts"] for record in records),
            "ragas": _numeric_summary(ragas_frame),
        }
        if ragas_coverage:
            summary["per_mode"][mode]["ragas_coverage"] = ragas_coverage

    summary = _sanitize(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Flat CSV for quick comparison in a spreadsheet.
    try:
        import pandas as pd

        flat_rows = []
        for mode, data in summary["per_mode"].items():
            row = {"mode": mode, "n": data["n"]}
            row.update({f"ir_{key}": value for key, value in data["used_ir"].items()})
            row.update({f"latency_{key}": value for key, value in data["latency_s"].items()})
            row.update({f"ttft_{key}": value for key, value in data["ttft_s"].items()})
            row["cost_total_usd"] = data["cost_usd"]["total"]
            row["cost_mean_usd"] = data["cost_usd"]["mean"]
            row["input_tokens"] = data["usage"]["input_tokens"]
            row["output_tokens"] = data["usage"]["output_tokens"]
            row["guardrail_blocks"] = data["guardrail_blocks"]
            row.update(data["ragas"])
            flat_rows.append(row)
        pd.DataFrame(flat_rows).to_csv(output_dir / "summary.csv", index=False)
    except Exception as error:  # pragma: no cover
        print(f"Could not write summary.csv: {error}")

    print("\n=== Summary ===")
    print(json.dumps(summary["per_mode"], indent=2))
    print(f"\nArtifacts: {output_dir}")
    if args.resume and not args.retrieval_only and not args.no_ragas:
        cache_path = (
            Path(args.cache_path)
            if args.cache_path
            else output_dir / "ragas_cache.jsonl"
        )
        print(f"RAGAS resume cache: {cache_path}")
        print("Re-run the same command later to score only what is still missing.")


if __name__ == "__main__":
    main()
