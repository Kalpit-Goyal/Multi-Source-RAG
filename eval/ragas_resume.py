"""Resumable RAGAS scoring for tight (free-tier) judge quotas.

``ragas.evaluate()`` is all-or-nothing: if the judge quota runs out halfway
through a mode, every score from that run is lost. This module scores one
``(mode, question, metric)`` triple at a time and appends each success to an
append-only JSONL cache *immediately* (with ``fsync``), so a later run with the
same arguments skips finished work and continues where it stopped.

Files written next to the run artifacts:

* ``ragas_cache.jsonl``    -- successful scores; safe to reuse across days.
* ``ragas_failures.jsonl`` -- attempts that raised (quota, timeout, parse).
  Failures are **not** cached, so the next run retries them.

The cache key includes the judge model, mode, metric, and normalized question,
so switching judge models or modes never mixes scores.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

# RAGAS posts anonymous usage events on import; disable for offline/CI runs.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

CACHE_FILENAME = "ragas_cache.jsonl"
FAILURES_FILENAME = "ragas_failures.jsonl"


def metric_builders() -> dict:
    """Return ``{name: factory}`` for the RAGAS metrics used by run_eval."""
    from ragas.metrics import (
        AnswerCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
        SemanticSimilarity,
    )

    return {
        "faithfulness": Faithfulness,
        # Groq rejects n > 1, which response relevancy requests by default.
        "answer_relevancy": lambda: ResponseRelevancy(strictness=1),
        "context_precision": LLMContextPrecisionWithReference,
        "context_recall": LLMContextRecall,
        "answer_correctness": AnswerCorrectness,
        "semantic_similarity": SemanticSimilarity,
    }


def cache_key(mode: str, metric: str, question: str, judge_model: str) -> str:
    normalized = " ".join((question or "").split())
    raw = "\x1f".join([judge_model, mode, metric, normalized])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, dict]:
    """Read the JSONL cache into ``{key: entry}``; ignore partial/bad lines."""
    cache: dict[str, dict] = {}
    if not path.is_file():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("key") and entry.get("score") is not None:
                cache[entry["key"]] = entry
    return cache


def _append_jsonl(path: Path, payload: dict) -> None:
    """Append one record and flush it to disk before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _is_valid_score(score) -> bool:
    if score is None:
        return False
    try:
        return not math.isnan(float(score))
    except (TypeError, ValueError):
        return False


def score_mode(
    records: list[dict],
    mode: str,
    metric_names: list[str],
    judge_model: str,
    timeout: int,
    retries: int,
    cache_path: Path,
    failures_path: Path,
    max_consecutive_failures: int = 3,
    sleep_s: float = 0.2,
    strict: bool = False,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """Score one mode question-by-question, resuming from the cache.

    Returns ``(rows, coverage)`` where ``rows`` is aligned with ``records`` and
    ``coverage[metric] = {"scored": n, "total": N}``.
    """
    from langchain_groq import ChatGroq
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    from vector_store import get_embeddings

    builders = metric_builders()
    unknown = [name for name in metric_names if name not in builders]
    if unknown:
        raise SystemExit(f"Unknown RAGAS metrics: {', '.join(unknown)}")

    llm = LangchainLLMWrapper(
        ChatGroq(
            model=judge_model,
            temperature=0,
            max_retries=retries,
            timeout=timeout,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    run_config = RunConfig(max_workers=1, timeout=timeout)

    metrics = {}
    for name in metric_names:
        metric = builders[name]()
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings
        metric.init(run_config)
        metrics[name] = metric

    cache = load_cache(cache_path)
    total = len(records) * len(metric_names)
    cached = sum(
        1
        for record in records
        for name in metric_names
        if cache_key(mode, name, record["question"], judge_model) in cache
    )
    print(f"  resume cache: {cached}/{total} scores present ({cache_path})")

    consecutive_failures = 0
    new_scores = 0
    failed = 0
    stopped_early = False

    for index, record in enumerate(records, start=1):
        if stopped_early:
            break
        question = record["question"]
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=record["contexts"] or ["No context was retrieved."],
            response=record["answer"] or "(empty answer)",
            reference=record["ground_truth"],
        )
        for name in metric_names:
            key = cache_key(mode, name, question, judge_model)
            if key in cache:
                continue

            if sleep_s:
                time.sleep(sleep_s)
            try:
                score = metrics[name].single_turn_score(sample)
                if not _is_valid_score(score):
                    raise ValueError("judge returned no numeric score")
            except Exception as error:  # noqa: BLE001 - quota/parse/timeout
                if strict:
                    raise
                failed += 1
                consecutive_failures += 1
                message = f"{type(error).__name__}: {error}"
                _append_jsonl(
                    failures_path,
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "mode": mode,
                        "metric": name,
                        "question": question,
                        "error": message[:500],
                    },
                )
                print(f"  ! {mode}/{name} q{index}: {message[:180]}")
                if consecutive_failures >= max_consecutive_failures:
                    print(
                        f"  stopping {mode}: {consecutive_failures} consecutive judge "
                        f"failures (quota likely exhausted). Cached scores are kept; "
                        f"re-run the same command later to continue."
                    )
                    stopped_early = True
                    break
                continue

            consecutive_failures = 0
            new_scores += 1
            score = float(score)
            _append_jsonl(
                cache_path,
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "key": key,
                    "judge_model": judge_model,
                    "mode": mode,
                    "metric": name,
                    "question": question,
                    "score": score,
                },
            )
            cache[key] = {"key": key, "score": score}
            print(
                f"  [{index}/{len(records)}] {name} -> {score:.4f} "
                f"(new {new_scores}, failed {failed})"
            )

    # Rebuild every row from the cache so partial runs still produce aligned output.
    rows = [
        {
            name: (cache.get(cache_key(mode, name, record["question"], judge_model)) or {}).get("score")
            for name in metric_names
        }
        for record in records
    ]
    coverage = _coverage(records, metric_names, cache, mode, judge_model)
    print(
        f"  done: {new_scores} new scores, {failed} failures, "
        f"{sum(item['scored'] for item in coverage.values())}/{total} cached"
    )
    return rows, coverage


def _coverage(records, metric_names, cache, mode, judge_model) -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = {}
    for name in metric_names:
        scored = sum(
            1
            for record in records
            if cache_key(mode, name, record["question"], judge_model) in cache
        )
        coverage[name] = {"scored": scored, "total": len(records)}
    return coverage
