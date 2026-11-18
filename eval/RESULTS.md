# Evaluation results: 20-question public run (2026-11-14)

Fixed run for reproducible numbers. Setup: `eval/dataset.public.jsonl`
(Amnesty QA `english_v2`, first 20 questions), corpus `chroma_db_public`
(65 chunks), `k=4`, answer and judge model `openai/gpt-oss-120b` on the Groq
free tier.

RAGAS: 120 scores (`context_precision` + `context_recall` x 20 questions x
3 modes), all cached in
`eval/results/free-tier-resume-20261025/ragas-20q-context-resume/`.

## RAGAS (LLM judge)

| Mode | n | Context Precision | Context Recall |
| --- | --- | --- | --- |
| `vector` | 20 | 0.8042 | 0.9500 |
| `hybrid` | 20 | **0.8792** | **1.0000** |
| `hybrid_rerank` | 20 | 0.8750 | 0.9500 |

## Retrieval IR (zero-Groq, gold chunk ids)

| Mode | Hit Rate@4 | Recall@4 | Precision@4 | MRR | nDCG@4 | Retrieval P50 |
| --- | --- | --- | --- | --- | --- | --- |
| `vector` | 1.000 | 0.888 | 0.713 | 0.967 | 0.889 | 0.026 s |
| `hybrid` | 1.000 | 0.888 | 0.713 | 0.942 | 0.877 | 0.035 s |
| `hybrid_rerank` | 1.000 | 0.900 | 0.725 | 0.925 | 0.872 | 0.459 s |

## End-to-end (Groq free tier)

| Mode | P50 latency | P50 TTFT | Mean cost / question | Cost, 20 questions |
| --- | --- | --- | --- | --- |
| `vector` | 11.77 s | 11.55 s | $0.00045 | $0.0091 |
| `hybrid` | 12.10 s | 11.77 s | $0.00046 | $0.0092 |
| `hybrid_rerank` | 12.72 s | 12.35 s | $0.00048 | $0.0096 |

## Token usage and stage latency

| Mode | Input tokens | Output tokens | Retrieve | Grade | Generate |
| --- | --- | --- | --- | --- | --- |
| `vector` | 26,530 | 8,492 | 0.055 s | 6.010 s | 5.619 s |
| `hybrid` | 27,462 | 8,436 | 0.050 s | 5.749 s | 5.599 s |
| `hybrid_rerank` | 27,676 | 9,073 | 0.550 s | 5.733 s | 6.191 s |

Stage latency is the per-question mean from the run summary. `grade` and
`generate` each make one sequential Groq call, so they dominate end-to-end
time; retrieval is 0.03-0.46 s.

## Findings

- Hybrid retrieval is the real gain: RAGAS context precision **0.804 -> 0.879**
  and context recall **0.950 -> 1.000** versus vector-only.
- Cross-encoder reranking does **not** improve this corpus: 0.875 precision /
  0.950 recall, slightly below plain hybrid. Keep it as an option, not a
  default win.
- End-to-end latency is dominated by two sequential Groq calls per question
  (`grade` median 5.9 s, `generate` median 6.3 s). Retrieval itself is
  0.03-0.46 s. Free-tier queueing is part of these numbers; they are not
  production latency.
- Context precision is structurally capped near 0.9 here: `k=4` but only 1-3
  gold chunks per question, so some retrieved slots are filled by chunks
  relevant to other questions.
- These are **n=20 on a public QA set with an LLM judge**, not a benchmark.
  Quote the dataset, `n`, `k`, model, judge model, and metric selection with
  any of these numbers.

## Reproduce

```bash
python eval/import_public_dataset.py
python eval/run_eval.py --dataset eval/dataset.public.jsonl --k 4 \
  --modes vector hybrid hybrid_rerank --mode ragas
```

Raw artifacts, per-question scores, caches, and retry logs live under
`eval/results/free-tier-resume-20261025/` (git-ignored; regenerate with the
commands above).
