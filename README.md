# Multi-source Agentic RAG assistant

Index PDFs, spreadsheets, Word documents, text files, images, public websites, and YouTube
transcripts into a local ChromaDB knowledge base, then chat across all of them. Chat runs on
Groq; embeddings and reranking run locally with Hugging Face, so only the chat model needs an
API key.

## Features

- Add many files, many websites, or many YouTube URLs at once.
- **Include/exclude any source per question without deleting it.** Unselecting a source removes
  its vectors from retrieval; the source stays indexed. A second metadata filter narrows
  retrieval by source type (`pdf`, `website`, `youtube`, ...).
- Remove a single source or wipe the whole knowledge base from the sidebar.
- Three retrieval modes, switchable at query time: vector-only, hybrid, and hybrid + cross-encoder
  reranking. Hybrid fuses ChromaDB dense search with BM25 using reciprocal rank fusion.
- **Streaming answers with time-to-first-token (TTFT).** The Streamlit UI renders tokens as they
  arrive; the CLI has `--stream`; the evaluation harness records TTFT.
- **Multi-query retrieval and query decomposition.** One extra LLM call produces paraphrases and
  sub-questions; their ranked lists are fused with RRF and reranked once.
- **Parent-document retrieval.** Child chunks are ranked, then expanded into a larger surrounding
  parent window before generation.
- **Persistent conversation memory** in SQLite via the LangGraph checkpointer. Chats survive
  reruns, page reloads, and app restarts.
- **SQLite session storage** for the sidebar transcript, sources, timings, token usage, cost, and
  guardrail flags. The legacy `chat_sessions.json` is imported once and renamed aside.
- **Explicit retry/backoff** with jitter, `Retry-After` handling, timeouts, and optional pacing for
  Groq rate limits.
- **Prompt-injection defenses and output guardrails:** input screening, escaped untrusted context,
  system-prompt-leak blocking, and citation/context-overlap checks.
- **USD cost tracking** from Groq token usage, accumulated per node, per message, and per eval run.
- **LangSmith tracing** with per-run metadata (retrieval mode, `k`, scope, source types,
  multi-query, parent expansion, model).
- **RAGAS evaluation harness** comparing retrieval modes with IR metrics, RAGAS scores, latency
  percentiles, TTFT, token usage, and estimated cost.
- Website ingestion strips navigation/scripts, caps downloads, and rejects private hosts (SSRF
  guard).

## LangGraph pipeline

`rag.py` builds a `StateGraph` with eight nodes and a SQLite checkpointer:

```text
START -> (seed) ----------------------------------------------> END   # history migration
     \-> condense -> expand -> retrieve -> grade
                        ^          |-- useful ------------------> generate -> END
                        |          |-- not useful, retry left -> refine
                        |          '-- retries exhausted ------> no_answer -> END
                        '-----------------------------------------'
```

| Node | Purpose |
| --- | --- |
| `seed` | Imports pre-checkpointer history into an empty thread. |
| `condense` | Rewrites a follow-up question into a standalone search query using chat history. Skipped on the first turn. |
| `expand` | Optionally creates paraphrases/sub-questions for multi-query retrieval. |
| `retrieve` | Runs the selected retrieval mode with source-id/source-type filters, then expands hits to parent windows. Broadens `k` on a retry. |
| `grade` | Asks the LLM which numbered chunks actually help, and keeps only those. |
| `refine` | Rewrites the query with broader terms when grading found nothing. |
| `generate` | Streams or returns an answer from the graded context and cites `(source, chunk N)` or `(source, chunks N-M)`. |
| `no_answer` | Returns a grounded "not found" message instead of guessing. |

Memory is persisted by `SqliteSaver` in `memory.sqlite3`. The graph receives only a `thread_id`
(the chat session id); it does not depend on Streamlit's process state.

## Retrieval

`search_transcript(..., mode=...)` exposes three modes over the same chunk corpus:

| Mode | Candidates | Ranker |
| --- | --- | --- |
| `vector` | 20 dense hits | Chroma cosine similarity |
| `hybrid` | 20 dense + 20 BM25 | Reciprocal rank fusion (`1 / (60 + rank)`) |
| `hybrid_rerank` | 20 fused candidates | Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2` by default), top `k` |

`search_multi_query(...)` runs each query through the selected base mode, fuses the lists with
RRF, and reranks the fused pool once for `hybrid_rerank`. Reranking every sub-query separately
would multiply the most expensive step.

### Metadata filters

Both retrieval arms apply the same filters:

- `source_ids`: per-source **Use** checkboxes in the app; `--source-type` is separate.
- `source_types`: `pdf`, `website`, `youtube`, `text`, `image`, `docx`, `csv`, `excel`, etc.

Chroma receives a `$and` metadata filter; BM25 applies an in-memory predicate. An explicit empty
list means "match nothing", not "no filter". If every source is unchecked, or every source type is
cleared, the chat input is disabled.

### Parent-document retrieval

`chunking.py` still indexes 1,200-character child chunks with 200-character overlap, so existing
chunk ids stay stable. Each child also stores a denormalized parent window (`PARENT_WINDOW=2` by
default):

- `parent_id`, `parent_index`, `parent_start`, `parent_end`, `parent_text`.

Retrieval ranks children; `expand_to_parents` collapses multiple hits from the same window into
one parent document. The parent document keeps `child_chunk_ids`, so IR metrics can still compare
against gold child ids. Old chunks without parent metadata pass through unchanged. Re-index a
source to get parent metadata for it.

## Streaming and TTFT

`AnswerStream` wraps `PIPELINE.stream(..., stream_mode="messages")`, filters chunks to the
`generate` node, and yields text as the model produces it. After the stream ends it reads the
final graph state and exposes:

- `stream.result.answer`
- `stream.result.documents`
- `stream.result.meta` with `ttft_s`, total latency, usage, cost, and guardrail flags.

If a provider returns no token chunks (for example, `no_answer`), the handle yields the final
stored answer once. TTFT is `None` in that case. For non-streaming `answer_question`, TTFT is also
`None`; use `eval/run_eval.py --stream` to measure it.

The Streamlit UI clears the streamed bubble and shows the safe response if the output guardrail
blocks or replaces an answer after generation.

## Reliability and rate limits

`resilience.py` adds explicit behavior around LangChain's built-in retries:

- Retries 429, 408, 425, 5xx, timeout, and connection errors.
- Exponential backoff with jitter; honors `Retry-After` or Groq's "try again in 6.9s" message.
- `LLM_RETRY_ATTEMPTS`, `LLM_RETRY_BASE_DELAY`, `LLM_RETRY_MAX_DELAY`, `LLM_RETRY_JITTER`.
- `LLM_TIMEOUT_S` request timeout.
- `LLM_MIN_INTERVAL_S` optional process-wide minimum interval to avoid bursting a free-tier quota.
- `friendly_error` turns 429/5xx/timeout exceptions into short UI messages.

Non-streaming LLM calls use the outer retry wrapper. Streaming generation does not replay a
partially emitted answer: it relies on `ChatGroq`'s internal retry for failures before the stream
starts. If a stream fails midway, the UI reports the error rather than duplicating tokens.

## Guardrails

Three layers are applied:

1. **Input screening** (`guardrails.screen_input`) blocks high-confidence jailbreak, role
   override, system-prompt extraction, and secret-exfiltration attempts. Ordinary security
   questions that merely mention "ignore previous instructions" are allowed through.
2. **Context isolation** wraps every retrieved chunk in escaped `<document>` tags and instructs
   the model to treat the content as untrusted data, never as instructions. Instruction-like
   phrases found in retrieved text are recorded as `context_flags`.
3. **Output checks** (`guardrails.check_output`) block empty answers and obvious system-prompt
   leakage. Missing citations and low context overlap are flagged, not blocked, because
   paraphrase and short answers can legitimately have low lexical overlap.

These are heuristics, not a complete security boundary. Guardrail flags are stored on assistant
messages and in evaluation records.

## USD cost tracking

`pricing.py` extracts usage from Groq/LangChain responses and estimates cost from per-million-token
rates. Defaults cover current Groq models and can be overridden without changing code:

```text
MODEL_PRICING_JSON={"openai/gpt-oss-120b":{"input":0.15,"output":0.60}}
```

Usage is accumulated across `condense`, `expand`, `grade`, `refine`, and `generate` for each turn.
The app shows per-message and total estimated cost; the eval summary reports total and mean cost,
input/output tokens, and per-mode usage. If a provider omits usage, the fallback is a rough
four-characters-per-token estimate. Treat all costs as local estimates, not billing data.

## Persistent memory and sessions

Two databases are intentional:

| Database | Contents | Owner |
| --- | --- | --- |
| `memory.sqlite3` | LangGraph checkpoints: the LLM-visible conversation state | `memory.py` |
| `sessions.sqlite3` | Sidebar metadata and rendered transcript, plus sources, latency, TTFT, usage, cost, and guardrail flags | `sessions.py` |

- New chats get a UUID `thread_id`; every turn is checkpointed by LangGraph.
- Legacy `chat_sessions.json` is imported into `sessions.sqlite3` once and renamed
  `chat_sessions.json.migrated`. Existing chats are preserved.
- `seed_thread` imports legacy transcript messages into an empty LangGraph thread.
- Deleting a chat removes its SQLite session/messages and its checkpoint thread.
- Prompts use the last 8 messages by default (`HISTORY_MESSAGES`), while the checkpointer keeps
  the full transcript. This avoids unbounded prompt growth without losing history.
- The CLI uses the same checkpointer with `--thread cli` (override with `--thread`), so command
  line conversations also survive restarts. Use `--reset-memory` to start fresh.

SQLite is the right amount of infrastructure for this single-user local app: it is transactional,
survives restarts, requires no server, and keeps the repo portable. If the app becomes
multi-user, replace `sessions.py`/`memory.py` with Postgres-backed storage; the call sites already
use a small load/save API.

## RAGAS evaluation

RAGAS is a library for scoring RAG pipelines with an LLM judge. It answers: *did we retrieve the
right evidence, and did the model use it without hallucinating?* The harness in `eval/run_eval.py`
runs every question through each selected retrieval mode and reports:

| Metric | What it measures |
| --- | --- |
| `context_precision` | Of the retrieved chunks, how many were actually relevant (ranking-aware). |
| `context_recall` | How much of the reference answer's evidence appears in the retrieved context. |
| `faithfulness` | Fraction of answer claims supported by the retrieved context (hallucination). |
| `answer_relevancy` | How directly the answer addresses the question. |
| `answer_correctness` | Agreement with the reference answer (factual + semantic). |
| `semantic_similarity` | Embedding cosine similarity between answer and reference. |

It also computes retrieval-only IR metrics that RAGAS does not cover: Hit Rate@k, Recall@k,
Precision@k, MRR, and NDCG@k, both on the pre-grading candidate set and on the chunks actually
sent to the generator. Latency is reported per node and as P50/P95/P99, and streaming runs add
TTFT percentiles.

> `answer_relevancy` uses `strictness=1` because Groq rejects requests with `n > 1`.
> `answer_correctness` is used instead of the standalone `factual_correctness` metric, which
> returned 0.0 on exact-match answers with the current Groq judge.

RAGAS is an offline judge, not a runtime dependency. It uses the same `GROQ_API_KEY` by default;
set `RAGAS_JUDGE_MODEL` to use a different model.

### Datasets

`eval/dataset.jsonl` is a local starter set over the currently indexed document. It is gitignored
because it contains personal source content. Each row uses:

```json
{
  "question": "...",
  "ground_truth": "...",
  "ground_truth_chunk_ids": ["file:073b59a118126a18:0"],
  "source_ids": ["file:073b59a118126a18"],
  "source_types": ["pdf"]
}
```

`source_ids` and `source_types` are optional; omit or set `null` to search the whole corpus.
`ground_truth_chunk_ids` use `source_id:chunk_index`. If you replace the underlying file, its
content hash changes and the ids must be regenerated.

For a public, reproducible run, use the importer:

```powershell
# Amnesty QA: human reference answers; contexts become the corpus
python eval/import_public_dataset.py --source amnesty --limit 50 --reset `
  --persist-directory chroma_db_public --output eval/dataset.public.jsonl

# One context per question; useful for both IR and RAGAS
python eval/import_public_dataset.py --source rag12000 --limit 50 --reset `
  --persist-directory chroma_db_public --output eval/dataset.public.jsonl

# RAGBench: model-generated reference responses; RAGAS only, not IR
python eval/import_public_dataset.py --source ragbench --config hotpotqa --limit 50 --reset `
  --persist-directory chroma_db_public --output eval/dataset.public.jsonl
```

Public dataset options:

| Dataset | Fields | Best for | Caveat |
| --- | --- | --- | --- |
| [`explodinggradients/amnesty_qa`](https://huggingface.co/datasets/explodinggradients/amnesty_qa) (`english_v2`) | `question`, `ground_truth`, `contexts` | IR + RAGAS with human references | Contexts are already snippets, so the corpus is a benchmark corpus, not a full web corpus. |
| [`neural-bridge/rag-dataset-12000`](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000) | `context`, `question`, `answer` | IR + RAGAS with one known context per question | Single-context questions are easier than open-corpus retrieval. |
| [`rungalileo/ragbench`](https://huggingface.co/datasets/rungalileo/ragbench) | `question`, `documents`, `response` | RAGAS on a multi-document corpus | `response` is model-generated, not human gold; sentence ids are not chunk ids. |
| [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) | `passages` + `question-answer` | Large-corpus RAGAS experiments | The QA split has no gold passage mapping, so IR metrics need a separate mapping. |

The importer uses Hugging Face `datasets`, so the first run downloads and caches the
dataset. You can also browse the dataset cards at the links above and export a JSONL file
yourself; `eval/run_eval.py` only needs `question`, `ground_truth`, and optional
`ground_truth_chunk_ids`.

The importer indexes the dataset's contexts into `chroma_db_public` by default and writes gold
child-chunk ids. It does not touch your app corpus unless you explicitly point
`--persist-directory` at it. Use `--reset` only with a dedicated directory such as
`chroma_db_public`; the importer refuses to reset the default `chroma_db`.

### Run

```powershell
# Zero-Groq retrieval evaluation: IR metrics only, all questions and modes
python eval/run_eval.py --dataset eval/dataset.public.jsonl `
  --persist-directory chroma_db_public --modes vector hybrid hybrid_rerank `
  --k 4 --retrieval-only

# End-to-end smoke without judge calls (still uses Groq for grade/generate)
python eval/run_eval.py --dataset eval/dataset.public.jsonl `
  --persist-directory chroma_db_public --modes hybrid_rerank --k 4 `
  --limit 3 --stream --no-ragas --workers 1 --sleep 1

# Full RAGAS on one mode (after the no-ragas records exist)
python eval/run_eval.py --from-records eval/results/<timestamp>/per_question.jsonl `
  --modes hybrid_rerank --workers 1

# Free-tier judge run: one question, two metrics (use a one-record subset file)
python eval/run_eval.py --from-records eval/results/<run>/ragas-1q.jsonl `
  --modes hybrid_rerank `
  --metrics faithfulness context_recall --workers 1

# Full pipeline + RAGAS for all three modes
python eval/run_eval.py --dataset eval/dataset.public.jsonl `
  --persist-directory chroma_db_public --modes vector hybrid hybrid_rerank --k 4 --stream

# Resume generation at the next question
python eval/run_eval.py --dataset eval/dataset.public.jsonl --offset 3 --limit 1 --no-ragas

# Compare two modes only
python eval/run_eval.py --modes hybrid hybrid_rerank

# Latency/token/cost report from LangSmith traces
python eval/latency_report.py
```

`--retrieval-only` makes no Groq calls at all. It runs the same dense/BM25/reranker
retrieval and parent expansion as the graph, but skips condense, expand, grade,
refine, and generate. `--no-ragas` is different: the pipeline still calls Groq
for grading and generation.

`--metrics` limits the RAGAS judge to selected metrics; `--offset`/`--limit`
let you split generation across days. `--from-records` always reuses stored
answers and contexts. Set `RAGAS_JUDGE_RETRIES=0` to fail fast when the daily
quota is exhausted.

Artifacts land in `eval/results/<timestamp>/`: `per_question.jsonl`, `ragas_<mode>.csv`,
`summary.json`, and `summary.csv`. `--from-records` skips retrieval and generation and only
recomputes RAGAS/summary, which is the cheapest way to add judge metrics after a no-ragas run.

### Free-tier guidance

Groq free-tier limits dominate wall-clock latency and can make a large RAGAS batch take much
longer than the pipeline itself. Before quoting numbers:

- Warm up embeddings, BM25, and the reranker (`run_eval.py` does this by default).
- Run `--retrieval-only` first: it evaluates retrieval and IR with zero Groq calls.
- Then run a small end-to-end sample with `--no-ragas`; this still uses Groq for
  grading/generation.
- Add RAGAS from stored records with `--from-records`; never regenerate answers
  just to add judge metrics.
- Limit judge work with `--metrics` (for example `faithfulness context_recall`)
  and set `RAGAS_JUDGE_RETRIES=0` to fail fast when quota is gone.
- Split generation across days with `--offset`/`--limit`.
- Start with one mode and one or two questions; increase only after the artifact looks sane.
- Use `--workers 1` and `--sleep 1` (or higher) on free tier.
- Do not compare free-tier TTFT with paid-tier TTFT; queueing is part of the measurement.
- Report the dataset, question count, model, `k`, mode, selected metrics, and judge model with every metric.

> The numbers in a resume must come from a fixed run of this harness. Old claims such as
> `0.625 → 1.000`, `1.10s P50`, and `37 traced runs` are not reproduced by the current code and
> must not be reused. Chunking, prompts, model, rate limits, and `k` all change the results.

### Results: 20-question public run (2026-11-14)

Fixed run for reproducible numbers. Setup: `eval/dataset.public.jsonl`
(Amnesty QA `english_v2`, first 20 questions), corpus `chroma_db_public` (65 chunks),
`k=4`, answer and judge model `openai/gpt-oss-120b` on Groq free tier.
RAGAS: 120 scores (`context_precision` + `context_recall` × 20 questions × 3 modes),
all cached in `eval/results/free-tier-resume-20261025/ragas-20q-context-resume/`.

**RAGAS (LLM judge)**

| Mode | n | Context Precision | Context Recall |
| --- | --- | --- | --- |
| `vector` | 20 | 0.8042 | 0.9500 |
| `hybrid` | 20 | **0.8792** | **1.0000** |
| `hybrid_rerank` | 20 | 0.8750 | 0.9500 |

**Retrieval IR (zero-Groq, gold chunk ids)**

| Mode | Hit Rate@4 | Recall@4 | Precision@4 | MRR | nDCG@4 | Retrieval P50 |
| --- | --- | --- | --- | --- | --- | --- |
| `vector` | 1.000 | 0.888 | 0.713 | 0.967 | 0.889 | 0.026 s |
| `hybrid` | 1.000 | 0.888 | 0.713 | 0.942 | 0.877 | 0.035 s |
| `hybrid_rerank` | 1.000 | 0.900 | 0.725 | 0.925 | 0.872 | 0.459 s |

**End-to-end (Groq free tier)**

| Mode | P50 latency | P50 TTFT | Mean cost / question | Cost, 20 questions |
| --- | --- | --- | --- | --- |
| `vector` | 11.77 s | 11.55 s | $0.00045 | $0.0091 |
| `hybrid` | 12.10 s | 11.77 s | $0.00046 | $0.0092 |
| `hybrid_rerank` | 12.72 s | 12.35 s | $0.00048 | $0.0096 |

**Findings**

- Hybrid retrieval is the real gain: RAGAS context precision **0.804 → 0.879** and
  context recall **0.950 → 1.000** versus vector-only.
- Cross-encoder reranking does **not** improve this corpus: 0.875 precision / 0.950 recall,
  slightly below plain hybrid. Keep it as an option, not a default win.
- End-to-end latency is dominated by two sequential Groq calls per question (`grade` median
  5.9 s, `generate` median 6.3 s). Retrieval itself is 0.03–0.46 s. Free-tier queueing is part
  of these numbers; they are not production latency.
- Context precision is structurally capped near 0.9 here: `k=4` but only 1–3 gold chunks per
  question, so some retrieved slots are filled by chunks relevant to other questions.
- These are **n=20 on a public QA set with an LLM judge**, not a benchmark. Quote the dataset,
  `n`, `k`, model, judge model, and metric selection with any of these numbers.

## LangSmith

Set these in `.env`:

```text
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=multi-source-rag
LANGSMITH_TRACING=true
```

Every graph invocation is named `rag-<mode>` and tagged with the mode, `k`, multi-query/parent
expansion flags, and source types. Metadata includes scope size, question length, model, and
source types. `python eval/latency_report.py` aggregates root runs into P50/P95/P99 latency,
TTFT (when available), average token usage, and estimated cost.

Without `LANGSMITH_API_KEY`, tracing is disabled and `run_config` stays minimal; the app still
works. No traces are fabricated.

## Supported sources

| Type | Extensions | Extraction |
| --- | --- | --- |
| PDF | `.pdf` | Text layer per page (`pypdf`) |
| Excel | `.xlsx`, `.xlsm` | Rows as `column: value` blocks (`openpyxl`) |
| CSV | `.csv` | Rows as `column: value` blocks |
| Word | `.docx` | Paragraphs and tables (`python-docx`) |
| Text | `.txt`, `.md` | UTF-8 or cp1252 |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff` | OCR (`rapidocr-onnxruntime`) |
| Website | `http://`, `https://` | Readable prose via `requests` + BeautifulSoup; PDF links supported |
| YouTube | URL | English captions (`youtube-transcript-api`) |

Website ingestion follows redirects, caps downloads at 5 MB (`MAX_WEBSITE_BYTES`), caps extracted
text (`MAX_WEBSITE_CHARS`), canonicalizes URLs (drops `utm_*`/fragments), and rejects
private/localhost targets.

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `GROQ_API_KEY`
   (free at <https://console.groq.com/keys>). Add `LANGSMITH_API_KEY` to enable tracing.

4. Optional: adjust retry/pacing, parent window, multi-query count, and pricing in `.env`.
   See `.env.example` for every variable.

## Web app

```powershell
streamlit run app.py
```

Use the sidebar to upload files, add websites or YouTube URLs, remove sources, choose the
retrieval mode, and switch chats. Each indexed source has a **Use** checkbox: uncheck it to keep
the source indexed but exclude its vectors from retrieval. The **Source types** multiselect is an
additional metadata filter. **Multi-query + decomposition** and **Expand hits to parent windows**
are query-time toggles.

Assistant messages show retrieval mode, TTFT, total latency, estimated cost, retrieval attempts,
and any guardrail flags. Chat history and metrics are stored in `sessions.sqlite3`.

## Command line

Index one or more sources and chat with persistent memory:

```powershell
python main.py "https://example.com/article" report.pdf sales.xlsx
```

Ask one question and exit (memory still persists under `--thread cli`):

```powershell
python main.py invoice.png --question "What is the total?"
```

Useful flags:

- `--mode hybrid|hybrid_rerank|vector`
- `--k 4`
- `--thread my-chat`
- `--source-type pdf website`
- `--multi-query`
- `--parent-expansion` / `--no-parent-expansion` (default from `PARENT_DOCUMENT_RETRIEVAL`)
- `--stream`
- `--reset-memory`

## Files

| File | Role |
| --- | --- |
| `app.py` | Streamlit UI: knowledge base, per-source scope, source-type filter, modes, streaming, chats, timings, cost |
| `rag.py` | LangGraph pipeline, Groq model config, streaming, checkpointed memory, seeding |
| `guardrails.py` | Input screening, untrusted-context formatting, output checks |
| `resilience.py` | Retry/backoff, `Retry-After`, timeout, rate limiter, friendly errors |
| `pricing.py` | Token usage extraction and USD cost estimates |
| `memory.py` | SQLite LangGraph checkpointer (`SqliteSaver`) |
| `sessions.py` | SQLite sidebar sessions/messages with legacy JSON migration |
| `observability.py` | LangSmith configuration and run metadata |
| `vector_store.py` | Chroma access, source registry, metadata filters, hybrid/multi-query search, parent expansion |
| `lexical.py` | BM25 tokenizer and index used by hybrid retrieval |
| `rerank.py` | Cross-encoder reranker |
| `loaders.py` | Text extraction for files, websites, and YouTube |
| `transcript.py` | YouTube URL parsing and caption fetching |
| `chunking.py` | Child splitter, parent-window metadata, per-source chunking |
| `eval/run_eval.py` | RAGAS + IR + latency/TTFT/cost harness across retrieval modes |
| `eval/import_public_dataset.py` | Index a public QA corpus and write a RAGAS dataset |
| `eval/make_dataset.py` | Draft a starter dataset from indexed chunks |
| `eval/latency_report.py` | P50/P95/P99, token usage, and cost from LangSmith traces |
| `eval/_ragas_compat.py` | Compatibility shim for ragas 0.4.3 on langchain-community 0.4.x |
| `main.py` | Command line entry point |
| `tests/` | Pytest suite (73 tests: guardrails, retries, pricing, SQLite sessions, parent retrieval, retrieval-only eval, streaming, retrieval, loaders) |

## Notes

- Groq has no embeddings endpoint, so embeddings run locally with
  `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, downloaded on first run, ~90 MB).
- The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is downloaded on first reranked query.
- **Cold start:** the first query loads the embedding model, BM25 index, and reranker, so measure
  latency after a warm-up. The eval harness warms up all models before timing.
- **Groq rate limits:** the grader and optional multi-query expansion make extra LLM calls per
  question, and retries can add more. Run latency benchmarks separately from large RAGAS batches;
  otherwise queueing dominates the numbers. `eval/run_eval.py --sleep` adds pacing.
- Images are processed with OCR, so only text inside the image is searchable.
- Scanned PDFs have no text layer and will report "No readable text found".
- `youtube-transcript-api` is configured for English captions.
- Changing the embedding model invalidates existing vectors; delete `chroma_db/` and re-index.
- Changing `PARENT_WINDOW` affects new/re-indexed sources; old chunks keep their existing parent
  metadata until re-indexed.
- `sessions.sqlite3`, `memory.sqlite3`, `chat_sessions.json.migrated`, `chroma_db/`,
  `chroma_db_public/`, `eval/results/`, and local datasets are excluded from Git.
- Never commit `.env`, API keys, `.venv`, or the generated `chroma_db` directory.
