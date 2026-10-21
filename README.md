# Multi-source RAG assistant

Index PDFs, spreadsheets, Word documents, text files, images, and YouTube transcripts into a
local ChromaDB knowledge base, then chat across all of them. Chat runs on Groq; embeddings run
locally with Hugging Face, so no embedding API key is needed.

## Features

- Add many files at once and many YouTube URLs at once.
- Remove a single source or wipe the whole knowledge base from the sidebar.
- Chat across every source, or narrow retrieval to a subset.
- Chat sessions with full history in the sidebar, persisted to `chat_sessions.json`.
- Previous messages stay visible when you continue a conversation.

## LangGraph pipeline

`rag.py` builds a `StateGraph` with six nodes:

```text
START -> condense -> retrieve -> grade
                                   |-- useful ---------------> generate -> END
                                   |-- not useful, retry left -> refine -> retrieve
                                   '-- retries exhausted ----> no_answer -> END
```

| Node | Purpose |
| --- | --- |
| `condense` | Rewrites a follow-up question into a standalone search query using chat history. Skipped on the first turn. |
| `retrieve` | Similarity search over Chroma, optionally filtered to selected sources. Broadens `k` on a retry. |
| `grade` | Asks the LLM which numbered chunks actually help, and keeps only those. |
| `refine` | Rewrites the query with broader terms when grading found nothing. |
| `generate` | Answers from the graded chunks and cites `(source, chunk N)`. |
| `no_answer` | Returns a grounded "not found" message instead of guessing. |

## Supported sources

| Type | Extensions | Extraction |
| --- | --- | --- |
| PDF | `.pdf` | Text layer per page (`pypdf`) |
| Excel | `.xlsx`, `.xlsm` | Rows as `column: value` blocks (`openpyxl`) |
| CSV | `.csv` | Rows as `column: value` blocks |
| Word | `.docx` | Paragraphs and tables (`python-docx`) |
| Text | `.txt`, `.md` | UTF-8 or cp1252 |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff` | OCR (`rapidocr-onnxruntime`) |
| YouTube | URL | English captions (`youtube-transcript-api`) |

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

3. Copy `.env.example` to `.env` and set `GROQ_API_KEY` (free at <https://console.groq.com/keys>).

## Web app

```powershell
streamlit run app.py
```

Use the sidebar to upload files, paste YouTube URLs, remove sources, and switch chats. The first
question in a chat names that chat; nothing is lost when you switch between them.

To use a different Groq chat model, set `GROQ_MODEL` in `.env` (default: `openai/gpt-oss-120b`).
Availability varies by account — list yours at <https://console.groq.com/docs/models>.

## Command line

Index one or more sources and open a chat:

```powershell
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" report.pdf sales.xlsx
```

Ask one question and exit:

```powershell
python main.py invoice.png --question "What is the total?"
```

The CLI shares the same ChromaDB knowledge base as the web app, but keeps its conversation
history only in memory.

## Files

| File | Role |
| --- | --- |
| `app.py` | Streamlit UI: knowledge base, chat sessions, message history |
| `rag.py` | LangGraph pipeline and the Groq model configuration |
| `loaders.py` | Text extraction for every supported source type |
| `chunking.py` | Splitter and per-source chunking that preserves metadata |
| `vector_store.py` | Chroma access, source registry, add/search/delete |
| `sessions.py` | JSON persistence for chat sessions |
| `transcript.py` | YouTube URL parsing and caption fetching |
| `main.py` | Command line entry point |

## Notes

- Groq has no embeddings endpoint, so embeddings run locally with `sentence-transformers/all-MiniLM-L6-v2`
  (384 dimensions, downloaded on first run, ~90 MB).
- Images are processed with OCR, so only text inside the image is searchable. Photos or diagrams
  without text produce nothing; convert them to text first.
- Scanned PDFs have no text layer and will report "No readable text found". Upload page images instead.
- `youtube-transcript-api` is configured for English captions.
- Changing the embedding model invalidates existing vectors; delete `chroma_db/` and re-index.
- `chat_sessions.json` holds local chat history and is excluded from Git.
- Never commit `.env`, API keys, `.venv`, or the generated `chroma_db` directory.
