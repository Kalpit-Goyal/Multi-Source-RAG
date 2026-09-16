# YouTube RAG Chatbot

Index a YouTube transcript in a local ChromaDB collection and ask questions about the video using OpenAI embeddings and chat completion.

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

3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

## Web app

Start the Streamlit interface:

```powershell
streamlit run app.py
```

Paste a YouTube URL in the sidebar, index it, and ask questions. Embeddings and transcript chunks are stored in the local `chroma_db/` directory, which is excluded from Git.

## Command line

Index a video and open an interactive chat:

```powershell
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Ask one question and exit:

```powershell
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --question "What is the main idea?"
```

## Notes

- The application currently requests English captions with `youtube-transcript-api`.
- Each video is stored under stable IDs, so re-indexing replaces its old chunks rather than duplicating them.
- Never commit `.env`, API keys, `.venv`, or the generated `chroma_db` directory.