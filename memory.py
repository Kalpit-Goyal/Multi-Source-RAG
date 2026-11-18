"""Persistent LangGraph checkpointer.

Conversation state lives in SQLite (not in the Streamlit process), so a chat
survives reruns, page reloads, and application restarts. ``thread_id`` is the
chat session id, which is the LangGraph equivalent of a conversation key.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


DEFAULT_MEMORY_DB = Path(__file__).parent / "memory.sqlite3"
MEMORY_DB_PATH = Path(os.getenv("MEMORY_DB_PATH") or DEFAULT_MEMORY_DB)

MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# SQLite connections are bound to the creating thread by default; Streamlit
# serves sessions from a thread pool, so the connection must be shareable.
_CONNECTION = sqlite3.connect(str(MEMORY_DB_PATH), check_same_thread=False)
_CONNECTION.execute("PRAGMA journal_mode=WAL")
_CONNECTION.execute("PRAGMA synchronous=NORMAL")

CHECKPOINTER = SqliteSaver(_CONNECTION)
CHECKPOINTER.setup()


def memory_path() -> str:
    return str(MEMORY_DB_PATH)


def delete_thread(thread_id: str) -> None:
    """Remove every checkpoint for one conversation."""
    CHECKPOINTER.delete_thread(thread_id)
