"""Chat session persistence backed by SQLite.

The Streamlit sidebar keeps a dict of sessions in memory for rendering, but
every save is now transactional SQLite. This replaces the old
``chat_sessions.json`` mirror with a real database while keeping the public
``load_sessions``/``save_sessions`` API used by the app.

Two databases are intentional:

* ``memory.sqlite3`` stores LangGraph checkpoints (the LLM-visible history).
* ``sessions.sqlite3`` stores sidebar metadata and the rendered transcript,
  including sources, timings, token usage, and guardrail flags.

For a single-user local app this is the right amount of infrastructure. If the
app becomes multi-user, swap this module for Postgres; the call sites do not
need to change.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent / "sessions.sqlite3"
DB_PATH = Path(os.getenv("SESSIONS_DB_PATH") or DEFAULT_DB_PATH)
LEGACY_JSON = Path(os.getenv("LEGACY_SESSIONS_JSON") or (Path(__file__).parent / "chat_sessions.json"))
MAX_TITLE = 60
_WRITE_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,
    retrieval TEXT,
    latency REAL,
    attempts INTEGER,
    ttft REAL,
    cost_usd REAL,
    usage TEXT,
    guardrail_flags TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def _blank() -> dict:
    return {"sessions": {}, "current": None}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(_SCHEMA)
    return connection


def _decode_json(value, fallback):
    if value in (None, ""):
        return fallback
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback
    return decoded if isinstance(decoded, type(fallback)) else fallback


def _message_from_row(row: sqlite3.Row) -> dict:
    message = {
        "role": row["role"],
        "content": row["content"],
    }
    sources = _decode_json(row["sources"], [])
    if sources:
        message["sources"] = sources
    if row["retrieval"]:
        message["retrieval"] = row["retrieval"]
    if row["latency"] is not None:
        message["latency"] = row["latency"]
    if row["attempts"] is not None:
        message["attempts"] = row["attempts"]
    if row["ttft"] is not None:
        message["ttft"] = row["ttft"]
    if row["cost_usd"] is not None:
        message["cost_usd"] = row["cost_usd"]
    usage = _decode_json(row["usage"], {})
    if usage:
        message["usage"] = usage
    flags = _decode_json(row["guardrail_flags"], [])
    if flags:
        message["guardrail_flags"] = flags
    return message


def _read_sessions(connection: sqlite3.Connection) -> dict:
    rows = connection.execute(
        "SELECT * FROM sessions ORDER BY created_at, id"
    ).fetchall()
    sessions: dict[str, dict] = {}
    for row in rows:
        messages = [
            _message_from_row(message)
            for message in connection.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY id",
                (row["id"],),
            ).fetchall()
        ]
        sessions[row["id"]] = {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "messages": messages,
        }

    current_row = connection.execute(
        "SELECT value FROM app_state WHERE key='current'"
    ).fetchone()
    current = current_row["value"] if current_row else None
    if current not in sessions:
        current = max(
            sessions,
            key=lambda session_id: sessions[session_id].get("created_at", 0),
            default=None,
        )
    return {"sessions": sessions, "current": current}


def _write_sessions(connection: sqlite3.Connection, data: dict) -> None:
    sessions = data.get("sessions") if isinstance(data, dict) else {}
    if not isinstance(sessions, dict):
        sessions = {}

    existing = {
        row["id"]
        for row in connection.execute("SELECT id FROM sessions").fetchall()
    }
    removed = existing - set(sessions)
    for session_id in removed:
        connection.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    for session_id, session in sessions.items():
        if not isinstance(session, dict):
            continue
        created_at = float(session.get("created_at") or time.time())
        updated_at = float(session.get("updated_at") or created_at)
        title = str(session.get("title") or "New chat")
        connection.execute(
            """
            INSERT INTO sessions(id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                created_at=excluded.created_at,
                updated_at=excluded.updated_at
            """,
            (session_id, title, created_at, updated_at),
        )
        connection.execute("DELETE FROM messages WHERE session_id=?", (session_id,))

        for message in session.get("messages") or []:
            if not isinstance(message, dict):
                continue
            connection.execute(
                """
                INSERT INTO messages(
                    session_id, role, content, sources, retrieval, latency,
                    attempts, ttft, cost_usd, usage, guardrail_flags, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    str(message.get("role") or "user"),
                    str(message.get("content") or ""),
                    json.dumps(message.get("sources") or [], ensure_ascii=False),
                    message.get("retrieval"),
                    float(message["latency"]) if message.get("latency") is not None else None,
                    int(message["attempts"]) if message.get("attempts") is not None else None,
                    float(message["ttft"]) if message.get("ttft") is not None else None,
                    float(message["cost_usd"]) if message.get("cost_usd") is not None else None,
                    json.dumps(message.get("usage") or {}, ensure_ascii=False),
                    json.dumps(message.get("guardrail_flags") or [], ensure_ascii=False),
                    float(message.get("created_at") or time.time()),
                ),
            )

    current = data.get("current") if isinstance(data, dict) else None
    if current in sessions:
        connection.execute(
            "INSERT INTO app_state(key, value) VALUES ('current', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(current),),
        )
    else:
        connection.execute("DELETE FROM app_state WHERE key='current'")


def _migrate_legacy_json(connection: sqlite3.Connection) -> None:
    """Import the old JSON mirror exactly once, then rename it aside."""
    if not LEGACY_JSON.is_file():
        return
    try:
        data = json.loads(LEGACY_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict) or not isinstance(data.get("sessions"), dict):
        return

    _write_sessions(connection, data)
    connection.commit()

    target = LEGACY_JSON.with_name(LEGACY_JSON.name + ".migrated")
    if target.exists():
        target = LEGACY_JSON.with_name(
            f"{LEGACY_JSON.name}.migrated.{int(time.time())}"
        )
    try:
        LEGACY_JSON.replace(target)
    except OSError:
        # The import succeeded; a locked file can be moved on the next run.
        pass


def init_db() -> None:
    """Create the schema if needed (useful for tests and first-run checks)."""
    with contextlib.closing(_connect()) as connection:
        connection.commit()


def load_sessions() -> dict:
    """Read sessions from SQLite, migrating the legacy JSON file if needed."""
    with contextlib.closing(_connect()) as connection:
        data = _read_sessions(connection)
        if not data["sessions"] and LEGACY_JSON.is_file():
            _migrate_legacy_json(connection)
            data = _read_sessions(connection)
        return data


def save_sessions(data: dict) -> None:
    """Replace the persisted session snapshot in one transaction."""
    with _WRITE_LOCK, contextlib.closing(_connect()) as connection:
        _write_sessions(connection, data)
        connection.commit()


def create_session() -> dict:
    return {
        "id": uuid.uuid4().hex[:12],
        "title": "New chat",
        "created_at": time.time(),
        "messages": [],
    }


def touch_title(session: dict, text: str) -> None:
    """Name a chat after its first question."""
    if session.get("title") not in (None, "", "New chat"):
        return
    title = " ".join(text.split())
    session["title"] = (
        title if len(title) <= MAX_TITLE else title[: MAX_TITLE - 3].rstrip() + "..."
    )
