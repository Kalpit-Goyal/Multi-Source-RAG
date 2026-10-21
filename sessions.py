"""Chat session persistence for the Streamlit app."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


SESSIONS_PATH = Path(__file__).parent / "chat_sessions.json"
MAX_TITLE = 60


def _blank() -> dict:
    return {"sessions": {}, "current": None}


def load_sessions() -> dict:
    if not SESSIONS_PATH.exists():
        return _blank()
    try:
        data = json.loads(SESSIONS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _blank()
    if not isinstance(data, dict) or not isinstance(data.get("sessions"), dict):
        return _blank()
    return {"sessions": data["sessions"], "current": data.get("current")}


def save_sessions(data: dict) -> None:
    """Write sessions atomically so a crash cannot corrupt the history."""
    temp = SESSIONS_PATH.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp.replace(SESSIONS_PATH)


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
