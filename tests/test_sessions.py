import json

import sessions


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "DB_PATH", tmp_path / "sessions.sqlite3")
    monkeypatch.setattr(sessions, "LEGACY_JSON", tmp_path / "chat_sessions.json")


def test_round_trip_preserves_metrics(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    data = {
        "sessions": {
            "s1": {
                "id": "s1",
                "title": "Policy chat",
                "created_at": 1.0,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "hello",
                        "sources": [{"source": "Policy", "chunk": 1}],
                        "retrieval": "Hybrid + rerank",
                        "latency": 1.25,
                        "attempts": 1,
                        "ttft": 0.5,
                        "cost_usd": 0.000123,
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                        "guardrail_flags": ["missing_citation"],
                    },
                ],
            }
        },
        "current": "s1",
    }

    sessions.save_sessions(data)
    loaded = sessions.load_sessions()

    assert loaded["current"] == "s1"
    message = loaded["sessions"]["s1"]["messages"][1]
    assert message["sources"][0]["source"] == "Policy"
    assert message["ttft"] == 0.5
    assert message["cost_usd"] == 0.000123
    assert message["usage"]["total_tokens"] == 15
    assert message["guardrail_flags"] == ["missing_citation"]


def test_saving_without_a_session_deletes_it(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    sessions.save_sessions(
        {
            "sessions": {
                "s1": {
                    "id": "s1",
                    "title": "x",
                    "created_at": 1.0,
                    "messages": [],
                }
            },
            "current": "s1",
        }
    )
    sessions.save_sessions({"sessions": {}, "current": None})
    assert sessions.load_sessions()["sessions"] == {}


def test_legacy_json_is_imported_once(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    legacy = tmp_path / "chat_sessions.json"
    legacy.write_text(
        json.dumps(
            {
                "sessions": {
                    "old": {
                        "id": "old",
                        "title": "Old chat",
                        "created_at": 1.0,
                        "messages": [{"role": "user", "content": "hello"}],
                    }
                },
                "current": "old",
            }
        ),
        encoding="utf-8",
    )

    loaded = sessions.load_sessions()
    assert loaded["current"] == "old"
    assert loaded["sessions"]["old"]["messages"][0]["content"] == "hello"
    assert not legacy.exists()
    assert (tmp_path / "chat_sessions.json.migrated").exists()


def test_touch_title_only_names_new_chats():
    session = sessions.create_session()
    sessions.touch_title(session, "What is the refund policy?")
    assert session["title"] == "What is the refund policy?"

    sessions.touch_title(session, "A different question")
    assert session["title"] == "What is the refund policy?"
