"""LangSmith tracing configuration and per-run metadata.

LangChain/LangGraph emit traces automatically once ``LANGSMITH_TRACING`` is
truthy and ``LANGSMITH_API_KEY`` is present. This module only centralizes the
defaults and builds the ``config`` dict that names each run and attaches the
retrieval settings, so traces can be filtered by mode/k in the LangSmith UI.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_PROJECT = "multi-source-rag"
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY


def configure() -> None:
    """Normalize tracing env vars; never enable tracing without an API key."""
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    os.environ.setdefault(
        "LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", DEFAULT_PROJECT)
    )
    if not api_key:
        return
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() in _FALSY:
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    # Older LangChain versions still read this name.
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")


configure()


def langsmith_enabled() -> bool:
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    return bool(api_key) and _flag("LANGSMITH_TRACING")


def project_name() -> str:
    return os.getenv("LANGSMITH_PROJECT", DEFAULT_PROJECT)


def run_config(
    *,
    thread_id: str,
    mode: str,
    k: int,
    source_ids: list[str] | None = None,
    question: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Build the LangGraph config for one invocation.

    ``configurable.thread_id`` drives checkpoint persistence. The remaining
    keys are LangSmith run metadata, ignored when tracing is off.
    """
    config: dict = {"configurable": {"thread_id": thread_id}}
    if not langsmith_enabled():
        return config

    metadata = {
        "retrieval_mode": mode,
        "k": k,
        "scoped": bool(source_ids),
        "source_count": len(source_ids or []),
        **(extra or {}),
    }
    if question is not None:
        metadata["question_chars"] = len(question)

    tags = ["rag", mode, f"k={k}"]
    if extra:
        if extra.get("multi_query"):
            tags.append("multi-query")
        if extra.get("parent_expansion"):
            tags.append("parent-expansion")
        for source_type in extra.get("source_types") or []:
            tags.append(f"type={source_type}")

    config.update(
        {
            "run_name": f"rag-{mode}",
            "tags": tags,
            "metadata": metadata,
        }
    )
    return config
