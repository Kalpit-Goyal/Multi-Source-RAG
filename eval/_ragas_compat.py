"""Compatibility shim for ragas 0.4.x on langchain-community 0.4.x.

ragas imports ``langchain_community.chat_models.vertexai.ChatVertexAI`` at
module load time. langchain-community 0.4 removed that module (Vertex AI now
lives in ``langchain-google-vertexai``), so importing ragas fails before any
metric can run. The class is only used for an ``isinstance`` check, so a stub
is enough for Groq/OpenAI judges.

Import this module *before* importing ragas::

    from eval import _ragas_compat  # noqa: F401
    from ragas import evaluate
"""

from __future__ import annotations

import sys
import types

try:  # pragma: no cover - depends on installed versions
    from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - langchain-community >= 0.4
    module = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # type: ignore[no-redef]
        """Placeholder for ragas' isinstance check."""

    module.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = module

    import langchain_community.chat_models as _chat_models

    _chat_models.vertexai = module
