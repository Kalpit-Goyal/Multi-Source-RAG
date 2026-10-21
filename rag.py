"""LangGraph RAG pipeline: condense -> retrieve -> grade -> generate."""

from __future__ import annotations

import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from vector_store import DEFAULT_DB_DIR, search_transcript

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_ATTEMPTS = 2
HISTORY_MESSAGES = 6


class RAGState(TypedDict, total=False):
    """Data passed between graph nodes."""

    question: str
    history: list[dict]
    source_ids: list[str] | None
    k: int
    persist_directory: str
    search_query: str
    documents: list[Document]
    relevant: bool
    answer: str
    cited_documents: list[Document]
    attempts: int


CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest question as a standalone search query. "
            "Resolve pronouns and references to earlier turns, but keep the meaning. "
            "Return only the query text.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade retrieval quality. You receive a question and numbered context chunks. "
            "Reply with a comma-separated list of the chunk numbers that help answer the "
            "question, for example: 1, 3. If none of them help, reply with exactly NONE.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

REFINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "A search returned nothing useful. Rewrite the question into a broader search "
            "query using synonyms and fewer specific terms. Return only the query text.",
        ),
        ("human", "Question: {question}\nFailed query: {search_query}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using only the retrieved context from the user's indexed "
            "sources. If the context does not contain the answer, say you do not know. "
            "Cite evidence inline as (source name, chunk N). Be concise.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def _llm(temperature: float = 0) -> ChatGroq:
    return ChatGroq(model=MODEL, temperature=temperature)


def _history_messages(history: list[dict] | None) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in (history or [])[-HISTORY_MESSAGES:]:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if item.get("role") == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


def _chunk_label(document: Document) -> str:
    metadata = document.metadata
    name = metadata.get("source_name", "source")
    locator = metadata.get("locator")
    suffix = f", {locator}" if locator else ""
    return f"({name}, chunk {metadata.get('chunk_index', '?')}{suffix})"


def _format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"{_chunk_label(document)}\n{document.page_content}"
        for document in documents
    )


def _relevant_documents(text: str, documents: list[Document]) -> list[Document]:
    """Map the grader's chunk numbers back to documents."""
    letters = re.sub(r"[^A-Z]", "", (text or "").upper())
    if "NONE" in letters:
        return []

    numbers = {
        int(number)
        for number in re.findall(r"\d+", text or "")
        if 1 <= int(number) <= len(documents)
    }
    if not numbers:
        # Unclear verdict: keep everything and let generation stay grounded.
        return list(documents)
    return [documents[number - 1] for number in sorted(numbers)]


def condense_question(state: RAGState) -> dict:
    history = _history_messages(state.get("history"))
    question = state["question"]
    if not history:
        return {"search_query": question}

    response = (CONDENSE_PROMPT | _llm()).invoke(
        {"history": history, "question": question}
    )
    query = str(response.content).strip().strip('"')
    return {"search_query": query or question}


def retrieve(state: RAGState) -> dict:
    attempts = state.get("attempts", 0) + 1
    k = state.get("k", 4) + (attempts - 1) * 3
    documents = search_transcript(
        state.get("search_query") or state["question"],
        persist_directory=state.get("persist_directory", DEFAULT_DB_DIR),
        source_ids=state.get("source_ids"),
        k=k,
    )
    return {"documents": documents, "attempts": attempts}


def grade_documents(state: RAGState) -> dict:
    documents = state.get("documents") or []
    if not documents:
        return {"relevant": False}

    context = "\n\n".join(
        f"[{index}] {_chunk_label(document)}\n{document.page_content[:600]}"
        for index, document in enumerate(documents, start=1)
    )
    response = (GRADE_PROMPT | _llm()).invoke(
        {"question": state["question"], "context": context}
    )
    relevant = _relevant_documents(str(response.content), documents)
    return {"documents": relevant, "relevant": bool(relevant)}


def refine_query(state: RAGState) -> dict:
    response = (REFINE_PROMPT | _llm()).invoke(
        {
            "question": state["question"],
            "search_query": state.get("search_query") or state["question"],
        }
    )
    query = str(response.content).strip().strip('"')
    return {"search_query": query or state["question"]}


def generate_answer(state: RAGState) -> dict:
    documents = state.get("documents") or []
    response = (ANSWER_PROMPT | _llm()).invoke(
        {
            "context": _format_context(documents),
            "history": _history_messages(state.get("history")),
            "question": state["question"],
        }
    )
    return {
        "answer": str(response.content).strip(),
        "cited_documents": documents,
    }


def no_answer(state: RAGState) -> dict:
    return {
        "answer": (
            "I couldn't find that in the indexed sources. Try rephrasing the question, "
            "or add a source that covers it."
        ),
        "cited_documents": [],
    }


def route_after_grading(state: RAGState) -> str:
    if state.get("relevant"):
        return "generate"
    if state.get("attempts", 0) < MAX_ATTEMPTS:
        return "refine"
    return "no_answer"


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("condense", condense_question)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade_documents)
    graph.add_node("refine", refine_query)
    graph.add_node("generate", generate_answer)
    graph.add_node("no_answer", no_answer)

    graph.add_edge(START, "condense")
    graph.add_edge("condense", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grading,
        {"generate": "generate", "refine": "refine", "no_answer": "no_answer"},
    )
    graph.add_edge("refine", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("no_answer", END)
    return graph.compile()


PIPELINE = build_graph()


def answer_question(
    question: str,
    persist_directory=DEFAULT_DB_DIR,
    source_ids: list[str] | None = None,
    history: list[dict] | None = None,
    k: int = 4,
) -> tuple[str, list[Document]]:
    """Run the graph and return the answer plus the chunks it was based on."""
    result = PIPELINE.invoke(
        {
            "question": question,
            "history": history or [],
            "source_ids": source_ids or None,
            "k": k,
            "persist_directory": str(persist_directory),
            "attempts": 0,
        }
    )
    return result.get("answer", ""), result.get("cited_documents", [])
