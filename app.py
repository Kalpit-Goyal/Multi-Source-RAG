import time

import streamlit as st

from chunking import chunk_documents
from loaders import (
    SUPPORTED_SUFFIXES,
    load_file,
    load_website,
    load_youtube,
)
from pricing import format_cost
from rag import (
    DEFAULT_PARENT_EXPANSION,
    delete_thread,
    seed_thread,
    stream_answer_question,
)
from resilience import friendly_error
from sessions import create_session, load_sessions, save_sessions, touch_title
from vector_store import (
    delete_all_sources,
    delete_source,
    index_source,
    list_sources,
)


st.set_page_config(page_title="Multi-source RAG", page_icon="🗂", layout="wide")

RETRIEVAL_OPTIONS = {
    "Hybrid + rerank": "hybrid_rerank",
    "Hybrid (dense + BM25)": "hybrid",
    "Vector only": "vector",
}


def _short(text: str, limit: int = 30) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _payload(documents) -> list[dict]:
    return [
        {
            "source": document.metadata.get("source_name", "source"),
            "chunk": document.metadata.get("chunk_index"),
            "chunk_end": document.metadata.get("chunk_end"),
            "locator": document.metadata.get("locator"),
            "preview": " ".join(document.page_content.split())[:350],
        }
        for document in documents
    ]


def _render_sources(items: list[dict]) -> None:
    if not items:
        return
    with st.expander(f"Sources ({len(items)})"):
        for item in items:
            location = item.get("locator") or ""
            chunk = item.get("chunk")
            end = item.get("chunk_end")
            chunk_label = ""
            if chunk is not None:
                chunk_label = (
                    f"chunks {chunk}-{end}"
                    if end is not None and str(end) != str(chunk)
                    else f"chunk {chunk}"
                )
            detail = " · ".join(part for part in (location, chunk_label) if part)
            st.markdown(f"**{item['source']}**" + (f" — {detail}" if detail else ""))
            st.caption(item.get("preview", ""))


def _ingest_files(files, existing: set[str]):
    added, skipped, errors = [], [], []
    for upload in files:
        try:
            loaded = load_file(upload.name, upload.getvalue())
            if loaded.source_id in existing:
                skipped.append(upload.name)
                continue
            chunks = chunk_documents(loaded.documents, loaded.source_id)
            index_source(loaded, chunks)
        except Exception as error:
            errors.append(f"{upload.name}: {error}")
            continue
        existing.add(loaded.source_id)
        added.append(f"{upload.name} ({len(chunks)} chunks)")
    return added, skipped, errors


def _ingest_urls(urls: list[str], existing: set[str], loader):
    added, skipped, errors = [], [], []
    for url in urls:
        try:
            loaded = loader(url)
            if loaded.source_id in existing:
                skipped.append(loaded.source_name)
                continue
            chunks = chunk_documents(loaded.documents, loaded.source_id)
            index_source(loaded, chunks)
        except Exception as error:
            errors.append(f"{url}: {error}")
            continue
        existing.add(loaded.source_id)
        added.append(f"{loaded.source_name} ({len(chunks)} chunks)")
    return added, skipped, errors


def _report(added, skipped, errors) -> None:
    flash = []
    if added:
        flash.append(("success", "Added:\n" + "\n".join(f"- {item}" for item in added)))
    if skipped:
        flash.append(
            ("info", "Already indexed, skipped:\n" + "\n".join(f"- {item}" for item in skipped))
        )
    if errors:
        flash.append(("error", "Could not index:\n" + "\n".join(f"- {item}" for item in errors)))
    st.session_state["_flash"] = flash


def _detail(
    *,
    retrieval: str | None,
    latency: float | None,
    ttft: float | None,
    cost: float | None,
    attempts: int | None,
    multi_query: bool = False,
    parent_expansion: bool = False,
) -> str:
    parts = [retrieval or ""]
    if ttft is not None:
        parts.append(f"TTFT {ttft:.2f}s")
    if latency is not None:
        parts.append(f"total {latency:.2f}s")
    if cost is not None:
        parts.append(format_cost(cost))
    if attempts:
        parts.append(f"{attempts} retrieval attempt(s)")
    if multi_query:
        parts.append("multi-query")
    if parent_expansion:
        parts.append("parent windows")
    return " · ".join(part for part in parts if part)


data = st.session_state.get("chat_data")
if data is None:
    data = load_sessions()
    st.session_state["chat_data"] = data

if not data["sessions"]:
    session = create_session()
    data["sessions"][session["id"]] = session
    data["current"] = session["id"]
if data.get("current") not in data["sessions"]:
    data["current"] = max(
        data["sessions"], key=lambda key: data["sessions"][key].get("created_at", 0)
    )

for kind, message in st.session_state.pop("_flash", []):
    getattr(st, kind)(message)

st.session_state.setdefault("upload_key", 0)
sources = list_sources()
source_ids = [source.source_id for source in sources]
all_types = sorted({source.type for source in sources})
scope: list[str] = []


with st.sidebar:
    st.header("Knowledge base")

    source_mode = st.radio(
        "Add from",
        ["Files", "YouTube", "Website"],
        horizontal=True,
    )

    with st.form("add-sources"):
        uploads = []
        urls = ""
        if source_mode == "Files":
            uploads = st.file_uploader(
                "Upload files",
                type=[suffix.lstrip(".") for suffix in sorted(SUPPORTED_SUFFIXES)],
                accept_multiple_files=True,
                key=f"uploads-{st.session_state['upload_key']}",
                help="PDF, Excel, CSV, Word, text, Markdown, or image files.",
            )
        elif source_mode == "YouTube":
            urls = st.text_area(
                "YouTube URLs",
                height=110,
                placeholder="One URL per line\nhttps://www.youtube.com/watch?v=...",
            )
        else:
            urls = st.text_area(
                "Website URLs",
                height=110,
                placeholder="One URL per line\nhttps://example.com/article",
                help="Public http(s) pages. HTML is cleaned to readable text; PDF links work too.",
            )
        submitted = st.form_submit_button(
            "Add to knowledge base", type="primary", use_container_width=True
        )

    if submitted:
        existing = set(source_ids)
        if source_mode == "Files":
            if uploads:
                added, skipped, errors = _ingest_files(uploads, existing)
            else:
                added, skipped, errors = [], [], ["Choose at least one file."]
        else:
            targets = [line.strip() for line in urls.splitlines() if line.strip()]
            if targets:
                loader = load_youtube if source_mode == "YouTube" else load_website
                added, skipped, errors = _ingest_urls(targets, existing, loader)
            else:
                added, skipped, errors = [], [], [f"Paste at least one {source_mode} URL."]

        _report(added, skipped, errors)
        if added:
            st.session_state["upload_key"] += 1
        st.rerun()

    st.divider()
    st.subheader(f"Indexed sources ({len(sources)})")
    if not sources:
        st.caption("Nothing indexed yet.")
    else:
        use_all_col, use_none_col = st.columns(2)

        def _use_all_sources() -> None:
            for source_id in source_ids:
                st.session_state[f"use-{source_id}"] = True

        def _use_none_sources() -> None:
            for source_id in source_ids:
                st.session_state[f"use-{source_id}"] = False

        use_all_col.button(
            "Use all",
            on_click=_use_all_sources,
            use_container_width=True,
        )
        use_none_col.button(
            "Use none",
            on_click=_use_none_sources,
            use_container_width=True,
        )
        st.caption("Uncheck a source to exclude its vectors. Nothing is deleted.")

    for source_id in source_ids:
        st.session_state.setdefault(f"use-{source_id}", True)

    for source in sources:
        use_col, name_col, action_col = st.columns([0.14, 0.66, 0.2])
        enabled = use_col.checkbox(
            f"Use {source.name}",
            key=f"use-{source.source_id}",
            label_visibility="collapsed",
            help="Include this source in answers",
        )
        if enabled:
            scope.append(source.source_id)
        name_col.markdown(
            f"**{_short(source.name, 26)}**  \n"
            f"`{source.type}` · {source.chunks} chunks"
        )
        if action_col.button(
            "✕",
            key=f"delete-{source.source_id}",
            help=f"Remove {source.name}",
            use_container_width=True,
        ):
            delete_source(source.source_id)
            st.rerun()

    if sources and st.button(
        "Remove all sources", use_container_width=True, type="secondary"
    ):
        delete_all_sources()
        st.rerun()

    st.divider()
    st.subheader("Retrieval")

    selected_types = st.multiselect(
        "Source types",
        options=all_types,
        default=all_types,
        format_func=lambda value: str(value).replace("_", " ").title(),
        help="Metadata filter applied to Chroma and BM25. Clear it to disable retrieval.",
    )
    source_types = (
        []
        if not selected_types
        else None
        if set(selected_types) == set(all_types)
        else selected_types
    )

    retrieval_label = st.radio(
        "Retrieval mode",
        list(RETRIEVAL_OPTIONS),
        index=0,
        help=(
            "Vector only: dense similarity. Hybrid: dense + BM25 with reciprocal "
            "rank fusion. Hybrid + rerank: adds a cross-encoder over the fused candidates."
        ),
    )
    retrieval_mode = RETRIEVAL_OPTIONS[retrieval_label]

    multi_query = st.toggle(
        "Multi-query + decomposition",
        value=False,
        help=(
            "One extra LLM call creates paraphrases and sub-questions. Each is "
            "retrieved, then the results are fused with RRF."
        ),
    )
    parent_expansion = st.toggle(
        "Expand hits to parent windows",
        value=DEFAULT_PARENT_EXPANSION,
        help=(
            "Rank small chunks, then send the surrounding parent window to the "
            "generator. Existing chunks without parent metadata are unchanged."
        ),
    )
    st.caption(
        "Rerank scores each (query, chunk) pair; parent expansion then widens the "
        "context around each hit."
    )

    st.divider()
    st.subheader("Chats")
    if st.button("New chat", use_container_width=True):
        session = create_session()
        data["sessions"][session["id"]] = session
        data["current"] = session["id"]
        save_sessions(data)
        st.rerun()

    for session_id, chat_session in sorted(
        data["sessions"].items(),
        key=lambda item: item[1].get("created_at", 0),
        reverse=True,
    ):
        open_col, drop_col = st.columns([0.8, 0.2])
        active = session_id == data["current"]
        if open_col.button(
            ("● " if active else "") + _short(chat_session.get("title", "New chat"), 24),
            key=f"open-{session_id}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            data["current"] = session_id
            save_sessions(data)
            st.rerun()
        if drop_col.button(
            "✕", key=f"drop-{session_id}", help="Delete this chat", use_container_width=True
        ):
            data["sessions"].pop(session_id, None)
            delete_thread(session_id)
            st.session_state.get("seeded_threads", set()).discard(session_id)
            if not data["sessions"]:
                fresh = create_session()
                data["sessions"][fresh["id"]] = fresh
                data["current"] = fresh["id"]
            elif data["current"] == session_id:
                data["current"] = max(
                    data["sessions"],
                    key=lambda key: data["sessions"][key].get("created_at", 0),
                )
            save_sessions(data)
            st.rerun()


session = data["sessions"][data["current"]]

# Existing installs have JSON history but no graph checkpoints; import it once.
seeded_threads = st.session_state.setdefault("seeded_threads", set())
if session["id"] not in seeded_threads:
    seed_thread(session["id"], session.get("messages", []))
    seeded_threads.add(session["id"])

total_cost = sum(
    float(message.get("cost_usd") or 0.0)
    for message in session["messages"]
    if message.get("role") == "assistant"
)
st.title("Multi-source RAG assistant")
caption = f"{len(sources)} source(s) · {len(session['messages'])} message(s)"
if total_cost:
    caption += f" · {format_cost(total_cost)} estimated"
st.caption(caption)

scope_ready = bool(sources) and bool(scope) and bool(selected_types)
if not sources:
    st.info("Add a PDF, spreadsheet, image, website, or YouTube video in the sidebar to get started.")
elif not scope:
    st.warning(
        "No sources are selected for answers. Select at least one source in the sidebar, "
        "or add a new one."
    )
elif not selected_types:
    st.warning(
        "No source types are selected. Choose at least one type in the Retrieval section."
    )

for message in session["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        _render_sources(message.get("sources", []))
        if message.get("role") == "assistant":
            detail = _detail(
                retrieval=message.get("retrieval"),
                latency=message.get("latency"),
                ttft=message.get("ttft"),
                cost=message.get("cost_usd"),
                attempts=message.get("attempts"),
            )
            if detail:
                st.caption(detail)
            flags = message.get("guardrail_flags") or []
            if flags:
                st.caption("⚠ Guardrail flags: " + ", ".join(flags))

question = st.chat_input(
    "Ask across your sources",
    disabled=not scope_ready,
)
if question:
    session["messages"].append({"role": "user", "content": question})
    touch_title(session, question)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        handle = stream_answer_question(
            question,
            thread_id=session["id"],
            source_ids=scope,
            source_types=source_types,
            mode=retrieval_mode,
            multi_query=multi_query,
            parent_expansion=parent_expansion,
        )
        status_slot = st.empty()
        status_slot.caption("Retrieving and generating…")
        slot = st.empty()
        try:
            with slot.container():
                st.write_stream(handle)
        except Exception as error:
            status_slot.empty()
            slot.empty()
            st.error(f"Answer generation failed: {friendly_error(error)}")
            save_sessions(data)
            st.stop()
        status_slot.empty()

        meta = handle.result.meta or {}
        answer = handle.result.answer
        documents = handle.result.documents
        flags = meta.get("guardrail_flags") or []

        if meta.get("blocked"):
            slot.empty()
            st.warning(
                "The question was blocked by the input guardrail. "
                "The safe response is shown below."
            )
            st.markdown(answer)
        elif "output_blocked" in flags:
            slot.empty()
            st.warning(
                "The answer was replaced by the output guardrail. "
                "The safe response is shown below."
            )
            st.markdown(answer)
        elif not answer:
            st.warning("The model returned an empty answer. Try rephrasing the question.")

        payload = _payload(documents)
        _render_sources(payload)
        latency = meta.get("timings", {}).get("total")
        detail = _detail(
            retrieval=retrieval_label,
            latency=latency,
            ttft=meta.get("ttft_s"),
            cost=meta.get("cost_usd"),
            attempts=meta.get("attempts"),
            multi_query=multi_query,
            parent_expansion=parent_expansion,
        )
        if detail:
            st.caption(detail)
        if meta.get("context_flags"):
            st.caption(
                "⚠ Retrieved text contained instruction-like phrases. It was "
                "treated as untrusted data, not followed."
            )
        if flags:
            st.caption("⚠ Guardrail flags: " + ", ".join(sorted(set(flags))))

    session["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": payload,
            "retrieval": retrieval_label,
            "latency": round(latency, 2) if latency is not None else None,
            "attempts": meta.get("attempts"),
            "ttft": meta.get("ttft_s"),
            "cost_usd": meta.get("cost_usd"),
            "usage": meta.get("usage") or {},
            "guardrail_flags": flags,
            "created_at": time.time(),
        }
    )
    save_sessions(data)
    st.rerun()
