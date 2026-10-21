import streamlit as st

from chunking import chunk_documents
from loaders import SUPPORTED_SUFFIXES, load_file, load_youtube
from rag import answer_question
from sessions import create_session, load_sessions, save_sessions, touch_title
from vector_store import (
    delete_all_sources,
    delete_source,
    index_source,
    list_sources,
)


st.set_page_config(page_title="Multi-source RAG", page_icon="🗂", layout="wide")


def _short(text: str, limit: int = 30) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _payload(documents) -> list[dict]:
    return [
        {
            "source": document.metadata.get("source_name", "source"),
            "chunk": document.metadata.get("chunk_index"),
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
            detail = " · ".join(
                part
                for part in (location, f"chunk {chunk}" if chunk is not None else "")
                if part
            )
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


def _ingest_urls(urls: list[str], existing: set[str]):
    added, skipped, errors = [], [], []
    for url in urls:
        try:
            loaded = load_youtube(url)
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
labels = {source.source_id: f"{_short(source.name, 24)} ({source.chunks})" for source in sources}


with st.sidebar:
    st.header("Knowledge base")

    mode = st.radio("Source type", ["Files", "YouTube"], horizontal=True)

    with st.form("add-sources"):
        uploads = []
        urls = ""
        if mode == "Files":
            uploads = st.file_uploader(
                "Upload files",
                type=[suffix.lstrip(".") for suffix in sorted(SUPPORTED_SUFFIXES)],
                accept_multiple_files=True,
                key=f"uploads-{st.session_state['upload_key']}",
                help="PDF, Excel, CSV, Word, text, Markdown, or image files.",
            )
        else:
            urls = st.text_area(
                "YouTube URLs",
                height=110,
                placeholder="One URL per line\nhttps://www.youtube.com/watch?v=...",
            )
        submitted = st.form_submit_button(
            "Add to knowledge base", type="primary", use_container_width=True
        )

    if submitted:
        existing = set(source_ids)
        if mode == "Files":
            if uploads:
                added, skipped, errors = _ingest_files(uploads, existing)
            else:
                added, skipped, errors = [], [], ["Choose at least one file."]
        else:
            targets = [line.strip() for line in urls.splitlines() if line.strip()]
            if targets:
                added, skipped, errors = _ingest_urls(targets, existing)
            else:
                added, skipped, errors = [], [], ["Paste at least one YouTube URL."]

        _report(added, skipped, errors)
        if added:
            st.session_state["upload_key"] += 1
        st.rerun()

    st.divider()
    st.subheader(f"Indexed sources ({len(sources)})")
    if not sources:
        st.caption("Nothing indexed yet.")
    for source in sources:
        name_col, action_col = st.columns([0.8, 0.2])
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
    scope = st.multiselect(
        "Search in",
        options=source_ids,
        default=source_ids,
        format_func=lambda source_id: labels.get(source_id, source_id),
        help="Narrow retrieval to a subset of sources. Empty means all.",
    )
    scope = [source_id for source_id in scope if source_id in source_ids]

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

st.title("Multi-source RAG assistant")
st.caption(f"{len(sources)} source(s) · {len(session['messages'])} message(s)")

if not sources:
    st.info("Add a PDF, spreadsheet, image, or YouTube video in the sidebar to get started.")

for message in session["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        _render_sources(message.get("sources", []))

question = st.chat_input("Ask across your sources", disabled=not sources)
if question:
    session["messages"].append({"role": "user", "content": question})
    touch_title(session, question)

    with st.chat_message("user"):
        st.markdown(question)

    history = session["messages"][:-1]
    with st.chat_message("assistant"):
        with st.spinner("Searching your sources…"):
            answer, documents = answer_question(
                question,
                source_ids=scope or None,
                history=history,
            )
        st.markdown(answer)
        payload = _payload(documents)
        _render_sources(payload)

    session["messages"].append(
        {"role": "assistant", "content": answer, "sources": payload}
    )
    save_sessions(data)
    st.rerun()
