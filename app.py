import streamlit as st

from chunking import create_chunks
from rag import answer_question
from transcript import get_transcript, get_video_id
from vector_store import store_transcript


st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="▶")
st.title("YouTube RAG Chatbot")
st.caption("Index a video transcript, then ask questions grounded in its content.")

with st.sidebar:
    st.header("Index a video")
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("Index transcript", type="primary", use_container_width=True):
        if not url.strip():
            st.error("Enter a YouTube URL first.")
        else:
            try:
                with st.spinner("Fetching and indexing transcript..."):
                    video_id = get_video_id(url)
                    chunks = create_chunks(get_transcript(video_id))
                    count = store_transcript(video_id, chunks)
                st.session_state["video_id"] = video_id
                st.success(f"Indexed {count} chunks.")
            except Exception as error:
                st.error(str(error))

video_id = st.session_state.get("video_id")
if not video_id:
    st.info("Use the sidebar to index a video before asking a question.")
else:
    st.write(f"Active video: `{video_id}`")
    question = st.chat_input("Ask about the transcript")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching transcript..."):
                answer, sources = answer_question(question, video_id=video_id)
            st.write(answer)
            with st.expander("Sources"):
                for source in sources:
                    chunk_index = source.metadata.get("chunk_index", "?")
                    st.write(f"Chunk {chunk_index}: {source.page_content}")