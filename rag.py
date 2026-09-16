from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from vector_store import DEFAULT_DB_DIR, search_transcript


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about YouTube transcripts. Use only the provided "
            "context. If the answer is not in the context, say you do not know. "
            "Keep the answer concise and cite the chunk number when useful.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def answer_question(
    question: str,
    persist_directory=DEFAULT_DB_DIR,
    video_id: str | None = None,
    k: int = 4,
) -> tuple[str, list]:
    """Answer a question and return the answer plus its source documents."""
    documents = search_transcript(
        question,
        persist_directory=persist_directory,
        video_id=video_id,
        k=k,
    )
    if not documents:
        return "I could not find any indexed transcript content.", []

    context = "\n\n".join(
        f"Chunk {document.metadata.get('chunk_index', '?')}: {document.page_content}"
        for document in documents
    )
    response = (PROMPT | ChatOpenAI(model="gpt-4o-mini", temperature=0)).invoke(
        {"context": context, "question": question}
    )
    return str(response.content), documents