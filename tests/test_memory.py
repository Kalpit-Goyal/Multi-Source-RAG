import uuid

import pytest

from rag import answer_question, delete_thread, seed_thread, thread_messages


@pytest.fixture
def thread_id():
    value = f"test-{uuid.uuid4().hex[:8]}"
    yield value
    delete_thread(value)


def test_seed_imports_json_history(thread_id):
    history = [
        {"role": "user", "content": "What is the leave policy?"},
        {"role": "assistant", "content": "Twenty days per year."},
    ]
    seed_thread(thread_id, history)

    messages = thread_messages(thread_id)
    assert [message.content for message in messages] == [
        "What is the leave policy?",
        "Twenty days per year.",
    ]


def test_seed_is_idempotent(thread_id):
    history = [{"role": "user", "content": "hello"}]
    seed_thread(thread_id, history)
    seed_thread(thread_id, history)
    assert len(thread_messages(thread_id)) == 1


def test_delete_thread_clears_memory(thread_id):
    seed_thread(thread_id, [{"role": "user", "content": "hello"}])
    delete_thread(thread_id)
    assert thread_messages(thread_id) == []


def test_unknown_mode_is_rejected(thread_id):
    with pytest.raises(ValueError):
        answer_question("hello", thread_id=thread_id, mode="nope")
