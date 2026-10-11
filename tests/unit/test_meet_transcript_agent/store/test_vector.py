from meet_transcript_agent.config import settings
from meet_transcript_agent.domain import Chunk
from meet_transcript_agent.models import get_embeddings
from meet_transcript_agent.store.vector import VectorStore


def make_chunk(chunk_id: str, doc_id: str, text: str, speaker: str = "alice") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        index=0,
        text=text,
        metadata={
            "doc_id": doc_id,
            "index": 0,
            "source_uri": f"{speaker}.wav",
            "speaker": speaker,
            "start_ts": 0.0,
            "end_ts": 30.0,
            "meeting_id": "test-meeting",
        },
    )


def make_store(tmp_path, monkeypatch) -> VectorStore:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return VectorStore(embedding_function=get_embeddings())


def test_search_finds_relevant_transcript(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)

    chunks = [
        make_chunk("m1-alice:0", "m1-alice", "We need to push the deadline to Friday", "alice"),
        make_chunk("m1-bob:0", "m1-bob", "The deployment pipeline is broken again", "bob"),
        make_chunk("m1-carol:0", "m1-carol", "I booked the conference room for Thursday", "carol"),
    ]
    store.add(chunks)

    results = store.search("What about the deadline?", k=3)

    assert results[0][0].doc_id == "m1-alice"


def test_delete_doc_removes_only_that_speakers_chunks(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)

    chunks = [
        make_chunk("m1-alice:0", "m1-alice", "First point from Alice", "alice"),
        make_chunk("m1-alice:1", "m1-alice", "Second point from Alice", "alice"),
        make_chunk("m1-bob:0", "m1-bob", "Point from Bob", "bob"),
    ]
    store.add(chunks)
    assert store.count() == 3

    store.delete_doc("m1-alice")
    assert store.count() == 1

    results = store._collection.get()
    assert results["ids"] == ["m1-bob:0"]


def test_data_survives_store_restart(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    store.add([make_chunk("m1-alice:0", "m1-alice", "Discussion about the API")])
    assert store.count() == 1

    new_store = make_store(tmp_path, monkeypatch)
    assert new_store.count() == 1

    results = new_store.search("API discussion", k=1)
    assert results[0][0].chunk_id == "m1-alice:0"


def test_re_adding_same_chunk_id_does_not_duplicate(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    chunk = make_chunk("m1-alice:0", "m1-alice", "Original text")
    store.add([chunk])
    store.add([chunk])
    assert store.count() == 1


def test_re_adding_same_chunk_id_updates_text(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    original = make_chunk("m1-alice:0", "m1-alice", "Original text")
    updated = make_chunk("m1-alice:0", "m1-alice", "Updated transcript text")
    store.add([original])
    store.add([updated])
    assert store.count() == 1

    results = store._collection.get(ids=["m1-alice:0"])
    assert results["documents"] == ["Updated transcript text"]
