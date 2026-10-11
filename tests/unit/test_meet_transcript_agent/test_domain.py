from meet_transcript_agent.domain import chunk_id_for, doc_id_for_transcript, hash_text


def test_transcript_doc_id_is_stable():
    id1 = doc_id_for_transcript("meeting-1", "alice")
    id2 = doc_id_for_transcript("meeting-1", "alice")
    assert id1 == id2


def test_transcript_doc_id_differs_by_speaker():
    id_alice = doc_id_for_transcript("meeting-1", "alice")
    id_bob = doc_id_for_transcript("meeting-1", "bob")
    assert id_alice != id_bob


def test_transcript_doc_id_differs_by_meeting():
    id1 = doc_id_for_transcript("meeting-1", "alice")
    id2 = doc_id_for_transcript("meeting-2", "alice")
    assert id1 != id2


def test_chunk_id_format():
    assert chunk_id_for("abc123", 0) == "abc123:0"
    assert chunk_id_for("abc123", 5) == "abc123:5"


def test_hash_text_changes_when_text_changes():
    assert hash_text("Hello") != hash_text("Goodbye")


def test_hash_text_is_stable():
    assert hash_text("Hello") == hash_text("Hello")
