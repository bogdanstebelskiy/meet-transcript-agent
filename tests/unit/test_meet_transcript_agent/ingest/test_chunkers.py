import json

from meet_transcript_agent.config import Settings
from meet_transcript_agent.domain import TRANSCRIPT_CONTENT_TYPE, Document, hash_text


from meet_transcript_agent.ingest.chunkers import TranscriptChunker


def _make_segments(count, duration=5.0):
    return [
        {"start": i * duration, "end": (i + 1) * duration, "text": f"Segment {i}"}
        for i in range(count)
    ]


def _make_doc(segments):
    text = "\n".join(f"[{int(s['start']//60):02d}:{int(s['start']%60):02d}] {s['text']}" for s in segments)
    return Document(
        doc_id="test-doc",
        source_uri="test.wav",
        content_hash=hash_text(text),
        content_type=TRANSCRIPT_CONTENT_TYPE,
        text=text,
        metadata={
            "meeting_id": "meeting-1",
            "meeting_title": "Test",
            "meeting_date": "2026-01-01",
            "speaker": "alice",
            "segments_json": json.dumps(segments),
        },
    )


def test_short_transcript_produces_one_chunk():
    segments = _make_segments(3, duration=5.0)
    doc = _make_doc(segments)
    settings = Settings(chunk_duration=30.0, chunk_overlap_duration=5.0, data_dir=".")

    chunks = TranscriptChunker().split(doc, settings)

    assert len(chunks) == 1
    assert chunks[0].metadata["speaker"] == "alice"
    assert chunks[0].metadata["meeting_id"] == "meeting-1"


def test_long_transcript_produces_multiple_chunks():
    segments = _make_segments(20, duration=5.0)
    doc = _make_doc(segments)
    settings = Settings(chunk_duration=30.0, chunk_overlap_duration=5.0, data_dir=".")

    chunks = TranscriptChunker().split(doc, settings)

    assert len(chunks) > 1


def test_chunks_have_correct_timestamps():
    segments = _make_segments(10, duration=5.0)
    doc = _make_doc(segments)
    settings = Settings(chunk_duration=30.0, chunk_overlap_duration=5.0, data_dir=".")

    chunks = TranscriptChunker().split(doc, settings)

    first = chunks[0]
    assert first.metadata["start_ts"] == 0.0
    assert first.metadata["end_ts"] > 0.0


def test_chunk_ids_are_sequential():
    segments = _make_segments(20, duration=5.0)
    doc = _make_doc(segments)
    settings = Settings(chunk_duration=30.0, chunk_overlap_duration=5.0, data_dir=".")

    chunks = TranscriptChunker().split(doc, settings)

    assert [c.chunk_id for c in chunks] == [f"test-doc:{i}" for i in range(len(chunks))]


def test_overlap_produces_shared_segments():
    segments = _make_segments(20, duration=5.0)
    doc = _make_doc(segments)
    settings = Settings(chunk_duration=30.0, chunk_overlap_duration=10.0, data_dir=".")

    chunks = TranscriptChunker().split(doc, settings)

    if len(chunks) >= 2:
        first_text = chunks[0].text
        second_text = chunks[1].text
        first_lines = set(first_text.splitlines())
        second_lines = set(second_text.splitlines())
        assert first_lines & second_lines


def test_empty_segments_produces_no_chunks():
    doc = Document(
        doc_id="test-doc",
        source_uri="test.wav",
        content_hash=hash_text(""),
        content_type=TRANSCRIPT_CONTENT_TYPE,
        text="",
        metadata={
            "meeting_id": "meeting-1",
            "meeting_title": "Test",
            "meeting_date": "2026-01-01",
            "speaker": "alice",
            "segments_json": "[]",
        },
    )
    settings = Settings(chunk_duration=30.0, chunk_overlap_duration=5.0, data_dir=".")

    chunks = TranscriptChunker().split(doc, settings)

    assert chunks == []
