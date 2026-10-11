from pathlib import Path
from unittest.mock import patch

from meet_transcript_agent.config import Settings
from meet_transcript_agent.domain import TRANSCRIPT_CONTENT_TYPE, Document, doc_id_for_transcript
from meet_transcript_agent.ingest.loaders import TranscriptLoader
from meet_transcript_agent.transcribe import Segment


def _fake_transcribe(audio_path, settings):
    return [
        Segment(start=0.0, end=5.0, text="Hello everyone"),
        Segment(start=5.0, end=10.0, text="Let's get started"),
    ]


def test_supports_audio_extensions():
    loader = TranscriptLoader()
    assert loader.supports(Path("test.wav"))
    assert loader.supports(Path("test.mp3"))
    assert loader.supports(Path("test.ogg"))
    assert loader.supports(Path("test.flac"))
    assert loader.supports(Path("test.m4a"))
    assert not loader.supports(Path("test.txt"))
    assert not loader.supports(Path("test.json"))


def test_load_produces_document(tmp_path):
    audio = tmp_path / "alice.wav"
    audio.touch()

    meeting_meta = {
        "meeting_id": "test-meeting",
        "title": "Standup",
        "date": "2026-01-01",
    }
    settings = Settings(data_dir=tmp_path)

    with patch("meet_transcript_agent.ingest.loaders.transcribe", side_effect=_fake_transcribe):
        doc = TranscriptLoader().load(audio, meeting_meta, settings)

    assert isinstance(doc, Document)
    assert doc.content_type == TRANSCRIPT_CONTENT_TYPE
    assert doc.doc_id == doc_id_for_transcript("test-meeting", "alice")
    assert doc.metadata["speaker"] == "alice"
    assert doc.metadata["meeting_id"] == "test-meeting"
    assert doc.metadata["meeting_title"] == "Standup"
    assert "[00:00]" in doc.text
    assert "Hello everyone" in doc.text


def test_speaker_name_from_filename(tmp_path):
    audio = tmp_path / "bob.wav"
    audio.touch()

    meeting_meta = {"meeting_id": "m1", "title": "Test", "date": "2026-01-01"}
    settings = Settings(data_dir=tmp_path)

    with patch("meet_transcript_agent.ingest.loaders.transcribe", side_effect=_fake_transcribe):
        doc = TranscriptLoader().load(audio, meeting_meta, settings)

    assert doc.metadata["speaker"] == "bob"
