import json
from unittest.mock import MagicMock, patch

from meet_transcript_agent.config import Settings
from meet_transcript_agent.domain import doc_id_for_transcript
from meet_transcript_agent.ingest.pipeline import IngestPipeline
from meet_transcript_agent.store.manifest import Manifest
from meet_transcript_agent.store.vector import VectorStore
from meet_transcript_agent.transcribe import Segment


def _fake_transcribe(audio_path, settings):
    return [
        Segment(start=0.0, end=5.0, text="Hello from this speaker"),
        Segment(start=5.0, end=10.0, text="Second sentence here"),
        Segment(start=10.0, end=15.0, text="Third sentence here"),
    ]


def _make_meeting_dir(tmp_path, speakers=("alice", "bob")):
    meeting_dir = tmp_path / "test-meeting"
    meeting_dir.mkdir()

    meta = {
        "meeting_id": "meeting-1",
        "title": "Test standup",
        "date": "2026-01-01",
        "participants": list(speakers),
    }
    (meeting_dir / "meeting.json").write_text(json.dumps(meta), encoding="utf-8")

    for speaker in speakers:
        (meeting_dir / f"{speaker}.wav").touch()

    return meeting_dir


def _make_pipeline(tmp_path, vector_store=None):
    vs = vector_store or MagicMock(spec=VectorStore)
    return IngestPipeline(
        settings=Settings(data_dir=tmp_path),
        vector_store=vs,
        manifest=Manifest(tmp_path / "manifest"),
    ), vs


@patch("meet_transcript_agent.ingest.loaders.transcribe", side_effect=_fake_transcribe)
def test_first_run_adds_all_speakers(mock_transcribe, tmp_path):
    meeting_dir = _make_meeting_dir(tmp_path, speakers=("alice", "bob"))
    pipeline, store = _make_pipeline(tmp_path)

    report = pipeline.ingest(meeting_dir)

    assert report.added == 2
    assert report.skipped == 0
    assert report.updated == 0
    assert store.add.call_count == 2


@patch("meet_transcript_agent.ingest.loaders.transcribe", side_effect=_fake_transcribe)
def test_second_run_skips_all(mock_transcribe, tmp_path):
    meeting_dir = _make_meeting_dir(tmp_path)
    pipeline, _ = _make_pipeline(tmp_path)

    pipeline.ingest(meeting_dir)
    report = pipeline.ingest(meeting_dir)

    assert report.added == 0
    assert report.skipped == 2
    assert report.updated == 0


@patch("meet_transcript_agent.ingest.loaders.transcribe")
def test_changed_transcript_reports_updated(mock_transcribe, tmp_path):
    meeting_dir = _make_meeting_dir(tmp_path, speakers=("alice",))
    pipeline, store = _make_pipeline(tmp_path)

    mock_transcribe.side_effect = lambda *a, **kw: [
        Segment(start=0.0, end=5.0, text="Original content"),
    ]
    pipeline.ingest(meeting_dir)

    store.reset_mock()
    mock_transcribe.side_effect = lambda *a, **kw: [
        Segment(start=0.0, end=5.0, text="Changed content"),
    ]
    report = pipeline.ingest(meeting_dir)

    assert report.updated == 1
    assert report.skipped == 0
    assert store.delete_doc.call_count == 1
    assert store.add.call_count == 1


@patch("meet_transcript_agent.ingest.loaders.transcribe", side_effect=_fake_transcribe)
def test_single_speaker_meeting(mock_transcribe, tmp_path):
    meeting_dir = _make_meeting_dir(tmp_path, speakers=("alice",))
    pipeline, store = _make_pipeline(tmp_path)

    report = pipeline.ingest(meeting_dir)

    assert report.added == 1
    assert store.add.call_count == 1
