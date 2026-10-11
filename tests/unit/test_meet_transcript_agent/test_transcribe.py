import json
from pathlib import Path
from unittest.mock import patch

import pytest

from meet_transcript_agent.config import Settings
from meet_transcript_agent.transcribe import Segment, transcribe


def _whisper_output(segments):
    return {
        "transcription": [
            {
                "timestamps": {"from": "00:00:00,000", "to": "00:00:05,000"},
                "offsets": {"from": int(s["start"] * 1000), "to": int(s["end"] * 1000)},
                "text": s["text"],
            }
            for s in segments
        ]
    }


def test_transcribe_parses_whisper_output(tmp_path):
    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    whisper_data = _whisper_output([
        {"start": 0.0, "end": 5.0, "text": " Hello everyone"},
        {"start": 5.0, "end": 10.0, "text": " Let's begin"},
    ])

    def fake_run(cmd, **kwargs):
        output_prefix = None
        for i, arg in enumerate(cmd):
            if arg == "-of" and i + 1 < len(cmd):
                output_prefix = cmd[i + 1]
        json_path = Path(f"{output_prefix}.json")
        json_path.write_text(json.dumps(whisper_data), encoding="utf-8")

        from unittest.mock import MagicMock
        result = MagicMock()
        result.returncode = 0
        return result

    settings = Settings(
        whisper_bin=Path("whisper-cli"),
        whisper_model=Path("model.bin"),
        data_dir=tmp_path,
    )

    with patch("meet_transcript_agent.transcribe.subprocess.run", side_effect=fake_run):
        segments = transcribe(audio_file, settings)

    assert len(segments) == 2
    assert segments[0] == Segment(start=0.0, end=5.0, text="Hello everyone")
    assert segments[1] == Segment(start=5.0, end=10.0, text="Let's begin")


def test_transcribe_skips_empty_segments(tmp_path):
    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    whisper_data = _whisper_output([
        {"start": 0.0, "end": 5.0, "text": " Hello"},
        {"start": 5.0, "end": 10.0, "text": "   "},
    ])

    def fake_run(cmd, **kwargs):
        output_prefix = None
        for i, arg in enumerate(cmd):
            if arg == "-of" and i + 1 < len(cmd):
                output_prefix = cmd[i + 1]
        json_path = Path(f"{output_prefix}.json")
        json_path.write_text(json.dumps(whisper_data), encoding="utf-8")

        from unittest.mock import MagicMock
        result = MagicMock()
        result.returncode = 0
        return result

    settings = Settings(
        whisper_bin=Path("whisper-cli"),
        whisper_model=Path("model.bin"),
        data_dir=tmp_path,
    )

    with patch("meet_transcript_agent.transcribe.subprocess.run", side_effect=fake_run):
        segments = transcribe(audio_file, settings)

    assert len(segments) == 1


def test_transcribe_raises_on_nonzero_exit(tmp_path):
    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    def fake_run(cmd, **kwargs):
        from unittest.mock import MagicMock
        result = MagicMock()
        result.returncode = 1
        result.stderr = "model not found"
        return result

    settings = Settings(
        whisper_bin=Path("whisper-cli"),
        whisper_model=Path("model.bin"),
        data_dir=tmp_path,
    )

    with patch("meet_transcript_agent.transcribe.subprocess.run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match="model not found"):
            transcribe(audio_file, settings)
