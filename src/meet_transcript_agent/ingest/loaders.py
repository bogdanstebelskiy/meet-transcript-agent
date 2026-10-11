import json
from pathlib import Path

from meet_transcript_agent.config import Settings
from meet_transcript_agent.config import settings as default_settings
from meet_transcript_agent.domain import (
    TRANSCRIPT_CONTENT_TYPE,
    Document,
    doc_id_for_transcript,
    hash_text,
)
from meet_transcript_agent.transcribe import transcribe

AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".ogg", ".flac", ".m4a"})


def _format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


class TranscriptLoader:
    def supports(self, path: Path | str) -> bool:
        return Path(path).suffix.lower() in AUDIO_EXTENSIONS

    def load(
        self,
        path: Path | str,
        meeting_meta: dict,
        settings: Settings = default_settings,
    ) -> Document:
        path = Path(path)
        speaker = path.stem

        segments = transcribe(path, settings)

        lines = [
            f"[{_format_timestamp(seg.start)}] {seg.text}"
            for seg in segments
        ]
        transcript_text = "\n".join(lines)

        segments_json = json.dumps([s.model_dump() for s in segments])

        return Document(
            doc_id=doc_id_for_transcript(meeting_meta["meeting_id"], speaker),
            source_uri=str(path.resolve()),
            content_hash=hash_text(transcript_text),
            content_type=TRANSCRIPT_CONTENT_TYPE,
            text=transcript_text,
            metadata={
                "meeting_id": meeting_meta["meeting_id"],
                "meeting_title": meeting_meta.get("title", ""),
                "meeting_date": meeting_meta.get("date", ""),
                "speaker": speaker,
                "segments_json": segments_json,
            },
        )
