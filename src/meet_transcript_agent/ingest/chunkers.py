import json

from meet_transcript_agent.config import Settings
from meet_transcript_agent.config import settings as default_settings
from meet_transcript_agent.domain import Chunk, Document, chunk_id_for


def _format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


class TranscriptChunker:
    def split(
        self, doc: Document, settings: Settings = default_settings
    ) -> list[Chunk]:
        segments = json.loads(doc.metadata["segments_json"])

        if not segments:
            return []

        duration = settings.chunk_duration
        overlap = settings.chunk_overlap_duration
        step = duration - overlap

        last_end = max(seg["end"] for seg in segments)
        window_start = 0.0
        chunks = []

        while window_start < last_end:
            window_end = window_start + duration

            window_segments = [
                seg for seg in segments
                if seg["end"] > window_start and seg["start"] < window_end
            ]

            if window_segments:
                lines = [
                    f"[{_format_timestamp(seg['start'])}] {seg['text']}"
                    for seg in window_segments
                ]
                chunk_text = "\n".join(lines)
                chunk_idx = len(chunks)

                actual_start = window_segments[0]["start"]
                actual_end = window_segments[-1]["end"]

                chunks.append(Chunk(
                    chunk_id=chunk_id_for(doc.doc_id, chunk_idx),
                    doc_id=doc.doc_id,
                    index=chunk_idx,
                    text=chunk_text,
                    metadata={
                        "meeting_id": doc.metadata["meeting_id"],
                        "meeting_title": doc.metadata["meeting_title"],
                        "meeting_date": doc.metadata["meeting_date"],
                        "speaker": doc.metadata["speaker"],
                        "start_ts": actual_start,
                        "end_ts": actual_end,
                        "doc_id": doc.doc_id,
                        "source_uri": doc.source_uri,
                        "index": chunk_idx,
                    },
                ))

            window_start += step

        return chunks
