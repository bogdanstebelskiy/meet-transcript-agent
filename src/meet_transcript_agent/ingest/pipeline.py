import json
from dataclasses import dataclass
from pathlib import Path

from meet_transcript_agent.config import Settings
from meet_transcript_agent.config import settings as default_settings
from meet_transcript_agent.ingest.chunkers import TranscriptChunker
from meet_transcript_agent.ingest.loaders import TranscriptLoader
from meet_transcript_agent.store.manifest import Manifest
from meet_transcript_agent.store.vector import VectorStore


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    skipped: int = 0


class IngestPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        vector_store: VectorStore | None = None,
        manifest: Manifest | None = None,
    ) -> None:
        self._settings = settings or default_settings
        self._loader = TranscriptLoader()
        self._chunker = TranscriptChunker()
        self._vector_store = vector_store or VectorStore.default(self._settings)
        self._manifest = manifest or Manifest.default(self._settings)

    def ingest(self, meeting_dir: Path | str) -> IngestReport:
        meeting_dir = Path(meeting_dir)
        report = IngestReport()

        meta_path = meeting_dir / "meeting.json"
        with open(meta_path, encoding="utf-8") as f:
            meeting_meta = json.load(f)

        for path in sorted(meeting_dir.iterdir()):
            if not path.is_file() or not self._loader.supports(path):
                continue

            doc = self._loader.load(path, meeting_meta, self._settings)

            existing = self._manifest.get(doc.doc_id)
            if existing is not None and existing.content_hash == doc.content_hash:
                report.skipped += 1
                continue

            chunks = self._chunker.split(doc, self._settings)

            if existing is not None:
                self._vector_store.delete_doc(doc.doc_id)
                report.updated += 1
            else:
                report.added += 1

            self._vector_store.add(chunks)
            self._manifest.upsert(doc.doc_id, doc.source_uri, doc.content_hash)

        return report
