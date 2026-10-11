import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from meet_transcript_agent.config import Settings
from meet_transcript_agent.config import settings as default_settings


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    doc_id: str
    source_uri: str
    content_hash: str


class Manifest:
    def __init__(self, data_dir: Path):
        self._db_path = data_dir / "document_agent.sqlite"

        data_dir.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(self._db_path)

        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents
            (
                doc_id TEXT PRIMARY KEY,
                source_uri TEXT NOT NULL,
                content_hash TEXT NOT NULL
            )
            """
        )

        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get(self, doc_id: str) -> ManifestRecord | None:
        cursor = self._connection.execute(
            """
            SELECT doc_id, source_uri, content_hash
            FROM documents
            WHERE doc_id = ?
            """,
            (doc_id,),
        )

        row = cursor.fetchone()
        if row is None:
            return None
        return ManifestRecord(*row)

    def upsert(self, doc_id: str, source_uri: str, content_hash: str) -> None:
        self._connection.execute(
            """
            INSERT INTO documents (doc_id, source_uri, content_hash)
            VALUES (?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                source_uri = excluded.source_uri,
                content_hash = excluded.content_hash
            """,
            (doc_id, source_uri, content_hash),
        )

        self._connection.commit()

    def delete(self, doc_id: str) -> None:
        self._connection.execute(
            """
            DELETE FROM documents
            WHERE doc_id = ?
            """,
            (doc_id,),
        )

        self._connection.commit()

    def all(self) -> list[ManifestRecord]:
        cursor = self._connection.execute(
            """
            SELECT doc_id, source_uri, content_hash
            FROM documents
            """
        )

        return [ManifestRecord(*row) for row in cursor.fetchall()]

    @staticmethod
    def default(settings: Settings = default_settings) -> "Manifest":
        return Manifest(settings.data_dir)
