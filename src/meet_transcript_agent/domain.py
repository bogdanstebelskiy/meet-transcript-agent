import hashlib

from pydantic import BaseModel

MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]

TRANSCRIPT_CONTENT_TYPE = "text/transcript"


class Document(BaseModel):
    doc_id: str
    source_uri: str
    content_hash: str
    content_type: str
    text: str
    metadata: Metadata


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    index: int
    text: str
    metadata: Metadata


def doc_id_for_transcript(meeting_id: str, speaker: str) -> str:
    key = f"{meeting_id}:{speaker}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def chunk_id_for(doc_id: str, index: int) -> str:
    return f"{doc_id}:{index}"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
