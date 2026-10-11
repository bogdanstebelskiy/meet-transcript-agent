from langchain_chroma import Chroma

from meet_transcript_agent.config import Settings
from meet_transcript_agent.config import settings as default_settings
from meet_transcript_agent.domain import Chunk
from meet_transcript_agent.models import get_embeddings


class VectorStore:
    def __init__(self, embedding_function, settings: Settings = default_settings):
        self._collection = Chroma(
            collection_name="document_agent",
            embedding_function=embedding_function,
            persist_directory=str(settings.data_dir / "chroma"),
        )

    @staticmethod
    def default(settings: Settings = default_settings) -> "VectorStore":
        return VectorStore(embedding_function=get_embeddings(), settings=settings)

    def add(self, chunks: list[Chunk]):
        self._collection.add_texts(
            texts=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    **chunk.metadata,
                    "chunk_id": chunk.chunk_id,
                }
                for chunk in chunks
            ],
            ids=[chunk.chunk_id for chunk in chunks],
        )

    def delete_doc(self, doc_id: str):
        self._collection.delete(
            where={"doc_id": doc_id},
        )

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        results = self._collection.similarity_search_with_score(
            query,
            k=k,
        )

        return [
            (
                Chunk(
                    chunk_id=document.metadata["chunk_id"],
                    doc_id=document.metadata["doc_id"],
                    index=document.metadata["index"],
                    text=document.page_content,
                    metadata=document.metadata,
                ),
                score,
            )
            for document, score in results
        ]

    def count(self) -> int:
        return self._collection._collection.count()
