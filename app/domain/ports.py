from typing import Protocol

from app.domain.models import Chunk, Document, RetrievedChunk


class Embedder(Protocol):
    model_name: str
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LLM(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...


class VectorStore(Protocol):
    def ensure_collection(self, name: str, dimension: int, model_name: str) -> None: ...

    def upsert(
        self,
        collection: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        model_name: str,
    ) -> None: ...

    def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]: ...

    def delete_document(self, collection: str, document_id: str) -> None: ...


class DocumentRepo(Protocol):
    def add(self, doc: Document) -> None: ...

    def update(self, doc: Document) -> None: ...

    def get(self, doc_id: str) -> Document | None: ...

    def find_by_hash(self, collection: str, content_hash: str) -> Document | None: ...

    def find_latest_by_filename(self, collection: str, filename: str) -> Document | None: ...

    def list(self, collection: str) -> list[Document]: ...

    def delete(self, doc_id: str) -> None: ...
