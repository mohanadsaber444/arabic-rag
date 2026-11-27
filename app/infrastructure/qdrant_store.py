from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.domain.errors import EmbeddingModelMismatchError
from app.domain.models import Chunk, RetrievedChunk


class QdrantVectorStore:
    def __init__(self, url: str):
        self._client = QdrantClient(url=url)

    def ensure_collection(self, name: str, dimension: int, model_name: str) -> None:
        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            for field in ("document_id", "language"):
                self._client.create_payload_index(name, field, PayloadSchemaType.KEYWORD)
            return
        existing = self._client.get_collection(name).config.params.vectors.size
        if existing != dimension:
            raise EmbeddingModelMismatchError(
                f"Collection '{name}' uses {existing} dimensions; model gives {dimension}. Re-embed."
            )
        points, _ = self._client.scroll(name, limit=1, with_payload=["embedding_model"])
        if points and points[0].payload.get("embedding_model") != model_name:
            raise EmbeddingModelMismatchError(
                f"Collection '{name}' was built with a different embedding model. Re-embed."
            )

    def upsert(
        self, collection: str, chunks: list[Chunk], vectors: list[list[float]], model_name: str = ""
    ) -> None:
        points = [
            PointStruct(
                id=c.id, vector=v, payload={**c.model_dump(), "embedding_model": model_name}
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=collection, points=points)

    def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if not self._client.collection_exists(collection):
            return []  # nothing ingested yet: no hits, so the answer is a clean refusal
        query_filter = None
        if document_ids:
            query_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchAny(any=document_ids))]
            )
        result = self._client.query_points(
            collection_name=collection,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                chunk=Chunk(**{k: v for k, v in p.payload.items() if k != "embedding_model"}),
                score=p.score,
            )
            for p in result.points
        ]

    def delete_document(self, collection: str, document_id: str) -> None:
        selector = FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )
        )
        self._client.delete(collection_name=collection, points_selector=selector)
