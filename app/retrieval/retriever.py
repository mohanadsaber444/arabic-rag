from app.domain.models import RetrievedChunk
from app.domain.ports import Embedder, VectorStore
from app.ingestion.normalize import normalize_for_embedding


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = 5,
                 min_score: float = 0.0):
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._min_score = min_score

    def retrieve(self, collection: str, question: str,
                 document_ids: list[str] | None = None) -> list[RetrievedChunk]:
        vector = self._embedder.embed_query(normalize_for_embedding(question))
        hits = self._store.search(collection, vector, self._top_k, document_ids)
        return [h for h in hits if h.score >= self._min_score]
