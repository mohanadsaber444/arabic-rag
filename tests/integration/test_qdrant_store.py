import uuid

import pytest

from app.domain.errors import EmbeddingModelMismatchError
from app.domain.models import Chunk
from app.infrastructure.qdrant_store import QdrantVectorStore

pytestmark = pytest.mark.integration
DIM = 8


@pytest.fixture
def store_and_col():
    store = QdrantVectorStore("http://localhost:6333")
    col = f"test_{uuid.uuid4().hex[:8]}"
    yield store, col
    store._client.delete_collection(col)


def make_chunks(col, n=3):
    return [
        Chunk(
            id=str(uuid.uuid4()),
            document_id="d1",
            document_version=1,
            collection=col,
            filename="a.pdf",
            ordinal=i,
            page=1,
            text=f"chunk {i}",
        )
        for i in range(n)
    ]


def make_vectors(n=3):
    return [[1.0 if j == i else 0.0 for j in range(DIM)] for i in range(n)]


def test_upsert_search_filter_delete(store_and_col):
    store, col = store_and_col
    vecs = make_vectors()
    store.ensure_collection(col, DIM, "m1")
    store.upsert(col, make_chunks(col), vecs, "m1")
    top = store.search(col, vecs[0], top_k=3)[0]
    assert top.chunk.text == "chunk 0" and top.score > 0.99
    assert len(store.search(col, vecs[0], 3, document_ids=["d1"])) == 3
    assert store.search(col, vecs[0], 3, document_ids=["nope"]) == []
    store.delete_document(col, "d1")
    assert store.search(col, vecs[0], 3) == []


def test_dimension_mismatch_raises(store_and_col):
    store, col = store_and_col
    store.ensure_collection(col, DIM, "m1")
    with pytest.raises(EmbeddingModelMismatchError):
        store.ensure_collection(col, DIM * 2, "m1")


def test_model_mismatch_raises(store_and_col):
    store, col = store_and_col
    store.ensure_collection(col, DIM, "m1")
    store.upsert(col, make_chunks(col, 1), make_vectors(1), "m1")
    with pytest.raises(EmbeddingModelMismatchError):
        store.ensure_collection(col, DIM, "m2")


def test_search_on_a_missing_collection_returns_nothing():
    store = QdrantVectorStore("http://localhost:6333")
    assert store.search("missing_" + uuid.uuid4().hex[:8], make_vectors(1)[0], 3) == []
