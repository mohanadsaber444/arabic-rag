from app.domain.models import Chunk, RetrievedChunk
from app.retrieval.context import build_context
from app.retrieval.retriever import Retriever


def hit(doc, ordinal, score, text="نص", page=1):
    c = Chunk(id=f"{doc}-{ordinal}", document_id=doc, document_version=1, collection="t",
              filename=f"{doc}.pdf", ordinal=ordinal, page=page, text=text)
    return RetrievedChunk(chunk=c, score=score)


class FakeEmbedder:
    model_name, dimension = "fake", 3

    def __init__(self):
        self.queries = []

    def embed_texts(self, texts):
        return [[0.0] * 3 for _ in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0, 0.0, 0.0]


class FakeStore:
    def __init__(self, hits):
        self.hits, self.args = hits, None

    def search(self, collection, vector, top_k, document_ids=None):
        self.args = (collection, top_k, document_ids)
        return self.hits


def test_retriever_normalizes_query_and_filters_by_score():
    emb = FakeEmbedder()
    store = FakeStore([hit("a", 0, 0.9), hit("a", 1, 0.2)])
    result = Retriever(emb, store, top_k=5, min_score=0.5).retrieve("hr", "أحمد")
    assert emb.queries == ["احمد"]
    assert [h.score for h in result] == [0.9]
    assert store.args == ("hr", 5, None)


def test_no_hits_above_threshold_returns_empty():
    store = FakeStore([hit("a", 0, 0.1)])
    assert Retriever(FakeEmbedder(), store, min_score=0.5).retrieve("hr", "q") == []


def test_context_orders_by_document_and_position_and_numbers_sources():
    hits = [hit("b", 1, 0.9, "ثاني"), hit("a", 2, 0.8, "ثالث"), hit("a", 0, 0.7, "أول")]
    context, used = build_context(hits)
    assert [h.chunk.ordinal for h in used] == [0, 2, 1]
    assert context.index("[1] a.pdf") < context.index("[2] a.pdf") < context.index("[3] b.pdf")


def test_context_stops_at_budget_but_keeps_at_least_one():
    hits = [hit("a", i, 0.9, "x" * 100) for i in range(5)]
    _, used = build_context(hits, max_chars=250)
    assert 1 <= len(used) < 5
