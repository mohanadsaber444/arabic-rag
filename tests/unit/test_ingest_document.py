import pytest

from app.application.ingest_document import IngestDocument
from app.domain.errors import DuplicateDocumentError, EmptyDocumentError, UnsupportedFileTypeError
from app.domain.models import DocumentStatus
from app.infrastructure.sqlite_repo import SqliteDocumentRepo


class FakeEmbedder:
    model_name, dimension = "fake", 3

    def __init__(self, fail=False):
        self.fail = fail

    def embed_texts(self, texts):
        if self.fail:
            raise RuntimeError("embedding service down")
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


class FakeStore:
    def __init__(self):
        self.upserted, self.deleted, self.ensured = [], [], []

    def ensure_collection(self, name, dimension, model_name):
        self.ensured.append((name, dimension, model_name))

    def upsert(self, collection, chunks, vectors, model_name=""):
        self.upserted.extend(chunks)

    def search(self, collection, vector, top_k, document_ids=None):
        return []

    def delete_document(self, collection, document_id):
        self.deleted.append((collection, document_id))


def make(tmp_path, fail=False):
    repo = SqliteDocumentRepo(tmp_path / "rag.db")
    store = FakeStore()
    use_case = IngestDocument(repo, store, FakeEmbedder(fail), tmp_path / "uploads", 5, 200, 30)
    return use_case, repo, store


TEXT = ("هذا نص عربي للاختبار عن سياسة الإجازات. " * 6).encode("utf-8")


def test_text_file_becomes_ready_and_is_stored(tmp_path):
    use_case, repo, store = make(tmp_path)
    doc = use_case.execute("hr", "policy.txt", TEXT)
    assert doc.status == DocumentStatus.READY and doc.version == 1
    assert repo.get(doc.id).status == DocumentStatus.READY
    assert store.upserted and store.ensured == [("hr", 3, "fake")]


def test_same_file_twice_is_rejected(tmp_path):
    use_case, _, _ = make(tmp_path)
    use_case.execute("hr", "policy.txt", TEXT)
    with pytest.raises(DuplicateDocumentError):
        use_case.execute("hr", "policy.txt", TEXT)


def test_new_version_replaces_the_old_one_after_it_is_ready(tmp_path):
    use_case, repo, store = make(tmp_path)
    first = use_case.execute("hr", "policy.txt", TEXT)
    second = use_case.execute("hr", "policy.txt", TEXT + " فقرة جديدة.".encode())
    assert second.version == 2
    assert ("hr", first.id) in store.deleted
    assert repo.get(first.id) is None and repo.get(second.id) is not None


def test_embedding_failure_marks_failed_and_cleans_vectors(tmp_path):
    use_case, repo, store = make(tmp_path, fail=True)
    with pytest.raises(RuntimeError):
        use_case.execute("hr", "policy.txt", TEXT)
    doc = repo.list("hr")[0]
    assert doc.status == DocumentStatus.FAILED and "embedding" in doc.error
    assert ("hr", doc.id) in store.deleted


def test_unsupported_type_is_rejected_before_anything_is_saved(tmp_path):
    use_case, repo, _ = make(tmp_path)
    with pytest.raises(UnsupportedFileTypeError):
        use_case.execute("hr", "run.exe", b"MZ....")
    assert repo.list("hr") == []


def test_blank_text_file_fails_as_empty(tmp_path):
    use_case, repo, _ = make(tmp_path)
    with pytest.raises(EmptyDocumentError):
        use_case.execute("hr", "blank.txt", b"   \n  ")
    assert repo.list("hr")[0].status == DocumentStatus.FAILED
