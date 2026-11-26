import pytest

from app.application.manage_documents import ManageDocuments
from app.domain.errors import DocumentNotFoundError
from app.domain.models import Document, DocumentStatus
from app.infrastructure.sqlite_repo import SqliteDocumentRepo


class FakeStore:
    def __init__(self):
        self.deleted = []

    def delete_document(self, collection, document_id):
        self.deleted.append((collection, document_id))


def setup(tmp_path):
    repo = SqliteDocumentRepo(tmp_path / "rag.db")
    store = FakeStore()
    upload_dir = tmp_path / "uploads"
    repo.add(
        Document(
            id="d1",
            collection="hr",
            filename="a.txt",
            content_hash="h",
            status=DocumentStatus.READY,
        )
    )
    file = upload_dir / "hr" / "d1.txt"
    file.parent.mkdir(parents=True)
    file.write_text("x")
    return ManageDocuments(repo, store, upload_dir), repo, store, file


def test_list_returns_only_the_collection_documents(tmp_path):
    use_case, _, _, _ = setup(tmp_path)
    assert [d.id for d in use_case.list("hr")] == ["d1"]
    assert use_case.list("other") == []


def test_delete_removes_vectors_file_and_row(tmp_path):
    use_case, repo, store, file = setup(tmp_path)
    use_case.delete("hr", "d1")
    assert store.deleted == [("hr", "d1")]
    assert not file.exists() and repo.get("d1") is None


def test_unknown_id_raises(tmp_path):
    use_case, _, _, _ = setup(tmp_path)
    with pytest.raises(DocumentNotFoundError):
        use_case.delete("hr", "nope")


def test_cannot_delete_from_another_collection(tmp_path):
    use_case, repo, store, file = setup(tmp_path)
    with pytest.raises(DocumentNotFoundError):
        use_case.delete("other", "d1")
    assert repo.get("d1") is not None and store.deleted == [] and file.exists()
