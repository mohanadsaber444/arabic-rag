import pytest

from app.domain.models import Document, DocumentStatus
from app.infrastructure.sqlite_repo import SqliteDocumentRepo


def make_doc(
    id="d1", filename="a.pdf", content_hash="h1", version=1, status=DocumentStatus.PROCESSING
):
    return Document(
        id=id,
        collection="hr",
        filename=filename,
        content_hash=content_hash,
        version=version,
        status=status,
    )


@pytest.fixture
def repo(tmp_path):
    return SqliteDocumentRepo(tmp_path / "db" / "rag.db")


def test_add_get_and_update(repo):
    doc = make_doc()
    repo.add(doc)
    assert repo.get("d1").status == DocumentStatus.PROCESSING
    doc.status = DocumentStatus.READY
    repo.update(doc)
    assert repo.get("d1").status == DocumentStatus.READY
    assert repo.get("missing") is None


def test_find_by_hash_only_returns_ready_documents(repo):
    doc = make_doc()
    repo.add(doc)
    assert repo.find_by_hash("hr", "h1") is None
    doc.status = DocumentStatus.READY
    repo.update(doc)
    assert repo.find_by_hash("hr", "h1").id == "d1"
    assert repo.find_by_hash("other", "h1") is None


def test_latest_version_by_filename(repo):
    for i, version in enumerate((1, 2), start=1):
        repo.add(
            make_doc(id=f"d{i}", content_hash=f"h{i}", version=version, status=DocumentStatus.READY)
        )
    assert repo.find_latest_by_filename("hr", "a.pdf").version == 2


def test_list_and_delete(repo):
    repo.add(make_doc("d1", filename="a.pdf", content_hash="h1"))
    repo.add(make_doc("d2", filename="b.pdf", content_hash="h2"))
    assert {d.id for d in repo.list("hr")} == {"d1", "d2"}
    repo.delete("d1")
    assert [d.id for d in repo.list("hr")] == ["d2"]
