import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_answer, get_ingest, get_manage
from app.domain.errors import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    UnsupportedFileTypeError,
)
from app.domain.models import Answer, Chunk, Citation, Document, DocumentStatus, RetrievedChunk
from app.main import app


def make_doc(collection="hr", filename="a.txt"):
    return Document(
        id="d1",
        collection=collection,
        filename=filename,
        content_hash="h",
        status=DocumentStatus.READY,
    )


class FakeIngest:
    def execute(self, collection, filename, data):
        if filename.endswith(".exe"):
            raise UnsupportedFileTypeError("Unsupported file type: .exe")
        if filename == "dup.txt":
            raise DuplicateDocumentError("This exact file is already in the collection")
        return make_doc(collection, filename)


class FakeManage:
    def list(self, collection):
        return [make_doc(collection)]

    def delete(self, collection, doc_id):
        if doc_id == "nope":
            raise DocumentNotFoundError(doc_id)


class FakeAnswer:
    def execute(self, collection, question, document_ids=None):
        chunk = Chunk(
            id="c1",
            document_id="d1",
            document_version=1,
            collection=collection,
            filename="a.txt",
            ordinal=0,
            page=1,
            text="30 days",
        )
        return Answer(
            text="30 days [1].",
            citations=[Citation(chunk_id="c1", filename="a.txt", page=1, quote="30 days")],
            sources=[RetrievedChunk(chunk=chunk, score=0.9)],
        )


@pytest.fixture
def client():
    app.dependency_overrides[get_ingest] = FakeIngest
    app.dependency_overrides[get_manage] = FakeManage
    app.dependency_overrides[get_answer] = FakeAnswer
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client, name="a.txt", collection="hr"):
    return client.post(f"/collections/{collection}/documents", files={"file": (name, b"hello")})


def test_upload_returns_201_and_ready_status(client):
    r = upload(client)
    assert r.status_code == 201 and r.json()["status"] == "ready"


def test_unsupported_type_is_415(client):
    r = upload(client, "run.exe")
    assert r.status_code == 415 and r.json()["error"] == "UnsupportedFileTypeError"


def test_duplicate_is_409(client):
    assert upload(client, "dup.txt").status_code == 409


def test_list_documents(client):
    r = client.get("/collections/hr/documents")
    assert r.status_code == 200 and r.json()[0]["filename"] == "a.txt"


def test_delete_ok_and_unknown_id(client):
    assert client.delete("/collections/hr/documents/d1").status_code == 204
    assert client.delete("/collections/hr/documents/nope").status_code == 404


def test_invalid_collection_name_is_422(client):
    assert upload(client, collection="UP").status_code == 422


def test_query_returns_answer_with_citations_and_sources(client):
    r = client.post("/collections/hr/query", json={"question": "كم مدة الإشعار؟"})
    body = r.json()
    assert r.status_code == 200 and body["grounded"] is True
    assert body["citations"][0]["filename"] == "a.txt" and body["sources"][0]["index"] == 1


def test_empty_question_is_422(client):
    assert client.post("/collections/hr/query", json={"question": ""}).status_code == 422


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_ready_is_503_when_services_are_down(client, monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr("app.api.routes.health.httpx.get", boom)
    r = client.get("/ready")
    assert r.status_code == 503 and r.json() == {"qdrant": False, "ollama": False}


def test_ready_is_200_when_services_answer(client, monkeypatch):
    monkeypatch.setattr("app.api.routes.health.httpx.get", lambda *a, **k: httpx.Response(200))
    r = client.get("/ready")
    assert r.status_code == 200 and all(r.json().values())
