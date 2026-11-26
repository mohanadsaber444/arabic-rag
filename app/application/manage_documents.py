from pathlib import Path

from app.domain.errors import DocumentNotFoundError
from app.domain.models import Document
from app.domain.ports import DocumentRepo, VectorStore


class ManageDocuments:
    def __init__(self, repo: DocumentRepo, store: VectorStore, upload_dir: Path):
        self._repo, self._store, self._upload_dir = repo, store, upload_dir

    def list(self, collection: str) -> list[Document]:
        return self._repo.list(collection)

    def delete(self, collection: str, doc_id: str) -> None:
        doc = self._repo.get(doc_id)
        if doc is None or doc.collection != collection:  # also blocks cross-collection deletes
            raise DocumentNotFoundError(doc_id)
        self._store.delete_document(collection, doc_id)
        for file in (self._upload_dir / collection).glob(f"{doc_id}.*"):
            file.unlink(missing_ok=True)
        self._repo.delete(doc_id)
