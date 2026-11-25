import contextlib
import hashlib
import logging
import uuid
from pathlib import Path

import app.ingestion.parsers  # noqa: F401 (importing registers the parsers)
from app.domain.errors import DuplicateDocumentError, EmptyDocumentError, NeedsOcrError
from app.domain.models import Document, DocumentStatus
from app.domain.ports import DocumentRepo, Embedder, VectorStore
from app.ingestion.chunking import chunk_document
from app.ingestion.detector import validate_upload
from app.ingestion.normalize import normalize_for_embedding
from app.ingestion.parsers.base import get_parser

log = logging.getLogger(__name__)


class IngestDocument:
    def __init__(
        self,
        repo: DocumentRepo,
        store: VectorStore,
        embedder: Embedder,
        upload_dir: Path,
        max_upload_mb: int,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self._repo, self._store, self._embedder = repo, store, embedder
        self._upload_dir = upload_dir
        self._max_bytes = max_upload_mb * 1024 * 1024
        self._size, self._overlap = chunk_size, chunk_overlap

    def execute(self, collection: str, filename: str, data: bytes) -> Document:
        ext = validate_upload(filename, data[:8], len(data), self._max_bytes)
        content_hash = hashlib.sha256(data).hexdigest()
        if self._repo.find_by_hash(collection, content_hash):
            raise DuplicateDocumentError("This exact file is already in the collection")
        previous = self._repo.find_latest_by_filename(collection, filename)
        doc = Document(
            id=str(uuid.uuid4()),
            collection=collection,
            filename=filename,
            content_hash=content_hash,
            version=(previous.version + 1) if previous else 1,
        )
        self._repo.add(doc)
        try:
            path = self._upload_dir / collection / f"{doc.id}{ext}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            parsed = get_parser(ext).parse(path)
            if parsed.needs_ocr:
                raise NeedsOcrError("This PDF looks scanned; OCR is not enabled yet")
            chunks = chunk_document(doc, parsed, self._size, self._overlap)
            if not chunks:
                raise EmptyDocumentError("No text could be extracted from this file")
            vectors = self._embedder.embed_texts([normalize_for_embedding(c.text) for c in chunks])
            self._store.ensure_collection(
                collection, self._embedder.dimension, self._embedder.model_name
            )
            self._store.upsert(collection, chunks, vectors, self._embedder.model_name)
            doc.status = DocumentStatus.READY
            self._repo.update(doc)
            if previous:  # replace the old version only after the new one is searchable
                self._store.delete_document(collection, previous.id)
                self._repo.delete(previous.id)
            log.info("ingested document=%s chunks=%d version=%d", doc.id, len(chunks), doc.version)
            return doc
        except Exception as exc:
            doc.status, doc.error = DocumentStatus.FAILED, str(exc)
            self._repo.update(doc)
            with contextlib.suppress(Exception):  # remove any partial vectors
                self._store.delete_document(collection, doc.id)
            log.warning("ingestion failed document=%s error=%s", doc.id, type(exc).__name__)
            raise
