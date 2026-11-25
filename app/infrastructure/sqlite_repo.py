import sqlite3
from pathlib import Path

from app.domain.models import Document

_COLUMNS = "id, collection, filename, content_hash, version, language, status, error, created_at"


class SqliteDocumentRepo:
    def __init__(self, path: Path):
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, collection TEXT NOT NULL, filename TEXT NOT NULL,
                content_hash TEXT NOT NULL, version INTEGER NOT NULL, language TEXT,
                status TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_docs_col ON documents(collection)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)  # one short-lived connection per call: thread safe
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _to_doc(row: sqlite3.Row) -> Document:
        return Document(**dict(row))

    def add(self, doc: Document) -> None:
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO documents ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    doc.id,
                    doc.collection,
                    doc.filename,
                    doc.content_hash,
                    doc.version,
                    doc.language,
                    doc.status.value,
                    doc.error,
                    doc.created_at.isoformat(),
                ),
            )

    def update(self, doc: Document) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status=?, error=?, language=? WHERE id=?",
                (doc.status.value, doc.error, doc.language, doc.id),
            )

    def get(self, doc_id: str) -> Document | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return self._to_doc(row) if row else None

    def find_by_hash(self, collection: str, content_hash: str) -> Document | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE collection=? AND content_hash=? AND status='ready'",
                (collection, content_hash),
            ).fetchone()
        return self._to_doc(row) if row else None

    def find_latest_by_filename(self, collection: str, filename: str) -> Document | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE collection=? AND filename=? AND status='ready' "
                "ORDER BY version DESC LIMIT 1",
                (collection, filename),
            ).fetchone()
        return self._to_doc(row) if row else None

    def list(self, collection: str) -> list[Document]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE collection=? ORDER BY created_at DESC", (collection,)
            ).fetchall()
        return [self._to_doc(r) for r in rows]

    def delete(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
