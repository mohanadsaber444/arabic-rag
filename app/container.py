from functools import lru_cache

from app.application.answer_question import AnswerQuestion
from app.application.ingest_document import IngestDocument
from app.application.manage_documents import ManageDocuments
from app.config import Settings, get_settings
from app.generation.generator import AnswerGenerator
from app.infrastructure.ollama_embedder import OllamaEmbedder
from app.infrastructure.ollama_llm import OllamaLLM
from app.infrastructure.qdrant_store import QdrantVectorStore
from app.infrastructure.sqlite_repo import SqliteDocumentRepo
from app.retrieval.retriever import Retriever


class Container:
    def __init__(self, s: Settings):
        upload_dir = s.data_dir / "uploads"
        self.repo = SqliteDocumentRepo(s.data_dir / "rag.db")
        self.store = QdrantVectorStore(s.qdrant_url)
        self.embedder = OllamaEmbedder(s.ollama_url, s.embedding_model)
        self.llm = OllamaLLM(s.ollama_url, s.llm_model)
        self.retriever = Retriever(self.embedder, self.store, s.top_k)
        self.ingest = IngestDocument(
            self.repo,
            self.store,
            self.embedder,
            upload_dir,
            s.max_upload_mb,
            s.chunk_size,
            s.chunk_overlap,
        )
        self.manage = ManageDocuments(self.repo, self.store, upload_dir)
        self.answer = AnswerQuestion(self.retriever, AnswerGenerator(self.llm))


@lru_cache
def get_container() -> Container:
    return Container(get_settings())
