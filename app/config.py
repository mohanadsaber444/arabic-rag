from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_")

    qdrant_url: str = "http://localhost:6333"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "CHOOSE_AFTER_TESTING"
    llm_model: str = "CHOOSE_AFTER_TESTING"
    gemini_api_key: str = ""
    groq_api_key: str = ""
    data_dir: Path = Path("data")
    max_upload_mb: int = 25
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()