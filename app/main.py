import logging

from fastapi import FastAPI

from app.api.errors import register_error_handlers
from app.api.routes import documents, health, query
from app.config import get_settings

logging.basicConfig(
    level=get_settings().log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

app = FastAPI(title="Arabic RAG", version="0.1.0")
register_error_handlers(app)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
