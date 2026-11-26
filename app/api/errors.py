from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.domain import errors as e

_STATUS = {
    e.UnsupportedFileTypeError: 415,
    e.FileTooLargeError: 413,
    e.EmptyDocumentError: 422,
    e.NeedsOcrError: 422,
    e.DuplicateDocumentError: 409,
    e.EmbeddingModelMismatchError: 409,
    e.DocumentNotFoundError: 404,
    e.ExternalServiceError: 503,
}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(e.RagError)
    async def handle_rag_error(request: Request, exc: e.RagError) -> JSONResponse:
        return JSONResponse(
            status_code=_STATUS.get(type(exc), 400),
            content={"error": type(exc).__name__, "message": str(exc)},
        )

    @app.exception_handler(ConnectionError)
    async def handle_connection_error(request: Request, exc: ConnectionError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": "ServiceUnavailable",
                "message": "A required service (Ollama or Qdrant) is not reachable",
            },
        )
