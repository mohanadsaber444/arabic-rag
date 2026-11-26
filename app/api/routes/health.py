import httpx
from fastapi import APIRouter, Response

from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}  # liveness: the process is up, no dependency checks


@router.get("/ready")
def ready(response: Response) -> dict:
    s = get_settings()
    checks: dict[str, bool] = {}
    for name, url in {"qdrant": f"{s.qdrant_url}/readyz", "ollama": s.ollama_url}.items():
        try:
            checks[name] = httpx.get(url, timeout=2).status_code < 500
        except httpx.HTTPError:
            checks[name] = False
    if not all(checks.values()):
        response.status_code = 503
    return checks
