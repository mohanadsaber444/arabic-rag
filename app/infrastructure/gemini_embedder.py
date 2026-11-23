import logging
import time
from collections.abc import Callable

import httpx

from app.domain.errors import ExternalServiceError

log = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RETRY_STATUS = {429, 500, 502, 503, 504}  # busy or temporary errors: worth trying again


class GeminiEmbedder:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-embedding-001",
        dimension: int = 768,
        batch_size: int = 50,
        retries: int = 4,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model_name = model_name
        self.dimension = dimension
        self._api_key = api_key
        self._batch_size = batch_size
        self._retries = retries
        self._client = client or httpx.Client(timeout=60)
        self._sleep = sleep

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._embed(batch, "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        body = {
            "requests": [
                {
                    "model": f"models/{self.model_name}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": self.dimension,
                }
                for text in texts
            ]
        }
        url = f"{BASE_URL}/models/{self.model_name}:batchEmbedContents"
        embeddings = self._post(url, body).json().get("embeddings", [])
        if len(embeddings) != len(texts):
            raise ExternalServiceError("Gemini returned a different number of vectors than texts")
        return [item["values"] for item in embeddings]

    def _post(self, url: str, body: dict) -> httpx.Response:
        for attempt in range(self._retries + 1):
            try:
                response = self._client.post(url, json=body, headers={"x-goog-api-key": self._api_key})
            except httpx.HTTPError as exc:
                problem = f"network error ({type(exc).__name__})"
            else:
                if response.status_code == 200:
                    return response
                if response.status_code not in RETRY_STATUS:
                    raise ExternalServiceError(f"Gemini rejected the request (status {response.status_code})")
                problem = f"Gemini answered with status {response.status_code}"
            if attempt == self._retries:
                raise ExternalServiceError(f"{problem}; gave up after {attempt + 1} attempts")
            delay = 2**attempt  # 1, 2, 4, 8 seconds
            log.warning("%s; retrying in %s s", problem, delay)
            self._sleep(delay)
        raise ExternalServiceError("unreachable")  # keeps type checkers happy