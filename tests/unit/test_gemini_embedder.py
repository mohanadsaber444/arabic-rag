import json

import httpx
import pytest

from app.domain.errors import ExternalServiceError
from app.infrastructure.gemini_embedder import GeminiEmbedder


def make_embedder(handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiEmbedder("secret-key", client=client, sleep=kwargs.pop("sleep", lambda s: None), **kwargs)


def ok_reply(request):
    count = len(json.loads(request.content)["requests"])
    return httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2, 0.3]} for _ in range(count)]})


def test_texts_are_sent_in_batches_and_come_back_in_order():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return ok_reply(request)

    vectors = make_embedder(handler, batch_size=2).embed_texts(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5
    assert [len(call["requests"]) for call in calls] == [2, 2, 1]


def test_documents_and_queries_use_different_task_types():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["requests"][0]["taskType"])
        return ok_reply(request)

    embedder = make_embedder(handler)
    embedder.embed_texts(["نص"])
    embedder.embed_query("سؤال")
    assert seen == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


def test_requested_dimension_and_key_header_are_sent():
    captured = {}

    def handler(request):
        captured["key"] = request.headers["x-goog-api-key"]
        captured["body"] = json.loads(request.content)
        return ok_reply(request)

    make_embedder(handler, dimension=768).embed_query("x")
    assert captured["key"] == "secret-key"
    assert captured["body"]["requests"][0]["outputDimensionality"] == 768
    assert "secret-key" not in json.dumps(captured["body"])


def test_busy_service_is_retried_with_growing_delay():
    attempts, delays = [], []

    def handler(request):
        attempts.append(1)
        return httpx.Response(429) if len(attempts) < 3 else ok_reply(request)

    vector = make_embedder(handler, sleep=delays.append).embed_query("x")
    assert vector == [0.1, 0.2, 0.3]
    assert delays == [1, 2]


def test_gives_up_with_a_clear_error_after_the_retries():
    embedder = make_embedder(lambda request: httpx.Response(429), retries=2)
    with pytest.raises(ExternalServiceError, match="gave up after 3 attempts"):
        embedder.embed_query("x")


def test_a_rejected_request_is_not_retried():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(400)

    with pytest.raises(ExternalServiceError, match="400"):
        make_embedder(handler).embed_query("x")
    assert len(attempts) == 1


def test_empty_input_makes_no_request():
    def handler(request):
        raise AssertionError("no request expected")

    assert make_embedder(handler).embed_texts([]) == []