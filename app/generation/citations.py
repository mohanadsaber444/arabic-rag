import re

from app.domain.models import Citation, RetrievedChunk
from app.ingestion.normalize import normalize_for_embedding

_MARKER = re.compile(r"\[(\d+)\]")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def extract_citations(answer: str, sources: list[RetrievedChunk]) -> list[Citation]:
    """Keep only markers that point to a real source; ignore invented numbers."""
    seen: list[int] = []
    citations: list[Citation] = []
    for match in _MARKER.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= len(sources) and n not in seen:
            seen.append(n)
            c = sources[n - 1].chunk
            citations.append(
                Citation(chunk_id=c.id, filename=c.filename, page=c.page, quote=c.text[:240])
            )
    return citations


def unsupported_numbers(answer: str, sources: list[RetrievedChunk]) -> list[str]:
    """Numbers in the answer that appear in no source: a cheap hallucination signal."""
    source_text = normalize_for_embedding(" ".join(s.chunk.text for s in sources))
    answer_text = normalize_for_embedding(_MARKER.sub("", answer))
    return [n for n in _NUMBER.findall(answer_text) if n not in source_text]
