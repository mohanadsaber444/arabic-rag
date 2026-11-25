from app.domain.models import RetrievedChunk


def build_context(hits: list[RetrievedChunk],
                  max_chars: int = 6000) -> tuple[str, list[RetrievedChunk]]:
    """Return the context text and the chunks actually used, numbered from 1."""
    ordered = sorted(hits, key=lambda h: (h.chunk.document_id, h.chunk.ordinal))
    used: list[RetrievedChunk] = []
    blocks: list[str] = []
    total = 0
    for hit in ordered:
        c = hit.chunk
        label = f"[{len(used) + 1}] {c.filename}" + (f", page {c.page}" if c.page else "")
        block = f"{label}\n{c.text}"
        if total + len(block) > max_chars and used:
            break
        used.append(hit)
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks), used
