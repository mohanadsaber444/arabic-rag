from pydantic import BaseModel, Field

from app.domain.models import Answer, Document


class DocumentOut(BaseModel):
    id: str
    filename: str
    version: int
    status: str
    error: str | None = None

    @classmethod
    def from_domain(cls, d: Document) -> "DocumentOut":
        return cls(
            id=d.id, filename=d.filename, version=d.version, status=d.status.value, error=d.error
        )


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] | None = None


class CitationOut(BaseModel):
    filename: str
    page: int | None
    quote: str


class SourceOut(BaseModel):
    index: int
    filename: str
    page: int | None
    score: float
    text: str


class QueryOut(BaseModel):
    answer: str
    grounded: bool
    citations: list[CitationOut]
    sources: list[SourceOut]

    @classmethod
    def from_domain(cls, a: Answer) -> "QueryOut":
        return cls(
            answer=a.text,
            grounded=a.grounded,
            citations=[
                CitationOut(filename=c.filename, page=c.page, quote=c.quote) for c in a.citations
            ],
            sources=[
                SourceOut(
                    index=i,
                    filename=s.chunk.filename,
                    page=s.chunk.page,
                    score=round(s.score, 4),
                    text=s.chunk.text,
                )
                for i, s in enumerate(a.sources, start=1)
            ],
        )
