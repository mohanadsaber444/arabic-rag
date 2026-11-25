from app.domain.models import Answer, RetrievedChunk
from app.domain.ports import LLM
from app.generation.citations import extract_citations, unsupported_numbers
from app.generation.prompts import NOT_ENOUGH, SYSTEM_PROMPT, USER_TEMPLATE
from app.ingestion.chunking import detect_language

SENTINEL = "INSUFFICIENT_CONTEXT"


class AnswerGenerator:
    def __init__(self, llm: LLM):
        self._llm = llm

    def not_enough(self, question: str) -> Answer:
        lang = "ar" if detect_language(question) == "ar" else "en"
        return Answer(text=NOT_ENOUGH[lang], grounded=False)

    def generate(self, question: str, context: str, sources: list[RetrievedChunk]) -> Answer:
        prompt = USER_TEMPLATE.format(context=context, question=question)
        raw = self._llm.generate(SYSTEM_PROMPT, prompt).strip()
        if SENTINEL in raw:
            return self.not_enough(question)
        citations = extract_citations(raw, sources)
        suspicious = unsupported_numbers(raw, sources)
        return Answer(
            text=raw,
            citations=citations,
            sources=sources,
            grounded=bool(citations) and not suspicious,
        )
