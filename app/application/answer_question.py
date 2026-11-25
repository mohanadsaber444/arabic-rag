from app.domain.models import Answer
from app.generation.generator import AnswerGenerator
from app.retrieval.context import build_context
from app.retrieval.retriever import Retriever


class AnswerQuestion:
    def __init__(self, retriever: Retriever, generator: AnswerGenerator):
        self._retriever = retriever
        self._generator = generator

    def execute(
        self, collection: str, question: str, document_ids: list[str] | None = None
    ) -> Answer:
        hits = self._retriever.retrieve(collection, question, document_ids)
        if not hits:
            return self._generator.not_enough(question)
        context, used = build_context(hits)
        return self._generator.generate(question, context, used)
