from app.domain.models import Chunk, RetrievedChunk
from app.generation.generator import AnswerGenerator


class FakeLLM:
    def __init__(self, reply: str):
        self.reply, self.calls = reply, 0

    def generate(self, system: str, prompt: str) -> str:
        self.calls += 1
        return self.reply


def make_hit(text: str = "The notice period is 30 days.") -> RetrievedChunk:
    chunk = Chunk(
        id="c1",
        document_id="d1",
        document_version=1,
        collection="t",
        filename="policy.pdf",
        ordinal=0,
        page=2,
        text=text,
    )
    return RetrievedChunk(chunk=chunk, score=0.9)


def test_valid_citation_is_kept():
    answer = AnswerGenerator(FakeLLM("The notice period is 30 days [1].")).generate(
        "q", "ctx", [make_hit()]
    )
    assert answer.grounded and answer.citations[0].page == 2


def test_invented_citation_is_dropped():
    answer = AnswerGenerator(FakeLLM("Thirty days [7].")).generate("q", "ctx", [make_hit()])
    assert answer.citations == [] and not answer.grounded


def test_invented_number_marks_answer_ungrounded():
    answer = AnswerGenerator(FakeLLM("The notice period is 90 days [1].")).generate(
        "q", "ctx", [make_hit()]
    )
    assert not answer.grounded


def test_sentinel_returns_not_enough():
    answer = AnswerGenerator(FakeLLM("INSUFFICIENT_CONTEXT")).generate(
        "What is the capital?", "ctx", [make_hit()]
    )
    assert not answer.grounded and answer.citations == []


def test_sentinel_answers_in_the_question_language():
    answer = AnswerGenerator(FakeLLM("INSUFFICIENT_CONTEXT")).generate(
        "ما هي العاصمة؟", "ctx", [make_hit()]
    )
    assert "المستندات" in answer.text and not answer.grounded
