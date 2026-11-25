from app.application.answer_question import AnswerQuestion
from app.domain.models import Chunk, RetrievedChunk
from app.generation.generator import AnswerGenerator


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def generate(self, system, prompt):
        self.calls += 1
        return self.reply


class FakeRetriever:
    def __init__(self, hits):
        self.hits, self.args = hits, None

    def retrieve(self, collection, question, document_ids=None):
        self.args = (collection, question, document_ids)
        return self.hits


def hit(text="The notice period is 30 days."):
    c = Chunk(
        id="c1",
        document_id="d1",
        document_version=1,
        collection="t",
        filename="policy.pdf",
        ordinal=0,
        page=2,
        text=text,
    )
    return RetrievedChunk(chunk=c, score=0.9)


def test_no_hits_never_calls_the_llm():
    llm = FakeLLM("anything")
    answer = AnswerQuestion(FakeRetriever([]), AnswerGenerator(llm)).execute(
        "hr", "What is the notice?"
    )
    assert llm.calls == 0 and not answer.grounded and answer.citations == []


def test_hits_produce_a_cited_answer_and_pass_the_filter():
    llm = FakeLLM("The notice period is 30 days [1].")
    retriever = FakeRetriever([hit()])
    answer = AnswerQuestion(retriever, AnswerGenerator(llm)).execute("hr", "q", ["d1"])
    assert llm.calls == 1 and answer.grounded
    assert answer.citations[0].filename == "policy.pdf"
    assert retriever.args == ("hr", "q", ["d1"])
