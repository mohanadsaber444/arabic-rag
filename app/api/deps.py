from app.application.answer_question import AnswerQuestion
from app.application.ingest_document import IngestDocument
from app.application.manage_documents import ManageDocuments
from app.container import get_container


def get_ingest() -> IngestDocument:
    return get_container().ingest


def get_manage() -> ManageDocuments:
    return get_container().manage


def get_answer() -> AnswerQuestion:
    return get_container().answer
