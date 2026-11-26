from typing import Annotated

from fastapi import APIRouter, Depends, Path

from app.api.deps import get_answer
from app.api.schemas import QueryIn, QueryOut
from app.application.answer_question import AnswerQuestion

router = APIRouter(prefix="/collections/{collection}", tags=["query"])
Collection = Annotated[str, Path(pattern=r"^[a-z0-9_-]{1,64}$")]


@router.post("/query", response_model=QueryOut)
def query(
    body: QueryIn, collection: Collection, use_case: Annotated[AnswerQuestion, Depends(get_answer)]
) -> QueryOut:
    return QueryOut.from_domain(use_case.execute(collection, body.question, body.document_ids))
