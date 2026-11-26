from typing import Annotated

from fastapi import APIRouter, Depends, Path, UploadFile

from app.api.deps import get_ingest, get_manage
from app.api.schemas import DocumentOut
from app.application.ingest_document import IngestDocument
from app.application.manage_documents import ManageDocuments
from app.config import get_settings

router = APIRouter(prefix="/collections/{collection}/documents", tags=["documents"])
Collection = Annotated[str, Path(pattern=r"^[a-z0-9_-]{1,64}$")]  # keeps names and paths safe


@router.post("", response_model=DocumentOut, status_code=201)
def upload(
    file: UploadFile,
    collection: Collection,
    use_case: Annotated[IngestDocument, Depends(get_ingest)],
) -> DocumentOut:
    limit = get_settings().max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)  # never read more than the limit plus one byte
    doc = use_case.execute(collection, file.filename or "unnamed", data)
    return DocumentOut.from_domain(doc)


@router.get("", response_model=list[DocumentOut])
def list_documents(
    collection: Collection, use_case: Annotated[ManageDocuments, Depends(get_manage)]
) -> list[DocumentOut]:
    return [DocumentOut.from_domain(d) for d in use_case.list(collection)]


@router.delete("/{doc_id}", status_code=204)
def delete_document(
    doc_id: str, collection: Collection, use_case: Annotated[ManageDocuments, Depends(get_manage)]
) -> None:
    use_case.delete(collection, doc_id)
