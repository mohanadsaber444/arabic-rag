import sys
from pathlib import Path

from app.container import get_container
from app.domain.errors import DuplicateDocumentError


def main(path: str, collection: str, *questions: str) -> None:
    c = get_container()
    file = Path(path)
    try:
        doc = c.ingest.execute(collection, file.name, file.read_bytes())
        print(f"ingested: {doc.filename} status={doc.status.value} version={doc.version}")
    except DuplicateDocumentError:
        print("already ingested, skipping upload")
    for question in questions:
        answer = c.answer.execute(collection, question)
        print(f"\nQ: {question}\ngrounded={answer.grounded}\n{answer.text}")
        for cit in answer.citations:
            print(f"  - {cit.filename}, page {cit.page}")


main(*sys.argv[1:])
