import csv
import json
import sys
from datetime import date
from pathlib import Path

from app.config import get_settings
from app.container import get_container
from app.domain.errors import DuplicateDocumentError

DOCS = Path("eval/docs")


def norm(text: str) -> str:
    return text.replace(",", "").lower()


def pct(flags: list[bool]) -> float:
    return 100 * sum(flags) / max(len(flags), 1)


def main(path: str = "eval/golden_set.jsonl", collection: str = "eval") -> None:
    c = get_container()
    s = get_settings()
    for f in sorted(DOCS.glob("*.txt")):
        print(f"ingesting {f.name} ...", flush=True)
        try:
            c.ingest.execute(collection, f.name, f.read_bytes())
        except DuplicateDocumentError:
            pass
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    hit_flags, ranks, correct, refused = [], [], [], []
    relevant_scores, irrelevant_scores, failures = [], [], []
    for n, row in enumerate(rows, 1):
        print(f"[{n}/{len(rows)}] {row['question']}", flush=True)
        hits = c.retriever.retrieve(collection, row["question"])
        answer = c.answer.execute(collection, row["question"])
        if not row["answerable"]:
            refused.append(not answer.grounded)
            if hits:
                irrelevant_scores.append(hits[0].score)
            if answer.grounded:
                failures.append((row["id"], "should refuse", answer.text[:70]))
            continue
        rank = next(
            (
                i
                for i, h in enumerate(hits, 1)
                if h.chunk.filename == row["expected_file"]
                and norm(row["expected_text"]) in norm(h.chunk.text)
            ),
            None,
        )
        hit_flags.append(rank is not None)
        ranks.append(1 / rank if rank else 0.0)
        if rank:
            relevant_scores.append(hits[rank - 1].score)
        ok = answer.grounded and norm(row["answer_contains"]) in norm(answer.text)
        correct.append(ok)
        if not ok:
            failures.append((row["id"], "wrong or ungrounded", answer.text[:70]))
    mrr = sum(ranks) / max(len(ranks), 1)
    print(f"Questions: {len(rows)} ({len(correct)} answerable, {len(refused)} unanswerable)")
    print(f"Retrieval hit rate : {pct(hit_flags):.0f}%")
    print(f"Mean reciprocal rank: {mrr:.2f}")
    print(f"Answer accuracy    : {pct(correct):.0f}%")
    print(f"Correct refusals   : {pct(refused):.0f}%")
    if relevant_scores and irrelevant_scores:
        print(
            f"Score of correct chunks   : min {min(relevant_scores):.3f}, "
            f"avg {sum(relevant_scores) / len(relevant_scores):.3f}"
        )
        print(f"Top score when unanswerable: max {max(irrelevant_scores):.3f}")
    for qid, why, text in failures:
        print(f"FAIL {qid}: {why} -> {text}")
    results = Path("eval/results.csv")
    new = not results.exists()
    with results.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "date",
                    "questions",
                    "hit_rate",
                    "mrr",
                    "answer_accuracy",
                    "refusal_rate",
                    "chunk_size",
                    "top_k",
                    "embedding_model",
                    "llm_model",
                ]
            )
        w.writerow(
            [
                date.today(),
                len(rows),
                f"{pct(hit_flags):.0f}",
                f"{mrr:.2f}",
                f"{pct(correct):.0f}",
                f"{pct(refused):.0f}",
                s.chunk_size,
                s.top_k,
                s.embedding_model,
                s.llm_model,
            ]
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
