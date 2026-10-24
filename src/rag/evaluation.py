"""Measure retrieval quality against a small hand-labelled question set.

For each question the dataset lists the chunk ids that answer it. A method is
run on the question, and its top-k chunk ids are compared with those labels.

Metrics (both are averaged over all questions, so a miss counts as 0):
- Recall@k: the share of the question's relevant chunks found in the top k.
- MRR: the reciprocal rank (1 / position) of the first relevant chunk in the
  top k, or 0 if none of them is there.

Nothing here runs retrieval itself or calls a model: evaluate() takes functions
that return ranked chunks, so the existing retrieval code is used as it is.
"""

import csv
import json
import re
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from config import Config
from src.rag.bm25_retrieval import query_bm25
from src.rag.dense_retrieval import query_dense
from src.rag.hybrid_retrieval import query_hybrid

DEFAULT_K = 3
METHODS = ["Dense", "BM25", "Hybrid", "Graph-enhanced hybrid"]

Retriever = Callable[[str], list[tuple[Document, float]]]


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Share of the relevant chunks that appear in the first k retrieved ids."""
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("A question needs at least one relevant chunk id.")
    return len(relevant & set(retrieved_ids[:k])) / len(relevant)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """1 / rank of the first relevant id within the first k, or 0.0 if there is none."""
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("A question needs at least one relevant chunk id.")
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_questions(path: Path) -> list[dict]:
    """Read the dataset. Each item has id, question and relevant [{chunk_id, evidence}]."""
    questions = json.loads(path.read_text(encoding="utf-8"))
    for item in questions:
        if not item.get("relevant"):
            raise ValueError(f"Question {item.get('id')} has no relevant chunks.")
    return questions


def check_labels(questions: list[dict], chunks: list[Document]) -> list[str]:
    """Problems with the labels: duplicate ids, unknown chunk ids, or evidence not in the chunk text."""
    text_by_id = {c.metadata["chunk_id"]: _squash(c.page_content) for c in chunks}
    problems = []
    seen = set()
    for item in questions:
        if item["id"] in seen:
            problems.append(f"{item['id']}: duplicate question id")
        seen.add(item["id"])
        ids = [r["chunk_id"] for r in item["relevant"]]
        if len(ids) != len(set(ids)):
            problems.append(f"{item['id']}: a chunk id is listed twice")
        for label in item["relevant"]:
            text = text_by_id.get(label["chunk_id"])
            if text is None:
                problems.append(f"{item['id']}: unknown chunk id {label['chunk_id']}")
            elif _squash(label["evidence"]) not in text:
                problems.append(f"{item['id']}: evidence not found in {label['chunk_id']}")
    return problems


def evaluate(
    questions: list[dict], retrievers: dict[str, Retriever], k: int = DEFAULT_K
) -> dict:
    """Run every retriever on every question.

    Returns {"k", "rows", "summary"}. rows has one entry per question and method
    with the retrieved ids and both metrics; summary has the averages per method.
    """
    rows = []
    for item in questions:
        relevant_ids = [r["chunk_id"] for r in item["relevant"]]
        for method, retrieve in retrievers.items():
            retrieved_ids = [chunk.metadata["chunk_id"] for chunk, _ in retrieve(item["question"])][:k]
            rows.append(
                {
                    "question_id": item["id"],
                    "question": item["question"],
                    "method": method,
                    "relevant_ids": relevant_ids,
                    "retrieved_ids": retrieved_ids,
                    "recall": recall_at_k(retrieved_ids, relevant_ids, k),
                    "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids, k),
                }
            )

    summary = {}
    for method in retrievers:
        method_rows = [r for r in rows if r["method"] == method]
        count = len(method_rows)
        summary[method] = {
            "questions": count,
            "recall": sum(r["recall"] for r in method_rows) / count if count else 0.0,
            "mrr": sum(r["reciprocal_rank"] for r in method_rows) / count if count else 0.0,
            "misses": sum(1 for r in method_rows if r["reciprocal_rank"] == 0.0),
        }
    return {"k": k, "rows": rows, "summary": summary}


def build_retrievers(config: Config, k: int = DEFAULT_K) -> dict[str, Retriever]:
    """The four existing retrieval functions, each limited to the top k."""
    return {
        "Dense": lambda q: query_dense(q, config, k=k),
        "BM25": lambda q: query_bm25(q, config, k=k),
        "Hybrid": lambda q: query_hybrid(q, config, k=k),
        "Graph-enhanced hybrid": lambda q: query_hybrid(q, config, k=k, use_graph=True),
    }


def write_csv(result: dict, path: Path) -> None:
    k = result["k"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["question_id", "method", "relevant_ids", "retrieved_ids", f"recall_at_{k}", "reciprocal_rank"]
        )
        for row in result["rows"]:
            writer.writerow(
                [
                    row["question_id"],
                    row["method"],
                    " ".join(row["relevant_ids"]),
                    " ".join(row["retrieved_ids"]),
                    f"{row['recall']:.4f}",
                    f"{row['reciprocal_rank']:.4f}",
                ]
            )


def _cell(retrieved_ids: list[str], relevant_ids: list[str]) -> str:
    """Retrieved ids in rank order; relevant ones in bold."""
    if not retrieved_ids:
        return "none"
    return "<br>".join(
        f"{rank}. " + (f"**{cid}**" if cid in relevant_ids else cid)
        for rank, cid in enumerate(retrieved_ids, start=1)
    )


def markdown_report(result: dict, corpus_note: str = "") -> str:
    k = result["k"]
    summary = result["summary"]
    methods = list(summary)
    count = next(iter(summary.values()))["questions"] if summary else 0

    lines = ["# Retrieval evaluation results", ""]
    lines.append(f"{count} questions, top {k} results per method. {corpus_note}".strip())
    lines += [
        "",
        f"Recall@{k} is the share of a question's relevant chunks found in the top {k}. "
        f"MRR is the average of 1 / rank of the first relevant chunk in the top {k} (0 if none). "
        "A miss means no relevant chunk was in the top results. See `evaluation/README.md` for the limits of these numbers.",
        "",
        "## Aggregate results",
        "",
        f"| Method | Recall@{k} | MRR | Misses |",
        "|---|---|---|---|",
    ]
    for method in methods:
        s = summary[method]
        lines.append(f"| {method} | {s['recall']:.3f} | {s['mrr']:.3f} | {s['misses']} / {s['questions']} |")

    lines += ["", "## Per question", "", "Relevant chunks that were retrieved are in bold.", ""]
    lines.append("| ID | Question | Relevant | " + " | ".join(methods) + " |")
    lines.append("|---|---|---|" + "---|" * len(methods))
    by_question: dict[str, dict] = {}
    for row in result["rows"]:
        by_question.setdefault(row["question_id"], {"row": row, "methods": {}})["methods"][row["method"]] = row
    for qid, entry in by_question.items():
        first = entry["row"]
        cells = [_cell(entry["methods"][m]["retrieved_ids"], first["relevant_ids"]) for m in methods]
        lines.append(
            f"| {qid} | {first['question']} | {'<br>'.join(first['relevant_ids'])} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines) + "\n"
