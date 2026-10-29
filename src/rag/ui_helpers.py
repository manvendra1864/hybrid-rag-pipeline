"""Logic behind the Streamlit app (app.py), kept free of Streamlit calls.

run_question() uses the existing retrieval and answer functions; nothing here
reimplements them. It returns a plain dict that the app only has to draw.
"""

import html
import importlib.util
import logging
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path

import networkx as nx
from langchain_core.documents import Document

from config import PROJECT_ROOT, Config
from src.rag.answer_generation import (
    AnswerGenerationError,
    MissingApiKeyError,
    create_llm,
    generate_answer,
    get_api_key,
)
from src.rag.bm25_retrieval import query_bm25
from src.rag.chunk_store import load_chunks
from src.rag.dense_retrieval import query_dense
from src.rag.hybrid_retrieval import query_hybrid
from src.rag.knowledge_graph import find_concepts, load_graph

METHODS = ["Dense", "BM25", "Hybrid", "Graph-enhanced hybrid"]

SCORE_NOTES = {
    "Dense": "Score is the embedding distance. Lower is closer.",
    "BM25": "Score is the BM25 keyword score. Higher is better.",
    "Hybrid": "Score is the RRF score from dense and BM25.",
    "Graph-enhanced hybrid": "Score is the RRF score with the knowledge graph added as a third list.",
}

logger = logging.getLogger(__name__)

_KEY_VARIABLES = ("GROQ_API_KEY",)

EVALUATION_RESULTS = PROJECT_ROOT / "evaluation" / "results.md"


class EvaluationResultsError(Exception):
    """The saved evaluation results are missing or cannot be read."""


def validate_question(text: str | None) -> str | None:
    """The trimmed question, or None if there is nothing to ask."""
    cleaned = (text or "").strip()
    return cleaned or None


def missing_files(config: Config) -> list[str]:
    """Names of the generated index files that are not there yet."""
    needed = {
        "chunk store": config.chunks_file,
        "dense index": config.chroma_dir,
        "knowledge graph": config.graph_file,
    }
    return [name for name, path in needed.items() if not path.exists()]


def _dense_index_rows(chroma_dir: Path) -> int | None:
    """Number of vectors stored in the Chroma index, or None if it cannot be read.

    The SQLite file is opened read-only and closed straight away, so nothing is
    left holding the index open when ingestion has to replace it.
    """
    database = chroma_dir / "chroma.sqlite3"
    if not database.exists():
        return None
    try:
        connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
        try:
            return connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        finally:
            connection.close()
    except sqlite3.Error:
        return None


def index_problems(config: Config) -> list[str]:
    """Which generated index parts are missing or unusable (empty list: ready).

    The chunk store and graph must load, and the dense index must hold exactly
    one vector per chunk, which also catches a build that was cut short.
    """
    try:
        chunk_count = len(load_chunks(config.chunks_file))
    except Exception:
        chunk_count = 0

    problems = []
    if chunk_count == 0:
        problems.append("chunk store")
    if chunk_count == 0 or _dense_index_rows(config.chroma_dir) != chunk_count:
        problems.append("dense index")
    try:
        load_graph(config.graph_file)
    except Exception:
        problems.append("knowledge graph")
    return problems


def build_index(config: Config) -> None:
    """Build the chunk store, knowledge graph and dense index with scripts/ingest.py.

    The script is loaded from its file, so the app runs the same ingestion code
    as `python scripts/ingest.py` and nothing is copied here.
    """
    path = PROJECT_ROOT / "scripts" / "ingest.py"
    spec = importlib.util.spec_from_file_location("nexusrag_ingest_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(config)


def ensure_index(config: Config) -> str | None:
    """Make sure the index exists, building it if needed. None means it is ready.

    On failure it returns a short message for the page; the full traceback goes
    to the server log.
    """
    if not index_problems(config):
        return None

    try:
        build_index(config)
    except Exception as error:
        logger.exception("Building the search index failed")
        detail = redact(str(error), _env_secrets()).strip().splitlines()[0:1]
        return f"Could not build the search index ({type(error).__name__}): {detail[0][:200] if detail else 'no details'}"

    remaining = index_problems(config)
    if remaining:
        return f"The search index was built but is not usable yet: {', '.join(remaining)}."
    return None


def api_key_available() -> bool:
    try:
        get_api_key()
    except MissingApiKeyError:
        return False
    return True


def redact(text: str, secrets: list[str]) -> str:
    """Hide any secret value that ended up in a message."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def _env_secrets() -> list[str]:
    return [os.environ.get(name, "").strip() for name in _KEY_VARIABLES]


def graph_stats(graph: nx.DiGraph) -> dict[str, int]:
    types = Counter(data.get("type") for _, data in graph.nodes(data=True))
    return {
        "Documents": types["document"],
        "Concepts": types["concept"],
        "Chunks": types["chunk"],
        "Relationships": graph.number_of_edges(),
    }


def load_evaluation_summary(path: Path) -> dict:
    """Read the aggregate table that scripts/evaluate_retrieval.py wrote to results.md.

    The numbers are shown as saved; nothing is recalculated here. Returns
    {"questions", "top_k", "columns", "rows"} with the Method, Recall@k and MRR columns.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise EvaluationResultsError(
            "No evaluation results yet. Run `python scripts/evaluate_retrieval.py` to create them."
        ) from None
    except (OSError, UnicodeDecodeError) as error:
        raise EvaluationResultsError(f"The evaluation results could not be read ({type(error).__name__}).") from error

    section = text.split("## Aggregate results", 1)
    table = []
    for line in section[1].splitlines() if len(section) == 2 else []:
        if line.startswith("|"):
            table.append([cell.strip() for cell in line.strip().strip("|").split("|")])
        elif table:
            break  # the table ends at the first line that is not part of it
    if len(table) < 3 or table[0][0] != "Method":
        raise EvaluationResultsError("The evaluation results file is not in the expected format.")

    header, rows = table[0], table[2:]
    keep = [i for i, name in enumerate(header) if i == 0 or name == "MRR" or name.startswith("Recall@")]
    if len(keep) < 3 or any(len(row) != len(header) for row in rows):
        raise EvaluationResultsError("The evaluation results file is not in the expected format.")

    counts = re.search(r"^(\d+) questions, top (\d+)", text, flags=re.MULTILINE)
    return {
        "questions": int(counts.group(1)) if counts else None,
        "top_k": int(counts.group(2)) if counts else None,
        "columns": [header[i] for i in keep],
        "rows": [[row[i] for i in keep] for row in rows],
    }


def rank_notes(hybrid: list[tuple[Document, float]], enhanced: list[tuple[Document, float]]) -> dict[str, str]:
    """For each graph-enhanced result, the rank it had in ordinary hybrid."""
    hybrid_rank = {c.metadata["chunk_id"]: i for i, (c, _) in enumerate(hybrid, start=1)}
    return {
        c.metadata["chunk_id"]: (
            f"hybrid rank {hybrid_rank[c.metadata['chunk_id']]}"
            if c.metadata["chunk_id"] in hybrid_rank
            else f"not in hybrid top {len(hybrid)}"
        )
        for c, _ in enhanced
    }


def chunk_records(results: list[tuple[Document, float]], notes: dict[str, str] | None = None) -> list[dict]:
    """Plain, display-ready rows for a list of (chunk, score) results."""
    return [
        {
            "rank": rank,
            "source": chunk.metadata["source"],
            "chunk_id": chunk.metadata["chunk_id"],
            "page": chunk.metadata.get("page"),
            "score": float(score),
            "text": chunk.page_content.strip(),
            "note": (notes or {}).get(chunk.metadata["chunk_id"], ""),
        }
        for rank, (chunk, score) in enumerate(results, start=1)
    ]


def citation_chips_html(sources: list[tuple[str, str]]) -> str:
    """Source chips as HTML. Everything is escaped before it goes into the page."""
    return "".join(
        f'<span class="chip">{html.escape(source)}<i>·</i><b>{html.escape(chunk_id)}</b></span>'
        for source, chunk_id in sources
    )


def run_question(question: str, config: Config, k: int) -> dict:
    """Retrieve with all four methods, then try to write the answer.

    Expected problems (missing index, no API key, an API error) come back as a
    notice on the returned dict instead of an exception, and secret values are
    removed from any message.
    """
    turn = {
        "question": question,
        "answer": None,
        "sufficient": None,
        "sources": [],
        "notice": None,
        "retrieved": [],
        "comparison": {},
        "concepts": [],
    }

    try:
        dense = query_dense(question, config, k=k)
        bm25 = query_bm25(question, config, k=k)
        hybrid = query_hybrid(question, config, k=k)
        enhanced = query_hybrid(question, config, k=k, use_graph=True)
        graph = load_graph(config.graph_file)
    except FileNotFoundError as error:
        turn["notice"] = {"level": "error", "text": str(error)}
        return turn
    except Exception as error:
        text = redact(f"Retrieval failed ({type(error).__name__}): {error}", _env_secrets())
        turn["notice"] = {"level": "error", "text": text}
        return turn

    notes = rank_notes(hybrid, enhanced)
    turn["comparison"] = {
        "Dense": chunk_records(dense),
        "BM25": chunk_records(bm25),
        "Hybrid": chunk_records(hybrid),
        "Graph-enhanced hybrid": chunk_records(enhanced, notes),
    }
    turn["retrieved"] = turn["comparison"]["Graph-enhanced hybrid"]
    turn["concepts"] = [graph.nodes[n]["label"] for n in find_concepts(graph, question)]

    chunks = [chunk for chunk, _ in enhanced]
    try:
        llm = create_llm(config) if chunks else None
        answer = generate_answer(question, chunks, config, llm=llm)
    except MissingApiKeyError as error:
        turn["notice"] = {
            "level": "warning",
            "text": f"{error}\n\nThe retrieved chunks are shown below, but no answer was written.",
        }
    except AnswerGenerationError as error:
        turn["notice"] = {"level": "error", "text": redact(str(error), _env_secrets())}
    except Exception as error:
        text = redact(f"Could not write the answer ({type(error).__name__}): {error}", _env_secrets())
        turn["notice"] = {"level": "error", "text": text}
    else:
        turn["answer"] = answer.text
        turn["sufficient"] = answer.sufficient
        turn["sources"] = answer.sources
    return turn
