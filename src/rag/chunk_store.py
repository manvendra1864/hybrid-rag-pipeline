"""Persist chunks to a JSON file so dense and BM25 retrieval share them.

Both retrieval paths must run over the exact same logical chunks. Rather
than re-loading and re-chunking the corpus independently in each script,
ingestion writes the chunk list here once, and both retrieval scripts
read it back.
"""

import json
from pathlib import Path

from langchain_core.documents import Document


def save_chunks(chunks: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"page_content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_chunks(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(
            f"No chunk store at {path}. Run scripts/ingest.py first."
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(page_content=r["page_content"], metadata=r["metadata"])
        for r in records
    ]
