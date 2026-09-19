"""Sparse keyword retrieval using BM25 over the same chunks as dense search.

LangChain's community BM25Retriever wraps rank_bm25 but does not expose
per-result scores through its standard interface, and score visibility is
useful for comparing BM25 against dense retrieval. Using rank_bm25 directly
is a small, clear alternative that keeps scores available.
"""

import re

from rank_bm25 import BM25Okapi

from config import Config
from langchain_core.documents import Document
from src.rag.chunk_store import load_chunks

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def query_bm25(
    query: str, config: Config, k: int | None = None
) -> list[tuple[Document, float]]:
    """Return (chunk, bm25_score) pairs ordered best-match first."""
    chunks = load_chunks(config.chunks_file)
    tokenized_corpus = [_tokenize(chunk.page_content) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)

    top_k = k or config.bm25_top_k
    return ranked[:top_k]
