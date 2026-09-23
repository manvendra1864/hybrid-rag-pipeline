"""Combine dense and BM25 results into one ranking with Reciprocal Rank Fusion.

RRF only needs each method's rank position, not its raw score, which
avoids having to make dense distances and BM25 scores comparable (they
are on completely different scales).
"""

from langchain_core.documents import Document

from config import Config
from src.rag.bm25_retrieval import query_bm25
from src.rag.dense_retrieval import query_dense


def reciprocal_rank_fusion(
    dense_results: list[tuple[Document, float]],
    bm25_results: list[tuple[Document, float]],
    rrf_k: int,
) -> list[tuple[Document, float]]:
    """Merge two ranked result lists (best match first) into one.

    A chunk's fused score is the sum of 1 / (rrf_k + rank) over every
    list it appears in, using its 1-indexed rank in that list. A chunk
    found by both methods accumulates two terms and so tends to rank
    above one found by only a single method.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Document] = {}

    for results in (dense_results, bm25_results):
        for rank, (chunk, _) in enumerate(results, start=1):
            chunk_id = chunk.metadata["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            chunks.setdefault(chunk_id, chunk)

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [(chunks[cid], scores[cid]) for cid in ranked_ids]


def query_hybrid(
    query: str, config: Config, k: int | None = None
) -> list[tuple[Document, float]]:
    """Run dense and BM25 retrieval and fuse the two rankings with RRF."""
    dense_results = query_dense(query, config)
    bm25_results = query_bm25(query, config)
    fused = reciprocal_rank_fusion(dense_results, bm25_results, rrf_k=config.rrf_k)
    top_k = k or config.hybrid_top_k
    return fused[:top_k]
