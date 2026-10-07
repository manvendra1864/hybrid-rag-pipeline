"""Combine dense, BM25 and (optionally) graph results with Reciprocal Rank Fusion.

RRF only needs each method's rank position, not its raw score, which
avoids having to make dense distances, BM25 scores and graph scores
comparable (they are on completely different scales).
"""

from langchain_core.documents import Document

from config import Config
from src.rag.bm25_retrieval import query_bm25
from src.rag.dense_retrieval import query_dense
from src.rag.knowledge_graph import query_graph


def reciprocal_rank_fusion(
    dense_results: list[tuple[Document, float]],
    bm25_results: list[tuple[Document, float]],
    rrf_k: int,
    graph_results: list[tuple[Document, float]] | None = None,
    graph_weight: float = 1.0,
) -> list[tuple[Document, float]]:
    """Merge ranked result lists (best match first) into one.

    A chunk's fused score is the sum of weight / (rrf_k + rank) over every
    list it appears in, using its 1-indexed rank in that list. Dense and
    BM25 count with weight 1.0; the optional graph list counts with
    graph_weight, so it can be a lighter vote than the other two. A chunk
    found by several methods accumulates several terms and so tends to
    rank above one found by a single method.
    """
    ranked_lists = [(dense_results, 1.0), (bm25_results, 1.0)]
    if graph_results:
        ranked_lists.append((graph_results, graph_weight))

    scores: dict[str, float] = {}
    chunks: dict[str, Document] = {}

    for results, weight in ranked_lists:
        for rank, (chunk, _) in enumerate(results, start=1):
            chunk_id = chunk.metadata["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)
            chunks.setdefault(chunk_id, chunk)

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [(chunks[cid], scores[cid]) for cid in ranked_ids]


def query_hybrid(
    query: str, config: Config, k: int | None = None, use_graph: bool = False
) -> list[tuple[Document, float]]:
    """Run dense and BM25 retrieval and fuse the rankings with RRF.

    With use_graph=True the knowledge graph adds a third ranked list. If
    the query names no concept in the graph that list is empty and the
    result is the same as ordinary hybrid retrieval.
    """
    dense_results = query_dense(query, config)
    bm25_results = query_bm25(query, config)
    graph_results = query_graph(query, config) if use_graph else None
    fused = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        rrf_k=config.rrf_k,
        graph_results=graph_results,
        graph_weight=config.graph_weight,
    )
    top_k = k or config.hybrid_top_k
    return fused[:top_k]
