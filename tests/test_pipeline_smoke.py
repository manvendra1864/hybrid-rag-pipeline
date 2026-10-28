"""End-to-end check: load -> chunk -> index -> retrieve for both methods."""

from src.rag.bm25_retrieval import query_bm25
from src.rag.dense_retrieval import build_dense_index, query_dense


def test_full_pipeline_runs_and_returns_results(test_config, ingested_chunks):
    assert len(ingested_chunks) > 0

    build_dense_index(ingested_chunks, test_config)

    dense_results = query_dense("How does a computer keep running many programs when its memory fills up?", test_config)
    bm25_results = query_bm25("What does DNS stand for?", test_config)

    assert len(dense_results) > 0
    assert len(bm25_results) > 0
