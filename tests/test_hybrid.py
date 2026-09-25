from langchain_core.documents import Document

from src.rag.dense_retrieval import build_dense_index
from src.rag.hybrid_retrieval import query_hybrid, reciprocal_rank_fusion


def _doc(chunk_id, source="test.md", page=1):
    return Document(
        page_content=f"content for {chunk_id}",
        metadata={"chunk_id": chunk_id, "source": source, "page": page},
    )


def test_chunk_found_by_both_methods_ranks_highest():
    doc_a = _doc("a")
    doc_b = _doc("b")
    doc_c = _doc("c")

    dense_results = [(doc_b, 0.1), (doc_a, 0.5)]
    bm25_results = [(doc_b, 9.0), (doc_c, 4.0)]

    fused = reciprocal_rank_fusion(dense_results, bm25_results, rrf_k=60)
    fused_ids = [chunk.metadata["chunk_id"] for chunk, _ in fused]

    assert fused_ids[0] == "b"
    assert set(fused_ids) == {"a", "b", "c"}


def test_fusion_score_matches_rrf_formula():
    doc_a = _doc("a")
    doc_b = _doc("b")

    dense_results = [(doc_a, 0.1), (doc_b, 0.2)]
    bm25_results = [(doc_b, 5.0), (doc_a, 3.0)]

    fused = reciprocal_rank_fusion(dense_results, bm25_results, rrf_k=60)
    scores = {chunk.metadata["chunk_id"]: score for chunk, score in fused}

    # a is rank 1 in dense, rank 2 in bm25; b is rank 2 in dense, rank 1 in bm25
    assert scores["a"] == 1 / 61 + 1 / 62
    assert scores["b"] == 1 / 62 + 1 / 61


def test_chunk_in_one_list_only_is_still_included():
    doc_a = _doc("a")
    doc_only_dense = _doc("only-dense")

    fused = reciprocal_rank_fusion(
        dense_results=[(doc_a, 0.1), (doc_only_dense, 0.2)],
        bm25_results=[(doc_a, 5.0)],
        rrf_k=60,
    )
    fused_ids = [chunk.metadata["chunk_id"] for chunk, _ in fused]
    assert "only-dense" in fused_ids


def test_query_hybrid_on_real_corpus(test_config, ingested_chunks):
    build_dense_index(ingested_chunks, test_config)
    results = query_hybrid("What is a foreign key?", test_config, k=3)

    assert results
    top_chunk, _ = results[0]
    assert top_chunk.metadata["source"] == "databases.md"


def test_query_hybrid_preserves_metadata(test_config, ingested_chunks):
    build_dense_index(ingested_chunks, test_config)
    results = query_hybrid("what is an IP address", test_config, k=3)

    assert results
    for chunk, score in results:
        assert chunk.metadata["chunk_id"]
        assert chunk.metadata["source"]
        assert chunk.metadata["page"] == 1
        assert isinstance(score, float)
