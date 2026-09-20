from src.rag.dense_retrieval import build_dense_index, query_dense


def test_semantic_paraphrase_finds_remote_work_chunk(test_config, ingested_chunks):
    build_dense_index(ingested_chunks, test_config)
    results = query_dense(
        "Can I live somewhere else for a few weeks and keep working?",
        test_config,
        k=3,
    )
    top_chunk, _ = results[0]
    assert top_chunk.metadata["source"] == "hr_policies.md"


def test_results_preserve_metadata(test_config, ingested_chunks):
    build_dense_index(ingested_chunks, test_config)
    results = query_dense("health insurance coverage", test_config, k=3)
    assert results
    for chunk, distance in results:
        assert chunk.metadata["chunk_id"]
        assert chunk.metadata["source"]
        assert chunk.metadata["page"] == 1
        assert isinstance(distance, float)
