from src.rag.dense_retrieval import build_dense_index, query_dense


def test_semantic_paraphrase_finds_exceptions_chunk(test_config, ingested_chunks):
    build_dense_index(ingested_chunks, test_config)
    results = query_dense(
        "How can a program keep running smoothly even if a piece of user "
        "input turns out to be invalid?",
        test_config,
        k=3,
    )
    top_chunk, _ = results[0]
    assert top_chunk.metadata["source"] == "python.md"


def test_results_preserve_metadata(test_config, ingested_chunks):
    build_dense_index(ingested_chunks, test_config)
    results = query_dense("what is an IP address", test_config, k=3)
    assert results
    for chunk, distance in results:
        assert chunk.metadata["chunk_id"]
        assert chunk.metadata["source"]
        assert chunk.metadata["page"] == 1
        assert isinstance(distance, float)
