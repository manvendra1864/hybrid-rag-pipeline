from src.rag.bm25_retrieval import query_bm25


def test_exact_keyword_query_finds_foreign_key(test_config, ingested_chunks):
    results = query_bm25("What is a foreign key?", test_config, k=3)
    top_chunk, _ = results[0]
    assert top_chunk.metadata["source"] == "databases.md"
    assert "foreign key" in top_chunk.page_content.lower()


def test_results_preserve_metadata(test_config, ingested_chunks):
    results = query_bm25("primary key", test_config, k=3)
    assert results
    for chunk, score in results:
        assert chunk.metadata["chunk_id"]
        assert chunk.metadata["source"]
        assert chunk.metadata["page"] == 1
        assert isinstance(score, float)
