import dataclasses

import pytest
from langchain_core.documents import Document

import src.rag.hybrid_retrieval as hybrid_module
from src.rag.chunk_store import save_chunks
from src.rag.dense_retrieval import build_dense_index
from src.rag.hybrid_retrieval import query_hybrid, reciprocal_rank_fusion
from src.rag.knowledge_graph import build_graph, save_graph


def _doc(chunk_id, text="text", source="db.md"):
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "source": source, "page": 1},
    )


def _ids(results):
    return [chunk.metadata["chunk_id"] for chunk, _ in results]


def test_fusion_without_graph_is_unchanged_by_the_new_parameters():
    a, b, c = _doc("a"), _doc("b"), _doc("c")
    dense = [(a, 0.1), (b, 0.2)]
    bm25 = [(b, 5.0), (c, 3.0)]

    plain = reciprocal_rank_fusion(dense, bm25, rrf_k=60)

    assert reciprocal_rank_fusion(dense, bm25, rrf_k=60, graph_results=None) == plain
    assert reciprocal_rank_fusion(dense, bm25, rrf_k=60, graph_results=[]) == plain
    assert dict((ch.metadata["chunk_id"], s) for ch, s in plain)["b"] == 1 / 62 + 1 / 61


def test_graph_vote_is_scaled_by_graph_weight():
    only_graph = _doc("g")
    fused = reciprocal_rank_fusion(
        [(_doc("a"), 0.1)], [], rrf_k=60, graph_results=[(only_graph, 1.0)], graph_weight=0.5
    )

    scores = {ch.metadata["chunk_id"]: s for ch, s in fused}
    assert scores["a"] == 1 / 61
    assert scores["g"] == 0.5 / 61


def test_graph_vote_lifts_a_chunk_that_dense_already_found():
    a, b = _doc("a"), _doc("b")

    fused = reciprocal_rank_fusion(
        [(a, 0.1), (b, 0.2)], [], rrf_k=60, graph_results=[(b, 1.0)], graph_weight=0.5
    )

    assert _ids(fused) == ["b", "a"]


def test_light_graph_weight_does_not_beat_a_chunk_both_retrievers_ranked():
    a, b, c, g = _doc("a"), _doc("b"), _doc("c"), _doc("g")

    fused = reciprocal_rank_fusion(
        [(a, 0.1), (b, 0.2), (c, 0.3)], [], rrf_k=60, graph_results=[(g, 1.0)], graph_weight=0.5
    )

    assert _ids(fused) == ["a", "b", "c", "g"]


def test_chunk_in_all_three_lists_appears_once():
    a = _doc("a")

    fused = reciprocal_rank_fusion(
        [(a, 0.1)], [(a, 5.0)], rrf_k=60, graph_results=[(a, 1.0)], graph_weight=0.5
    )

    assert _ids(fused) == ["a"]
    assert fused[0][1] == 1 / 61 + 1 / 61 + 0.5 / 61


def _small_chunks():
    return [
        _doc("db-p1-c0", "## Primary Keys\n\nA primary key identifies each row."),
        _doc("db-p1-c1", "## Foreign Keys\n\nA foreign key points to a primary key."),
        _doc("db-p1-c2", "## Views\n\nA view is a saved query."),
        _doc("db-p1-c3", "## Triggers\n\nA trigger runs when a row changes."),
    ]


@pytest.fixture
def small_stores(test_config, monkeypatch):
    chunks = _small_chunks()
    save_chunks(chunks, test_config.chunks_file)
    save_graph(build_graph(chunks), test_config.graph_file)
    by_id = {c.metadata["chunk_id"]: c for c in chunks}

    # Dense ranks Views, then Triggers, then Foreign Keys; BM25 finds Views only.
    dense = [(by_id["db-p1-c2"], 0.2), (by_id["db-p1-c3"], 0.4), (by_id["db-p1-c1"], 0.5)]
    bm25 = [(by_id["db-p1-c2"], 3.0)]
    monkeypatch.setattr(hybrid_module, "query_dense", lambda q, c: dense)
    monkeypatch.setattr(hybrid_module, "query_bm25", lambda q, c: bm25)
    return by_id


def test_graph_enhanced_query_promotes_the_named_concept(test_config, small_stores):
    plain = query_hybrid("What is a foreign key?", test_config)
    enhanced = query_hybrid("What is a foreign key?", test_config, use_graph=True)

    assert _ids(plain) == ["db-p1-c2", "db-p1-c3", "db-p1-c1"]
    # the graph lifts the Foreign Keys chunk past Triggers, but a chunk both
    # retrievers ranked first is not displaced
    assert _ids(enhanced)[:2] == ["db-p1-c2", "db-p1-c1"]


def test_unmatched_query_falls_back_to_ordinary_hybrid(test_config, small_stores):
    plain = query_hybrid("tell me about astronomy", test_config)
    enhanced = query_hybrid("tell me about astronomy", test_config, use_graph=True)

    assert enhanced == plain


def test_default_path_does_not_need_the_graph(test_config, small_stores):
    config = dataclasses.replace(test_config, graph_file=test_config.graph_file.parent / "missing.json")

    results = query_hybrid("What is a foreign key?", config)

    assert _ids(results) == ["db-p1-c2", "db-p1-c3", "db-p1-c1"]


def test_graph_enhanced_query_without_a_graph_file_reports_it(test_config, small_stores):
    config = dataclasses.replace(test_config, graph_file=test_config.graph_file.parent / "missing.json")

    with pytest.raises(FileNotFoundError, match="ingest"):
        query_hybrid("What is a foreign key?", config, use_graph=True)


def test_graph_enhanced_results_respect_k_and_have_no_duplicates(test_config, small_stores):
    results = query_hybrid("foreign key and primary key", test_config, k=2, use_graph=True)

    ids = _ids(results)
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_real_corpus_graph_enhanced_query_keeps_metadata(test_config, ingested_chunks):
    build_dense_index(ingested_chunks, test_config)
    save_graph(build_graph(ingested_chunks), test_config.graph_file)

    results = query_hybrid("What does DNS do when I open a website?", test_config, use_graph=True)

    assert results
    assert results[0][0].metadata["chunk_id"] == "networks-p1-c5"
    for chunk, score in results:
        assert chunk.metadata["source"] == "networks.md"
        assert chunk.metadata["page"] == 1
        assert isinstance(score, float)


def test_real_corpus_original_hybrid_path_still_works(test_config, ingested_chunks):
    build_dense_index(ingested_chunks, test_config)

    results = query_hybrid("What is a foreign key?", test_config)

    assert results[0][0].metadata["chunk_id"] == "databases-p1-c3"
