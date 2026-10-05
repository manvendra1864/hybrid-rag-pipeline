from langchain_core.documents import Document

from src.rag.chunk_store import save_chunks
from src.rag.knowledge_graph import (
    build_graph,
    graph_chunk_scores,
    query_graph,
    save_graph,
)


def _chunk(chunk_id, text, source="db.md"):
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "source": source, "page": 1},
    )


def _small_chunks():
    # c0 ends with the bare "## Foreign Keys" heading; the body is in c1.
    return [
        _chunk("db-p1-c0", "# DB\n\n## Primary Keys\n\nA primary key identifies each row.\n\n## Foreign Keys"),
        _chunk(
            "db-p1-c1",
            "A foreign key points to a primary key in another table.\n\n"
            "## Indexes\n\nAn index speeds up lookups.",
        ),
        _chunk("db-p1-c2", "## Views\n\nA view is a saved query."),
    ]


def _write_stores(test_config, chunks):
    save_chunks(chunks, test_config.chunks_file)
    save_graph(build_graph(chunks), test_config.graph_file)


def test_covered_in_edge_records_share_of_section_text():
    graph = build_graph(_small_chunks())

    heading_only = graph.edges["concept:db/foreign-keys", "chunk:db-p1-c0"]
    body = graph.edges["concept:db/foreign-keys", "chunk:db-p1-c1"]
    assert heading_only["share"] == 0.0
    assert body["share"] == 1.0


def test_matched_concept_chunks_come_first_then_related_chunks():
    graph = build_graph(_small_chunks())

    scored = graph_chunk_scores(graph, "What is a foreign key?")

    # c1 holds the foreign key text; c0 is only reached through the related
    # "Primary Keys" concept that the foreign key section mentions.
    assert [cid for cid, _ in scored] == ["db-p1-c1", "db-p1-c0"]


def test_heading_only_chunk_is_not_scored_for_its_own_concept():
    # c0 only ends with the bare "## Foreign Keys" heading, and Foreign
    # Keys names no other concept, so c1 must be the only result.
    chunks = [
        _chunk("x-p1-c0", "## Primary Keys\n\nText about identifiers.\n\n## Foreign Keys", "x.md"),
        _chunk("x-p1-c1", "Text about links between tables.", "x.md"),
    ]
    graph = build_graph(chunks)

    scored = graph_chunk_scores(graph, "foreign key")

    assert [cid for cid, _ in scored] == ["x-p1-c1"]


def test_related_chunk_never_outranks_a_chunk_of_the_matched_concept():
    # "Memory" has a long body chunk and a short tail chunk. "Threads"
    # mentions memory and has one long chunk of its own.
    chunks = [
        _chunk("os-p1-c0", "## Threads\n\n" + "Threads run in a process and share memory. " * 6, "os.md"),
        _chunk("os-p1-c1", "## Memory\n\n" + "Memory holds data for a running program. " * 6, "os.md"),
        _chunk("os-p1-c2", "Disk is slower than RAM.", "os.md"),
    ]
    graph = build_graph(chunks)

    scored = graph_chunk_scores(graph, "How does memory work?")

    assert [cid for cid, _ in scored] == ["os-p1-c1", "os-p1-c2", "os-p1-c0"]


def test_each_chunk_appears_once_when_several_concepts_match():
    graph = build_graph(_small_chunks())

    scored = graph_chunk_scores(graph, "foreign key and primary key and index")

    ids = [cid for cid, _ in scored]
    assert len(ids) == len(set(ids))
    assert set(ids) == {"db-p1-c0", "db-p1-c1"}


def test_query_naming_no_concept_returns_nothing():
    graph = build_graph(_small_chunks())

    assert graph_chunk_scores(graph, "tell me about astronomy") == []


def test_query_graph_returns_chunks_with_metadata(test_config):
    _write_stores(test_config, _small_chunks())

    results = query_graph("What is a foreign key?", test_config)

    assert [c.metadata["chunk_id"] for c, _ in results] == ["db-p1-c1", "db-p1-c0"]
    for chunk, score in results:
        assert chunk.metadata["source"] == "db.md"
        assert chunk.metadata["page"] == 1
        assert score > 0


def test_query_graph_respects_k(test_config):
    _write_stores(test_config, _small_chunks())

    results = query_graph("What is a foreign key?", test_config, k=1)

    assert [c.metadata["chunk_id"] for c, _ in results] == ["db-p1-c1"]


def test_query_graph_empty_for_unmatched_query(test_config):
    _write_stores(test_config, _small_chunks())

    assert query_graph("tell me about astronomy", test_config) == []


def test_query_graph_skips_chunks_missing_from_chunk_store(test_config):
    chunks = _small_chunks()
    save_graph(build_graph(chunks), test_config.graph_file)
    save_chunks([c for c in chunks if c.metadata["chunk_id"] != "db-p1-c0"], test_config.chunks_file)

    results = query_graph("What is a foreign key?", test_config)

    assert [c.metadata["chunk_id"] for c, _ in results] == ["db-p1-c1"]


def test_real_corpus_foreign_key_query_ranks_definition_chunk_first(test_config, ingested_chunks):
    save_graph(build_graph(ingested_chunks), test_config.graph_file)

    results = query_graph("What is a foreign key?", test_config)

    assert results[0][0].metadata["chunk_id"] == "databases-p1-c3"


def test_real_corpus_memory_query_keeps_its_own_tail_chunk(test_config, ingested_chunks):
    save_graph(build_graph(ingested_chunks), test_config.graph_file)

    results = query_graph("How does a computer avoid running out of memory?", test_config, k=3)
    ids = [c.metadata["chunk_id"] for c, _ in results]

    # both chunks of the memory section come before related concepts
    assert ids[:2] == ["operating_systems-p1-c3", "operating_systems-p1-c4"]
