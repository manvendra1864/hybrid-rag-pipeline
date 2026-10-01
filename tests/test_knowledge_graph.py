from langchain_core.documents import Document

from src.rag.knowledge_graph import (
    build_graph,
    describe_concept,
    find_concepts,
    load_graph,
    save_graph,
)


def _chunk(chunk_id, text, source="db.md"):
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "source": source, "page": 1},
    )


def _small_chunks():
    return [
        _chunk(
            "db-p1-c0",
            "# DB\n\n## Primary Keys\n\nA primary key identifies each row.\n\n"
            "## Foreign Keys\n\nA foreign key points to a primary key in another table.",
        ),
        _chunk("db-p1-c1", "A foreign key links two tables together."),
    ]


def _edges(graph, relation):
    return {
        (u, v): d for u, v, d in graph.edges(data=True) if d["relation"] == relation
    }


def test_nodes_for_documents_concepts_and_chunks():
    graph = build_graph(_small_chunks())
    types = {n: d["type"] for n, d in graph.nodes(data=True)}

    assert types["doc:db.md"] == "document"
    assert types["concept:db/primary-keys"] == "concept"
    assert types["concept:db/foreign-keys"] == "concept"
    assert types["chunk:db-p1-c0"] == "chunk"
    assert types["chunk:db-p1-c1"] == "chunk"


def test_concepts_link_to_document_and_to_every_chunk_they_span():
    graph = build_graph(_small_chunks())

    contains = _edges(graph, "contains")
    assert ("doc:db.md", "concept:db/foreign-keys") in contains

    covered = _edges(graph, "covered_in")
    assert ("concept:db/foreign-keys", "chunk:db-p1-c0") in covered
    assert ("concept:db/foreign-keys", "chunk:db-p1-c1") in covered
    assert ("concept:db/primary-keys", "chunk:db-p1-c1") not in covered


def test_mentions_edge_records_the_evidence_chunk():
    graph = build_graph(_small_chunks())
    mentions = _edges(graph, "mentions")

    edge = mentions[("concept:db/foreign-keys", "concept:db/primary-keys")]
    assert edge["chunk_ids"] == ["db-p1-c0"]
    assert edge["weight"] == 1
    # primary keys never names foreign keys
    assert ("concept:db/primary-keys", "concept:db/foreign-keys") not in mentions


def test_intro_text_does_not_replace_the_contains_edge():
    chunks = [
        _chunk(
            "db-p1-c0",
            "# DB\n\nThis file is about the primary key.\n\n## Primary Keys\n\nBody text.",
        )
    ]
    graph = build_graph(chunks)

    edge = graph.edges["doc:db.md", "concept:db/primary-keys"]
    assert edge["relation"] == "contains"


def test_concept_does_not_mention_itself():
    graph = build_graph(_small_chunks())
    for u, v in _edges(graph, "mentions"):
        assert u != v


def test_mentions_do_not_cross_documents():
    chunks = [
        _chunk("a-p1-c0", "## Processes\n\nA process runs a program.", source="a.md"),
        _chunk("b-p1-c0", "## Normalization\n\nThe process of removing duplication.", source="b.md"),
    ]
    graph = build_graph(chunks)

    assert _edges(graph, "mentions") == {}


def test_save_and_load_round_trip(tmp_path):
    graph = build_graph(_small_chunks())
    path = tmp_path / "graph.json"
    save_graph(graph, path)
    loaded = load_graph(path)

    assert set(loaded.nodes) == set(graph.nodes)
    assert set(loaded.edges) == set(graph.edges)
    assert loaded.nodes["concept:db/foreign-keys"]["aliases"] == ["foreign keys"]
    assert loaded.edges["concept:db/foreign-keys", "concept:db/primary-keys"]["chunk_ids"] == ["db-p1-c0"]


def test_find_concepts_matches_query_wording():
    graph = build_graph(_small_chunks())

    assert find_concepts(graph, "What is a foreign key?") == ["concept:db/foreign-keys"]
    assert find_concepts(graph, "something unrelated") == []


def test_describe_concept_lists_neighbours_and_chunks():
    graph = build_graph(_small_chunks())

    info = describe_concept(graph, "concept:db/primary-keys")
    assert info["mentioned_by"] == ["Foreign Keys"]
    assert info["mentions"] == []
    assert info["chunk_ids"] == ["db-p1-c0"]


def test_graph_on_real_corpus(ingested_chunks):
    graph = build_graph(ingested_chunks)

    concepts = [n for n, d in graph.nodes(data=True) if d["type"] == "concept"]
    documents = [n for n, d in graph.nodes(data=True) if d["type"] == "document"]
    assert len(documents) == 4
    assert len(concepts) == 22

    # every chunk node is an existing chunk, and every concept sits in one
    chunk_ids = {c.metadata["chunk_id"] for c in ingested_chunks}
    graph_chunk_ids = {d["label"] for _, d in graph.nodes(data=True) if d["type"] == "chunk"}
    assert graph_chunk_ids == chunk_ids
    for concept in concepts:
        assert describe_concept(graph, concept)["chunk_ids"]
        source = graph.nodes[concept]["source"]
        assert graph.edges[f"doc:{source}", concept]["relation"] == "contains"

    foreign = find_concepts(graph, "What is a foreign key?")
    assert foreign == ["concept:databases/foreign-keys"]
    assert "Primary Keys" in describe_concept(graph, foreign[0])["mentions"]


def test_graph_construction_is_deterministic(ingested_chunks):
    first = build_graph(ingested_chunks)
    second = build_graph(ingested_chunks)

    assert list(first.nodes(data=True)) == list(second.nodes(data=True))
    assert list(first.edges(data=True)) == list(second.edges(data=True))
