"""Build and query a small knowledge graph over the existing chunks.

Nodes:
    document  one per source file                  doc:python.md
    concept   one per '## Heading'                 concept:python/exceptions
    chunk     one per existing chunk               chunk:python-p1-c5

Edges:
    document -contains->   concept
    concept  -covered_in-> chunk     the chunk holds part of that section;
                                     share is the fraction of the section's
                                     text that sits in this chunk (0 when
                                     the chunk only ends with its heading)
    concept  -mentions->   concept   the section text names another concept
                                     from the same file (chunk_ids records
                                     where)

Chunk nodes reuse the chunk_id from chunking, so anything found in the
graph can be mapped back to the same chunks dense and BM25 retrieval use.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
from langchain_core.documents import Document

from config import Config
from src.rag.chunk_store import load_chunks
from src.rag.graph_extraction import (
    concept_aliases,
    mentions_alias,
    split_sections,
    tokenize,
)


def _concept_id(source: str, heading: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
    return f"concept:{Path(source).stem}/{slug}"


def build_graph(chunks: list[Document]) -> nx.DiGraph:
    graph = nx.DiGraph()
    segments = split_sections(chunks)
    section_chars: dict[tuple[str, str], int] = defaultdict(int)

    for source, section, chunk_id, text in segments:
        document_node = f"doc:{source}"
        chunk_node = f"chunk:{chunk_id}"
        graph.add_node(document_node, type="document", label=source)
        graph.add_node(chunk_node, type="chunk", label=chunk_id, source=source)

        if section:
            concept_node = _concept_id(source, section)
            graph.add_node(
                concept_node,
                type="concept",
                label=section,
                source=source,
                aliases=concept_aliases(section),
            )
            graph.add_edge(document_node, concept_node, relation="contains")
            graph.add_edge(concept_node, chunk_node, relation="covered_in")
            section_chars[(concept_node, chunk_node)] += len(text)

    totals: dict[str, int] = defaultdict(int)
    for (concept_node, _), chars in section_chars.items():
        totals[concept_node] += chars
    for (concept_node, chunk_node), chars in section_chars.items():
        share = chars / totals[concept_node] if totals[concept_node] else 0.0
        graph.edges[concept_node, chunk_node]["share"] = share

    concepts = {
        node: (data["source"], data["aliases"])
        for node, data in graph.nodes(data=True)
        if data["type"] == "concept"
    }

    # Only link concepts within the same file: a heading like "Processes"
    # is unambiguous in its own document but matches ordinary words
    # ("the process of ...") in others. Intro text before the first heading
    # is skipped; it would only repeat the document -> concept link.
    evidence: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source, section, chunk_id, text in segments:
        if not section:
            continue
        speaker = _concept_id(source, section)
        tokens = tokenize(text)
        for concept_node, (concept_source, aliases) in concepts.items():
            if concept_node == speaker or concept_source != source:
                continue
            if any(mentions_alias(tokens, alias) for alias in aliases):
                evidence[(speaker, concept_node)].add(chunk_id)

    for (speaker, concept_node), chunk_ids in sorted(evidence.items()):
        graph.add_edge(
            speaker,
            concept_node,
            relation="mentions",
            weight=len(chunk_ids),
            chunk_ids=sorted(chunk_ids),
        )

    return graph


def save_graph(graph: nx.DiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(graph, edges="edges")
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_graph(path: Path) -> nx.DiGraph:
    if not path.exists():
        raise FileNotFoundError(f"No knowledge graph at {path}. Run scripts/ingest.py first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return nx.node_link_graph(data, edges="edges")


def find_concepts(graph: nx.DiGraph, query: str) -> list[str]:
    """Concept nodes whose name (or any alias) appears in the query."""
    tokens = tokenize(query)
    return [
        node
        for node, data in graph.nodes(data=True)
        if data["type"] == "concept"
        and any(mentions_alias(tokens, alias) for alias in data["aliases"])
    ]


def describe_concept(graph: nx.DiGraph, node: str) -> dict:
    """Labels of related concepts and the chunk_ids covering this concept."""
    def label(n: str) -> str:
        return graph.nodes[n]["label"]

    return {
        "label": label(node),
        "source": graph.nodes[node]["source"],
        "mentions": sorted(
            label(n) for _, n, d in graph.out_edges(node, data=True) if d["relation"] == "mentions"
        ),
        "mentioned_by": sorted(
            label(n) for n, _, d in graph.in_edges(node, data=True) if d["relation"] == "mentions"
        ),
        "chunk_ids": sorted(
            label(n) for _, n, d in graph.out_edges(node, data=True) if d["relation"] == "covered_in"
        ),
    }


def graph_chunk_scores(graph: nx.DiGraph, query: str) -> list[tuple[str, float]]:
    """Rank chunk_ids by how directly the graph ties them to the query.

    Chunks of concepts named in the query come first, scored by the share
    of each concept's text they hold (summed over matched concepts). After
    them come chunks of concepts one `mentions` edge away (either
    direction), scored the same way. A related concept's chunk never
    outranks a chunk of a concept the query named. Chunks holding none of
    a concept's text, such as one that only ends with its heading, are
    left out. Ties keep document order. Returns [] if the query names no
    concept.
    """
    seeds = find_concepts(graph, query)
    if not seeds:
        return []

    neighbours: set[str] = set()
    for seed in seeds:
        for _, other, data in graph.out_edges(seed, data=True):
            if data["relation"] == "mentions":
                neighbours.add(other)
        for other, _, data in graph.in_edges(seed, data=True):
            if data["relation"] == "mentions":
                neighbours.add(other)
    neighbours -= set(seeds)

    def chunk_shares(concepts) -> dict[str, float]:
        shares: dict[str, float] = defaultdict(float)
        for concept in concepts:
            for _, chunk_node, data in graph.out_edges(concept, data=True):
                if data["relation"] == "covered_in" and data.get("share", 1.0) > 0:
                    shares[chunk_node] += data.get("share", 1.0)
        return shares

    direct = chunk_shares(seeds)
    related = {
        node: score
        for node, score in chunk_shares(sorted(neighbours)).items()
        if node not in direct
    }

    position = {node: i for i, node in enumerate(graph.nodes)}
    ranked = [
        node
        for group in (direct, related)
        for node in sorted(group, key=lambda n: (-group[n], position[n]))
    ]
    scores = {**related, **direct}
    return [(graph.nodes[node]["label"], scores[node]) for node in ranked]


def query_graph(
    query: str, config: Config, k: int | None = None
) -> list[tuple[Document, float]]:
    """Return (chunk, graph_score) pairs ordered best first.

    Empty when the query names no concept in the graph.
    """
    graph = load_graph(config.graph_file)
    chunks = {c.metadata["chunk_id"]: c for c in load_chunks(config.chunks_file)}
    scored = graph_chunk_scores(graph, query)

    top_k = k or config.graph_top_k
    return [(chunks[cid], score) for cid, score in scored if cid in chunks][:top_k]
