"""Build and query a small knowledge graph over the existing chunks.

Nodes:
    document  one per source file                  doc:python.md
    concept   one per '## Heading'                 concept:python/exceptions
    chunk     one per existing chunk               chunk:python-p1-c5

Edges:
    document -contains->   concept
    concept  -covered_in-> chunk     the chunk holds part of that section
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

    for source, section, chunk_id, _ in segments:
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
