"""Look up concepts in the knowledge graph.

Usage:
    python scripts/query_graph.py                       (graph summary)
    python scripts/query_graph.py "What is a foreign key?"
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from src.rag.knowledge_graph import describe_concept, find_concepts, load_graph


def _print_summary(graph) -> None:
    node_types = Counter(d["type"] for _, d in graph.nodes(data=True))
    relations = Counter(d["relation"] for _, _, d in graph.edges(data=True))
    print(f"\nNodes: {dict(node_types)}")
    print(f"Edges: {dict(relations)}\n")

    for node, data in graph.nodes(data=True):
        if data["type"] != "document":
            continue
        concepts = [
            graph.nodes[c]["label"]
            for _, c, d in graph.out_edges(node, data=True)
            if d["relation"] == "contains"
        ]
        print(f"{data['label']}: {', '.join(concepts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge graph lookup over the demo corpus.")
    parser.add_argument("query", nargs="?", help="Question or search text. Omit for a graph summary.")
    args = parser.parse_args()

    config = load_config()
    graph = load_graph(config.graph_file)

    if not args.query:
        _print_summary(graph)
        return

    print(f'\nQuery: "{args.query}"\n')
    matches = find_concepts(graph, args.query)
    if not matches:
        print("No matching concepts.")
        return

    for node in matches:
        info = describe_concept(graph, node)
        print(f"Concept: {info['label']}  ({info['source']})")
        print(f"  mentions:     {', '.join(info['mentions']) or '-'}")
        print(f"  mentioned by: {', '.join(info['mentioned_by']) or '-'}")
        print(f"  chunks:       {', '.join(info['chunk_ids'])}")
        print()


if __name__ == "__main__":
    main()
