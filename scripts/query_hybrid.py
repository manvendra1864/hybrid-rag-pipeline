"""Run a query through dense, BM25, hybrid, and graph-enhanced hybrid retrieval.

Usage:
    python scripts/query_hybrid.py "What is a foreign key?"
    python scripts/query_hybrid.py "..." --k 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from src.rag.bm25_retrieval import query_bm25
from src.rag.dense_retrieval import query_dense
from src.rag.hybrid_retrieval import query_hybrid
from src.rag.knowledge_graph import find_concepts, load_graph


def _print_results(results, notes=None) -> None:
    for rank, (chunk, score) in enumerate(results, start=1):
        meta = chunk.metadata
        note = f"  [{notes[meta['chunk_id']]}]" if notes else ""
        print(f"Rank {rank}  |  score={score:.4f}  |  chunk_id={meta['chunk_id']}{note}")
        print(f"  source={meta['source']}  page={meta['page']}")
        print(f"  text: {chunk.page_content.strip()[:300]}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NexusRAG: compare dense, BM25, hybrid, and graph-enhanced hybrid retrieval for a query."
    )
    parser.add_argument("query", help="Question or search text.")
    parser.add_argument("--k", type=int, default=None, help="Number of hybrid results (default from config).")
    args = parser.parse_args()

    config = load_config()

    print(f'\nQuery: "{args.query}"')

    print("\n--- Dense ---")
    _print_results(query_dense(args.query, config))

    print("--- BM25 ---")
    _print_results(query_bm25(args.query, config))

    print("--- Hybrid (Reciprocal Rank Fusion) ---")
    hybrid = query_hybrid(args.query, config, k=args.k)
    _print_results(hybrid)

    print("--- Graph-enhanced hybrid (RRF + knowledge graph) ---")
    try:
        graph = load_graph(config.graph_file)
    except FileNotFoundError as error:
        print(f"Skipped: {error}")
        return

    concepts = [graph.nodes[n]["label"] for n in find_concepts(graph, args.query)]
    if concepts:
        print(f"Concepts matched in graph: {', '.join(concepts)}\n")
    else:
        print("No concept matched in graph, so this is the same as hybrid.\n")

    hybrid_rank = {c.metadata["chunk_id"]: i for i, (c, _) in enumerate(hybrid, start=1)}
    enhanced = query_hybrid(args.query, config, k=args.k, use_graph=True)
    notes = {
        c.metadata["chunk_id"]: (
            f"hybrid rank {hybrid_rank[c.metadata['chunk_id']]}"
            if c.metadata["chunk_id"] in hybrid_rank
            else f"not in hybrid top {len(hybrid)}"
        )
        for c, _ in enhanced
    }
    _print_results(enhanced, notes)


if __name__ == "__main__":
    main()
