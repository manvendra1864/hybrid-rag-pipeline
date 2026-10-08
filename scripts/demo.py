"""Demonstration: run each retrieval method on a few contrasting queries.

Run scripts/ingest.py first. Then:
    python scripts/demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from src.rag.bm25_retrieval import query_bm25
from src.rag.dense_retrieval import query_dense
from src.rag.hybrid_retrieval import query_hybrid

EXACT_KEYWORD_QUERY = "What is a foreign key in a database?"
SEMANTIC_QUERY = "How can a program keep running smoothly even if a piece of user input turns out to be invalid?"
GRAPH_QUERY = "What does DNS do when I open a website?"


def _print_results(label: str, results) -> None:
    for rank, (chunk, score) in enumerate(results, start=1):
        meta = chunk.metadata
        print(f"  [{label}] Rank {rank}  score={score:.4f}  chunk_id={meta['chunk_id']}  source={meta['source']}")
        print(f"      {chunk.page_content.strip()[:150]}")


def main() -> None:
    config = load_config()

    examples = [
        ("Query 1 (exact keyword)", EXACT_KEYWORD_QUERY),
        ("Query 2 (semantic paraphrase, names no concept)", SEMANTIC_QUERY),
        ("Query 3 (concept with a related concept)", GRAPH_QUERY),
    ]

    for title, query in examples:
        print("=" * 70)
        print(f'{title}: "{query}"')
        print("=" * 70)
        print("\nBM25:")
        _print_results("BM25", query_bm25(query, config, k=3))
        print("\nDense:")
        _print_results("Dense", query_dense(query, config, k=3))
        print("\nHybrid (RRF):")
        _print_results("Hybrid", query_hybrid(query, config, k=3))
        print("\nGraph-enhanced hybrid (RRF + knowledge graph):")
        _print_results("Graph", query_hybrid(query, config, k=3, use_graph=True))
        print()


if __name__ == "__main__":
    main()
