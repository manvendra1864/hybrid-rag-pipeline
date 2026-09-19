"""Week 4 demonstration: run both retrieval methods on two contrasting queries.

Run scripts/ingest.py first. Then:
    python scripts/demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from src.rag.bm25_retrieval import query_bm25
from src.rag.dense_retrieval import query_dense

EXACT_KEYWORD_QUERY = "What is policy HR-204?"
SEMANTIC_QUERY = "Can I live somewhere else for a few weeks and keep working?"


def _print_results(label: str, results) -> None:
    for rank, (chunk, score) in enumerate(results, start=1):
        meta = chunk.metadata
        print(f"  [{label}] Rank {rank}  score={score:.4f}  chunk_id={meta['chunk_id']}  source={meta['source']}")
        print(f"      {chunk.page_content.strip()[:150]}")


def main() -> None:
    config = load_config()

    print("=" * 70)
    print(f'Query 1 (exact keyword): "{EXACT_KEYWORD_QUERY}"')
    print("=" * 70)
    print("\nBM25:")
    _print_results("BM25", query_bm25(EXACT_KEYWORD_QUERY, config, k=3))
    print("\nDense:")
    _print_results("Dense", query_dense(EXACT_KEYWORD_QUERY, config, k=3))

    print()
    print("=" * 70)
    print(f'Query 2 (semantic paraphrase): "{SEMANTIC_QUERY}"')
    print("=" * 70)
    print("\nBM25:")
    _print_results("BM25", query_bm25(SEMANTIC_QUERY, config, k=3))
    print("\nDense:")
    _print_results("Dense", query_dense(SEMANTIC_QUERY, config, k=3))
    print()


if __name__ == "__main__":
    main()
