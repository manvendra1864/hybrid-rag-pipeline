"""Run a query through dense, BM25, and the fused hybrid ranking.

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


def _print_results(results) -> None:
    for rank, (chunk, score) in enumerate(results, start=1):
        meta = chunk.metadata
        print(f"Rank {rank}  |  score={score:.4f}  |  chunk_id={meta['chunk_id']}")
        print(f"  source={meta['source']}  page={meta['page']}")
        print(f"  text: {chunk.page_content.strip()[:300]}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dense, BM25, and hybrid (RRF) retrieval for a query."
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
    _print_results(query_hybrid(args.query, config, k=args.k))


if __name__ == "__main__":
    main()
