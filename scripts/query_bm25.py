"""Run a query against the BM25 (keyword) retriever.

Usage:
    python scripts/query_bm25.py "What is policy HR-204?"
    python scripts/query_bm25.py "..." --k 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from src.rag.bm25_retrieval import query_bm25


def main() -> None:
    parser = argparse.ArgumentParser(description="NexusRAG: BM25 retrieval over the demo corpus.")
    parser.add_argument("query", help="Question or search text.")
    parser.add_argument("--k", type=int, default=None, help="Number of results (default from config).")
    args = parser.parse_args()

    config = load_config()
    results = query_bm25(args.query, config, k=args.k)

    print(f'\nQuery: "{args.query}"')
    print("Method: BM25 (keyword)\n")

    if not results:
        print("No results.")
        return

    for rank, (chunk, score) in enumerate(results, start=1):
        meta = chunk.metadata
        print(f"Rank {rank}  |  score={score:.4f}  |  chunk_id={meta['chunk_id']}")
        print(f"  source={meta['source']}  page={meta['page']}")
        print(f"  text: {chunk.page_content.strip()[:300]}")
        print()


if __name__ == "__main__":
    main()
