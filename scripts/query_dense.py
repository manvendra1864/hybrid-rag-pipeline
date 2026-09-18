"""Run a query against the dense (embedding) retriever.

Usage:
    python scripts/query_dense.py "What is the remote work policy?"
    python scripts/query_dense.py "..." --k 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from src.rag.dense_retrieval import query_dense


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense retrieval over the demo corpus.")
    parser.add_argument("query", help="Question or search text.")
    parser.add_argument("--k", type=int, default=None, help="Number of results (default from config).")
    args = parser.parse_args()

    config = load_config()
    results = query_dense(args.query, config, k=args.k)

    print(f'\nQuery: "{args.query}"')
    print(f"Method: dense (embedding similarity, {config.embedding_model})\n")

    if not results:
        print("No results.")
        return

    for rank, (chunk, distance) in enumerate(results, start=1):
        meta = chunk.metadata
        print(f"Rank {rank}  |  distance={distance:.4f}  |  chunk_id={meta['chunk_id']}")
        print(f"  source={meta['source']}  page={meta['page']}")
        print(f"  text: {chunk.page_content.strip()[:300]}")
        print()


if __name__ == "__main__":
    main()
