"""Ask a question and get an answer grounded in the retrieved documents.

Retrieval is the graph-enhanced hybrid ranking; the top chunks are passed to
Gemini as the only context it may answer from. Set GEMINI_API_KEY (or
GOOGLE_API_KEY) in the environment or in a .env file first.

Usage:
    python scripts/ask.py "How are primary keys and foreign keys related?"
    python scripts/ask.py "..." --k 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from config import PROJECT_ROOT, load_config
from src.rag.answer_generation import (
    AnswerGenerationError,
    MissingApiKeyError,
    create_llm,
    generate_answer,
)
from src.rag.hybrid_retrieval import query_hybrid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Answer a question from the documents using graph-enhanced hybrid retrieval and Gemini."
    )
    parser.add_argument("query", help="The question to answer.")
    parser.add_argument("--k", type=int, default=None, help="Number of chunks to retrieve (default from config).")
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config()

    try:
        # Created first so a missing API key fails before the slower retrieval.
        llm = create_llm(config)
        results = query_hybrid(args.query, config, k=args.k, use_graph=True)
        chunks = [chunk for chunk, _ in results]
        answer = generate_answer(args.query, chunks, config, llm=llm)
    except (MissingApiKeyError, AnswerGenerationError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f'\nQuestion: "{args.query}"')
    retrieved = ", ".join(f"{c.metadata['chunk_id']} ({c.metadata['source']})" for c in chunks)
    print(f"Retrieved chunks: {retrieved or 'none'}")
    print(f"\nAnswer:\n{answer.text}")

    if answer.sufficient and answer.sources:
        print("\nSources cited:")
        for source, chunk_id in answer.sources:
            print(f"  - {source} (chunk {chunk_id})")
    elif answer.sufficient:
        print("\nSources cited: none (the answer did not cite a retrieved chunk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
