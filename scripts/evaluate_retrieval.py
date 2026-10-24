"""Evaluate dense, BM25, hybrid and graph-enhanced hybrid retrieval on a labelled question set.

No language model is called. Run scripts/ingest.py first.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --k 3 --output-dir evaluation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT, load_config
from src.rag.chunk_store import load_chunks
from src.rag.evaluation import (
    DEFAULT_K,
    build_retrievers,
    check_labels,
    evaluate,
    load_questions,
    markdown_report,
    write_csv,
)

DEFAULT_QUESTIONS = PROJECT_ROOT / "evaluation" / "questions.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the four retrieval methods on the labelled questions.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="Question file (JSON).")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Results kept per method (default 3).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where results.md and results.csv go.")
    args = parser.parse_args(argv)

    config = load_config()
    try:
        chunks = load_chunks(config.chunks_file)
        questions = load_questions(args.questions)
        problems = check_labels(questions, chunks)
        if problems:
            print("The labels do not match the ingested chunks. Fix the labels or re-run scripts/ingest.py:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        result = evaluate(questions, build_retrievers(config, args.k), args.k)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    documents = len({c.metadata["source"] for c in chunks})
    note = f"Corpus: {documents} documents, {len(chunks)} chunks."
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.md").write_text(markdown_report(result, note), encoding="utf-8")
    write_csv(result, args.output_dir / "results.csv")

    print(f"{len(questions)} questions, top {args.k}")
    print(f"{'Method':<24}{'Recall@' + str(args.k):>10}{'MRR':>8}{'Misses':>8}")
    for method, s in result["summary"].items():
        print(f"{method:<24}{s['recall']:>10.3f}{s['mrr']:>8.3f}{s['misses']:>8}")
    print(f"\nWrote {args.output_dir / 'results.md'} and {args.output_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
