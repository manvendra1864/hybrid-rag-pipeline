"""Load the corpus, chunk it, and build the dense + BM25 indexes.

Usage:
    python scripts/ingest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from src.rag.chunker import chunk_documents
from src.rag.chunk_store import save_chunks
from src.rag.dense_retrieval import build_dense_index
from src.rag.loader import load_documents


def main() -> None:
    config = load_config()

    print(f"Loading documents from {config.docs_dir} ...")
    documents = load_documents(config.docs_dir)
    print(f"Loaded {len(documents)} document(s).")

    print(f"Chunking (size={config.chunk_size}, overlap={config.chunk_overlap}) ...")
    chunks = chunk_documents(documents, config.chunk_size, config.chunk_overlap)
    print(f"Produced {len(chunks)} chunk(s).")

    save_chunks(chunks, config.chunks_file)
    print(f"Saved chunk store to {config.chunks_file}")

    print(f"Building dense index at {config.chroma_dir} with {config.embedding_model} ...")
    build_dense_index(chunks, config)
    print("Dense index ready.")

    print("BM25 does not need a separate build step; it indexes chunks.json at query time.")


if __name__ == "__main__":
    main()
