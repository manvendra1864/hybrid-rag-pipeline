import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT, Config
from src.rag.chunker import chunk_documents
from src.rag.chunk_store import save_chunks
from src.rag.loader import load_documents


@pytest.fixture
def test_config(tmp_path):
    """Config pointing at the real demo corpus but scratch storage paths."""
    return Config(
        docs_dir=PROJECT_ROOT / "data" / "documents",
        chroma_dir=tmp_path / "chroma",
        chunks_file=tmp_path / "chunks.json",
        graph_file=tmp_path / "graph.json",
    )


@pytest.fixture
def ingested_chunks(test_config):
    """Load + chunk the real corpus and write it to the scratch chunk store."""
    documents = load_documents(test_config.docs_dir)
    chunks = chunk_documents(documents, test_config.chunk_size, test_config.chunk_overlap)
    save_chunks(chunks, test_config.chunks_file)
    return chunks


@pytest.fixture
def dense_index_builder():
    """Build a small but real Chroma index with fake embeddings, so no model is loaded."""
    from langchain_chroma import Chroma
    from langchain_core.embeddings import DeterministicFakeEmbedding

    def build(chunks, chroma_dir):
        Chroma.from_documents(
            documents=chunks,
            embedding=DeterministicFakeEmbedding(size=8),
            persist_directory=str(chroma_dir),
            ids=[chunk.metadata["chunk_id"] for chunk in chunks],
        )

    return build
