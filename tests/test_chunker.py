from config import PROJECT_ROOT
from src.rag.chunker import chunk_documents
from src.rag.loader import load_documents

DOCS_DIR = PROJECT_ROOT / "data" / "documents"


def _build_chunks():
    documents = load_documents(DOCS_DIR)
    return chunk_documents(documents, chunk_size=500, chunk_overlap=50)


def test_chunks_are_produced():
    assert len(_build_chunks()) > 0


def test_chunks_have_ids_and_metadata():
    for chunk in _build_chunks():
        assert chunk.metadata["chunk_id"]
        assert chunk.metadata["source"]
        assert chunk.metadata["page"] == 1


def test_chunk_ids_are_unique():
    ids = [c.metadata["chunk_id"] for c in _build_chunks()]
    assert len(ids) == len(set(ids))


def test_chunk_ids_are_deterministic_across_runs():
    first_ids = [c.metadata["chunk_id"] for c in _build_chunks()]
    second_ids = [c.metadata["chunk_id"] for c in _build_chunks()]
    assert first_ids == second_ids
