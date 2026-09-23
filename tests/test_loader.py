from config import PROJECT_ROOT
from src.rag.loader import load_documents

DOCS_DIR = PROJECT_ROOT / "data" / "documents"


def test_loads_all_corpus_files():
    documents = load_documents(DOCS_DIR)
    sources = {doc.metadata["source"] for doc in documents}
    assert sources == {
        "python.md",
        "databases.md",
        "operating_systems.md",
        "networks.md",
    }


def test_documents_have_content_and_type():
    documents = load_documents(DOCS_DIR)
    for doc in documents:
        assert doc.page_content.strip()
        assert doc.metadata["doc_type"] == "md"
