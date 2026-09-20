from config import PROJECT_ROOT
from src.rag.loader import load_documents

DOCS_DIR = PROJECT_ROOT / "data" / "documents"


def test_loads_all_corpus_files():
    documents = load_documents(DOCS_DIR)
    sources = {doc.metadata["source"] for doc in documents}
    assert sources == {
        "hr_policies.md",
        "it_security_policy.md",
        "benefits_guide.md",
        "onboarding_guide.md",
    }


def test_documents_have_content_and_type():
    documents = load_documents(DOCS_DIR)
    for doc in documents:
        assert doc.page_content.strip()
        assert doc.metadata["doc_type"] == "md"
