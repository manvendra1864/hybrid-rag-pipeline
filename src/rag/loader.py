"""Load raw documents from the corpus directory into LangChain Documents."""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_GLOB = "*.md"


def load_documents(docs_dir: Path) -> list[Document]:
    """Load every supported file in docs_dir as one Document per file.

    Metadata always includes 'source' (filename) and 'doc_type'. A 'page'
    key is not set here because plain text/markdown files have no native
    pagination; the chunker treats that case as page 1.
    """
    loader = DirectoryLoader(
        str(docs_dir),
        glob=SUPPORTED_GLOB,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    documents = loader.load()

    for doc in documents:
        source_path = Path(doc.metadata["source"])
        doc.metadata["source"] = source_path.name
        doc.metadata["doc_type"] = source_path.suffix.lstrip(".")

    return documents
