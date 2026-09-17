"""Split loaded documents into chunks with deterministic IDs."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split documents into chunks and stamp each with a stable chunk_id.

    The ID is derived from source filename + page + chunk index within
    that page, so re-running this on the same corpus reproduces the same
    IDs. Dense and BM25 retrieval both build their index from these same
    chunks, so the ID is what ties a result back to the same logical
    chunk across both retrieval paths.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[Document] = []
    for doc in documents:
        source = Path(doc.metadata["source"]).stem
        # Plain text/markdown has no native pagination.
        page = doc.metadata.get("page", 0) + 1

        doc_chunks = splitter.split_documents([doc])
        for index, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_id"] = f"{source}-p{page}-c{index}"
            chunk.metadata["page"] = page
            chunks.append(chunk)

    return chunks
