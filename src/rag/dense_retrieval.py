"""Embedding-based retrieval backed by a local Chroma vector store."""

import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import Config


def get_embeddings(config: Config) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": "cpu"},
    )


def build_dense_index(chunks: list[Document], config: Config) -> None:
    """(Re)build the Chroma index from scratch for the given chunks."""
    if config.chroma_dir.exists():
        shutil.rmtree(config.chroma_dir)

    embeddings = get_embeddings(config)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(config.chroma_dir),
        ids=[chunk.metadata["chunk_id"] for chunk in chunks],
    )


def query_dense(
    query: str, config: Config, k: int | None = None
) -> list[tuple[Document, float]]:
    """Return (chunk, distance) pairs ordered best-match first.

    Distance is Chroma's raw similarity distance: lower means closer.
    """
    if not config.chroma_dir.exists():
        raise FileNotFoundError(
            f"No dense index at {config.chroma_dir}. Run scripts/ingest.py first."
        )

    embeddings = get_embeddings(config)
    store = Chroma(
        persist_directory=str(config.chroma_dir),
        embedding_function=embeddings,
    )
    return store.similarity_search_with_score(query, k=k or config.dense_top_k)
