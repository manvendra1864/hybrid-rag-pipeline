"""Central configuration for NexusRAG.

Values can be overridden with environment variables, but sensible
defaults are provided so the project runs out of the box.
"""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    docs_dir: Path = PROJECT_ROOT / "data" / "documents"
    chroma_dir: Path = PROJECT_ROOT / "storage" / "chroma"
    chunks_file: Path = PROJECT_ROOT / "storage" / "chunks.json"
    graph_file: Path = PROJECT_ROOT / "storage" / "graph.json"

    # Chunking. These are starting values for a demo-sized corpus, not
    # tuned for any particular document type.
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Small model that runs comfortably on CPU with 8GB RAM.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    dense_top_k: int = 3
    bm25_top_k: int = 3
    hybrid_top_k: int = 3

    # Smoothing constant for reciprocal rank fusion. 60 is the value used
    # in the original RRF paper and is not sensitive to small corpus size.
    rrf_k: int = 60

    # Graph-enhanced retrieval. graph_weight scales the graph's vote in
    # fusion relative to dense and BM25 (1.0 = equal), so the graph can
    # nudge the ranking without overriding the other two.
    graph_top_k: int = 3
    graph_weight: float = 0.5

    # Groq model used for answers. The API key is read from the environment
    # (GROQ_API_KEY), never stored here.
    llm_model: str = "openai/gpt-oss-120b"
    # Seconds allowed for each Groq request attempt.
    llm_timeout_seconds: int = 60


def load_config() -> Config:
    return Config(
        docs_dir=Path(os.environ.get("RAG_DOCS_DIR", Config.docs_dir)),
        chroma_dir=Path(os.environ.get("RAG_CHROMA_DIR", Config.chroma_dir)),
        chunks_file=Path(os.environ.get("RAG_CHUNKS_FILE", Config.chunks_file)),
        graph_file=Path(os.environ.get("RAG_GRAPH_FILE", Config.graph_file)),
        chunk_size=int(os.environ.get("RAG_CHUNK_SIZE", Config.chunk_size)),
        chunk_overlap=int(os.environ.get("RAG_CHUNK_OVERLAP", Config.chunk_overlap)),
        embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", Config.embedding_model),
        dense_top_k=int(os.environ.get("RAG_DENSE_TOP_K", Config.dense_top_k)),
        bm25_top_k=int(os.environ.get("RAG_BM25_TOP_K", Config.bm25_top_k)),
        hybrid_top_k=int(os.environ.get("RAG_HYBRID_TOP_K", Config.hybrid_top_k)),
        rrf_k=int(os.environ.get("RAG_RRF_K", Config.rrf_k)),
        graph_top_k=int(os.environ.get("RAG_GRAPH_TOP_K", Config.graph_top_k)),
        graph_weight=float(os.environ.get("RAG_GRAPH_WEIGHT", Config.graph_weight)),
        llm_model=os.environ.get("RAG_LLM_MODEL", Config.llm_model),
        llm_timeout_seconds=int(os.environ.get("RAG_LLM_TIMEOUT", Config.llm_timeout_seconds)),
    )
