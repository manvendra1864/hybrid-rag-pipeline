# Hybrid RAG Pipeline

A small retrieval-augmented generation pipeline built with LangChain. It
loads a set of documents, splits them into chunks, and lets you search
those chunks with dense (embedding) retrieval, BM25 (keyword) retrieval,
or a hybrid ranking that combines both.

## Tech stack

- Python
- LangChain (`langchain-community`, `langchain-text-splitters`,
  `langchain-huggingface`, `langchain-chroma`)
- Chroma (vector store)
- sentence-transformers (`all-MiniLM-L6-v2`) for embeddings
- rank_bm25 for keyword scoring

## Project structure

```
config.py                    Paths, chunk size, top-k, embedding model
data/documents/               Sample corpus (Python, databases, OS, networks)
src/rag/loader.py             Loads documents
src/rag/chunker.py            Splits documents into chunks
src/rag/chunk_store.py        Saves/loads chunks as JSON
src/rag/dense_retrieval.py    Builds and queries the Chroma index
src/rag/bm25_retrieval.py     BM25 query
src/rag/hybrid_retrieval.py   Combines dense + BM25 rankings with RRF
scripts/ingest.py             Loads, chunks, and builds the dense index
scripts/query_dense.py        Run a dense query from the command line
scripts/query_bm25.py         Run a BM25 query from the command line
scripts/query_hybrid.py       Run a query through dense, BM25, and hybrid
scripts/demo.py               Runs all three methods on two example queries
tests/                        pytest suite
storage/                      Generated index files (gitignored)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the embedding model and caches it locally.

## Usage

Build the chunk store and dense index:

```bash
python scripts/ingest.py
```

Run a dense query:

```bash
python scripts/query_dense.py "How does a computer avoid running out of memory?"
```

Run a BM25 query:

```bash
python scripts/query_bm25.py "What is a foreign key?"
```

Run a hybrid query (dense + BM25 combined with reciprocal rank fusion):

```bash
python scripts/query_hybrid.py "How does a computer avoid running out of memory?"
```

All three accept `--k N` to change the number of results.

Run the demo (compares all three methods on two example queries):

```bash
python scripts/demo.py
```

Run the tests:

```bash
python -m pytest -v
```

## How it works

1. Documents in `data/documents/` are loaded into LangChain `Document`
   objects.
2. Each document is split into overlapping chunks, and each chunk gets a
   stable id based on its source file and position.
3. Dense retrieval embeds the chunks and the query, and returns the
   chunks whose embeddings are closest to the query's.
4. BM25 retrieval scores chunks by keyword overlap with the query,
   without using embeddings at all.
5. Hybrid retrieval runs both methods and combines their rankings with
   reciprocal rank fusion (RRF): each chunk gets a score based on its
   rank position in each list (1 / (60 + rank)), added up across both
   lists. A chunk found by both methods ranks higher than one found by
   only one.

All three methods search the same chunks, so their results can be
compared directly.
