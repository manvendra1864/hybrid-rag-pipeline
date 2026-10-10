# Hybrid RAG Pipeline

A small retrieval-augmented generation pipeline built with LangChain. It
loads a set of documents, splits them into chunks, and lets you search
those chunks with dense (embedding) retrieval, BM25 (keyword) retrieval,
or a hybrid ranking that combines both. A small knowledge graph built
from the same chunks can be added to the hybrid ranking.

## Tech stack

- Python
- LangChain (`langchain-community`, `langchain-text-splitters`,
  `langchain-huggingface`, `langchain-chroma`)
- Chroma (vector store)
- sentence-transformers (`all-MiniLM-L6-v2`) for embeddings
- rank_bm25 for keyword scoring
- NetworkX for the knowledge graph

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
src/rag/graph_extraction.py   Finds sections and concept names in chunks
src/rag/knowledge_graph.py    Builds, saves, and looks up the knowledge graph
scripts/ingest.py             Loads, chunks, builds the dense index and graph
scripts/query_dense.py        Run a dense query from the command line
scripts/query_bm25.py         Run a BM25 query from the command line
scripts/query_hybrid.py       Compare dense, BM25, hybrid, and graph-enhanced hybrid
scripts/query_graph.py        Look up concepts in the knowledge graph
scripts/demo.py               Runs the methods on three example queries
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

Build the chunk store, dense index, and knowledge graph:

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

Run a hybrid query. This prints dense, BM25, ordinary hybrid (dense +
BM25 combined with reciprocal rank fusion), and graph-enhanced hybrid
(the same fusion plus the knowledge graph) one after another, so they
can be compared. The graph-enhanced results are labelled with the
rank each chunk had in ordinary hybrid, or "not in hybrid top N":

```bash
python scripts/query_hybrid.py "What does DNS do when I open a website?"
```

All of the query scripts accept `--k N` to change the number of results.

Look up concepts in the knowledge graph (run without a query for a summary):

```bash
python scripts/query_graph.py "What is a foreign key?"
```

Run the demo (compares the methods on three example queries):

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
6. Graph-enhanced hybrid adds the knowledge graph as a third ranked list
   in the same fusion (see below).

All methods search the same chunks, so their results can be compared
directly.

The knowledge graph is built from the same chunks during ingestion:

- Each `## Heading` in a document is a concept (for example "Foreign
  Keys"). Concepts are linked to their document and to the chunks that
  hold their text.
- If the text under one concept names another concept from the same
  file, the first concept gets a `mentions` edge to the second, with the
  chunk ids where it was found.
- Names are matched by simple rules (headings and plural forms), not by
  a language model.
- Each concept-to-chunk link records what share of the concept's text
  is in that chunk. A chunk that only ends with a concept's heading has
  share 0 and is not treated as covering it.
- The graph is saved to `storage/graph.json`. Chunk nodes use the same
  chunk ids as retrieval.

### Graph-enhanced retrieval

`query_hybrid(query, config, use_graph=True)` adds the graph to the
fusion:

1. Concepts named in the query are looked up in the graph (the same
   name matching as above).
2. Chunks holding those concepts' text are ranked first, by share of the
   concept's text. Chunks of concepts one `mentions` edge away, in
   either direction, follow. Related chunks never outrank chunks of a
   concept the query named. At most `graph_top_k` chunks are kept.
3. That list is fused with the dense and BM25 lists by RRF. The graph's
   vote is multiplied by `graph_weight` (default 0.5), so it can lift a
   chunk dense or BM25 already found but does not override them.
4. If the query names no concept, the graph list is empty and the result
   is exactly ordinary hybrid retrieval.

`graph_top_k` and `graph_weight` are set in `config.py` and can be
overridden with `RAG_GRAPH_TOP_K` and `RAG_GRAPH_WEIGHT`. The
`graph.json` file has to be rebuilt with `scripts/ingest.py` after
upgrading, because the share values are new.

Limitations: concepts are only the `##` headings and are matched by
name, so a query that describes a concept without naming it (for
example "errors" for "Exceptions") gets no graph help. Related concepts
are only linked within one file. The pipeline stops at retrieval; no
answer is generated from the retrieved chunks yet.
