# Hybrid RAG Pipeline

A hybrid retrieval-augmented generation pipeline built with LangChain, combining
dense (embedding) retrieval with sparse (BM25) keyword retrieval.

Full target architecture:

```
Documents -> loading -> chunking -> dense retrieval -> hybrid fusion -> LLM -> grounded answer
                                  -> BM25 retrieval  ------^
```

## Current scope (Week 4)

This is an internship project built incrementally. Week 4 covers everything up
to independent dense and BM25 retrieval:

```
Documents
    -> Document loader
    -> Chunking + metadata
    -> Dense retrieval (Chroma + sentence-transformers)
    -> BM25 retrieval (rank_bm25)
```

Hybrid fusion, LLM-based answer generation, and evaluation are later work (see
"What's next" below). Dense and BM25 can currently be run and inspected
independently, but they are not yet combined.

## Architecture

1. **Loader** (`src/rag/loader.py`) - reads every `.md` file in `data/documents/`
   into a LangChain `Document` per file, using `DirectoryLoader` +
   `TextLoader`. Metadata: `source` (filename), `doc_type`.
2. **Chunker** (`src/rag/chunker.py`) - splits each document with
   `RecursiveCharacterTextSplitter` and stamps every chunk with a
   deterministic `chunk_id` (`<filename>-p<page>-c<index>`), plus `page`.
   Re-running this on the same corpus produces the same IDs.
3. **Chunk store** (`src/rag/chunk_store.py`) - `scripts/ingest.py` writes the
   chunk list to `storage/chunks.json` once, so dense and BM25 retrieval
   operate on the exact same chunks instead of re-chunking independently.
4. **Dense retrieval** (`src/rag/dense_retrieval.py`) - embeds chunks with a
   local `sentence-transformers` model via `langchain-huggingface`, indexes
   them in a persisted `Chroma` vector store, and returns the closest chunks
   for a query (with a distance score, lower = closer).
5. **BM25 retrieval** (`src/rag/bm25_retrieval.py`) - tokenizes chunk text and
   scores it against a query with `rank_bm25.BM25Okapi`. The BM25 index
   rebuilds from `chunks.json` on every query call; this is cheap at this
   corpus size and keeps the index trivially in sync with the chunk store.

## Project structure

```
config.py                    Central configuration (paths, chunk size, top-k, model)
data/documents/               Demo corpus (Acme Corp internal docs, synthetic)
src/rag/loader.py             Document loading
src/rag/chunker.py            Chunking + deterministic chunk IDs
src/rag/chunk_store.py        Save/load the shared chunk list as JSON
src/rag/dense_retrieval.py    Chroma index build + query
src/rag/bm25_retrieval.py     BM25 query
scripts/ingest.py             Load -> chunk -> save chunk store -> build dense index
scripts/query_dense.py        CLI: run a dense query
scripts/query_bm25.py         CLI: run a BM25 query
scripts/demo.py               Runs both methods on two contrasting example queries
tests/                        pytest suite (see Testing below)
storage/                      Generated: chunks.json + Chroma index (gitignored)
```

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

First run downloads the embedding model (`sentence-transformers/all-MiniLM-L6-v2`,
~80MB) from Hugging Face and caches it locally.

## Usage

Build the chunk store and dense index from the corpus in `data/documents/`:

```bash
python scripts/ingest.py
```

Run a dense (semantic) query:

```bash
python scripts/query_dense.py "How does remote work approval work?"
```

Run a BM25 (keyword) query:

```bash
python scripts/query_bm25.py "What is policy HR-204?"
```

Both accept `--k N` to override the number of results.

Run the side-by-side demonstration used for the Week 4 walkthrough:

```bash
python scripts/demo.py
```

## Example queries

**Exact keyword** - `"What is policy HR-204?"`: BM25 ranks the chunk
containing the literal "HR-204" first because of the direct term match.

**Semantic paraphrase** - `"Can I live somewhere else for a few weeks and
keep working?"`: this doesn't share much vocabulary with the source text.
Dense retrieval ranks the correct remote-work chunk first by embedding
similarity; BM25 ranks it third, behind chunks that happen to share more
words ("working", "weeks") but are less relevant. This contrast is the
main reason both retrieval methods exist.

## Configuration

Set via environment variables, all optional (see `config.py` for defaults):

| Variable            | Default                                   | Meaning                          |
|----------------------|-------------------------------------------|-----------------------------------|
| `RAG_DOCS_DIR`       | `data/documents`                          | Corpus directory                  |
| `RAG_STORAGE_DIR`    | `storage`                                 | Base storage directory            |
| `RAG_CHROMA_DIR`     | `storage/chroma`                          | Chroma persistence directory      |
| `RAG_CHUNKS_FILE`    | `storage/chunks.json`                     | Shared chunk store                |
| `RAG_CHUNK_SIZE`     | `500`                                     | Characters per chunk              |
| `RAG_CHUNK_OVERLAP`  | `50`                                      | Character overlap between chunks  |
| `RAG_EMBEDDING_MODEL`| `sentence-transformers/all-MiniLM-L6-v2`  | Embedding model                   |
| `RAG_DENSE_TOP_K`    | `3`                                       | Dense results per query           |
| `RAG_BM25_TOP_K`     | `3`                                       | BM25 results per query            |

Chunk size/overlap are starting values picked for this small demo corpus, not
tuned values.

## Testing

```bash
python -m pytest -v
```

Covers: document loading, chunk generation and metadata, deterministic chunk
IDs across repeated runs, BM25 keyword retrieval, dense semantic retrieval,
metadata preservation through both retrieval paths, and a full pipeline smoke
test.

## Decisions and limitations

- Corpus is plain Markdown, loaded with `TextLoader`/`DirectoryLoader`. These
  files have no native pagination, so `page` is always `1`; a paginated
  format such as PDF would populate it per-page instead.
- BM25 uses `rank_bm25` directly rather than LangChain's `BM25Retriever`
  wrapper, because the wrapper doesn't expose per-result scores, and seeing
  BM25 scores next to dense distances is useful for comparing the two.
  Everything else (loaders, splitter, embeddings, vector store) uses current
  LangChain integration packages.
- `langchain-community` (used for `DirectoryLoader`/`TextLoader`) currently
  prints a deprecation notice about being sunset in favor of standalone
  packages; there is no standalone replacement for these generic loaders yet,
  so this is the current correct import path.
- No LLM or API key configuration yet - not needed until hybrid fusion and
  answer generation.

## What's next (Week 5)

Week 5 adds hybrid fusion (combining dense + BM25 rankings, e.g. with
reciprocal rank fusion) and passes the fused context to an LLM to produce a
grounded answer with citations. Not implemented here.
