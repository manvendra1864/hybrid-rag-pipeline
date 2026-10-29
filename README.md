# NexusRAG

This is my internship project. It is a question answering system over a small set of
study notes (Python, databases, operating systems and networks). It splits the notes
into chunks and finds the ones that match a question using three methods: meaning
based search (dense), keyword search (BM25), and a small knowledge graph. The results
are combined into one ranking, and the best chunks are given to an LLM which writes
an answer and says which chunks it used.

## Technologies Used

- Python
- LangChain
- Chroma (vector database)
- sentence-transformers (all-MiniLM-L6-v2) for embeddings
- rank_bm25 for keyword search
- NetworkX for the knowledge graph
- Groq (through langchain-groq) for the answers
- Streamlit for the web interface
- pytest for testing

## Running the web interface

Build the index once, then start the app from the project root:

```
python scripts/ingest.py
streamlit run app.py
```

It opens at http://localhost:8501. Ask a question to see the answer with its sources,
the retrieved chunks, and a comparison of dense, BM25, hybrid and graph-enhanced
hybrid retrieval. Set `GROQ_API_KEY` in your environment or
in a `.env` file to get answers. Without a key the app still shows the retrieval results.
