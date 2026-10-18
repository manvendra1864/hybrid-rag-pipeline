# Hybrid RAG Pipeline

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
- Google Gemini (through langchain-google-genai) for the answers
- pytest for testing
