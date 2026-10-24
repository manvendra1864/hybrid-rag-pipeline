# Retrieval evaluation

This measures how well each retrieval method finds the right chunks for a set of
questions. It runs without calling an LLM, so it needs no API
key and gives the same numbers every run.

## Run it

Build the index once, then run the evaluation from the project root:

```
python scripts/ingest.py
python scripts/evaluate_retrieval.py
```

It prints a summary and writes `evaluation/results.md` (aggregate table and a
per-question table) and `evaluation/results.csv` (one row per question and method).
Options: `--k` (results kept per method, default 3), `--questions` (another question
file), `--output-dir`.

The four methods are the ones the project already has, called as they are: dense
(`query_dense`), BM25 (`query_bm25`), hybrid (`query_hybrid`) and graph-enhanced hybrid
(`query_hybrid` with `use_graph=True`). The evaluation only reads their ranked results.

## The questions

`questions.json` holds 20 questions about the four documents. Each one lists the chunks
that answer it:

```json
{
  "id": "q01",
  "question": "What is a primary key?",
  "relevant": [
    {"chunk_id": "databases-p1-c2", "evidence": "whose value uniquely identifies each row"}
  ]
}
```

`evidence` is a short phrase copied from the chunk's text. Before every run, and in the
tests, each `chunk_id` must exist and its `evidence` must appear in that chunk. If the
documents or chunking change, the run stops and lists the labels that no longer match,
so old labels cannot be used by mistake.

A chunk was marked relevant only if its own text contains part of the answer. A chunk
that only ends with the next section's heading was not marked. When an answer is split
across two chunks (for example normalization, TCP and UDP, virtual memory), both are
marked and the question counts as fully found only if both are retrieved.

## Metrics

Both are worked out on each method's top `k` results (3 by default) and averaged over
all questions.

- **Recall@3**: the share of a question's relevant chunks that are in the top 3. With
  one relevant chunk it is 1 or 0. With two, it can also be 0.5.
- **MRR** (mean reciprocal rank): for each question, 1 divided by the position of the
  first relevant chunk in the top 3 (1, 0.5, 0.33). If no relevant chunk is in the top 3,
  the question scores 0. The average is taken over all questions, so misses lower it.
- **Misses**: how many questions had no relevant chunk at all in the top 3.

## Limits of these numbers

- **The corpus is tiny.** There are 31 chunks, and each method returns 3 of them. Even a
  weak method finds a lot by chance, so scores are close to the maximum and the methods
  are hard to tell apart.
- **Twenty questions is a small sample.** One question can change MRR by up to 0.05 (less
  for Recall). A gap of one or two questions between methods should not be read as one
  method being better.
- **The labels are one person's judgment.** Relevance was decided by hand, by one
  labeler, from the text. Another person might also mark a neighbouring chunk. Relevance
  is either yes or no, with no partial credit for a chunk that helps only a little.
- **The questions were written after reading the chunks.** Many reuse words from the
  text, which favours BM25 and dense search alike. A few are reworded on purpose (for
  example q03, q09 and q16), but the set is not a sample of real user questions.
- **Only retrieval is measured.** It says nothing about how good the final answer is,
  and the graph-enhanced method depends on the rule-based concepts found in the question.
- **Hybrid uses its own inner top 3.** `query_hybrid` takes the top 3 of dense and of BM25
  (from `config.py`) and fuses them, then keeps `k`. Changing `--k` changes how many fused
  results are kept, not how many each inner method contributes.

Use the results to spot clear failures and to check that a change did not break
retrieval. To compare methods properly, the corpus and the question set would both need
to be much larger and labeled by more than one person.
