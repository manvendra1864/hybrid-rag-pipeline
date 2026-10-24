import csv
import importlib.util
import json

import pytest
from langchain_core.documents import Document

import src.rag.evaluation as evaluation
from config import PROJECT_ROOT
from src.rag.evaluation import (
    build_retrievers,
    check_labels,
    evaluate,
    load_questions,
    markdown_report,
    recall_at_k,
    reciprocal_rank,
    write_csv,
)

QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.json"


def _chunk(chunk_id, text="text"):
    return Document(page_content=text, metadata={"chunk_id": chunk_id, "source": "x.md", "page": 1})


def _results(*ids):
    return [(_chunk(i), 0.0) for i in ids]


# ---- metrics ---------------------------------------------------------------


def test_recall_counts_the_share_of_relevant_chunks_found():
    assert recall_at_k(["a", "b", "c"], ["a"], 3) == 1.0
    assert recall_at_k(["a", "x", "y"], ["a", "b"], 3) == 0.5
    assert recall_at_k(["a", "b", "x"], ["a", "b"], 3) == 1.0


def test_recall_only_looks_at_the_first_k():
    assert recall_at_k(["x", "y", "z", "a"], ["a"], 3) == 0.0
    assert recall_at_k(["x", "y", "z", "a"], ["a"], 4) == 1.0


def test_recall_is_zero_when_nothing_relevant_was_retrieved():
    assert recall_at_k(["x", "y", "z"], ["a", "b"], 3) == 0.0
    assert recall_at_k([], ["a"], 3) == 0.0


def test_a_repeated_retrieved_id_is_not_counted_twice():
    assert recall_at_k(["a", "a", "a"], ["a", "b"], 3) == 0.5


def test_reciprocal_rank_uses_the_first_relevant_chunk():
    assert reciprocal_rank(["a", "x", "y"], ["a"], 3) == 1.0
    assert reciprocal_rank(["x", "a", "y"], ["a"], 3) == 0.5
    assert reciprocal_rank(["x", "y", "a"], ["a"], 3) == pytest.approx(1 / 3)
    # a later relevant chunk does not change it
    assert reciprocal_rank(["x", "b", "a"], ["a", "b"], 3) == 0.5


def test_reciprocal_rank_is_zero_with_no_relevant_chunk_in_the_top_k():
    assert reciprocal_rank(["x", "y", "z"], ["a"], 3) == 0.0
    assert reciprocal_rank(["x", "y", "z", "a"], ["a"], 3) == 0.0
    assert reciprocal_rank([], ["a"], 3) == 0.0


def test_a_question_without_relevant_chunks_is_an_error_not_a_silent_score():
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], 3)
    with pytest.raises(ValueError):
        reciprocal_rank(["a"], [], 3)


# ---- evaluate() ------------------------------------------------------------

QUESTIONS = [
    {"id": "q1", "question": "first?", "relevant": [{"chunk_id": "a", "evidence": "e"}]},
    {"id": "q2", "question": "second?", "relevant": [{"chunk_id": "b", "evidence": "e"}, {"chunk_id": "c", "evidence": "e"}]},
]


def _stub_retrievers():
    answers = {
        "Good": {"first?": ["a", "x", "y"], "second?": ["b", "c", "x"]},
        "Late": {"first?": ["x", "a", "y"], "second?": ["x", "y", "b"]},
        "Blind": {"first?": ["x", "y", "z"], "second?": ["x", "y", "z"]},
    }
    return {name: (lambda q, table=table: _results(*table[q])) for name, table in answers.items()}


def test_evaluate_scores_every_method_on_every_question():
    result = evaluate(QUESTIONS, _stub_retrievers(), k=3)

    assert len(result["rows"]) == 6
    row = next(r for r in result["rows"] if r["question_id"] == "q2" and r["method"] == "Late")
    assert row["retrieved_ids"] == ["x", "y", "b"]
    assert row["recall"] == 0.5
    assert row["reciprocal_rank"] == pytest.approx(1 / 3)


def test_the_summary_averages_over_all_questions_and_counts_misses_as_zero():
    summary = evaluate(QUESTIONS, _stub_retrievers(), k=3)["summary"]

    assert summary["Good"] == {"questions": 2, "recall": 1.0, "mrr": 1.0, "misses": 0}
    assert summary["Late"]["recall"] == pytest.approx((1.0 + 0.5) / 2)
    assert summary["Late"]["mrr"] == pytest.approx((0.5 + 1 / 3) / 2)
    assert summary["Late"]["misses"] == 0
    # a method that finds nothing scores 0 and still counts both questions
    assert summary["Blind"] == {"questions": 2, "recall": 0.0, "mrr": 0.0, "misses": 2}


def test_evaluate_keeps_only_the_top_k_even_if_a_retriever_returns_more():
    retrievers = {"Long": lambda q: _results("x", "y", "z", "a")}

    result = evaluate(QUESTIONS[:1], retrievers, k=3)

    assert result["rows"][0]["retrieved_ids"] == ["x", "y", "z"]
    assert result["summary"]["Long"]["recall"] == 0.0


def test_evaluate_passes_each_question_text_to_the_retrievers():
    seen = []
    retrievers = {"Spy": lambda q: seen.append(q) or _results("a")}

    evaluate(QUESTIONS, retrievers, k=3)

    assert seen == ["first?", "second?"]


# ---- dataset ---------------------------------------------------------------


def test_the_question_file_has_15_to_20_questions_with_labels():
    questions = load_questions(QUESTIONS_FILE)

    assert 15 <= len(questions) <= 20
    assert all(q["question"].strip() and q["relevant"] for q in questions)


def test_every_label_matches_a_real_chunk_and_its_text(ingested_chunks):
    questions = load_questions(QUESTIONS_FILE)

    assert check_labels(questions, ingested_chunks) == []


def test_the_questions_cover_all_four_documents(ingested_chunks):
    questions = load_questions(QUESTIONS_FILE)
    sources = {c.metadata["chunk_id"]: c.metadata["source"] for c in ingested_chunks}

    covered = {sources[r["chunk_id"]] for q in questions for r in q["relevant"]}

    assert covered == {"python.md", "databases.md", "operating_systems.md", "networks.md"}


def test_check_labels_reports_bad_labels():
    chunks = [_chunk("a", "A primary key\nuniquely identifies a row."), _chunk("b", "something else")]
    questions = [
        {"id": "q1", "question": "?", "relevant": [{"chunk_id": "a", "evidence": "primary key uniquely identifies"}]},
        {"id": "q2", "question": "?", "relevant": [{"chunk_id": "zzz", "evidence": "x"}]},
        {"id": "q3", "question": "?", "relevant": [{"chunk_id": "b", "evidence": "not in that chunk"}]},
        {"id": "q3", "question": "?", "relevant": [{"chunk_id": "a", "evidence": "primary key"}, {"chunk_id": "a", "evidence": "row"}]},
    ]

    problems = check_labels(questions, chunks)

    assert "q2: unknown chunk id zzz" in problems
    assert "q3: evidence not found in b" in problems
    assert "q3: duplicate question id" in problems
    assert "q3: a chunk id is listed twice" in problems
    assert not any(p.startswith("q1") for p in problems)  # whitespace differences are ignored


def test_load_questions_rejects_a_question_without_labels(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"id": "q1", "question": "?", "relevant": []}]), encoding="utf-8")

    with pytest.raises(ValueError, match="q1"):
        load_questions(path)


# ---- retrievers use the existing functions ---------------------------------


def test_build_retrievers_call_the_four_existing_functions_with_k(test_config, monkeypatch):
    calls = []
    monkeypatch.setattr(evaluation, "query_dense", lambda q, c, k=None: calls.append(("dense", q, k)) or [])
    monkeypatch.setattr(evaluation, "query_bm25", lambda q, c, k=None: calls.append(("bm25", q, k)) or [])
    monkeypatch.setattr(
        evaluation,
        "query_hybrid",
        lambda q, c, k=None, use_graph=False: calls.append(("graph" if use_graph else "hybrid", q, k)) or [],
    )

    retrievers = build_retrievers(test_config, k=3)
    for retrieve in retrievers.values():
        retrieve("what?")

    assert list(retrievers) == ["Dense", "BM25", "Hybrid", "Graph-enhanced hybrid"]
    assert calls == [("dense", "what?", 3), ("bm25", "what?", 3), ("hybrid", "what?", 3), ("graph", "what?", 3)]


# ---- output ----------------------------------------------------------------


def test_csv_has_one_row_per_question_and_method(tmp_path):
    result = evaluate(QUESTIONS, _stub_retrievers(), k=3)
    path = tmp_path / "out" / "results.csv"

    write_csv(result, path)

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 6
    late = next(r for r in rows if r["question_id"] == "q2" and r["method"] == "Late")
    assert late["relevant_ids"] == "b c"
    assert late["retrieved_ids"] == "x y b"
    assert late["recall_at_3"] == "0.5000"
    assert late["reciprocal_rank"] == "0.3333"


def test_markdown_report_shows_the_measured_aggregates_and_marks_hits():
    result = evaluate(QUESTIONS, _stub_retrievers(), k=3)

    report = markdown_report(result, "Corpus: test.")

    assert "| Good | 1.000 | 1.000 | 0 / 2 |" in report
    assert "| Late | 0.750 | 0.417 | 0 / 2 |" in report
    assert "| Blind | 0.000 | 0.000 | 2 / 2 |" in report
    assert "Corpus: test." in report
    assert "1. **a**<br>2. x<br>3. y" in report  # the relevant chunk is bold


def test_the_same_inputs_give_identical_output(tmp_path):
    first = markdown_report(evaluate(QUESTIONS, _stub_retrievers(), k=3))
    second = markdown_report(evaluate(QUESTIONS, _stub_retrievers(), k=3))

    assert first == second


# ---- script ----------------------------------------------------------------


@pytest.fixture
def script():
    path = PROJECT_ROOT / "scripts" / "evaluate_retrieval.py"
    spec = importlib.util.spec_from_file_location("evaluate_retrieval_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_writes_results_without_calling_a_model(script, test_config, ingested_chunks, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(script, "load_config", lambda: test_config)

    def first_chunk_retriever(config, k):
        return {name: (lambda q: [(ingested_chunks[0], 0.0)]) for name in evaluation.METHODS}

    monkeypatch.setattr(script, "build_retrievers", first_chunk_retriever)
    monkeypatch.setenv("GROQ_API_KEY", "")
    out = tmp_path / "results"

    code = script.main(["--output-dir", str(out)])

    assert code == 0
    assert (out / "results.md").exists() and (out / "results.csv").exists()
    printed = capsys.readouterr().out
    assert "Graph-enhanced hybrid" in printed and "Recall@3" in printed
    assert "Corpus: 4 documents, 31 chunks." in (out / "results.md").read_text(encoding="utf-8")


def test_the_script_stops_when_labels_do_not_match_the_chunks(script, test_config, monkeypatch, tmp_path, capsys):
    from src.rag.chunk_store import save_chunks

    save_chunks([_chunk("only-p1-c0", "nothing relevant")], test_config.chunks_file)
    monkeypatch.setattr(script, "load_config", lambda: test_config)

    code = script.main(["--output-dir", str(tmp_path / "out")])

    assert code == 1
    assert "unknown chunk id" in capsys.readouterr().err
    assert not (tmp_path / "out" / "results.md").exists()


def test_the_script_explains_a_missing_chunk_store(script, test_config, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(script, "load_config", lambda: test_config)

    code = script.main(["--output-dir", str(tmp_path / "out")])

    assert code == 1
    assert "ingest.py" in capsys.readouterr().err
