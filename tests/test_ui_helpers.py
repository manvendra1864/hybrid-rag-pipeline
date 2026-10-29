import importlib.util
from types import SimpleNamespace

import groq
import httpx
import pytest
from langchain_core.documents import Document

import src.rag.ui_helpers as ui
from config import PROJECT_ROOT
from src.rag.answer_generation import INSUFFICIENT_MESSAGE
from src.rag.chunk_store import load_chunks, save_chunks
from src.rag.chunker import chunk_documents
from src.rag.dense_retrieval import query_dense as real_query_dense
from src.rag.knowledge_graph import build_graph, save_graph
from src.rag.loader import load_documents


def _chunk(chunk_id, text, source="databases.md"):
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "source": source, "page": 1},
    )


SLOW = "Groq took too long to answer. Please try again shortly."
PRIMARY = _chunk("databases-p1-c0", "## Primary Keys\n\nA primary key identifies each row.")
FOREIGN = _chunk("databases-p1-c1", "## Foreign Keys\n\nA foreign key points to a primary key.")
OTHER = _chunk("databases-p1-c2", "## Views\n\nA view is a saved query.")


class FakeLLM:
    def __init__(self, reply="", error=None):
        self.reply, self.error = reply, error

    def invoke(self, messages):
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.reply)


def _status_error(code, message):
    """The error the Groq client raises for an HTTP error response."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.APIStatusError(message, response=httpx.Response(code, request=request), body=None)


@pytest.fixture(autouse=True)
def no_real_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


@pytest.fixture
def stubbed(test_config, monkeypatch):
    """Graph on disk, retrieval functions replaced, calls recorded."""
    chunks = [PRIMARY, FOREIGN, OTHER]
    save_graph(build_graph(chunks), test_config.graph_file)
    calls = []

    monkeypatch.setattr(ui, "query_dense", lambda q, c, k=None: calls.append(("dense", k)) or [(OTHER, 0.2), (FOREIGN, 0.4)])
    monkeypatch.setattr(ui, "query_bm25", lambda q, c, k=None: calls.append(("bm25", k)) or [(FOREIGN, 7.5)])

    def fake_hybrid(q, c, k=None, use_graph=False):
        calls.append(("graph" if use_graph else "hybrid", k))
        return [(FOREIGN, 0.04), (PRIMARY, 0.03)] if use_graph else [(OTHER, 0.03), (FOREIGN, 0.02)]

    monkeypatch.setattr(ui, "query_hybrid", fake_hybrid)
    return calls


# ---- small pure helpers ---------------------------------------------------


@pytest.mark.parametrize("text, expected", [(None, None), ("", None), ("   \n", None), ("  hi there ", "hi there")])
def test_validate_question(text, expected):
    assert ui.validate_question(text) == expected


def test_missing_files_lists_what_is_not_built_yet(test_config):
    assert ui.missing_files(test_config) == ["chunk store", "dense index", "knowledge graph"]

    test_config.chunks_file.write_text("[]")
    test_config.chroma_dir.mkdir()
    test_config.graph_file.write_text("{}")
    assert ui.missing_files(test_config) == []


def test_api_key_available_follows_the_environment(monkeypatch):
    assert not ui.api_key_available()
    monkeypatch.setenv("GROQ_API_KEY", "some-key")
    assert ui.api_key_available()


def test_graph_stats_counts_node_types_and_relationships():
    graph = build_graph([PRIMARY, FOREIGN, OTHER])

    stats = ui.graph_stats(graph)

    assert stats["Documents"] == 1
    assert stats["Concepts"] == 3
    assert stats["Chunks"] == 3
    assert stats["Relationships"] == graph.number_of_edges()


def test_chunk_records_keep_rank_source_id_score_and_text():
    records = ui.chunk_records([(FOREIGN, 0.5), (PRIMARY, 0.25)], notes={"databases-p1-c1": "hybrid rank 2"})

    assert [r["rank"] for r in records] == [1, 2]
    assert records[0]["source"] == "databases.md"
    assert records[0]["chunk_id"] == "databases-p1-c1"
    assert records[0]["score"] == 0.5
    assert records[0]["note"] == "hybrid rank 2"
    assert records[1]["note"] == ""
    assert records[0]["text"].startswith("## Foreign Keys")


def test_rank_notes_compare_graph_enhanced_with_hybrid():
    hybrid = [(OTHER, 0.1), (FOREIGN, 0.09)]
    enhanced = [(FOREIGN, 0.2), (PRIMARY, 0.1)]

    assert ui.rank_notes(hybrid, enhanced) == {
        "databases-p1-c1": "hybrid rank 2",
        "databases-p1-c0": "not in hybrid top 2",
    }


def test_citation_chips_escape_html():
    html_text = ui.citation_chips_html([("a<b>.md", "x-p1-c0"), ("databases.md", '"><script>alert(1)</script>')])

    assert "<script>" not in html_text
    assert "a&lt;b&gt;.md" in html_text
    assert html_text.count('class="chip"') == 2


def test_redact_hides_secrets_and_ignores_empty_ones():
    assert ui.redact("bad key abc123 and abc123 again", ["abc123", ""]) == "bad key *** and *** again"


# ---- saved evaluation results ----------------------------------------------

RESULTS_MD = """# Retrieval evaluation results

7 questions, top 2 results per method. Corpus: 1 documents, 3 chunks.

## Aggregate results

| Method | Recall@2 | MRR | Misses |
|---|---|---|---|
| Dense | 0.500 | 0.250 | 3 / 7 |
| Graph-enhanced hybrid | 0.123 | 0.456 | 0 / 7 |

## Per question

| ID | Question | Relevant | Dense | Graph-enhanced hybrid |
|---|---|---|---|---|
| q01 | What? | a | 1. a | 1. a |
"""


def test_evaluation_summary_is_read_as_saved_not_recalculated(tmp_path):
    path = tmp_path / "results.md"
    path.write_text(RESULTS_MD, encoding="utf-8")

    summary = ui.load_evaluation_summary(path)

    assert summary == {
        "questions": 7,
        "top_k": 2,
        "columns": ["Method", "Recall@2", "MRR"],
        "rows": [["Dense", "0.500", "0.250"], ["Graph-enhanced hybrid", "0.123", "0.456"]],
    }


def test_the_saved_results_file_in_the_project_can_be_read():
    summary = ui.load_evaluation_summary(ui.EVALUATION_RESULTS)

    assert [row[0] for row in summary["rows"]] == ui.METHODS
    assert summary["columns"] == ["Method", f"Recall@{summary['top_k']}", "MRR"]
    assert summary["questions"] == 20


def test_missing_evaluation_results_explain_how_to_create_them(tmp_path):
    with pytest.raises(ui.EvaluationResultsError, match="evaluate_retrieval.py"):
        ui.load_evaluation_summary(tmp_path / "nope.md")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "# Results\n\nnothing useful here\n",
        "## Aggregate results\n\n| Method | Misses |\n|---|---|\n| Dense | 0 |\n",
        "## Aggregate results\n\n| Method | Recall@3 | MRR |\n|---|---|---|\n| Dense | 1.0 |\n",
    ],
)
def test_unreadable_evaluation_results_raise_a_clear_error(tmp_path, text):
    path = tmp_path / "results.md"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ui.EvaluationResultsError, match="expected format"):
        ui.load_evaluation_summary(path)


def test_a_results_file_that_is_not_text_gives_a_clear_error(tmp_path):
    path = tmp_path / "results.md"
    path.write_bytes(bytes([0xFF, 0xFE, 0x00, 0x62]))

    with pytest.raises(ui.EvaluationResultsError, match="could not be read"):
        ui.load_evaluation_summary(path)


# ---- search index checks and automatic build --------------------------------


def _write_index(config, chunks, dense_index_builder, dense_chunks=None):
    save_chunks(chunks, config.chunks_file)
    save_graph(build_graph(chunks), config.graph_file)
    dense_index_builder(chunks if dense_chunks is None else dense_chunks, config.chroma_dir)


def test_a_complete_index_has_no_problems(test_config, dense_index_builder):
    _write_index(test_config, [PRIMARY, FOREIGN, OTHER], dense_index_builder)

    assert ui.index_problems(test_config) == []


def test_everything_missing_is_reported(test_config):
    assert ui.index_problems(test_config) == ["chunk store", "dense index", "knowledge graph"]


@pytest.mark.parametrize("content", ["not json at all", "[]"])
def test_an_unreadable_or_empty_chunk_store_is_a_problem(test_config, dense_index_builder, content):
    _write_index(test_config, [PRIMARY, FOREIGN], dense_index_builder)
    test_config.chunks_file.write_text(content, encoding="utf-8")

    assert ui.index_problems(test_config) == ["chunk store", "dense index"]


def test_a_corrupt_graph_file_is_a_problem(test_config, dense_index_builder):
    _write_index(test_config, [PRIMARY, FOREIGN], dense_index_builder)
    test_config.graph_file.write_text("{broken", encoding="utf-8")

    assert ui.index_problems(test_config) == ["knowledge graph"]


def test_a_dense_folder_without_an_index_is_a_problem(test_config):
    save_chunks([PRIMARY, FOREIGN], test_config.chunks_file)
    save_graph(build_graph([PRIMARY, FOREIGN]), test_config.graph_file)
    test_config.chroma_dir.mkdir()

    assert ui.index_problems(test_config) == ["dense index"]


def test_a_dense_index_cut_short_is_a_problem(test_config, dense_index_builder):
    _write_index(test_config, [PRIMARY, FOREIGN], dense_index_builder, dense_chunks=[PRIMARY])

    assert ui.index_problems(test_config) == ["dense index"]


def test_an_unreadable_dense_database_is_a_problem(test_config):
    save_chunks([PRIMARY, FOREIGN], test_config.chunks_file)
    save_graph(build_graph([PRIMARY, FOREIGN]), test_config.graph_file)
    test_config.chroma_dir.mkdir()
    (test_config.chroma_dir / "chroma.sqlite3").write_bytes(b"this is not a database" * 50)

    assert ui.index_problems(test_config) == ["dense index"]


def test_ensure_index_does_nothing_when_the_index_is_ready(test_config, dense_index_builder, monkeypatch):
    _write_index(test_config, [PRIMARY, FOREIGN], dense_index_builder)
    calls = []
    monkeypatch.setattr(ui, "build_index", lambda config: calls.append(config))

    assert ui.ensure_index(test_config) is None
    assert calls == []


def test_ensure_index_builds_what_is_missing(test_config, dense_index_builder, monkeypatch):
    calls = []

    def fake_build(config):
        calls.append(config)
        _write_index(config, [PRIMARY, FOREIGN], dense_index_builder)

    monkeypatch.setattr(ui, "build_index", fake_build)

    assert ui.ensure_index(test_config) is None
    assert calls == [test_config]
    assert ui.index_problems(test_config) == []


def test_ensure_index_reports_a_failed_build_briefly_and_logs_the_traceback(test_config, monkeypatch, caplog):
    def failing(config):
        raise OSError("disk full" + chr(10) + "second line with technical details" + "x" * 500)

    monkeypatch.setattr(ui, "build_index", failing)

    message = ui.ensure_index(test_config)

    assert message == "Could not build the search index (OSError): disk full"
    assert any(record.exc_info for record in caplog.records)  # the full traceback is in the log


def test_a_very_long_build_error_is_cut_short(test_config, monkeypatch):
    monkeypatch.setattr(ui, "build_index", lambda config: (_ for _ in ()).throw(RuntimeError("y" * 1000)))

    assert len(ui.ensure_index(test_config)) < 300


def test_a_build_error_never_shows_the_api_key(test_config, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sekret-build-key")
    monkeypatch.setattr(ui, "build_index", lambda config: (_ for _ in ()).throw(RuntimeError("failed with sekret-build-key")))

    message = ui.ensure_index(test_config)

    assert "sekret-build-key" not in message
    assert "***" in message


def test_ensure_index_reports_an_index_that_is_still_unusable(test_config, monkeypatch):
    monkeypatch.setattr(ui, "build_index", lambda config: None)  # "succeeds" without writing anything

    assert ui.ensure_index(test_config) == (
        "The search index was built but is not usable yet: chunk store, dense index, knowledge graph."
    )


def test_build_index_runs_the_real_ingestion_script_and_gives_a_usable_index(test_config):
    assert ui.index_problems(test_config) != []

    ui.build_index(test_config)

    assert ui.index_problems(test_config) == []
    corpus = chunk_documents(load_documents(test_config.docs_dir), test_config.chunk_size, test_config.chunk_overlap)
    assert len(load_chunks(test_config.chunks_file)) == len(corpus)
    results = real_query_dense("What is a foreign key?", test_config)  # the dense index really answers
    assert results and results[0][0].metadata["chunk_id"].startswith("databases")


def test_the_ingest_script_still_works_from_the_command_line_without_arguments(test_config, dense_index_builder, monkeypatch):
    for name, path in (
        ("RAG_CHUNKS_FILE", test_config.chunks_file),
        ("RAG_GRAPH_FILE", test_config.graph_file),
        ("RAG_CHROMA_DIR", test_config.chroma_dir),
    ):
        monkeypatch.setenv(name, str(path))
    monkeypatch.setattr("src.rag.dense_retrieval.build_dense_index", lambda chunks, config: dense_index_builder(chunks, config.chroma_dir))
    spec = importlib.util.spec_from_file_location("ingest_script_for_test", PROJECT_ROOT / "scripts" / "ingest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main()  # no config given: it reads the environment, as `python scripts/ingest.py` does

    assert ui.index_problems(test_config) == []


# ---- run_question ----------------------------------------------------------


def test_run_question_uses_all_four_methods_and_writes_the_answer(test_config, stubbed, monkeypatch):
    llm = FakeLLM("A foreign key points to a primary key [databases.md, databases-p1-c1].")
    monkeypatch.setattr(ui, "create_llm", lambda config: llm)

    turn = ui.run_question("What is a foreign key?", test_config, k=2)

    assert sorted(stubbed) == sorted([("dense", 2), ("bm25", 2), ("hybrid", 2), ("graph", 2)])
    assert list(turn["comparison"]) == ui.METHODS
    assert turn["notice"] is None
    assert turn["sufficient"] is True
    assert "foreign key points to a primary key" in turn["answer"]
    assert turn["sources"] == [("databases.md", "databases-p1-c1")]
    # the chunks shown as retrieved are the graph-enhanced ones, with rank notes
    assert [r["chunk_id"] for r in turn["retrieved"]] == ["databases-p1-c1", "databases-p1-c0"]
    assert turn["retrieved"][0]["note"] == "hybrid rank 2"
    assert turn["retrieved"][1]["note"] == "not in hybrid top 2"
    assert turn["concepts"] == ["Foreign Keys"]


def test_answer_is_written_from_the_graph_enhanced_chunks(test_config, stubbed, monkeypatch):
    seen = {}

    class Capture(FakeLLM):
        def invoke(self, messages):
            seen["context"] = messages[1].content
            return SimpleNamespace(content="ok")

    monkeypatch.setattr(ui, "create_llm", lambda config: Capture())

    ui.run_question("q", test_config, k=2)

    assert "databases-p1-c1" in seen["context"] and "databases-p1-c0" in seen["context"]
    assert "databases-p1-c2" not in seen["context"]  # only in plain hybrid, not graph-enhanced


def test_missing_api_key_still_returns_the_retrieval(test_config, stubbed):
    turn = ui.run_question("What is a foreign key?", test_config, k=2)

    assert turn["answer"] is None
    assert turn["notice"]["level"] == "warning"
    assert "No Groq API key found" in turn["notice"]["text"]
    assert "\n\n" in turn["notice"]["text"]
    assert len(turn["retrieved"]) == 2
    assert list(turn["comparison"]) == ui.METHODS


def test_api_error_is_shown_without_the_key(test_config, stubbed, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sekret-value-123")
    llm = FakeLLM(error=RuntimeError("request rejected for key sekret-value-123"))
    monkeypatch.setattr(ui, "create_llm", lambda config: llm)

    turn = ui.run_question("q", test_config, k=2)

    assert turn["answer"] is None
    assert turn["notice"]["level"] == "error"
    assert "sekret-value-123" not in str(turn)
    assert turn["notice"]["text"] == "The Groq request failed (RuntimeError). Please try again."


def test_a_503_shows_a_short_retry_message_and_keeps_the_retrieval(test_config, stubbed, monkeypatch):
    error = _status_error(503, "overloaded " * 50)
    monkeypatch.setattr(ui, "create_llm", lambda config: FakeLLM(error=error))

    turn = ui.run_question("q", test_config, k=2)

    assert turn["answer"] is None
    assert turn["notice"] == {
        "level": "error",
        "text": "Groq is temporarily unavailable (503). Please try again in a few moments.",
    }
    assert "overloaded" not in str(turn["notice"])
    assert len(turn["retrieved"]) == 2


def test_a_timeout_shows_a_short_retry_message(test_config, stubbed, monkeypatch):
    monkeypatch.setattr(ui, "create_llm", lambda config: FakeLLM(error=httpx.ReadTimeout("read timed out")))

    turn = ui.run_question("q", test_config, k=2)

    assert turn["notice"]["level"] == "error"
    assert turn["notice"]["text"] == SLOW


def test_a_daily_quota_error_is_shown_with_the_retrieval_kept(test_config, stubbed, monkeypatch):
    error = _status_error(429, "Rate limit reached on tokens per day (TPD): Limit 200000, Used 199900. " * 3)
    monkeypatch.setattr(ui, "create_llm", lambda config: FakeLLM(error=error))

    turn = ui.run_question("q", test_config, k=2)

    assert turn["answer"] is None
    assert turn["notice"]["level"] == "error"
    assert turn["notice"]["text"].startswith("The daily Groq request limit for this API key is used up (429).")
    assert "Limit 200000" not in str(turn["notice"])
    assert len(turn["retrieved"]) == 2


def test_a_504_deadline_exceeded_shows_the_same_message_as_a_timeout(test_config, stubbed, monkeypatch):
    error = _status_error(504, "late " * 50)
    monkeypatch.setattr(ui, "create_llm", lambda config: FakeLLM(error=error))

    turn = ui.run_question("q", test_config, k=2)

    assert turn["answer"] is None
    assert turn["notice"] == {"level": "error", "text": SLOW}
    assert "late" not in str(turn["notice"])
    assert len(turn["retrieved"]) == 2


def test_unexpected_client_error_is_contained_and_redacted(test_config, stubbed, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sekret-value-456")

    def broken_client(config):
        raise ValueError("cannot build client with sekret-value-456")

    monkeypatch.setattr(ui, "create_llm", broken_client)

    turn = ui.run_question("q", test_config, k=2)

    assert turn["notice"]["level"] == "error"
    assert "sekret-value-456" not in str(turn)


def test_insufficient_answer_has_no_sources(test_config, stubbed, monkeypatch):
    monkeypatch.setattr(ui, "create_llm", lambda config: FakeLLM(INSUFFICIENT_MESSAGE))

    turn = ui.run_question("Who won the 1998 World Cup?", test_config, k=2)

    assert turn["sufficient"] is False
    assert turn["sources"] == []


def test_nothing_retrieved_needs_no_model_or_key(test_config, stubbed, monkeypatch):
    monkeypatch.setattr(ui, "query_hybrid", lambda q, c, k=None, use_graph=False: [])

    def must_not_run(config):
        raise AssertionError("no client should be created when nothing was retrieved")

    monkeypatch.setattr(ui, "create_llm", must_not_run)

    turn = ui.run_question("q", test_config, k=2)

    assert turn["notice"] is None
    assert turn["sufficient"] is False
    assert turn["retrieved"] == []


def test_missing_index_becomes_a_message_not_an_exception(test_config, monkeypatch):
    def no_index(q, c, k=None):
        raise FileNotFoundError("No dense index at x. Run scripts/ingest.py first.")

    monkeypatch.setattr(ui, "query_dense", no_index)

    turn = ui.run_question("q", test_config, k=2)

    assert turn["notice"]["level"] == "error"
    assert "ingest.py" in turn["notice"]["text"]
    assert turn["comparison"] == {}
