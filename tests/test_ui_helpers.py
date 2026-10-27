from types import SimpleNamespace

import groq
import httpx
import pytest
from langchain_core.documents import Document

import src.rag.ui_helpers as ui
from src.rag.answer_generation import INSUFFICIENT_MESSAGE
from src.rag.knowledge_graph import build_graph, save_graph


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
