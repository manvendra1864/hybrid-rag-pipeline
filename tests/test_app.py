"""Run app.py headlessly with Streamlit's test runner. No browser, model, network or API key."""

import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import groq
import httpx
import streamlit as st
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

import src.rag.ui_helpers as ui
from config import PROJECT_ROOT
from src.rag.chunk_store import save_chunks
from src.rag.knowledge_graph import build_graph, save_graph

SECRET = "sekret-app-key-98765"


def _chunk(chunk_id, text):
    return Document(page_content=text, metadata={"chunk_id": chunk_id, "source": "databases.md", "page": 1})


PRIMARY = _chunk("databases-p1-c0", "## Primary Keys\n\nA primary key identifies each row.")
FOREIGN = _chunk("databases-p1-c1", "## Foreign Keys\n\nA foreign key points to a primary key.")


class FakeLLM:
    def __init__(self, reply="", error=None):
        self.reply, self.error = reply, error

    def invoke(self, messages):
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.reply)


EVAL_RESULTS = """# Retrieval evaluation results

9 questions, top 3 results per method.

## Aggregate results

| Method | Recall@3 | MRR | Misses |
|---|---|---|---|
| Dense | 0.111 | 0.222 | 1 / 9 |
| BM25 | 0.333 | 0.444 | 2 / 9 |
| Hybrid | 0.555 | 0.666 | 3 / 9 |
| Graph-enhanced hybrid | 0.777 | 0.888 | 4 / 9 |

## Per question

| ID | Question |
|---|---|
| q01 | What? |
"""


def _status_error(code, message):
    """The error the Groq client raises for an HTTP error response."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.APIStatusError(message, response=httpx.Response(code, request=request), body=None)


def _visible_text(at):
    """Every piece of text the page shows."""
    parts = []
    for group in (at.markdown, at.caption, at.warning, at.error, at.sidebar.markdown, at.sidebar.caption):
        parts += [str(element.value) for element in group]
    parts += [e.label for e in at.expander] + [b.label for b in at.button]
    return "\n".join(parts)


@pytest.fixture
def empty_app(test_config, monkeypatch):
    """The app with every external piece replaced and no search index on disk yet."""
    # these tests patch the already imported modules, so app.py must not re-import them
    monkeypatch.setattr(sys, "_nexusrag_loaded_at", time.time(), raising=False)
    for name, path in (
        ("RAG_CHUNKS_FILE", test_config.chunks_file),
        ("RAG_GRAPH_FILE", test_config.graph_file),
        ("RAG_CHROMA_DIR", test_config.chroma_dir),
    ):
        monkeypatch.setenv(name, str(path))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    # never read a real .env or load the real embedding model
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr("src.rag.dense_retrieval.get_embeddings", lambda config: object())
    monkeypatch.setattr(ui, "query_dense", lambda q, c, k=None: [(PRIMARY, 0.3), (FOREIGN, 0.5)])
    monkeypatch.setattr(ui, "query_bm25", lambda q, c, k=None: [(FOREIGN, 6.0)])
    monkeypatch.setattr(
        ui,
        "query_hybrid",
        lambda q, c, k=None, use_graph=False: [(FOREIGN, 0.04), (PRIMARY, 0.03)] if use_graph else [(PRIMARY, 0.03), (FOREIGN, 0.02)],
    )
    st.cache_resource.clear()
    results = test_config.chunks_file.parent / "results.md"
    results.write_text(EVAL_RESULTS, encoding="utf-8")
    monkeypatch.setattr(ui, "EVALUATION_RESULTS", results)

    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
    yield at
    st.cache_resource.clear()


@pytest.fixture
def app(empty_app, test_config, dense_index_builder):
    """The same app with a small, valid search index already on disk."""
    chunks = [PRIMARY, FOREIGN]
    save_chunks(chunks, test_config.chunks_file)
    save_graph(build_graph(chunks), test_config.graph_file)
    dense_index_builder(chunks, test_config.chroma_dir)
    return empty_app


def _ask(at, text):
    at.chat_input[0].set_value(text).run()
    return at


def test_first_page_shows_graph_stats_settings_and_examples(app):
    app.run()

    assert not app.exception
    metrics = {m.label: m.value for m in app.sidebar.metric}
    assert metrics["Documents"] == "1"
    assert metrics["Concepts"] == "2"
    assert metrics["Chunks"] == "2"
    assert app.sidebar.slider[0].value == 3
    assert len([b for b in app.button if b.label.startswith(("How are", "What does", "How can"))]) == 3
    assert "API key <b>missing</b>" in _visible_text(app)


def _project_module_names():
    return [name for name in sys.modules if name in ("config", "src") or name.startswith("src.")]


@pytest.fixture
def stale_server(monkeypatch):
    """Put the project's modules back as they were after a test that makes app.py re-import them."""
    saved = {name: sys.modules[name] for name in _project_module_names()}
    yield
    for name in _project_module_names():
        del sys.modules[name]
    sys.modules.update(saved)


@pytest.fixture
def no_model(monkeypatch):
    """The re-imported modules are new objects, so the embedding model is replaced at its source."""
    monkeypatch.setattr("langchain_huggingface.HuggingFaceEmbeddings", lambda **kwargs: object())


def test_a_redeploy_with_old_modules_still_in_memory_does_not_crash(app, stale_server, no_model, monkeypatch):
    # What happened after a redeploy: the server still held the previous src/rag/ui_helpers.py,
    # which has no ensure_index, and the new app.py failed on its import at line 16.
    old_module = types.ModuleType("src.rag.ui_helpers")
    added_by_the_new_version = ("ensure_index", "index_problems", "build_index")
    vars(old_module).update({k: v for k, v in vars(ui).items() if k not in added_by_the_new_version})
    monkeypatch.setitem(sys.modules, "src.rag.ui_helpers", old_module)
    monkeypatch.delattr(sys, "_nexusrag_loaded_at", raising=False)  # no earlier run of this app.py

    app.run()

    assert not app.exception
    assert [t.value for t in app.title] == ["NexusRAG"]
    assert len(app.chat_input) == 1
    assert hasattr(sys.modules["src.rag.ui_helpers"], "ensure_index")


def test_a_source_file_changed_after_the_last_run_is_loaded_again(app, stale_server, no_model, monkeypatch):
    before = sys.modules["src.rag.ui_helpers"]
    # the last run was just before the source file's own modification time, so the file is newer
    long_ago = Path(ui.__file__).stat().st_mtime - 10
    monkeypatch.setattr(sys, "_nexusrag_loaded_at", long_ago)

    app.run()

    assert not app.exception
    assert sys.modules["src.rag.ui_helpers"] is not before
    assert sys._nexusrag_loaded_at > long_ago


def test_unchanged_modules_are_kept_between_runs(app):
    before = {name: sys.modules[name] for name in _project_module_names()}

    app.run()
    app.run()

    assert not app.exception
    assert all(sys.modules[name] is module for name, module in before.items())


def test_the_page_shows_the_project_name_and_description(app):
    app.run()

    assert not app.exception
    assert [t.value for t in app.title] == ["NexusRAG"]
    assert (
        "A hybrid RAG system combining dense retrieval, BM25, and Knowledge Graph-enhanced "
        "retrieval to generate context-aware answers from documents."
    ) in _visible_text(app)


def test_the_saved_evaluation_results_are_shown_as_a_table_with_an_explanation(app):
    app.run()

    assert not app.exception
    assert "Retrieval evaluation" in [e.label for e in app.expander]
    table = app.dataframe[0].value
    assert list(table.columns) == ["Method", "Recall@3", "MRR"]
    assert table.values.tolist() == [
        ["Dense", "0.111", "0.222"],
        ["BM25", "0.333", "0.444"],
        ["Hybrid", "0.555", "0.666"],
        ["Graph-enhanced hybrid", "0.777", "0.888"],
    ]
    text = _visible_text(app)
    assert "9 hand-labelled questions, top 3 results per method" in text
    assert "<b>Recall@3</b>" in text and "<b>MRR</b>" in text


def test_the_evaluation_section_is_collapsed_by_default(app):
    app.run()

    section = next(e for e in app.expander if e.label == "Retrieval evaluation")
    assert section.proto.expanded is False
    assert len(app.chat_input) == 1  # the chat stays the main focus


def test_missing_evaluation_results_do_not_break_the_page(app, test_config):
    (test_config.chunks_file.parent / "results.md").unlink()

    app.run()

    assert not app.exception
    assert "No evaluation results yet" in _visible_text(app)
    assert len(app.dataframe) == 0
    assert len(app.chat_input) == 1  # the chat still works


def test_unreadable_evaluation_results_do_not_break_the_page(app, test_config):
    (test_config.chunks_file.parent / "results.md").write_text("not a results file", encoding="utf-8")

    app.run()

    assert not app.exception
    assert "not in the expected format" in _visible_text(app)
    assert len(app.chat_input) == 1


def test_a_question_gets_an_answer_with_sources_and_details(app, monkeypatch):
    monkeypatch.setattr(ui, "create_llm", lambda config: FakeLLM("A foreign key points to a primary key [databases.md, databases-p1-c1]."))
    app.run()

    _ask(app, "What is a foreign key?")

    assert not app.exception
    text = _visible_text(app)
    assert "What is a foreign key?" in text
    assert "A foreign key points to a primary key" in text
    assert "databases-p1-c1" in text  # the source chip
    labels = [e.label for e in app.expander]
    assert "Retrieved chunks (2)" in labels
    assert "Compare retrieval methods" in labels
    assert [t.label for t in app.tabs] == ["Dense", "BM25", "Hybrid", "Graph-enhanced hybrid"]
    assert len(app.chat_message) == 2


def test_example_buttons_disappear_once_a_question_was_asked(app, monkeypatch):
    monkeypatch.setattr(ui, "create_llm", lambda config: FakeLLM("ok [databases.md, databases-p1-c0]"))
    app.run()

    _ask(app, "anything")

    assert not [b for b in app.button if b.label.startswith(("How are", "What does", "How can"))]


def test_empty_question_asks_for_text_and_does_not_search(app, monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError("retrieval should not run for an empty question")

    monkeypatch.setattr(ui, "query_dense", must_not_run)
    app.run()

    _ask(app, "   ")

    assert not app.exception
    assert [w.value for w in app.warning] == ["Type a question first."]
    assert len(app.chat_message) == 0


def test_missing_api_key_still_shows_retrieval_and_says_why(app):
    app.run()

    _ask(app, "What is a foreign key?")

    assert not app.exception
    assert any("No Groq API key found" in w.value for w in app.warning)
    assert "Retrieved chunks (2)" in [e.label for e in app.expander]


def test_api_error_is_reported_and_the_key_is_not_shown(app, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    monkeypatch.setattr(ui, "create_llm", lambda config: FakeLLM(error=RuntimeError(f"rejected {SECRET}")))
    app.run()

    _ask(app, "What is a foreign key?")

    assert not app.exception
    assert any("Groq" in e.value for e in app.error)
    assert SECRET not in _visible_text(app)
    assert "API key <b>found</b>" in _visible_text(app)


def test_a_503_is_shown_as_a_short_message_without_raw_details(app, monkeypatch):
    raw = "RAW-DETAILS " * 40
    error = _status_error(503, raw)
    monkeypatch.setattr(ui, "create_llm", lambda config: FakeLLM(error=error))
    app.run()

    _ask(app, "What is a foreign key?")

    assert not app.exception
    assert [e.value for e in app.error] == ["Groq is temporarily unavailable (503). Please try again in a few moments."]
    assert "RAW-DETAILS" not in _visible_text(app)
    assert "Retrieved chunks (2)" in [e.label for e in app.expander]


def test_a_504_is_shown_as_a_short_message_without_raw_details(app, monkeypatch):
    error = _status_error(504, "RAW-DETAILS " * 40)
    monkeypatch.setattr(ui, "create_llm", lambda config: FakeLLM(error=error))
    app.run()

    _ask(app, "What is a foreign key?")

    assert not app.exception
    assert [e.value for e in app.error] == ["Groq took too long to answer. Please try again shortly."]
    assert "RAW-DETAILS" not in _visible_text(app)
    assert "Retrieved chunks (2)" in [e.label for e in app.expander]


def test_api_key_value_is_never_rendered_even_on_success(app, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    monkeypatch.setattr(ui, "create_llm", lambda config: FakeLLM("fine [databases.md, databases-p1-c1]"))
    app.run()

    _ask(app, "What is a foreign key?")

    assert SECRET not in _visible_text(app)
    assert "API key <b>found</b>" in _visible_text(app)


@pytest.fixture
def builder(test_config, monkeypatch, dense_index_builder):
    """Stand in for the real ingestion: fast, records its calls, and can be made to fail."""
    state = {"calls": 0, "errors": []}

    def fake_build(config):
        state["calls"] += 1
        if state["errors"]:
            raise state["errors"].pop(0)
        chunks = [PRIMARY, FOREIGN]
        save_chunks(chunks, config.chunks_file)
        save_graph(build_graph(chunks), config.graph_file)
        dense_index_builder(chunks, config.chroma_dir)

    monkeypatch.setattr(ui, "build_index", fake_build)
    return state


def test_a_ready_index_is_not_rebuilt(app, builder):
    app.run()

    assert not app.exception
    assert builder["calls"] == 0
    assert len(app.chat_input) == 1


def test_missing_storage_is_built_automatically_before_the_chat_is_enabled(empty_app, builder):
    empty_app.run()

    assert not empty_app.exception
    assert builder["calls"] == 1
    assert not empty_app.error
    assert len(empty_app.chat_input) == 1
    assert {m.label: m.value for m in empty_app.sidebar.metric}["Chunks"] == "2"


def test_the_index_is_built_once_not_on_every_rerun(empty_app, builder, monkeypatch):
    monkeypatch.setattr(ui, "create_llm", lambda config: FakeLLM("ok [databases.md, databases-p1-c1]"))
    empty_app.run()

    empty_app.run()
    _ask(empty_app, "What is a foreign key?")

    assert builder["calls"] == 1
    assert len(empty_app.chat_message) == 2


def test_a_failed_build_shows_a_clear_error_and_keeps_retrieval_off(empty_app, builder):
    builder["errors"].append(RuntimeError("no space left on device"))

    empty_app.run()

    assert not empty_app.exception
    assert [e.value for e in empty_app.error] == ["Could not build the search index (RuntimeError): no space left on device"]
    assert "Run `python scripts/ingest.py`" not in _visible_text(empty_app)  # not the old misleading message
    assert len(empty_app.chat_input) == 0
    assert "Try again" in [b.label for b in empty_app.button]


def test_a_failed_build_is_not_retried_on_every_rerun(empty_app, builder):
    builder["errors"].append(RuntimeError("boom"))
    empty_app.run()

    empty_app.sidebar.slider[0].set_value(2).run()
    empty_app.run()

    assert builder["calls"] == 1
    assert len(empty_app.error) == 1


def test_try_again_builds_once_more_and_then_enables_the_chat(empty_app, builder):
    builder["errors"].append(RuntimeError("network down"))
    empty_app.run()

    next(b for b in empty_app.button if b.label == "Try again").click().run()

    assert not empty_app.exception
    assert builder["calls"] == 2
    assert not empty_app.error
    assert len(empty_app.chat_input) == 1


def test_a_build_that_leaves_an_unusable_index_is_reported(empty_app, monkeypatch):
    monkeypatch.setattr(ui, "build_index", lambda config: None)  # "succeeds" but writes nothing

    empty_app.run()

    assert not empty_app.exception
    assert any("built but is not usable" in e.value for e in empty_app.error)
    assert len(empty_app.chat_input) == 0
