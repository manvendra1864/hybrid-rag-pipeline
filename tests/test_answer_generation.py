import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import src.rag.answer_generation as answer_module
from config import PROJECT_ROOT, load_config
from src.rag.answer_generation import (
    INSUFFICIENT_MESSAGE,
    SYSTEM_PROMPT,
    AnswerGenerationError,
    MissingApiKeyError,
    build_messages,
    cited_sources,
    create_llm,
    format_context,
    generate_answer,
    get_api_key,
)


def _chunk(chunk_id, text, source="databases.md"):
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "source": source, "page": 1},
    )


PRIMARY = _chunk("databases-p1-c2", "A primary key uniquely identifies each row.")
FOREIGN = _chunk("databases-p1-c3", "A foreign key points to the primary key of another table.")


class FakeLLM:
    def __init__(self, reply="", error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.reply)


@pytest.fixture(autouse=True)
def no_real_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_format_context_includes_source_filename_chunk_id_and_text():
    context = format_context([PRIMARY, FOREIGN])

    assert "Source: databases.md | Chunk ID: databases-p1-c2" in context
    assert "A primary key uniquely identifies each row." in context
    assert "Source: databases.md | Chunk ID: databases-p1-c3" in context
    assert context.index("databases-p1-c2") < context.index("databases-p1-c3")


def test_format_context_keeps_each_chunks_own_source():
    other = _chunk("python-p1-c5", "Exceptions stop the program.", source="python.md")

    context = format_context([FOREIGN, other])

    assert "Source: python.md | Chunk ID: python-p1-c5\nExceptions stop the program." in context


def test_messages_contain_the_context_question_and_grounding_rules():
    system, human = build_messages("How are keys related?", [PRIMARY, FOREIGN])

    assert system.content == SYSTEM_PROMPT
    assert "only" in SYSTEM_PROMPT and INSUFFICIENT_MESSAGE in SYSTEM_PROMPT
    assert "chunk_id" in SYSTEM_PROMPT
    assert "databases-p1-c2" in human.content
    assert "Question: How are keys related?" in human.content


def test_generate_answer_with_mocked_llm_returns_answer_and_cited_sources(test_config):
    llm = FakeLLM("A foreign key points to a primary key [databases.md, databases-p1-c3].")

    answer = generate_answer("How are keys related?", [PRIMARY, FOREIGN], test_config, llm=llm)

    assert answer.sufficient
    assert "foreign key points to a primary key" in answer.text
    assert answer.sources == [("databases.md", "databases-p1-c3")]
    assert len(llm.calls) == 1
    assert "databases-p1-c2" in llm.calls[0][1].content


def test_citation_to_a_chunk_that_was_not_supplied_is_not_reported():
    answer_text = "See [databases.md, databases-p1-c3] and [databases.md, databases-p1-c99]."

    assert cited_sources(answer_text, [PRIMARY, FOREIGN]) == [("databases.md", "databases-p1-c3")]


def test_chunk_id_is_not_matched_inside_a_longer_id():
    long_chunk = _chunk("databases-p1-c30", "x")
    short_chunk = _chunk("databases-p1-c3", "y")

    assert cited_sources("[databases.md, databases-p1-c30]", [short_chunk, long_chunk]) == [
        ("databases.md", "databases-p1-c30")
    ]


def test_empty_retrieval_does_not_call_the_model(test_config):
    llm = FakeLLM(error=AssertionError("the model must not be called"))

    answer = generate_answer("What is a foreign key?", [], test_config, llm=llm)

    assert answer.text == INSUFFICIENT_MESSAGE
    assert not answer.sufficient
    assert answer.sources == []
    assert llm.calls == []


def test_empty_retrieval_does_not_need_an_api_key(test_config):
    answer = generate_answer("What is a foreign key?", [], test_config)

    assert not answer.sufficient


def test_reply_saying_context_is_insufficient_is_reported_as_insufficient(test_config):
    llm = FakeLLM(INSUFFICIENT_MESSAGE)

    answer = generate_answer("Who won the 1998 World Cup?", [PRIMARY], test_config, llm=llm)

    assert answer.text == INSUFFICIENT_MESSAGE
    assert not answer.sufficient
    assert answer.sources == []


def test_insufficient_reply_is_recognised_with_different_wording_of_the_ending(test_config):
    llm = FakeLLM("the available documents do not contain sufficient information to answer this question")

    assert not generate_answer("q", [PRIMARY], test_config, llm=llm).sufficient


def test_content_returned_as_blocks_is_joined(test_config):
    llm = FakeLLM([{"type": "text", "text": "Keys relate "}, "[databases.md, databases-p1-c2]."])

    answer = generate_answer("q", [PRIMARY], test_config, llm=llm)

    assert answer.text == "Keys relate [databases.md, databases-p1-c2]."
    assert answer.sources == [("databases.md", "databases-p1-c2")]


def test_api_failure_becomes_a_clear_error(test_config):
    llm = FakeLLM(error=ConnectionError("network down"))

    with pytest.raises(AnswerGenerationError, match="ConnectionError.*network down"):
        generate_answer("q", [PRIMARY], test_config, llm=llm)


def test_empty_model_reply_is_an_error(test_config):
    with pytest.raises(AnswerGenerationError, match="empty"):
        generate_answer("q", [PRIMARY], test_config, llm=FakeLLM("   "))


def test_missing_api_key_gives_a_clear_error():
    with pytest.raises(MissingApiKeyError, match="GEMINI_API_KEY") as error:
        get_api_key()

    assert "GOOGLE_API_KEY" in str(error.value)
    assert ".env" in str(error.value)


def test_blank_api_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")

    with pytest.raises(MissingApiKeyError):
        get_api_key()


def test_gemini_key_is_preferred_and_google_key_is_the_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    assert get_api_key() == "google-key"

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert get_api_key() == "gemini-key"


def test_generate_answer_without_a_key_fails_before_calling_anything(test_config):
    with pytest.raises(MissingApiKeyError):
        generate_answer("q", [PRIMARY], test_config)


def test_create_llm_uses_the_configured_model_and_key(test_config, monkeypatch):
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return "client"

    monkeypatch.setattr(answer_module, "ChatGoogleGenerativeAI", fake_chat)
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key")
    config = dataclasses.replace(test_config, llm_model="some-other-model")

    assert create_llm(config) == "client"
    assert captured["model"] == "some-other-model"
    assert captured["google_api_key"] == "secret-key"
    assert captured["temperature"] == 0


def test_llm_model_comes_from_rag_llm_model(monkeypatch):
    assert load_config().llm_model == "gemini-3.8-flash"

    monkeypatch.setenv("RAG_LLM_MODEL", "my-model")
    assert load_config().llm_model == "my-model"


# ---- scripts/ask.py -------------------------------------------------------


@pytest.fixture
def ask_script(monkeypatch):
    spec = importlib.util.spec_from_file_location("ask_script", PROJECT_ROOT / "scripts" / "ask.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # never read a real .env file during tests
    monkeypatch.setattr(module, "load_dotenv", lambda *args, **kwargs: None)
    return module


def test_ask_prints_answer_and_sources(ask_script, monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    llm = FakeLLM("Foreign keys reference primary keys [databases.md, databases-p1-c3].")
    monkeypatch.setattr(ask_script, "create_llm", lambda config: llm)
    seen = {}

    def fake_query_hybrid(query, config, k=None, use_graph=False):
        seen.update(query=query, k=k, use_graph=use_graph)
        return [(PRIMARY, 0.03), (FOREIGN, 0.02)]

    monkeypatch.setattr(ask_script, "query_hybrid", fake_query_hybrid)

    code = ask_script.main(["How are primary keys and foreign keys related?", "--k", "2"])

    out = capsys.readouterr().out
    assert code == 0
    assert seen == {"query": "How are primary keys and foreign keys related?", "k": 2, "use_graph": True}
    assert "Retrieved chunks: databases-p1-c2 (databases.md), databases-p1-c3 (databases.md)" in out
    assert "Foreign keys reference primary keys" in out
    assert "- databases.md (chunk databases-p1-c3)" in out


def test_ask_reports_insufficient_context_without_sources(ask_script, monkeypatch, capsys):
    monkeypatch.setattr(ask_script, "create_llm", lambda config: FakeLLM(INSUFFICIENT_MESSAGE))
    monkeypatch.setattr(ask_script, "query_hybrid", lambda *a, **k: [(PRIMARY, 0.03)])

    code = ask_script.main(["Who won the 1998 World Cup?"])

    out = capsys.readouterr().out
    assert code == 0
    assert INSUFFICIENT_MESSAGE in out
    assert "Sources cited" not in out


def test_ask_with_nothing_retrieved_says_so_without_calling_the_model(ask_script, monkeypatch, capsys):
    llm = FakeLLM(error=AssertionError("the model must not be called"))
    monkeypatch.setattr(ask_script, "create_llm", lambda config: llm)
    monkeypatch.setattr(ask_script, "query_hybrid", lambda *a, **k: [])

    code = ask_script.main(["anything"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Retrieved chunks: none" in out
    assert INSUFFICIENT_MESSAGE in out


def test_ask_without_an_api_key_exits_with_a_clear_message(ask_script, monkeypatch, capsys):
    def retrieval_must_not_run(*args, **kwargs):
        raise AssertionError("retrieval should not run without an API key")

    monkeypatch.setattr(ask_script, "query_hybrid", retrieval_must_not_run)

    code = ask_script.main(["How are primary keys and foreign keys related?"])

    captured = capsys.readouterr()
    assert code == 1
    assert "GEMINI_API_KEY" in captured.err
    assert captured.out == ""


def test_ask_reports_api_errors_without_a_traceback(ask_script, monkeypatch, capsys):
    monkeypatch.setattr(ask_script, "create_llm", lambda config: FakeLLM(error=TimeoutError("slow")))
    monkeypatch.setattr(ask_script, "query_hybrid", lambda *a, **k: [(PRIMARY, 0.03)])

    code = ask_script.main(["q"])

    captured = capsys.readouterr()
    assert code == 1
    assert "Error: The Gemini request failed (TimeoutError): slow" in captured.err


def test_ask_reports_a_missing_index_clearly(ask_script, monkeypatch, capsys):
    monkeypatch.setattr(ask_script, "create_llm", lambda config: FakeLLM("unused"))

    def missing_index(*args, **kwargs):
        raise FileNotFoundError("No dense index at x. Run scripts/ingest.py first.")

    monkeypatch.setattr(ask_script, "query_hybrid", missing_index)

    code = ask_script.main(["q"])

    assert code == 1
    assert "ingest.py" in capsys.readouterr().err
