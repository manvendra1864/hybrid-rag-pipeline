import dataclasses
import importlib.util
import sys
from types import SimpleNamespace

import groq
import httpx
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
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


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

    with pytest.raises(AnswerGenerationError, match="ConnectionError") as error:
        generate_answer("q", [PRIMARY], test_config, llm=llm)

    assert "network down" not in str(error.value)


SLOW = "Groq took too long to answer. Please try again shortly."


REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _status_error(code, message="long raw message " * 30):
    """The error the Groq client raises for an HTTP error response."""
    return groq.APIStatusError(message, response=httpx.Response(code, request=REQUEST), body=None)


def _api_error(code, status, message="long raw message " * 30):
    return _status_error(code, message)


def _server_error(code, status):
    return _status_error(code, "x " * 200)


@pytest.mark.parametrize(
    "error, expected",
    [
        (_server_error(503, "UNAVAILABLE"), "temporarily unavailable (503). Please try again in a few moments"),
        (_api_error(429, "RESOURCE_EXHAUSTED"), "too many requests (429)"),
        (_api_error(401, "UNAUTHENTICATED"), "rejected the API key"),
        (_api_error(403, "PERMISSION_DENIED"), "rejected the API key"),
        (_api_error(404, "NOT_FOUND"), "RAG_LLM_MODEL"),
        (_server_error(500, "INTERNAL"), "failed (HTTP 500)"),
        (_server_error(504, "DEADLINE_EXCEEDED"), SLOW),
        (_api_error(504, "DEADLINE_EXCEEDED"), SLOW),
        (httpx.ReadTimeout("read timed out"), SLOW),
        (groq.APITimeoutError(request=REQUEST), SLOW),
        (TimeoutError("slow"), SLOW),
        (RuntimeError("something odd"), "failed (RuntimeError)"),
    ],
)
def test_request_errors_become_short_messages(test_config, error, expected):
    with pytest.raises(AnswerGenerationError) as raised:
        generate_answer("q", [PRIMARY], test_config, llm=FakeLLM(error=error))

    message = str(raised.value)
    assert expected in message
    assert len(message) < 120
    assert "raw message" not in message and "x x x" not in message
    assert raised.value.__cause__ is error


def _wrapped(inner):
    # the client libraries can re-raise an HTTP error inside their own error type
    try:
        try:
            raise inner
        except Exception as caught:
            raise RuntimeError("wrapped by the client") from caught
    except RuntimeError as wrapped:
        return wrapped


@pytest.mark.parametrize(
    "inner, expected",
    [
        (_server_error(503, "UNAVAILABLE"), "temporarily unavailable"),
        (_server_error(504, "DEADLINE_EXCEEDED"), SLOW),
        (httpx.ReadTimeout("read timed out"), SLOW),
    ],
)
def test_wrapped_errors_are_still_recognised(test_config, inner, expected):
    with pytest.raises(AnswerGenerationError) as raised:
        generate_answer("q", [PRIMARY], test_config, llm=FakeLLM(error=_wrapped(inner)))

    assert expected in str(raised.value)


def test_a_groq_timeout_is_recognised_without_a_status_code():
    assert answer_module.describe_request_error(groq.APITimeoutError(request=REQUEST)) == SLOW


DAILY_QUOTA_TEXT = (
    "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` service tier "
    "`on_demand` on tokens per day (TPD): Limit 200000, Used 199900, Requested 500. "
    "Please try again in 10m. Need more tokens? Upgrade to Dev Tier today."
)
DAILY_QUOTA_MESSAGE = (
    "The daily Groq request limit for this API key is used up (429). "
    "Try again later, or use a key with a higher limit."
)


@pytest.mark.parametrize(
    "text",
    [DAILY_QUOTA_TEXT, "Rate limit reached ... on requests per day (RPD): Limit 1000, Used 1000."],
)
def test_a_daily_quota_error_says_so_instead_of_asking_for_a_short_wait(test_config, text):
    error = _status_error(429, text)

    with pytest.raises(AnswerGenerationError) as raised:
        generate_answer("q", [PRIMARY], test_config, llm=FakeLLM(error=error))

    assert str(raised.value) == DAILY_QUOTA_MESSAGE


def test_a_wrapped_daily_quota_error_is_still_recognised(test_config):
    error = _wrapped(_status_error(429, DAILY_QUOTA_TEXT))

    with pytest.raises(AnswerGenerationError) as raised:
        generate_answer("q", [PRIMARY], test_config, llm=FakeLLM(error=error))

    assert str(raised.value) == DAILY_QUOTA_MESSAGE


def test_a_per_minute_rate_limit_still_asks_for_a_short_wait(test_config):
    text = "Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute (TPM): Limit 8000. Please try again in 5s."

    with pytest.raises(AnswerGenerationError, match="too many requests .429.. Please wait a little"):
        generate_answer("q", [PRIMARY], test_config, llm=FakeLLM(error=_status_error(429, text)))


def test_a_504_message_does_not_show_raw_details(test_config):
    error = _server_error(504, "DEADLINE_EXCEEDED")

    with pytest.raises(AnswerGenerationError) as raised:
        generate_answer("q", [PRIMARY], test_config, llm=FakeLLM(error=error))

    assert str(raised.value) == SLOW


def test_the_api_key_is_not_in_the_short_message(test_config, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-key-123")
    error = _status_error(503, "service unavailable for key secret-key-123")

    with pytest.raises(AnswerGenerationError) as raised:
        generate_answer("q", [PRIMARY], test_config, llm=FakeLLM(error=error))

    assert "secret-key-123" not in str(raised.value)


def test_empty_model_reply_is_an_error(test_config):
    with pytest.raises(AnswerGenerationError, match="empty"):
        generate_answer("q", [PRIMARY], test_config, llm=FakeLLM("   "))


def test_missing_api_key_gives_a_clear_error():
    with pytest.raises(MissingApiKeyError, match="GROQ_API_KEY") as error:
        get_api_key()

    assert ".env" in str(error.value)


def test_blank_api_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "   ")

    with pytest.raises(MissingApiKeyError):
        get_api_key()


def test_the_groq_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "  groq-key  ")

    assert get_api_key() == "groq-key"


def test_an_old_gemini_key_is_not_used_for_groq(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    with pytest.raises(MissingApiKeyError):
        get_api_key()


def test_generate_answer_without_a_key_fails_before_calling_anything(test_config):
    with pytest.raises(MissingApiKeyError):
        generate_answer("q", [PRIMARY], test_config)


def test_create_llm_uses_the_configured_model_and_key(test_config, monkeypatch):
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return "client"

    monkeypatch.setattr(answer_module, "ChatGroq", fake_chat)
    monkeypatch.setenv("GROQ_API_KEY", "secret-key")
    config = dataclasses.replace(test_config, llm_model="some-other-model")

    assert create_llm(config) == "client"
    assert captured["model"] == "some-other-model"
    assert captured["api_key"] == "secret-key"
    assert captured["temperature"] == 0


def test_create_llm_sets_a_request_timeout_and_keeps_the_retries(test_config, monkeypatch):
    captured = {}
    monkeypatch.setattr(answer_module, "ChatGroq", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setenv("GROQ_API_KEY", "secret-key")

    create_llm(test_config)

    assert captured["timeout"] == 60
    assert captured["max_retries"] == answer_module.MAX_RETRIES == 1  # one retry: two attempts in all


def test_the_real_groq_client_is_built_from_the_config_without_a_network_call(test_config, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-key-xyz")

    llm = create_llm(test_config)

    assert llm.model_name == "openai/gpt-oss-120b"
    assert llm.request_timeout == 60
    assert llm.max_retries == 1
    assert llm.temperature == pytest.approx(0, abs=1e-6)  # the client turns 0 into a tiny value
    assert "secret-key-xyz" not in repr(llm) and "secret-key-xyz" not in str(llm.groq_api_key)


def test_the_timeout_comes_from_the_config(test_config, monkeypatch):
    captured = {}
    monkeypatch.setattr(answer_module, "ChatGroq", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setenv("GROQ_API_KEY", "secret-key")

    create_llm(dataclasses.replace(test_config, llm_timeout_seconds=90))

    assert captured["timeout"] == 90


def test_llm_timeout_can_be_set_with_rag_llm_timeout(monkeypatch):
    assert load_config().llm_timeout_seconds == 60

    monkeypatch.setenv("RAG_LLM_TIMEOUT", "120")
    assert load_config().llm_timeout_seconds == 120


def test_llm_model_comes_from_rag_llm_model(monkeypatch):
    assert load_config().llm_model == "openai/gpt-oss-120b"

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
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
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
    assert "GROQ_API_KEY" in captured.err
    assert captured.out == ""


def test_ask_reports_api_errors_without_a_traceback(ask_script, monkeypatch, capsys):
    monkeypatch.setattr(ask_script, "create_llm", lambda config: FakeLLM(error=TimeoutError("slow")))
    monkeypatch.setattr(ask_script, "query_hybrid", lambda *a, **k: [(PRIMARY, 0.03)])

    code = ask_script.main(["q"])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.strip() == f"Error: {SLOW}"


def test_ask_prints_an_answer_even_if_the_console_cannot_encode_it(ask_script, monkeypatch):
    import io

    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    reply = "Keys link tables‑and rows [databases.md, databases-p1-c3]."
    monkeypatch.setattr(ask_script, "create_llm", lambda config: FakeLLM(reply))
    monkeypatch.setattr(ask_script, "query_hybrid", lambda *a, **k: [(PRIMARY, 0.03), (FOREIGN, 0.02)])

    code = ask_script.main(["q"])
    sys.stdout.flush()

    assert code == 0
    assert "Keys link tables" in raw.getvalue().decode("cp1252")


def test_ask_reports_a_daily_quota_error_with_the_same_message_as_the_app(ask_script, monkeypatch, capsys):
    error = _status_error(429, DAILY_QUOTA_TEXT)
    monkeypatch.setattr(ask_script, "create_llm", lambda config: FakeLLM(error=error))
    monkeypatch.setattr(ask_script, "query_hybrid", lambda *a, **k: [(PRIMARY, 0.03)])

    code = ask_script.main(["q"])

    assert code == 1
    assert capsys.readouterr().err.strip() == f"Error: {DAILY_QUOTA_MESSAGE}"


def test_ask_reports_a_504_with_the_same_short_message_as_the_app(ask_script, monkeypatch, capsys):
    monkeypatch.setattr(ask_script, "create_llm", lambda config: FakeLLM(error=_server_error(504, "DEADLINE_EXCEEDED")))
    monkeypatch.setattr(ask_script, "query_hybrid", lambda *a, **k: [(PRIMARY, 0.03)])

    code = ask_script.main(["q"])

    assert code == 1
    assert capsys.readouterr().err.strip() == f"Error: {SLOW}"


def test_ask_reports_a_503_with_the_same_short_message_as_the_app(ask_script, monkeypatch, capsys):
    monkeypatch.setattr(ask_script, "create_llm", lambda config: FakeLLM(error=_server_error(503, "UNAVAILABLE")))
    monkeypatch.setattr(ask_script, "query_hybrid", lambda *a, **k: [(PRIMARY, 0.03)])

    code = ask_script.main(["q"])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.strip() == "Error: Groq is temporarily unavailable (503). Please try again in a few moments."


def test_ask_reports_a_missing_index_clearly(ask_script, monkeypatch, capsys):
    monkeypatch.setattr(ask_script, "create_llm", lambda config: FakeLLM("unused"))

    def missing_index(*args, **kwargs):
        raise FileNotFoundError("No dense index at x. Run scripts/ingest.py first.")

    monkeypatch.setattr(ask_script, "query_hybrid", missing_index)

    code = ask_script.main(["q"])

    assert code == 1
    assert "ingest.py" in capsys.readouterr().err
