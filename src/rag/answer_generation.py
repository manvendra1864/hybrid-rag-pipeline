"""Generate an answer from retrieved chunks with Groq, using only those chunks.

The chunks are formatted into a context block that keeps each chunk's source
filename and chunk id, and the model is told to answer only from that context
and to cite the passages it used. The citations the model writes are then
checked against the chunks that were actually supplied.
"""

import os
import re
from dataclasses import dataclass

import httpx
from groq import APITimeoutError
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import Config

INSUFFICIENT_MESSAGE = (
    "The available documents do not contain sufficient information to answer this question."
)

SYSTEM_PROMPT = f"""You answer questions using only the context passages supplied by the user.

Rules:
- Use only facts stated in the context. Do not use outside knowledge and do not guess.
- If the context does not contain enough information to answer the question, reply with exactly: {INSUFFICIENT_MESSAGE}
- After each statement, cite the passage it came from as [source_file, chunk_id], copying both values exactly from that passage's header. Never cite a passage that is not in the context.
- Keep the answer short and direct."""

_API_KEY_VARIABLES = ("GROQ_API_KEY",)

# Retries after the first attempt, so a failing request is tried twice in total.
MAX_RETRIES = 1


class MissingApiKeyError(RuntimeError):
    pass


class AnswerGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    sufficient: bool
    # (source filename, chunk_id) of the supplied chunks that the answer cites
    sources: list[tuple[str, str]]


def get_api_key() -> str:
    for name in _API_KEY_VARIABLES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise MissingApiKeyError(
        "No Groq API key found. Set GROQ_API_KEY in your environment or in a "
        ".env file. A key can be created at https://console.groq.com/keys"
    )


def create_llm(config: Config) -> ChatGroq:
    return ChatGroq(
        model=config.llm_model,
        api_key=get_api_key(),
        temperature=0,
        timeout=config.llm_timeout_seconds,
        max_retries=MAX_RETRIES,
    )


def describe_request_error(error: BaseException) -> str:
    """A short message for a failed Groq request, without the raw error text.

    The raw text can be long and is not useful to the person asking, so only
    the kind of failure is reported.
    """
    chain = []
    current = error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__

    code = next((e.status_code for e in chain if isinstance(getattr(e, "status_code", None), int)), None)
    timed_out = code in (504, 408) or any(
        isinstance(e, (TimeoutError, httpx.TimeoutException, APITimeoutError)) for e in chain
    )
    if timed_out:
        # Our own client timeout and a 504 from the service mean the same thing.
        return "Groq took too long to answer. Please try again shortly."
    if code == 503:
        return "Groq is temporarily unavailable (503). Please try again in a few moments."
    if code == 429:
        if any(re.search(r"per ?day|\((?:RPD|TPD)\)", str(e), flags=re.IGNORECASE) for e in chain):
            return (
                "The daily Groq request limit for this API key is used up (429). "
                "Try again later, or use a key with a higher limit."
            )
        return "Groq is receiving too many requests (429). Please wait a little and try again."
    if code in (401, 403):
        return "Groq rejected the API key. Check that the key is valid and allowed to use this model."
    if code == 404:
        return "Groq could not find the configured model. Check RAG_LLM_MODEL."
    detail = f"HTTP {code}" if code else type(error).__name__
    return f"The Groq request failed ({detail}). Please try again."


def format_context(chunks: list[Document]) -> str:
    """One block per chunk, headed by its source filename and chunk id."""
    passages = [
        f"Source: {chunk.metadata['source']} | Chunk ID: {chunk.metadata['chunk_id']}\n"
        f"{chunk.page_content.strip()}"
        for chunk in chunks
    ]
    return "\n\n---\n\n".join(passages)


def build_messages(query: str, chunks: list[Document]) -> list:
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n\n{format_context(chunks)}\n\nQuestion: {query}"),
    ]


def _response_text(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    # Some models return a list of content blocks instead of one string.
    parts = [block if isinstance(block, str) else block.get("text", "") for block in content]
    return "".join(parts)


def cited_sources(answer: str, chunks: list[Document]) -> list[tuple[str, str]]:
    """Supplied chunks whose chunk_id appears in the answer, in supplied order.

    A citation the model invents, one that matches none of the supplied
    chunks, is not returned.
    """
    cited = []
    for chunk in chunks:
        chunk_id = chunk.metadata["chunk_id"]
        if re.search(rf"(?<![\w-]){re.escape(chunk_id)}(?![\w-])", answer):
            cited.append((chunk.metadata["source"], chunk_id))
    return cited


def generate_answer(
    query: str, chunks: list[Document], config: Config, llm=None
) -> GroundedAnswer:
    """Answer the query from the chunks only.

    With no chunks the model is not called and the answer says the documents
    are insufficient. A reply that says so is returned as insufficient, with
    no sources.
    """
    if not chunks:
        return GroundedAnswer(INSUFFICIENT_MESSAGE, sufficient=False, sources=[])

    llm = llm or create_llm(config)
    try:
        response = llm.invoke(build_messages(query, chunks))
    except Exception as error:
        raise AnswerGenerationError(describe_request_error(error)) from error

    text = _response_text(response).strip()
    if not text:
        raise AnswerGenerationError("The model returned an empty response.")

    if INSUFFICIENT_MESSAGE.rstrip(".").lower() in text.lower():
        return GroundedAnswer(INSUFFICIENT_MESSAGE, sufficient=False, sources=[])

    return GroundedAnswer(text, sufficient=True, sources=cited_sources(text, chunks))
