"""Pull sections and concept names out of Markdown chunks.

The corpus is structured study notes, so every "## Heading" is treated as
a concept, and the text under it is what that concept says about other
concepts. This is rule-based on purpose: it needs no model and gives the
same result every run.
"""

import re

from langchain_core.documents import Document

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ALIAS_SPLIT_RE = re.compile(r",\s*(?:and\s+)?|\s+and\s+")


def _stem(token: str) -> str:
    """Crude plural stripping so 'keys' matches 'key' and 'indexes' matches 'index'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and token[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower())]


def concept_aliases(heading: str) -> list[str]:
    """Names a concept can be referred to by.

    'Tables, Rows, and Columns' -> ['tables', 'rows', 'columns'].
    """
    parts = _ALIAS_SPLIT_RE.split(heading.lower())
    return [p.strip() for p in parts if p.strip()]


def mentions_alias(tokens: list[str], alias: str) -> bool:
    """True if the alias appears as a run of consecutive tokens."""
    alias_tokens = tokenize(alias)
    n = len(alias_tokens)
    return any(tokens[i : i + n] == alias_tokens for i in range(len(tokens) - n + 1))


def split_sections(chunks: list[Document]) -> list[tuple[str, str | None, str, str]]:
    """Split chunks at '## ' headings.

    Returns (source, section, chunk_id, text) tuples. A chunk that holds
    the end of one section and the start of the next gives two tuples, and
    a chunk with no heading of its own continues the section the previous
    chunk of that file ended in. Text before the first heading has
    section None. Heading lines are not part of the text.
    """
    segments = []
    current_section: dict[str, str | None] = {}

    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk_id = chunk.metadata["chunk_id"]
        section = current_section.get(source)
        pieces: list[list] = [[section, []]]

        for line in chunk.page_content.splitlines():
            match = _HEADING_RE.match(line)
            if match:
                section = match.group(1)
                pieces.append([section, []])
            elif not line.startswith("# "):
                pieces[-1][1].append(line)

        current_section[source] = section

        for index, (name, lines) in enumerate(pieces):
            text = "\n".join(lines).strip()
            if index == 0 and not text:
                continue
            segments.append((source, name, chunk_id, text))

    return segments
