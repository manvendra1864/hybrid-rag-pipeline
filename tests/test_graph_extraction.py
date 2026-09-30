from langchain_core.documents import Document

from src.rag.graph_extraction import (
    concept_aliases,
    mentions_alias,
    split_sections,
    tokenize,
)


def _chunk(chunk_id, text, source="notes.md"):
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "source": source, "page": 1},
    )


def test_heading_with_and_is_split_into_aliases():
    assert concept_aliases("Tables, Rows, and Columns") == ["tables", "rows", "columns"]
    assert concept_aliases("TCP and UDP") == ["tcp", "udp"]
    assert concept_aliases("Primary Keys") == ["primary keys"]


def test_alias_matches_plural_and_singular_forms():
    assert mentions_alias(tokenize("Each table needs a primary key."), "primary keys")
    assert mentions_alias(tokenize("Create an index on the column."), "indexes")
    assert not mentions_alias(tokenize("A key without the primary part."), "primary keys")


def test_split_sections_attributes_text_to_the_right_section():
    chunk = _chunk(
        "notes-p1-c0",
        "# Notes\n\nIntro text.\n\n## Alpha\n\nAlpha body.\n\n## Beta\n\nBeta body.",
    )
    segments = split_sections([chunk])

    assert segments == [
        ("notes.md", None, "notes-p1-c0", "Intro text."),
        ("notes.md", "Alpha", "notes-p1-c0", "Alpha body."),
        ("notes.md", "Beta", "notes-p1-c0", "Beta body."),
    ]


def test_chunk_without_heading_continues_previous_section():
    first = _chunk("notes-p1-c0", "## Alpha\n\nStart of alpha.")
    second = _chunk("notes-p1-c1", "More of alpha.")
    segments = split_sections([first, second])

    assert segments[-1] == ("notes.md", "Alpha", "notes-p1-c1", "More of alpha.")


def test_sections_do_not_carry_across_files():
    a = _chunk("a-p1-c0", "## Alpha\n\nText.", source="a.md")
    b = _chunk("b-p1-c0", "Plain intro with no heading.", source="b.md")
    segments = split_sections([a, b])

    assert segments[-1] == ("b.md", None, "b-p1-c0", "Plain intro with no heading.")
