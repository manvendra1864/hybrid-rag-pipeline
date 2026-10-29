"""Streamlit chat UI for NexusRAG.

Run from the project root:
    streamlit run app.py
"""

import html
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


def drop_stale_project_modules() -> None:
    """Make a redeploy take effect without restarting the server.

    Streamlit runs app.py from disk on every run but keeps imported modules in memory, so
    after new code is deployed the new app.py can meet the old modules and fail with
    "cannot import name ...". The project's modules are dropped, and so imported again from
    disk, if no earlier run has loaded them yet or if a source file changed since then.
    """
    loaded_at = getattr(sys, "_nexusrag_loaded_at", None)
    project = [name for name in sys.modules if name in ("config", "src") or name.startswith("src.")]

    def changed(name: str) -> bool:
        source = getattr(sys.modules[name], "__file__", None)
        try:
            return bool(source) and Path(source).stat().st_mtime > loaded_at
        except OSError:
            return True

    if loaded_at is None or any(changed(name) for name in project):
        for name in project:
            sys.modules.pop(name, None)
    sys._nexusrag_loaded_at = time.time()


drop_stale_project_modules()

from config import PROJECT_ROOT, load_config
from src.rag.dense_retrieval import get_embeddings
from src.rag.knowledge_graph import load_graph
from src.rag.ui_helpers import (
    EVALUATION_RESULTS,
    METHODS,
    SCORE_NOTES,
    EvaluationResultsError,
    api_key_available,
    citation_chips_html,
    ensure_index,
    graph_stats,
    load_evaluation_summary,
    missing_files,
    run_question,
    validate_question,
)

EXAMPLES = [
    "How are primary keys and foreign keys related?",
    "What does DNS do when I open a website?",
    "How can a program keep running smoothly even if a piece of user input turns out to be invalid?",
]

CSS = """
.block-container { max-width: 860px; padding-top: 2.4rem; padding-bottom: 6rem; }
h1 { font-weight: 600; letter-spacing: -0.02em; font-size: 1.7rem !important; padding-bottom: 0; }
.app-sub { color: #8b929c; font-size: 0.93rem; margin: -0.35rem 0 1.4rem 0; }
.label { text-transform: uppercase; letter-spacing: 0.09em; font-size: 0.68rem; color: #7d8590; margin: 1rem 0 0.35rem 0; }
.chip { display: inline-block; padding: 2px 10px; margin: 2px 6px 2px 0; border: 1px solid #2a303b; border-radius: 999px;
        background: #121620; font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: 0.78rem; color: #c3c8d0; }
.chip i { font-style: normal; color: #555d69; margin: 0 6px; }
.chip b { color: #d4a15a; font-weight: 500; }
.chunk { border-left: 2px solid #2a303b; padding: 0.15rem 0 0.15rem 0.9rem; margin: 0.8rem 0; }
.chunk-meta { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: 0.78rem; color: #8b929c; }
.chunk-meta b { color: #d4a15a; font-weight: 500; }
.chunk-text { white-space: pre-wrap; font-size: 0.87rem; line-height: 1.5; color: #c3c8d0; max-height: 10.5rem; overflow: auto; margin-top: 0.3rem; }
.concepts { color: #8b929c; font-size: 0.85rem; margin-bottom: 0.4rem; }
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 500; }
[data-testid="stMetricLabel"] p { font-size: 0.74rem; color: #8b929c; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stExpander"] { border-color: #232833; }
.status { font-size: 0.85rem; color: #8b929c; }
.status b { color: #e4e5e8; font-weight: 500; }
"""

st.set_page_config(page_title="NexusRAG", page_icon=":material/menu_book:", layout="centered")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

load_dotenv(PROJECT_ROOT / ".env")
config = load_config()


@st.cache_resource(show_spinner="Loading the embedding model...")
def warm_up(model_name: str, _config):
    return get_embeddings(_config)


@st.cache_resource
def read_graph(path: str, modified: float):
    # `modified` is part of the cache key, so re-running ingestion refreshes the graph.
    return load_graph(Path(path))


@st.cache_resource(show_spinner="Building the search index. This only happens on the first start and can take a few minutes...")
def prepare_index(_config):
    # Runs once per server process; other sessions wait for it. The result, including a
    # failure message, is cached so reruns do not start another build. "Try again" clears it.
    return ensure_index(_config)


def chunk_html(record: dict) -> str:
    # Newlines become &#10;: a real blank line would end the HTML block in st.markdown.
    meta = (
        f"#{record['rank']} &nbsp;{html.escape(record['source'])} &middot; "
        f"<b>{html.escape(record['chunk_id'])}</b> &middot; score {record['score']:.4f}"
    )
    if record["note"]:
        meta += f" &middot; {html.escape(record['note'])}"
    return (
        f'<div class="chunk"><div class="chunk-meta">{meta}</div>'
        f'<div class="chunk-text">{html.escape(record["text"]).replace(chr(10), "&#10;")}</div></div>'
    )


def render_details(turn: dict) -> None:
    """The expandable sections under an answer."""
    if not turn["comparison"]:
        return

    with st.expander(f"Retrieved chunks ({len(turn['retrieved'])})"):
        st.markdown(
            '<div class="concepts">Concepts matched in the graph: '
            + (html.escape(", ".join(turn["concepts"])) if turn["concepts"] else "none")
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("".join(chunk_html(r) for r in turn["retrieved"]), unsafe_allow_html=True)

    with st.expander("Compare retrieval methods"):
        for tab, method in zip(st.tabs(METHODS), METHODS):
            with tab:
                st.caption(SCORE_NOTES[method])
                rows = turn["comparison"][method]
                if not rows:
                    st.write("No results.")
                    continue
                st.dataframe(
                    [
                        {
                            "Rank": r["rank"],
                            "Source": r["source"],
                            "Chunk ID": r["chunk_id"],
                            "Score": round(r["score"], 4),
                            **({"vs hybrid": r["note"]} if method == "Graph-enhanced hybrid" else {}),
                            "Text": r["text"][:110].replace("\n", " "),
                        }
                        for r in rows
                    ],
                    hide_index=True,
                    width="stretch",
                )


def render_evaluation() -> None:
    """Saved results from scripts/evaluate_retrieval.py, shown as they were written."""
    with st.expander("Retrieval evaluation", expanded=False):
        try:
            summary = load_evaluation_summary(EVALUATION_RESULTS)
        except EvaluationResultsError as error:
            st.caption(str(error))
            return

        k = summary["top_k"]
        recall = f"Recall@{k}" if k else "Recall"
        st.caption(
            f"{summary['questions'] or 'Several'} hand-labelled questions, top {k or 'few'} results per method. "
            "It is a small test set, so small gaps between methods are not meaningful."
        )
        st.dataframe(
            [dict(zip(summary["columns"], row)) for row in summary["rows"]],
            hide_index=True,
            width="stretch",
        )
        st.markdown(
            f'<div class="concepts"><b>{html.escape(recall)}</b>: the share of the relevant chunks for a question '
            "that appear in the top results. <b>MRR</b>: the average of 1 / rank of the first relevant chunk "
            "(0 if there is none). For both, higher is better.</div>",
            unsafe_allow_html=True,
        )


def render_turn(turn: dict) -> None:
    with st.chat_message("assistant", avatar=":material/menu_book:"):
        notice = turn["notice"]
        if notice:
            (st.warning if notice["level"] == "warning" else st.error)(notice["text"])
        if turn["answer"] is not None:
            st.markdown(turn["answer"])
            if turn["sources"]:
                st.markdown('<div class="label">Sources</div>' + citation_chips_html(turn["sources"]), unsafe_allow_html=True)
            elif turn["sufficient"] is False:
                st.caption("No sources: the retrieved chunks did not have enough information.")
        render_details(turn)


def ask(question: str, k: int) -> None:
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)
    with st.spinner("Searching the notes and writing the answer..."):
        turn = run_question(question, config, k)
    render_turn(turn)
    st.session_state.turns.append(turn)


# ---- header and search index -----------------------------------------------
st.title("NexusRAG")
st.markdown(
    '<div class="app-sub">A hybrid RAG system combining dense retrieval, BM25, and Knowledge Graph-enhanced retrieval to generate context-aware answers from documents.</div>',
    unsafe_allow_html=True,
)
index_error = prepare_index(config)

# ---- sidebar ---------------------------------------------------------------
missing = missing_files(config)

with st.sidebar:
    st.markdown("### Knowledge graph")
    if index_error or "knowledge graph" in missing:
        st.caption("Not built yet.")
    else:
        stats = graph_stats(read_graph(str(config.graph_file), config.graph_file.stat().st_mtime))
        left, right = st.columns(2)
        for i, (label, value) in enumerate(stats.items()):
            (left if i % 2 == 0 else right).metric(label, value)

    st.markdown("### Settings")
    k = st.slider("Chunks to retrieve", 1, 6, min(max(config.hybrid_top_k, 1), 6))
    key_state = "found" if api_key_available() else "missing"
    st.markdown(
        f'<div class="status">Model <b>{html.escape(config.llm_model)}</b><br>API key <b>{key_state}</b></div>',
        unsafe_allow_html=True,
    )
    if st.button("Clear conversation", width="stretch"):
        st.session_state.turns = []
        st.rerun()

# ---- main ------------------------------------------------------------------
render_evaluation()

if index_error:
    st.error(index_error)
    if st.button("Try again"):
        prepare_index.clear()
        st.rerun()
    st.stop()

warm_up(config.embedding_model, config)

if "turns" not in st.session_state:
    st.session_state.turns = []
if "pending" not in st.session_state:
    st.session_state.pending = None

typed = st.chat_input("Ask a question about the notes")
question = st.session_state.pending or typed
st.session_state.pending = None

for past in st.session_state.turns:
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(past["question"])
    render_turn(past)

if not st.session_state.turns and question is None:
    st.markdown('<div class="label">Try one</div>', unsafe_allow_html=True)
    for example in EXAMPLES:
        if st.button(example, key=f"ex-{example}", width="stretch"):
            st.session_state.pending = example
            st.rerun()

if question is not None:
    cleaned = validate_question(question)
    if cleaned is None:
        st.warning("Type a question first.")
    else:
        ask(cleaned, k)
