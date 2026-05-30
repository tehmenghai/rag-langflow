"""Streamlit app — Classic RAG vs Agentic RAG comparison.

Backend modes
─────────────
• Local LangChain  – ingest.py + query.py (Ollama + local ChromaDB, no Langflow needed)
• Langflow         – Langflow Desktop API; LangChain processes the returned JSON
"""
from __future__ import annotations

import concurrent.futures
import os
import tempfile

import streamlit as st

import config
from ingest import ingest, list_sources, delete_by_source, get_all_chunks
from json_flow_runner import FlowSettings, parse_langflow_json, ask_from_flow
from langflow_client import ClassicRAGTrace, AgenticRAGTrace, LangflowClient, RetrievedChunk
from query import ask, ask_agentic

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Playground", page_icon="🔍", layout="wide")

# ── Session state defaults ────────────────────────────────────────────────────

_DEFAULTS: dict = {
    "backend_mode": "Local LangChain",
    "connection_ok": False,
    "flows": [],
    "classic_flow_id": "",
    "agentic_flow_id": "",
    "classic_trace": None,
    "agentic_trace": None,
    "local_trace": None,
    "local_agentic_trace": None,
    "json_trace": None,
    "json_flow_settings": None,
    "langflow_url": config.LANGFLOW_BASE_URL,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _local_ask(question: str) -> ClassicRAGTrace:
    """Run local LangChain RAG and return a ClassicRAGTrace."""
    import time as _time
    t0 = _time.time()
    try:
        result = ask(question)
        chunks = [
            RetrievedChunk(content=src["content"], source=src["source"])
            for src in result["sources"]
        ]
        return ClassicRAGTrace(
            query=question,
            retrieved_chunks=chunks,
            model_used=config.LLM_MODEL,
            answer=result["answer"],
            elapsed_ms=(_time.time() - t0) * 1000,
        )
    except Exception as e:
        return ClassicRAGTrace(query=question, error=str(e))


def _local_ask_agentic(question: str) -> AgenticRAGTrace:
    """Run local agentic RAG with loop + rerank and return an AgenticRAGTrace."""
    try:
        return ask_agentic(question)
    except Exception as e:
        return AgenticRAGTrace(query=question, error=str(e))


def _json_ask(question: str, settings: FlowSettings) -> ClassicRAGTrace:
    """Run RAG using settings parsed from a Langflow JSON and return a ClassicRAGTrace."""
    try:
        result = ask_from_flow(question, settings)
        chunks = [
            RetrievedChunk(content=src["content"], source=src["source"])
            for src in result["sources"]
        ]
        return ClassicRAGTrace(
            query=question,
            retrieved_chunks=chunks,
            model_used=result["model_used"],
            prompt_template=result["prompt_template"],
            answer=result["answer"],
        )
    except Exception as e:
        return ClassicRAGTrace(query=question, error=str(e))


# ── Panel render functions ────────────────────────────────────────────────────

def render_classic_panel(trace: ClassicRAGTrace | None, title: str = "⚙️ Classic RAG") -> None:
    st.subheader(title)

    if trace is None:
        st.info("Run a query to see the pipeline trace here.")
        return

    if trace.error:
        st.error(f"**Error:** {trace.error}")
        return

    with st.expander("**1 — Query**", expanded=True):
        st.write(trace.query)

    n = len(trace.retrieved_chunks)
    with st.expander(f"**2 — Retrieve** ({n} chunk{'s' if n != 1 else ''})", expanded=True):
        if not trace.retrieved_chunks:
            st.caption("No chunks returned.")
        for i, chunk in enumerate(trace.retrieved_chunks, 1):
            st.markdown(f"**[{i}] {chunk.source}**")
            if chunk.score is not None:
                st.caption(f"Similarity score: {chunk.score:.4f}")
            st.text(chunk.content[:400])
            if i < len(trace.retrieved_chunks):
                st.divider()

    with st.expander("**3 — Generate**", expanded=False):
        if trace.model_used:
            st.caption(f"Model: `{trace.model_used}`")
        if trace.prompt_template:
            st.code(trace.prompt_template, language="text")
        else:
            st.caption("Prompt template not exposed by this pipeline.")

    with st.expander("**4 — Answer**", expanded=True):
        st.markdown(trace.answer if trace.answer else "_No answer returned._")


def render_agentic_panel(trace: AgenticRAGTrace | None) -> None:
    st.subheader("🤖 Agentic RAG")

    if trace is None:
        st.info("Run a query to see agent reasoning here.")
        st.caption("_Select an Agentic RAG flow in the sidebar to enable this panel._")
        return

    if trace.error:
        st.error(f"**Error:** {trace.error}")
        return

    # ── Local agentic mode (loop-based, has iterations + citations) ───────────
    if trace.iterations:
        n_loops = len(trace.iterations)
        passed = any(it.passed for it in trace.iterations)

        if passed:
            passing_loop = next(it.loop_num for it in trace.iterations if it.passed)
            st.success(
                f"Good match found on loop {passing_loop} of {n_loops} "
                f"(score {next(it.best_score for it in trace.iterations if it.passed):.4f})"
            )
        elif trace.reranked:
            st.warning(f"Threshold not met after {n_loops} loops — LLM reranking applied")

        with st.expander("**Answer**", expanded=True):
            st.markdown(trace.answer if trace.answer else "_No answer returned._")

        if trace.citations:
            n_cit = len(trace.citations)
            with st.expander(f"**Citations** ({n_cit})", expanded=True):
                for i, chunk in enumerate(trace.citations, 1):
                    st.markdown(f"**[{i}] {chunk.source}**")
                    if chunk.score is not None:
                        st.caption(f"L2 distance: {chunk.score:.4f}")
                    st.text(chunk.content[:400])
                    if i < n_cit:
                        st.divider()
        return

    # ── Langflow agentic mode (ReAct steps) ───────────────────────────────────
    n_steps = len(trace.steps)
    with st.expander(f"**Agent Reasoning** ({n_steps} step{'s' if n_steps != 1 else ''})", expanded=True):
        if not trace.steps:
            st.caption("No reasoning steps captured.")
        for i, step in enumerate(trace.steps, 1):
            st.markdown(f"**Step {i}**")
            if step.thought:
                st.markdown(f"_Thought:_ {step.thought}")
            if step.action:
                st.markdown(f"_Action:_ `{step.action}`")
            if step.action_input:
                st.markdown(f"_Input:_ {step.action_input}")
            if step.observation:
                st.markdown(f"_Observation:_ {step.observation[:400]}")
            if i < len(trace.steps):
                st.divider()

    with st.expander("**Final Answer**", expanded=True):
        st.markdown(trace.answer if trace.answer else "_No answer returned._")


def render_observability_pane(
    classic: ClassicRAGTrace | None,
    agentic: AgenticRAGTrace | None,
) -> None:
    with st.expander("🔍 Observability & Trace", expanded=False):
        obs_classic, obs_agentic = st.tabs(["Classic RAG", "Agentic RAG"])

        with obs_classic:
            if classic is None:
                st.caption("No trace yet — run a query.")
            elif classic.error:
                st.error(classic.error)
            else:
                if classic.elapsed_ms is not None:
                    st.caption(f"Elapsed: **{classic.elapsed_ms:.0f} ms** | Model: `{classic.model_used}`")
                if not classic.retrieved_chunks:
                    st.caption("No chunks retrieved.")
                for i, chunk in enumerate(classic.retrieved_chunks, 1):
                    st.markdown(f"**[{i}] `{chunk.source}`**")
                    if chunk.score is not None:
                        st.caption(f"L2 distance: {chunk.score:.4f}")
                    st.caption(chunk.content[:300])
                    if i < len(classic.retrieved_chunks):
                        st.divider()

        with obs_agentic:
            if agentic is None:
                st.caption("No trace yet — run a query.")
            elif agentic.error:
                st.error(agentic.error)
            elif not agentic.iterations:
                st.caption("Langflow mode — loop trace not available for remote flows.")
            else:
                if agentic.elapsed_ms is not None:
                    st.caption(f"Total elapsed: **{agentic.elapsed_ms:.0f} ms** | Threshold: `{config.SIMILARITY_THRESHOLD}` (L2)")

                for it in agentic.iterations:
                    status = "✅ passed" if it.passed else "❌ retry"
                    label = f"Loop {it.loop_num} — {status} | best score: {it.best_score:.4f}"
                    with st.expander(label, expanded=it.passed):
                        st.caption(f"Query used: _{it.query_used}_")
                        for j, chunk in enumerate(it.chunks, 1):
                            st.markdown(f"**[{j}]** score `{chunk.score:.4f}` — `{chunk.source}`")
                            st.caption(chunk.content[:200])

                if agentic.reranked:
                    st.divider()
                    st.warning("Reranking triggered — LLM ranked all candidate chunks by relevance")
                    st.caption("Final citation order after rerank:")
                    for i, c in enumerate(agentic.citations, 1):
                        st.caption(f"[{i}] `{c.source}` (original score: {c.score:.4f})")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔧 Configuration")

    # Backend mode selector
    _modes = ["Local LangChain", "Langflow JSON", "Langflow"]
    _cur_idx = _modes.index(st.session_state.backend_mode) if st.session_state.backend_mode in _modes else 0
    backend_mode = st.radio(
        "Backend",
        _modes,
        index=_cur_idx,
        help=(
            "Local LangChain: Ollama + ChromaDB via config.py\n"
            "Langflow JSON: load a flow JSON and run offline\n"
            "Langflow: call the live Langflow Desktop API"
        ),
    )
    st.session_state.backend_mode = backend_mode

    st.divider()

    if backend_mode == "Local LangChain":
        st.subheader("Local Settings")
        st.caption(f"Ollama: `{config.OLLAMA_BASE_URL}`")
        st.caption(f"Embed model: `{config.EMBED_MODEL}`")
        st.caption(f"LLM: `{config.LLM_MODEL}`")
        st.caption(f"ChromaDB: `{config.CHROMA_DIR}`")

    elif backend_mode == "Langflow JSON":
        st.subheader("Langflow JSON Flow")
        uploaded_json = st.file_uploader(
            "Upload a Langflow flow JSON",
            type=["json"],
            key="_json_uploader",
        )
        if uploaded_json is not None:
            try:
                import tempfile as _tmp
                with _tmp.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                    tf.write(uploaded_json.read())
                    tf_path = tf.name
                settings = parse_langflow_json(tf_path)
                st.session_state.json_flow_settings = settings
                for w in settings.warnings:
                    st.warning(w)
                st.success(f"Loaded: **{settings.source_file}**")
                st.caption(f"LLM: `{settings.llm_model}` @ `{settings.llm_base_url}`")
                st.caption(f"Embed: `{settings.embed_model}`")
                st.caption(f"Collection: `{settings.collection_name}`")
                st.caption(f"Persist dir: `{settings.persist_dir}`")
                st.caption(f"Top-K: `{settings.top_k}`")
            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")
                st.session_state.json_flow_settings = None
        elif st.session_state.json_flow_settings:
            s = st.session_state.json_flow_settings
            st.caption(f"Loaded: **{s.source_file}**")
            st.caption(f"LLM: `{s.llm_model}` @ `{s.llm_base_url}`")
            st.caption(f"Embed: `{s.embed_model}`")
            st.caption(f"Collection: `{s.collection_name}`")

    else:  # Langflow
        st.subheader("Langflow Connection")
        langflow_url = st.text_input(
            "Langflow URL",
            value=st.session_state.langflow_url,
            placeholder="http://localhost:7860",
            key="_url_input",
        )
        st.session_state.langflow_url = langflow_url

        if st.button("Test Connection", use_container_width=True):
            _probe = LangflowClient(base_url=langflow_url, timeout=5)
            ok, msg = _probe.test_connection()
            if ok:
                st.session_state.connection_ok = True
                try:
                    st.session_state.flows = LangflowClient(base_url=langflow_url).get_flows()
                    st.success(msg)
                except Exception as e:
                    st.session_state.flows = []
                    st.warning(f"Connected but could not fetch flows: {e}")
            else:
                st.session_state.connection_ok = False
                st.session_state.flows = []
                st.error(msg)

        connected = st.session_state.connection_ok
        st.markdown(f"Status: {'🟢 Connected' if connected else '🔴 Disconnected'}")

        st.divider()
        st.subheader("Flow Selection")

        flows = st.session_state.flows
        flow_map: dict[str, str] = {f["name"]: f["id"] for f in flows}
        flow_names = list(flow_map.keys())
        no_flows = not flow_names

        if no_flows:
            st.caption("Connect to Langflow to load flows.")
            flow_names = ["(none)"]

        classic_name = st.selectbox(
            "Classic RAG Flow",
            flow_names,
            disabled=no_flows,
            key="_classic_select",
        )
        agentic_name = st.selectbox(
            "Agentic RAG Flow",
            flow_names,
            disabled=no_flows,
            key="_agentic_select",
        )

        st.session_state.classic_flow_id = flow_map.get(classic_name, "")
        st.session_state.agentic_flow_id = flow_map.get(agentic_name, "")

        if not no_flows:
            cid = st.session_state.classic_flow_id
            aid = st.session_state.agentic_flow_id
            st.caption(f"Classic: `{cid[:8]}…`" if cid else "Classic: not set")
            st.caption(f"Agentic: `{aid[:8]}…`" if aid else "Agentic: not set")


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🔍 RAG Playground")
mode_label = "Local LangChain" if st.session_state.backend_mode == "Local LangChain" else "Langflow + LangChain"
st.caption(f"Backend: **{mode_label}**")

if st.session_state.backend_mode in ("Local LangChain", "Langflow JSON"):
    tab_ask, tab_ingest, tab_browse = st.tabs(["Ask", "Ingest", "Browse Collection"])
else:
    tab_ask = st.tabs(["Ask"])[0]
    tab_ingest = None
    tab_browse = None


# ── Ask tab ───────────────────────────────────────────────────────────────────

with tab_ask:
    query = st.text_area(
        "Your question",
        placeholder="What is the leave policy?",
        height=80,
        key="_query_input",
    )

    run_clicked = st.button("▶ Run", type="primary", disabled=not query.strip())

    if run_clicked and query.strip():
        if st.session_state.backend_mode == "Local LangChain":
            with st.spinner("Running Classic & Agentic RAG pipelines…"):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    f_classic = executor.submit(_local_ask, query)
                    f_agentic = executor.submit(_local_ask_agentic, query)
                    st.session_state.local_trace = f_classic.result()
                    st.session_state.local_agentic_trace = f_agentic.result()
        elif st.session_state.backend_mode == "Langflow JSON":
            settings = st.session_state.json_flow_settings
            if not settings:
                st.warning("Upload a Langflow JSON file in the sidebar first.")
                st.stop()
            with st.spinner(f"Running offline — model: `{settings.llm_model}`…"):
                st.session_state.json_trace = _json_ask(query, settings)
        else:
            client = LangflowClient(
                base_url=st.session_state.langflow_url,
                timeout=config.LANGFLOW_REQUEST_TIMEOUT,
            )
            classic_id = st.session_state.classic_flow_id
            agentic_id = st.session_state.agentic_flow_id

            with st.spinner("Running both Langflow pipelines…"):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    f_classic = executor.submit(client.run_classic_rag, classic_id, query)
                    f_agentic = executor.submit(client.run_agentic_rag, agentic_id, query)
                    st.session_state.classic_trace = f_classic.result()
                    st.session_state.agentic_trace = f_agentic.result()

    st.divider()

    col_classic, col_agentic = st.columns(2, gap="large")

    if st.session_state.backend_mode == "Local LangChain":
        with col_classic:
            render_classic_panel(st.session_state.local_trace, title="⚙️ Classic RAG")
        with col_agentic:
            render_agentic_panel(st.session_state.local_agentic_trace)
        st.divider()
        render_observability_pane(
            st.session_state.local_trace,
            st.session_state.local_agentic_trace,
        )
    elif st.session_state.backend_mode == "Langflow JSON":
        with col_classic:
            s = st.session_state.json_flow_settings
            title = f"⚙️ Classic RAG — {s.source_file}" if s else "⚙️ Classic RAG"
            render_classic_panel(st.session_state.json_trace, title=title)
        with col_agentic:
            st.subheader("🤖 Agentic RAG")
            st.info("Agentic RAG coming soon.")
    else:
        with col_classic:
            render_classic_panel(st.session_state.classic_trace)
        with col_agentic:
            render_agentic_panel(st.session_state.agentic_trace)
        st.divider()
        render_observability_pane(
            st.session_state.classic_trace,
            st.session_state.agentic_trace,
        )


# ── Ingest tab (Local LangChain + Langflow JSON) ──────────────────────────────

if tab_ingest is not None:
    with tab_ingest:
        # Resolve ingest settings based on backend mode
        _json_settings: FlowSettings | None = st.session_state.json_flow_settings
        _is_json_mode = st.session_state.backend_mode == "Langflow JSON"

        if _is_json_mode:
            st.header("Ingest Documents")
            if not _json_settings:
                st.warning("Upload a Langflow JSON file in the sidebar first.")
                st.stop()
            st.caption(
                f"Embedding into collection **`{_json_settings.collection_name}`** "
                f"using `{_json_settings.embed_model}` — settings loaded from `{_json_settings.source_file}`."
            )
            _persist_dir = _json_settings.persist_dir
            _collection = _json_settings.collection_name
            _embed_model = _json_settings.embed_model
            _embed_base_url = _json_settings.embed_base_url
            _default_chunk_size = _json_settings.chunk_size
            _default_chunk_overlap = _json_settings.chunk_overlap
        else:
            st.header("Ingest Documents")
            st.caption(
                "Chunks and embeds documents into the **local ChromaDB** — "
                "used by the Local LangChain backend."
            )
            _persist_dir = config.CHROMA_DIR
            _collection = config.CHROMA_COLLECTION
            _embed_model = config.EMBED_MODEL
            _embed_base_url = config.OLLAMA_BASE_URL
            _default_chunk_size = config.CHUNK_SIZE
            _default_chunk_overlap = config.CHUNK_OVERLAP

        uploaded_files = st.file_uploader(
            "Upload files (.txt, .pdf, .md)",
            accept_multiple_files=True,
            type=["txt", "pdf", "md"],
        )

        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider(
                "Chunk size (tokens)",
                min_value=100,
                max_value=2000,
                value=_default_chunk_size,
                step=50,
                help="Smaller chunks = finer retrieval; larger chunks = more context per hit.",
            )
        with col2:
            chunk_overlap = st.slider(
                "Chunk overlap (tokens)",
                min_value=0,
                max_value=400,
                value=_default_chunk_overlap,
                step=10,
            )

        if st.button("Ingest", type="primary", disabled=not uploaded_files):
            with tempfile.TemporaryDirectory() as tmp_dir:
                for f in uploaded_files:
                    dest = os.path.join(tmp_dir, f.name)
                    with open(dest, "wb") as out:
                        out.write(f.getbuffer())

                try:
                    from ingest import load_documents, split_documents, embed_and_store

                    with st.spinner("Loading and chunking documents…"):
                        docs = load_documents(tmp_dir)
                        if not docs:
                            st.error("No readable content found in the uploaded file(s).")
                            st.stop()
                        chunks = split_documents(docs, chunk_size, chunk_overlap)

                    n_chunks = len(chunks)
                    st.info(f"Created **{n_chunks} chunks** from {len(uploaded_files)} file(s). Starting embedding…")

                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    def _update_progress(done: int, total: int) -> None:
                        progress_bar.progress(done / total)
                        status_text.caption(f"Embedding chunk {done} / {total}")

                    embed_and_store(
                        chunks, _persist_dir, _collection,
                        _embed_model, _embed_base_url,
                        progress_callback=_update_progress,
                    )
                    progress_bar.progress(1.0)
                    status_text.empty()
                    st.success(f"Done! Stored {n_chunks} chunks from {len(uploaded_files)} file(s).")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

        if not _is_json_mode:
            st.divider()
            st.subheader("Or ingest sample docs")
            if st.button("Ingest sample docs"):
                if not os.path.isdir(config.DOCS_DIR):
                    st.warning(f"Sample docs directory `{config.DOCS_DIR}` not found.")
                else:
                    with st.spinner("Ingesting sample docs …"):
                        try:
                            n = ingest(persist_dir=config.CHROMA_DIR)
                            st.success(f"Done! Created {n} chunks.")
                        except Exception as e:
                            st.error(f"Failed: {e}")

        st.divider()
        st.subheader("Manage Collection")
        st.caption("Remove previously ingested documents from ChromaDB.")

        try:
            sources = list_sources(_persist_dir, _collection, _embed_model, _embed_base_url)
        except Exception:
            sources = []

        if not sources:
            st.info("No documents found in ChromaDB.")
        else:
            to_delete = st.multiselect(
                "Select documents to delete",
                options=sources,
                placeholder="Choose one or more files…",
            )
            if st.button("Delete selected", type="primary", disabled=not to_delete):
                total = 0
                errors = []
                for src in to_delete:
                    try:
                        total += delete_by_source(src, _persist_dir, _collection,
                                                  _embed_model, _embed_base_url)
                    except Exception as e:
                        errors.append(f"{src}: {e}")
                if errors:
                    st.error("Some deletions failed:\n" + "\n".join(errors))
                else:
                    st.success(f"Deleted {total} chunk(s) from {len(to_delete)} document(s).")
                    st.rerun()


# ── Browse Collection tab (Local LangChain only) ──────────────────────────────

if tab_browse is not None:
    with tab_browse:
        st.header("Browse Collection")

        if st.button("Refresh", key="_browse_refresh"):
            st.rerun()

        _b_settings: FlowSettings | None = st.session_state.json_flow_settings
        _b_is_json = st.session_state.backend_mode == "Langflow JSON"
        _b_persist = _b_settings.persist_dir if _b_is_json and _b_settings else config.CHROMA_DIR
        _b_collection = _b_settings.collection_name if _b_is_json and _b_settings else config.CHROMA_COLLECTION
        _b_embed = _b_settings.embed_model if _b_is_json and _b_settings else config.EMBED_MODEL
        _b_base_url = _b_settings.embed_base_url if _b_is_json and _b_settings else config.OLLAMA_BASE_URL

        if _b_is_json and not _b_settings:
            st.warning("Upload a Langflow JSON file in the sidebar first.")
            st.stop()

        try:
            data = get_all_chunks(_b_persist, _b_collection, _b_embed, _b_base_url)
        except Exception as e:
            st.error(f"Could not load ChromaDB: {e}")
            data = None

        if data is not None:
            ids = data["ids"]
            docs = data["documents"]
            metas = data["metadatas"]

            if not ids:
                st.info("ChromaDB is empty. Ingest some documents first.")
            else:
                # Build per-source index
                from collections import defaultdict
                from pathlib import Path as _Path

                source_chunks: dict[str, list] = defaultdict(list)
                for doc_id, content, meta in zip(ids, docs, metas):
                    src = _Path(meta.get("source", "unknown")).name if meta else "unknown"
                    source_chunks[src].append({"id": doc_id, "content": content, "meta": meta})

                # Summary metrics
                col_a, col_b = st.columns(2)
                col_a.metric("Total chunks", len(ids))
                col_b.metric("Documents", len(source_chunks))

                st.divider()

                # Per-document expanders
                for src_name, chunks in sorted(source_chunks.items()):
                    with st.expander(f"**{src_name}** — {len(chunks)} chunk(s)"):
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(f"**Chunk {i}** `{chunk['id'][:8]}…`")
                            st.text(chunk["content"][:500] + ("…" if len(chunk["content"]) > 500 else ""))
                            if chunk["meta"]:
                                extra = {k: v for k, v in chunk["meta"].items() if k != "source"}
                                if extra:
                                    st.caption("  ".join(f"{k}: {v}" for k, v in extra.items()))
                            if i < len(chunks):
                                st.divider()
