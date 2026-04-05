"""
AInsights — Main Application
Single-page layout with logo hero, agent descriptions, and tabbed workflow.
All three agent interfaces live here as tabs — no separate pages needed.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AInsights",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CSS_PATH    = Path(__file__).parent / "assets" / "style.css"

# ─────────────────────────────────────────────────────────────────────────────
# CSS injection
# ─────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
def init_session() -> None:
    defaults: dict = {
        "pipeline_status":  "idle",
        "chat_history":     [],
        "schema_info":      None,
        "uploaded_files":   [],     # list of {name, size_kb, rows, columns, method}
        "chart_figures":    [],
        "current_job_id":   None,
        "active_tab":       0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        # ── Logo ────────────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="padding: 1rem 0 1.25rem;">
              <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:36px; height:36px; background:linear-gradient(135deg,#6366f1,#34d399);
                            border-radius:10px; display:flex; align-items:center;
                            justify-content:center; font-size:18px; flex-shrink:0;">
                  ⬡
                </div>
                <div>
                  <div style="font-size:1.15rem; font-weight:700;
                              letter-spacing:-0.03em; color:#f1f5f9;">AInsights</div>
                  <div style="font-size:0.68rem; color:#475569;
                              letter-spacing:0.1em; text-transform:uppercase;">
                    Local-first · 100% private
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='margin:0 0 1rem;'>", unsafe_allow_html=True)

        # ── Pipeline status ─────────────────────────────────────────────────
        status = st.session_state.pipeline_status
        status_map = {
            "idle":       ("idle",    "⬡", "Waiting for data"),
            "processing": ("warning", "⟳", "Pipeline running …"),
            "ready":      ("success", "✓", "Data ready"),
            "error":      ("error",   "✕", "Error occurred"),
        }
        badge_cls, icon, label = status_map.get(status, ("idle", "⬡", status))

        st.markdown(
            '<div style="margin-bottom:1.25rem;">'
            '<div style="font-size:0.7rem;color:#475569;text-transform:uppercase;'
            'letter-spacing:0.09em;margin-bottom:7px;">Pipeline</div>'
            f'<span class="badge badge-{badge_cls}">{icon} {label}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Uploaded files ──────────────────────────────────────────────────
        uploaded_files = st.session_state.get("uploaded_files", [])
        if uploaded_files:
            st.markdown(
                """<div style="font-size:0.7rem; color:#475569; text-transform:uppercase;
                               letter-spacing:0.09em; margin-bottom:8px;">
                     Loaded datasets
                   </div>""",
                unsafe_allow_html=True,
            )
            for f in uploaded_files[-3:]:   # show last 3
                ext_icon = {
                    ".csv": "📊", ".xlsx": "📗", ".xls": "📗",
                    ".json": "📋", ".xml": "📄", ".pdf": "📕",
                }.get(Path(f["name"]).suffix.lower(), "📁")

                st.markdown(
                    f"""
                    <div class="sidebar-file-item">
                      <span class="sidebar-file-icon">{ext_icon}</span>
                      <div style="min-width:0;">
                        <div class="sidebar-file-name">{f["name"]}</div>
                        <div class="sidebar-file-meta">{f.get("size_kb","?")} KB</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Dataset stats ───────────────────────────────────────────────────
        schema = st.session_state.get("schema_info")
        if schema:
            st.markdown(
                """<div style="font-size:0.7rem; color:#475569; text-transform:uppercase;
                               letter-spacing:0.09em; margin:1rem 0 8px;">
                     Dataset info
                   </div>""",
                unsafe_allow_html=True,
            )
            rows    = schema.get("rows", "—")
            cols    = schema.get("columns", "—")
            col_map = schema.get("schema", {})

            # Count semantic types
            type_counts: dict[str, int] = {}
            for meta in col_map.values():
                t = meta.get("dtype", "object")
                key = "numeric" if "int" in t or "float" in t else \
                      "datetime" if "datetime" in t else "text"
                type_counts[key] = type_counts.get(key, 0) + 1

            stat_rows = [
                ("Rows",    f"{rows:,}" if isinstance(rows, int) else rows),
                ("Columns", cols),
                ("Numeric", type_counts.get("numeric", 0)),
                ("Text",    type_counts.get("text", 0)),
                ("Datetime",type_counts.get("datetime", 0)),
            ]
            stats_html = "".join(
                f"""<div class="sidebar-stat-row">
                      <span class="sidebar-stat-key">{k}</span>
                      <span class="sidebar-stat-value">{v}</span>
                    </div>"""
                for k, v in stat_rows
            )
            st.markdown(
                f'<div style="border:1px solid var(--glass-border); '
                f'border-radius:var(--radius-md); padding:0.6rem 0.85rem; '
                f'background:var(--glass-highlight);">{stats_html}</div>',
                unsafe_allow_html=True,
            )

            # Top columns
            if col_map:
                st.markdown(
                    """<div style="font-size:0.7rem; color:#475569; text-transform:uppercase;
                                   letter-spacing:0.09em; margin:1rem 0 6px;">
                         Columns
                       </div>""",
                    unsafe_allow_html=True,
                )
                for col_name, meta in list(col_map.items())[:8]:
                    dtype = meta.get("dtype", "")
                    color = "#818cf8" if ("int" in dtype or "float" in dtype) else \
                            "#34d399" if "datetime" in dtype else "#94a3b8"
                    st.markdown(
                        f"""<div style="display:flex; justify-content:space-between;
                                        align-items:center; padding:3px 0;
                                        font-size:0.78rem; border-bottom:1px solid rgba(30,41,59,0.5);">
                              <span style="color:#e2e8f0; overflow:hidden; text-overflow:ellipsis;
                                           white-space:nowrap; max-width:65%;">{col_name}</span>
                              <span style="color:{color}; font-size:0.68rem;
                                           font-family:'JetBrains Mono',monospace;">{dtype}</span>
                            </div>""",
                        unsafe_allow_html=True,
                    )
                if len(col_map) > 8:
                    st.markdown(
                        f'<div style="font-size:0.72rem; color:#475569; '
                        f'padding-top:5px;">+{len(col_map)-8} more columns</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)

        # ── Version footer ──────────────────────────────────────────────────
        st.markdown(
            """<div style="font-size:0.68rem; color:#334155; padding-bottom:0.5rem;">
                 AInsights v1.0 &nbsp;·&nbsp; qwen2.5-coder:7b &nbsp;·&nbsp; ChromaDB
               </div>""",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now().strftime("%H:%M")


def _client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(timeout=timeout)


def _fetch_schema() -> dict | None:
    try:
        r = httpx.get(f"{BACKEND_URL}/data/schema", timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _poll_job(job_id: str) -> dict:
    """Poll job endpoint until terminal state. Returns final job dict."""
    backoff = 1.2
    for _ in range(80):
        try:
            r = httpx.get(f"{BACKEND_URL}/jobs/{job_id}", timeout=8)
            job = r.json()
            if job.get("status") in ("complete", "failed"):
                return job
        except Exception:
            pass
        time.sleep(backoff)
        backoff = min(backoff * 1.35, 7)
    return {"status": "timeout", "detail": "Job timed out."}


# ─────────────────────────────────────────────────────────────────────────────
# Section: Hero Landing
# ─────────────────────────────────────────────────────────────────────────────
def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-wrapper">
          <div class="hero-logo">
            <div class="hero-logo-icon">⬡</div>
            <div class="hero-logo-text">AInsights</div>
          </div>
          <div class="hero-tagline">
            A privacy-first, local-first Business Intelligence platform powered by on-device AI.
            Upload your data, visualise it instantly, and query it in plain English —
            entirely offline, with no data ever leaving your machine.
          </div>
          <div class="hero-badges">
            <span class="badge badge-info">100% Local</span>
            <span class="badge badge-success">Zero Cloud</span>
            <span class="badge badge-pink">RAG-Powered</span>
            <span class="badge badge-warning">qwen2.5-coder:7b</span>
            <span class="badge badge-idle">ChromaDB</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Agent architecture cards
    st.markdown(
        """<div style="font-size:0.7rem; color:#475569; text-transform:uppercase;
                       letter-spacing:0.11em; margin:0.5rem 0 1rem; text-align:center;">
             Three-agent pipeline architecture
           </div>""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    agents = [
        (
            c1, "agent-icon-a", "🔧", "Agent A",
            "Data Engineer",
            "Autonomous ETL pipeline that profiles your data's schema, generates Python cleaning "
            "code via LLM, and executes it in a secure sandbox.",
            [
                ("Supports CSV, Excel, JSON, XML, PDF tables", "#fb923c"),
                ("LLM-generated cleaning with rule-based fallback", "#fb923c"),
                ("Zero tolerance for column loss or data destruction", "#fb923c"),
            ]
        ),
        (
            c2, "agent-icon-b", "📊", "Agent B",
            "Visualizer",
            "Reads the cleaned dataset and autonomously selects the most mathematically "
            "appropriate Plotly charts based on detected column types.",
            [
                ("Auto-detects numeric, categorical, and datetime columns", "#34d399"),
                ("Renders correlation heatmaps, time series, box plots, and more", "#34d399"),
                ("No manual chart configuration required", "#34d399"),
            ]
        ),
        (
            c3, "agent-icon-c", "💬", "Agent C",
            "Analyst",
            "Dual-context conversational BI. Combines vector retrieval from ChromaDB "
            "with live Pandas calculations to answer questions without hallucinating numbers.",
            [
                ("RAG retrieval over documents and tabular rows", "#818cf8"),
                ("Live Pandas execution for precise statistics", "#818cf8"),
                ("Grounded, citation-aware answers", "#818cf8"),
            ]
        ),
        (
            c4, "agent-icon-r", "🧠", "RAG Engine",
            "Memory Pipeline",
            "Embeds documents and CSV rows into a local ChromaDB vector store using "
            "the all-MiniLM-L6-v2 model, enabling semantic search over your entire dataset.",
            [
                ("Chunks and embeds PDF, TXT, and Markdown documents", "#f472b6"),
                ("Converts tabular rows into searchable sentences", "#f472b6"),
                ("Fully offline — HuggingFace model runs locally", "#f472b6"),
            ]
        ),
    ]

    for col, icon_cls, icon, label, name, desc, features in agents:
        with col:
            feature_html = "".join(
                f"""<div class="agent-feature">
                      <div class="agent-feature-dot"
                           style="background:{color};"></div>
                      <span>{text}</span>
                    </div>"""
                for text, color in features
            )
            st.markdown(
                f"""
                <div class="agent-card">
                  <div class="agent-icon {icon_cls}">{icon}</div>
                  <div class="agent-label">{label}</div>
                  <div class="agent-name">{name}</div>
                  <div class="agent-desc">{desc}</div>
                  {feature_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Upload & Clean (Agent A)
# ─────────────────────────────────────────────────────────────────────────────
def render_upload_tab() -> None:
    st.markdown(
        """
        <div class="section-header">
          <div class="section-icon agent-icon-a">🔧</div>
          <div>
            <div class="section-title">Upload & Clean</div>
            <div class="section-subtitle">
              Agent A profiles your schema and cleans your data automatically
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    inner_tab_data, inner_tab_docs = st.tabs(["📂 Structured Data", "📄 Documents & PDFs"])

    # ── Structured data ────────────────────────────────────────────────────
    with inner_tab_data:
        st.markdown(
            """
            <div class="glass-card" style="padding:1rem 1.25rem;">
              <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
                <span style="font-size:0.78rem; color:#64748b; margin-right:4px;">Supported:</span>
                <span class="badge badge-info">CSV</span>
                <span class="badge badge-info">Excel (.xlsx / .xls)</span>
                <span class="badge badge-info">JSON</span>
                <span class="badge badge-info">XML</span>
                <span class="badge badge-info">PDF tables</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_data = st.file_uploader(
            "Drop your data file here",
            type=["csv", "xlsx", "xls", "json", "xml", "pdf"],
            key="data_uploader",
            label_visibility="collapsed",
        )

        if uploaded_data:
            col_meta, col_btn = st.columns([4, 1])
            with col_meta:
                size_kb = round(uploaded_data.size / 1024, 1)
                st.markdown(
                    f"""<div style="display:flex; align-items:center; gap:12px;
                                    padding:0.5rem 0;">
                          <span style="font-size:1.2rem;">📎</span>
                          <div>
                            <div style="color:#f1f5f9; font-size:0.9rem;
                                        font-weight:500;">{uploaded_data.name}</div>
                            <div style="color:#64748b; font-size:0.78rem;">{size_kb} KB</div>
                          </div>
                        </div>""",
                    unsafe_allow_html=True,
                )
            with col_btn:
                run_btn = st.button(
                    "▶ Run Pipeline",
                    type="primary",
                    use_container_width=True,
                    key="run_pipeline_btn",
                )

            if run_btn:
                _run_pipeline(uploaded_data)

    # ── Documents ──────────────────────────────────────────────────────────
    with inner_tab_docs:
        st.markdown(
            """
            <div class="glass-card">
              <div style="font-size:0.88rem; color:#94a3b8; line-height:1.65;">
                Upload PDFs, text files, or Markdown documents to enrich Agent C's knowledge base.
                Documents are chunked, embedded with <code style="color:#818cf8;
                font-family:JetBrains Mono,monospace; font-size:0.82rem;">all-MiniLM-L6-v2</code>,
                and stored in the local ChromaDB vector store.
                Agent C will retrieve relevant passages when answering your questions.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_doc = st.file_uploader(
            "Drop a document here",
            type=["pdf", "txt", "md"],
            key="doc_uploader",
            label_visibility="collapsed",
        )

        if uploaded_doc:
            col_i, col_b = st.columns([4, 1])
            with col_i:
                st.markdown(
                    f"""<div style="color:#f1f5f9; font-size:0.9rem;
                                    font-weight:500; padding:0.5rem 0;">
                          📎 {uploaded_doc.name}
                        </div>""",
                    unsafe_allow_html=True,
                )
            with col_b:
                ingest_btn = st.button(
                    "🔍 Ingest",
                    type="primary",
                    use_container_width=True,
                    key="ingest_doc_btn",
                )

            if ingest_btn:
                _ingest_document(uploaded_doc)


def _run_pipeline(uploaded_file) -> None:
    """Upload file → Agent A ETL → RAG ingestion with live status updates."""
    st.session_state.pipeline_status = "processing"

    # Live status container
    status_container = st.empty()

    def _render_steps(steps: list[tuple[str, str, str]]) -> None:
        items = ""
        for state, label, detail in steps:
            detail_html = f'<div style="font-size:0.78rem;opacity:0.75;margin-top:2px;">{detail}</div>' if detail else ""
            items += (
                f'<div class="pipeline-step {state}">'
                f'<div class="pipeline-dot {state}"></div>'
                f'<div><div style="font-weight:500;">{label}</div>{detail_html}</div>'
                f'</div>'
            )
        html = (
            '<div class="glass-card" style="padding:1.25rem;">'
            '<div style="font-size:0.78rem;color:#64748b;text-transform:uppercase;'
            'letter-spacing:0.08em;margin-bottom:0.75rem;">Pipeline progress</div>'
            + items +
            '</div>'
        )
        status_container.markdown(html, unsafe_allow_html=True)

    steps = [
        ("active",  "Uploading file to Agent A …", ""),
        ("pending", "Schema profiling",             ""),
        ("pending", "LLM cleaning code generation", ""),
        ("pending", "Sandbox execution",            ""),
        ("pending", "RAG ingestion",                ""),
    ]
    _render_steps(steps)

    # Upload
    try:
        r = httpx.post(
            f"{BACKEND_URL}/upload/data",
            files={"file": (uploaded_file.name, uploaded_file.getvalue())},
            timeout=45,
        )
        r.raise_for_status()
        job_id = r.json()["job_id"]
        st.session_state.current_job_id = job_id
    except Exception as exc:
        st.session_state.pipeline_status = "error"
        st.error(f"Upload failed: {exc}")
        return

    # Poll with live step updates
    steps[0] = ("done", "File uploaded", "")
    steps[1] = ("active", "Schema profiling …", "Analysing column types and statistics")
    _render_steps(steps)

    backoff = 1.5
    for _ in range(90):
        try:
            r = httpx.get(f"{BACKEND_URL}/jobs/{job_id}", timeout=8)
            job = r.json()
            status = job.get("status", "")
            detail = job.get("detail", "")

            # Map backend detail messages to step states
            if "profiling" in detail.lower() or "schema" in detail.lower():
                steps[1] = ("active",  "Schema profiling …",             detail)
                steps[2] = ("pending", "LLM cleaning code generation",   "")
            elif "llm" in detail.lower() or "cleaning" in detail.lower() or "agent a" in detail.lower():
                steps[1] = ("done",   "Schema profiling complete",        "")
                steps[2] = ("active", "LLM cleaning code generation …",  detail)
                steps[3] = ("active", "Sandbox execution …",             "Running generated code")
            elif "rag" in detail.lower() or "ingest" in detail.lower():
                steps[1] = ("done",   "Schema profiling complete", "")
                steps[2] = ("done",   "Cleaning code executed",    "")
                steps[3] = ("done",   "Sandbox execution complete","")
                steps[4] = ("active", "RAG ingestion …",           "Embedding rows into ChromaDB")
            _render_steps(steps)

            if status == "complete":
                steps[4] = ("done", "RAG ingestion complete", "")
                _render_steps(steps)

                # Record the uploaded file in session
                result = job.get("result", {})
                size_kb = round(uploaded_file.size / 1024, 1)
                file_record = {
                    "name":    uploaded_file.name,
                    "size_kb": size_kb,
                    "rows":    result.get("cleaned_shape", [0])[0],
                    "columns": result.get("cleaned_shape", [0, 0])[1] if len(result.get("cleaned_shape", [])) > 1 else 0,
                    "method":  result.get("cleaning_method", "unknown"),
                }
                files = st.session_state.get("uploaded_files", [])
                files.append(file_record)
                st.session_state.uploaded_files = files[-5:]   # keep last 5

                # Fetch and store schema
                schema = _fetch_schema()
                if schema:
                    st.session_state.schema_info = schema

                st.session_state.pipeline_status = "ready"
                _show_pipeline_results(result, schema)
                return

            if status == "failed":
                st.session_state.pipeline_status = "error"
                st.error(f"Pipeline failed: {detail}")
                return

        except Exception:
            pass

        time.sleep(backoff)
        backoff = min(backoff * 1.3, 6)

    st.session_state.pipeline_status = "error"
    st.error("Pipeline timed out. Check backend logs.")


def _show_pipeline_results(result: dict, schema: dict | None) -> None:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, result.get("original_shape",  [0])[0],   "Original rows",   None),
        (c2, result.get("cleaned_shape",   [0])[0],   "Cleaned rows",    None),
        (c3, result.get("null_cells_before", 0),       "Nulls before",    None),
        (c4, result.get("null_cells_after",  0),       "Nulls after",     None),
    ]
    for col, val, label, _ in kpis:
        with col:
            st.markdown(
                f"""<div class="kpi-card">
                      <div class="kpi-value">{val:,}</div>
                      <div class="kpi-label">{label}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    # Method + elapsed
    method  = result.get("cleaning_method", "unknown")
    elapsed = result.get("elapsed_seconds", 0)
    badge   = "success" if "llm" in method else "warning"
    st.markdown(
        f"""<div style="margin-top:0.85rem; display:flex; gap:12px;
                        align-items:center; flex-wrap:wrap;">
              <span class="badge badge-{badge}">Method: {method}</span>
              <span style="font-size:0.8rem; color:#64748b;">
                Completed in {elapsed}s
              </span>
            </div>""",
        unsafe_allow_html=True,
    )

    # Schema preview
    if schema:
        st.markdown(
            """<div style="margin-top:1.5rem; margin-bottom:0.5rem;
                           font-size:0.88rem; color:#94a3b8; font-weight:500;">
                 Schema preview
               </div>""",
            unsafe_allow_html=True,
        )
        import pandas as pd
        rows = [
            {
                "Column":  col_name,
                "Type":    meta["dtype"],
                "Nulls":   meta["nulls"],
                "Unique":  meta["unique"],
                "Sample":  str(meta["sample"])[:55],
            }
            for col_name, meta in schema.get("schema", {}).items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Download
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        try:
            csv_bytes = httpx.get(f"{BACKEND_URL}/data/cleaned", timeout=20).content
            st.download_button(
                "⬇ Download cleaned CSV",
                data=csv_bytes,
                file_name="ainsights_cleaned_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception:
            pass

    st.success("✓ Pipeline complete — switch to the Visualize or Chat tab to explore your data.")


def _ingest_document(uploaded_doc) -> None:
    with st.status("Ingesting document into RAG engine …", expanded=True) as status_box:
        st.write(f"Uploading {uploaded_doc.name} …")
        try:
            r = httpx.post(
                f"{BACKEND_URL}/upload/document",
                files={"file": (uploaded_doc.name, uploaded_doc.getvalue())},
                timeout=30,
            )
            r.raise_for_status()
            job_id = r.json()["job_id"]
        except Exception as exc:
            status_box.update(label="Upload failed", state="error")
            st.error(str(exc))
            return

        st.write("Chunking and embedding …")
        final = _poll_job(job_id)

        if final.get("status") == "complete":
            status_box.update(label="Document indexed successfully", state="complete")
            st.success(f"✓ {final.get('detail', 'Indexed.')}")
        else:
            status_box.update(label="Ingestion failed", state="error")
            st.error(final.get("detail", "Unknown error"))


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Visualize (Agent B)
# ─────────────────────────────────────────────────────────────────────────────
def render_visualize_tab() -> None:
    st.markdown(
        """
        <div class="section-header">
          <div class="section-icon agent-icon-b">📊</div>
          <div>
            <div class="section-title">Visualize</div>
            <div class="section-subtitle">
              Agent B auto-selects and renders the most appropriate charts for your dataset
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.pipeline_status not in ("ready",):
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:3.5rem 2rem;">
              <div style="font-size:2.5rem; margin-bottom:1rem;">📊</div>
              <div style="font-size:1rem; color:#94a3b8; font-weight:500;
                          margin-bottom:0.4rem;">No data loaded</div>
              <div style="font-size:0.84rem; color:#64748b;">
                Upload a dataset in the <strong style="color:#818cf8;">Upload & Clean</strong>
                tab to generate visualisations.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Controls
    col_btn, col_info = st.columns([1, 5])
    with col_btn:
        refresh = st.button("⟳ Generate", type="primary", use_container_width=True)
    with col_info:
        if st.session_state.schema_info:
            info = st.session_state.schema_info
            st.markdown(
                f"""<div style="padding-top:0.55rem; font-size:0.82rem; color:#64748b;">
                      Analysing {info.get('rows','?'):,} rows × {info.get('columns','?')} columns
                    </div>""",
                unsafe_allow_html=True,
            )

    # Fetch charts
    if refresh or not st.session_state.get("chart_figures"):
        with st.status("Agent B is analysing your data and generating charts …", expanded=True) as s:
            st.write("Detecting column types …")
            st.write("Selecting optimal chart types …")
            try:
                r = httpx.get(f"{BACKEND_URL}/visualize", timeout=120)
                r.raise_for_status()
                st.session_state.chart_figures = r.json().get("charts", [])
                st.write(f"Generated {len(st.session_state.chart_figures)} charts.")
                s.update(label="Charts ready", state="complete")
            except Exception as exc:
                s.update(label="Failed", state="error")
                st.error(str(exc))
                return

    figures = st.session_state.get("chart_figures", [])
    if not figures:
        st.info("No charts generated. Ensure your cleaned data has numeric or categorical columns.")
        return

    import plotly.graph_objects as go

    # Separate summary table from visual charts
    summary = [f for f in figures if "Descriptive Statistics" in
               f.get("layout", {}).get("title", {}).get("text", "")]
    charts  = [f for f in figures if f not in summary]

    st.markdown(
        f"""<div style="font-size:0.78rem; color:#64748b; margin-bottom:1.25rem;">
              {len(figures)} chart{'s' if len(figures) != 1 else ''} generated
              from {st.session_state.schema_info.get('rows','?'):,} rows
            </div>""",
        unsafe_allow_html=True,
    )

    # 2-column grid
    for i in range(0, len(charts), 2):
        cols = st.columns(2, gap="medium")
        for j, col in enumerate(cols):
            if i + j < len(charts):
                fig_dict = charts[i + j]
                fig      = go.Figure(fig_dict)
                layout   = fig_dict.get("layout", {})
                title    = layout.get("title", {}).get("text", f"Chart {i+j+1}")
                schema   = st.session_state.schema_info or {}
                n_rows   = schema.get("rows", "")
                subtitle = _chart_subtitle(fig_dict, n_rows)

                with col:
                    st.markdown(
                        f"""<div class="chart-card">
                              <div class="chart-card-header">
                                <div class="chart-card-title">{title}</div>
                                <div class="chart-card-subtitle">{subtitle}</div>
                              </div>
                              <div class="chart-card-body">""",
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": True, "displaylogo": False,
                                "modeBarButtonsToRemove": ["toImage"]},
                    )
                    st.markdown("</div></div>", unsafe_allow_html=True)

    # Full-width summary table
    for fig_dict in summary:
        fig = go.Figure(fig_dict)
        schema   = st.session_state.schema_info or {}
        n_cols   = schema.get("columns", "")
        subtitle = f"Descriptive statistics across {n_cols} numeric columns" if n_cols else ""
        st.markdown(
            f"""<div class="chart-card">
                  <div class="chart-card-header">
                    <div class="chart-card-title">Descriptive Statistics</div>
                    <div class="chart-card-subtitle">{subtitle}</div>
                  </div>
                  <div class="chart-card-body">""",
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.markdown("</div></div>", unsafe_allow_html=True)


def _chart_subtitle(fig_dict: dict, n_rows: int | str) -> str:
    """Generate a descriptive subtitle for each chart type."""
    title = fig_dict.get("layout", {}).get("title", {}).get("text", "").lower()
    row_str = f"{n_rows:,}" if isinstance(n_rows, int) else str(n_rows)

    if "correlation" in title:
        return f"Pearson r matrix across numeric columns · {row_str} rows"
    if "over time" in title:
        return f"Trend analysis · {row_str} data points"
    if "distribution" in title:
        return f"Frequency histogram with 40 bins · {row_str} rows"
    if "average" in title or "by" in title.split("average")[-1:][0:1]:
        return f"Mean value grouped by category · {row_str} rows"
    if "vs" in title:
        return f"Scatter plot with OLS trendline · up to 2,000 sampled points"
    if "distribution by" in title:
        return f"Quartile box plot with outliers · {row_str} rows"
    return f"{row_str} rows analysed"


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Chat (Agent C)
# ─────────────────────────────────────────────────────────────────────────────
SUGGESTIONS = [
    "What are the top 5 values by the highest numeric column?",
    "Show me summary statistics for all numeric columns.",
    "What is the average value grouped by the main category?",
    "Are there any outliers or anomalies in the data?",
    "What trends do you see over time?",
]


def render_chat_tab() -> None:
    st.markdown(
        """
        <div class="section-header">
          <div class="section-icon agent-icon-c">💬</div>
          <div>
            <div class="section-title">Chat with Data</div>
            <div class="section-subtitle">
              Agent C retrieves vector context and runs live Pandas calculations
              to ground every answer in real data
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.pipeline_status not in ("ready",):
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:3.5rem 2rem;">
              <div style="font-size:2.5rem; margin-bottom:1rem;">💬</div>
              <div style="font-size:1rem; color:#94a3b8; font-weight:500;
                          margin-bottom:0.4rem;">No data loaded</div>
              <div style="font-size:0.84rem; color:#64748b;">
                Upload and clean a dataset in the
                <strong style="color:#818cf8;">Upload & Clean</strong>
                tab before starting a conversation.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Suggestion chips (only when history is empty)
    if not st.session_state.chat_history:
        st.markdown(
            """<div style="font-size:0.75rem; color:#475569; text-transform:uppercase;
                           letter-spacing:0.09em; margin-bottom:0.75rem;">
                 Suggested questions
               </div>""",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(SUGGESTIONS))
        for col, sug in zip(cols, SUGGESTIONS):
            with col:
                if st.button(
                    sug[:40] + ("…" if len(sug) > 40 else ""),
                    use_container_width=True,
                    key=f"sug_{hash(sug)}",
                ):
                    st.session_state._pending_q = sug
                    st.rerun()

    # Chat history
    for msg in st.session_state.chat_history:
        _render_chat_message(msg)

    # Input row
    col_input, col_send, col_clear = st.columns([8, 1, 1])
    with col_input:
        question = st.text_input(
            "Ask a question …",
            key="chat_input",
            label_visibility="collapsed",
            placeholder="e.g. What is the average revenue by region?",
        )
    with col_send:
        send = st.button("➤", type="primary", use_container_width=True, key="send_btn")
    with col_clear:
        if st.button("🗑", use_container_width=True, key="clear_btn", help="Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

    # Handle pending suggestion
    pending = st.session_state.pop("_pending_q", None)
    if pending:
        question = pending
        send = True

    # Send
    if send and question and question.strip():
        _send_chat(question.strip())

    # Export
    if st.session_state.chat_history:
        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
        col_exp, _ = st.columns([1, 6])
        with col_exp:
            text = "\n\n".join(
                f"[{m['role'].upper()} {m.get('time','')}]\n{m['content']}"
                for m in st.session_state.chat_history
            )
            st.download_button(
                "⬇ Export chat",
                data=text,
                file_name="ainsights_chat.txt",
                mime="text/plain",
                use_container_width=True,
            )


def _render_chat_message(msg: dict) -> None:
    role    = msg["role"]
    content = msg["content"]
    t       = msg.get("time", "")

    if role == "user":
        html = (
            '<div class="chat-bubble chat-user">'
            + content
            + f'<div class="chat-meta">{t}</div>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
    else:
        pandas_html = ""
        if msg.get("pandas_result"):
            pandas_html = (
                '<div class="pandas-result">'
                f'⚡ Pandas result: {msg["pandas_result"]}'
                '</div>'
            )

        pills = "".join(
            f'<span class="source-pill">{s}</span>'
            for s in msg.get("sources", [])
        )
        sources_html = f'<div class="source-pills">{pills}</div>' if pills else ""

        html = (
            '<div class="chat-bubble chat-assistant">'
            + content
            + pandas_html
            + sources_html
            + '<div class="chat-meta">'
            '<span style="color:#818cf8;">⬡ AInsights</span>'
            f'&nbsp;·&nbsp; {t}'
            '</div>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

def _send_chat(question: str) -> None:
    """POST question to Agent C and update chat history."""
    st.session_state.chat_history.append(
        {"role": "user", "content": question, "time": _now()}
    )

    # Show thinking indicator
    thinking = st.empty()
    thinking.markdown(
    '<div class="chat-thinking">'
    '<div class="thinking-dots"><span></span><span></span><span></span></div>'
    'Agent C is thinking …'
    '</div>',
    unsafe_allow_html=True,
)

    try:
        # Long timeout — LLM inference + Pandas can take 60-180s
        r = httpx.post(
            f"{BACKEND_URL}/chat",
            json={"question": question, "session_id": "default"},
            timeout=240.0,
        )
        r.raise_for_status()
        data = r.json()

        thinking.empty()
        st.session_state.chat_history.append({
            "role":          "assistant",
            "content":       data.get("answer", "No answer returned."),
            "pandas_result": data.get("pandas_result"),
            "sources":       data.get("sources", []),
            "time":          _now(),
        })

    except httpx.TimeoutException:
        thinking.empty()
        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": (
                "⚠ The request took longer than expected. "
                "Try a simpler or more specific question, "
                "or check that the Ollama server is running."
            ),
            "time": _now(),
        })
    except Exception as exc:
        thinking.empty()
        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": f"⚠ Error communicating with the backend: {exc}",
            "time":    _now(),
        })

    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
inject_css()
init_session()
render_sidebar()

render_hero()

st.markdown("<hr>", unsafe_allow_html=True)

tab_upload, tab_viz, tab_chat = st.tabs([
    "⬆  Upload & Clean",
    "📊  Visualize",
    "💬  Chat with Data",
])

with tab_upload:
    render_upload_tab()

with tab_viz:
    render_visualize_tab()

with tab_chat:
    render_chat_tab()