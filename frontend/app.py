"""
AInsights — Main Application v5
Changes:
  - Chat: messages + thinking indicator always appear ABOVE the input row
  - Pipeline: stepper positioned directly below the Run Pipeline button
    with tiny descriptions beneath each stage
  - Sidebar: Ollama LLM Online/Offline status indicator (cached 30 s)
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st

st.set_page_config(
    page_title="AInsights",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
OLLAMA_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CSS_PATH    = Path(__file__).parent / "assets" / "style.css"


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def init_session() -> None:
    defaults: dict = {
        "pipeline_status":   "idle",
        "chat_history":      [],
        "schema_info":       None,
        "uploaded_files":    [],
        "chart_figures":     [],
        "custom_figures":    [],
        "current_job_id":    None,
        "chat_input_key":    0,
        "pipeline_result":   {},
        # Ollama status cache
        "_ollama_online":    False,
        "_ollama_check_ts":  0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now().strftime("%H:%M")


def _fetch_schema() -> dict | None:
    try:
        r = httpx.get(f"{BACKEND_URL}/data/schema", timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _poll_job(job_id: str) -> dict:
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


def _get_ollama_status() -> bool:
    """
    Check whether Ollama is reachable. Cached in session state for 30 s
    so the sidebar doesn't hammer the server on every Streamlit rerun.
    """
    last = st.session_state.get("_ollama_check_ts", 0.0)
    if time.time() - last < 30:
        return st.session_state.get("_ollama_online", False)
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        online = r.status_code == 200
    except Exception:
        online = False
    st.session_state["_ollama_online"]   = online
    st.session_state["_ollama_check_ts"] = time.time()
    return online


def _generate_suggestions(schema_info: dict | None) -> list[str]:
    base = ["Explain the dataset", "Find outliers"]
    if not schema_info:
        return base + ["Show summary statistics", "Main trends?", "Top 5 by value"]
    col_meta = schema_info.get("schema", {})
    num_cols = [c for c, m in col_meta.items()
                if "int" in m.get("dtype", "") or "float" in m.get("dtype", "")]
    cat_cols = [c for c, m in col_meta.items()
                if "object" in m.get("dtype", "") or "category" in m.get("dtype", "")]
    suggestions = list(base)
    if num_cols:
        suggestions.insert(1, f"Average {num_cols[0]}")
    if len(num_cols) >= 2:
        suggestions.append(f"{num_cols[0]} vs {num_cols[1]}")
    if cat_cols:
        suggestions.append(f"Group by {cat_cols[0]}")
    return suggestions[:5]


def _friendly_method(method: str) -> tuple[str, str]:
    if not method or method == "unknown":
        return "idle", "Unknown"
    m = method.lower()
    if "llm" in m:
        attempt = method.split("_")[-1] if "_" in method else "1"
        return "success", f"AI cleaning · attempt {attempt}"
    if "rule" in m or "fallback" in m:
        return "warning", "Rule-based fallback"
    return "idle", method


# ─────────────────────────────────────────────────────────────────────────────
# Horizontal stepper
# ─────────────────────────────────────────────────────────────────────────────
_STEP_LABELS = ["Upload",       "Schema",           "Clean",         "Execute",      "Embed"]
_STEP_DESCS  = ["Saving file",  "Profiling columns","LLM writes code","Sandbox run", "ChromaDB index"]


def _render_stepper(
    states:      list[str],
    placeholder: st.delta_generator.DeltaGenerator,
) -> None:
    """
    Render a horizontal progress stepper with tiny descriptions.
    states = list of state strings, one per step, length must match _STEP_LABELS.
    state ∈ { 'pending', 'active', 'done', 'error' }
    """
    icon_map = {"done": "✓", "active": "●", "pending": "·", "error": "✕"}
    inner = ""

    for i, state in enumerate(states):
        label = _STEP_LABELS[i] if i < len(_STEP_LABELS) else f"Step {i+1}"
        desc  = _STEP_DESCS[i]  if i < len(_STEP_DESCS)  else ""
        icon  = icon_map.get(state, str(i + 1))

        inner += (
            f'<div style="display:flex;flex-direction:column;align-items:center;flex:1;min-width:0;">'
            f'<div class="hstep-circle {state}">{icon}</div>'
            f'<div class="hstep-label {state}">{label}</div>'
            f'<div class="hstep-desc {state}">{desc}</div>'
            f'</div>'
        )

        if i < len(states) - 1:
            # Connector takes the state of the left step
            conn = "done" if state == "done" else ("active" if state == "active" else "pending")
            inner += f'<div class="hstep-connector {conn}"></div>'

    placeholder.markdown(
        '<div class="glass-card" style="padding:0.85rem 1.5rem 1rem;margin-top:0.75rem;">'
        '<div style="font-size:0.67rem;color:#64748b;text-transform:uppercase;'
        'letter-spacing:0.09em;margin-bottom:0.85rem;">Pipeline progress</div>'
        '<div style="display:flex;align-items:flex-start;">'
        + inner +
        '</div></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        # Logo
        st.markdown(
            '<div style="padding:1rem 0 1.25rem;">'
            '<div style="display:flex;align-items:center;gap:10px;">'
            '<div style="width:36px;height:36px;background:linear-gradient(135deg,#6366f1,#34d399);'
            'border-radius:10px;display:flex;align-items:center;justify-content:center;'
            'font-size:18px;flex-shrink:0;">⬡</div>'
            '<div>'
            '<div style="font-size:1.15rem;font-weight:700;letter-spacing:-0.03em;color:#f1f5f9;">'
            'AInsights</div>'
            '<div style="font-size:0.68rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;">'
            'Local-first · 100% private</div>'
            '</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='margin:0 0 1rem;'>", unsafe_allow_html=True)

        # ── LLM status ──────────────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.7rem;color:#475569;text-transform:uppercase;'
            'letter-spacing:0.09em;margin-bottom:7px;">AI model</div>',
            unsafe_allow_html=True,
        )
        ollama_online = _get_ollama_status()
        dot_cls   = "online"  if ollama_online else "offline"
        txt_cls   = "online"  if ollama_online else "offline"
        txt_label = "Online"  if ollama_online else "Offline"
        model_name = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

        st.markdown(
            f'<div class="llm-status">'
            f'<div class="llm-status-dot {dot_cls}"></div>'
            f'<span class="llm-status-text {txt_cls}">{txt_label}</span>'
            f'<span class="llm-status-model">{model_name}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Refresh LLM status button
        if st.button("↻ Refresh status", use_container_width=True, key="refresh_llm"):
            st.session_state["_ollama_check_ts"] = 0.0   # force re-check
            st.rerun()

        st.markdown("<hr style='margin:0.75rem 0;'>", unsafe_allow_html=True)

        # ── Pipeline status ──────────────────────────────────────────────────
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

        # ── Download cleaned CSV ─────────────────────────────────────────────
        if st.session_state.pipeline_status == "ready":
            st.markdown(
                '<div style="font-size:0.7rem;color:#475569;text-transform:uppercase;'
                'letter-spacing:0.09em;margin-bottom:8px;">Export</div>',
                unsafe_allow_html=True,
            )
            try:
                csv_bytes = httpx.get(f"{BACKEND_URL}/data/cleaned", timeout=20).content
                st.download_button(
                    "⬇ Download cleaned CSV",
                    data=csv_bytes,
                    file_name="ainsights_cleaned_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="sidebar_download",
                )
            except Exception:
                st.caption("CSV unavailable")
            st.markdown("<hr style='margin:0.75rem 0;'>", unsafe_allow_html=True)

        # ── Uploaded files ───────────────────────────────────────────────────
        uploaded_files = st.session_state.get("uploaded_files", [])
        if uploaded_files:
            st.markdown(
                '<div style="font-size:0.7rem;color:#475569;text-transform:uppercase;'
                'letter-spacing:0.09em;margin-bottom:8px;">Loaded datasets</div>',
                unsafe_allow_html=True,
            )
            for f in uploaded_files[-3:]:
                ext_icon = {
                    ".csv": "📊", ".xlsx": "📗", ".xls": "📗",
                    ".json": "📋", ".xml": "📄", ".pdf": "📕",
                }.get(Path(f["name"]).suffix.lower(), "📁")
                st.markdown(
                    f'<div class="sidebar-file-item">'
                    f'<span class="sidebar-file-icon">{ext_icon}</span>'
                    f'<div style="min-width:0;">'
                    f'<div class="sidebar-file-name">{f["name"]}</div>'
                    f'<div class="sidebar-file-meta">{f.get("size_kb","?")} KB</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        # ── Dataset stats ────────────────────────────────────────────────────
        schema = st.session_state.get("schema_info")
        if schema:
            st.markdown(
                '<div style="font-size:0.7rem;color:#475569;text-transform:uppercase;'
                'letter-spacing:0.09em;margin:1rem 0 8px;">Dataset info</div>',
                unsafe_allow_html=True,
            )
            rows    = schema.get("rows", "—")
            cols    = schema.get("columns", "—")
            col_map = schema.get("schema", {})
            type_counts: dict[str, int] = {}
            for meta in col_map.values():
                t = meta.get("dtype", "object")
                key = ("numeric"  if ("int" in t or "float" in t) else
                       "datetime" if "datetime" in t else "text")
                type_counts[key] = type_counts.get(key, 0) + 1

            stat_rows = [
                ("Rows",     f"{rows:,}" if isinstance(rows, int) else rows),
                ("Columns",  cols),
                ("Numeric",  type_counts.get("numeric",  0)),
                ("Text",     type_counts.get("text",     0)),
                ("Datetime", type_counts.get("datetime", 0)),
            ]
            stats_html = "".join(
                f'<div class="sidebar-stat-row">'
                f'<span class="sidebar-stat-key">{k}</span>'
                f'<span class="sidebar-stat-value">{v}</span>'
                f'</div>'
                for k, v in stat_rows
            )
            st.markdown(
                '<div style="border:1px solid var(--glass-border);border-radius:var(--radius-md);'
                f'padding:0.6rem 0.85rem;background:var(--glass-highlight);">{stats_html}</div>',
                unsafe_allow_html=True,
            )

            if col_map:
                st.markdown(
                    '<div style="font-size:0.7rem;color:#475569;text-transform:uppercase;'
                    'letter-spacing:0.09em;margin:1rem 0 6px;">Columns</div>',
                    unsafe_allow_html=True,
                )
                for col_name, meta in list(col_map.items())[:8]:
                    dtype = meta.get("dtype", "")
                    color = ("#818cf8" if ("int" in dtype or "float" in dtype) else
                             "#34d399" if "datetime" in dtype else "#94a3b8")
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'align-items:center;padding:3px 0;font-size:0.78rem;'
                        f'border-bottom:1px solid rgba(30,41,59,0.5);">'
                        f'<span style="color:#e2e8f0;overflow:hidden;text-overflow:ellipsis;'
                        f'white-space:nowrap;max-width:65%;">{col_name}</span>'
                        f'<span style="color:{color};font-size:0.68rem;'
                        f'font-family:\'JetBrains Mono\',monospace;">{dtype}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                if len(col_map) > 8:
                    st.markdown(
                        f'<div style="font-size:0.72rem;color:#475569;padding-top:5px;">'
                        f'+{len(col_map)-8} more columns</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.68rem;color:#334155;padding-bottom:0.5rem;">'
            'AInsights v1.0 &nbsp;·&nbsp; ChromaDB</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab: About
# ─────────────────────────────────────────────────────────────────────────────
def render_about_tab() -> None:
    st.markdown(
        '<div class="hero-wrapper">'
        '<div class="hero-logo">'
        '<div class="hero-logo-icon">⬡</div>'
        '<div class="hero-logo-text">AInsights</div>'
        '</div>'
        '<div class="hero-tagline">'
        'A privacy-first, local-first Business Intelligence platform powered by on-device AI. '
        'Upload your data, visualise it instantly, and query it in plain English — '
        'entirely offline, with no data ever leaving your machine.'
        '</div>'
        '<div class="hero-badges">'
        '<span class="badge badge-info">100% Local</span>'
        '<span class="badge badge-success">Zero Cloud</span>'
        '<span class="badge badge-pink">RAG-Powered</span>'
        '<span class="badge badge-warning">qwen2.5-coder:7b</span>'
        '<span class="badge badge-idle">ChromaDB</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size:0.7rem;color:#475569;text-transform:uppercase;'
        'letter-spacing:0.11em;margin:0.5rem 0 1rem;text-align:center;">'
        'Three-agent pipeline architecture'
        '</div>',
        unsafe_allow_html=True,
    )

    agents = [
        (
            "agent-icon-a", "🔧", "Agent A", "Data Engineer",
            "Autonomous ETL pipeline that cleans and standardises your data using LLM-generated code.",
            ["CSV, Excel, JSON, XML and PDF table support",
             "LLM cleaning with rule-based fallback",
             "Zero tolerance for column loss"],
            "#fb923c",
        ),
        (
            "agent-icon-b", "📊", "Agent B", "Visualizer",
            "Reads the cleaned dataset and auto-selects the most appropriate Plotly charts.",
            ["Detects numeric, categorical and datetime columns",
             "Renders heatmaps, time series, box plots and more",
             "Accepts custom on-demand chart requests"],
            "#34d399",
        ),
        (
            "agent-icon-c", "💬", "Agent C", "Analyst",
            "Conversational BI using ChromaDB retrieval combined with live Pandas calculations.",
            ["RAG retrieval over documents and tabular rows",
             "Live Pandas execution for precise numbers",
             "Grounded answers with data citations"],
            "#818cf8",
        ),
        (
            "agent-icon-r", "🧠", "RAG Engine", "Memory Pipeline",
            "Embeds documents and CSV rows into local ChromaDB using all-MiniLM-L6-v2.",
            ["Chunks and embeds PDF, TXT and Markdown files",
             "Converts rows into searchable semantic sentences",
             "Fully offline — no external API calls"],
            "#f472b6",
        ),
    ]

    cols = st.columns(4, gap="medium")
    for col, (icon_cls, icon, label, name, desc, bullets, color) in zip(cols, agents):
        bullet_html = "".join(
            f'<div class="agent-feature">'
            f'<div class="agent-feature-dot" style="background:{color};"></div>'
            f'<span>{b}</span>'
            f'</div>'
            for b in bullets
        )
        with col:
            st.markdown(
                f'<div class="agent-card">'
                f'<div class="agent-icon {icon_cls}">{icon}</div>'
                f'<div class="agent-label">{label}</div>'
                f'<div class="agent-name">{name}</div>'
                f'<div class="agent-desc">{desc}</div>'
                f'{bullet_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="glass-card" style="padding:1.5rem 2rem;">'
        '<div style="font-size:1rem;font-weight:600;color:#f1f5f9;margin-bottom:1rem;">⚡ Quick Start</div>'
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;">'
        '<div style="text-align:center;">'
        '<div style="font-size:1.5rem;margin-bottom:0.4rem;">1</div>'
        '<div style="font-size:0.85rem;color:#94a3b8;font-weight:500;">Upload Data</div>'
        '<div style="font-size:0.78rem;color:#64748b;margin-top:4px;">'
        'Drop any CSV, Excel, JSON or PDF</div></div>'
        '<div style="text-align:center;">'
        '<div style="font-size:1.5rem;margin-bottom:0.4rem;">2</div>'
        '<div style="font-size:0.85rem;color:#94a3b8;font-weight:500;">Auto Clean</div>'
        '<div style="font-size:0.78rem;color:#64748b;margin-top:4px;">'
        'Agent A cleans your data automatically</div></div>'
        '<div style="text-align:center;">'
        '<div style="font-size:1.5rem;margin-bottom:0.4rem;">3</div>'
        '<div style="font-size:0.85rem;color:#94a3b8;font-weight:500;">Visualize</div>'
        '<div style="font-size:0.78rem;color:#64748b;margin-top:4px;">'
        'Agent B renders charts automatically</div></div>'
        '<div style="text-align:center;">'
        '<div style="font-size:1.5rem;margin-bottom:0.4rem;">4</div>'
        '<div style="font-size:0.85rem;color:#94a3b8;font-weight:500;">Chat</div>'
        '<div style="font-size:0.78rem;color:#64748b;margin-top:4px;">'
        'Ask Agent C in plain English</div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Upload & Clean
# ─────────────────────────────────────────────────────────────────────────────
def render_upload_tab() -> None:
    st.markdown(
        '<div class="section-header">'
        '<div class="section-icon agent-icon-a">🔧</div>'
        '<div>'
        '<div class="section-title">Upload & Clean</div>'
        '<div class="section-subtitle">Agent A profiles your schema and cleans your data automatically</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    inner_data, inner_docs = st.tabs(["📂 Structured Data", "📄 Documents"])

    with inner_data:
        st.markdown(
            '<div class="glass-card" style="padding:1rem 1.25rem;">'
            '<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">'
            '<span style="font-size:0.78rem;color:#64748b;margin-right:4px;">Supported:</span>'
            '<span class="badge badge-info">CSV</span>'
            '<span class="badge badge-info">Excel (.xlsx/.xls)</span>'
            '<span class="badge badge-info">JSON</span>'
            '<span class="badge badge-info">XML</span>'
            '<span class="badge badge-info">PDF tables</span>'
            '</div></div>',
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
                    '<div style="display:flex;align-items:center;gap:12px;padding:0.5rem 0;">'
                    '<span style="font-size:1.2rem;">📎</span>'
                    '<div>'
                    f'<div style="color:#f1f5f9;font-size:0.9rem;font-weight:500;">{uploaded_data.name}</div>'
                    f'<div style="color:#64748b;font-size:0.78rem;">{size_kb} KB</div>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )
            with col_btn:
                run_clicked = st.button(
                    "▶ Run Pipeline",
                    type="primary",
                    use_container_width=True,
                    key="run_pipeline_btn",
                )

            # ── Stepper placeholder directly below the button row ───────────
            stepper_placeholder = st.empty()

            if run_clicked:
                _run_pipeline(uploaded_data, stepper_placeholder)

        # ── Results — rendered on rerun after pipeline completes ────────────
        # Lives outside the `if uploaded_data` block so it always renders
        # after the file widget and stepper, never duplicated.
        if (
            st.session_state.pipeline_status == "ready"
            and st.session_state.get("pipeline_result")
        ):
            _show_pipeline_results(
                st.session_state.pipeline_result,
                st.session_state.schema_info,
            )

    with inner_docs:
        st.markdown(
            '<div class="glass-card">'
            '<div style="font-size:0.88rem;color:#94a3b8;line-height:1.65;">'
            'Upload PDFs, text files, or Markdown documents to enrich Agent C\'s knowledge base. '
            'Documents are chunked, embedded with '
            '<code style="color:#818cf8;font-family:JetBrains Mono,monospace;font-size:0.82rem;">'
            'all-MiniLM-L6-v2</code>, and stored in ChromaDB.'
            '</div></div>',
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
                    f'<div style="color:#f1f5f9;font-size:0.9rem;font-weight:500;padding:0.5rem 0;">'
                    f'📎 {uploaded_doc.name}</div>',
                    unsafe_allow_html=True,
                )
            with col_b:
                if st.button("🔍 Ingest", type="primary",
                             use_container_width=True, key="ingest_doc_btn"):
                    _ingest_document(uploaded_doc)


def _run_pipeline(
    uploaded_file,
    stepper_placeholder: st.delta_generator.DeltaGenerator,
) -> None:
    """
    Full ETL pipeline with live horizontal stepper.
    Stepper renders in the placeholder that sits directly below the Run button.
    Calls st.rerun() on completion — results rendered by the caller's block (no duplication).
    """
    st.session_state.pipeline_status = "processing"
    st.session_state.pipeline_result = {}

    # Initial state: first step active, rest pending
    states = ["active", "pending", "pending", "pending", "pending"]
    _render_stepper(states, stepper_placeholder)

    # ── Upload ────────────────────────────────────────────────────────────
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
        stepper_placeholder.empty()
        st.error(f"Upload failed: {exc}")
        return

    states[0] = "done"
    states[1] = "active"
    _render_stepper(states, stepper_placeholder)

    # ── Poll with live stepper updates ───────────────────────────────────
    backoff = 1.5
    for _ in range(90):
        try:
            r   = httpx.get(f"{BACKEND_URL}/jobs/{job_id}", timeout=8)
            job = r.json()
            s   = job.get("status", "")
            d   = job.get("detail", "").lower()

            if "profiling" in d or "schema" in d:
                states[1] = "active"
                states[2] = "pending"

            elif "llm" in d or "cleaning" in d or "agent a" in d:
                states[1] = "done"
                states[2] = "active"
                states[3] = "active"

            elif "rag" in d or "ingest" in d or "embed" in d:
                states[1] = "done"
                states[2] = "done"
                states[3] = "done"
                states[4] = "active"

            _render_stepper(states, stepper_placeholder)

            if s == "complete":
                states = ["done", "done", "done", "done", "done"]
                _render_stepper(states, stepper_placeholder)

                result  = job.get("result", {})
                size_kb = round(uploaded_file.size / 1024, 1)

                st.session_state.pipeline_result = result
                files = st.session_state.get("uploaded_files", [])
                files.append({
                    "name":    uploaded_file.name,
                    "size_kb": size_kb,
                    "rows":    result.get("cleaned_shape", [0])[0],
                    "columns": (result.get("cleaned_shape", [0, 0])[1]
                                if len(result.get("cleaned_shape", [])) > 1 else 0),
                    "method":  result.get("cleaning_method", "unknown"),
                })
                st.session_state.uploaded_files = files[-5:]

                schema = _fetch_schema()
                if schema:
                    st.session_state.schema_info = schema

                st.session_state.pipeline_status = "ready"
                time.sleep(0.6)   # let user see all-done state briefly
                st.rerun()
                return

            if s == "failed":
                states = ["done" if st == "done" else "error"
                          for st in states]
                _render_stepper(states, stepper_placeholder)
                st.session_state.pipeline_status = "error"
                st.error(f"Pipeline failed: {job.get('detail')}")
                return

        except Exception:
            pass

        time.sleep(backoff)
        backoff = min(backoff * 1.3, 6)

    st.session_state.pipeline_status = "error"
    stepper_placeholder.empty()
    st.error("Pipeline timed out. Check backend logs.")


def _show_pipeline_results(result: dict, schema: dict | None) -> None:
    if not result:
        return
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, result.get("original_shape",    [0])[0], "Original rows"),
        (c2, result.get("cleaned_shape",     [0])[0], "Cleaned rows"),
        (c3, result.get("null_cells_before", 0),      "Nulls before"),
        (c4, result.get("null_cells_after",  0),      "Nulls after"),
    ]
    for col, val, label in kpis:
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value">{val:,}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    method_raw            = result.get("cleaning_method", "")
    elapsed               = result.get("elapsed_seconds", 0)
    badge_cls, method_lbl = _friendly_method(method_raw)
    st.markdown(
        '<div style="margin-top:0.85rem;display:flex;gap:12px;align-items:center;flex-wrap:wrap;">'
        f'<span class="badge badge-{badge_cls}">Cleaning: {method_lbl}</span>'
        f'<span style="font-size:0.8rem;color:#64748b;">Completed in {elapsed}s</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if schema:
        st.markdown(
            '<div style="margin-top:1.5rem;margin-bottom:0.5rem;'
            'font-size:0.88rem;color:#94a3b8;font-weight:500;">Schema preview</div>',
            unsafe_allow_html=True,
        )
        import pandas as pd
        rows = [
            {
                "Column": col_name,
                "Type":   meta["dtype"],
                "Nulls":  meta["nulls"],
                "Unique": meta["unique"],
                "Sample": str(meta["sample"])[:55],
            }
            for col_name, meta in schema.get("schema", {}).items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.success("✓ Pipeline complete — open Visualizations or Chat to explore your data.")


def _ingest_document(uploaded_doc) -> None:
    with st.status("Ingesting document …", expanded=True) as s:
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
            s.update(label="Upload failed", state="error")
            st.error(str(exc))
            return
        st.write("Chunking and embedding …")
        final = _poll_job(job_id)
        if final.get("status") == "complete":
            s.update(label="Document indexed", state="complete")
            st.success(f"✓ {final.get('detail','Indexed.')}")
        else:
            s.update(label="Failed", state="error")
            st.error(final.get("detail", "Unknown error"))


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Visualizations
# ─────────────────────────────────────────────────────────────────────────────
def render_visualize_tab() -> None:
    st.markdown(
        '<div class="section-header">'
        '<div class="section-icon agent-icon-b">📊</div>'
        '<div>'
        '<div class="section-title">Visualizations</div>'
        '<div class="section-subtitle">Auto-generated charts plus on-demand custom chart requests</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.pipeline_status != "ready":
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:3.5rem 2rem;">'
            '<div style="font-size:2.5rem;margin-bottom:1rem;">📊</div>'
            '<div style="font-size:1rem;color:#94a3b8;font-weight:500;margin-bottom:0.4rem;">'
            'No data loaded</div>'
            '<div style="font-size:0.84rem;color:#64748b;">Upload a dataset in the '
            '<strong style="color:#818cf8;">Upload Data</strong> tab first.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # Auto charts
    st.markdown(
        '<div style="font-size:0.82rem;font-weight:600;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.75rem;">'
        '⬡ Auto-generated charts</div>',
        unsafe_allow_html=True,
    )
    col_btn, col_info = st.columns([1, 5])
    with col_btn:
        refresh = st.button("⟳ Generate", type="primary",
                            use_container_width=True, key="refresh_charts")
    with col_info:
        if st.session_state.schema_info:
            info    = st.session_state.schema_info
            rows    = info.get("rows", "?")
            cols    = info.get("columns", "?")
            row_str = f"{rows:,}" if isinstance(rows, int) else str(rows)
            st.markdown(
                f'<div style="padding-top:0.55rem;font-size:0.82rem;color:#64748b;">'
                f'Analysing {row_str} rows × {cols} columns</div>',
                unsafe_allow_html=True,
            )

    if refresh or not st.session_state.get("chart_figures"):
        with st.status("Agent B is generating charts …", expanded=True) as s:
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

    _render_chart_grid(st.session_state.get("chart_figures", []))

    # Custom chart
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.82rem;font-weight:600;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.75rem;">'
        '✏ Custom chart request</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="glass-card" style="padding:1rem 1.25rem;">'
        '<div style="font-size:0.84rem;color:#94a3b8;line-height:1.6;">'
        'Describe any chart you want. Agent B will generate it using your dataset. '
        'Be as specific as you like — column names, chart type, grouping, colour.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    custom_request = st.text_input(
        "Describe your chart …",
        placeholder="e.g. Histogram of Price with 20 bins, coloured by Bedrooms",
        key="custom_chart_input",
        label_visibility="collapsed",
    )
    col_gen, col_clear = st.columns([1, 5])
    with col_gen:
        gen_btn = st.button("⚡ Generate chart", type="primary",
                            use_container_width=True, key="gen_custom_chart")
    with col_clear:
        if st.button("✕ Clear custom", use_container_width=True, key="clear_custom"):
            st.session_state.custom_figures = []
            st.rerun()

    if gen_btn and custom_request.strip():
        with st.spinner(f'Generating: "{custom_request}" …'):
            try:
                r = httpx.post(
                    f"{BACKEND_URL}/visualize/custom",
                    json={"request": custom_request.strip()},
                    timeout=180,
                )
                r.raise_for_status()
                data = r.json()
                if data.get("chart"):
                    existing = st.session_state.get("custom_figures", [])
                    existing.insert(0, {
                        "figure":  data["chart"],
                        "request": custom_request.strip(),
                    })
                    st.session_state.custom_figures = existing[:6]
                    st.rerun()
                else:
                    st.warning(data.get("error",
                               "The model could not generate a chart for that request."))
            except Exception as exc:
                st.error(f"Custom chart error: {exc}")

    custom_figs = st.session_state.get("custom_figures", [])
    if custom_figs:
        import plotly.graph_objects as go
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        for item in custom_figs:
            fig = go.Figure(item["figure"])
            st.markdown(
                f'<div class="chart-card">'
                f'<div class="chart-card-header">'
                f'<div class="chart-card-title">Custom: {item["request"]}</div>'
                f'<div class="chart-card-subtitle">Generated on demand by Agent B</div>'
                f'</div><div class="chart-card-body">',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": True, "displaylogo": False})
            st.markdown("</div></div>", unsafe_allow_html=True)


def _render_chart_grid(figures: list[dict]) -> None:
    if not figures:
        st.info("No charts generated yet. Click ⟳ Generate.")
        return

    import plotly.graph_objects as go

    summary = [f for f in figures
               if "Descriptive Statistics" in
               f.get("layout", {}).get("title", {}).get("text", "")]
    charts  = [f for f in figures if f not in summary]
    schema  = st.session_state.schema_info or {}
    n_rows  = schema.get("rows", "")

    st.markdown(
        f'<div style="font-size:0.78rem;color:#64748b;margin-bottom:1.25rem;">'
        f'{len(figures)} chart{"s" if len(figures)!=1 else ""} generated</div>',
        unsafe_allow_html=True,
    )

    for i in range(0, len(charts), 2):
        cols = st.columns(2, gap="medium")
        for j, col in enumerate(cols):
            if i + j < len(charts):
                fd       = charts[i + j]
                fig      = go.Figure(fd)
                title    = fd.get("layout", {}).get("title", {}).get("text", f"Chart {i+j+1}")
                subtitle = _chart_subtitle(fd, n_rows)
                with col:
                    st.markdown(
                        f'<div class="chart-card">'
                        f'<div class="chart-card-header">'
                        f'<div class="chart-card-title">{title}</div>'
                        f'<div class="chart-card-subtitle">{subtitle}</div>'
                        f'</div><div class="chart-card-body">',
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": True, "displaylogo": False,
                                            "modeBarButtonsToRemove": ["toImage"]})
                    st.markdown("</div></div>", unsafe_allow_html=True)

    for fd in summary:
        fig      = go.Figure(fd)
        n_cols   = schema.get("columns", "")
        subtitle = f"Descriptive statistics across {n_cols} columns" if n_cols else ""
        st.markdown(
            f'<div class="chart-card">'
            f'<div class="chart-card-header">'
            f'<div class="chart-card-title">Descriptive Statistics</div>'
            f'<div class="chart-card-subtitle">{subtitle}</div>'
            f'</div><div class="chart-card-body">',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.markdown("</div></div>", unsafe_allow_html=True)


def _chart_subtitle(fig_dict: dict, n_rows: int | str) -> str:
    title   = fig_dict.get("layout", {}).get("title", {}).get("text", "").lower()
    row_str = f"{n_rows:,}" if isinstance(n_rows, int) else str(n_rows)
    if "correlation" in title:     return f"Pearson r matrix · {row_str} rows"
    if "over time"   in title:     return f"Trend over time · {row_str} data points"
    if "distribution of" in title: return f"Frequency distribution · {row_str} rows"
    if "average"     in title:     return f"Mean by category · {row_str} rows"
    if " vs "        in title:     return "Scatter with OLS trendline · up to 2,000 sampled points"
    if "distribution by" in title: return f"Quartile box plot · {row_str} rows"
    return f"{row_str} rows analysed"


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Chat
# Chat messages and the "thinking" indicator always appear ABOVE the input row.
# This is achieved by creating msg_container first (Streamlit renders containers
# in creation order) then input_container, then filling them in any order.
# ─────────────────────────────────────────────────────────────────────────────
def render_chat_tab() -> None:
    st.markdown(
        '<div class="section-header">'
        '<div class="section-icon agent-icon-c">💬</div>'
        '<div>'
        '<div class="section-title">Chat with Data</div>'
        '<div class="section-subtitle">'
        'Agent C retrieves context and runs live Pandas calculations to ground every answer'
        '</div></div></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.pipeline_status != "ready":
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:3.5rem 2rem;">'
            '<div style="font-size:2.5rem;margin-bottom:1rem;">💬</div>'
            '<div style="font-size:1rem;color:#94a3b8;font-weight:500;margin-bottom:0.4rem;">'
            'No data loaded</div>'
            '<div style="font-size:0.84rem;color:#64748b;">Upload a dataset in the '
            '<strong style="color:#818cf8;">Upload Data</strong> tab first.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Create containers in display order ────────────────────────────────
    # msg_container is created FIRST → always renders above input_container
    msg_container   = st.container()
    input_container = st.container()

    # ── Fill messages container ───────────────────────────────────────────
    with msg_container:
        history = st.session_state.chat_history

        # Suggestions only when conversation is empty
        if not history:
            suggestions = _generate_suggestions(st.session_state.schema_info)
            st.markdown(
                '<div style="font-size:0.75rem;color:#475569;text-transform:uppercase;'
                'letter-spacing:0.09em;margin-bottom:0.75rem;">Suggested questions</div>',
                unsafe_allow_html=True,
            )
            sug_cols = st.columns(len(suggestions))
            for col, sug in zip(sug_cols, suggestions):
                with col:
                    if st.button(sug, use_container_width=True, key=f"sug_{hash(sug)}"):
                        st.session_state._pending_q = sug
                        st.rerun()

        # Render existing messages
        for msg in history:
            _render_chat_message(msg)

        # Reserve a slot for the thinking indicator + live response
        # This slot lives INSIDE msg_container → always above the input
        thinking_slot = st.empty()

    # ── Fill input container ──────────────────────────────────────────────
    with input_container:
        input_key = f"chat_input_{st.session_state.chat_input_key}"
        col_input, col_send, col_clear = st.columns([8, 1, 1])

        with col_input:
            question = st.text_input(
                "Ask …",
                key=input_key,
                label_visibility="collapsed",
                placeholder="e.g. What is the average price by location?",
            )
        with col_send:
            send = st.button("➤", type="primary", use_container_width=True, key="send_btn")
        with col_clear:
            if st.button("🗑", use_container_width=True, key="clear_chat",
                         help="Clear conversation"):
                st.session_state.chat_history   = []
                st.session_state.chat_input_key += 1
                st.rerun()

        # Handle pending suggestion tap
        pending = st.session_state.pop("_pending_q", None)
        if pending:
            question = pending
            send     = True

        if send and question and question.strip():
            q = question.strip()

            # Increment key so textbox clears on rerun
            st.session_state.chat_input_key += 1

            # Add user message to history
            st.session_state.chat_history.append(
                {"role": "user", "content": q, "time": _now()}
            )

            # Show thinking indicator in msg_container (above the input)
            thinking_slot.markdown(
                '<div class="chat-thinking">'
                '<div class="thinking-dots">'
                '<span></span><span></span><span></span>'
                '</div>'
                'Agent C is thinking …'
                '</div>',
                unsafe_allow_html=True,
            )

            # Blocking API call — browser shows indicator during this wait
            try:
                r = httpx.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": q, "session_id": "default"},
                    timeout=240.0,
                )
                r.raise_for_status()
                data = r.json()
                thinking_slot.empty()
                st.session_state.chat_history.append({
                    "role":          "assistant",
                    "content":       data.get("answer", "No answer returned."),
                    "pandas_result": data.get("pandas_result"),
                    "sources":       data.get("sources", []),
                    "time":          _now(),
                })
            except httpx.TimeoutException:
                thinking_slot.empty()
                st.session_state.chat_history.append({
                    "role":    "assistant",
                    "content": (
                        "⚠ The request timed out. Try a simpler question "
                        "or check that the Ollama server is running."
                    ),
                    "time": _now(),
                })
            except Exception as exc:
                thinking_slot.empty()
                st.session_state.chat_history.append({
                    "role":    "assistant",
                    "content": f"⚠ Backend error: {exc}",
                    "time":    _now(),
                })

            # Rerun renders the new message in msg_container, above the input
            st.rerun()

        # Export chat
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
        st.markdown(
            '<div class="chat-bubble chat-user">'
            + content
            + f'<div class="chat-meta">{t}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
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
        st.markdown(
            '<div class="chat-bubble chat-assistant">'
            + content
            + pandas_html
            + sources_html
            + '<div class="chat-meta">'
            '<span style="color:#818cf8;">⬡ AInsights</span>'
            f'&nbsp;·&nbsp; {t}'
            '</div></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main — tab order: About → Upload → Visualize → Chat
# ─────────────────────────────────────────────────────────────────────────────
inject_css()
init_session()
render_sidebar()

tab_about, tab_upload, tab_viz, tab_chat = st.tabs([
    "⬡  About AInsights",
    "⬆  Upload Data",
    "📊  Visualizations",
    "💬  Chat",
])

with tab_about:
    render_about_tab()

with tab_upload:
    render_upload_tab()

with tab_viz:
    render_visualize_tab()

with tab_chat:
    render_chat_tab()