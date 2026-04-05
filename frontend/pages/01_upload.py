"""
Page 1 — Upload & Clean
Handles structured data files (Agent A) and documents (RAG Engine).
Polls the backend job endpoint until completion.
"""

import os
import time
from pathlib import Path

import httpx
import streamlit as st

# Inherit CSS + session from app.py
from app import inject_css, init_session, render_sidebar, BACKEND_URL

inject_css()
init_session()
render_sidebar()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _poll_job(job_id: str, placeholder: st.delta_generator.DeltaGenerator) -> dict:
    """
    Poll /jobs/{job_id} with exponential back-off.
    Updates the placeholder with live status. Returns the final job dict.
    """
    backoff = 1.5
    for _ in range(60):   # max ~90 s of polling
        r = httpx.get(f"{BACKEND_URL}/jobs/{job_id}", timeout=10)
        job = r.json()
        status  = job.get("status", "unknown")
        detail  = job.get("detail", "")

        badge_cls = {"pending": "warning", "running": "warning",
                     "complete": "success", "failed": "error"}.get(status, "idle")

        placeholder.markdown(
            f"""
            <div class="glass-card">
              <span class="badge badge-{badge_cls}">{status}</span>
              <span style="margin-left:10px; font-size:0.88rem;
                           color:#94a3b8;">{detail}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if status in ("complete", "failed"):
            return job

        time.sleep(backoff)
        backoff = min(backoff * 1.4, 8)

    return {"status": "timeout", "detail": "Job did not complete in time."}


def _fetch_schema() -> dict | None:
    try:
        r = httpx.get(f"{BACKEND_URL}/data/schema", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-title">Upload & Clean</div>
    <div class="page-subtitle">
      Upload your data file to trigger the ETL pipeline, or add documents to the knowledge base.
    </div>
    """,
    unsafe_allow_html=True,
)

tab_data, tab_docs = st.tabs(["📂 Structured Data", "📄 Documents & PDFs"])

# ── Tab 1: Structured data ────────────────────────────────────────────────────
with tab_data:
    st.markdown(
        """
        <div class="glass-card">
          <div style="font-size:0.82rem; color:#64748b; margin-bottom:0.5rem;">
            Supported formats
          </div>
          <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <span class="badge badge-info">CSV</span>
            <span class="badge badge-info">Excel (.xlsx/.xls)</span>
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
        col_name, col_size, col_btn = st.columns([3, 1, 1])
        with col_name:
            st.markdown(
                f"""<div style="color:#f1f5f9; font-size:0.9rem;
                               padding-top:0.5rem;">📎 {uploaded_data.name}</div>""",
                unsafe_allow_html=True,
            )
        with col_size:
            size_kb = round(uploaded_data.size / 1024, 1)
            st.markdown(
                f"""<div style="color:#64748b; font-size:0.82rem;
                               padding-top:0.55rem;">{size_kb} KB</div>""",
                unsafe_allow_html=True,
            )
        with col_btn:
            run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

        if run_btn:
            st.session_state.pipeline_status = "processing"
            status_placeholder = st.empty()

            with st.spinner("Uploading to Agent A …"):
                try:
                    r = httpx.post(
                        f"{BACKEND_URL}/upload/data",
                        files={"file": (uploaded_data.name, uploaded_data.getvalue())},
                        timeout=30,
                    )
                    r.raise_for_status()
                    job = r.json()
                    job_id = job["job_id"]
                    st.session_state.current_job_id = job_id

                except Exception as exc:
                    st.error(f"Upload failed: {exc}")
                    st.session_state.pipeline_status = "error"
                    st.stop()

            # Poll until done
            final_job = _poll_job(job_id, status_placeholder)

            if final_job["status"] == "complete":
                st.session_state.pipeline_status = "ready"
                schema = _fetch_schema()
                if schema:
                    st.session_state.schema_info = schema

                result = final_job.get("result", {})

                # KPI row
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                kpis = [
                    (c1, result.get("original_shape", [0])[0], "Original Rows"),
                    (c2, result.get("cleaned_shape",  [0])[0], "Clean Rows"),
                    (c3, result.get("null_cells_before", 0),   "Nulls Before"),
                    (c4, result.get("null_cells_after", 0),    "Nulls After"),
                ]
                for col, val, label in kpis:
                    with col:
                        st.markdown(
                            f"""<div class="kpi-card">
                              <div class="kpi-value">{val:,}</div>
                              <div class="kpi-label">{label}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                # Method badge
                method = result.get("cleaning_method", "unknown")
                badge = "success" if "llm" in method else "warning"
                st.markdown(
                    f"""<div style="margin-top:0.75rem;">
                      <span style="font-size:0.8rem; color:#64748b;">Cleaning method: </span>
                      <span class="badge badge-{badge}">{method}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Schema preview
                if schema:
                    st.markdown(
                        "<div style='margin-top:1.5rem; font-size:0.88rem; "
                        "color:#94a3b8; font-weight:500;'>Schema preview</div>",
                        unsafe_allow_html=True,
                    )
                    import pandas as pd
                    rows = []
                    for col_name, meta in schema.get("schema", {}).items():
                        rows.append({
                            "Column":  col_name,
                            "Type":    meta["dtype"],
                            "Nulls":   meta["nulls"],
                            "Unique":  meta["unique"],
                            "Sample":  str(meta["sample"])[:60],
                        })
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                # Download button
                st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
                col_dl, _ = st.columns([1, 3])
                with col_dl:
                    csv_bytes = httpx.get(
                        f"{BACKEND_URL}/data/cleaned", timeout=15
                    ).content
                    st.download_button(
                        "⬇ Download cleaned CSV",
                        data=csv_bytes,
                        file_name="ainsights_cleaned_data.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            else:
                st.session_state.pipeline_status = "error"
                st.error(f"Pipeline failed: {final_job.get('detail', 'Unknown error')}")

# ── Tab 2: Documents ──────────────────────────────────────────────────────────
with tab_docs:
    st.markdown(
        """
        <div class="glass-card">
          <div style="font-size:0.88rem; color:#94a3b8; line-height:1.6;">
            Upload PDFs, text files, or Markdown documents. They are chunked,
            embedded, and stored in the local vector database so Agent C can
            retrieve relevant passages when answering your questions.
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
        col_info, col_btn2 = st.columns([4, 1])
        with col_info:
            st.markdown(
                f"""<div style="color:#f1f5f9; font-size:0.9rem;
                               padding-top:0.5rem;">📎 {uploaded_doc.name}</div>""",
                unsafe_allow_html=True,
            )
        with col_btn2:
            ingest_btn = st.button("🔍 Ingest", type="primary", use_container_width=True)

        if ingest_btn:
            doc_placeholder = st.empty()
            with st.spinner("Uploading document …"):
                try:
                    r = httpx.post(
                        f"{BACKEND_URL}/upload/document",
                        files={"file": (uploaded_doc.name, uploaded_doc.getvalue())},
                        timeout=30,
                    )
                    r.raise_for_status()
                    job_id = r.json()["job_id"]
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")
                    st.stop()

            final = _poll_job(job_id, doc_placeholder)
            if final["status"] == "complete":
                st.success(f"✓ Document indexed: {final.get('detail', '')}")
            else:
                st.error(f"Ingestion failed: {final.get('detail')}")