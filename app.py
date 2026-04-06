from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from core.llm_client import OllamaClient
from core.orchestrator import Orchestrator


st.set_page_config(
    page_title="AInsights",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_llm_client() -> OllamaClient:
    return OllamaClient(
        base_url="http://localhost:11434",
        model="qwen2.5:1.5b-instruct",
        timeout=300,
    )


@st.cache_data(show_spinner=False)
def load_dataframe_from_bytes(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    from utils.file_reader import read_any_file

    return read_any_file(tmp_path)


@st.cache_data(show_spinner=False)
def build_quality_report(df: pd.DataFrame) -> dict:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    object_cols = [c for c in df.columns if df[c].dtype == "object"]

    duplicates = int(df.duplicated().sum())
    missing = (df.isna().mean() * 100).round(2).to_dict()

    outliers = {}
    for col in numeric_cols[:10]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 4:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers[col] = int(((s < lower) | (s > upper)).sum())

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicates": duplicates,
        "missing_percent": missing,
        "numeric_columns": numeric_cols,
        "categorical_columns": object_cols,
        "outliers_iqr": outliers,
    }


def init_state() -> None:
    defaults = {
        "orch": None,
        "df": None,
        "raw_df": None,
        "engineer_report": None,
        "visual_output": None,
        "messages": [],
        "file_name": None,
        "quality_report": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def ensure_orchestrator() -> Orchestrator:
    if st.session_state.orch is None:
        llm = get_llm_client()
        st.session_state.orch = Orchestrator(llm)
    return st.session_state.orch


def render_dataset_overview(df: pd.DataFrame) -> None:
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicates", f"{int(df.duplicated().sum()):,}")

    st.dataframe(df.head(20), use_container_width=True)


def render_quality_report(report: dict) -> None:
    st.subheader("Data quality report")
    st.json(report)


def render_charts(visual_output) -> None:
    st.subheader("Auto visuals")
    if not visual_output or not visual_output.charts:
        st.info("No chart could be generated from the current dataset.")
        return

    for idx, chart in enumerate(visual_output.charts, start=1):
        st.altair_chart(chart, use_container_width=True)
        st.caption(f"Chart {idx}")

    if visual_output.kpis:
        st.subheader("KPIs")
        cols = st.columns(min(len(visual_output.kpis), 4))
        for i, (k, v) in enumerate(visual_output.kpis.items()):
            cols[i % len(cols)].metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))

    if visual_output.insights:
        st.subheader("Insights")
        for item in visual_output.insights:
            st.write(f"- {item}")


def export_buttons(df: pd.DataFrame, engineer_report: dict | None) -> None:
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cleaned CSV",
        data=csv_data,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

    report_payload = json.dumps(engineer_report or {}, indent=2).encode("utf-8")
    st.download_button(
        "Download insights report",
        data=report_payload,
        file_name="insights_report.json",
        mime="application/json",
    )


def main() -> None:
    init_state()
    orch = ensure_orchestrator()

    st.title("AInsights — Local Agentic BI Platform")

    with st.sidebar:
        st.header("Upload")
        uploaded = st.file_uploader(
            "File",
            type=["csv", "xlsx", "xls", "json", "txt", "pdf"],
            accept_multiple_files=False,
        )

        st.caption("Local-only execution. One LLM. No vector database.")
        st.caption("Model: qwen2.5:1.5b-instruct via Ollama")

        if st.button("Reset session"):
            for key in ["df", "raw_df", "engineer_report", "visual_output", "messages", "file_name", "quality_report"]:
                st.session_state[key] = None if key != "messages" else []
            st.rerun()

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        if st.session_state.file_name != uploaded.name:
            progress = st.progress(0, text="Loading file...")
            df = load_dataframe_from_bytes(file_bytes, uploaded.name)
            progress.progress(35, text="Cleaning with engineer agent...")
            engineer_result = orch.engineer.clean(df)
            progress.progress(70, text="Building retrieval memory...")
            orch.analyst.build_memory(engineer_result.dataframe)
            progress.progress(100, text="Ready.")

            st.session_state.raw_df = df
            st.session_state.df = engineer_result.dataframe
            st.session_state.engineer_report = engineer_result.report
            st.session_state.visual_output = orch.visualizer.build(engineer_result.dataframe)
            st.session_state.quality_report = build_quality_report(engineer_result.dataframe)
            st.session_state.file_name = uploaded.name

    if st.session_state.df is None:
        st.info("Upload a CSV, XLSX, JSON, TXT, or PDF file to start.")
        return

    df = st.session_state.df

    current_view = st.radio("View", ["Overview", "Visuals", "Chat", "Export"], horizontal=True, label_visibility="collapsed")

    if current_view == "Overview":
        render_dataset_overview(df)
        if st.session_state.quality_report:
            render_quality_report(st.session_state.quality_report)

        if st.session_state.engineer_report:
            with st.expander("Cleaning report"):
                st.json(st.session_state.engineer_report)

    elif current_view == "Visuals":
        render_charts(st.session_state.visual_output)

    elif current_view == "Chat":
        st.subheader("Chat with the dataset")
        history = st.session_state.messages

        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_query = st.chat_input("Ask about trends, causes, missing values, categories, or comparisons.")
        if user_query:
            temp_history = history + [{"role": "user", "content": user_query}]
            
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    answer, meta = orch.handle_query(df, user_query, temp_history)
                    st.markdown(answer)

                    with st.expander("Execution trace"):
                        st.json(meta)

            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": answer})
            st.session_state.messages = history

    elif current_view == "Export":
        export_buttons(df, st.session_state.engineer_report)

if __name__ == "__main__":
    main()