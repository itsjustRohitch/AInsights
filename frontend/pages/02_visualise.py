"""
Page 2 — Visualize
Calls Agent B's /visualize endpoint and renders the Plotly charts.
"""

import os

import httpx
import plotly.io as pio
import streamlit as st
from plotly import graph_objects as go

from app import inject_css, init_session, render_sidebar, BACKEND_URL

inject_css()
init_session()
render_sidebar()

st.markdown(
    """
    <div class="page-title">Visualize</div>
    <div class="page-subtitle">
      Agent B analyses your data's column types and auto-selects the most appropriate charts.
    </div>
    """,
    unsafe_allow_html=True,
)

# Guard: require data to be loaded
if st.session_state.get("pipeline_status") not in ("ready",):
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding:3rem;">
          <div style="font-size:2rem; margin-bottom:1rem;">📂</div>
          <div style="color:#64748b; font-size:0.92rem;">
            No data loaded yet. Upload a file on the
            <strong style="color:#818cf8;">Upload &amp; Clean</strong> page first.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ── Controls ──────────────────────────────────────────────────────────────────
col_btn, col_info = st.columns([1, 4])
with col_btn:
    refresh = st.button("⟳ Generate Charts", type="primary", use_container_width=True)

with col_info:
    if st.session_state.schema_info:
        info = st.session_state.schema_info
        st.markdown(
            f"""<div style="padding-top:0.5rem; font-size:0.82rem; color:#64748b;">
              Analysing {info.get('rows','?')} rows × {info.get('columns','?')} columns
            </div>""",
            unsafe_allow_html=True,
        )


# ── Fetch and render charts ───────────────────────────────────────────────────
if refresh or "chart_figures" not in st.session_state:
    with st.spinner("Agent B is selecting and rendering charts …"):
        try:
            r = httpx.get(f"{BACKEND_URL}/visualize", timeout=90)
            r.raise_for_status()
            st.session_state.chart_figures = r.json().get("charts", [])
        except Exception as exc:
            st.error(f"Visualisation failed: {exc}")
            st.stop()

figures = st.session_state.get("chart_figures", [])

if not figures:
    st.info("No charts could be generated. Ensure your data has numeric or categorical columns.")
    st.stop()

# Render charts in a 2-column masonry-style grid
st.markdown(
    f"""<div style="font-size:0.82rem; color:#64748b;
                    margin-bottom:1rem;">{len(figures)} charts generated</div>""",
    unsafe_allow_html=True,
)

# Summary table renders full-width; other charts go in a 2-col grid
summary_figs = [f for f in figures if "Descriptive Statistics" in
                f.get("layout", {}).get("title", {}).get("text", "")]
chart_figs   = [f for f in figures if f not in summary_figs]

# 2-column chart grid
for i in range(0, len(chart_figs), 2):
    cols = st.columns(2, gap="medium")
    for j, col in enumerate(cols):
        if i + j < len(chart_figs):
            fig_dict = chart_figs[i + j]
            fig      = go.Figure(fig_dict)
            title    = fig_dict.get("layout", {}).get("title", {}).get("text", f"Chart {i+j+1}")
            with col:
                st.markdown(
                    f"""<div class="glass-card" style="padding:0.5rem;">
                      <div style="font-size:0.78rem; color:#475569;
                                  text-transform:uppercase; letter-spacing:0.06em;
                                  padding:0.5rem 0.5rem 0;">
                        {title}
                      </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": True, "displaylogo": False},
                )

# Full-width summary table
for fig_dict in summary_figs:
    fig = go.Figure(fig_dict)
    st.markdown(
        """<div class="glass-card" style="padding:0.5rem;">
          <div style="font-size:0.78rem; color:#475569;
                      text-transform:uppercase; letter-spacing:0.06em;
                      padding:0.5rem 0.5rem 0;">
            Descriptive Statistics
          </div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})