"""
Page 3 — Chat with Data
Conversational interface for Agent C (dual-context analyst).
Supports both REST polling and WebSocket streaming.
"""

import asyncio
import json
import os
import time
from datetime import datetime

import httpx
import streamlit as st

from app import inject_css, init_session, render_sidebar, BACKEND_URL

inject_css()
init_session()
render_sidebar()

WS_URL = BACKEND_URL.replace("http://", "ws://").replace("https://", "wss://")

st.markdown(
    """
    <div class="page-title">Chat with Data</div>
    <div class="page-subtitle">
      Ask questions in plain English. Agent C retrieves vector context
      and runs live Pandas calculations to ground every answer.
    </div>
    """,
    unsafe_allow_html=True,
)

# Guard
if st.session_state.get("pipeline_status") not in ("ready",):
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding:3rem;">
          <div style="font-size:2rem; margin-bottom:1rem;">💬</div>
          <div style="color:#64748b; font-size:0.92rem;">
            Upload and clean a dataset first before starting a conversation.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ── Suggested questions ────────────────────────────────────────────────────────
SUGGESTIONS = [
    "What are the top 5 values by the highest numeric column?",
    "Show me summary statistics for all numeric columns.",
    "What is the average value grouped by the main category?",
    "Are there any outliers in the data?",
    "What trends do you see over time?",
]

def _now() -> str:
    return datetime.now().strftime("%H:%M")

if not st.session_state.chat_history:
    st.markdown(
        """<div style="font-size:0.8rem; color:#475569;
                       text-transform:uppercase; letter-spacing:0.08em;
                       margin-bottom:0.75rem;">Suggested questions</div>""",
        unsafe_allow_html=True,
    )
    suggestion_cols = st.columns(len(SUGGESTIONS))
    for col, suggestion in zip(suggestion_cols, SUGGESTIONS):
        with col:
            if st.button(
                suggestion[:42] + ("…" if len(suggestion) > 42 else ""),
                use_container_width=True,
                key=f"sug_{suggestion[:10]}",
            ):
                st.session_state.chat_history.append(
                    {"role": "user", "content": suggestion, "time": _now()}
                )
                st.session_state._pending_question = suggestion
                st.rerun()

# ── Render chat history ───────────────────────────────────────────────────────
def _render_message(msg: dict) -> None:
    role    = msg["role"]
    content = msg["content"]
    t       = msg.get("time", "")

    if role == "user":
        st.markdown(
            f"""<div class="chat-bubble chat-user">
              {content}
              <div class="chat-meta">{t}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        pandas_html = ""
        if msg.get("pandas_result"):
            pandas_html = f"""<div class="pandas-result">
              ⚡ Live calculation: {msg['pandas_result']}</div>"""

        sources_html = ""
        if msg.get("sources"):
            pills = "".join(
                f'<span class="source-pill">{s}</span>' for s in msg["sources"]
            )
            sources_html = f'<div class="source-pills">{pills}</div>'

        st.markdown(
            f"""<div class="chat-bubble chat-assistant">
              {content}
              {pandas_html}
              {sources_html}
              <div class="chat-meta">AInsights · {t}</div>
            </div>""",
            unsafe_allow_html=True,
        )


chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        _render_message(msg)


# ── Input ─────────────────────────────────────────────────────────────────────
col_input, col_send, col_clear = st.columns([8, 1, 1])

with col_input:
    question = st.text_input(
        "Ask a question …",
        key="chat_input",
        label_visibility="collapsed",
        placeholder="e.g. What is the average revenue by region?",
    )

with col_send:
    send = st.button("➤", type="primary", use_container_width=True)

with col_clear:
    if st.button("🗑", use_container_width=True, help="Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

# Handle pending question from suggestion buttons
if "_pending_question" in st.session_state:
    question = st.session_state.pop("_pending_question")
    send = True


# ── Send question ─────────────────────────────────────────────────────────────
def _ask(q: str) -> None:
    """POST to /chat, update session state, and rerun."""
    st.session_state.chat_history.append(
        {"role": "user", "content": q, "time": _now()}
    )

    typing_placeholder = st.empty()
    typing_placeholder.markdown(
        """<div class="chat-bubble chat-assistant" style="opacity:0.5;">
          <span style="animation: pulse 1s infinite;">
            ● ● ● &nbsp; Agent C is thinking …
          </span>
        </div>""",
        unsafe_allow_html=True,
    )

    try:
        r = httpx.post(
            f"{BACKEND_URL}/chat",
            json={"question": q, "session_id": "default"},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()

        typing_placeholder.empty()
        st.session_state.chat_history.append({
            "role":          "assistant",
            "content":       data.get("answer", "No answer returned."),
            "pandas_result": data.get("pandas_result"),
            "sources":       data.get("sources", []),
            "time":          _now(),
        })

    except httpx.TimeoutException:
        typing_placeholder.empty()
        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": "⚠ The request timed out. Try a simpler question or check the backend.",
            "time":    _now(),
        })
    except Exception as exc:
        typing_placeholder.empty()
        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": f"⚠ Error: {exc}",
            "time":    _now(),
        })

    st.rerun()


if (send or st.session_state.get("_submit_on_enter")) and question.strip():
    _ask(question.strip())

# Enter-key submission via JS injection
st.markdown(
    """
    <script>
    const inputs = window.parent.document.querySelectorAll(
      '[data-testid="stTextInput"] input'
    );
    inputs.forEach(inp => {
      inp.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
          const btn = window.parent.document.querySelector(
            '.stButton button[kind="primary"]'
          );
          if (btn) btn.click();
        }
      });
    });
    </script>
    """,
    unsafe_allow_html=True,
)

# ── Export chat ───────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    export_col, _ = st.columns([1, 5])
    with export_col:
        chat_text = "\n\n".join(
            f"[{m['role'].upper()} {m.get('time','')}]\n{m['content']}"
            for m in st.session_state.chat_history
        )
        st.download_button(
            "⬇ Export chat",
            data=chat_text,
            file_name="ainsights_chat.txt",
            mime="text/plain",
            use_container_width=True,
        )