import streamlit as st
import pandas as pd
import os
import glob
from langchain_ollama import OllamaLLM

from src.agent_a_engineer import AgentA_Engineer
from src.agent_b_visualizer import AgentB_Visualizer
from src.agent_c_analyst import AgentC_Analyst
from src.rag_engine import process_uploaded_file, process_cleaned_csv

st.set_page_config(
    page_title="AInsights | Agentic Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0B0E14;
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="block-container"] {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }

    [data-testid="stSidebarUserContent"] {
        padding-top: 0rem !important;
        padding-bottom: 1rem !important;
    }

    [data-testid="stSidebar"] {
        background-color: #06080C !important;
        border-right: 1px solid #1E293B;
    }

    .sb-ingestion { color: #00F2FE; font-weight: 800; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 1.2px; }
    .sb-rag { color: #A855F7; font-weight: 800; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 1.2px; }

    .brand-text {
        background: linear-gradient(135deg, #00F2FE 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1.5px;
    }

    .agent-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 16px;
        margin-bottom: 0px;
        transition: transform 0.2s ease;
        height: 100%;
    }
    
    .agent-card:hover {
        border: 1px solid rgba(0, 242, 254, 0.3);
        transform: translateY(-2px);
    }

    .stButton>button {
        background: linear-gradient(90deg, #1E293B, #0F172A);
        color: #00F2FE;
        border: 1px solid rgba(0, 242, 254, 0.4);
        border-radius: 8px;
        font-weight: 600;
        height: 3em;
    }

    .stTabs [aria-selected="true"] {
        color: #00F2FE !important;
        border-bottom: 2px solid #00F2FE !important;
    }

    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.5);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #1E293B;
    }
    </style>
    """, unsafe_allow_html=True)

if 'data' not in st.session_state: st.session_state.data = None
if 'engineer_logs' not in st.session_state: st.session_state.engineer_logs = []
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

with st.sidebar:
    st.markdown("""
        <h1 style='font-size: 2rem; margin-top: 0; margin-bottom: 0;'><span class='brand-text'>⚡ AInsights</span></h1>
        <p style='font-size: 0.95rem; color: #94A3B8; margin-top: -5px; margin-bottom: 15px;'>
            Autonomous Agentic BI System
        </p>
    """, unsafe_allow_html=True)

    try:
        llm = OllamaLLM(model="llama3.2:1b")
        status_color = "#10B981"
        status_text = "Llama 3.2 Online"
    except Exception:
        llm = None
        status_color = "#EF4444"
        status_text = "Brain Offline"

    st.markdown(f"""
        <div style='padding: 8px 12px; border-radius: 8px; border: 1px solid #1E293B; background: #0F172A; margin-bottom: 15px;'>
            <span style='color: {status_color};'>●</span> 
            <span style='font-size: 0.8rem; color: #E2E8F0; font-weight: 600;'>{status_text}</span>
        </div>
        <hr style='border-color: rgba(255,255,255,0.1); margin: 0 0 15px 0;'>
        <p class="sb-ingestion" style="margin-bottom: 2px;">Ingestion Engine</p>
        <p style="font-size: 0.75rem; color: #64748B; margin-top: 0; margin-bottom: 10px;">Upload raw data for automated cleaning and profiling.</p>
    """, unsafe_allow_html=True)

    uploaded_data = st.file_uploader(
        "Upload Raw Data", 
        type=['csv', 'xlsx', 'json', 'txt', 'pdf', 'html', 'xml'], 
        label_visibility="collapsed",
        help="Agent A accepts multi-format files and cleans them without renaming columns."
    )

    if uploaded_data and st.button("Run Triple-Agent Relay", use_container_width=True):
        with st.status("Agent A is cleaning and indexing...", expanded=False) as status:
            engineer = AgentA_Engineer(llm_engine=llm)
            clean_df, logs = engineer.run(uploaded_data)
            
            if clean_df is not None:
                st.session_state.data = clean_df
                st.session_state.engineer_logs = logs
                
                list_of_files = glob.glob('data/cleaned_data_*.csv') 
                if list_of_files:
                    latest_file = max(list_of_files, key=os.path.getctime)
                    rag_status = process_cleaned_csv(latest_file)
                    st.success(rag_status)
                
                status.update(label="Engineering Finalized", state="complete")
                st.toast("Agent A: Pipeline & Indexing Success!")

    if st.session_state.engineer_logs:
        with st.expander("Agent A: Reasoning & Logs"):
            for log in st.session_state.engineer_logs:
                st.write(log)

    st.markdown("""
        <hr style='border-color: rgba(255,255,255,0.1); margin: 20px 0 15px 0;'>
        <p class="sb-rag" style="margin-bottom: 2px;">RAG Memory</p>
        <p style="font-size: 0.75rem; color: #64748B; margin-top: 0; margin-bottom: 10px;">Upload PDF reports to inject knowledge into FAISS.</p>
    """, unsafe_allow_html=True)
    
    uploaded_doc = st.file_uploader("Upload contextual files", type=['pdf', 'txt'], label_visibility="collapsed")
    
    if uploaded_doc:
        with st.spinner("Updating RAG Memory..."):
            msg = process_uploaded_file(uploaded_doc)
            st.success(msg)

if st.session_state.data is not None:
    st.markdown("<h2 style='letter-spacing:-1px;'>Intelligence Dashboard</h2>", unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows Scanned", f"{st.session_state.data.shape[0]:,}")
    m2.metric("Features", st.session_state.data.shape[1])
    m3.metric("Agent Status", "Grounded", delta="Ready")

    tab_viz, tab_chat = st.tabs(["VISUALIZER", "ANALYST"])

    with tab_viz:
        st.markdown("### Agent B: Visual Analysis")
        visualizer = AgentB_Visualizer(st.session_state.data)
        visualizer.render_overview()
    
    with tab_chat:
        st.markdown("### Agent C: Senior Analyst")
        chat_box = st.container(height=500, border=True)
        
        with chat_box:
            for q, a in st.session_state.chat_history:
                with st.chat_message("user"): st.markdown(q)
                with st.chat_message("assistant"): st.markdown(a)

        user_query = st.chat_input("Ask a question about your business trends...")

        if user_query:
            with chat_box:
                with st.chat_message("user"): st.markdown(user_query)
                
                if llm:
                    with st.chat_message("assistant"):
                        with st.spinner("Synthesizing reasoning..."):
                            analyst = AgentC_Analyst(llm, st.session_state.data)
                            response = analyst.get_response(user_query, st.session_state.chat_history)
                            st.markdown(response)
                            st.session_state.chat_history.append((user_query, response))
                            st.rerun()
                else:
                    st.error("Agent C cannot think: LLM is not connected.")

else:
    st.markdown("""
        <div style='text-align: center; padding: 0px 0px 20px 0px;'>
            <h1 style='font-size: 3.5rem; margin-bottom: 5px;'><span class='brand-text'>⚡ AInsights</span></h1>
            <p style='color: #94A3B8; font-size: 1.1rem; max-width: 800px; margin: 0 auto 30px auto; line-height: 1.5;'>
                A privacy-first BI system leveraging RAG to transform unstructured data into actionable business strategy.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class='agent-card'>
                <h3 style='color: #00F2FE; margin-top: 0; margin-bottom: 5px; text-align: center;'>Agent A</h3>
                <p style='font-size: 1rem; color: #E2E8F0; text-align: center; font-weight: 600; margin-bottom: 0;'>The Universal Data Engineer</p>
                <hr style='border-color: rgba(255,255,255,0.1); margin: 10px 0;'>
                <p style='font-size: 0.88rem; color: #94A3B8; margin-top: 10px; margin-bottom: 0; line-height: 1.4;'>
                Responsible for ingestion and sanitization. Normalizes data structures and ensures clean data lineage for subsequent analysis.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class='agent-card'>
                <h3 style='color: #00F2FE; margin-top: 0; margin-bottom: 5px; text-align: center;'>Agent B</h3>
                <p style='font-size: 1rem; color: #E2E8F0; text-align: center; font-weight: 600; margin-bottom: 0;'>The Adaptive Visualizer</p>
                <hr style='border-color: rgba(255,255,255,0.1); margin: 10px 0;'>
                <p style='font-size: 0.88rem; color: #94A3B8; margin-top: 10px; margin-bottom: 0; line-height: 1.4;'>
                Automatically generates interactive Exploratory Data Analysis (EDA) and KPI dashboards based on data-type heuristics.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class='agent-card'>
                <h3 style='color: #00F2FE; margin-top: 0; margin-bottom: 5px; text-align: center;'>Agent C</h3>
                <p style='font-size: 1rem; color: #E2E8F0; text-align: center; font-weight: 600; margin-bottom: 0;'>The Reasoning Analyst</p>
                <hr style='border-color: rgba(255,255,255,0.1); margin: 10px 0;'>
                <p style='font-size: 0.88rem; color: #94A3B8; margin-top: 10px; margin-bottom: 0; line-height: 1.4;'>
                Synthesizes dashboard metrics, document knowledge, and chat history to provide strategic business reasoning using a RAG Engine.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <hr style='border-color: rgba(255,255,255,0.1); margin: 30px 0 25px 0;'>
        <div style='text-align: center; color: #FFFFFF; font-size: 1rem; font-weight: 600; letter-spacing: 0.5px;'>
            <span style='margin-right: 8px; color: #00F2FE;'>←</span> Upload files in the sidebar to get started
        </div>
    """, unsafe_allow_html=True)