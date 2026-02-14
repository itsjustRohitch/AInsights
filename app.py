import streamlit as st
import pandas as pd
import os      
import glob   
from langchain_ollama import OllamaLLM

from src.agent_a_engineer import AgentA_Engineer
from src.agent_b_visualizer import AgentB_Visualizer
from src.agent_c_analyst import AgentC_Analyst
from src.rag_engine import process_uploaded_file, process_cleaned_csv

#1. PAGE CONFIGURATION
st.set_page_config(
    page_title="AInsights: Agentic Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

#2. SESSION STATE MANAGEMENT
if 'data' not in st.session_state: st.session_state.data = None
if 'engineer_logs' not in st.session_state: st.session_state.engineer_logs = []
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

#3. SYSTEM INITIALIZATION
st.sidebar.title("🤖 AInsights System")

try:
    llm = OllamaLLM(model="llama3.2:1b")
    st.sidebar.success("● Brain: Llama 3.2 (Online)")
except Exception:
    llm = None
    st.sidebar.error("● Brain: Offline (Ollama needed)")

st.sidebar.markdown("---")

#4. SIDEBAR: AGENT A (THE DATA ENGINEER)
st.sidebar.header("📂 Agent A: Ingestion")
uploaded_data = st.sidebar.file_uploader(
    "Upload Raw Data", 
    type=['csv', 'xlsx', 'json', 'txt', 'pdf', 'html', 'xml'],
    help="Agent A accepts multi-format files and cleans them without renaming columns."
)

if uploaded_data and st.sidebar.button("▶ Start Engineering Pipeline"):
    with st.spinner("Agent A is cleaning and indexing..."):
        engineer = AgentA_Engineer(llm_engine=llm)
        clean_df, logs = engineer.run(uploaded_data)
        
        if clean_df is not None:
            st.session_state.data = clean_df
            st.session_state.engineer_logs = logs
            
            list_of_files = glob.glob('data/cleaned_data_*.csv') 
            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)
                rag_status = process_cleaned_csv(latest_file)
                st.sidebar.success(rag_status)
            
            st.sidebar.toast("Agent A: Pipeline & Indexing Success!", icon="✅")

# Sidebar Knowledge Base (For RAG)
st.sidebar.markdown("---")
st.sidebar.header("🧠 Knowledge Base")
uploaded_doc = st.sidebar.file_uploader("Upload PDF Reports", type=['pdf', 'txt'])
if uploaded_doc:
    with st.spinner("Updating RAG Memory..."):
        msg = process_uploaded_file(uploaded_doc)
        st.sidebar.info(msg)

# Show Engineer's Reasoning
if st.session_state.engineer_logs:
    with st.sidebar.expander("Agent A: Reasoning & Logs"):
        for log in st.session_state.engineer_logs:
            st.write(log)

# 5. MAIN DASHBOARD: AGENT B (THE VISUALIZER)
st.title("📊 Executive Agentic Dashboard")

if st.session_state.data is not None:
    st.markdown("### 👁️ Agent B: Visual Analysis")
    visualizer = AgentB_Visualizer(st.session_state.data)
    visualizer.render_overview()
    
    st.markdown("---")

    # 6. AGENT C (THE REASONING ANALYST)
    st.markdown("### 🧠 Agent C: Senior Analyst")
    st.caption("Analyzing live data + document context in real-time.")

    chat_box = st.container(border=True)
    
    with chat_box:
        for q, a in st.session_state.chat_history:
            with st.chat_message("user"): st.write(q)
            with st.chat_message("assistant", avatar="🧠"): st.write(a)

        user_query = st.chat_input("Ask a question about your business trends...")

        if user_query:
            with st.chat_message("user"): st.write(user_query)
            
            if llm:
                with st.chat_message("assistant", avatar="🧠"):
                    with st.spinner("Synthesizing reasoning..."):
                        analyst = AgentC_Analyst(llm, st.session_state.data)
                        response = analyst.get_response(user_query, st.session_state.chat_history)
                        st.write(response)
                        st.session_state.chat_history.append((user_query, response))
                        st.rerun() 
            else:
                st.error("Agent C cannot think: LLM is not connected.")

else:
    st.info("👋 Welcome to AInsights. Please upload a file via the sidebar to activate the agents.")
    st.markdown("""
    #### System Capabilities:
    1. **Agent A (Engineer)**: Ingests any file type and performs deep cleaning while preserving original column names.
    2. **Agent B (Visualizer)**: Automatically detects data types and builds grounded, interactive charts.
    3. **Agent C (Analyst)**: Uses RAG and Live Data reasoning to explain 'Why' your business trends are changing.
    """)