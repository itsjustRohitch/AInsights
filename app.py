import streamlit as st
import pandas as pd
from src.rag_engine import get_retriever

# --- PAGE CONFIG ---
st.set_page_config(page_title="AInsights", layout="wide")

# --- SIDEBAR: CONFIGURATION ---
st.sidebar.title("⚙️ Dashboard Config")

# 1. AI Model Selection
st.sidebar.subheader("Intelligence Engine")
model_choice = st.sidebar.radio(
    "Select Model:",
    ["☁️ Cloud (Gemini Pro)", "🔒 Local (Llama 3.2 - Fast)"]
)

# Initialize LLM
llm = None
if model_choice == "☁️ Cloud (Gemini Pro)":
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = st.sidebar.text_input("Gemini API Key:", type="password")
        if api_key:
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
            st.sidebar.success("Online Mode Ready ✅")
    except ImportError:
        st.sidebar.error("Run: pip install langchain-google-genai")

elif model_choice == "🔒 Local (Llama 3.2 - Fast)":
    try:
        # NEW: Using the faster, modern library
        from langchain_ollama import OllamaLLM
        st.sidebar.info("Status: Offline Mode (Edge AI)")
        llm = OllamaLLM(model="llama3.2:1b") 
        st.sidebar.success("Local Engine Ready ✅")
    except ImportError:
        st.sidebar.error("Run: pip install langchain-ollama")

st.sidebar.markdown("---")

# --- MAIN DASHBOARD ---
st.title("📊 AInsights: Executive Dashboard")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('data/sales_data.csv')

try:
    df = load_data()

    # --- 2. SLICERS ---
    st.sidebar.subheader("Filters")
    region_options = ["All"] + list(df['Region'].unique())
    selected_region = st.sidebar.selectbox("Filter by Region:", region_options)

    product_options = ["All"] + list(df['Product'].unique())
    selected_product = st.sidebar.selectbox("Filter by Product:", product_options)

    # Apply Filters
    df_filtered = df.copy()
    if selected_region != "All":
        df_filtered = df_filtered[df_filtered['Region'] == selected_region]
    if selected_product != "All":
        df_filtered = df_filtered[df_filtered['Product'] == selected_product]

    # --- LAYOUT ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📈 Market Overview ({selected_region})")
        k1, k2, k3 = st.columns(3)
        k1.metric("Revenue", f"${df_filtered['Sales'].sum():,}")
        k2.metric("Avg. Sale", f"${int(df_filtered['Sales'].mean())}")
        k3.metric("Profit", f"${df_filtered['Profit'].sum():,}")
        
        tab1, tab2 = st.tabs(["Trend", "Category"])
        with tab1:
            st.line_chart(df_filtered.set_index('Date')['Sales'])
        with tab2:
            st.bar_chart(df_filtered.groupby('Product')['Sales'].sum())

    with col2:
        st.subheader("🤖 AI Analyst")
        query = st.text_input("Input Query:", placeholder="Why are sales high?")
        
        if st.button("Generate Insight"):
            if not query:
                st.warning("Enter a question.")
            elif not llm:
                st.error("Configure Model first.")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        # 1. RETRIEVE
                        retriever = get_retriever()
                        docs = retriever.invoke(query)
                        
                        # 2. PROMPT
                        context = "\n".join([d.page_content for d in docs])
                        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
                        
                        # 3. GENERATE
                        response = llm.invoke(prompt)
                        st.info(response)
                            
                    except Exception as e:
                        st.error(f"Error: {e}")

except FileNotFoundError:
    st.error("Run 'python src/generate_data.py'")