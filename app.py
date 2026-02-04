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
    ["☁️ Cloud (Gemini Pro)", "🔒 Local (Ollama/Mistral)"]
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

elif model_choice == "🔒 Local (Ollama/Mistral)":
    try:
        from langchain_community.llms import Ollama
        st.sidebar.info("Status: Offline Mode")
        llm = Ollama(model="mistral") 
        st.sidebar.success("Local Engine Ready ✅")
    except ImportError:
        st.sidebar.error("Run: pip install langchain-community")

st.sidebar.markdown("---")

# --- MAIN DASHBOARD ---
st.title("📊 AInsights: Executive Dashboard")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('data/sales_data.csv')

try:
    df = load_data()

    # --- 2. SLICERS (Week 3 Requirement) ---
    st.sidebar.subheader("Filters")
    
    # Region Slicer
    region_options = ["All"] + list(df['Region'].unique())
    selected_region = st.sidebar.selectbox("Filter by Region:", region_options)

    # Product Slicer
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

    # LEFT COLUMN: Dynamic Visuals
    with col1:
        st.subheader(f"📈 Market Overview ({selected_region})")
        
        # KPI Cards
        k1, k2, k3 = st.columns(3)
        total_rev = df_filtered['Sales'].sum()
        avg_sale = df_filtered['Sales'].mean()
        total_profit = df_filtered['Profit'].sum()
        
        k1.metric("Revenue", f"${total_rev:,.0f}", delta="Dynamic")
        k2.metric("Avg. Transaction", f"${avg_sale:.0f}")
        k3.metric("Net Profit", f"${total_profit:,.0f}")
        
        # Charts
        tab1, tab2 = st.tabs(["Sales Trend", "Category Split"])
        with tab1:
            st.line_chart(df_filtered.set_index('Date')['Sales'])
        with tab2:
            st.bar_chart(df_filtered.groupby('Product')['Sales'].sum())

        with st.expander("📄 View Raw Data"):
            st.dataframe(df_filtered.head(10))

    # RIGHT COLUMN: AI Analysis
    with col2:
        st.subheader("🤖 AI Analyst")
        st.caption(f"Ask questions about the {selected_region} region data.")
        
        query = st.text_input("Input Query:", placeholder="Why are sales down in North?")
        
        if st.button("Generate Insight"):
            if not query:
                st.warning("Please enter a question.")
            elif not llm:
                st.error("Please configure the Model in the Sidebar.")
            else:
                with st.spinner("Analyzing context..."):
                    try:
                        # 1. RETRIEVE
                        retriever = get_retriever()
                        docs = retriever.get_relevant_documents(query)
                        context_text = "\n\n".join([d.page_content for d in docs])
                        
                        # 2. PROMPT
                        final_prompt = f"""
                        You are an expert analyst. Answer based on this context:
                        {context_text}
                        
                        User Question: {query}
                        """
                        
                        # 3. GENERATE
                        response = llm.invoke(final_prompt)
                        if hasattr(response, 'content'):
                            st.info(response.content)
                        else:
                            st.info(response)
                            
                    except Exception as e:
                        st.error(f"Error: {e}")

except FileNotFoundError:
    st.error("Data file not found. Run 'python src/generate_data.py'")