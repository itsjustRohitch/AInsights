import streamlit as st
import pandas as pd
from src.rag_engine import get_retriever

# --- PAGE CONFIG ---
st.set_page_config(page_title="AInsights", layout="wide")

# --- SIDEBAR: MODEL CONFIG ---
st.sidebar.title("⚙️ System Config")
model_choice = st.sidebar.radio(
    "Select Intelligence Engine:",
    ["☁️ Cloud (Gemini Pro)", "🔒 Local (Ollama/Mistral)"]
)

# Initialize the correct LLM
llm = None
if model_choice == "☁️ Cloud (Gemini Pro)":
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")
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

# --- MAIN DASHBOARD ---
st.title("📊 AInsights: Executive Dashboard")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('data/sales_data.csv')

try:
    df = load_data()
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Market Overview")
        st.metric("Total Revenue", f"${df['Sales'].sum():,}")
        st.line_chart(df.set_index('Date')['Sales'])

    with col2:
        st.subheader("🤖 AI Insights")
        query = st.text_input("Input Query:", placeholder="Ask about the market report...")
        
        if st.button("Generate Insight"):
            if not query:
                st.warning("Please enter a question.")
            elif not llm:
                st.error("Please configure the Model in the Sidebar.")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        # 1. RETRIEVE (Get relevant text manually)
                        retriever = get_retriever()
                        docs = retriever.get_relevant_documents(query)
                        
                        # 2. COMBINE (Make a single text block)
                        context_text = "\n\n".join([d.page_content for d in docs])
                        
                        # 3. GENERATE (Send directly to LLM)
                        # We construct the prompt manually to avoid 'Chain' errors
                        final_prompt = f"""
                        You are an expert business analyst. Answer the question based ONLY on the context below.
                        
                        Context:
                        {context_text}
                        
                        Question:
                        {query}
                        """
                        
                        # Handle different model types (Gemini returns object, Ollama returns string)
                        response = llm.invoke(final_prompt)
                        
                        # Clean output
                        if hasattr(response, 'content'):
                            st.info(response.content)
                        else:
                            st.info(response)
                            
                    except Exception as e:
                        st.error(f"Error: {e}")

except FileNotFoundError:
    st.error("Data file not found. Run 'python src/generate_data.py'")