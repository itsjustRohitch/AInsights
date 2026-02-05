import pandas as pd
from src.rag_engine import get_retriever

class AgentC_Analyst:
    """
    Agent C: The Senior Business Analyst.
    
    Role:
    1. Reason across structured data (Agent A's output).
    2. Synthesize unstructured knowledge (RAG documents).
    3. Provide conversational insights with memory.
    """

    def __init__(self, llm_engine, df):
        self.llm = llm_engine
        self.df = df
        self.retriever = get_retriever()

    def _generate_data_brief(self):
        """Calculates a high-level statistical summary for the AI's short-term memory."""
        try:
            # Identify numeric and categorical columns
            num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            
            summary = "--- LIVE DATA SUMMARY ---\n"
            
            for col in num_cols:
                summary += f"- {col}: Total={self.df[col].sum():,.0f}, Average={self.df[col].mean():,.0f}\n"
            
            for col in cat_cols:
                top_val = self.df[col].value_counts().idxmax()
                summary += f"- Top {col}: {top_val}\n"
                
            return summary
        except Exception as e:
            return f"Data summary generation failed: {str(e)}"

    def get_response(self, user_query, chat_history):
        if self.llm is None:
            return "Agent C: Brain offline."

        # 1. RETRIEVE DOCUMENT CONTEXT
        context_docs = "NO DOCUMENT CONTEXT AVAILABLE."
        if self.retriever:
            try:
                docs = self.retriever.invoke(user_query)
                context_docs = "\n".join([d.page_content for d in docs])
            except: pass

        # 2. DATA CONTEXT
        data_context = self._generate_data_brief()

        # 3. THE STRICT GROUNDING PROMPT
        prompt = f"""
        [STRICT INSTRUCTION]: 
        You are Agent C. You MUST only answer based on the PROVIDED DATA and DOCUMENTS below.
        If the information is not in the data or documents, say "I do not have enough specific data to answer that." 
        DO NOT use your general training knowledge about regions or shipping.

        --- PROVIDED DATA (AGENT A OUTPUT) ---
        {data_context}
        
        --- PROVIDED DOCUMENTS (RAG CONTEXT) ---
        {context_docs}
        
        --- USER QUESTION ---
        {user_query}

        --- RESPONSE PROTOCOL ---
        1. Look at the LIVE DATA STATS first.
        2. Look at the DOCUMENT CONTEXT second.
        3. If there is a contradiction, report it.
        4. Provide an answer based ONLY on these two sources.
        """

        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            return f"Agent C Error: {str(e)}"