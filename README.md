# AInsights: RAG-Enhanced Business Intelligence Dashboard

**Course:** CSE3232 - Emerging Tools & Technology  
**Timeline:** Jan 2026 - March 2026  
**Developer:** [Your Name]

## 🚀 Project Objective
AInsights is a decision-support system that bridges the gap between raw data and actionable intelligence. Unlike traditional dashboards, it integrates **Streamlit** for visualization with a **Hybrid RAG (Retrieval-Augmented Generation)** backend. [cite_start]This allows users to receive context-aware insights and natural language explanations for data anomalies[cite: 8, 9, 10].

## 🛠 Tech Stack
* **Frontend:** Streamlit (Python-based interactive UI)
* **Data Processing:** Pandas & NumPy
* **AI Engine (Hybrid):** * *Cloud Mode:* Google Gemini Pro (via API)
    * *Offline Mode:* Ollama (Mistral/Llama3) - **Works without Internet**
* **Vector Database:** FAISS (Local storage for high-speed retrieval)
* **Embeddings:** `all-MiniLM-L6-v2` (Hugging Face)

## ⚙️ Installation & Setup
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/itsjustRohitch/AInsights.git](https://github.com/itsjustRohitch/AInsights.git)
    cd AInsights
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard**
    ```bash
    streamlit run app.py
    ```

## 🧠 Architecture
1.  **Data Layer:** Ingests structured sales data (CSV) and unstructured market reports (PDF/TXT).
2.  **Vector Store:** Uses FAISS to index text chunks for semantic search.
3.  **RAG Pipeline:** * Retrieves relevant context based on user query.
    * Injects context into the LLM prompt.
    * Generates accurate, evidence-based answers.

## 📸 Features (Week 3 Status)
* ✅ **Dynamic Slicers:** Filter by Region and Product.
* ✅ **Interactive Charts:** Auto-updating Line and Bar charts.
* ✅ **Offline Capability:** Fully functional AI analysis without internet connection.