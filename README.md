🤖 AInsights: Autonomous Agentic BI System
AInsights is a privacy-first, local-intelligence platform designed to transform raw, unstructured data into actionable business strategy. Powered by Llama 3.2 (1B) and running entirely without external API dependencies, it orchestrates a Triple-Agent Relay to clean, visualize, and reason through complex datasets.

🏗️ Architecture: The Agentic Trinity
AInsights is built on a modular "Separation of Concerns" architecture where three specialized agents collaborate to provide a complete BI solution:

1. Agent A: The Universal Data Engineer
Role: Ingestion, Sanitization, and Standardization.

Capabilities: Native support for CSV, XLSX, JSON, PDF, and HTML.

Core Logic: Employs a "Pure Cleaning" philosophy—performing deduplication, currency normalization, and date standardizing while strictly preserving original column names to ensure data lineage and integrity.

2. Agent B: The Adaptive Visualizer
Role: Automated UI Generation and Exploratory Data Analysis (EDA).

Capabilities: Automatically classifies columns into Metrics (Numeric) and Dimensions (Categorical/Time).

Core Logic: Dynamically builds interactive Time-Series Trends, Heatmaps, and KPI Cards based on data-type heuristics without requiring hardcoded column mappings.

3. Agent C: The Reasoning Analyst (The "Ultimatum")
Role: Strategic Insight and Question Answering.

Capabilities: Advanced Retrieval-Augmented Generation (RAG) using FAISS vector storage.

Core Logic: Operates a Triple-Context Reasoning Loop, synthesizing live dashboard statistics, document-based knowledge from PDF reports, and multi-turn conversational history to explain the "Why" behind business performance.

⚡ Scaling & Performance
Local-First Architecture: 100% data privacy and zero API costs.

Persistent RAG Engine: Uses HuggingFace (MiniLM-L6) embeddings to index thousands of data rows and document pages into a local FAISS vector store.

Optimized Ingestion: Features recursive character splitting for faster indexing of large-scale CSVs and complex PDF reports.

🚀 Installation & Setup
Prerequisites
Python 3.10+

Ollama: Install and run ollama run llama3.2:1b.

Quick Start
Clone the Repository:

Bash
git clone https://github.com/itsjustRohitch/AInsights.git
cd AInsights
Install Requirements:

Bash
pip install -r requirements.txt
Run the System:

Bash
streamlit run app.py
🔮 Future Roadmap
Multi-Modal Ingestion: Support for image-based receipts and charts.

Distributed Vector Storage: Moving from FAISS to Qdrant for enterprise-scale indexing.

Agent Autonomy: Allowing Agent A to automatically suggest new metrics based on detected data anomalies.