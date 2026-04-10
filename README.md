# 🔷 AInsights: Autonomous Agentic BI System

**AInsights** is a privacy-first, local-intelligence platform designed to transform raw, unstructured data into actionable business strategy. Powered by **Qwen2.5-Coder (7B)** and running entirely without external API dependencies, it orchestrates a **Triple-Agent Relay** to clean, visualize, and reason through complex datasets.

---

## Architecture: The Agentic Trinity

AInsights is built on a modular **"Separation of Concerns"** architecture where three specialized agents collaborate to provide a complete BI solution:

### 1. Agent A: The Universal Data Engineer

- **Role:** Ingestion, Sanitization, and Standardization.
- **Capabilities:** Native support for `CSV`, `XLSX`, `JSON`, `PDF`, and `HTML`.
- **Core Logic:** Employs a **"Pure Cleaning"** philosophy—performing deduplication, currency normalization, and date standardizing while strictly preserving original column names to ensure data lineage and integrity.

### 2. Agent B: The Adaptive Visualizer

- **Role:** Automated UI Generation and Exploratory Data Analysis (EDA).
- **Capabilities:** Automatically classifies columns into **Metrics** (Numeric) and **Dimensions** (Categorical/Time).
- **Core Logic:** Dynamically builds interactive Time-Series Trends, Heatmaps, and KPI Cards based on data-type heuristics without requiring hardcoded column mappings.

### 3. Agent C: The Reasoning Analyst (The "Ultimatum")

- **Role:** Strategic Insight and Question Answering.
- **Capabilities:** Advanced **Retrieval-Augmented Generation (RAG)** using **ChromaDB** for persistent vector storage.
- **Core Logic:** Operates a **Triple-Context Reasoning Loop**, synthesizing live dashboard statistics, document-based knowledge from PDF reports, and multi-turn conversational history to explain the "Why" behind business performance.

---

## Technical Implementation & Benefits

### Advanced RAG with ChromaDB

To eliminate hallucinations and provide evidence-based answers, AInsights leverages a local RAG pipeline:

- **Implementation:** We utilize **ChromaDB** as our high-performance vector database. Raw text from documents is chunked and embedded using the `all-MiniLM-L6-v2` transformer model. These embeddings are stored locally, allowing the model to perform semantic searches to find the most relevant context for any user query.
- **Benefits:** \* **Data Sovereignty:** Your data never leaves your machine to be indexed by third parties.
  - **Persistence:** Unlike in-memory stores, ChromaDB persists your data across restarts, avoiding redundant processing.
  - **Accuracy:** The model provides answers based on _your_ specific data facts rather than general training knowledge.

### Full Dockerization

The system is architected as a multi-service ecosystem managed via **Docker Compose**:

- **Implementation:** The platform is split into two distinct containers: a **FastAPI** backend container (handling the heavy lifting, agents, and RAG logic) and a **Streamlit** frontend container (handling the user interface).
- **Benefits:**
  - **Consistency:** Eliminates "it works on my machine" issues by standardizing dependencies and environment variables across any OS.
  - **Isolation:** Keeps your local system clean by containing all libraries and the database within the Docker environment.
  - **Simplified Networking:** Docker Compose automatically handles the communication bridge between the UI and the API layer.

---

## Scaling & Performance

- **Local-First Architecture:** 100% data privacy and zero API costs.
- **FastAPI Backend:** Orchestrates agent logic and RAG workflows via a high-performance, asynchronous API layer running in the background for low-latency reasoning.
- **Full Dockerization:** Ensures consistent performance across different environments, simplified dependency management, and easy one-command deployments.

---

## Installation & Setup

### Prerequisites

- Python 3.11+
- **Ollama:** [Install Ollama](https://ollama.com/) and pull the required model:
  ```bash
  ollama run qwen2.5-coder:7b
  ```

### Local Installation

This method requires running the backend and frontend services in separate terminal sessions.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/itsjustRohitch/AInsights.git
    cd AInsights
    ```
2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Start the FastAPI Backend:**
    ```bash
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    ```
4.  **Launch the Streamlit UI (In a new terminal):**
    ```bash
    streamlit run app.py
    ```

### Docker Installation

Running via Docker ensures consistency across environments, isolating dependencies and handling networking between the Streamlit frontend and FastAPI backend.

1.  **Clone and Launch:**
    ```bash
    git clone https://github.com/itsjustRohitch/AInsights.git
    cd AInsights
    docker-compose up --build
    ```
2.  **Access the UI:** Open `http://localhost:8501` in your browser.

---

> **Created by Ishan Ravishankar and Chinta Sri Durga Rohit**

---
