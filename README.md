# 🔷 AInsights: Secure On-Device Agentic BI System

A privacy-first, Business Intelligence platform powered entirely by on-device AI.  
AInsights automates the full BI pipeline — data ingestion, cleaning, visualisation, and conversational analysis — while ensuring that no data ever leaves your machine.

---

## Overview

AInsights is built around a three-agent architecture that handles structured and unstructured data through a sequential pipeline. Each agent has a clearly defined responsibility, and all agents read from a single verified source of truth: a cleaned CSV produced by Agent A.

The platform is designed for deployment in two modes: a fully containerised Docker stack for server or cloud environments, and a standalone Windows desktop executable that manages all background services automatically.

---

## Architecture

```
User
  |
  v
Streamlit Frontend
  |
  v
FastAPI Gateway
  |
  +------------> Agent A (Data Engineer)
  |              performs code-driven data cleaning ---> produces cleaned_data.csv
  |
  +------------> RAG Engine
  |              ingests cleaned_data.csv ---> tabular + document embeddings
  |
  +------------> Agent B (Visualizer)
  |              reads cleaned_data.csv ---> generates charts
  |
  +------------> Agent C (Analyst)
                 queries RAG Engine  ---> conversational analysis

--------------------------------------------------

All agents (A, B, C) call:
 Ollama -- qwen2.5-coder:7b (local, offline)
```

---

## Agents

### Agent A: Data Engineer

Accepts raw files (CSV, Excel, JSON, XML, PDF tables), profiles the schema, and uses the local LLM to write and execute Python cleaning code in a sandboxed environment. If LLM-generated code fails validation, Agent A falls back to deterministic rule-based Pandas cleaning. Original column names and row integrity are preserved under all conditions.

### Agent B: Visualizer

Reads `cleaned_data.csv` and autonomously selects the most statistically appropriate Plotly charts based on detected column types. Also accepts free-form natural language chart requests, generating bespoke visualisations on demand via the LLM.

### Agent C: Analyst

A conversational BI interface that uses a dual-context approach to answer questions without hallucinating figures. For each question it performs two operations in parallel: vector retrieval from ChromaDB (covering both uploaded documents and tabular row embeddings), and live Pandas calculations on `cleaned_data.csv`. Both results are synthesised into a single grounded answer.

### RAG Engine

A dedicated background pipeline that embeds content into a local ChromaDB vector store using the `all-MiniLM-L6-v2` sentence transformer model. Handles two ingestion paths: unstructured documents (PDF, TXT, Markdown) via chunked text, and structured tabular data by converting CSV rows into natural-language sentences before embedding.

---

## Technology Stack

| Layer             | Technology                               |
| ----------------- | ---------------------------------------- |
| LLM runtime       | Ollama with `qwen2.5-coder:7b`           |
| LLM orchestration | LangChain, langchain-ollama              |
| Vector database   | ChromaDB (persistent, local)             |
| Embeddings        | sentence-transformers (all-MiniLM-L6-v2) |
| Backend API       | FastAPI, Uvicorn (async)                 |
| Frontend UI       | Streamlit with custom CSS                |
| Data processing   | Pandas, NumPy, PyArrow                   |
| File parsing      | pdfplumber, openpyxl, lxml, chardet      |
| Visualisation     | Plotly                                   |
| Data validation   | Pydantic                                 |
| Containerisation  | Docker, Docker Compose                   |
| Desktop packaging | PyInstaller                              |
| Language          | Python 3.11+                             |

---

## Features

- **Fully offline** — No external API calls, no cloud dependency
- **Multi-format ingestion** — CSV, Excel (.xlsx, .xls), JSON, XML, PDF tables
- **LLM-powered cleaning** — Dynamically generated and sandboxed Python code with automatic rule-based fallback
- **Tabular RAG** — Structured data rows are embedded as semantic sentences, making spreadsheet contents retrievable via vector search
- **Auto-visualisation** — Correlation matrices, time series, histograms, bar charts, scatter plots, box plots, and descriptive statistics tables
- **On-demand chart generation** — describe any chart in plain English; Agent B generates it
- **Streaming chat** — WebSocket-based token streaming for real-time LLM responses
- **Persistent job store** — File-backed job state survives server restarts and works across multiple workers
- **Docker and .exe support** — Single codebase, two deployment targets

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- [Ollama](https://ollama.com/download) installed and running
- The `qwen2.5-coder:7b` model pulled locally

```bash
ollama pull qwen2.5-coder:7b
ollama serve
```

### Local Development

```bash
# Clone and set up the environment
git clone https://github.com/itsjustRohitch/AInsights.git
cd AInsights
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Create required data directories
mkdir -p backend/data/uploads backend/data/vectorstore backend/data/jobs

# Terminal 1 — FastAPI backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit frontend
streamlit run frontend/app.py --server.port 8501
```

| Service                | URL                             |
| ---------------------- | ------------------------------- |
| Streamlit UI           | http://localhost:8501           |
| FastAPI docs (Swagger) | http://localhost:8000/docs      |
| Ollama API             | http://localhost:11434/api/tags |

---

## Docker Deployment

This setup assumes Ollama is already running on the host machine. The containers reach it via `host.docker.internal`.

### Configuration

Edit `.env` before starting:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen2.5-coder:7b
BACKEND_URL=http://backend:8000
DATA_DIR=/app/data
UPLOAD_DIR=/app/data/uploads
CHROMA_PERSIST_DIR=/app/data/vectorstore
```

### Starting the stack

```bash
# First run — builds images (5-10 minutes)
docker compose up --build

# Subsequent runs
docker compose up -d

# Stop (data volumes preserved)
docker compose down

# Full reset including data
docker compose down -v
```

| Service            | URL                        |
| ------------------ | -------------------------- |
| Streamlit frontend | http://localhost:8501      |
| FastAPI backend    | http://localhost:8000/docs |

The `app_data` Docker volume persists all uploaded files, cleaned CSVs, ChromaDB vectors, and job state across container restarts.

---

## Windows Desktop Executable

The `.exe` build bundles the Python runtime, all dependencies, the embedding model, and application source into a single distributable directory. The orchestrator manages the full lifecycle: starting Ollama silently, pulling the model on first launch, booting the backend and frontend, and opening the browser automatically.

### Build

```bat
.venv\Scripts\activate
python packaging/build.py
```

### Output

```
dist/
+-- AInsights/
    +-- AInsights.exe       Double-click to launch
    +-- _internal/
    +-- ainsights_data/
```

Distribute the entire `AInsights/` folder as a zip. End users require only:

1. Ollama installed.
2. The unzipped `AInsights/` folder
3. A double-click on `AInsights.exe`

---

## API Reference

The FastAPI backend exposes the following endpoints. Full interactive documentation is available at `/docs` when the server is running.

| Method | Endpoint            | Description                                  |
| ------ | ------------------- | -------------------------------------------- |
| GET    | `/health`           | Service health and readiness status          |
| POST   | `/upload/data`      | Upload structured data file, trigger Agent A |
| POST   | `/upload/document`  | Upload document for RAG ingestion            |
| GET    | `/data/cleaned`     | Download the cleaned CSV                     |
| GET    | `/data/schema`      | Get dataset schema and column metadata       |
| GET    | `/visualize`        | Trigger Agent B auto chart generation        |
| POST   | `/visualize/custom` | Request a specific chart from Agent B        |
| POST   | `/chat`             | Send a question to Agent C                   |
| WS     | `/chat/stream`      | WebSocket streaming chat                     |

---

## Data Privacy

All processing — including LLM inference, vector embedding, data cleaning, and storage — is executed locally on the host machine, with no data transmitted to external services.

This architecture is ideal for business intelligence workflows involving financial documents or other sensitive data, ensuring maximum privacy and control.

---

> **Created by Ishan Ravishankar and Chinta Sri Durga Rohit**

---
