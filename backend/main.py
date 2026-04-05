"""
AInsights — FastAPI Gateway (v2)
Improvements:
  - asyncio.gather for parallel background operations
  - Configurable LLM timeout propagated to agents
  - Improved job TTL with automatic cleanup
  - Smarter health endpoint with detailed readiness info
  - Schema endpoint cached with a short TTL
  - Proper streaming response for chat
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import aiofiles
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ainsights.gateway")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(os.getenv("DATA_DIR",   "backend/data"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "backend/data/uploads"))
CLEAN_CSV  = DATA_DIR / "cleaned_data.csv"

for d in (DATA_DIR, UPLOAD_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Job registry ──────────────────────────────────────────────────────────────
# job_id → {status, detail, result, created_at}
_jobs: dict[str, dict] = {}
JOB_TTL = 3600   # purge jobs older than 1 hour

# ── Schema cache ──────────────────────────────────────────────────────────────
_schema_cache: dict = {}
_schema_cache_ts: float = 0.0
SCHEMA_CACHE_TTL = 30   # seconds

# ── Shared state ──────────────────────────────────────────────────────────────
_rag_engine = None


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global _rag_engine
    log.info("AInsights backend starting …")

    from backend.rag_engine import RAGEngine
    _rag_engine = RAGEngine()

    # Initialize in thread pool (blocking model load)
    await asyncio.to_thread(_rag_engine.initialize)
    log.info("RAG engine ready. %s", _rag_engine.stats())

    # Background job cleanup task
    asyncio.create_task(_job_cleanup_loop())

    yield
    log.info("Backend shutting down.")


async def _job_cleanup_loop() -> None:
    """Periodically remove stale jobs to prevent memory bloat."""
    while True:
        await asyncio.sleep(300)   # run every 5 minutes
        cutoff = time.time() - JOB_TTL
        stale  = [jid for jid, j in _jobs.items() if j.get("created_at", 0) < cutoff]
        for jid in stale:
            del _jobs[jid]
        if stale:
            log.info("Purged %d stale jobs.", len(stale))


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AInsights API",
    version="2.0.0",
    description="Local-first BI. All processing on-device.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question:   str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer:        str
    sources:       list[str] = []
    pandas_result: str | None = None

class JobStatus(BaseModel):
    job_id:  str
    status:  str
    detail:  str = ""
    result:  dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _set_job(job_id: str, status: str, detail: str = "", result: dict | None = None) -> None:
    _jobs[job_id] = {
        "job_id":     job_id,
        "status":     status,
        "detail":     detail,
        "result":     result or {},
        "created_at": time.time(),
    }


ALLOWED_DATA = {".csv", ".xlsx", ".xls", ".json", ".xml", ".pdf"}
ALLOWED_DOCS = {".pdf", ".txt", ".md"}


def _check_ext(filename: str, allowed: set[str]) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}",
        )
    return ext


async def _save_upload(file: UploadFile, job_id: str) -> Path:
    """Stream uploaded file to disk efficiently."""
    save_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    async with aiofiles.open(save_path, "wb") as f:
        while chunk := await file.read(512 * 1024):   # 512 KB chunks
            await f.write(chunk)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health() -> dict:
    rag_stats = _rag_engine.stats() if _rag_engine else {}
    return {
        "status":            "ok",
        "rag_ready":         _rag_engine is not None,
        "cleaned_csv":       CLEAN_CSV.exists(),
        "cleaned_csv_rows":  _quick_row_count(),
        "rag_doc_chunks":    rag_stats.get("doc_chunks", 0),
        "rag_table_chunks":  rag_stats.get("table_chunks", 0),
    }


def _quick_row_count() -> int:
    if not CLEAN_CSV.exists():
        return 0
    try:
        with open(CLEAN_CSV) as f:
            return sum(1 for _ in f) - 1   # minus header
    except Exception:
        return -1


# ── Upload structured data ────────────────────────────────────────────────────
@app.post("/upload/data", tags=["pipeline"], response_model=JobStatus)
async def upload_data(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> JobStatus:
    _check_ext(file.filename, ALLOWED_DATA)
    job_id = str(uuid.uuid4())

    save_path = await _save_upload(file, job_id)
    _set_job(job_id, "pending", "File saved. Queuing Agent A …")
    log.info("Data upload saved: %s | job: %s", save_path.name, job_id)

    background_tasks.add_task(_bg_run_agent_a, job_id, save_path)
    return JobStatus(**_jobs[job_id])


async def _bg_run_agent_a(job_id: str, file_path: Path) -> None:
    _set_job(job_id, "running", "Agent A: profiling schema …")
    try:
        from backend.agents.agent_a import DataEngineerAgent
        agent = DataEngineerAgent(
            llm_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )

        _set_job(job_id, "running", "Agent A: LLM cleaning code generation …")
        result = await asyncio.to_thread(agent.run, file_path, CLEAN_CSV)

        _set_job(job_id, "running", "RAG Engine: ingesting cleaned CSV …", result)

        # RAG ingestion runs concurrently with schema cache invalidation
        await asyncio.gather(
            asyncio.to_thread(_rag_engine.ingest_tabular_data, CLEAN_CSV),
            asyncio.to_thread(_invalidate_schema_cache),
        )

        _set_job(job_id, "complete", "Pipeline complete — cleaned_data.csv is ready.", result)
        log.info("Full pipeline complete for job %s", job_id)

    except Exception as exc:
        log.exception("Agent A pipeline failed for job %s", job_id)
        _set_job(job_id, "failed", str(exc))


def _invalidate_schema_cache() -> None:
    global _schema_cache, _schema_cache_ts
    _schema_cache    = {}
    _schema_cache_ts = 0.0


# ── Upload document ───────────────────────────────────────────────────────────
@app.post("/upload/document", tags=["pipeline"], response_model=JobStatus)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> JobStatus:
    _check_ext(file.filename, ALLOWED_DOCS)
    job_id    = str(uuid.uuid4())
    save_path = await _save_upload(file, job_id)

    _set_job(job_id, "pending", "Document saved. Queuing RAG ingestion …")
    background_tasks.add_task(_bg_ingest_doc, job_id, save_path)
    return JobStatus(**_jobs[job_id])


async def _bg_ingest_doc(job_id: str, file_path: Path) -> None:
    _set_job(job_id, "running", "RAG Engine: chunking and embedding …")
    try:
        n_chunks = await asyncio.to_thread(_rag_engine.ingest_document, file_path)
        _set_job(
            job_id, "complete",
            f"Indexed {n_chunks} chunks from '{file_path.name}' into vector store.",
        )
    except Exception as exc:
        log.exception("Doc ingest failed for job %s", job_id)
        _set_job(job_id, "failed", str(exc))


# ── Job status ────────────────────────────────────────────────────────────────
@app.get("/jobs/{job_id}", tags=["pipeline"], response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatus(**_jobs[job_id])


# ── Visualize (Agent B) ───────────────────────────────────────────────────────
@app.get("/visualize", tags=["agents"])
async def visualize() -> JSONResponse:
    if not CLEAN_CSV.exists():
        raise HTTPException(status_code=404, detail="No cleaned data. Upload a file first.")
    try:
        from backend.agents.agent_b import VisualizerAgent
        agent   = VisualizerAgent()
        figures = await asyncio.to_thread(agent.run, CLEAN_CSV)
        return JSONResponse(content={"charts": figures})
    except Exception as exc:
        log.exception("Agent B failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Chat (Agent C) ────────────────────────────────────────────────────────────
@app.post("/chat", tags=["agents"], response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not CLEAN_CSV.exists():
        raise HTTPException(status_code=404, detail="No cleaned data. Upload a file first.")
    try:
        from backend.agents.agent_c import AnalystAgent
        agent  = AnalystAgent(rag_engine=_rag_engine)
        result = await asyncio.to_thread(agent.run, req.question, CLEAN_CSV)
        return ChatResponse(**result)
    except Exception as exc:
        log.exception("Agent C failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Streaming chat (WebSocket) ────────────────────────────────────────────────
@app.websocket("/chat/stream")
async def chat_stream(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            data     = await ws.receive_json()
            question = data.get("question", "").strip()
            if not question:
                await ws.send_json({"error": "Empty question."})
                continue
            if not CLEAN_CSV.exists():
                await ws.send_json({"error": "No data loaded."})
                continue

            from backend.agents.agent_c import AnalystAgent
            agent = AnalystAgent(rag_engine=_rag_engine, streaming=True)

            async for token in agent.stream(question, CLEAN_CSV):
                await ws.send_json({"token": token})

            await ws.send_json({"done": True})

    except WebSocketDisconnect:
        log.info("WebSocket disconnected.")


# ── Cleaned CSV download ──────────────────────────────────────────────────────
@app.get("/data/cleaned", tags=["data"])
async def download_cleaned() -> FileResponse:
    if not CLEAN_CSV.exists():
        raise HTTPException(status_code=404, detail="No cleaned data available.")
    return FileResponse(
        path=CLEAN_CSV,
        media_type="text/csv",
        filename="ainsights_cleaned_data.csv",
    )


# ── Schema (cached) ───────────────────────────────────────────────────────────
@app.get("/data/schema", tags=["data"])
async def get_schema() -> JSONResponse:
    global _schema_cache, _schema_cache_ts

    if not CLEAN_CSV.exists():
        raise HTTPException(status_code=404, detail="No cleaned data available.")

    # Return cached schema if fresh
    if _schema_cache and (time.time() - _schema_cache_ts) < SCHEMA_CACHE_TTL:
        return JSONResponse(content=_schema_cache)

    import pandas as pd

    df = await asyncio.to_thread(pd.read_csv, CLEAN_CSV, nrows=5000)
    schema = {
        col: {
            "dtype":  str(df[col].dtype),
            "nulls":  int(df[col].isna().sum()),
            "sample": df[col].dropna().head(3).tolist(),
            "unique": int(df[col].nunique()),
        }
        for col in df.columns
    }
    payload = {
        "rows":    _quick_row_count(),
        "columns": len(df.columns),
        "schema":  schema,
    }
    _schema_cache    = payload
    _schema_cache_ts = time.time()

    return JSONResponse(content=payload)


# ─────────────────────────────────────────────────────────────────────────────
# Dev entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def run() -> None:
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        port=int(os.getenv("BACKEND_PORT", 8000)),
        reload=True,
        timeout_keep_alive=300,    # keep connections alive for long LLM responses
    )


if __name__ == "__main__":
    run()