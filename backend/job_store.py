"""
AInsights — Persistent Job Store
Replaces the in-memory _jobs dict with a file-backed store so that:
  - Multiple uvicorn workers all read/write the same state.
  - A --reload restart does not lose running jobs.
  - The API never returns 404 for a recently-created job.

Jobs are stored as individual JSON files in DATA_DIR/jobs/.
Each file is named {job_id}.json and is written atomically via
a temp-file rename so partial writes never corrupt state.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

log = logging.getLogger("ainsights.job_store")

_JOBS_DIR: Path | None = None
JOB_TTL = 7200   


def _jobs_dir() -> Path:
    global _JOBS_DIR
    if _JOBS_DIR is None:
        base = Path(os.getenv("DATA_DIR", "backend/data"))
        _JOBS_DIR = base / "jobs"
        _JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return _JOBS_DIR


def _job_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.json"

def set_job(
    job_id: str,
    status: str,
    detail: str = "",
    result: dict | None = None,
) -> None:
    """Write or update a job record atomically."""
    payload = {
        "job_id":     job_id,
        "status":     status,
        "detail":     detail,
        "result":     result or {},
        "created_at": time.time(),
    }
    path = _job_path(job_id)

    try:
        fd, tmp = tempfile.mkstemp(
            dir=str(_jobs_dir()), prefix=f".{job_id}_", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception as exc:
        log.error("Failed to write job %s: %s", job_id, exc)
        try:
            path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass


def get_job(job_id: str) -> dict | None:
    """Read a job record. Returns None if not found."""
    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception as exc:
        log.error("Failed to read job %s: %s", job_id, exc)
        return None

def job_exists(job_id: str) -> bool:
    return _job_path(job_id).exists()

def cleanup_old_jobs() -> int:
    cutoff = time.time() - JOB_TTL
    deleted = 0
    try:
        for f in _jobs_dir().glob("*.json"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
                    deleted += 1
            except Exception:
                pass
    except Exception as exc:
        log.warning("Job cleanup error: %s", exc)
    return deleted