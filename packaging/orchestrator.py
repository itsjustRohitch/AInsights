
from __future__ import annotations

import multiprocessing
multiprocessing.freeze_support()

import os
import sys
import tempfile
from pathlib import Path

_LOG_FILE = Path(tempfile.gettempdir()) / "ainsights_launcher.log"
_log_fh   = open(_LOG_FILE, "a", encoding="utf-8", buffering=1)  # noqa: SIM115

if sys.stdout is None:
    sys.stdout = _log_fh
if sys.stderr is None:
    sys.stderr = _log_fh

import atexit
import logging
import platform
import signal
import subprocess
import threading
import time
import webbrowser
from typing import Optional

import httpx

IS_WINDOWS = platform.system() == "Windows"
IS_FROZEN  = getattr(sys, "frozen", False)

OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_URL   = "http://127.0.0.1:11434"
BACKEND_URL  = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:8501"

OLLAMA_HEALTH_TIMEOUT   = 120
BACKEND_HEALTH_TIMEOUT  = 120
FRONTEND_HEALTH_TIMEOUT = 60
MODEL_PULL_TIMEOUT      = 1800
POLL_INTERVAL           = 2.0
GRACE_PERIOD            = 8
_CREATE_NO_WINDOW       = 0x08000000 if IS_WINDOWS else 0

if IS_FROZEN:
    BASE_DIR = Path(sys._MEIPASS)           # type: ignore[attr-defined]
    EXE_DIR  = Path(sys.executable).parent
    DATA_DIR = EXE_DIR / "ainsights_data"
else:
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "backend" / "data"

LOG_FILE = _LOG_FILE  

def _run_as_service(service: str) -> None:
    for _d in [DATA_DIR, DATA_DIR / "uploads",
               DATA_DIR / "vectorstore", DATA_DIR / "jobs"]:
        _d.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("BACKEND_URL",     BACKEND_URL)
    os.environ.setdefault("OLLAMA_BASE_URL", OLLAMA_URL)

    if service == "frontend":
        app_path = str(BASE_DIR / "frontend" / "app.py")
        sys.argv = [
            "streamlit", "run", app_path,
            "--server.port=8501",
            "--server.address=127.0.0.1",
            "--server.headless=true",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
            "--browser.gatherUsageStats=false",
            "--theme.base=dark",
        ]
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        try:
            from streamlit.web.cli import main as st_main
            st_main(standalone_mode=False)
        except SystemExit:
            pass
        except Exception as exc:
            _log_fh.write(f"[frontend service] crash: {exc}\n")
            _log_fh.flush()
    else:
        sys.stderr.write(f"Unknown service: {service}\n")
        sys.exit(1)


if len(sys.argv) >= 3 and sys.argv[1] == "--ainsights-service":
    _run_as_service(sys.argv[2])
    sys.exit(0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(_log_fh),
        logging.StreamHandler(sys.__stdout__ or _log_fh),
    ],
)
log = logging.getLogger("ainsights.orchestrator")

# Data directories
for _d in [DATA_DIR, DATA_DIR / "uploads",
           DATA_DIR / "vectorstore", DATA_DIR / "jobs"]:
    _d.mkdir(parents=True, exist_ok=True)

if IS_WINDOWS:
    _ollama_candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles",  "")) / "Ollama" / "ollama.exe",
        Path("ollama.exe"),
    ]
else:
    _ollama_candidates = [
        Path("/usr/local/bin/ollama"),
        Path("/usr/bin/ollama"),
        Path("ollama"),
    ]
OLLAMA_BIN: Optional[Path] = next(
    (p for p in _ollama_candidates if p.exists()), None
)

_shutdown_event  = threading.Event()
_fatal_lock      = threading.Lock()
_fatal_fired     = False
_uvicorn_server  = None
_frontend_proc: Optional[subprocess.Popen] = None


# Shutdown
def _terminate_all() -> None:
    log.info("Initiating graceful shutdown …")
    global _uvicorn_server
    if _uvicorn_server is not None:
        try:
            _uvicorn_server.should_exit = True
        except Exception:
            pass
    if _frontend_proc and _frontend_proc.poll() is None:
        try:
            _frontend_proc.terminate()
            _frontend_proc.wait(timeout=GRACE_PERIOD)
        except subprocess.TimeoutExpired:
            _frontend_proc.kill()
        except Exception as exc:
            log.error("Frontend termination error: %s", exc)
    log.info("Shutdown complete.")

atexit.register(_terminate_all)

def _handle_signal(signum: int, _frame: object) -> None:
    log.info("Signal %d — shutting down.", signum)
    _shutdown_event.set()
    _terminate_all()
    sys.exit(0)

for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, _handle_signal)
    except OSError:
        pass

def _fatal(title: str, message: str) -> None:
    global _fatal_fired
    with _fatal_lock:
        if _fatal_fired:
            return
        _fatal_fired = True

    log.error("[FATAL] %s — %s", title, message)
    _shutdown_event.set()
    _terminate_all()

    if IS_WINDOWS:
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, message, f"AInsights — {title}", 0x10
            )
        except Exception:
            pass
    sys.exit(1)

def _info_dialog(message: str) -> None:
    if not IS_WINDOWS:
        log.info("[INFO] %s", message)
        return
    try:
        import ctypes
        threading.Thread(
            target=ctypes.windll.user32.MessageBoxW,
            args=(0, message, "AInsights — Starting Up", 0x40),
            daemon=True,
        ).start()
    except Exception:
        pass

def _wait_http(
    url:     str,
    name:    str,
    timeout: int,
    thread:  Optional[threading.Thread] = None,
    proc:    Optional[subprocess.Popen] = None,
) -> bool:
    deadline = time.time() + timeout
    attempt  = 0
    while time.time() < deadline:
        if _shutdown_event.is_set():
            return False
        attempt += 1
        if thread is not None and not thread.is_alive():
            log.error("[%s] thread died — see log for traceback.", name)
            return False
        if proc is not None and proc.poll() is not None:
            log.error("[%s] process exited (code %d).", name, proc.returncode)
            return False
        try:
            if httpx.get(url, timeout=3).status_code == 200:
                log.info("[%s] healthy after %d poll(s).", name, attempt)
                return True
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)
    log.error("[%s] did not become healthy within %ds.", name, timeout)
    return False

def _free_port(port: int) -> None:
    if not IS_WINDOWS:
        return
    try:
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid   = int(parts[-1])
                if pid > 4:
                    log.info("Freeing port %d — killing PID %d", port, pid)
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/F"],
                        capture_output=True, timeout=5,
                    )
    except Exception as exc:
        log.debug("Port %d cleanup: %s", port, exc)


def start_ollama() -> bool:
    try:
        if httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3).status_code == 200:
            log.info("Ollama already running — skipping spawn.")
            return True
    except Exception:
        pass

    if OLLAMA_BIN is None:
        _fatal(
            "Ollama Not Found",
            "AInsights requires Ollama.\n\n"
            "Download from: https://ollama.com/download\n\n"
            "After installing, restart AInsights.",
        )
        return False

    log.info("Spawning Ollama from: %s", OLLAMA_BIN)
    try:
        subprocess.Popen(
            [str(OLLAMA_BIN), "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW,
        )
    except Exception as exc:
        log.error("Failed to spawn Ollama: %s", exc)
        return False

    return _wait_http(f"{OLLAMA_URL}/api/tags", "Ollama", OLLAMA_HEALTH_TIMEOUT)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Model
# ─────────────────────────────────────────────────────────────────────────────
def ensure_model() -> bool:
    try:
        models = [m["name"] for m in
                  httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10).json().get("models", [])]
        if any(OLLAMA_MODEL in m for m in models):
            log.info("Model '%s' already present.", OLLAMA_MODEL)
            return True
    except Exception as exc:
        log.warning("Could not check model list: %s", exc)

    log.info("Pulling model '%s' …", OLLAMA_MODEL)
    _info_dialog(
        f"Downloading AI model: {OLLAMA_MODEL}\n\n"
        "This only happens once on first launch (~4 GB).\n"
        "AInsights will open automatically when ready."
    )
    try:
        result = subprocess.run(
            [str(OLLAMA_BIN), "pull", OLLAMA_MODEL],
            timeout=MODEL_PULL_TIMEOUT,
            creationflags=_CREATE_NO_WINDOW,
        )
        if result.returncode == 0:
            log.info("Model pulled successfully.")
            return True
        log.error("Model pull failed (exit code %d).", result.returncode)
        return False
    except Exception as exc:
        log.error("Model pull error: %s", exc)
        return False


def start_backend() -> bool:
    global _uvicorn_server
    _free_port(8000)

    os.environ.update({
        "OLLAMA_BASE_URL":            OLLAMA_URL,
        "OLLAMA_MODEL":               OLLAMA_MODEL,
        "DATA_DIR":                   str(DATA_DIR),
        "UPLOAD_DIR":                 str(DATA_DIR / "uploads"),
        "CHROMA_PERSIST_DIR":         str(DATA_DIR / "vectorstore"),
        "BACKEND_HOST":               "127.0.0.1",
        "BACKEND_PORT":               "8000",
        "TRANSFORMERS_OFFLINE":       "1",
        "HF_DATASETS_OFFLINE":        "1",
        "HF_HOME":                    str(BASE_DIR / "hf_cache"),
        "SENTENCE_TRANSFORMERS_HOME": str(BASE_DIR / "hf_cache"),
    })

    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    log.info("Configuring uvicorn backend …")
    try:
        import uvicorn
        config = uvicorn.Config(
            "backend.main:app",
            host="127.0.0.1",
            port=8000,
            workers=1,
            log_level="info",
            log_config=None,   
            access_log=False,
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None
        _uvicorn_server = server
    except Exception as exc:
        log.error("Failed to configure uvicorn: %s", exc, exc_info=True)
        return False

    backend_thread = threading.Thread(
        target=_backend_thread,
        args=(server,),
        daemon=True,
        name="backend",
    )
    backend_thread.start()
    log.info("Backend thread started.")
    return _wait_http(
        f"{BACKEND_URL}/health",
        "Backend",
        BACKEND_HEALTH_TIMEOUT,
        thread=backend_thread,
    )


def _backend_thread(server) -> None:
    try:
        server.run()
    except Exception as exc:
        log.error("Backend crashed: %s", exc, exc_info=True)


def start_frontend() -> bool:
    global _frontend_proc
    _free_port(8501)

    env = {
        **os.environ,
        "BACKEND_URL":     BACKEND_URL,
        "OLLAMA_BASE_URL": OLLAMA_URL,
    }
    args = [sys.executable, "--ainsights-service", "frontend"]
    log.info("Spawning Streamlit via service mode …")
    try:
        _frontend_proc = subprocess.Popen(
            args,
            env=env,
            stdout=_log_fh,
            stderr=_log_fh,
            creationflags=_CREATE_NO_WINDOW,
            cwd=str(BASE_DIR),
        )
        log.info("[frontend] PID %d", _frontend_proc.pid)
    except Exception as exc:
        log.error("Failed to spawn frontend: %s", exc, exc_info=True)
        return False

    return _wait_http(
        f"{FRONTEND_URL}/_stcore/health",
        "Frontend",
        FRONTEND_HEALTH_TIMEOUT,
        proc=_frontend_proc,
    )


def _watchdog() -> None:
    while not _shutdown_event.is_set():
        if not any(t.name == "backend" for t in threading.enumerate()):
            log.error("Backend thread died unexpectedly.")
            _fatal("Backend Crashed",
                   f"The backend stopped unexpectedly.\n\nCheck:\n{LOG_FILE}")
            return
        if _frontend_proc and _frontend_proc.poll() is not None:
            log.error("Frontend exited (code %d).", _frontend_proc.returncode)
            _fatal("Frontend Crashed",
                   f"The UI stopped unexpectedly.\n\nCheck:\n{LOG_FILE}")
            return
        time.sleep(3)


def main() -> None:
    log.info("=" * 60)
    log.info("AInsights Desktop v1.0")
    log.info("Python  : %s", sys.version.split()[0])
    log.info("Frozen  : %s", IS_FROZEN)
    log.info("Base    : %s", BASE_DIR)
    log.info("Data    : %s", DATA_DIR)
    log.info("Log     : %s", LOG_FILE)
    log.info("=" * 60)

    steps = [
        ("Starting Ollama …",         start_ollama),
        ("Verifying AI model …",       ensure_model),
        ("Starting backend engine …",  start_backend),
        ("Starting UI server …",       start_frontend),
    ]

    for description, fn in steps:
        log.info(description)
        if not fn():
            _fatal(
                "Startup Failed",
                f"AInsights could not start at:\n\n  {description}\n\n"
                f"Check the log:\n{LOG_FILE}",
            )
            return

    threading.Thread(target=_watchdog, name="watchdog", daemon=True).start()
    log.info("All services healthy — opening browser.")
    webbrowser.open(FRONTEND_URL)
    log.info("Running at %s", FRONTEND_URL)

    try:
        while not _shutdown_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Keyboard interrupt.")
    finally:
        _terminate_all()


if __name__ == "__main__":
    main()
