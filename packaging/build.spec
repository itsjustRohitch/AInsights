# -*- mode: python ; coding: utf-8 -*-
"""
AInsights — PyInstaller spec file 

"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(SPECPATH).parent  # project root

# ─────────────────────────────────────────────────────────────────────────────
# Resolve source paths on the build machine
# ─────────────────────────────────────────────────────────────────────────────
import transformers as _tf
import sentence_transformers as _st

TRANSFORMERS_SRC = Path(_tf.__file__).parent
ST_SRC           = Path(_st.__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Data files
# ─────────────────────────────────────────────────────────────────────────────
datas = []

# Standard asset collection (JSON, CSS, fonts, SQL etc.)
datas += collect_data_files("streamlit")
datas += collect_data_files("altair")
datas += collect_data_files("chromadb")
datas += collect_data_files("pdfminer")
datas += collect_data_files("tokenizers")

# ── Full source trees — REQUIRED for os.listdir() at runtime ─────────────────
# transformers/utils/import_utils.py calls os.listdir(module_path) where
# module_path is a live filesystem path. This path must actually exist.
datas.append((str(TRANSFORMERS_SRC),   "transformers"))
datas.append((str(ST_SRC),             "sentence_transformers"))

# ── HuggingFace cached model — bundles embedding model for offline use ────────
hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
_model_found = False
if hf_cache.exists():
    for _d in hf_cache.glob("models--sentence-transformers--all-MiniLM-L6-v2"):
        datas.append((str(_d), f"hf_cache/{_d.name}"))
        _model_found = True
        print(f"[spec] HF model bundled: {_d.name}")
if not _model_found:
    print("[spec] WARNING: HF model cache not found — run build.py to pre-download it")

# ── Application source ────────────────────────────────────────────────────────
datas += [
    (str(ROOT / "backend"),      "backend"),
    (str(ROOT / "frontend"),     "frontend"),
    (str(ROOT / ".env.example"), ".env"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Native binaries
# ─────────────────────────────────────────────────────────────────────────────
binaries  = collect_dynamic_libs("chromadb")
binaries += collect_dynamic_libs("onnxruntime")

# ─────────────────────────────────────────────────────────────────────────────
# Hidden imports
# ─────────────────────────────────────────────────────────────────────────────
hiddenimports = [

    # Uvicorn
    "uvicorn", "uvicorn.config", "uvicorn.main", "uvicorn.server",
    "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto",
    "uvicorn.loops.asyncio", "uvicorn.protocols",
    "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.protocols.websockets.wsproto_impl",
    "uvicorn.lifespan", "uvicorn.lifespan.on",
    "uvicorn.middleware", "uvicorn.middleware.proxy_headers",

    # FastAPI / Starlette
    "fastapi", "fastapi.middleware", "fastapi.middleware.cors",
    "starlette", "starlette.middleware", "starlette.middleware.cors",
    "aiofiles", "aiofiles.os", "aiofiles.threadpool",

    # LangChain
    "langchain_ollama", "langchain_community", "langchain",
    "langchain.schema", "langchain.callbacks",

    # ChromaDB
    "chromadb", "chromadb.api", "chromadb.api.client",
    "chromadb.db", "chromadb.db.impl", "chromadb.db.impl.sqlite",
    "chromadb.migrations", "chromadb.telemetry", "chromadb.segment",
    "chromadb.segment.impl", "chromadb.segment.impl.vector",
    "chromadb.segment.impl.vector.local_persistent_hnsw",

    # Transformers / sentence-transformers
    # Source is on disk (see datas) — hiddenimports covers compiled entry points
    "sentence_transformers",
    "sentence_transformers.models",
    "sentence_transformers.cross_encoder",
    "sentence_transformers.cross_encoder.CrossEncoder",
    "transformers",
    "transformers.models",
    "transformers.utils",
    "transformers.utils.import_utils",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
    "safetensors.torch",

    # Data
    "pandas", "numpy", "openpyxl", "xlrd",
    "lxml", "lxml.etree", "pdfplumber",
    "pdfminer", "pdfminer.high_level", "pdfminer.layout",
    "chardet", "pyarrow",

    # Visualisation
    "plotly", "plotly.express", "plotly.graph_objects", "altair",

    # Streamlit
    "streamlit", "streamlit.web", "streamlit.web.cli",
    "streamlit.web.server", "streamlit.web.server.server",
    "streamlit.runtime", "streamlit.runtime.scriptrunner",
    "streamlit.runtime.state", "streamlit.runtime.uploaded_file_manager",
    "streamlit.elements", "streamlit.components", "streamlit.components.v1",

    # Multiprocessing (sandbox/executor.py uses threading now, but
    # ChromaDB and other deps may still import multiprocessing)
    "multiprocessing",
    "multiprocessing.popen_spawn_win32",
    "multiprocessing.synchronize",
    "multiprocessing.managers",

    # Async / HTTP
    "httpx", "httpcore", "anyio", "anyio._backends._asyncio",
    "h11", "websockets", "wsproto",

    # Pydantic
    "pydantic", "pydantic.v1", "pydantic_core", "pydantic_settings",

    # Misc
    "dotenv", "sklearn", "sklearn.metrics", "sklearn.preprocessing",
    "scipy", "scipy.spatial", "PIL", "PIL.Image", "zstandard",
]

# Broad sweeps for packages with many dynamic sub-imports
hiddenimports += collect_submodules("streamlit")
hiddenimports += collect_submodules("chromadb")
hiddenimports += collect_submodules("langchain")
hiddenimports += collect_submodules("langchain_community")
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("sentence_transformers")

# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────
a = Analysis(
    [str(ROOT / "packaging" / "orchestrator.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(ROOT / "packaging" / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "matplotlib.pyplot",
        "IPython", "jupyter", "notebook",
        "pytest", "mypy", "black", "ruff",
        "PyQt5", "PyQt6", "PySide2", "PySide6",
        "wx", "gi", "tkinter", "_tkinter", "tornado",
    ],
    noarchive=False,
    # optimize=0: safest — higher levels strip __doc__ strings that dynamic
    # importers like transformers sometimes inspect at runtime
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AInsights",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no CMD window on Windows
    disable_windowed_traceback=False,
    icon=str(ROOT / "frontend" / "assets" / "icon.ico")
         if (ROOT / "frontend" / "assets" / "icon.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=["vcruntime*.dll", "msvcp*.dll", "python*.dll", "*.pyd"],
    name="AInsights",
)
