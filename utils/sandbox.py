from __future__ import annotations

import ast
import multiprocessing as mp
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


BLOCKED_NODE_TYPES = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.Try,
    ast.While,
    ast.For,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.Global,
    ast.Nonlocal,
)

BLOCKED_NAMES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "builtins",
    "eval",
    "exec",
    "open",
    "compile",
    "__import__",
    "input",
    "locals",
    "globals",
    "help",
    "dir",
    "getattr",
    "setattr",
    "delattr",
}


def _validate_code(code: str) -> None:
    tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, BLOCKED_NODE_TYPES):
            raise ValueError(f"Blocked code construct: {type(node).__name__}")

        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ValueError(f"Blocked name usage: {node.id}")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_NAMES:
                raise ValueError(f"Blocked function call: {node.func.id}")

    # Hard guard: disallow double-underscore attributes often used for introspection
    if re.search(r"\.__[A-Za-z_][A-Za-z0-9_]*__", code):
        raise ValueError("Dunder attribute access is blocked.")


def _safe_builtins() -> dict[str, Any]:
    allowed = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    return allowed


def _worker(code: str, df: pd.DataFrame, conn) -> None:
    try:
        _validate_code(code)

        safe_globals = {
            "__builtins__": _safe_builtins(),
            "pd": pd,
            "np": np,
            "re": re,
            "datetime": datetime,
        }
        safe_locals = {
            "df": df.copy(),
            "cleaned_df": None,
        }

        exec(code, safe_globals, safe_locals)

        result = safe_locals.get("cleaned_df", safe_locals.get("df"))
        if not isinstance(result, pd.DataFrame):
            raise ValueError("Cleaning code must leave a pandas DataFrame in df or cleaned_df.")

        conn.send(("ok", result))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        conn.close()


@dataclass
class SandboxResult:
    ok: bool
    dataframe: pd.DataFrame | None = None
    error: str | None = None


def run_cleaning_code(code: str, df: pd.DataFrame, timeout_seconds: int = 5) -> SandboxResult:
    """
    Execute LLM-generated cleaning code in a separate process.
    Hard timeout: process gets killed if it exceeds the budget.
    """
    if not isinstance(code, str) or not code.strip():
        return SandboxResult(ok=False, error="Empty cleaning code.")

    parent_conn, child_conn = mp.Pipe(duplex=False)
    ctx = mp.get_context("spawn")
    process = ctx.Process(target=_worker, args=(code, df, child_conn))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return SandboxResult(ok=False, error="Sandbox timeout exceeded.")

    if parent_conn.poll():
        status, payload = parent_conn.recv()
        if status == "ok":
            cleaned_df = payload

            # Enforce no column renaming
            if list(cleaned_df.columns) != list(df.columns):
                return SandboxResult(ok=False, error="Column renaming or reordering is not allowed.")

            return SandboxResult(ok=True, dataframe=cleaned_df)

        return SandboxResult(ok=False, error=str(payload))

    return SandboxResult(ok=False, error="Sandbox returned no result.")