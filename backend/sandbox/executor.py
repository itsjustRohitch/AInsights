"""
SafeExecutor — restricted Python sandbox for Agent A.
Runs LLM-generated code with a timeout and a restricted set of allowed globals.
Uses multiprocessing to enforce the timeout without threading issues.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import textwrap
from typing import Any

import pandas as pd

log = logging.getLogger("ainsights.sandbox")

SANDBOX_TIMEOUT = 30   # seconds — kill the process if it runs longer


def _worker(code: str, df_serialised: bytes, result_queue: mp.Queue) -> None:
    """
    Runs inside a child process. Deserialises the DataFrame, executes the
    LLM-generated clean() function, and puts the result back on the queue.
    """
    import io
    import numpy as np
    import pandas as pd
    import re

    try:
        df = pd.read_parquet(io.BytesIO(df_serialised))

        # Restricted globals — LLM code can ONLY access these names
        safe_globals: dict[str, Any] = {
            "__builtins__": {
                "len": len, "range": range, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter,
                "list": list, "dict": dict, "set": set, "tuple": tuple,
                "str": str, "int": int, "float": float, "bool": bool,
                "print": print, "isinstance": isinstance, "type": type,
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "sorted": sorted, "reversed": reversed,
                "any": any, "all": all,
            },
            "pd": pd,
            "np": np,
            "re": re,
        }
        local_ns: dict = {}
        exec(code, safe_globals, local_ns)   # noqa: S102

        if "clean" not in local_ns:
            result_queue.put(("error", "No 'clean' function found in generated code."))
            return

        cleaned_df = local_ns["clean"](df)

        if not isinstance(cleaned_df, pd.DataFrame):
            result_queue.put(("error", "clean() did not return a DataFrame."))
            return

        buf = io.BytesIO()
        cleaned_df.to_parquet(buf, index=True)
        result_queue.put(("ok", buf.getvalue()))

    except Exception as exc:
        result_queue.put(("error", str(exc)))


class SafeExecutor:
    """Executes LLM-generated cleaning code in an isolated subprocess."""

    def run_cleaning_function(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """
        Execute the LLM-generated clean(df) function in a child process.
        Raises RuntimeError if execution fails or times out.
        """
        import io

        # Serialise the DataFrame for IPC via Parquet (preserves dtypes)
        buf = io.BytesIO()
        df.to_parquet(buf, index=True)
        df_bytes = buf.getvalue()

        queue: mp.Queue = mp.Queue()
        proc = mp.Process(
            target=_worker,
            args=(code, df_bytes, queue),
            daemon=True,
        )
        proc.start()
        proc.join(timeout=SANDBOX_TIMEOUT)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            raise RuntimeError(
                f"Sandbox timeout: LLM code did not complete within {SANDBOX_TIMEOUT}s."
            )

        if queue.empty():
            raise RuntimeError("Sandbox returned no result (process crashed).")

        status, payload = queue.get_nowait()
        if status == "error":
            raise RuntimeError(f"Sandbox execution error: {payload}")

        result_df = pd.read_parquet(io.BytesIO(payload))
        log.info("Sandbox execution successful. Output shape: %s", result_df.shape)
        return result_df