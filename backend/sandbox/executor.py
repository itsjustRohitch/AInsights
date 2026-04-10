"""
AInsights — Safe Sandbox Executor v2

Replaces multiprocessing with threading.
Rationale: multiprocessing.spawn requires the child process to re-import
all modules — this fails silently inside uvicorn --reload because the
module paths are not set up identically in the child. The queue returns
empty, both LLM attempts raise RuntimeError, and we always fall back.

Threading approach:
  - No pickling / spawn issues
  - Same restricted __builtins__ namespace
  - Hard timeout via thread.join(timeout)
  - Daemon thread — won't block process exit if it overruns
  - Fully compatible with uvicorn, asyncio, and PyInstaller
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("ainsights.sandbox")

SANDBOX_TIMEOUT = 60  


_SAFE_BUILTINS: dict[str, Any] = {
    # Core types
    "object":     object,
    "type":       type,
    "bool":       bool,
    "int":        int,
    "float":      float,
    "str":        str,
    "bytes":      bytes,
    "list":       list,
    "dict":       dict,
    "set":        set,
    "tuple":      tuple,
    "frozenset":  frozenset,
    "None":       None,
    "True":       True,
    "False":      False,
    # Iterables
    "len":        len,
    "range":      range,
    "enumerate":  enumerate,
    "zip":        zip,
    "map":        map,
    "filter":     filter,
    "reversed":   reversed,
    "sorted":     sorted,
    "iter":       iter,
    "next":       next,
    # Math
    "abs":        abs,
    "round":      round,
    "min":        min,
    "max":        max,
    "sum":        sum,
    "pow":        pow,
    "divmod":     divmod,
    # Logic / introspection
    "any":        any,
    "all":        all,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "callable":   callable,
    "hasattr":    hasattr,
    "getattr":    getattr,
    "setattr":    setattr,
    "print":      print,
    "repr":       repr,
    "str":        str,
    # Exceptions the LLM may catch
    "Exception":       Exception,
    "ValueError":      ValueError,
    "TypeError":       TypeError,
    "KeyError":        KeyError,
    "IndexError":      IndexError,
    "AttributeError":  AttributeError,
    "StopIteration":   StopIteration,
    "RuntimeError":    RuntimeError,
    "NotImplementedError": NotImplementedError,
}

class SafeExecutor:

    def run_cleaning_function(
        self, df: pd.DataFrame, code: str
    ) -> pd.DataFrame:
        result: list[pd.DataFrame | None] = [None]
        error:  list[Exception | None]    = [None]

        def _run() -> None:
            try:
                safe_globals: dict[str, Any] = {
                    "__builtins__": _SAFE_BUILTINS,
                    "pd":  pd,
                    "np":  np,
                    "re":  re,
                }
                local_ns: dict[str, Any] = {}

                exec(code, safe_globals, local_ns)  # noqa: S102

                if "clean" not in local_ns:
                    raise RuntimeError(
                        "Generated code did not define a function named 'clean'.\n"
                        f"Defined names: {list(local_ns.keys())}"
                    )

                cleaned = local_ns["clean"](df)

                if not isinstance(cleaned, pd.DataFrame):
                    raise RuntimeError(
                        f"clean() returned {type(cleaned).__name__}, "
                        "expected pd.DataFrame."
                    )

                result[0] = cleaned

            except Exception as exc:
                error[0] = exc

        thread = threading.Thread(target=_run, daemon=True, name="sandbox-exec")
        thread.start()
        thread.join(timeout=SANDBOX_TIMEOUT)

        if thread.is_alive():
            raise RuntimeError(
                f"Sandbox timeout: code did not complete within {SANDBOX_TIMEOUT}s."
            )

        if error[0] is not None:
            raise RuntimeError(f"Sandbox execution error: {error[0]}") from error[0]

        if result[0] is None:
            raise RuntimeError("Sandbox returned no result (unknown failure).")

        log.info("Sandbox OK — output shape: %s", result[0].shape)
        return result[0]