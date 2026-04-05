"""
AInsights — Agent C: The Analyst (v2)
Dual-context conversational BI.
Fixes: LLM request timeout, context truncation, retry logic,
       more robust Pandas code execution, tighter prompts for qwen2.5-coder:7b.
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Any, AsyncGenerator

import pandas as pd
from langchain_ollama import OllamaLLM

log = logging.getLogger("ainsights.agent_c")

OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Context budget — keep prompts lean for the 7b model
MAX_DOC_CONTEXT_CHARS   = 1200
MAX_TABLE_CONTEXT_CHARS = 1200
MAX_PANDAS_RESULT_CHARS = 600
MAX_SCHEMA_CHARS        = 800
MAX_ROWS_PANDAS         = 50_000

# Ollama request timeout — 7b model on CPU can take 90-150s
LLM_TIMEOUT = 180   # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Prompts — kept tight for qwen2.5-coder:7b
# ─────────────────────────────────────────────────────────────────────────────

_PANDAS_PROMPT = """\
You are a Python data analyst. Write Python code to answer the question using a pandas DataFrame called `df`.

RULES:
1. Output ONLY raw Python code. No markdown, no explanation, no triple backticks.
2. The last line must be an expression that evaluates to the answer (scalar, Series, or small DataFrame ≤10 rows).
3. Use only: pandas (pd), numpy (np). No other imports.
4. Do not modify df in-place. Use .copy() if needed.
5. If the question cannot be answered with pandas, output exactly: NOT_APPLICABLE

DataFrame columns:
{schema}

Question: {question}

Code:"""


_SYNTHESIS_PROMPT = """\
You are AInsights, a professional BI analyst. Answer the business question below.

RULES:
1. Use ONLY the provided context. Never invent numbers.
2. If Pandas Result is present, always reference it as the authoritative number.
3. Write clear, concise professional prose. Use bullet points for 3+ items only.
4. Keep response under 300 words.
5. If the data cannot answer the question, say so clearly.

Pandas calculation result (ground truth numbers):
{pandas_result}

Retrieved document excerpts:
{doc_context}

Retrieved data rows:
{tabular_context}

Question: {question}

Answer:"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────
class AnalystAgent:

    def __init__(
        self,
        rag_engine: Any,
        llm_base_url: str = OLLAMA_BASE_URL,
        streaming: bool = False,
    ) -> None:
        self._rag    = rag_engine
        self._stream = streaming

        # Separate LLM instances with explicit timeouts
        self._llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=llm_base_url,
            temperature=0.1,
            num_predict=400,
            request_timeout=LLM_TIMEOUT,
        )
        self._code_llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=llm_base_url,
            temperature=0.0,
            num_predict=200,
            request_timeout=LLM_TIMEOUT,
        )

    # ── Public API ────────────────────────────────────────────────────────────
    def run(self, question: str, csv_path: Path) -> dict:
        log.info("Agent C: '%s'", question[:80])

        df = self._load_csv(csv_path)
        if df is None:
            return {
                "answer":        "Could not load the cleaned data file.",
                "sources":       [],
                "pandas_result": "",
            }

        # Step 1: RAG retrieval
        rag_context   = self._retrieve(question)

        # Step 2: Pandas calculation
        pandas_result = self._calculate(question, df)

        # Step 3: Synthesis
        answer        = self._synthesise(question, rag_context, pandas_result)
        sources       = self._extract_sources(rag_context)

        return {
            "answer":        answer,
            "sources":       sources,
            "pandas_result": pandas_result,
        }

    async def stream(
        self, question: str, csv_path: Path
    ) -> AsyncGenerator[str, None]:
        import asyncio

        df = await asyncio.to_thread(self._load_csv, csv_path)
        if df is None:
            yield "Could not load cleaned data."
            return

        rag_context   = await asyncio.to_thread(self._retrieve, question)
        pandas_result = await asyncio.to_thread(self._calculate, question, df)
        prompt        = self._build_synthesis_prompt(question, rag_context, pandas_result)

        for chunk in self._llm.stream(prompt):
            yield chunk

    # ── Data loading ──────────────────────────────────────────────────────────
    def _load_csv(self, csv_path: Path) -> pd.DataFrame | None:
        try:
            return pd.read_csv(csv_path, nrows=MAX_ROWS_PANDAS)
        except Exception as exc:
            log.error("Failed to load CSV: %s", exc)
            return None

    # ── RAG retrieval ─────────────────────────────────────────────────────────
    def _retrieve(self, question: str) -> dict[str, list[str]]:
        try:
            return self._rag.query(question)
        except Exception as exc:
            log.warning("RAG retrieval failed: %s", exc)
            return {"doc_chunks": [], "table_chunks": []}

    def _extract_sources(self, rag_context: dict) -> list[str]:
        sources: list[str] = []
        for chunk in rag_context.get("table_chunks", []):
            m = re.match(r"(Row \d+)", chunk)
            if m and m.group(1) not in sources:
                sources.append(m.group(1))
        return sources[:8]

    # ── Pandas calculation ────────────────────────────────────────────────────
    def _calculate(self, question: str, df: pd.DataFrame) -> str:
        schema = self._compact_schema(df)
        prompt = _PANDAS_PROMPT.format(schema=schema, question=question)

        for attempt in range(2):   # retry once on failure
            try:
                raw = self._code_llm.invoke(prompt).strip()
                raw = re.sub(r"```(?:python)?|```", "", raw).strip()

                if not raw or raw == "NOT_APPLICABLE":
                    return ""

                result = self._safe_exec(raw, df)
                formatted = self._format_result(result)
                if formatted:
                    return formatted

            except Exception as exc:
                log.warning("Pandas calc attempt %d failed: %s", attempt + 1, exc)

        return ""

    def _compact_schema(self, df: pd.DataFrame) -> str:
        """Compact schema string — stays within token budget."""
        lines = []
        for col in df.columns[:25]:   # cap at 25 cols to keep prompt short
            dtype  = str(df[col].dtype)
            sample = df[col].dropna().head(2).tolist()
            lines.append(f"  {col} ({dtype}): e.g. {sample}")
        if len(df.columns) > 25:
            lines.append(f"  … and {len(df.columns) - 25} more columns")
        lines.append(f"  Total rows: {len(df):,}")
        return textwrap.shorten("\n".join(lines), width=MAX_SCHEMA_CHARS, placeholder=" …")

    def _safe_exec(self, code: str, df: pd.DataFrame) -> Any:
        """Execute LLM-generated Pandas code in a restricted namespace."""
        import numpy as np

        safe_builtins = {
            "len": len, "range": range, "list": list, "dict": dict,
            "str": str, "int": int, "float": float, "bool": bool,
            "round": round, "abs": abs, "min": min, "max": max,
            "sum": sum, "sorted": sorted, "print": print,
            "isinstance": isinstance, "enumerate": enumerate,
            "zip": zip, "any": any, "all": all,
        }
        safe_globals: dict = {
            "__builtins__": safe_builtins,
            "pd": pd, "np": np,
            "df": df.copy(),
        }
        local_ns: dict = {}
        lines = code.strip().split("\n")

        if len(lines) == 1:
            return eval(lines[0], safe_globals)   # noqa: S307

        setup = "\n".join(lines[:-1])
        last  = lines[-1].strip()
        exec(setup, safe_globals, local_ns)        # noqa: S102
        safe_globals.update(local_ns)

        # Try the last line as an expression; fall back to exec
        try:
            return eval(last, safe_globals)        # noqa: S307
        except SyntaxError:
            exec(last, safe_globals, local_ns)     # noqa: S102
            # Return the last assigned variable if possible
            if local_ns:
                return list(local_ns.values())[-1]
            return None

    def _format_result(self, result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, pd.DataFrame):
            if result.empty:
                return "No matching rows found."
            return result.head(8).to_string(index=False)
        if isinstance(result, pd.Series):
            return result.head(8).to_string()
        if isinstance(result, float):
            return f"{result:,.4f}"
        formatted = str(result)
        return textwrap.shorten(formatted, width=MAX_PANDAS_RESULT_CHARS, placeholder=" …")

    # ── Synthesis ─────────────────────────────────────────────────────────────
    def _build_synthesis_prompt(
        self,
        question: str,
        rag_context: dict,
        pandas_result: str,
    ) -> str:
        doc_chunks   = rag_context.get("doc_chunks",   [])
        table_chunks = rag_context.get("table_chunks", [])

        doc_context = textwrap.shorten(
            "\n\n".join(doc_chunks) or "None.",
            width=MAX_DOC_CONTEXT_CHARS,
            placeholder=" [truncated]",
        )
        tabular_context = textwrap.shorten(
            "\n".join(table_chunks) or "None.",
            width=MAX_TABLE_CONTEXT_CHARS,
            placeholder=" [truncated]",
        )
        pandas_str = textwrap.shorten(
            pandas_result or "No calculation result.",
            width=MAX_PANDAS_RESULT_CHARS,
            placeholder=" …",
        )

        return _SYNTHESIS_PROMPT.format(
            question=question,
            pandas_result=pandas_str,
            doc_context=doc_context,
            tabular_context=tabular_context,
        )

    def _synthesise(
        self,
        question: str,
        rag_context: dict,
        pandas_result: str,
    ) -> str:
        prompt = self._build_synthesis_prompt(question, rag_context, pandas_result)
        try:
            answer = self._llm.invoke(prompt).strip()
            # Clean any markdown artefacts the model might add
            answer = re.sub(r"^(Answer:|Response:)\s*", "", answer, flags=re.IGNORECASE)
            return answer
        except Exception as exc:
            log.error("LLM synthesis failed: %s", exc)
            if pandas_result:
                return (
                    f"The data calculation returned: **{pandas_result}**\n\n"
                    "_(The language model encountered an error generating a full "
                    "response. The calculation result above is accurate.)_"
                )
            return (
                "I encountered an error generating a response. "
                "Please check that Ollama is running and the model is loaded."
            )