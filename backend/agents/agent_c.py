"""
AInsights — Agent C: Analyst
Optimized dual-context analyst:
- Head snippet in Pandas prompt gives LLM concrete column value examples
- Synthesis prompt stripped to minimum tokens
- Pandas code exec is single-pass with safe builtins
- All LLM calls have explicit request_timeout
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

OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

MAX_PANDAS_RESULT_CHARS = 500
MAX_CONTEXT_CHARS       = 1000
MAX_ROWS_PANDAS         = 50_000
LLM_TIMEOUT             = 180

# ── Prompts ───────────────────────────────────────────────────────────────────
# Agent C uses two LLM calls:
# 1. Code LLM  → generate Pandas expression (small, fast, temperature=0)
# 2. Synthesis → write the final answer prose (temperature=0.1)
#
# Optimizations:
# - Pandas prompt ends at the code start marker — model completes it directly
# - head(3) shows concrete values so the model picks correct column names
# - Synthesis prompt caps response length explicitly (prevents rambling)
# - Both prompts avoid filler instructions; every sentence earns its tokens

_PANDAS_PROMPT = """\
Write a Python expression to answer the question using DataFrame `df`.

RULES:
- One expression or short code block only
- Last line must evaluate to the answer (scalar, Series ≤10 rows, or DataFrame ≤10 rows)
- Use only: pd, np, df
- dtype args use string form — 'object' not object
- If unanswerable with pandas: output exactly NOT_APPLICABLE
- No markdown, no imports, no explanation

COLUMNS ({n_rows} rows):
{schema}

HEAD:
{head}

QUESTION: {question}

Answer:"""


_SYNTHESIS_PROMPT = """\
You are AInsights, a BI analyst. Answer concisely in ≤200 words.

RULES:
- Use ONLY the context below — never invent numbers
- If Pandas Result exists, cite it as the authoritative figure
- Plain prose; bullet points only for 3+ items
- If data cannot answer: say so in one sentence

Pandas result (ground truth):
{pandas_result}

Retrieved document context:
{doc_context}

Retrieved data rows:
{tabular_context}

Question: {question}
Answer:"""


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

        self._llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=llm_base_url,
            temperature=0.1,
            num_predict=350,
            request_timeout=LLM_TIMEOUT,
        )
        self._code_llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=llm_base_url,
            temperature=0.0,
            num_predict=200,
            request_timeout=LLM_TIMEOUT,
        )

    # ── Public ────────────────────────────────────────────────────────────────
    def run(self, question: str, csv_path: Path) -> dict:
        log.info("Agent C: '%s'", question[:80])

        df = self._load(csv_path)
        if df is None:
            return {"answer": "Could not load the cleaned data file.",
                    "sources": [], "pandas_result": ""}

        rag_ctx      = self._retrieve(question)
        pandas_result= self._calculate(question, df)
        answer       = self._synthesise(question, rag_ctx, pandas_result)

        return {
            "answer":        answer,
            "sources":       self._sources(rag_ctx),
            "pandas_result": pandas_result,
        }

    async def stream(
        self, question: str, csv_path: Path
    ) -> AsyncGenerator[str, None]:
        import asyncio
        df = await asyncio.to_thread(self._load, csv_path)
        if df is None:
            yield "Could not load cleaned data."
            return
        rag_ctx       = await asyncio.to_thread(self._retrieve, question)
        pandas_result = await asyncio.to_thread(self._calculate, question, df)
        prompt        = self._synthesis_prompt(question, rag_ctx, pandas_result)
        for chunk in self._llm.stream(prompt):
            yield chunk

    # ── Load ──────────────────────────────────────────────────────────────────
    def _load(self, csv_path: Path) -> pd.DataFrame | None:
        try:
            return pd.read_csv(csv_path, nrows=MAX_ROWS_PANDAS)
        except Exception as exc:
            log.error("CSV load failed: %s", exc)
            return None

    # ── RAG retrieval ─────────────────────────────────────────────────────────
    def _retrieve(self, question: str) -> dict[str, list[str]]:
        try:
            return self._rag.query(question)
        except Exception as exc:
            log.warning("RAG failed: %s", exc)
            return {"doc_chunks": [], "table_chunks": []}

    def _sources(self, ctx: dict) -> list[str]:
        seen: list[str] = []
        for chunk in ctx.get("table_chunks", []):
            m = re.match(r"(Row \d+)", chunk)
            if m and m.group(1) not in seen:
                seen.append(m.group(1))
        return seen[:8]

    # ── Pandas calculation ────────────────────────────────────────────────────
    def _calculate(self, question: str, df: pd.DataFrame) -> str:
        # Build compact schema + head
        schema_lines = []
        for col in df.columns[:20]:
            schema_lines.append(
                f"  {col} ({df[col].dtype}): {df[col].dropna().head(2).tolist()}"
            )
        if len(df.columns) > 20:
            schema_lines.append(f"  … +{len(df.columns)-20} more")

        try:
            head_str = df.head(3).to_string(
                max_cols=12, max_colwidth=20, show_dimensions=False
            )
        except Exception:
            head_str = ""

        prompt = _PANDAS_PROMPT.format(
            n_rows   = len(df),
            schema   = "\n".join(schema_lines),
            head     = head_str,
            question = question,
        )

        for attempt in range(2):
            try:
                raw = self._code_llm.invoke(prompt).strip()
                raw = re.sub(r"```(?:python)?|```", "", raw).strip()

                if not raw or raw == "NOT_APPLICABLE":
                    return ""

                result = self._exec(raw, df)
                fmt    = self._fmt(result)
                if fmt:
                    return fmt

            except Exception as exc:
                log.warning("Pandas calc attempt %d: %s", attempt + 1, exc)

        return ""

    def _exec(self, code: str, df: pd.DataFrame) -> Any:
        import numpy as np

        builtins = {
            "object": object, "type": type,
            "bool": bool, "int": int, "float": float,
            "str": str, "list": list, "dict": dict,
            "set": set, "tuple": tuple,
            "len": len, "range": range, "enumerate": enumerate,
            "zip": zip, "sorted": sorted, "reversed": reversed,
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "any": any, "all": all,
            "isinstance": isinstance, "callable": callable,
            "hasattr": hasattr, "getattr": getattr,
            "print": print,
            "ValueError": ValueError, "TypeError": TypeError,
            "KeyError": KeyError, "IndexError": IndexError,
            "Exception": Exception,
        }
        g: dict = {"__builtins__": builtins, "pd": pd, "np": np, "df": df.copy()}
        local: dict = {}

        lines = code.strip().splitlines()
        if len(lines) == 1:
            return eval(lines[0], g)          # noqa: S307

        exec("\n".join(lines[:-1]), g, local) # noqa: S102
        g.update(local)
        try:
            return eval(lines[-1], g)         # noqa: S307
        except SyntaxError:
            exec(lines[-1], g, local)         # noqa: S102
            return list(local.values())[-1] if local else None

    def _fmt(self, result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, pd.DataFrame):
            return "No matching rows." if result.empty else result.head(8).to_string(index=False)
        if isinstance(result, pd.Series):
            return result.head(8).to_string()
        if isinstance(result, float):
            return f"{result:,.4f}"
        return textwrap.shorten(str(result), width=MAX_PANDAS_RESULT_CHARS, placeholder=" …")

    # ── Synthesis ─────────────────────────────────────────────────────────────
    def _synthesis_prompt(
        self, question: str, ctx: dict, pandas_result: str
    ) -> str:
        def _trunc(chunks: list[str], limit: int) -> str:
            return textwrap.shorten(
                "\n".join(chunks) or "None.",
                width=limit, placeholder=" [truncated]"
            )

        return _SYNTHESIS_PROMPT.format(
            question        = question,
            pandas_result   = textwrap.shorten(
                                  pandas_result or "None.",
                                  width=MAX_PANDAS_RESULT_CHARS, placeholder=" …"),
            doc_context     = _trunc(ctx.get("doc_chunks",   []), MAX_CONTEXT_CHARS),
            tabular_context = _trunc(ctx.get("table_chunks", []), MAX_CONTEXT_CHARS),
        )

    def _synthesise(
        self, question: str, ctx: dict, pandas_result: str
    ) -> str:
        prompt = self._synthesis_prompt(question, ctx, pandas_result)
        try:
            answer = self._llm.invoke(prompt).strip()
            answer = re.sub(r"^(Answer:|Response:)\s*", "", answer,
                            flags=re.IGNORECASE)
            return answer
        except Exception as exc:
            log.error("Synthesis failed: %s", exc)
            if pandas_result:
                return (
                    f"The calculation returned: **{pandas_result}**\n\n"
                    "_(Language model synthesis failed — the number above is accurate.)_"
                )
            return ("I encountered an error. "
                    "Ensure Ollama is running and the model is loaded.")