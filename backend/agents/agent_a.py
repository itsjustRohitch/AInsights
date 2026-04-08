from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
from langchain_ollama import OllamaLLM

from backend.sandbox.executor import SafeExecutor
from backend.utils.file_parser import load_file
from backend.utils.schema_profiler import compact_schema_str, profile

log = logging.getLogger("ainsights.agent_a")

OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


_PROMPT = """\
Act as a Senior Data Engineer. Analyze the schema and sample data to identify inconsistencies and formatting errors. 
Provide a clean, robust implementation that ensures data integrity.

Rules:
- No imports
- dtype must be strings
- Keep column names unchanged
- Last line must be: return df

Write ONLY the indented body of this function.

def clean(df):
    return df

DATA:
{schema}

SAMPLE:
{head}
"""


class DataEngineerAgent:

    def __init__(self, llm_base_url: str = OLLAMA_BASE_URL) -> None:
        self._llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=llm_base_url,
            temperature=0.0,
            num_predict=2000,
            request_timeout=180,
        )
        self._executor = SafeExecutor()

    # ─────────────────────────────────────────────
    def run(self, input_path: Path, output_path: Path) -> dict:
        t0 = time.perf_counter()

        log.info("Agent A: processing '%s'", input_path.name)

        df_raw = load_file(input_path)
        original_columns = df_raw.columns.tolist()
        original_shape = df_raw.shape

        log.info("Loaded: %d rows × %d cols", *original_shape)

        schema = profile(df_raw)

        df_clean, method = self._clean(df_raw.copy(), schema)

        df_clean = self._restore_columns(df_clean, df_raw, original_columns)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)

        elapsed = round(time.perf_counter() - t0, 2)

        result = {
            "input_file": input_path.name,
            "original_shape": list(original_shape),
            "cleaned_shape": list(df_clean.shape),
            "rows_removed": original_shape[0] - df_clean.shape[0],
            "null_cells_before": int(df_raw.isna().sum().sum()),
            "null_cells_after": int(df_clean.isna().sum().sum()),
            "cleaning_method": method,
            "elapsed_seconds": elapsed,
        }

        log.info(
            "Agent A complete in %.2fs via '%s'. Shape %s→%s, nulls %d→%d",
            elapsed,
            method,
            original_shape,
            df_clean.shape,
            result["null_cells_before"],
            result["null_cells_after"],
        )

        return result

    # ─────────────────────────────────────────────
    def _clean(self, df: pd.DataFrame, schema: dict):

        head = schema.get("head", "")
        if isinstance(head, str) and len(head) > 800:
            head = head[:800] + "\n..."

        prompt = _PROMPT.format(
            schema=compact_schema_str(schema),
            head=head,
        )

        for attempt in range(1, 3):
            log.info("LLM cleaning attempt %d/2 …", attempt)

            try:
                raw = self._llm.invoke(prompt)

                log.debug(
                    "LLM raw output (attempt %d):\n%s",
                    attempt,
                    raw[:800],
                )

                if self._looks_truncated(raw):
                    log.warning(
                        "Attempt %d: output appears truncated. Raw tail:\n%s",
                        attempt,
                        raw[-200:],
                    )
                    continue

                body = self._extract(raw)

                if not body:
                    log.warning(
                        "Attempt %d: extraction failed. Raw output snippet:\n%s",
                        attempt,
                        raw[:400],
                    )
                    continue

                full_code = "def clean(df):\n" + body

                log.debug(
                    "Executing code (attempt %d):\n%s",
                    attempt,
                    full_code[:800],
                )

                df_result = self._executor.run_cleaning_function(df.copy(), full_code)

                ok, reason = self._safe(df_result, df)

                if ok:
                    log.info("LLM cleaning accepted on attempt %d.", attempt)
                    return df_result, f"llm_attempt_{attempt}"

                log.warning(
                    "Attempt %d: safety check rejected result — %s",
                    attempt,
                    reason,
                )

            except Exception as e:
                log.warning("Attempt %d failed: %s", attempt, e)

        log.warning("All LLM attempts exhausted — using rule-based fallback.")

        return self._rule_based(df, schema), "rule_based_fallback"

    # ─────────────────────────────────────────────
    def _looks_truncated(self, raw: str) -> bool:
        raw = raw.strip()
        if not raw:
            return True

        # Odd number of fences means an opening fence with no closing fence
        if raw.count("```") % 2 == 1:
            return True

        last = raw.splitlines()[-1].strip()

        # These characters genuinely indicate an incomplete statement.
        # Removed '"' and "'" — a line ending with a closed string is valid.
        return last.endswith(("=", ",", "(", "[", "{", ":"))

    # ─────────────────────────────────────────────
    def _extract(self, raw: str) -> str:
        # Remove markdown fences
        raw = re.sub(r"```(?:python)?", "", raw)
        raw = raw.replace("```", "")

        if "def clean" in raw:
            m = re.search(r"def clean\(.*?\):(.*)", raw, re.DOTALL)
            if m:
                raw = m.group(1)

        raw = raw.strip("\n")

        if "return" not in raw:
            log.debug("_extract: no return found.")
            return ""

        lines = raw.split("\n")
        clean_lines = []

        for ln in lines:
            if ln.strip():
                clean_lines.append("    " + ln.strip())
            else:
                clean_lines.append("")

        body = "\n".join(clean_lines)

        try:
            compile("def clean(df):\n" + body, "<llm>", "exec")
        except SyntaxError as e:
            log.warning("_extract: SyntaxError after normalization: %s", e)
            log.debug("Problematic body:\n%s", body[:400])
            return ""

        return body

    # ─────────────────────────────────────────────
    def _safe(self, result: pd.DataFrame, original: pd.DataFrame):

        missing = set(original.columns) - set(result.columns)
        if missing:
            return False, f"missing columns: {missing}"

        if len(original) > 0:
            retention = len(result) / len(original)
            if retention < 0.70:
                return False, f"dropped {(1-retention)*100:.1f}% rows"

        before = int(original.isna().sum().sum())
        after = int(result.isna().sum().sum())

        if before == 0:
            if after > len(original) * len(original.columns) * 0.05:
                return False, f"introduced too many nulls ({after})"
        else:
            if after > before * 1.15:
                return False, f"nulls increased {before}→{after}"

        return True, ""

    # ─────────────────────────────────────────────
    def _restore_columns(self, df_clean, df_raw, original_columns):
        for col in original_columns:
            if col not in df_clean.columns:
                log.warning("Restoring missing column '%s'", col)
                df_clean[col] = df_raw[col]
        return df_clean[original_columns]

    # ─────────────────────────────────────────────
    def _rule_based(self, df: pd.DataFrame, schema: dict):
        log.info("Rule-based cleaning started (%d cols)", len(df.columns))

        for col in df.columns:
            try:
                # ffill() is the pandas 2.x forward-fill method.
                # fillna(method="ffill") was deprecated in 2.1 and removed in 3.0.
                df[col] = df[col].ffill()
            except Exception as e:
                log.warning("Rule-based failed for '%s': %s", col, e)

        df = df.drop_duplicates().reset_index(drop=True)

        log.info("Rule-based cleaning complete. Output shape: %s", df.shape)

        return df