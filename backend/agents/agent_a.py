"""
AInsights — Agent A: The Data Engineer
=======================================
Autonomous ETL pipeline. Accepts raw files, profiles the schema,
uses the LLM to write cleaning code, executes it in a secure sandbox,
and outputs a standardised cleaned_data.csv.

ABSOLUTE MANDATE: Zero data destruction.
  - Original column names are ALWAYS preserved.
  - If LLM-generated code fails for ANY reason, the agent falls back
    to hardcoded rule-based Pandas cleaning automatically.
  - The output CSV is ALWAYS written, even if cleaning partially fails.
"""

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
from backend.utils.schema_profiler import profile

log = logging.getLogger("ainsights.agent_a")

OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MAX_LLM_RETRIES = 2   # attempts before falling back to rule-based cleaning


# ─────────────────────────────────────────────────────────────────────────────
# Prompt — heavily optimised for qwen2.5-coder:7b
# ─────────────────────────────────────────────────────────────────────────────
_CLEANING_PROMPT = """\
You are a senior Python data engineer. Your ONLY job is to write a Python \
function that cleans a pandas DataFrame.

## STRICT RULES — violating any rule causes the entire clean to be discarded:
1. The function signature MUST be exactly: def clean(df: pd.DataFrame) -> pd.DataFrame:
2. You MUST NOT rename, drop, or reorder any columns. Column names are sacred.
3. You MUST NOT drop rows unless the ENTIRE row is null.
4. Return the cleaned df at the end of the function.
5. Use ONLY: pandas (as pd), numpy (as np), re. No other imports.
6. Output ONLY the raw Python function. No markdown, no explanation, no ```python fences.

## DataFrame schema:
Shape: {shape}
Columns and types:
{column_summary}

## Cleaning operations to perform (ONLY these, in order):
1. Strip leading/trailing whitespace from all string columns.
2. For each column with inferred_semantic == "numeric": coerce to numeric \
   (pd.to_numeric with errors="coerce"). Fill remaining NaN with the column median.
3. For each column with inferred_semantic == "categorical": fill NaN with \
   the column mode (most frequent value). Strip whitespace and title-case the values.
4. For each column with inferred_semantic == "datetime": coerce with \
   pd.to_datetime(errors="coerce"). Leave NaT as NaT.
5. For columns with inferred_semantic == "id_or_text": fill NaN with empty string.
6. Remove exact duplicate rows (keep first).
7. Return df.

Write the function now:
"""


def _build_column_summary(schema: dict) -> str:
    lines = []
    for col, meta in schema["columns"].items():
        lines.append(
            f"  - '{col}': dtype={meta['dtype']}, "
            f"null_pct={meta['null_pct']}%, "
            f"semantic={meta['inferred_semantic']}, "
            f"sample={meta['sample_values'][:3]}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────
class DataEngineerAgent:
    """
    Agent A.
    Call agent.run(input_path, output_path) to execute the full ETL pipeline.
    """

    def __init__(self, llm_base_url: str = OLLAMA_BASE_URL) -> None:
        self._llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=llm_base_url,
            temperature=0.0,       # deterministic code generation
            num_predict=1024,
        )
        self._executor = SafeExecutor()

    # ── Public API ────────────────────────────────────────────────────────────
    def run(self, input_path: Path, output_path: Path) -> dict:
        """
        Full ETL pipeline.
        Returns a summary dict with cleaning stats.
        """
        t0 = time.perf_counter()
        log.info("Agent A starting: %s", input_path.name)

        # ① Load raw data
        df_raw = load_file(input_path)
        original_columns = df_raw.columns.tolist()
        original_shape   = df_raw.shape
        log.info("Raw data loaded: %s rows × %s cols", *original_shape)

        # ② Profile schema
        schema = profile(df_raw)

        # ③ Attempt LLM-generated cleaning
        df_clean, method = self._llm_clean(df_raw, schema)

        # ④ MANDATORY safety net — restore columns if any were lost/renamed
        df_clean = self._enforce_columns(df_clean, df_raw, original_columns)

        # ⑤ Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)

        elapsed  = round(time.perf_counter() - t0, 2)
        summary  = {
            "input_file":       input_path.name,
            "original_shape":   list(original_shape),
            "cleaned_shape":    list(df_clean.shape),
            "rows_removed":     original_shape[0] - df_clean.shape[0],
            "null_cells_before": int(df_raw.isna().sum().sum()),
            "null_cells_after":  int(df_clean.isna().sum().sum()),
            "cleaning_method":  method,
            "elapsed_seconds":  elapsed,
            "output_path":      str(output_path),
        }
        log.info("Agent A complete in %.2fs via '%s'. %s", elapsed, method, summary)
        return summary

    # ── LLM cleaning ─────────────────────────────────────────────────────────
    def _llm_clean(
        self, df: pd.DataFrame, schema: dict
    ) -> tuple[pd.DataFrame, str]:
        """
        Ask the LLM to generate a cleaning function, then execute it in a sandbox.
        Falls back to rule-based cleaning if the LLM fails MAX_LLM_RETRIES times.
        """
        col_summary = _build_column_summary(schema)
        prompt = _CLEANING_PROMPT.format(
            shape=schema["shape"],
            column_summary=col_summary,
        )

        for attempt in range(1, MAX_LLM_RETRIES + 1):
            log.info("LLM cleaning attempt %d/%d …", attempt, MAX_LLM_RETRIES)
            try:
                raw_code = self._llm.invoke(prompt)
                code     = self._extract_function(raw_code)

                if not code:
                    log.warning("Attempt %d: LLM returned no valid function.", attempt)
                    continue

                # Run in isolated sandbox
                df_result = self._executor.run_cleaning_function(df.copy(), code)

                # Sanity checks before accepting the result
                if self._is_safe_result(df_result, df, schema):
                    log.info("LLM cleaning accepted on attempt %d.", attempt)
                    return df_result, f"llm_attempt_{attempt}"
                else:
                    log.warning("Attempt %d: LLM result failed safety checks.", attempt)

            except Exception as exc:
                log.warning("Attempt %d: LLM/sandbox error: %s", attempt, exc)

        # ── All LLM attempts failed → fall back ──────────────────────────────
        log.warning("All LLM attempts exhausted. Falling back to rule-based cleaning.")
        return self._rule_based_clean(df, schema), "rule_based_fallback"

    def _extract_function(self, raw: str) -> str:
        """
        Strip markdown fences and extract only the clean() function definition.
        qwen2.5-coder sometimes wraps output in ```python blocks.
        """
        # Remove markdown code fences
        raw = re.sub(r"```(?:python)?", "", raw).strip()

        # Find the function definition
        match = re.search(r"(def clean\(df.*?)(?=\ndef |\Z)", raw, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _is_safe_result(
        self, df_result: pd.DataFrame, df_original: pd.DataFrame, schema: dict
    ) -> bool:
        """
        Validate that the LLM-cleaned DataFrame is safe to use.
        A result is REJECTED if:
          - It has fewer columns than the original
          - It lost more than 20% of rows (LLM shouldn't drop valid rows)
          - It has MORE null cells than the original (cleaning made things worse)
          - Any original column is missing
        """
        orig_cols = set(df_original.columns)
        result_cols = set(df_result.columns)

        if not orig_cols.issubset(result_cols):
            log.warning("Safety fail: missing columns %s", orig_cols - result_cols)
            return False

        row_loss_pct = 1.0 - (len(df_result) / max(len(df_original), 1))
        if row_loss_pct > 0.20:
            log.warning("Safety fail: %.1f%% of rows were dropped.", row_loss_pct * 100)
            return False

        nulls_before = df_original.isna().sum().sum()
        nulls_after  = df_result.isna().sum().sum()
        if nulls_after > nulls_before * 1.05:   # allow 5% tolerance
            log.warning("Safety fail: null count increased from %d to %d.", nulls_before, nulls_after)
            return False

        return True

    def _enforce_columns(
        self,
        df_clean: pd.DataFrame,
        df_raw: pd.DataFrame,
        original_columns: list[str],
    ) -> pd.DataFrame:
        """
        ABSOLUTE SAFETY NET.
        Restores any missing original columns from the raw DataFrame.
        This runs ALWAYS, even after a successful LLM clean.
        """
        missing = [c for c in original_columns if c not in df_clean.columns]
        if missing:
            log.warning("Restoring %d missing columns: %s", len(missing), missing)
            for col in missing:
                df_clean[col] = df_raw[col]
        # Preserve original column order
        return df_clean[original_columns]

    # ── Rule-based fallback cleaning ─────────────────────────────────────────
    def _rule_based_clean(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """
        Deterministic, hardcoded Pandas cleaning.
        Zero LLM involvement. Guaranteed to never lose columns.
        """
        log.info("Running rule-based fallback cleaning …")
        cols_meta = schema["columns"]

        for col in df.columns:
            semantic = cols_meta.get(col, {}).get("inferred_semantic", "text")

            if semantic == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())

            elif semantic == "categorical":
                df[col] = df[col].astype(str).str.strip().str.title()
                mode = df[col].mode()
                fill = mode.iloc[0] if not mode.empty else ""
                df[col] = df[col].replace("Nan", fill).replace("", fill)

            elif semantic == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")

            elif semantic == "id_or_text":
                df[col] = df[col].fillna("").astype(str).str.strip()

            else:   # generic string column
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace("nan", "")

        # Remove fully null rows only
        df = df.dropna(how="all")
        # Remove exact duplicates
        df = df.drop_duplicates(keep="first")
        return df.reset_index(drop=True)