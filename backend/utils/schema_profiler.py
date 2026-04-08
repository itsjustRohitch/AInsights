"""
AInsights — Schema Profiler
Generates the schema dict and compact string used in all three agent prompts.
Includes a df.head(3) snippet formatted to fit inside an LLM prompt.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("ainsights.schema_profiler")


def profile(df: pd.DataFrame) -> dict:
    """
    Returns:
      {
        "shape":   [rows, cols],
        "columns": { col: {dtype, null_pct, unique_count, sample_values,
                           inferred_semantic} },
        "head":    str   # df.head(3) as a compact string
      }
    """
    rows, cols = df.shape
    schema: dict = {"shape": [rows, cols], "columns": {}, "head": ""}

    for col in df.columns:
        series = df[col]
        schema["columns"][col] = {
            "dtype":             str(series.dtype),
            "null_pct":          round(series.isna().mean() * 100, 1),
            "unique_count":      int(series.nunique()),
            "sample_values":     series.dropna().head(3).tolist(),
            "inferred_semantic": _infer_semantic(series),
        }

    schema["head"] = _head_str(df)
    return schema


def compact_schema_str(schema: dict) -> str:
    """
    One line per column:
      col_name (dtype, semantic, X% null): [v1, v2, v3]

    Used directly in LLM prompts — kept deliberately short.
    """
    lines = []
    for col, meta in schema["columns"].items():
        lines.append(
            f"  {col} ({meta['dtype']}, {meta['inferred_semantic']}, "
            f"{meta['null_pct']}% null): {meta['sample_values']}"
        )
    return "\n".join(lines)


def _head_str(df: pd.DataFrame) -> str:
    """
    df.head(3) formatted as a compact string for LLM prompt injection.
    Caps columns and column width to avoid token bloat.
    """
    try:
        return df.head(3).to_string(
            max_cols=20,
            max_colwidth=28,
            show_dimensions=False,
        )
    except Exception as exc:
        log.debug("head_str failed: %s", exc)
        return "(sample unavailable)"


def _infer_semantic(series: pd.Series) -> str:
    """Heuristic semantic type — used by rule-based fallback and prompt context."""
    dtype_str = str(series.dtype)

    if "int"      in dtype_str: return "numeric"
    if "float"    in dtype_str: return "numeric"
    if "datetime" in dtype_str: return "datetime"
    if "bool"     in dtype_str: return "categorical"

    # Try to detect datetime strings
    sample = series.dropna().head(30).astype(str)
    date_pattern = r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
    if sample.str.match(date_pattern).mean() > 0.5:
        return "datetime"

    n_valid = max(len(series.dropna()), 1)
    if series.nunique() / n_valid > 0.85:
        return "id_or_text"
    if series.nunique() <= 25:
        return "categorical"

    return "text"