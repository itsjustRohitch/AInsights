"""
Schema profiler — generates a structured summary of a DataFrame's
columns, types, and statistics. Used by Agent A to build the LLM prompt.
"""

from __future__ import annotations

import pandas as pd


def profile(df: pd.DataFrame) -> dict:
    """
    Returns a compact schema summary:
    {
        "shape": [rows, cols],
        "columns": {
            "col_name": {
                "dtype": "float64",
                "null_pct": 12.5,
                "sample_values": [1.0, 2.5, null, "text"],
                "inferred_semantic": "numeric|categorical|datetime|text|id"
            },
            ...
        }
    }
    """
    rows, cols = df.shape
    schema: dict = {"shape": [rows, cols], "columns": {}}

    for col in df.columns:
        series    = df[col]
        null_pct  = round(series.isna().mean() * 100, 2)
        sample    = series.dropna().head(5).tolist()

        schema["columns"][col] = {
            "dtype":              str(series.dtype),
            "null_pct":           null_pct,
            "unique_count":       int(series.nunique()),
            "sample_values":      sample,
            "inferred_semantic":  _infer_semantic(series),
        }

    return schema


def _infer_semantic(series: pd.Series) -> str:
    """Heuristic semantic type inference."""
    dtype_str = str(series.dtype)

    if "int" in dtype_str or "float" in dtype_str:
        return "numeric"

    if "datetime" in dtype_str:
        return "datetime"

    # Try to detect datetime strings
    sample = series.dropna().head(20).astype(str)
    if sample.str.match(
        r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}"
    ).mean() > 0.5:
        return "datetime"

    # High-cardinality strings are likely IDs or free text
    if series.nunique() / max(len(series.dropna()), 1) > 0.9:
        return "id_or_text"

    if series.nunique() <= 30:
        return "categorical"

    return "text"