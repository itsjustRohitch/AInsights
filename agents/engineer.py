from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.llm_client import OllamaClient, extract_code_block
from utils.file_reader import read_any_file
from utils.sandbox import SandboxResult, run_cleaning_code


def basic_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    # Remove fully empty rows
    cleaned = cleaned.dropna(how="all")

    # Normalize obvious whitespace in string cells, not column names
    for col in cleaned.columns:
        if cleaned[col].dtype == "object":
            cleaned[col] = cleaned[col].astype(str).replace("nan", pd.NA)
            cleaned[col] = cleaned[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop duplicate rows
    cleaned = cleaned.drop_duplicates()

    return cleaned


def build_profile(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isna().sum().to_dict(),
        "missing_percent": (df.isna().mean() * 100).round(2).to_dict(),
        "sample_rows": df.head(5).to_dict(orient="records"),
    }


def build_cleaning_prompt(profile: dict[str, Any]) -> str:
    return f"""
You are generating Python cleaning code for a pandas DataFrame named df.

Rules:
- Output ONLY Python code.
- No markdown, no explanations.
- Do not import anything.
- Use only pd, np, re, datetime.
- Do not rename columns.
- Do not reorder columns.
- Keep the result in df or cleaned_df.
- Prefer safe, simple transformations.
- Avoid anything expensive.

Profile:
{profile}

Task:
Clean the dataframe conservatively.
Possible actions:
- drop fully empty rows
- fix obvious whitespace issues in cells
- standardize missing values
- remove duplicate rows
- parse obvious datetime-like columns only if safe
- remove impossible values only if strongly justified by the profile
""".strip()


@dataclass
class EngineerResult:
    dataframe: pd.DataFrame
    report: dict[str, Any]
    used_llm: bool
    sandbox_error: str | None = None


class DataEngineer:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def ingest(self, file_path: str) -> pd.DataFrame:
        return read_any_file(file_path)

    def clean(self, df: pd.DataFrame) -> EngineerResult:
        profile = build_profile(df)
        prompt = build_cleaning_prompt(profile)

        used_llm = True
        sandbox_error = None

        try:
            raw = self.llm.generate(
                prompt=prompt,
                system="You are a careful data-cleaning code generator.",
                temperature=0.0,
                num_predict=500,
            )
            code = extract_code_block(raw)
            sandbox: SandboxResult = run_cleaning_code(code, df, timeout_seconds=5)

            if sandbox.ok and sandbox.dataframe is not None:
                cleaned = sandbox.dataframe
            else:
                used_llm = False
                sandbox_error = sandbox.error
                cleaned = basic_cleanup(df)

        except Exception as e:
            used_llm = False
            sandbox_error = str(e)
            cleaned = basic_cleanup(df)

        report = {
            "original_shape": df.shape,
            "cleaned_shape": cleaned.shape,
            "rows_removed": int(df.shape[0] - cleaned.shape[0]),
            "columns": list(cleaned.columns),
            "profile": profile,
        }

        return EngineerResult(
            dataframe=cleaned,
            report=report,
            used_llm=used_llm,
            sandbox_error=sandbox_error,
        )