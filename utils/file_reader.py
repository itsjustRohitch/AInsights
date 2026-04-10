from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


CSV_SEPARATORS = [",", ";", "\t", "|"]
CSV_ENCODINGS = ["utf-8-sig", "utf-8", "latin-1"]


def read_csv_robust(path: str | Path) -> pd.DataFrame:
    best_df: Optional[pd.DataFrame] = None
    best_score = (-1, -1)  # (rows, cols)

    for encoding in CSV_ENCODINGS:
        for sep in CSV_SEPARATORS:
            try:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    encoding=encoding,
                    engine="python",
                    on_bad_lines="skip",
                )
                rows, cols = df.shape
                if rows > 0 and cols > 1:
                    score = (rows, cols)
                    if score > best_score:
                        best_df = df
                        best_score = score
            except Exception:
                continue

    if best_df is not None:
        return best_df

    raise ValueError("Could not parse CSV with supported separators/encodings.")


def read_excel_file(path: str | Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheet_name = xls.sheet_names[0]
    return pd.read_excel(path, sheet_name=sheet_name)


def read_json_file(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_json(path, lines=True)
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        raise ValueError("Unsupported JSON structure.")


def read_txt_file(path: str | Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return pd.DataFrame({"text": lines})


def read_pdf_file(path: str | Path) -> pd.DataFrame:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise ImportError("pypdf is required for PDF ingestion.") from e

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i + 1, "text": text})

    if not pages:
        raise ValueError("No readable text found in PDF.")

    return pd.DataFrame(pages)


def read_any_file(path: str | Path) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        return read_csv_robust(path)
    if suffix in {".xls", ".xlsx"}:
        return read_excel_file(path)
    if suffix == ".json":
        return read_json_file(path)
    if suffix == ".txt":
        return read_txt_file(path)
    if suffix == ".pdf":
        return read_pdf_file(path)

    raise ValueError(f"Unsupported file type: {suffix}")