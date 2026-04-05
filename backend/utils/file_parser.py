"""
Universal file parser — normalises CSV, Excel, JSON, XML, and PDF tables
into a single raw Pandas DataFrame for Agent A to consume.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pdfplumber

log = logging.getLogger("ainsights.file_parser")


def load_file(file_path: Path) -> pd.DataFrame:
    """
    Detect file type and return a raw (uncleaned) DataFrame.
    Raises ValueError on unsupported formats.
    """
    suffix = file_path.suffix.lower()
    log.info("Parsing file: %s (%s)", file_path.name, suffix)

    if suffix == ".csv":
        return _load_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return _load_excel(file_path)
    if suffix == ".json":
        return _load_json(file_path)
    if suffix == ".xml":
        return _load_xml(file_path)
    if suffix == ".pdf":
        return _load_pdf_tables(file_path)

    raise ValueError(f"Unsupported file format: '{suffix}'")


def _load_csv(path: Path) -> pd.DataFrame:
    import chardet
    raw = path.read_bytes()
    enc = chardet.detect(raw)["encoding"] or "utf-8"
    return pd.read_csv(path, encoding=enc, low_memory=False)


def _load_excel(path: Path) -> pd.DataFrame:
    # Read all sheets; concatenate if multiple
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    if len(sheets) == 1:
        return next(iter(sheets.values()))
    log.info("Excel has %d sheets — concatenating all.", len(sheets))
    return pd.concat(sheets.values(), ignore_index=True)


def _load_json(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        # Try common wrappers: {"data": [...]} or {"records": [...]}
        for key in ("data", "records", "rows", "items", "results"):
            if key in data and isinstance(data[key], list):
                return pd.DataFrame(data[key])
        # Fall back to treating the dict itself as a single-row table
        return pd.DataFrame([data])
    raise ValueError("JSON must be an array or an object with a list-valued key.")


def _load_xml(path: Path) -> pd.DataFrame:
    return pd.read_xml(path)


def _load_pdf_tables(path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            tables = page.extract_tables()
            for table in tables:
                if not table or not table[0]:
                    continue
                df = pd.DataFrame(table[1:], columns=table[0])
                df["_pdf_page"] = page_num
                frames.append(df)
    if not frames:
        raise ValueError("No extractable tables found in the PDF.")
    return pd.concat(frames, ignore_index=True)