from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def row_to_text(row: pd.Series) -> str:
    parts = []
    for col, val in row.items():
        if pd.isna(val):
            continue
        parts.append(f"{col}: {val}")
    return ", ".join(parts)


def dataframe_to_chunks(df: pd.DataFrame, rows_per_chunk: int = 25, max_chunks: int = 2000) -> list[str]:
    if df.empty:
        return []

    chunks: list[str] = []
    n = len(df)
    step = max(1, rows_per_chunk)

    for start in range(0, n, step):
        end = min(start + step, n)
        block = df.iloc[start:end]
        text = "\n".join(row_to_text(block.iloc[i]) for i in range(len(block)))
        if text.strip():
            chunks.append(text)
        if len(chunks) >= max_chunks:
            break

    return chunks


@dataclass
class TfidfMemory:
    vectorizer: TfidfVectorizer | None = None
    matrix: any = None
    chunks: list[str] = field(default_factory=list)

    def build(self, df: pd.DataFrame, rows_per_chunk: int = 10) -> None:
        self.chunks = dataframe_to_chunks(df, rows_per_chunk=rows_per_chunk)
        if not self.chunks:
            self.vectorizer = None
            self.matrix = None
            return

        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        if not self.vectorizer or self.matrix is None or not self.chunks:
            return []

        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        if scores.size == 0:
            return []

        idx = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in idx if scores[i] > 0]