"""
AInsights — RAG Engine
Owns the ChromaDB vector store and the HuggingFace embedding model.
Handles two ingestion pipelines:
  1. Unstructured documents (PDF, TXT, Markdown)  → chunked text
  2. Structured tabular data (cleaned_data.csv)   → semantic row sentences
"""

from __future__ import annotations

import logging
import os
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber
from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

log = logging.getLogger("ainsights.rag")

# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DIR       = Path(os.getenv("CHROMA_PERSIST_DIR", "backend/data/vectorstore"))
COLLECTION_DOCS  = "ainsights_documents"   # unstructured docs
COLLECTION_ROWS  = "ainsights_tabular"     # structured CSV rows

CHUNK_SIZE       = 512    # characters per text chunk
CHUNK_OVERLAP    = 64     # overlap between adjacent chunks
MAX_ROWS_EMBED   = 5_000  # embed at most this many CSV rows (perf guard)
RETRIEVAL_TOP_K  = 6      # how many chunks to return per query


class RAGEngine:
    """
    Dual-collection ChromaDB engine.
    - initialize()          : loads embedding model + connects to ChromaDB
    - ingest_document()     : chunks and embeds PDF/TXT files
    - ingest_tabular_data() : converts CSV rows → semantic sentences and embeds
    - query()               : retrieves relevant chunks from both collections
    """

    def __init__(self) -> None:
        self._embedder:   SentenceTransformer | None = None
        self._chroma:     PersistentClient | None    = None
        self._col_docs:   Any = None
        self._col_rows:   Any = None

    # ── Initialisation ────────────────────────────────────────────────────────
    def initialize(self) -> None:
        """Load the embedding model and connect to ChromaDB. Call once at startup."""
        log.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        log.info("Connecting to ChromaDB at %s", CHROMA_DIR)
        self._chroma = PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        self._col_docs = self._chroma.get_or_create_collection(
            name=COLLECTION_DOCS,
            metadata={"hnsw:space": "cosine"},
        )
        self._col_rows = self._chroma.get_or_create_collection(
            name=COLLECTION_ROWS,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("RAG engine initialized. docs=%d rows=%d",
                 self._col_docs.count(), self._col_rows.count())

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-encode a list of strings into embedding vectors."""
        assert self._embedder is not None, "RAGEngine.initialize() not called."
        return self._embedder.encode(texts, batch_size=64, show_progress_bar=False).tolist()

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline 1 — Unstructured document ingestion
    # ─────────────────────────────────────────────────────────────────────────
    def ingest_document(self, file_path: Path) -> int:
        """
        Parse a document and ingest it into the docs collection.
        Supports: .pdf, .txt, .md
        Returns the number of chunks created.
        """
        log.info("Ingesting document: %s", file_path.name)
        raw_text = self._extract_text(file_path)

        if not raw_text.strip():
            log.warning("Document '%s' yielded no extractable text.", file_path.name)
            return 0

        chunks   = self._chunk_text(raw_text)
        ids      = [f"{file_path.stem}__chunk_{i}" for i in range(len(chunks))]
        metadata = [{"source": file_path.name, "chunk": i} for i in range(len(chunks))]

        # Delete any prior versions of this document
        self._col_docs.delete(where={"source": file_path.name})

        embeddings = self._embed(chunks)
        self._col_docs.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata,
        )
        log.info("Indexed %d chunks from '%s'.", len(chunks), file_path.name)
        return len(chunks)

    def _extract_text(self, file_path: Path) -> str:
        """Extract raw text from PDF, TXT, or Markdown."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            pages: list[str] = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
            return "\n\n".join(pages)

        if suffix in {".txt", ".md"}:
            return file_path.read_text(encoding="utf-8", errors="replace")

        raise ValueError(f"Unsupported document format: {suffix}")

    def _chunk_text(self, text: str) -> list[str]:
        """
        Sliding-window chunker.
        Splits by paragraph first (double newline), then by CHUNK_SIZE characters,
        with CHUNK_OVERLAP overlap to preserve context across chunk boundaries.
        """
        # Prefer paragraph boundaries
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []

        for para in paragraphs:
            if len(para) <= CHUNK_SIZE:
                chunks.append(para)
            else:
                # Hard-split long paragraphs with overlap
                start = 0
                while start < len(para):
                    end = start + CHUNK_SIZE
                    chunks.append(para[start:end])
                    start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline 2 — Tabular RAG (structured CSV ingestion)
    # ─────────────────────────────────────────────────────────────────────────
    def ingest_tabular_data(self, csv_path: Path) -> int:
        """
        Reads cleaned_data.csv and converts each row into a semantic text sentence,
        then embeds and stores it in the tabular collection.

        Design decision: Converting rows to human-readable sentences (rather than
        raw JSON) dramatically improves retrieval quality because the embedding
        model was trained on natural language, not JSON syntax.

        Example output sentence for a sales row:
        "Row 42: Region is North, Product is Widget A, Sales is 1250.00,
         Profit is 340.00, Quantity is 5."
        """
        log.info("Ingesting tabular data from: %s", csv_path.name)

        df = pd.read_csv(csv_path)
        total_rows = len(df)

        if total_rows == 0:
            log.warning("CSV is empty — nothing to ingest.")
            return 0

        # Cap to avoid memory blowout on huge files
        if total_rows > MAX_ROWS_EMBED:
            log.warning(
                "CSV has %d rows; only the first %d will be embedded.",
                total_rows, MAX_ROWS_EMBED,
            )
            df = df.head(MAX_ROWS_EMBED)

        sentences = self._rows_to_sentences(df)
        ids       = [f"row_{i}" for i in df.index]
        metadata  = [{"row_index": int(i), "source": csv_path.name} for i in df.index]

        # Replace the entire tabular collection on each re-ingest
        # (Agent A always regenerates the full cleaned_data.csv)
        self._col_rows.delete(where={"source": csv_path.name})

        embeddings = self._embed(sentences)
        self._col_rows.add(
            ids=ids,
            documents=sentences,
            embeddings=embeddings,
            metadatas=metadata,
        )
        log.info("Indexed %d rows from '%s'.", len(sentences), csv_path.name)
        return len(sentences)

    def _rows_to_sentences(self, df: pd.DataFrame) -> list[str]:
        """
        Convert each DataFrame row into a natural-language sentence
        optimised for semantic similarity search.

        Format:
          "Row {idx}: {col1} is {val1}, {col2} is {val2}, ..."

        Numeric values are rounded to 2dp to avoid floating-point noise.
        NaN values are represented as 'N/A'.
        """
        sentences: list[str] = []
        columns = df.columns.tolist()

        for idx, row in df.iterrows():
            parts: list[str] = []
            for col in columns:
                val = row[col]
                if pd.isna(val):
                    parts.append(f"{col} is N/A")
                elif isinstance(val, float):
                    parts.append(f"{col} is {val:.2f}")
                else:
                    parts.append(f"{col} is {val}")

            sentence = f"Row {idx}: " + ", ".join(parts) + "."
            # Truncate runaway sentences (very wide tables)
            sentences.append(textwrap.shorten(sentence, width=800, placeholder=" …"))

        return sentences

    # ─────────────────────────────────────────────────────────────────────────
    # Query — retrieval for Agent C
    # ─────────────────────────────────────────────────────────────────────────
    def query(
        self,
        question: str,
        top_k: int = RETRIEVAL_TOP_K,
        include_tabular: bool = True,
        include_docs: bool = True,
    ) -> dict[str, list[str]]:
        """
        Query both collections and return retrieved chunks.
        Agent C calls this to get vector context before building its prompt.

        Returns:
          {
            "doc_chunks":   ["chunk text …", …],
            "table_chunks": ["Row 42: Region is …", …],
          }
        """
        q_embedding = self._embed([question])[0]

        doc_chunks: list[str]   = []
        table_chunks: list[str] = []

        if include_docs and self._col_docs.count() > 0:
            results = self._col_docs.query(
                query_embeddings=[q_embedding],
                n_results=min(top_k, self._col_docs.count()),
            )
            doc_chunks = results["documents"][0] if results["documents"] else []

        if include_tabular and self._col_rows.count() > 0:
            results = self._col_rows.query(
                query_embeddings=[q_embedding],
                n_results=min(top_k, self._col_rows.count()),
            )
            table_chunks = results["documents"][0] if results["documents"] else []

        log.info(
            "RAG query → %d doc chunks + %d table chunks",
            len(doc_chunks), len(table_chunks),
        )
        return {"doc_chunks": doc_chunks, "table_chunks": table_chunks}

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        """Return collection sizes — useful for the health endpoint."""
        return {
            "doc_chunks":   self._col_docs.count() if self._col_docs else 0,
            "table_chunks": self._col_rows.count() if self._col_rows else 0,
        }

    def clear_all(self) -> None:
        """Wipe both collections. Useful for testing."""
        if self._chroma:
            self._chroma.delete_collection(COLLECTION_DOCS)
            self._chroma.delete_collection(COLLECTION_ROWS)
            self.initialize()