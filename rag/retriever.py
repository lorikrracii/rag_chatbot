from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved chunk + its similarity score (lower/higher depends on backend; treat as 'for ranking')."""
    doc: Document
    score: float


def _get_env_int(name: str, default: int) -> int:
    """Read an int env var safely."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_vectorstore(
    persist_directory: str = "chromadb",
    collection_name: str = "rag_docs",
    embed_model: str = "nomic-embed-text",
) -> Chroma:
    """
    Loads the existing Chroma DB from disk and returns a ready-to-query vectorstore.
    IMPORTANT: embed_model must match ingestion.
    """
    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(
            f"Chroma persist directory not found: '{persist_directory}'. "
            f"Did you run ingestion and create it?"
        )

    embeddings = OllamaEmbeddings(model=embed_model)

    # This does NOT re-ingest. It just opens the persisted collection.
    vs = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    return vs


def retrieve(
    query: str,
    k: Optional[int] = None,
    *,
    persist_directory: str = "chromadb",
    collection_name: str = "rag_docs",
    embed_model: str = "nomic-embed-text",
) -> List[RetrievalResult]:
    """
    Runs Top-K similarity search and returns documents + scores.
    """
    if not query or not query.strip():
        return []

    top_k = k if isinstance(k, int) and k > 0 else _get_env_int("RAG_TOP_K", 4)

    vs = get_vectorstore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embed_model=embed_model,
    )

    # Returns: List[Tuple[Document, float]]
    docs_and_scores: List[Tuple[Document, float]] = vs.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    results: List[RetrievalResult] = [
        RetrievalResult(doc=doc, score=float(score)) for doc, score in docs_and_scores
    ]
    return results


def format_citation(doc: Document) -> str:
    """
    Builds a simple citation string from metadata you stored during ingestion.
    Expected metadata keys (best-case): source, page, chunk_id
    """
    md = doc.metadata or {}
    source = md.get("source", "unknown_source")
    page = md.get("page")
    chunk_id = md.get("chunk_id")

    # Make the filename nicer if it's a path
    filename = os.path.basename(str(source))

    parts = [filename]
    if page is not None:
        parts.append(f"p.{page}")
    if chunk_id is not None:
        parts.append(f"chunk:{chunk_id}")

    return " — ".join(parts)
