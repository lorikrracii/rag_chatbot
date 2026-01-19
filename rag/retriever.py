from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved chunk + its similarity score (Chroma: lower distance is better)."""
    doc: Document
    score: float


def _get_env_int(name: str, default: int) -> int:
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
    *,
    embed_model: Optional[str] = None,
) -> Chroma:
    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(
            f"Chroma persist directory not found: '{persist_directory}'. Did you run ingestion?"
        )

    model_name = (embed_model or os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")).strip()
    embeddings = OpenAIEmbeddings(model=model_name)

    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )


def _match_filename(source_value: str, wanted_filename: str) -> bool:
    if not source_value or not wanted_filename:
        return False
    src = str(source_value).replace("\\", "/").lower().strip()
    wanted = wanted_filename.replace("\\", "/").lower().strip()
    if os.path.basename(src) == wanted:
        return True
    return src.endswith("/" + wanted)


def retrieve(
    query: str,
    k: Optional[int] = None,
    *,
    source_filename: Optional[str] = None,
    persist_directory: str = "chromadb",
    collection_name: str = "rag_docs",
    embed_model: Optional[str] = None,
) -> List[RetrievalResult]:
    if not query or not query.strip():
        return []

    top_k = k if isinstance(k, int) and k > 0 else _get_env_int("RAG_TOP_K", 12)

    vs = get_vectorstore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embed_model=embed_model,
    )

    want_filter = bool(source_filename and source_filename.strip() and source_filename.strip().lower() != "all")
    prefetch_k = max(top_k * 8, 50) if want_filter else max(top_k * 4, top_k)

    docs_and_scores: List[Tuple[Document, float]] = vs.similarity_search_with_score(query=query, k=prefetch_k) or []

    if want_filter:
        wanted = source_filename.strip()
        docs_and_scores = [
            (doc, score)
            for (doc, score) in docs_and_scores
            if doc is not None and _match_filename((doc.metadata or {}).get("source", ""), wanted)
        ]

    seen = set()
    deduped: List[Tuple[Document, float]] = []
    for doc, score in docs_and_scores:
        md = doc.metadata or {}
        key = (md.get("source"), md.get("page"), md.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((doc, score))
        if len(deduped) >= top_k:
            break

    return [RetrievalResult(doc=d, score=float(s)) for d, s in deduped]


def format_citation(doc: Document) -> str:
    md = doc.metadata or {}
    source = md.get("source", "unknown_source")
    page = md.get("page")
    filename = os.path.basename(str(source).replace("\\", "/"))

    if page is not None:
        return f"{filename} — p.{int(page) + 1}"
    return f"{filename}"
