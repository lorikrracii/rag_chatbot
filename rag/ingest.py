from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)


@dataclass
class IngestConfig:
    data_dir: str = os.getenv("DATA_DIR", "data/raw")
    chroma_dir: str = os.getenv("CHROMA_DIR", "chromadb")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "rag_docs")

    embed_model: str = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1100"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "160"))

    reset_chroma: bool = os.getenv("RESET_CHROMA", "0").strip().lower() in {"1", "true", "yes"}


def _load_pdfs(data_dir: str) -> List:
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {base.resolve()}")

    pdf_paths = sorted(base.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found under: {base.resolve()}")

    docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pdf_docs = loader.load()
        for d in pdf_docs:
            d.metadata["source"] = str(pdf_path).replace("\\", "/")
        docs.extend(pdf_docs)

    return docs


def _chunk(docs: List, chunk_size: int, chunk_overlap: int) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def ingest(config: IngestConfig) -> dict:
    if config.reset_chroma and Path(config.chroma_dir).exists():
        shutil.rmtree(config.chroma_dir, ignore_errors=True)

    docs = _load_pdfs(config.data_dir)
    print(f"Loaded {len(docs)} pages")

    chunks = _chunk(docs, config.chunk_size, config.chunk_overlap)
    print(f"Created {len(chunks)} chunks")
    print("Starting embedding + indexing…")

    embeddings = OpenAIEmbeddings(model=config.embed_model)

    vectorstore = Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=config.chroma_dir,
    )
    vectorstore.add_documents(chunks)

    return {
        "pdf_pages_loaded": len(docs),
        "chunks_indexed": len(chunks),
        "chroma_dir": str(Path(config.chroma_dir).resolve()),
        "collection": config.collection_name,
        "embed_backend": "openai",
        "embed_model": config.embed_model,
    }


if __name__ == "__main__":
    cfg = IngestConfig()
    result = ingest(cfg)
    print("✅ Ingestion complete")
    for k, v in result.items():
        print(f"- {k}: {v}")
