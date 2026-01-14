from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


@dataclass
class IngestConfig:
    data_dir: str = os.getenv("DATA_DIR", "data/raw")
    chroma_dir: str = os.getenv("CHROMA_DIR", "chromadb")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "rag_docs")
    embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))




def _load_pdfs(data_dir: str) -> List:
    """Load all PDFs under data_dir into LangChain Documents."""
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {base.resolve()}")

    pdf_paths = sorted(base.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found under: {base.resolve()}")

    docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pdf_docs = loader.load()  # each doc usually has metadata incl. 'page'
        # Normalize source metadata for citations
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

    # Add stable chunk ids for citations/debugging
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i

    return chunks


def ingest(config: IngestConfig) -> dict:
    # 1) Load
    docs = _load_pdfs(config.data_dir)
    print(f"Loaded {len(docs)} pages")

    # 2) Chunk
    chunks = _chunk(docs, config.chunk_size, config.chunk_overlap)
    print(f"Created {len(chunks)} chunks")

    # 3) Embed + Index (Chroma persistent)
    print("Starting embedding + indexing…")

    embeddings = OllamaEmbeddings(model=config.embed_model)

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
        "embed_model": config.embed_model,
    }


if __name__ == "__main__":
    cfg = IngestConfig()
    result = ingest(cfg)
    print("✅ Ingestion complete")
    for k, v in result.items():
        print(f"- {k}: {v}")
