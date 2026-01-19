# rag/hf_models.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def _device() -> int:
    # HF Spaces free tier is usually CPU => device = -1
    # If you ever get GPU, it will work automatically
    return 0 if torch.cuda.is_available() else -1


@lru_cache(maxsize=1)
def get_embedder(model_name: str | None = None) -> SentenceTransformer:
    name = model_name or os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    return SentenceTransformer(name)


def embed_texts(texts: List[str], model_name: str | None = None) -> List[List[float]]:
    emb = get_embedder(model_name)
    vecs = emb.encode(texts, normalize_embeddings=True)
    return vecs.tolist()


@lru_cache(maxsize=1)
def get_llm_pipe(model_name: str | None = None):
    name = model_name or os.getenv("HF_LLM_MODEL", "google/flan-t5-large")
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name)
    return pipeline(
        "text2text-generation",
        model=mdl,
        tokenizer=tok,
        device=_device(),
        max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "256")),
        do_sample=False,
    )
