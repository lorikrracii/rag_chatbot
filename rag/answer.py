# rag/answer.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


from rag.retriever import RetrievalResult, format_citation, retrieve

load_dotenv(override=True)



NOT_FOUND = "Not found in the provided documents."


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _expand_queries(question: str) -> List[str]:
    """
    Expand short/vague questions into a few semantically related queries to increase recall.
    Also expands common acronyms (CCB/CCyB).
    """
    q = (question or "").strip()
    if not q:
        return []

    ql = q.lower()
    extras: List[str] = []

    if "ccyb" in ql:
        extras += ["countercyclical capital buffer", "countercyclical buffer ccyb"]
    if "ccb" in ql:
        extras += ["capital conservation buffer", "conservation buffer ccb"]

    # If user asks the phrase explicitly, also add acronym variants
    if "countercyclical capital buffer" in ql:
        extras += ["ccyb", "countercyclical buffer"]
    if "capital conservation buffer" in ql:
        extras += ["ccb", "capital conservation buffer requirement"]

    if len(q.split()) <= 4:
        return [
            q,
            f"{q} definition and requirements",
            f"How is {q} described in the document?",
            f"{q} related to capital requirements and risk-weighted assets",
            *extras,
        ]

    return [
        q,
        f"{q} definition and requirements",
        f"How is {q} described in the document?",
        *extras,
    ]


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]


def _keyword_score(query: str, chunk: str) -> float:
    q_tokens = set(_tokenize(query))
    c_tokens = set(_tokenize(chunk))
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens)
    return float(overlap) / float(len(q_tokens))


def _build_context(results: List[RetrievalResult], max_chars: int = 12000, max_chunks: int = 6) -> str:
    parts: List[str] = []
    total = 0

    for i, r in enumerate(results[:max_chunks], start=1):
        doc: Document = r.doc
        citation = format_citation(doc)
        chunk_text = (doc.page_content or "").strip()

        block = f"[Source {i}] {citation}\n{chunk_text}\n"
        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n---\n".join(parts).strip()


def _unique_citations(results: List[RetrievalResult]) -> List[str]:
    seen = set()
    citations: List[str] = []
    for r in results:
        c = format_citation(r.doc)
        if c not in seen:
            seen.add(c)
            citations.append(c)
    return citations


def answer_question(
    question: str,
    *,
    k: Optional[int] = None,
    llm_model: Optional[str] = None,
    persist_directory: str = "chromadb",
    collection_name: str = "rag_docs",
    embed_model: str = "nomic-embed-text",
    source_filename: Optional[str] = None,
    strict: Optional[bool] = None,
    user_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end RAG:
    - multi-query retrieval + dedupe + rerank
    - strict evidence gate (if strict=True)
    - Gemini generation grounded in context
    - citations appended by app
    """
    question = (question or "").strip()
    raw_user = (user_query or question).strip()

    if not question:
        return {"answer": "Please enter a question.", "citations": [], "retrieved": []}

    top_k = k if isinstance(k, int) and k > 0 else _get_env_int("RAG_TOP_K", 12)

    # Strict mode: param wins. Env is only a fallback for CLI usage.
    env_strict = os.getenv("RAG_USE_SCORE_GATE", "").strip().lower() in {"1", "true", "yes"}
    strict_mode = bool(strict) if strict is not None else env_strict

    def _too_short(q: str) -> bool:
        words = [w for w in (q or "").split() if w]
        return len(words) <= 1

    if strict_mode and _too_short(raw_user):
        return {
            "answer": (
                "Your question is too broad in Strict mode.\n\n"
                "Which one do you mean?\n"
                "- Capital Conservation Buffer (CCB)\n"
                "- Countercyclical Capital Buffer (CCyB)\n"
                "- Other Basel III buffers"
            ),
            "citations": [],
            "retrieved": [],
        }

    # Evidence thresholds (Chroma returns distance: lower is better)
    max_distance = _get_env_float("RAG_MAX_SCORE", 0.75)
    min_kw_overlap = _get_env_float("RAG_MIN_KW_OVERLAP", 0.12)

    # ---- Multi-query retrieval
    queries = _expand_queries(question) or [question]
    all_results: List[RetrievalResult] = []
    for q in queries:
        if not q or not q.strip():
            continue
        all_results.extend(
            retrieve(
                q,
                k=top_k,
                source_filename=source_filename,
                persist_directory=persist_directory,
                collection_name=collection_name,
                embed_model=embed_model,
            )
        )

    # ---- Deduplicate by (source, page, chunk_id)
    seen = set()
    deduped: List[RetrievalResult] = []
    for r in all_results:
        md = r.doc.metadata or {}
        key = (md.get("source"), md.get("page"), md.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    if not deduped:
        return {"answer": NOT_FOUND, "citations": [], "retrieved": []}

    # ---- Keyword overlap rerank (fast + helps acronym/definition questions)
    scored = []
    for idx, r in enumerate(deduped):
        s = _keyword_score(question, r.doc.page_content or "")
        scored.append((s, idx, r))
    scored.sort(key=lambda x: (-x[0], x[1]))
    results = [r for _, _, r in scored]

    # ---- Strict evidence gate
    best_distance = min((r.score for r in results), default=None)
    best_kw = max((_keyword_score(question, r.doc.page_content or "") for r in results[:8]), default=0.0)

    if strict_mode:
        if best_distance is None or best_distance > max_distance or best_kw < min_kw_overlap:
            return {
                "answer": NOT_FOUND,
                "citations": [],
                "retrieved": [{"citation": format_citation(r.doc), "score": r.score} for r in results],
            }

    context = _build_context(
        results,
        max_chars=_get_env_int("RAG_MAX_CONTEXT_CHARS", 12000),
        max_chunks=_get_env_int("RAG_MAX_CONTEXT_CHUNKS", 6),
    )

    llm_name = llm_model or os.getenv("RAG_LLM_MODEL", "qwen2.5:7b-instruct")

    llm = ChatOllama(
        model=llm_name,
        temperature=_get_env_float("RAG_TEMPERATURE", 0.2),
    )


    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a Retrieval-Augmented Generation (RAG) assistant.\n"
                    "You MUST answer using ONLY the provided context.\n\n"
                    "STRICT RULES (DO NOT BREAK):\n"
                    f"1) Only answer from context.\n"
                    f"2) If not in context, say exactly: \"{NOT_FOUND}\"\n"
                    "3) Do not use outside knowledge.\n"
                    "4) Ask ONE clarifying question if the user is vague.\n"
                    "5) Do NOT write citations.\n"
                ),
            ),
            ("human", "Question:\n{question}\n\nContext:\n{context}\n\nWrite the answer now."),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    response_text = (getattr(response, "content", "") or "").strip()

    # HARD CLAMP: never allow mixed outputs
    if response_text == NOT_FOUND or (NOT_FOUND in response_text):
        return {
            "answer": NOT_FOUND,
            "citations": [],
            "retrieved": [{"citation": format_citation(r.doc), "score": r.score} for r in results],
        }

    citations = _unique_citations(results)[:4]

    final_answer = response_text + "\n\nSources:\n"
    for i, c in enumerate(citations, start=1):
        final_answer += f"{i}. {c}\n"

    return {
        "answer": final_answer.strip(),
        "citations": citations,
        "retrieved": [
            {
                "citation": format_citation(r.doc),
                "score": r.score,
                "preview": (r.doc.page_content or "")[:200].replace("\n", " "),
            }
            for r in results
        ],
        "llm_model": llm_name,
        "top_k": top_k,
        "expanded_queries": queries,
        "strict": strict_mode,
    }
