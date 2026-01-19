from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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


def _get_env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def rewrite_followup_question(current_q: str, previous_q: str | None) -> str:
    """
    Rewrites vague follow-up questions using the previous user question.
    This enables short-term conversational memory without hallucination.
    """
    if not previous_q:
        return current_q

    q = current_q.lower()
    followup_triggers = [
        "it",
        "that",
        "the other",
        "one",
        "difference",
        "different",
        "differ",
        "compare",
        "vs",
    ]

    if any(t in q for t in followup_triggers):
        return (
            f"{current_q}\n\n"
            f"Context: The previous question was: {previous_q}"
        )

    return current_q

# ---------------------- query expansion ----------------------
def _expand_queries(question: str) -> List[str]:
    q = (question or "").strip()
    if not q:
        return []
    ql = q.lower()
    extras: List[str] = []

    if "ccyb" in ql:
        extras += ["countercyclical capital buffer", "countercyclical buffer ccyb"]
    if "ccb" in ql:
        extras += ["capital conservation buffer", "conservation buffer ccb"]

    if len(q.split()) <= 4:
        return [
            q,
            f"{q} definition and requirements",
            f"How is {q} described in the document?",
            *extras,
        ]

    return [q, f"{q} definition and requirements", f"How is {q} described in the document?", *extras]


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


def _build_context(results: List[RetrievalResult], max_chars: int, max_chunks: int) -> str:
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
    out: List[str] = []
    for r in results:
        c = format_citation(r.doc)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _make_evidence(results: List[RetrievalResult], question: str, limit: int = 6) -> List[Dict[str, Any]]:
    q_tokens = set(_tokenize(question))
    evidence: List[Dict[str, Any]] = []
    for r in results[:limit]:
        text = (r.doc.page_content or "").strip()
        # pick a "best" sentence by keyword overlap
        sentences = re.split(r"(?<=[\.\!\?])\s+", text)
        best = ""
        best_score = -1
        for s in sentences[:30]:
            s_tokens = set(_tokenize(s))
            score = len(q_tokens & s_tokens)
            if score > best_score:
                best_score = score
                best = s.strip()
        evidence.append(
            {
                "citation": format_citation(r.doc),
                "score": r.score,
                "snippet": best if best else (text[:240] + ("…" if len(text) > 240 else "")),
                "text": text[:1400] + ("…" if len(text) > 1400 else ""),
            }
        )
    return evidence


# ---------------------- Reranking (cheap) ----------------------
def _should_rerank() -> bool:
    return os.getenv("RAG_USE_RERANK", "0").strip().lower() in {"1", "true", "yes"}


def _rerank(results: List[RetrievalResult], question: str) -> List[RetrievalResult]:
    """LLM-based rerank: score top candidates 0..10 for relevance, then sort."""
    if not results:
        return results

    model = _get_env_str("RAG_RERANK_MODEL", _get_env_str("RAG_LLM_MODEL", "gpt-4o-mini"))
    keep = _get_env_int("RAG_RERANK_KEEP", 8)

    llm = ChatOpenAI(model=model, temperature=0.0)

    scored: List[Tuple[float, int, RetrievalResult]] = []
    for idx, r in enumerate(results):
        chunk = (r.doc.page_content or "").strip()
        chunk = chunk[:1200]  # cost control

        prompt = (
            "You are scoring relevance for retrieval.\n"
            "Score how relevant the CHUNK is to the QUESTION on a 0-10 scale.\n"
            "Return ONLY a number.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CHUNK:\n{chunk}\n"
        )
        try:
            resp = llm.invoke(prompt)
            txt = (getattr(resp, "content", "") or str(resp)).strip()
            m = re.search(r"(\d+(?:\.\d+)?)", txt)
            score = float(m.group(1)) if m else 0.0
        except Exception:
            score = 0.0

        scored.append((score, idx, r))

    scored.sort(key=lambda x: (-x[0], x[1]))
    reranked = [r for _, _, r in scored]
    return reranked[:keep]


# ---------------------- LLM build ----------------------
def _build_llm(llm_model_override: Optional[str] = None):
    model = llm_model_override or _get_env_str("RAG_LLM_MODEL", "gpt-4o-mini")
    temp = _get_env_float("RAG_TEMPERATURE", 0.2)
    return ChatOpenAI(model=model, temperature=temp)


def answer_question(
    question: str,
    *,
    k: Optional[int] = None,
    llm_model: Optional[str] = None,
    persist_directory: str = "chromadb",
    collection_name: str = "rag_docs",
    embed_model: Optional[str] = None,
    source_filename: Optional[str] = None,
    strict: Optional[bool] = None,
    user_query: Optional[str] = None,
) -> Dict[str, Any]:
    question = (question or "").strip()
    raw_user = (user_query or question).strip()

    # --- short-term conversational memory (FOLLOW-UP FIX) ---
    previous_q = None
    if user_query and user_query != question:
        previous_q = question

    effective_question = rewrite_followup_question(raw_user, previous_q)


    if not question:
        return {"answer": "Please enter a question.", "citations": [], "evidence": [], "retrieved": []}

    top_k = k if isinstance(k, int) and k > 0 else _get_env_int("RAG_TOP_K", 12)

    strict_default = os.getenv("RAG_STRICT_DEFAULT", "0").strip().lower() in {"1", "true", "yes"}
    strict_mode = bool(strict) if strict is not None else strict_default

    max_distance = _get_env_float("RAG_MAX_SCORE", 0.85)
    min_kw_overlap = _get_env_float("RAG_MIN_KW_OVERLAP", 0.08)

    model_name = (embed_model or os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")).strip()

    queries = _expand_queries(effective_question) or [effective_question]

    # Prefetch more if reranking
    rerank_prefetch = _get_env_int("RAG_RERANK_PREFETCH", 36)
    initial_k = rerank_prefetch if _should_rerank() else top_k

    all_results: List[RetrievalResult] = []
    for q in queries:
        all_results.extend(
            retrieve(
                q,
                k=initial_k,
                source_filename=source_filename,
                persist_directory=persist_directory,
                collection_name=collection_name,
                embed_model=model_name,
            )
        )

    # Dedup
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
        return {"answer": NOT_FOUND, "citations": [], "evidence": [], "retrieved": []}

    results = deduped

    # Rerank
    if _should_rerank():
        results = _rerank(results, effective_question)

    # Confidence gate (pre-answer)
    best_distance = min((r.score for r in results), default=None)
    best_kw = max((_keyword_score(effective_question, r.doc.page_content or "") for r in results[: min(10, len(results))]), default=0.0)

    if strict_mode:
        if best_distance is None or best_distance > max_distance or best_kw < min_kw_overlap:
            return {
                "answer": NOT_FOUND,
                "citations": [],
                "evidence": [],
                "retrieved": [{"citation": format_citation(r.doc), "score": r.score} for r in results[:12]],
            }

    context = _build_context(
        results,
        max_chars=_get_env_int("RAG_MAX_CONTEXT_CHARS", 12000),
        max_chunks=_get_env_int("RAG_MAX_CONTEXT_CHUNKS", 6),
    )

    llm = _build_llm(llm_model_override=llm_model)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a Retrieval-Augmented Generation (RAG) assistant.\n"
                    "You MUST answer using ONLY the provided context.\n\n"
                    "RULES:\n"
                    f'1) Only answer from context.\n'
                    f'2) If not in context, say exactly: "{NOT_FOUND}"\n'
                    "3) Do not use outside knowledge.\n"
                    "4) If the question is vague, ask ONE clarifying question.\n"
                    "5) Do NOT write citations.\n"
                ),
            ),
            ("human", "Question:\n{question}\n\nContext:\n{context}\n\nWrite the answer now."),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"question": effective_question, "context": context})
    response_text = (getattr(response, "content", "") or str(response) or "").strip()

    # Confidence gate (post-answer): if model says NOT_FOUND or is too generic
    if response_text == NOT_FOUND or (NOT_FOUND in response_text):
        return {"answer": NOT_FOUND, "citations": [], "evidence": [], "retrieved": []}

    if strict_mode:
        # If answer doesn't share any meaningful tokens with the question, reject.
        overlap = _keyword_score(effective_question, response_text)
        if overlap < 0.05:
            return {"answer": NOT_FOUND, "citations": [], "evidence": [], "retrieved": []}

    citations = _unique_citations(results)[:4]
    final_answer = response_text + "\n\nSources:\n" + "\n".join([f"{i}. {c}" for i, c in enumerate(citations, start=1)])

    evidence = _make_evidence(results, effective_question, limit=_get_env_int("RAG_MAX_CONTEXT_CHUNKS", 6))

    return {
        "answer": final_answer.strip(),
        "citations": citations,
        "evidence": evidence,
        "top_k": top_k,
        "expanded_queries": queries,
        "strict": strict_mode,
        "embed_model": model_name,
        "llm_model": _get_env_str("RAG_LLM_MODEL", "gpt-4o-mini"),
        "rerank": _should_rerank(),
    }
