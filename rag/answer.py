# rag/answer.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag.retriever import RetrievalResult, format_citation, retrieve

load_dotenv(override=True)

NOT_FOUND = "Not found in the provided documents."


# ---------------------- env helpers ----------------------
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


# ---------------------- query expansion ----------------------
def _expand_queries(question: str) -> List[str]:
    """
    Expand short/vague questions into related queries to increase recall.
    Includes acronym expansions and "losses/distribution constraints" variants.
    """
    q = (question or "").strip()
    if not q:
        return []

    ql = q.lower()
    extras: List[str] = []

    # Acronyms
    if "ccyb" in ql:
        extras += ["countercyclical capital buffer", "countercyclical buffer ccyb"]
    if "ccb" in ql:
        extras += ["capital conservation buffer", "conservation buffer ccb"]

    # Full phrase -> add acronym variants
    if "countercyclical capital buffer" in ql:
        extras += ["ccyb", "countercyclical buffer"]
    if "capital conservation buffer" in ql:
        extras += ["ccb", "capital conservation buffer requirement"]

    # Losses / drawdown / distributions
    if any(w in ql for w in ["loss", "losses", "incur", "draw down", "drawdown"]):
        extras += [
            "drawn down when losses are incurred",
            "constraints on capital distributions dividends buybacks bonuses",
            "distribution constraints conservation range",
        ]

    # Definitions / acronyms
    if any(w in ql for w in ["define", "stands for", "stand for", "meaning of", "what does"]):
        extras += ["definition", "abbreviation", "acronym"]

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


# ---------------------- scoring / rerank ----------------------
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


def _extract_percentages(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"\b\d+(?:\.\d+)?\s*%\b", text)


def _phrase_boost(question: str, text: str) -> float:
    ql = (question or "").lower()
    tl = (text or "").lower()
    boost = 0.0

    # Asking buffer percentage: reward the "2.5% CCB" style chunk; penalize "10.5%" total confusion
    if ("capital conservation buffer" in ql) or ("ccb" in ql):
        if "capital conservation buffer" in tl:
            boost += 0.25
        if any(w in ql for w in ["%", "percentage", "set at"]):
            if "2.5" in tl and "%" in tl and "capital conservation buffer" in tl:
                boost += 0.60
            if "10.5" in tl and "%" in tl:
                boost -= 0.30

    if ("countercyclical" in ql) or ("ccyb" in ql):
        if "countercyclical capital buffer" in tl:
            boost += 0.45
        if "capital conservation buffer" in tl and "ccyb" in ql:
            boost -= 0.25

    # Losses / constraints
    if any(w in ql for w in ["loss", "losses", "draw down", "incur"]):
        if any(w in tl for w in ["constraints", "dividend", "buyback", "bonus", "distribution"]):
            boost += 0.45

    # Definition questions: definitions often appear as "Full Term (ACR)"
    if any(w in ql for w in ["define", "meaning", "stands for", "stand for", "what does"]):
        if "(" in tl and ")" in tl:
            boost += 0.10

    return boost


# ---------------------- context building ----------------------
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


def _is_too_short(user_q: str) -> bool:
    words = [w for w in (user_q or "").split() if w]
    return len(words) <= 1


# ---------------------- FOLLOW-UP / MEMORY HELPERS ----------------------
_FOLLOWUP_HINTS = ("this", "that", "it", "the one", "earlier", "previous", "we discussed")


def _looks_like_followup(q: str) -> bool:
    ql = (q or "").lower()
    return any(h in ql for h in _FOLLOWUP_HINTS) and ("?" in ql or len(ql.split()) <= 10)


def _rewrite_followup_into_explicit(q: str) -> str:
    """
    Make common follow-ups explicit so retrieval has a chance.
    This is conservative: it doesn't invent facts; it just turns vague comparison prompts
    into explicit "compare CCB vs CCyB" style prompts that can be answered from docs.
    """
    ql = (q or "").lower()
    if "differ" in ql or "difference" in ql or "compare" in ql:
        return (
            "Compare the Capital Conservation Buffer (CCB) and the Countercyclical Capital Buffer (CCyB): "
            "purpose, when it is built up vs drawn down, and any constraints mentioned."
        )
    return q


# ---------------------- ACRONYM EXTRACTION (DETERMINISTIC) ----------------------
def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _extract_acronym_definition_from_context(context: str, acronym: str) -> Optional[str]:
    """
    Deterministically extract "Full Term (ACR)" from context.
    Returns the full term if we can find it, else None.
    """
    if not context or not acronym:
        return None

    a = acronym.strip()
    # Capture something like: "Countercyclical Capital Buffer (CCyB)"
    # (keep it generous but not too wild)
    pat = re.compile(rf"([A-Za-z][A-Za-z \-]{3,120})\(\s*{re.escape(a)}\s*\)", re.IGNORECASE)
    m = pat.search(context)
    if not m:
        return None
    term = _normalize_space(m.group(1))
    # sanity: must include at least 2 words
    if len(term.split()) < 2:
        return None
    return term


def _looks_like_acronym_question(user_q: str) -> Optional[str]:
    ql = (user_q or "").lower()
    if any(p in ql for p in ["stand for", "stands for", "meaning of", "what does"]):
        if "ccyb" in ql:
            return "CCyB"
        if "ccb" in ql:
            return "CCB"
    return None


# ---------------------- MAIN ----------------------
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
    question = (question or "").strip()
    raw_user = (user_query or question).strip()

    if not question:
        return {"answer": "Please enter a question.", "citations": [], "retrieved": []}

    top_k = k if isinstance(k, int) and k > 0 else _get_env_int("RAG_TOP_K", 12)

    env_strict = os.getenv("RAG_USE_SCORE_GATE", "").strip().lower() in {"1", "true", "yes"}
    strict_mode = bool(strict) if strict is not None else env_strict

    if strict_mode and _is_too_short(raw_user):
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

    # follow-up rewrite (helps “this/earlier” questions)
    effective_question = _rewrite_followup_into_explicit(raw_user) if _looks_like_followup(raw_user) else question

    max_distance = _get_env_float("RAG_MAX_SCORE", 0.75)
    min_kw_overlap = _get_env_float("RAG_MIN_KW_OVERLAP", 0.12)

    # ---- Multi-query retrieval
    queries = _expand_queries(effective_question) or [effective_question]
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

    # ---- Deduplicate
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

    # ---- Boosted rerank
    scored: List[Tuple[float, int, RetrievalResult]] = []
    for idx, r in enumerate(deduped):
        base = _keyword_score(effective_question, r.doc.page_content or "")
        boost = _phrase_boost(effective_question, r.doc.page_content or "")
        scored.append((base + boost, idx, r))
    scored.sort(key=lambda x: (-x[0], x[1]))
    results = [r for _, _, r in scored]

    # ---- Strict gate
    best_distance = min((r.score for r in results), default=None)
    best_kw = max((_keyword_score(effective_question, r.doc.page_content or "") for r in results[:8]), default=0.0)
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

    # ---- Deterministic acronym answers (works in BOTH Normal and Strict)
    asked_acr = _looks_like_acronym_question(raw_user)
    if asked_acr:
        term = _extract_acronym_definition_from_context(context, asked_acr)
        if term is None:
            return {"answer": NOT_FOUND, "citations": [], "retrieved": []}
        # short, safe response (no LLM required)
        citations = _unique_citations(results)[:4]
        final = f"{asked_acr} stands for {term}.\n\nSources:\n"
        for i, c in enumerate(citations, start=1):
            final += f"{i}. {c}\n"
        return {"answer": final.strip(), "citations": citations, "retrieved": []}

    # ---- Strict ambiguity check for percentages
    if strict_mode and any(w in raw_user.lower() for w in ["%", "percentage", "set at"]):
        pcts = _extract_percentages(context)
        if len(set(pcts)) >= 2:
            return {
                "answer": (
                    "I found multiple percentages in the provided context.\n\n"
                    "Which figure do you want?\n"
                    "- the buffer percentage itself\n"
                    "- or the total requirement including buffers"
                ),
                "citations": [],
                "retrieved": [{"citation": format_citation(r.doc), "score": r.score} for r in results[:8]],
            }

    # ---- LLM
    llm_name = llm_model or os.getenv("RAG_LLM_MODEL", "qwen2.5:7b-instruct")
    llm = ChatOllama(model=llm_name, temperature=_get_env_float("RAG_TEMPERATURE", 0.2))

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a Retrieval-Augmented Generation (RAG) assistant.\n"
                    "You MUST answer using ONLY the provided context.\n\n"
                    "RULES:\n"
                    f"1) Only answer from context.\n"
                    f"2) If not in context, say exactly: \"{NOT_FOUND}\"\n"
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
    response_text = (getattr(response, "content", "") or "").strip()

    if response_text == NOT_FOUND or (NOT_FOUND in response_text):
        return {"answer": NOT_FOUND, "citations": [], "retrieved": []}

    citations = _unique_citations(results)[:4]
    final_answer = response_text + "\n\nSources:\n"
    for i, c in enumerate(citations, start=1):
        final_answer += f"{i}. {c}\n"

    return {
        "answer": final_answer.strip(),
        "citations": citations,
        "retrieved": [],
        "llm_model": llm_name,
        "top_k": top_k,
        "expanded_queries": queries,
        "strict": strict_mode,
    }
