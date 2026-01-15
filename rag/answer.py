from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from rag.retriever import RetrievalResult, retrieve, format_citation


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


def _build_context(results: List[RetrievalResult], max_chars: int = 12000) -> str:
    """
    Turn retrieved chunks into a single context string.
    We keep citations next to each chunk so the model can reference them.
    """
    parts: List[str] = []
    total = 0

    for i, r in enumerate(results, start=1):
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
) -> Dict[str, Any]:
    """
    End-to-end RAG answer:
    - retrieve Top-K chunks
    - generate an answer STRICTLY grounded in retrieved context
    - return answer + citations + debug info
    """
    question = (question or "").strip()
    if not question:
        return {
            "answer": "Please enter a question.",
            "citations": [],
            "retrieved": [],
        }

    top_k = k if isinstance(k, int) and k > 0 else _get_env_int("RAG_TOP_K", 4)

    # Optional heuristic: if the best score is "too weak", we say not found.
    # IMPORTANT: Chroma score meaning can vary (distance vs similarity).
    # So we keep this OFF by default and let retrieval emptiness be the main gate.
    score_gate_enabled = os.getenv("RAG_USE_SCORE_GATE", "").strip().lower() in {"1", "true", "yes"}
    max_acceptable_score = _get_env_float("RAG_MAX_SCORE", 0.75)

    results = retrieve(
        question,
        k=top_k,
        persist_directory=persist_directory,
        collection_name=collection_name,
        embed_model=embed_model,
    )

    if not results:
        return {
            "answer": "Not found in the provided documents.",
            "citations": [],
            "retrieved": [],
        }

    if score_gate_enabled:
        best_score = results[0].score
        # If your backend uses distance: higher distance = worse match (often).
        # If it uses similarity: higher similarity = better match.
        # Because this can vary, you may need to flip this comparison later after you inspect scores.
        if best_score > max_acceptable_score:
            return {
                "answer": "Not found in the provided documents.",
                "citations": [],
                "retrieved": [{"citation": format_citation(r.doc), "score": r.score} for r in results],
            }

    context = _build_context(results, max_chars=_get_env_int("RAG_MAX_CONTEXT_CHARS", 12000))

    # Ollama chat model for answering (you can change via env: RAG_LLM_MODEL)
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
                    "1) If the answer is not explicitly supported by the context, say EXACTLY:\n"
                    "\"Not found in the provided documents.\"\n"
                    "2) DO NOT use outside knowledge.\n"
                    "3) DO NOT explain from general training data.\n"
                    "4) DO NOT speculate or add background information.\n"
                    "5) If the user input is vague (e.g., 'Basel III'), ask ONE clarifying question instead.\n"
                    "6) DO NOT write citations. The application adds citations.\n\n"

                    "STYLE:\n"
                    "- Be natural and conversational, but precise.\n"
                    "- Use short paragraphs.\n"
                    "- If the user asks a follow-up like 'why?' or 'what about that?', restate the subject briefly.\n"
                ),
            ),



            (
                "human",
                (
                    "Question:\n{question}\n\n"
                    "Context:\n{context}\n\n"
                    "Write the answer now."
                ),
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})

    response_text = (response.content or "").strip()


    citations = _unique_citations(results)[:4]  # limit to top 4 for cleanliness

    final_answer = response_text
    if response_text != "Not found in the provided documents.":
        final_answer += "\n\nSources:\n"
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
    }