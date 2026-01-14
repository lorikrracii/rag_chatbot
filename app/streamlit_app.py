# app/streamlit_app.py

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# --- Ensure repo root is importable when Streamlit runs from /app ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.answer import answer_question  # noqa: E402


# -----------------------------
# Page config + modern styling
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

st.markdown(
    """
<style>
/* Layout */
.block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1120px; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* App shell */
.rag-shell {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 22px;
  padding: 18px 18px 14px 18px;
  background:
    radial-gradient(1200px 600px at 0% 0%, rgba(99,102,241,0.18), transparent 55%),
    radial-gradient(1000px 500px at 100% 0%, rgba(16,185,129,0.14), transparent 55%),
    rgba(255,255,255,0.03);
  box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}
.rag-title { font-size: 28px; font-weight: 760; letter-spacing: -0.6px; margin: 0; }
.rag-subtitle { margin-top: 6px; opacity: 0.82; font-size: 14px; line-height: 1.35; }
.rag-divider { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 14px 0 10px 0; }

/* Chat layout */
.rag-chat { margin-top: 10px; }
.msg-row { display: flex; width: 100%; margin: 10px 0; }
.msg-left { justify-content: flex-start; }
.msg-right { justify-content: flex-end; }

/* Bubbles */
.bubble {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.03);
  box-shadow: 0 6px 18px rgba(0,0,0,0.10);
  white-space: pre-wrap;
  line-height: 1.45;
}
.bubble-user {
  max-width: 72%;
  background: linear-gradient(180deg, rgba(99,102,241,0.25), rgba(255,255,255,0.03));
  border-color: rgba(99,102,241,0.26);
}
.bubble-assistant {
  max-width: 82%;
  background: linear-gradient(180deg, rgba(16,185,129,0.18), rgba(255,255,255,0.03));
  border-color: rgba(16,185,129,0.22);
}

/* Meta line under assistant messages */
.meta {
  margin-top: 8px;
  opacity: 0.72;
  font-size: 12px;
}

/* Sources cards */
.source-card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 10px 12px;
  background: rgba(255,255,255,0.03);
  margin: 8px 0;
}
.source-card .idx { font-weight: 700; margin-right: 6px; }
.source-card .txt { opacity: 0.88; font-size: 13px; }

/* Sidebar button polish */
.stButton>button { border-radius: 14px; padding: 0.55rem 0.95rem; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def _extract_sources_from_answer(answer_text: str) -> List[str]:
    marker = "\nSources:\n"
    if not answer_text or marker not in answer_text:
        return []
    sources_part = answer_text.split(marker, 1)[1].strip()
    lines = [ln.strip() for ln in sources_part.splitlines() if ln.strip()]
    cleaned: List[str] = []
    for ln in lines:
        if len(ln) >= 3 and ln[0].isdigit() and ln[1:3] == ". ":
            cleaned.append(ln[3:].strip())
        else:
            cleaned.append(ln)
    return cleaned


def _answer_without_sources(answer_text: str) -> str:
    marker = "\nSources:\n"
    if not answer_text:
        return ""
    return answer_text.split(marker, 1)[0].strip() if marker in answer_text else answer_text.strip()


def _init_state() -> None:
    if "messages" not in st.session_state:
        # Each message:
        # { role: "user"/"assistant", content: str, sources?: List[str], meta?: str, debug?: List[dict] }
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Ask me anything about your banking PDFs. I’ll answer using retrieved passages and attach citations.",
                "sources": [],
                "meta": "",
                "debug": [],
            }
        ]


def _render_message(msg: Dict[str, Any], show_sources: bool, show_debug: bool) -> None:
    role = msg.get("role", "")
    content = msg.get("content", "") or ""
    sources: List[str] = msg.get("sources") or []
    meta: str = msg.get("meta", "") or ""
    debug_items: List[Dict[str, Any]] = msg.get("debug") or []

    if role == "user":
        st.markdown(
            f"<div class='msg-row msg-right'><div class='bubble bubble-user'>{content}</div></div>",
            unsafe_allow_html=True,
        )
        return

    # assistant
    st.markdown(
        f"<div class='msg-row msg-left'><div class='bubble bubble-assistant'>{content}</div></div>",
        unsafe_allow_html=True,
    )

    if meta:
        st.markdown(f"<div class='msg-row msg-left'><div class='meta'>{meta}</div></div>", unsafe_allow_html=True)

    if show_sources and sources and content.strip() != "Not found in the provided documents.":
        with st.expander("Sources", expanded=False):
            for i, s in enumerate(sources, start=1):
                st.markdown(
                    f"<div class='source-card'><span class='idx'>{i}.</span><span class='txt'>{s}</span></div>",
                    unsafe_allow_html=True,
                )

    if show_debug:
        with st.expander("Debug: Retrieved chunks", expanded=False):
            if not debug_items:
                st.caption("No debug info stored for this message.")
            for i, item in enumerate(debug_items, start=1):
                st.markdown(f"**{i}. {item.get('citation','')}**")
                st.caption(f"Score: {item.get('score')}")
                st.write(item.get("preview", ""))
                st.divider()


# -----------------------------
# Sidebar (clean + stable)
# -----------------------------
with st.sidebar:
    st.header("Controls")
    show_sources_panel = st.toggle("Show sources", value=True)
    show_debug = st.toggle("Debug mode", value=False)

    st.write("")
    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        _init_state()
        st.rerun()

    st.divider()
    st.caption("Answers are grounded in your PDFs. If it’s not in the documents, it will say so.")


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<div class="rag-shell">
  <div class="rag-title">RAG Chatbot</div>
  <div class="rag-subtitle">
    Retrieval-augmented answers grounded in your banking PDFs, with deterministic citations.
  </div>
  <hr class="rag-divider" />
  <div class="rag-subtitle">
    Try: <i>“What are risk-weighted assets (RWAs) and why are they important?”</i>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


# -----------------------------
# State + render history
# -----------------------------
_init_state()

st.markdown("<div class='rag-chat'></div>", unsafe_allow_html=True)
for m in st.session_state["messages"]:
    _render_message(m, show_sources_panel, show_debug)


# -----------------------------
# Input
# -----------------------------
prompt = st.chat_input("Ask a question…")

if prompt:
    # Store user message
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Call RAG
    with st.spinner("Thinking…"):
        t0 = time.time()
        # keep stable internal settings (simple and clean)
        out: Dict[str, Any] = answer_question(prompt, k=4, llm_model="llama3.2:3b")
        dt = time.time() - t0

    answer_full = out.get("answer", "") or ""
    answer_only = _answer_without_sources(answer_full)
    sources = _extract_sources_from_answer(answer_full)

    meta = f"Response time: {dt:.2f}s · Model: llama3.2:3b · Top-K: 4"
    debug_items = out.get("retrieved", []) or []

    # Store assistant message WITH sources + debug, so toggles never “lose” anything
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer_only,
            "sources": sources,
            "meta": meta,
            "debug": debug_items,
        }
    )

    st.rerun()


st.caption("Project #4 • Ollama + Chroma • Grounded answers with citations")
