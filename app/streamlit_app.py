# app/streamlit_app.py

from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.answer import answer_question  # noqa: E402


st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { padding-top: 0.9rem; padding-bottom: 6.2rem; max-width: 1250px; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Header */
.rag-shell {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 20px;
  padding: 16px 18px;
  background:
    radial-gradient(1200px 600px at 0% 0%, rgba(99,102,241,0.18), transparent 55%),
    radial-gradient(1000px 500px at 100% 0%, rgba(16,185,129,0.14), transparent 55%),
    rgba(255,255,255,0.03);
  box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}
.rag-title { font-size: 26px; font-weight: 760; letter-spacing: -0.5px; margin: 0; }
.rag-subtitle { margin-top: 6px; opacity: 0.82; font-size: 14px; line-height: 1.35; }
.pill {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  margin-right: 6px;
  opacity: 0.88;
}

/* Chat */
.msg-row { display: flex; width: 100%; margin: 10px 0; }
.msg-left { justify-content: flex-start; }
.msg-right { justify-content: flex-end; }

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
  max-width: 74%;
  background: linear-gradient(180deg, rgba(99,102,241,0.25), rgba(255,255,255,0.03));
  border-color: rgba(99,102,241,0.26);
}
.bubble-assistant {
  max-width: 86%;
  background: linear-gradient(180deg, rgba(16,185,129,0.18), rgba(255,255,255,0.03));
  border-color: rgba(16,185,129,0.22);
}
.meta {
  margin-top: 6px;
  opacity: 0.72;
  font-size: 12px;
}

/* Minimal sources */
.sources-line {
  margin-top: 8px;
  font-size: 12px;
  opacity: 0.78;
}
.sources-line code {
  font-size: 12px;
  padding: 1px 6px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  margin-right: 6px;
  display: inline-block;
  margin-top: 6px;
}

/* Right panel */
.panel {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 14px 14px 10px 14px;
  background: rgba(255,255,255,0.03);
  box-shadow: 0 10px 30px rgba(0,0,0,0.14);
  position: sticky;
  top: 14px;
}
.panel-title { font-weight: 700; font-size: 14px; margin-bottom: 10px; opacity: 0.9; }

/* Fixed composer */
.rag-composer {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 9999;
  padding: 14px 0 16px 0;
  background: linear-gradient(180deg, rgba(0,0,0,0), rgba(0,0,0,0.35), rgba(0,0,0,0.55));
  backdrop-filter: blur(10px);
}
.rag-composer-inner { max-width: 1250px; margin: 0 auto; padding: 0 1rem; }
.rag-composer-card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  background: rgba(15,15,18,0.55);
  box-shadow: 0 12px 30px rgba(0,0,0,0.22);
  padding: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)


def now_hhmm() -> str:
    return datetime.now().strftime("%H:%M")


def extract_sources(answer_text: str) -> List[str]:
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


def answer_without_sources(answer_text: str) -> str:
    marker = "\nSources:\n"
    if not answer_text:
        return ""
    return answer_text.split(marker, 1)[0].strip() if marker in answer_text else answer_text.strip()

def normalize_query(q: str) -> str:
    q = (q or "").strip()
    # short inputs like "CET1" or "Basel III" often retrieve poorly
    if len(q.split()) <= 3:
        return f"Explain {q} in the context of banking regulation and capital requirements."
    return q


_SMALLTALK = re.compile(
    r"^\s*(hi|hello|hey|yo|sup|thanks|thank you|thx|good morning|good afternoon|good evening|how are you|who are you)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def is_smalltalk(text: str) -> bool:
    return bool(_SMALLTALK.match(text or ""))


def smalltalk_reply(text: str) -> str:
    t = (text or "").strip().lower()
    if "who are you" in t:
        return "I’m your RAG assistant — I answer questions using your uploaded banking PDFs and cite the exact places I used."
    if "how are you" in t:
        return "Doing good 😄 Send me a question about your PDFs and I’ll pull the relevant parts."
    if "thank" in t or "thx" in t:
        return "Anytime. Want to test something harder from the documents?"
    return "Hey! 👋 Ask me anything about your banking PDFs (Basel III, CET1, RWAs, buffers, etc.)."


def render_message(msg: Dict[str, Any]) -> None:
    role = msg.get("role", "")
    content = msg.get("content", "") or ""
    ts = msg.get("ts", "")
    sources: List[str] = msg.get("sources", []) or ""

    if role == "user":
        st.markdown(
            f"<div class='msg-row msg-right'><div class='bubble bubble-user'>{content}</div></div>"
            f"<div class='msg-row msg-right'><div class='meta'>{ts}</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='msg-row msg-left'><div class='bubble bubble-assistant'>{content}</div></div>"
            f"<div class='msg-row msg-left'><div class='meta'>{ts}</div></div>",
            unsafe_allow_html=True,
        )
        if sources:
            st.markdown(
                "<div class='msg-row msg-left'><div class='sources-line'>Sources: "
                + "".join([f"<code>{s}</code>" for s in sources[:4]])
                + "</div></div>",
                unsafe_allow_html=True,
            )


# ---- State init
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hey — ask me anything about your banking PDFs.", "ts": now_hhmm(), "sources": []}
    ]
st.session_state.setdefault("last_sources", [])
st.session_state.setdefault("pending_prompt", None)


# ---- Sidebar
with st.sidebar:
    st.header("Controls")
    st.toggle("Show sources panel", key="show_sources_panel", value=True)
    if st.button("Clear chat"):
        st.session_state.clear()
        st.rerun()
    st.caption("Knowledge questions use your PDFs + citations. Small talk works too.")


# ---- Header
st.markdown(
    """
<div class="rag-shell">
  <div class="rag-title">RAG Chatbot</div>
  <div class="rag-subtitle">Grounded answers from your banking PDFs, plus normal conversation.</div>
  <div style="margin-top:10px;">
    <span class="pill">Chat</span>
    <span class="pill">RAG</span>
    <span class="pill">Citations</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([2.2, 1.0], gap="large")

with left:
    for m in st.session_state["messages"]:
        render_message(m)

with right:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Sources</div>", unsafe_allow_html=True)
    if st.session_state.get("show_sources_panel", True):
        srcs = st.session_state.get("last_sources", [])
        if not srcs:
            st.caption("Sources will appear here after a document-based question.")
        else:
            st.markdown("<div class='sources-line'>", unsafe_allow_html=True)
            st.markdown("".join([f"<code>{s}</code>" for s in srcs[:6]]), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.caption("Hidden.")
    st.markdown("</div>", unsafe_allow_html=True)


# ---- Fixed composer (bottom)
st.markdown("<div class='rag-composer'><div class='rag-composer-inner'><div class='rag-composer-card'>", unsafe_allow_html=True)

with st.form("composer", clear_on_submit=True):
    prompt = st.text_input("Message", label_visibility="collapsed", placeholder="Ask a question…")
    send = st.form_submit_button("Send")

st.markdown("</div></div></div>", unsafe_allow_html=True)

# ---- Two-step submit: show user immediately, answer on next run
if send and prompt.strip():
    st.session_state["messages"].append({"role": "user", "content": prompt, "ts": now_hhmm()})
    st.session_state["pending_prompt"] = prompt
    st.rerun()

pending: Optional[str] = st.session_state.get("pending_prompt")
if pending:
    # clear pending first so we don't double-run on reruns
    st.session_state["pending_prompt"] = None

    # small talk path (no retrieval, no sources)
    if is_smalltalk(pending):
        st.session_state["last_sources"] = []
        st.session_state["messages"].append(
            {"role": "assistant", "content": smalltalk_reply(pending), "ts": now_hhmm(), "sources": []}
        )
        st.rerun()

    # RAG path
    with st.spinner("Thinking…"):
        query = normalize_query(pending)
        out: Dict[str, Any] = answer_question(query, k=4, llm_model="llama3.2:3b")


    full = out.get("answer", "") or ""
    content = answer_without_sources(full)
    sources = extract_sources(full)
    if content.strip() == "Not found in the provided documents.":
        sources = []


    st.session_state["last_sources"] = sources
    st.session_state["messages"].append({"role": "assistant", "content": content, "ts": now_hhmm(), "sources": sources})

    st.rerun()

st.caption("Project #4 • Ollama + Chroma • Grounded answers with citations")
