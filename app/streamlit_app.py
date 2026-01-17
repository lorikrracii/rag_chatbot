from __future__ import annotations

import streamlit.components.v1 as components

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.answer import answer_question  # noqa: E402


st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None,  # Hide default menu but keep sidebar
)

# ---------- CSS ----------
st.markdown(
    """
<style>
/* -------- Base layout tweaks -------- */
.block-container {
  padding-top: 4rem; /* Extra space to account for Streamlit header */
  padding-bottom: 6.6rem; /* more space for fixed composer */
  max-width: 1320px;
}
/* Ensure main content area has proper spacing when sidebar is open */
.main .block-container {
  padding-left: 1rem;
  padding-right: 1rem;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* Keep header visible for sidebar toggle button but ensure proper spacing */
header { 
  visibility: visible;
  position: relative;
  z-index: 100;
}
/* Ensure content starts below header with proper spacing */
.stApp > div:first-child {
  padding-top: 0;
}
/* Add extra top margin to first content element to push hero below header */
.stApp .block-container > div:first-child {
  margin-top: 2rem;
}
/* Ensure sidebar is always visible and properly styled */
section[data-testid="stSidebar"] {
  visibility: visible !important;
  display: block !important;
}
/* Make sidebar toggle button visible */
button[data-testid="baseButton-header"],
button[kind="header"] {
  visibility: visible !important;
  display: block !important;
}
/* Improve sidebar visibility and styling */
.css-1d391kg {
  visibility: visible !important;
}
/* Ensure sidebar content is visible */
[data-testid="stSidebar"] > div {
  visibility: visible !important;
}

/* -------- Design tokens -------- */
:root{
  --card-bg: rgba(255,255,255,0.04);
  --card-border: rgba(255,255,255,0.10);
  --soft-border: rgba(255,255,255,0.08);
  --shadow: 0 14px 40px rgba(0,0,0,0.22);
  --shadow-soft: 0 10px 24px rgba(0,0,0,0.18);
  --radius-xl: 20px;
  --radius-lg: 16px;
  --radius-md: 12px;

  --ink: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.70);
  --muted2: rgba(255,255,255,0.58);

  --accentA: rgba(99,102,241,0.95);   /* indigo */
  --accentB: rgba(16,185,129,0.92);   /* emerald */
  --accentC: rgba(236,72,153,0.88);   /* pink */
}

/* -------- Top hero header -------- */
.rag-hero {
  border: 1px solid var(--card-border);
  border-radius: var(--radius-xl);
  padding: 18px 18px 14px 18px;
  margin-top: 1.5rem; /* Space from Streamlit header */
  margin-bottom: 1.5rem;
  background:
    radial-gradient(1200px 600px at 0% 0%, rgba(99,102,241,0.22), transparent 60%),
    radial-gradient(1000px 520px at 100% 0%, rgba(16,185,129,0.18), transparent 55%),
    radial-gradient(1100px 520px at 60% 120%, rgba(236,72,153,0.10), transparent 55%),
    rgba(255,255,255,0.03);
  box-shadow: var(--shadow);
  position: relative;
  z-index: 1;
}

.rag-hero-top {
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap: 12px;
}

.rag-brand {
  display:flex;
  flex-direction:column;
  gap: 6px;
}

.rag-title {
  margin:0;
  font-size: 22px;
  font-weight: 780;
  letter-spacing: -0.45px;
  color: var(--ink);
}

.rag-subtitle {
  margin:0;
  font-size: 13px;
  line-height: 1.35;
  color: var(--muted);
}

.hero-chips {
  display:flex;
  flex-wrap:wrap;
  gap: 8px;
  justify-content:flex-end;
  margin-top: 2px;
}

.chip {
  display:inline-flex;
  align-items:center;
  gap: 7px;
  padding: 5px 10px;
  border-radius: 999px;
  border: 1px solid var(--soft-border);
  background: rgba(255,255,255,0.03);
  font-size: 12px;
  color: var(--muted);
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  display:inline-block;
  background: rgba(255,255,255,0.30);
}

.dot-ok { background: rgba(16,185,129,0.92); box-shadow: 0 0 0 4px rgba(16,185,129,0.12); }
.dot-warn { background: rgba(245,158,11,0.95); box-shadow: 0 0 0 4px rgba(245,158,11,0.12); }

/* -------- Main columns wrappers -------- */
.rag-grid {
  display: grid;
  grid-template-columns: 1fr 360px;
  gap: 18px;
  align-items: start;
}
@media (max-width: 1100px) {
  .rag-grid { grid-template-columns: 1fr; }
}

/* -------- Chat surface -------- */
.chat-surface {
  border: 1px solid var(--card-border);
  border-radius: var(--radius-xl);
  background: rgba(255,255,255,0.02);
  box-shadow: var(--shadow-soft);
  overflow: hidden;
}

.chat-surface-head {
  padding: 12px 14px;
  border-bottom: 1px solid rgba(255,255,255,0.07);
  display:flex;
  align-items:center;
  justify-content:space-between;
}

.chat-surface-title {
  font-weight: 720;
  font-size: 13px;
  color: var(--muted);
  letter-spacing: 0.2px;
}

.chat-badges {
  display:flex; gap: 8px; flex-wrap:wrap;
}
.badge {
  font-size: 11px;
  padding: 4px 9px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
  color: var(--muted2);
}

.chat-scroll {
  padding: 12px 14px 6px 14px;
}

/* -------- Message bubbles -------- */
.msg-row {
  display:flex;
  width: 100%;
  margin: 10px 0;
  gap: 10px;
}

.msg-left { justify-content:flex-start; }
.msg-right { justify-content:flex-end; }

.avatar {
  width: 34px;
  height: 34px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.04);
  display:flex;
  align-items:center;
  justify-content:center;
  flex: 0 0 auto;
  box-shadow: 0 10px 24px rgba(0,0,0,0.14);
  font-size: 16px;
}

.bubble {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 10px 12px;
  background: rgba(255,255,255,0.03);
  box-shadow: 0 10px 24px rgba(0,0,0,0.12);
  white-space: pre-wrap;
  line-height: 1.48;
  font-size: 14px;
  color: var(--ink);
  max-width: 78%;
}

.bubble-user {
  background: linear-gradient(180deg, rgba(99,102,241,0.24), rgba(255,255,255,0.03));
  border-color: rgba(99,102,241,0.28);
}

.bubble-assistant {
  background: linear-gradient(180deg, rgba(16,185,129,0.16), rgba(255,255,255,0.03));
  border-color: rgba(16,185,129,0.20);
}

.meta {
  margin-top: 5px;
  font-size: 11px;
  color: var(--muted2);
}

.sources-line {
  margin-top: 6px;
  font-size: 12px;
  color: var(--muted);
}

.source-chip {
  display:inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  margin-right: 6px;
  margin-top: 6px;
  font-size: 12px;
  color: var(--muted);
}

/* -------- Right panel -------- */
.panel {
  border: 1px solid var(--card-border);
  border-radius: var(--radius-xl);
  padding: 14px;
  background: rgba(255,255,255,0.03);
  box-shadow: var(--shadow-soft);
  position: sticky;
  top: 14px;
}

.panel-title {
  font-weight: 760;
  font-size: 13px;
  color: var(--muted);
  letter-spacing: 0.2px;
  margin-bottom: 10px;
}

.panel-card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  background: rgba(255,255,255,0.02);
  padding: 10px;
  margin-bottom: 10px;
}

.panel-kv {
  display:flex;
  justify-content:space-between;
  gap: 10px;
  font-size: 12px;
  color: var(--muted);
  margin: 6px 0;
}
.panel-kv code {
  padding: 1px 6px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
}

/* -------- Fixed composer -------- */
.rag-composer {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 999;
  padding: 12px 0 14px 0;
  background:
    linear-gradient(180deg, rgba(10,10,12,0.0), rgba(10,10,12,0.88) 30%, rgba(10,10,12,0.92));
  border-top: 1px solid rgba(255,255,255,0.08);
  backdrop-filter: blur(8px);
}
/* Improve sidebar styling */
[data-testid="stSidebar"] {
  background-color: rgba(15, 15, 20, 0.95) !important;
  border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
}
/* Ensure sidebar content is readable */
[data-testid="stSidebar"] * {
  color: rgba(255, 255, 255, 0.9) !important;
}
/* Make sidebar headers more visible */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  color: rgba(255, 255, 255, 0.95) !important;
}

.rag-composer-inner {
  max-width: 1320px;
  margin: 0 auto;
  padding: 0 1rem;
}

.rag-composer-card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  background: rgba(18,18,22,0.88);
  box-shadow: 0 18px 50px rgba(0,0,0,0.35);
  padding: 10px;
}

.hintline {
  margin-top: 6px;
  font-size: 12px;
  color: var(--muted2);
}

/* Make Streamlit input + button look tighter inside composer */
.rag-composer-card input {
  border-radius: 14px !important;
}

/* Additional fixes for sidebar visibility */
.stApp > header {
  visibility: visible !important;
}
.stApp [data-testid="stSidebar"] {
  visibility: visible !important;
  min-width: 21rem !important;
}
/* Ensure sidebar toggle is always visible */
.stApp button[kind="header"] {
  visibility: visible !important;
  display: inline-flex !important;
}
</style>
<script>
// Force sidebar to be visible on page load
(function() {
  const sidebar = document.querySelector('[data-testid="stSidebar"]');
  if (sidebar) {
    sidebar.style.visibility = 'visible';
    sidebar.style.display = 'block';
  }
  // Also ensure toggle button is visible
  const toggleBtn = document.querySelector('button[kind="header"]');
  if (toggleBtn) {
    toggleBtn.style.visibility = 'visible';
    toggleBtn.style.display = 'inline-flex';
  }
})();
</script>
""",
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def now_hhmm() -> str:
    return datetime.now().strftime("%H:%M")


def autoscroll_to_bottom() -> None:
    st.markdown(
        """
        <script>
        const el = window.parent.document.querySelector('section.main');
        if (el) { el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' }); }
        </script>
        """,
        unsafe_allow_html=True,
    )


def list_pdf_filenames() -> List[str]:
    data_dir = os.getenv("DATA_DIR", "data/raw")
    base = Path(data_dir)
    if not base.exists():
        return []
    return sorted({p.name for p in base.rglob("*.pdf")})


def health_check() -> Dict[str, Any]:
    chroma_dir = os.getenv("CHROMA_DIR", "chromadb")
    collection = os.getenv("CHROMA_COLLECTION", "rag_docs")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    llm_model = os.getenv("RAG_LLM_MODEL", "qwen2.5:7b-instruct")


    info: Dict[str, Any] = {
        "chroma_dir": chroma_dir,
        "collection": collection,
        "embed_model": embed_model,
        "llm_model": llm_model,
        "chroma_dir_exists": Path(chroma_dir).exists(),
    }

    try:
        from rag.retriever import get_vectorstore

        vs = get_vectorstore(
            persist_directory=chroma_dir,
            collection_name=collection,
            embed_model=embed_model,
        )

        count = None
        try:
            count = vs._collection.count()
        except Exception:
            pass

        info["vectorstore_ok"] = True
        info["vector_count"] = count
    except Exception as e:
        info["vectorstore_ok"] = False
        info["vector_count"] = None
        info["error"] = str(e)

    return info


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
    if len(q.split()) <= 3:
        return f"Explain {q} in the context of banking regulation and capital requirements."
    return q


def build_memory_hint(messages: List[Dict[str, Any]], max_user_turns: int = 2) -> str:
    users = [m.get("content", "") for m in messages if m.get("role") == "user"]
    users = [u.strip() for u in users if u.strip()]
    last = users[-max_user_turns:]
    if not last:
        return ""
    return "Previous user questions:\n- " + "\n- ".join(last)


def autoscroll_to_anchor() -> None:
    components.html(
        """
        <script>
        const el = window.parent.document.getElementById("chat-bottom");
        if (el) { el.scrollIntoView({behavior: "smooth", block: "end"}); }
        </script>
        """,
        height=0,
    )


_SMALLTALK = re.compile(
    r"^\s*(hi|hello|hey|yo|sup|thanks|thank you|thx|good morning|good afternoon|good evening|how are you|who are you)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

_INJECTION = re.compile(
    r"(ignore (all|previous|prior) instructions|"
    r"reveal (the )?(system|developer) prompt|"
    r"(system|developer) prompt|developer message|"
    r"jailbreak|do anything now|DAN|"
    r"bypass|override|"
    r"hidden rules|"
    r"print your prompt)",
    re.IGNORECASE,
)

_PUNCT_ONLY = re.compile(r"^\s*[\W_]+\s*$")


def is_smalltalk(text: str) -> bool:
    return bool(_SMALLTALK.match(text or ""))


def is_injection(text: str) -> bool:
    return bool(_INJECTION.search(text or ""))


def is_garbage_input(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if _PUNCT_ONLY.match(t):
        return True
    if len(t) < 2:
        return True
    return False


def is_casual_statement(text: str) -> bool:
    """
    Detect casual statements that aren't questions about documents.
    Examples: "i am lorik", "my name is...", "i like...", etc.
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    
    # Check if it's a question
    question_words = ["what", "how", "why", "when", "where", "who", "which", "explain", "tell me", "describe", "define"]
    is_question = any(t.startswith(qw) for qw in question_words) or "?" in t
    
    # Banking/document-related keywords
    banking_keywords = ["cet1", "rwa", "capital", "buffer", "basel", "pra", "pillar", "banking", "regulation", 
                       "requirement", "ratio", "ccyb", "conservation", "countercyclical", "risk", "asset", 
                       "supervisory", "framework", "methodology", "reform"]
    has_banking_keywords = any(kw in t for kw in banking_keywords)
    
    # If it has banking keywords, it's probably a valid question even if casual
    if has_banking_keywords:
        return False
    
    # If it's a question, let it through
    if is_question:
        return False
    
    # Common casual statement patterns (first person statements)
    casual_patterns = [
        r"^(i|my|i'm|i am|my name is|i like|i love|i hate|i think|i feel|i want|i need|i have|i do|i did|i was|i will)",
        r"^(this is|that is|here is|there is|it is|it's)",
        r"^(nice|good|bad|great|awesome|cool|hello|hi|hey)",
        r"^(i am|i'm) [a-z]+$",  # "i am lorik", "i'm john", etc.
    ]
    
    is_casual = any(re.match(pattern, t) for pattern in casual_patterns)
    
    # If it's a casual statement without banking keywords and not a question
    return is_casual


def smalltalk_reply(text: str) -> str:
    t = (text or "").strip().lower()
    if "who are you" in t:
        return "I’m your RAG assistant — I answer questions using your uploaded banking PDFs and cite the exact places I used."
    if "how are you" in t:
        return "Doing good 😄 Send me a question about your PDFs and I’ll pull the relevant parts."
    if "thank" in t or "thx" in t:
        return "Anytime. Want to test something harder from the documents?"
    return "Hey! 👋 Ask me anything about your banking PDFs (Basel III, CET1, RWAs, buffers, etc.)."


def injection_reply() -> str:
    return (
        "I can’t help with attempts to override instructions or reveal system prompts.\n\n"
        "Ask a normal question about the PDFs (e.g., CET1, RWAs, capital buffers), and I’ll answer with citations."
    )


def render_message(msg: Dict[str, Any]) -> None:
    role = msg.get("role", "")
    content = msg.get("content", "") or ""
    ts = msg.get("ts", "")
    sources: List[str] = msg.get("sources", []) or []

    if role == "user":
        st.markdown(
            f"""
<div class="msg-row msg-right">
  <div class="bubble bubble-user">{content}</div>
  <div class="avatar">🧑</div>
</div>
<div class="msg-row msg-right">
  <div class="meta">{ts}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
<div class="msg-row msg-left">
  <div class="avatar">📚</div>
  <div>
    <div class="bubble bubble-assistant">{content}</div>
    <div class="meta">{ts}</div>
    {"" if not sources else '<div class="sources-line">Sources: ' + "".join([f'<span class="source-chip">{s}</span>' for s in sources[:4]]) + '</div>'}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


# ---------- State init ----------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hey — ask me anything about your banking PDFs.", "ts": now_hhmm(), "sources": []}
    ]
st.session_state.setdefault("last_sources", [])
st.session_state.setdefault("pending_prompt", None)
st.session_state.setdefault("is_processing", False)


# ---------- Sidebar ----------
with st.sidebar:
    is_locked = st.session_state.get("is_processing", False)

    st.header("Control Center")

    st.toggle("Show Sources / System panel", key="show_sources_panel", value=True, disabled=is_locked)

    if st.button("Clear chat", disabled=is_locked, use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.divider()

    pdf_names = list_pdf_filenames()
    doc_choice = st.selectbox(
        "Filter by document (filename)",
        options=["All"] + pdf_names,
        index=0,
        disabled=is_locked,
    )

    strict = st.selectbox("Answer strictness", ["Normal", "Strict"], index=0, disabled=is_locked)
    if strict == "Strict":
        os.environ["RAG_USE_SCORE_GATE"] = "true"
        os.environ["RAG_MAX_SCORE"] = "0.75"
    else:
        os.environ["RAG_USE_SCORE_GATE"] = "false"

    st.caption("Knowledge questions use your PDFs + citations. Small talk works too.")

    with st.expander("Health check", expanded=False):
        hc = health_check()

        st.write("Chroma dir:", f"`{hc['chroma_dir']}`")
        st.write("Collection:", f"`{hc['collection']}`")
        st.write("Embed model:", f"`{hc['embed_model']}`")
        st.write("LLM model:", f"`{hc['llm_model']}`")

        st.write("DB folder exists:", "✅" if hc["chroma_dir_exists"] else "❌")

        if hc.get("vectorstore_ok"):
            st.write("Vectorstore:", "✅ loaded")
            if hc.get("vector_count") is not None:
                st.write("Vectors indexed:", hc["vector_count"])
        else:
            st.write("Vectorstore:", "❌ error")
            st.caption(hc.get("error", "Unknown error"))


# ---------- Header ----------
# Add spacer to push hero below Streamlit header
st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)

hc_top = health_check()
db_ok = bool(hc_top.get("vectorstore_ok")) and bool(hc_top.get("chroma_dir_exists"))
st.markdown(
    f"""
<div class="rag-hero">
  <div class="rag-hero-top">
    <div class="rag-brand">
      <div class="rag-title">RAG Chatbot</div>
      <div class="rag-subtitle">Grounded answers from your banking PDFs — with citations you can verify.</div>
    </div>
    <div class="hero-chips">
      <span class="chip"><span class="dot dot-ok"></span>Gemini</span>
      <span class="chip"><span class="dot dot-ok"></span>Chroma</span>
      <span class="chip"><span class="dot {'dot-ok' if db_ok else 'dot-warn'}"></span>{'Index Ready' if db_ok else 'Index Check'}</span>
    </div>
  </div>
  <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
    <span class="badge">RAG</span>
    <span class="badge">Citations</span>
    <span class="badge">Injection Guard</span>
    <span class="badge">Short Memory</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# ---------- Main layout (chat + right panel) ----------
show_panel = st.session_state.get("show_sources_panel", True)

if show_panel:
    left, right = st.columns([1.75, 1], gap="large")
else:
    left = st.container()
    right = None

# ---- Chat surface (left) ----
with left:
    st.markdown(
        """
<div class="chat-surface">
  <div class="chat-surface-head">
    <div class="chat-surface-title">Conversation</div>
    <div class="chat-badges">
      <span class="badge">Top-K retrieval</span>
      <span class="badge">Grounded answers</span>
    </div>
  </div>
  <div class="chat-scroll">
""",
        unsafe_allow_html=True,
    )

    for m in st.session_state["messages"]:
        render_message(m)

    st.markdown("<div id='chat-bottom'></div>", unsafe_allow_html=True)
    autoscroll_to_anchor()

    st.markdown("</div></div>", unsafe_allow_html=True)

# ---- Right panel (sources + system) ----
if right is not None:
    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>Sources & System</div>", unsafe_allow_html=True)

        # Sources card
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title' style='margin-bottom:6px;'>Last answer sources</div>", unsafe_allow_html=True)

        srcs = st.session_state.get("last_sources", [])
        if srcs:
            st.markdown(
                "<div class='sources-line'>"
                + "".join([f"<span class='source-chip'>{s}</span>" for s in srcs[:10]])
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No sources yet — ask a question to see citations here.")
        st.markdown("</div>", unsafe_allow_html=True)

        # System card
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title' style='margin-bottom:6px;'>System</div>", unsafe_allow_html=True)
        hc = hc_top

        st.markdown(
            f"""
<div class="panel-kv"><span>Collection</span><code>{hc.get('collection','')}</code></div>
<div class="panel-kv"><span>Embed model</span><code>{hc.get('embed_model','')}</code></div>
<div class="panel-kv"><span>LLM model</span><code>{hc.get('llm_model','')}</code></div>
""",
            unsafe_allow_html=True,
        )
        if hc.get("vectorstore_ok") and hc.get("vector_count") is not None:
            st.markdown(
                f"<div class='panel-kv'><span>Vectors indexed</span><code>{hc.get('vector_count')}</code></div>",
                unsafe_allow_html=True,
            )
        elif not hc.get("vectorstore_ok"):
            st.caption(hc.get("error", "Vectorstore not available."))

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ---------- Fixed composer ----------
st.markdown(
    "<div class='rag-composer'><div class='rag-composer-inner'><div class='rag-composer-card'>",
    unsafe_allow_html=True,
)

with st.form("composer", clear_on_submit=True):
    c1, c2 = st.columns([6, 1])
    with c1:
        prompt = st.text_input("Message", label_visibility="collapsed", placeholder="Ask a question about Basel III, CET1, RWAs, buffers…")
    with c2:
        send = st.form_submit_button("Send", use_container_width=True)

if st.session_state.get("is_processing", False):
    st.markdown("<div class='hintline'>Thinking…</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='hintline'>Tip: keep questions specific for better citations.</div>", unsafe_allow_html=True)

st.markdown("</div></div></div>", unsafe_allow_html=True)


# ---------- Two-step submit ----------
if send and prompt.strip():
    st.session_state["messages"].append({"role": "user", "content": prompt, "ts": now_hhmm(), "sources": []})
    st.session_state["pending_prompt"] = prompt
    st.session_state["is_processing"] = True
    st.rerun()


pending: Optional[str] = st.session_state.get("pending_prompt")
if pending:
    st.session_state["pending_prompt"] = None
    st.session_state["is_processing"] = True

    # smalltalk (no retrieval)
    if is_smalltalk(pending):
        st.session_state["is_processing"] = False
        st.session_state["last_sources"] = []
        st.session_state["messages"].append(
            {"role": "assistant", "content": smalltalk_reply(pending), "ts": now_hhmm(), "sources": []}
        )
        st.rerun()

    # injection guard (no retrieval)
    if is_injection(pending):
        st.session_state["is_processing"] = False
        st.session_state["last_sources"] = []
        st.session_state["messages"].append(
            {"role": "assistant", "content": injection_reply(), "ts": now_hhmm(), "sources": []}
        )
        st.rerun()

    # garbage input guard (no retrieval)
    if is_garbage_input(pending):
        st.session_state["is_processing"] = False
        st.session_state["last_sources"] = []
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": "Type a real question (not just punctuation 🙂).",
                "ts": now_hhmm(),
                "sources": [],
            }
        )
        st.rerun()

    # casual statement guard (no retrieval)
    if is_casual_statement(pending):
        st.session_state["is_processing"] = False
        st.session_state["last_sources"] = []
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": "Nice to meet you! 👋 I'm here to answer questions about your banking PDFs (Basel III, CET1, RWAs, capital buffers, etc.). What would you like to know?",
                "ts": now_hhmm(),
                "sources": [],
            }
        )
        st.rerun()

    # RAG path with error handling
    try:
        with st.spinner("Thinking…"):
            memory = build_memory_hint(st.session_state["messages"], max_user_turns=2)
            base_query = normalize_query(pending)
            query = f"{memory}\n\nCurrent question:\n{base_query}" if memory else base_query

            out: Dict[str, Any] = answer_question(
                query,
                k=12,
                llm_model=None,
                source_filename=None if doc_choice == "All" else doc_choice,
                strict=(strict == "Strict"),
                user_query=pending,  # raw user input (before normalize_query)
            )

        st.session_state["is_processing"] = False

        full = out.get("answer", "") or ""
        content = answer_without_sources(full)
        sources = extract_sources(full)
        if content.strip() == "Not found in the provided documents.":
            sources = []

        st.session_state["last_sources"] = sources
        st.session_state["messages"].append({"role": "assistant", "content": content, "ts": now_hhmm(), "sources": sources})
        st.rerun()
    
    except Exception as e:
        st.session_state["is_processing"] = False
        st.session_state["last_sources"] = []
        error_msg = f"I encountered an error while processing your question. Please try rephrasing it or ask about banking regulations (CET1, RWAs, capital buffers, etc.).\n\nError: {str(e)}"
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": error_msg,
                "ts": now_hhmm(),
                "sources": [],
            }
        )
        st.rerun()

st.caption("Project #4 • Gemini + Chroma • Grounded answers with citations")
