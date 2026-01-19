from __future__ import annotations

import html
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Load env
load_dotenv(override=True)

# Path setup
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Core logic import
from rag.answer import answer_question  # noqa: E402


# ---------------- Page config ----------------
st.set_page_config(
    page_title="RAG Knowledge Navigator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- Modern UI CSS ----------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #f1f5f9;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 10rem;
    max-width: 1200px;
}

/* Hide some default elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { background: rgba(0,0,0,0) !important; }

/* Tokens */
:root {
    --accent: #6366f1;
    --accent-glow: rgba(99, 102, 241, 0.3);
    --bg-card: rgba(255, 255, 255, 0.03);
    --border: rgba(255, 255, 255, 0.1);
    --text-muted: #94a3b8;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: rgba(15, 15, 20, 0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.10) !important;
}
section[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.90) !important;
}

/* Hero */
.hero-section {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(16, 185, 129, 0.05) 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 2rem;
    margin-bottom: 2.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.hero-title {
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 8px;
    background: linear-gradient(to right, #fff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}

/* Messages */
.msg-row {
    display: flex;
    margin: 20px 0;
    width: 100%;
    align-items: flex-start;
    gap: 14px;
}
.msg-right { flex-direction: row-reverse; }

.bubble {
    padding: 16px 20px;
    border-radius: 20px;
    max-width: 75%;
    font-size: 15px;
    line-height: 1.6;
    border: 1px solid var(--border);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    white-space: pre-wrap;
}

.bubble-user {
    background: var(--accent);
    color: white;
    border-bottom-right-radius: 4px;
    box-shadow: 0 4px 15px var(--accent-glow);
}

.bubble-assistant {
    background: rgba(255,255,255,0.06);
    color: #e2e8f0;
    border-bottom-left-radius: 4px;
    backdrop-filter: blur(5px);
}

.avatar {
    width: 38px; height: 38px;
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    border: 1px solid var(--border);
}

.source-pill {
    display: inline-block;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    padding: 3px 10px;
    border-radius: 8px;
    font-size: 11px;
    margin-top: 10px;
    margin-right: 6px;
    color: #a5b4fc;
}

.time-stamp {
    font-size: 10px;
    color: #94a3b8;
    opacity: 0.8;
    margin-top: 8px;
    text-align: right;
    display: block;
    width: 100%;
}

/* Fixed composer */
.fixed-composer {
    position: fixed;
    bottom: 22px;
    left: 50%;
    transform: translateX(-50%);
    width: 85%;
    max-width: 900px;
    z-index: 1000;
    background: rgba(15, 15, 20, 0.85);
    backdrop-filter: blur(15px);
    padding: 18px 24px;
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 20px 50px rgba(0,0,0,0.5);
}

.status-badge {
    padding: 5px 14px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(255,255,255,0.05);
    color: var(--text-muted);
    border: 1px solid var(--border);
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- Helpers ----------------
def now_hhmm() -> str:
    return datetime.now().strftime("%H:%M")


def list_pdf_filenames() -> List[str]:
    data_dir = os.getenv("DATA_DIR", "data/raw")
    base = Path(data_dir)
    if not base.exists():
        return []
    return sorted({p.name for p in base.rglob("*.pdf")})


def _llm_info_from_env() -> Dict[str, str]:
    return {"backend": "openai", "model": os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")}


def _embed_info_from_env() -> Dict[str, str]:
    return {"backend": "openai", "model": os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")}

# --- Auto-ingest on Streamlit Cloud (first run) ---
def ensure_chroma_exists() -> None:
    chroma_dir = os.getenv("CHROMA_DIR", "chromadb")
    collection = os.getenv("CHROMA_COLLECTION", "rag_docs")

    # If DB folder exists and has content, do nothing
    if Path(chroma_dir).exists() and any(Path(chroma_dir).iterdir()):
        return

    # If no PDFs, can't ingest
    data_dir = os.getenv("DATA_DIR", "data/raw")
    pdfs = list(Path(data_dir).rglob("*.pdf")) if Path(data_dir).exists() else []
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {data_dir}. Add your PDFs to the repo under data/raw/."
        )

    from rag.ingest import ingest, IngestConfig

    cfg = IngestConfig()
    cfg.chroma_dir = chroma_dir
    cfg.collection_name = collection
    cfg.reset_chroma = True  # safe on first run

    with st.spinner("📦 First run: building the vector database (one-time)…"):
        ingest(cfg)

# Run once per session
if "BOOTSTRAPPED" not in st.session_state:
    ensure_chroma_exists()
    st.session_state["BOOTSTRAPPED"] = True


def health_check() -> Dict[str, Any]:
    chroma_dir = os.getenv("CHROMA_DIR", "chromadb")
    collection = os.getenv("CHROMA_COLLECTION", "rag_docs")

    llm = _llm_info_from_env()
    emb = _embed_info_from_env()

    info: Dict[str, Any] = {
        "chroma_dir": chroma_dir,
        "collection": collection,
        "embed_backend": emb["backend"],
        "embed_model": emb["model"],
        "llm_backend": llm["backend"],
        "llm_model": llm["model"],
        "chroma_dir_exists": Path(chroma_dir).exists(),
        "vectorstore_ok": False,
        "vector_count": None,
        "error": None,
    }

    try:
        from rag.retriever import get_vectorstore

        # ✅ OpenAI-only signature: get_vectorstore(..., embed_model=...)
        vs = get_vectorstore(
            persist_directory=chroma_dir,
            collection_name=collection,
            embed_model=emb["model"],
        )

        try:
            info["vector_count"] = vs._collection.count()
        except Exception:
            info["vector_count"] = None

        info["vectorstore_ok"] = True
    except Exception as e:
        info["vectorstore_ok"] = False
        info["error"] = f"{type(e).__name__}: {repr(e)}"

    return info

def highlight_terms(text: str, query: str) -> str:
    terms = sorted({t for t in re.findall(r"[A-Za-z0-9]{3,}", (query or "").lower())}, key=len, reverse=True)
    safe = html.escape(text or "")
    for t in terms[:12]:
        safe = re.sub(rf"(?i)\b({re.escape(t)})\b", r"<mark>\1</mark>", safe)
    return safe


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
        return f"Explain {q} in the context of the uploaded documents."
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


# Guards
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
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(p in t for p in ["hi", "hello", "hey", "yo", "sup", "how are you", "who are you", "thank you", "thanks", "thx"]) and len(t.split()) <= 8


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
    t = (text or "").strip().lower()
    if not t:
        return False

    question_words = ["what", "how", "why", "when", "where", "who", "which", "explain", "tell me", "describe", "define"]
    is_question = any(t.startswith(qw) for qw in question_words) or "?" in t

    banking_keywords = [
        "cet1", "rwa", "capital", "buffer", "basel", "pra", "pillar", "banking", "regulation",
        "requirement", "ratio", "ccyb", "ccb", "ccob", "conservation", "countercyclical", "risk", "asset",
        "supervisory", "framework", "methodology", "reform",
    ]
    has_banking_keywords = any(kw in t for kw in banking_keywords)

    if has_banking_keywords or is_question:
        return False

    casual_patterns = [
        r"^(i|my|i'm|i am|my name is|i like|i love|i hate|i think|i feel|i want|i need|i have|i do|i did|i was|i will)",
        r"^(this is|that is|here is|there is|it is|it's)",
        r"^(nice|good|bad|great|awesome|cool|hello|hi|hey)$",
        r"^(i am|i'm) [a-z]+$",
    ]
    return any(re.match(pattern, t) for pattern in casual_patterns)


def smalltalk_reply(text: str) -> str:
    t = (text or "").strip().lower()
    if "who are you" in t:
        return "I’m your RAG assistant — I answer using only the uploaded documents and cite the exact places used."
    if "how are you" in t:
        return "Doing good 😄 Ask me something from your documents and I’ll answer with sources."
    if "thank" in t or "thx" in t:
        return "Anytime. Want to test something harder from the documents?"
    return "Hey! 👋 Ask me anything about your uploaded documents."


def injection_reply() -> str:
    return (
        "I can’t help with attempts to override instructions or reveal system prompts.\n\n"
        "Ask a normal question about the PDFs and I’ll answer with citations."
    )


# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Systems online. Ask a question and I’ll answer using your documents with citations.",
            "ts": now_hhmm(),
            "sources": [],
        }
    ]

st.session_state.setdefault("last_sources", [])
st.session_state.setdefault("pending_prompt", None)
st.session_state.setdefault("is_processing", False)


# ---------------- Sidebar ----------------
with st.sidebar:
    is_locked = st.session_state.get("is_processing", False)

    st.markdown("### 🛠️ Configuration")

    pdf_names = list_pdf_filenames()
    doc_choice = st.selectbox(
        "Document Context",
        options=["All"] + pdf_names,
        index=0,
        disabled=is_locked,
    )

    strict_mode = st.toggle(
        "Strict Analysis",
        value=False,
        disabled=is_locked,
        help="Enable to prevent the AI from using general knowledge outside of the PDFs.",
    )

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True, disabled=is_locked):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🖥️ System Details")

    llm = _llm_info_from_env()
    emb = _embed_info_from_env()
    st.caption(f"**LLM backend:** `{llm['backend']}`")
    st.caption(f"**LLM model:** `{llm['model']}`")
    st.caption(f"**Embeddings:** `{emb['backend']}` / `{emb['model']}`")
    st.caption(f"**Collection:** `{os.getenv('CHROMA_COLLECTION', 'rag_docs')}`")

    with st.expander("System check", expanded=False):
        hc = health_check()
        st.write("Chroma dir:", f"`{hc['chroma_dir']}`")
        st.write("Collection:", f"`{hc['collection']}`")
        st.write("Embeddings backend:", f"`{hc['embed_backend']}`")
        st.write("Embeddings model:", f"`{hc['embed_model']}`")
        st.write("LLM backend:", f"`{hc['llm_backend']}`")
        st.write("LLM model:", f"`{hc['llm_model']}`")
        st.write("DB folder exists:", "✅" if hc["chroma_dir_exists"] else "❌")
        if hc.get("vectorstore_ok"):
            st.write("Vectorstore:", "✅ loaded")
            if hc.get("vector_count") is not None:
                st.write("Vectors indexed:", hc["vector_count"])
        else:
            st.write("Vectorstore:", "❌ error")
            st.caption(hc.get("error", "Unknown error"))


# ---------------- Main UI ----------------
hc_top = health_check()
db_ok = bool(hc_top.get("vectorstore_ok")) and bool(hc_top.get("chroma_dir_exists"))
llm = _llm_info_from_env()

st.markdown(
    f"""
<div class="hero-section">
    <div class="hero-title">RAG Knowledge Navigator</div>
    <div style="color: var(--text-muted); font-size: 15px; margin-bottom: 20px;">
        Grounded answers from your PDF knowledge base — with citations and safety guards.
    </div>
    <div style="display: flex; gap: 12px; flex-wrap: wrap;">
        <span class="status-badge">📡 Vector Store {"Connected" if db_ok else "Check"}</span>
        <span class="status-badge">🧠 LLM: {html.escape(llm["backend"])} / {html.escape(llm["model"])}</span>
        <span class="status-badge">📄 Docs: {"All" if doc_choice == "All" else html.escape(doc_choice)}</span>
        <span class="status-badge">🔒 Strict: {"On" if strict_mode else "Off"}</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- Render chat messages ----------------
for msg in st.session_state["messages"]:
    is_user = msg.get("role") == "user"
    row_class = "msg-right" if is_user else "msg-left"
    bubble_class = "bubble-user" if is_user else "bubble-assistant"

    avatar = "🧑‍💻" if is_user else "🤖"

    content = (msg.get("content", "") or "").replace("\n", "<br>")
    ts = msg.get("ts", "")
    sources = msg.get("sources", []) or []

    sources_html = ""
    if sources:
        chips = "".join([f'<span class="source-pill">{html.escape(s)}</span>' for s in sources[:6]])
        sources_html = f'<div style="margin-top:6px;">{chips}</div>'

    html_string = (
        f'<div class="msg-row {row_class}">'
        f'<div class="avatar">{avatar}</div>'
        f'<div class="bubble {bubble_class}">'
        f'<div>{content}</div>{sources_html}'
        f'<div class="time-stamp">{html.escape(ts)}</div>'
        f"</div></div>"
    )

    st.markdown(html_string, unsafe_allow_html=True)

st.markdown("<div id='chat-bottom'></div>", unsafe_allow_html=True)
autoscroll_to_anchor()


# ---------------- Fixed composer ----------------
st.markdown('<div class="fixed-composer">', unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    cols = st.columns([9, 1])
    user_input = cols[0].text_input(
        "Ask anything",
        placeholder="Ask a question about your PDFs…",
        label_visibility="collapsed",
        disabled=st.session_state.get("is_processing", False),
    )
    submitted = cols[1].form_submit_button("⏎", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Two-step submit ----------------
if submitted and user_input.strip():
    st.session_state["messages"].append(
        {"role": "user", "content": user_input, "ts": now_hhmm(), "sources": []}
    )
    st.session_state["pending_prompt"] = user_input
    st.session_state["is_processing"] = True
    st.rerun()


pending: Optional[str] = st.session_state.get("pending_prompt")
if pending:
    st.session_state["pending_prompt"] = None
    st.session_state["is_processing"] = True

    if is_smalltalk(pending):
        st.session_state["is_processing"] = False
        st.session_state["last_sources"] = []
        st.session_state["messages"].append(
            {"role": "assistant", "content": smalltalk_reply(pending), "ts": now_hhmm(), "sources": []}
        )
        st.rerun()

    if is_injection(pending):
        st.session_state["is_processing"] = False
        st.session_state["last_sources"] = []
        st.session_state["messages"].append(
            {"role": "assistant", "content": injection_reply(), "ts": now_hhmm(), "sources": []}
        )
        st.rerun()

    if is_garbage_input(pending):
        st.session_state["is_processing"] = False
        st.session_state["last_sources"] = []
        st.session_state["messages"].append(
            {"role": "assistant", "content": "Type a real question (not just punctuation 🙂).", "ts": now_hhmm(), "sources": []}
        )
        st.rerun()

    if is_casual_statement(pending):
        st.session_state["is_processing"] = False
        st.session_state["last_sources"] = []
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": "Nice to meet you! 👋 Ask me a question about your uploaded PDFs and I’ll answer with sources.",
                "ts": now_hhmm(),
                "sources": [],
            }
        )
        st.rerun()

    try:
        with st.spinner("Synthesizing response..."):
            memory = build_memory_hint(st.session_state["messages"], max_user_turns=2)
            base_query = normalize_query(pending)
            query = f"{memory}\n\nCurrent question:\n{base_query}" if memory else base_query

            out: Dict[str, Any] = answer_question(
                query,
                k=None,
                llm_model=None,
                source_filename=None if doc_choice == "All" else doc_choice,
                strict=strict_mode,
                user_query=pending,
            )

        st.session_state["is_processing"] = False

        full = out.get("answer", "") or ""
        content = answer_without_sources(full)
        sources = extract_sources(full)
        if content.strip() == "Not found in the provided documents.":
            sources = []

        st.session_state["last_sources"] = sources
        st.session_state["messages"].append(
            {"role": "assistant", "content": content, "ts": now_hhmm(), "sources": sources}
        )
        st.rerun()

    except Exception as e:
        st.session_state["is_processing"] = False
        st.session_state["last_sources"] = []
        tb = traceback.format_exc()

        error_msg = (
            "I encountered an error while processing your question.\n\n"
            f"Type: {type(e).__name__}\n"
            f"Repr: {repr(e)}\n\n"
            f"Traceback:\n{tb}"
        )

        st.session_state["messages"].append(
            {"role": "assistant", "content": error_msg, "ts": now_hhmm(), "sources": []}
        )
        st.rerun()

st.markdown(
    "<br><p style='text-align:center; color:gray; font-size:12px;'>Project #4 • RAG • Grounded answers with citations</p>",
    unsafe_allow_html=True,
)
