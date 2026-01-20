# 🤖 RAG Knowledge Navigator

**Project #4 — Retrieval-Augmented Generation (RAG) Chatbot**

A production-oriented **Retrieval-Augmented Generation (RAG)** chatbot that answers questions **strictly grounded in regulatory PDF documents**, with **citations, guardrails, reranking, metadata filtering, and short-term conversational memory**.

This project demonstrates how Large Language Models (LLMs) can be safely used in **high-risk, regulated domains** where hallucinations and vague answers are unacceptable.

---

## 🎯 Project Goal

The goal of this project is to build a chatbot that:

* Retrieves the **most relevant document passages**
* Generates **accurate, grounded answers**
* Shows **explicit citations**
* **Refuses to answer** when the information is not present
* Handles **follow-up questions** without hallucination

The system is designed to behave like a **regulatory knowledge assistant**, not a general-purpose chatbot.

---

## 📄 Knowledge Base: Document Scope & Meaning

This project is built around a **curated set of authoritative regulatory PDF documents** related to **banking capital adequacy and prudential regulation**, primarily under the **Basel III framework**.

These documents are **dense, technical, and rule-driven**, making them ideal for evaluating a Retrieval-Augmented Generation system.

### 🏦 Why These PDFs Were Chosen

The PDFs:

* Are **official regulatory texts**, not summaries or blogs
* Contain **precise definitions, thresholds, and conditions**
* Require **exact wording and context**
* Penalize hallucination or speculative answers

This makes them a **high-risk knowledge domain**, where incorrect answers would be unacceptable in real-world usage.

### 📘 What the Documents Cover

The documents include detailed explanations of:

* Capital Conservation Buffer (CCB)
* Countercyclical Capital Buffer (CCyB)
* Common Equity Tier 1 (CET1) capital
* Risk-Weighted Assets (RWA)
* Buffer activation and release mechanisms
* Supervisory intent and macroprudential regulation

Example documents used:

* *Basel III – Capital Buffers: Conservation & Countercyclical*
* *Basel III – A Global Regulatory Framework for More Resilient Banks*

### 🧠 Why This Matters for RAG

These PDFs are particularly suitable for RAG because:

* Answers must be **directly supported by retrieved text**
* Many questions are **contextual follow-ups** (“how does it differ?”, “when is it released?”)
* The system must correctly handle **short-term conversational memory**
* The assistant must **refuse answers** when evidence is missing

This dataset acts as a **stress test** for retrieval accuracy, grounding, and safety.

---

## 🧩 Core Functionality

### ✅ Document Ingestion Pipeline

* PDF loading
* Chunking with overlap
* Embedding generation
* Vector storage using **Chroma**

**Flow:** PDF → chunks → embeddings → vector database

---

### ✅ Semantic Retrieval (Top-K)

* Vector similarity search
* Deduplication by document, page, and chunk
* Optional document-level filtering

---

### ✅ Answer Generation with Citations

* LLM answers using **retrieved context only**
* Automatic refusal when information is missing
* Clear citations displayed per answer

---

### ✅ Short-Term Conversation Memory (Grounded)

* Follow-up questions rewritten using previous user context
* Improves retrieval without introducing hallucinations
* Example supported queries:

  * “How does it differ from the other one?”
  * “When is it released?”
  * “Compare them”

---

### ✅ Guardrails & Safety

**Prompt-Injection Protection**

* Detects attempts such as:
  * “ignore previous instructions”
  * “print system prompt”
  * “DAN”
* Safe refusal response

**Document-Only Enforcement**

* Strict mode prevents use of outside knowledge
* Returns *“Not found in the provided documents”* when needed

---

## 🎛️ User Interface

Built with **Streamlit**, featuring:

* Chat-style UI
* Document selector (metadata filter)
* Strict mode toggle
* System health panel
* Vector count visibility
* Backend & model transparency

---

## ⭐ Nice-to-Have Features (Implemented vs Not Implemented)

### ✅ Implemented

| Feature | Status |
| :--- | :--- |
| Reranking | ✅ Implemented |
| Metadata filtering (by document) | ✅ Implemented |
| Short-term conversation memory | ✅ Implemented |
| Guardrails (prompt injection + grounding) | ✅ Implemented |

---

### ❌ Not Implemented (By Design)

| Feature | Reason |
| :--- | :--- |
| Hybrid search (BM25 + vector) | Project focused on semantic retrieval |
| Observability dashboards | Out of scope for current evaluation |

These omissions are **intentional and documented**, not oversights.

---

## 🧪 Example Behaviors

✔️ Correct follow-up handling  
✔️ Context-aware comparisons  
✔️ Citations per answer  
✔️ Safe refusals when content is missing  

❌ No hallucinations  
❌ No outside knowledge injection  

---

## 🧠 Technical Stack

* **Python**
* **LangChain**
* **ChromaDB**
* **OpenAI embeddings & LLM**
* **Streamlit**

---

## 📌 Project Summary

This project demonstrates a **production-grade RAG system** designed for **regulated, high-risk domains**, where correctness, grounding, and explainability matter more than creativity.

By grounding every answer in authoritative regulatory documents, the system shows how LLMs can be safely integrated into compliance-sensitive workflows.

---

## 🚀 How to Run

```bash
# 1. Ingest documents
python rag/ingest.py

# 2. Start the app
streamlit run app/streamlit_app.py

📄 License
Academic / educational use.