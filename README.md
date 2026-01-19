🧠 RAG Knowledge Navigator

Grounded Question Answering over Basel III & Prudential Regulation Documents

Overview

RAG Knowledge Navigator is a Retrieval-Augmented Generation (RAG) application designed to answer complex regulatory and banking questions strictly grounded in official supervisory documents.

Unlike generic chatbots, this system:

Retrieves verifiable passages from regulatory PDFs

Generates answers only from retrieved evidence

Explicitly returns “Not found in the provided documents” when information is missing

Displays clear citations (document + page) for every answer

The project demonstrates how modern LLMs can be safely applied in high-risk domains such as banking regulation, where hallucinations are unacceptable.

Why this project exists

Banking regulation (Basel III, Pillar 2, capital buffers) is:

Dense

Fragmented across long PDFs

Hard to query precisely

This project turns static regulatory documents into an interactive, auditable knowledge system — something that could realistically be used by:

Risk analysts

Compliance teams

Regulatory reporting units

Documents Ingested

The knowledge base was built exclusively from the following official regulatory sources:

Basel III – A Global Regulatory Framework for More Resilient Banks

Basel III – Finalising Post-Crisis Reforms (2017)

Basel III Capital Buffers – Conservation & Countercyclical Buffers

PRA Supervisory Statement – Pillar 2 Capital Methodology

No external knowledge, websites, or pretrained facts are used during answering.

How it works (Architecture)

1. Ingestion Pipeline

PDFs are loaded and cleaned

Documents are split into semantic chunks

Each chunk is embedded and stored in ChromaDB

Metadata includes document name and page number

2. Retrieval

User query is optionally expanded (e.g. CCB → Capital Conservation Buffer)

Top-K relevant chunks are retrieved

Optional reranking improves relevance

Confidence gates reject weak matches

3. Answer Generation

The LLM receives:

The user question

Only the retrieved document context

Strict system rules enforce:

No external knowledge

Exact fallback message if answer is missing

4. Citation Layer

Final answers include a structured Sources section

Each citation maps directly to the ingested PDFs

Key Features
✅ Grounded Answers Only

If the documents do not contain the answer, the system responds:

“Not found in the provided documents.”

This is enforced both before and after generation.

🧾 Explicit Citations

Every valid answer includes:

Document name

Page reference

This makes responses auditable and defensible.

🔒 Strict Mode

When enabled:

Keyword overlap thresholds are enforced

Low-confidence retrievals are rejected

Prevents “reasonable-sounding” hallucinations

🧠 Short-Term Context Awareness

Follow-up questions such as:

“How does it differ from the countercyclical one?”

are rewritten internally using the previous user question, without fabricating context.

🛡️ Prompt Injection Guards

The UI actively detects and blocks:

Instruction overrides

Prompt leakage attempts

Jailbreak patterns

💬 Modern Chat UI

Streamlit-based interface

Fixed chat composer

Message bubbles with timestamps

Source pills per answer

Built for clarity, not demos.

Tech Stack

Python

LangChain

ChromaDB

OpenAI / Local LLM support

Streamlit

dotenv

The system is modular and backend-agnostic (LLM or embeddings can be swapped via .env).

Project Structure (High-Level)
rag/
├── ingest.py        # PDF loading, chunking, embedding
├── retriever.py     # Vector search + scoring
├── answer.py        # RAG logic, confidence gates, citations
app/
├── streamlit_app.py # UI and interaction layer
data/
├── raw/             # Original PDFs
chromadb/            # Persistent vector store

What this project demonstrates

Practical RAG design (not a toy example)

Safety-first LLM usage

Regulatory-grade answer grounding

Clear separation between retrieval, reasoning, and UI

Real-world document complexity handling

 Next Steps

Future improvements:

Better follow-up rewriting

Section-aware retrieval

Answer confidence scoring

Exportable audit logs

Final Note

This project is intentionally boring in the right way:

No flashy claims

No hallucinations

No fake intelligence

Just correct answers, or honest uncertainty — which is exactly what high-stakes domains require.