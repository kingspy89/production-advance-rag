# 🎯 RESUME DESCRIPTIONS & INTERVIEW PREPARATION — INFRAAGENT

## 📋 ONE-LINE SUMMARY
Stateful agentic RAG system for enterprise IT documentation (Kubernetes, Intel hardware, networking) featuring two-layer safety guardrails, cross-encoder semantic reranking, dual embedding support, and automated RAGAS evaluation.

---

## 🏆 OPTION 1: TECHNICAL DEPTH (ML / AI Infrastructure Roles)

**Enterprise Agentic RAG System (InfraAgent)** | Python, LangGraph, FastAPI, Qdrant, RAGAS  
*AI assistant for Kubernetes operations, Intel hardware acceleration, and enterprise networking*

• Built an **agentic RAG workflow** using LangGraph featuring a 3-node state machine (Planner → Retriever → Responder) and thread-level memory checkpointing (`MemorySaver`) to maintain multi-turn conversational context.

• Implemented a **two-stage retrieval pipeline** pairing Qdrant Cloud vector search (15 candidates) with local FlashRank ONNX cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`), reducing context chunks by **67%** (15→5) and improving RAGAS Context Precision from **0.59 to 0.78** (+0.19 absolute gain).

• Architected a **two-layer hybrid guardrail system** combining sub-millisecond deterministic regex pre-filters for jailbreaks, greetings, and technical patterns with NeMo Guardrails Colang flows; engineered fail-open error handling to prevent blocking engineers during external API timeouts.

• Integrated **dual embedding support** (Google Gemini 3,072-dim and local SentenceTransformers 768-dim) with runtime collection dimension validation in `qdrant_service.py` to eliminate silent vector dimension mismatches.

• Built an **automated evaluation platform** (RAGAS + Streamlit) with rate-limit cooldown buffers and dynamic upstream compatibility shims for LangChain 0.4+, measuring **0.89 Faithfulness**, **0.91 Answer Relevancy**, and **0.87 Answer Correctness** on a 21-case golden dataset.

• Added **production engineering features**: multi-dependency `/health` diagnostic probe, headless non-interactive Logfire telemetry fallback, and comprehensive `.env.example` configuration templates.

**Technical Stack:** Python 3.10+, FastAPI, LangGraph, Qdrant Cloud, FlashRank, NeMo Guardrails, Groq Cloud, Streamlit, Pydantic Logfire, LangSmith, RAGAS  

---

## 🚀 OPTION 2: IMPACT-FOCUSED (Full-Stack ML / Product Roles)

**Enterprise Documentation Assistant (InfraAgent)** | Full-Stack ML Engineer  
*Agentic technical documentation assistant with automated grounding evaluation*

• **Optimized retrieval context quality** by adding FlashRank cross-encoder reranking to vector search, reducing context chunk volume by **67%** (15→5 chunks) while raising RAGAS Context Precision from **0.59 to 0.78**.

• **Eliminated silent data corruption** in Qdrant by implementing runtime schema validation asserting `query_dim == collection_dim` prior to vector search execution.

• **Engineered layered input protection** using deterministic regex fast-paths and NeMo Colang rails, correctly classifying 6 golden guardrail scenarios (3/3 adversarial blocked, 3/3 technical allowed).

• **Automated quality benchmarking** by deploying a dedicated Streamlit evaluation dashboard running 6 RAGAS metric experiments with calibrated cooldown buffers to stay within judge LLM token limits.

• **Delivered full operational visibility** by adding a dependency health probe monitoring Qdrant vector database, Groq LLM, and guardrail engine status.

**Technologies:** Python, FastAPI, LangGraph, Qdrant Cloud, FlashRank, NeMo Guardrails, Groq, Streamlit, Pydantic Logfire, RAGAS  

---

## 💼 OPTION 3: CONCISE (CV Bullet Format)

**Enterprise Agentic RAG System (InfraAgent)** | Python, LangGraph, FastAPI, Qdrant, RAGAS

• Built a stateful agentic RAG system with **two-layer safety guardrails** (regex pre-filtering + NeMo Colang flows) and thread-level conversational memory checkpointing.

• Designed a **two-stage retrieval pipeline** combining Qdrant vector search (15 chunks) with local FlashRank cross-encoder reranking (top 5 chunks), achieving a **67% context reduction** and improving Context Precision from **0.59 to 0.78**.

• Hardened vector retrieval with **runtime dimension validation**, preventing silent search failures when switching between Gemini (3,072D) and local (768D) embedding models.

• Built an **automated evaluation harness** using RAGAS, achieving **0.89 Faithfulness** and **0.91 Answer Relevancy** on an offline 21-case golden benchmark.

---

## 🎤 OPTION 4: ELEVATOR PITCH (30 seconds)

"I built InfraAgent, an agentic RAG assistant for enterprise Kubernetes and Intel hardware documentation. It uses a LangGraph state machine with conversational memory, a two-stage retrieval pipeline with FlashRank cross-encoder reranking that trims context chunks by 67% and improves precision by 32%, and a two-layer guardrail system that fails open during API interruptions. It features runtime vector dimension validation to prevent data corruption and a complete automated evaluation dashboard scoring RAGAS metrics."

---

## 📊 VERIFIED BENCHMARKS (Golden Dataset Evaluation)

| Metric | Measured Value | Meaning |
|---|:---:|---|
| **Context Chunk Reduction** | **67%** (15 → 5) | Context compression via FlashRank cross-encoder |
| **Context Precision** | **0.78** (vs 0.59 baseline) | +0.19 gain in signal-to-noise ratio of retrieved context |
| **Faithfulness** | **0.89** | Factual grounding in retrieved source documents |
| **Answer Relevancy** | **0.91** | Response directness and query intent alignment |
| **Answer Correctness** | **0.87** | Semantic alignment with reference ground truth |
| **Tool Selection Correctness**| **1.00** | Accurate routing to `retrieve_documents` / `direct_answer` |
| **Guardrails Test Pass Rate** | **6/6 passed** | 3/3 adversarial blocked, 3/3 legitimate queries passed |

*Evaluation setup: 21 golden test cases (15 RAG domain questions + 6 guardrail test cases), evaluated using Groq's `llama-3.1-8b-instant` as judge LLM and `all-MiniLM-L6-v2` evaluation embeddings.*

---

## 🗣️ INTERVIEW TALKING POINTS

### "Tell me about a technical challenge you diagnosed and resolved"

**The Silent Vector Dimension Mismatch:**
"Qdrant accepts vectors of differing dimensions without raising database errors, which silently corrupts search results when switching between Gemini (3072D) and local (768D) embeddings. There is no runtime error—just nonsensical search results.

I diagnosed this by inspecting the collection schema and implemented runtime validation in `qdrant_service.py` that verifies `query_dim == collection_dim` before executing queries. If mismatched, it halts immediately with a clear error: 'Embedding dimension mismatch: query=768D, collection=3072D. Re-ingest data.' This ensures data integrity by forcing an explicit re-index when changing embedding providers."

---

### "Why did you implement cross-encoder reranking?"

**Context Compression & Precision Optimization:**
"Standard bi-encoder vector search evaluates query and document vectors independently via dot product or cosine distance. This is computationally fast for initial candidate retrieval, but misses nuanced cross-attention between question terms and technical context.

I implemented FlashRank, an ONNX-quantized cross-encoder (`ms-marco-MiniLM-L-6-v2`), running locally on CPU in ~12ms. We retrieve 15 candidate chunks from Qdrant and rerank them down to the top 5 most relevant chunks. This reduced the prompt context chunk count by 67% and raised RAGAS Context Precision from 0.59 to 0.78."

---

### "How do you handle guardrail reliability during system failures?"

**Fail-Open Operational Resilience:**
"The guardrail architecture has two tiers: Layer 1 runs deterministic regex checks (<1ms) for known injection patterns and greetings without API cost. Layer 2 routes ambiguous queries to NeMo Guardrails Colang flows backed by an LLM.

In enterprise IT infrastructure, blocking engineers during an active incident because an external safety API timed out is unacceptable. If the LLM guardrail service times out or hits rate limits, the guardrail layer is intentionally engineered to fail open for technical queries, logging the exception to Logfire while allowing the LangGraph agent to answer the operator's query."

---

### "Why LangGraph instead of simple sequential chains?"

**Stateful Decision Graphs:**
"Sequential chains force linear execution: every query gets embedded, searched, and generated even if the user simply asked a follow-up or said 'hello'. 

LangGraph allows us to define an explicit state graph with a Planner node. The Planner inspects conversation history and classifies whether the turn is conversational or technical. Conversational turns route straight to the Responder, skipping vector retrieval entirely. Furthermore, LangGraph's `MemorySaver` provides thread-level state checkpointing across turns."
