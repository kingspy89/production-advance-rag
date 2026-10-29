<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=6C63FF&center=true&vCenter=true&width=700&lines=InfraAgent+%E2%80%94+Enterprise+Agentic+RAG;Production+LangGraph+%2B+Qdrant+%2B+NeMo;Built+for+Infrastructure+Engineers" alt="InfraAgent Typing SVG" />

<br/>

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/LangGraph-Agentic_Workflow-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
<img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC244C?style=for-the-badge&logo=qdrant&logoColor=white" />
<img src="https://img.shields.io/badge/NeMo-Safety_Guardrails-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
<img src="https://img.shields.io/badge/RAGAS-Evaluation-8A2BE2?style=for-the-badge" />
<img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" />

<br/><br/>

<img src="https://img.shields.io/badge/Faithfulness-0.89-brightgreen?style=flat-square" />
<img src="https://img.shields.io/badge/Answer_Relevancy-0.91-brightgreen?style=flat-square" />
<img src="https://img.shields.io/badge/Context_Precision-0.78-yellow?style=flat-square" />
<img src="https://img.shields.io/badge/Answer_Correctness-0.87-brightgreen?style=flat-square" />
<img src="https://img.shields.io/badge/Tool_Correctness-1.00-brightgreen?style=flat-square" />

<br/><br/>

<a href="https://github.com/kingspy89/production-advance-rag/stargazers"><img src="https://img.shields.io/github/stars/kingspy89/production-advance-rag?style=social" /></a>
<a href="https://github.com/kingspy89/production-advance-rag/forks"><img src="https://img.shields.io/github/forks/kingspy89/production-advance-rag?style=social" /></a>
<a href="https://github.com/kingspy89/production-advance-rag/issues"><img src="https://img.shields.io/github/issues/kingspy89/production-advance-rag?color=red&style=flat-square" /></a>
<a href="https://github.com/kingspy89/production-advance-rag/blob/main/LICENSE"><img src="https://img.shields.io/github/license/kingspy89/production-advance-rag?style=flat-square" /></a>

</div>

---

<div align="center">

## 🤖 What is InfraAgent?

</div>

**InfraAgent** is a production-grade **Enterprise Agentic RAG** (Retrieval-Augmented Generation) system built for complex IT infrastructure domains — **Kubernetes cluster operations**, **Intel hardware acceleration** (Xeon, FPGAs, SR-IOV, NUMA), and **enterprise networking** (BGP, SDN, VLANs, TCP/IP).

> Standard LLMs hallucinate CLI flags, miss private documentation, and lack operational context. Naive RAG floods prompts with noisy, unranked chunks. InfraAgent solves both problems: a **LangGraph state machine** orchestrates intent-aware retrieval, **FlashRank cross-encoder reranking** surgically compresses context from 15 candidates to 5, and a **two-layer NeMo + Regex safety system** filters unsafe inputs before they ever reach the LLM.

---

## ✨ Key Capabilities

<table>
<tr>
<td width="50%">

### 🧠 Stateful Agentic Orchestration
LangGraph state machine with a **Planner → Router → Retriever/Responder** pipeline. Conversation history is persisted across turns via thread-level `MemorySaver` checkpointing. The planner classifies intent and rewrites queries before retrieval.

</td>
<td width="50%">

### 🎯 Coarse-to-Fine Retrieval
**15 candidate chunks** retrieved from Qdrant Cloud via cosine similarity → **5 final chunks** selected by FlashRank ONNX cross-encoder (`ms-marco-MiniLM-L-6-v2`) — a **67% reduction in candidate count** before generation.

</td>
</tr>
<tr>
<td width="50%">

### 🛡️ Two-Layer Hybrid Guardrails
**Layer 1** — sub-millisecond deterministic regex pre-filter for injections, jailbreaks, greetings, and technical keywords (typo-tolerant). **Layer 2** — LLM-backed NeMo Guardrails Colang flow for ambiguous inputs. Fail-open on API error to never block engineers during incidents.

</td>
<td width="50%">

### 📄 Multi-Format Document Ingestion
Ingests **PDF** (`pypdf` + `pdfplumber` fallback), **HTML** (`beautifulsoup4`), **Word** (`python-docx`), **PowerPoint** (`python-pptx`), and plain text — all with 1,500-character paragraph-aware chunking and automatic Qdrant upsert.

</td>
</tr>
<tr>
<td width="50%">

### 🔢 Dual Embedding + Runtime Validation
**Gemini** (`gemini-embedding-2-preview`, 3,072-dim) or **local SentenceTransformers** (`all-mpnet-base-v2`, 768-dim). Query vector dimensions are validated against the Qdrant collection schema at runtime — no silent mismatch failures.

</td>
<td width="50%">

### 📊 Automated RAGAS Evaluation Suite
Dedicated Streamlit eval dashboard (`evals/app.py`) running **6 RAGAS metrics** against a **21-case golden dataset** (15 RAG + 6 guardrail scenarios). Judge model: `llama-3.1-8b-instant` via a separate Groq API key.

</td>
</tr>
<tr>
<td>

### 🔭 Distributed Observability
**Pydantic Logfire** for structured request/retrieval tracing (headless-offline fallback enabled). **LangSmith** for state graph execution inspection. Operational `/health` dependency probe.

</td>
<td>

### 🔀 Provider Routing & Fallback
Portkey Gateway sits between the agent and Groq Cloud for provider-level routing and fallback. Active model: **`qwen/qwen3.8-27b`** (configurable via `GROQ_MODEL` env var).

</td>
</tr>
</table>

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    User([👤 User / Engineer]) --> UI["🖥️ Streamlit Chat Interface\n(:8501)"]
    UI --> API["⚡ FastAPI REST Server\n(:8000 /query)"]

    subgraph Safety_Layer ["🛡️ Two-Layer Guardrails"]
        API --> L1{"Layer 1\nRegex Filter\n⚡ sub-ms"}
        L1 -- "🚫 Disallowed / Greeting" --> FastResp["Direct Response"]
        L1 -- "❓ Ambiguous" --> L2["Layer 2\nNeMo Guardrails (Colang)"]
        L2 -- "🔴 Blocked" --> FastResp
    end

    L1 -- "✅ Technical Match" --> Agent
    L2 -- "✅ Pass" --> Agent
    L2 -. "⚠️ Error: fail-open bypass" .-> Agent

    subgraph Agent ["🤖 LangGraph Agent Engine"]
        Planner["🧠 Planner Node\n(Intent Classification & Query Refinement)"]
        Planner --> RouteDecision{"Routing\nDecision"}
        RouteDecision -- "💬 Conversational" --> Responder["📝 Responder Node"]
        RouteDecision -- "🔍 Technical" --> Retriever["📚 Retriever Node"]

        Retriever --> Qdrant[("🗄️ Qdrant Cloud DB\nTop 15 Chunks")]
        Qdrant --> FlashRank["⚡ FlashRank Reranker\nONNX Cross-Encoder → Top 5"]
        FlashRank --> Responder

        Checkpointer[("💾 MemorySaver\nThread Session State")] <--> Agent
    end

    subgraph Provider_Layer ["☁️ LLM Provider & Gateway"]
        Responder --> Provider{"Portkey\nModel Gateway"}
        Provider --> Groq["🚀 Groq Cloud\nqwen/qwen3.8-27b"]
    end

    Groq --> FinalResponse(["✨ Synthesized Answer\n+ Thought Process + Sources"])
    FinalResponse --> UI

    subgraph Observability_Evals ["📡 Telemetry & Evaluation"]
        EvalApp["📊 Streamlit Eval Dashboard\n(:8502)"] -.-> API
        API -.-> Logfire["🔥 Pydantic Logfire Spans"]
        Agent -.-> LangSmith["🦜 LangSmith Graph Tracing"]
    end
```

---

## 📊 Evaluation & Benchmark Results

> All scores are **offline local evaluation results** recorded against the golden dataset. They are not production-observed metrics.

### 🎯 RAGAS Metric Scores

Scored against **15 technical RAG questions** (IDs 1–15 in `evals/golden_dataset.json`). Judge LLM: `llama-3.1-8b-instant` (Groq). Evaluation embeddings: `sentence-transformers/all-MiniLM-L6-v2`. Batch size: 1, with 40s inter-sample and 62s inter-experiment cooldowns to stay within Groq's TPM limits. Results are captured in the Streamlit eval dashboard session — not auto-persisted to disk.

<div align="center">

| Metric | Score | Visual | What It Measures |
|--------|:-----:|--------|-----------------|
| 🔵 **Faithfulness** | **`0.89`** | `████████░` | Factual grounding — answers derived strictly from retrieved docs |
| 🟢 **Answer Relevancy** | **`0.91`** | `█████████░` | How directly the answer satisfies the query without digressions |
| 🟡 **Context Precision** | **`0.78`** | `███████░░` | Ground-truth context ranked at top (*+0.19 gain over raw cosine baseline of 0.59*) |
| 🔵 **Context Recall** | **`0.84`** | `████████░` | All necessary reference facts captured in retrieved chunks |
| 🟢 **Answer Correctness** | **`0.87`** | `████████░` | Semantic + factual alignment with verified reference answers |
| ✅ **Tool Correctness** | **`1.00`** | `██████████` | Planner routes to correct tool on all 15 samples (Jaccard similarity) |

</div>

### 🛡️ Guardrails Classification Results

Evaluated via `evals/guardrails_eval.py` across 6 golden safety scenarios:

| Test Set | Samples | Expected | Result | Status |
|----------|:-------:|----------|--------|:------:|
| Adversarial Injections / Jailbreaks | 3 | Block → Canned Refusal | 3 Blocked | ✅ Pass |
| Legitimate Technical Inquiries | 3 | Pass → RAG Pipeline | 3 Passed | ✅ Pass |

> ⚠️ **Scope note:** These 6 test cases validate correct classification on the curated golden safety scenarios. They do **not** establish general adversarial robustness across the full range of possible injection techniques or novel jailbreak strategies.

### ⚡ Context Reduction Benchmark

| Stage | Candidates | Engine | Effect |
|-------|:----------:|--------|--------|
| 🔍 Vector Search | **15 chunks** | Qdrant Cloud (Cosine Similarity) | Coarse retrieval pool |
| ⚡ Cross-Encoder Rerank | **5 chunks** | FlashRank `ms-marco-MiniLM-L-6-v2` | **67% reduction in candidate count** (15 → 5); token reduction is proportional to chunk sizes and is a secondary effect |

---

## 🛠️ Technology Stack

<div align="center">

<img src="https://skillicons.dev/icons?i=python,fastapi,docker,git" height="45" />
<br/>

</div>

<table>
<tr><td><b>Layer</b></td><td><b>Technology</b></td><td><b>Role</b></td></tr>
<tr><td>🐍 Runtime</td><td>Python 3.10+, Pydantic v2</td><td>Core language & data validation</td></tr>
<tr><td>⚡ API Backend</td><td>FastAPI, Uvicorn (ASGI)</td><td>REST endpoints, health probes, graph visualization</td></tr>
<tr><td>🤖 Agent Engine</td><td>LangGraph, LangChain Core</td><td>Stateful workflow graph, conditional routing, checkpointing</td></tr>
<tr><td>🗄️ Vector DB</td><td>Qdrant Cloud</td><td>Dense vector storage, cosine similarity search (<code>query_points</code> API)</td></tr>
<tr><td>⚡ Reranker</td><td>FlashRank (ONNX)</td><td>Local cross-encoder — no API call, quantized for speed</td></tr>
<tr><td>🛡️ Safety</td><td>NVIDIA NeMo Guardrails + Regex</td><td>Two-layer hybrid guardrail pipeline with fail-open semantics</td></tr>
<tr><td>🔢 Embeddings</td><td>Gemini <code>gemini-embedding-2-preview</code> / SentenceTransformers <code>all-mpnet-base-v2</code></td><td>Dual-provider with runtime dimension validation</td></tr>
<tr><td>🚀 LLM Provider</td><td>Groq Cloud (<code>qwen/qwen3.8-27b</code>), Portkey Gateway</td><td>Sub-second inference; Portkey handles routing & fallback. <code>llama-3.1-8b-instant</code> used as offline RAGAS judge only</td></tr>
<tr><td>📊 Evaluation</td><td>RAGAS, Pandas</td><td>6-metric automated evaluation harness</td></tr>
<tr><td>🔭 Observability</td><td>Pydantic Logfire, LangSmith</td><td>Structured traces, span-level graph inspection</td></tr>
<tr><td>🖥️ UI</td><td>Streamlit</td><td>Chat interface (:8501) + Evaluation dashboard (:8502)</td></tr>
<tr><td>🐳 Container</td><td>Docker</td><td>Single-container deployment manifest</td></tr>
</table>

---

## 📁 Project Structure

```text
production-advance-rag/
├── 📂 app/
│   ├── main.py                   # FastAPI entrypoint — /health, /graph, /query
│   ├── config.py                 # Pydantic Settings — env var schema & model routing
│   ├── 📂 agents/
│   │   ├── graph.py              # LangGraph StateGraph assembly, edges & checkpointing
│   │   ├── state.py              # TypedDict AgentState schema
│   │   └── 📂 nodes/            # Planner, Retriever, Responder nodes
│   ├── 📂 gateway/
│   │   └── client.py             # Portkey wrapper + direct Groq/OpenAI fallback
│   ├── 📂 guardrails/
│   │   ├── rails.py              # Regex engine, NeMo adapter & fail-open logic
│   │   └── colang_rules.py       # Colang policies & technical keyword allowlist
│   ├── 📂 ingestion/
│   │   ├── processor.py          # Batch ingestion CLI, chunking & Qdrant upsert
│   │   ├── 📂 chunking/         # Paragraph-aware 1500-char text splitter
│   │   └── 📂 loaders/          # PDF, HTML, DOCX, PPTX, TXT parsers
│   └── 📂 services/retrieval/
│       ├── embedding.py          # Dual-provider embedding manager (Gemini / Local)
│       ├── qdrant_service.py     # Qdrant search + dimension validation
│       └── ranking_service.py    # FlashRank local cross-encoder service
├── 📂 evals/
│   ├── app.py                    # Streamlit eval dashboard (:8502)
│   ├── pipeline.py               # Live API evaluation orchestrator
│   ├── metrics.py                # 6 RAGAS metric implementations
│   ├── guardrails_eval.py        # Binary classification harness
│   └── golden_dataset.json       # 21-sample ground-truth evaluation dataset
├── 📂 ui/
│   ├── app.py                    # Primary Streamlit chat interface (:8501)
│   └── st_cloud_ui.py            # In-process Streamlit deployment variant
├── .env.example                  # Environment configuration template
├── Dockerfile                    # Container build manifest
└── requirements.txt              # Pinned production dependencies
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Details |
|-------------|---------|
| 🐍 Python | 3.10 or higher |
| 🗄️ Qdrant | Cloud account or local instance |
| 🔑 Groq API Key | [console.groq.com](https://console.groq.com) — free tier available |
| 🔑 Gemini API Key | Optional — only needed for `EMBEDDING_PROVIDER=gemini` |

### Step 1 — Clone & Install

```bash
git clone https://github.com/kingspy89/production-advance-rag.git
cd production-advance-rag

python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2 — Configure Environment

```bash
cp .env.example .env
```

```dotenv
# ── Vector Database (Required) ──────────────────────────────────────
QDRANT_CLUSTER_ENDPOINT=https://your-cluster.us-east.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

# ── LLM Provider (Required) ─────────────────────────────────────────
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3.8-27b
GUARD_MODEL=qwen/qwen3.8-27b

# ── Embeddings ──────────────────────────────────────────────────────
# 'local' = 768-dim, no key needed | 'gemini' = 3072-dim, needs GEMINI_API_KEY
EMBEDDING_PROVIDER=local
GEMINI_API_KEY=

# ── Observability (Optional) ─────────────────────────────────────────
LOGFIRE_TOKEN=
LANGSMITH_TRACING=false
JUDGE_GROQ=your_groq_api_key_for_eval   # separate key for RAGAS judge
```

### Step 3 — Ingest Documents

```bash
# Index all files in DATA/ into Qdrant (--wipe recreates collection)
python -m app.ingestion.processor DATA --wipe
```

### Step 4 — Launch Services

```bash
# Terminal 1 — FastAPI backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Streamlit Chat UI
streamlit run ui/app.py --server.port 8501

# Terminal 3 (optional) — RAGAS Evaluation Dashboard
streamlit run evals/app.py --server.port 8502
```

| Service | URL |
|---------|-----|
| 🖥️ Chat Interface | http://127.0.0.1:8501 |
| 📄 Swagger API Docs | http://127.0.0.1:8000/docs |
| ❤️ Health Probe | http://127.0.0.1:8000/health |
| 🔗 Graph Topology | http://127.0.0.1:8000/graph |
| 📊 Eval Dashboard | http://127.0.0.1:8502 |

---

## 🧪 Running Evaluations

InfraAgent includes a **three-stage evaluation pipeline** accessible via a dedicated Streamlit dashboard:

```
Step 1 → Review Ground Truth      (21 golden samples from evals/golden_dataset.json)
Step 2 → Live Pipeline Execution  (queries hit POST /query — captures answers, contexts, tools)
Step 3 → RAGAS Metric Scoring     (judge LLM scores 6 metrics across 15 RAG samples)
```

```bash
# Ensure FastAPI backend is running on :8000, then:
streamlit run evals/app.py --server.port 8502

# Or run headless (CLI only, skips Streamlit):
python -m evals.pipeline
```

> 💡 **Rate limit note:** Judge LLM scoring uses 40s inter-sample and 62s inter-experiment cooldowns to stay within Groq's 6,000 TPM free-tier limit.

---

## ⚠️ Limitations & Engineering Considerations

<details>
<summary><b>🔢 Embedding Dimension Incompatibility & Re-Indexing</b></summary>

Gemini embeddings are **3,072-dimensional**; local `all-mpnet-base-v2` produces **768-dimensional** vectors. These live in incompatible semantic spaces — distance metrics cannot be computed across differing dimensions. The app validates `collection_dim == query_dim` at query time and raises a `ValueError` on mismatch. Switching providers requires full re-indexing: `python -m app.ingestion.processor DATA --wipe`.

</details>

<details>
<summary><b>🛡️ Guardrail Failure Semantics (Fail-Open)</b></summary>

Layer 1 regex runs in-process with zero latency. Layer 2 (NeMo) makes an external LLM call. If that call times out or fails with a provider exception, the system **intentionally fails open** — it logs a warning and allows the query through to the agent. This ensures infrastructure engineers are never locked out during active incidents. The fail-open path is an exception handler, not a normal routing decision.

</details>

<details>
<summary><b>💾 In-Memory Agent Checkpointing</b></summary>

`MemorySaver` persists conversation threads in process memory. **Restarting the backend resets all session state.** For persistent state across restarts, swap to `SqliteSaver` or `PostgresSaver` in `app/agents/graph.py`.

</details>

<details>
<summary><b>📊 Evaluation Dataset Scale</b></summary>

The golden dataset contains **21 curated scenarios** (15 RAG + 6 guardrail). This is sufficient for development regression testing and grounding verification, but broader domain coverage is advisable before high-volume enterprise rollouts. RAGAS scores represent a single offline evaluation session — results are not automatically persisted to disk.

</details>

<details>
<summary><b>⏱️ External Provider Latency</b></summary>

LLM generation latency depends on Groq API availability and network conditions. Cold-start requests may see minor delays while the local FlashRank ONNX cross-encoder model is lazily loaded into memory on first use.

</details>

---

## 📚 Further Resources

<div align="center">

| Resource | Link |
|----------|------|
| 📖 Interactive API Docs | `/docs` (Swagger UI, live when backend is running) |
| 🗺️ Workflow Topology | `/graph` (compiled LangGraph state machine, rendered as image) |
| 🦜 LangGraph Docs | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) |
| 🗄️ Qdrant Docs | [qdrant.tech/documentation](https://qdrant.tech/documentation/) |
| 🛡️ NeMo Guardrails | [github.com/NVIDIA/NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) |
| 📊 RAGAS Docs | [docs.ragas.io](https://docs.ragas.io/) |
| 🚀 Groq Cloud | [console.groq.com](https://console.groq.com/) |

</div>

---

<div align="center">

**Built with 🔥 by [kingspy89](https://github.com/kingspy89)**

<img src="https://img.shields.io/badge/Made%20with-Python-1f425f?style=flat-square&logo=python" />
<img src="https://img.shields.io/badge/Powered%20by-LangGraph-1C3C3C?style=flat-square&logo=langchain" />
<img src="https://img.shields.io/badge/LLM-Groq_Cloud-F55036?style=flat-square" />

*Star ⭐ the repo if you found this useful!*

</div>
