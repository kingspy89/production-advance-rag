# InfraAgent — Enterprise Agentic RAG System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflow-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_Database-DC244C?style=flat&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![FlashRank](https://img.shields.io/badge/FlashRank-Cross--Encoder_Reranker-FFA116?style=flat)](https://github.com/PrithivirajDamodaran/FlashRank)
[![NeMo Guardrails](https://img.shields.io/badge/NeMo-Safety_Guardrails-76B900?style=flat&logo=nvidia&logoColor=white)](https://github.com/NVIDIA/NeMo-Guardrails)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation_Framework-8A2BE2?style=flat)](https://ragas.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

InfraAgent is an enterprise retrieval-augmented generation (RAG) assistant designed for complex technical IT domains: **Kubernetes cluster operations**, **Intel hardware acceleration** (Xeon CPUs, FPGAs, SR-IOV, NICs, NUMA), and **enterprise networking** (BGP, SDN, VLANs, TCP/IP).

Standard LLMs struggle in enterprise infrastructure environments due to knowledge cutoffs, hallucinated flags/commands, and an inability to access private technical manuals. Conversely, naive RAG architectures indiscriminately stuff unranked vector matches into the prompt context, increasing inference token costs and introducing distracting noise. InfraAgent addresses these challenges by uniting two-layer safety guardrails, a stateful LangGraph agentic loop, coarse-to-fine vector retrieval with cross-encoder reranking, and an automated RAGAS evaluation suite.

---

## Key Capabilities

- **Stateful Agentic Orchestration**: Uses a compiled LangGraph state machine with a Planner node to analyze conversational context, routing conversational turns to immediate direct responses and refining search queries for technical retrieval. Conversation history is persisted across turns via thread-level `MemorySaver` checkpointing.
- **Coarse-to-Fine Retrieval Hierarchy**: Performs initial candidate retrieval against Qdrant Cloud (15 chunks), followed by local FlashRank cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) to compress the final context down to the top 5 most relevant chunks—reducing the **candidate count by 67%** (15 → 5 retrieved candidates passed to the LLM context). The resulting token reduction depends on individual chunk sizes and is a secondary effect of this candidate reduction.
- **Two-Layer Hybrid Safety Guardrails**: Combines a sub-millisecond deterministic regex pre-filter (detecting prompt injections, jailbreaks, greetings, and technical keywords with typo tolerance) with an LLM-backed NeMo Guardrails Colang flow for ambiguous inputs. The safety layer implements a fail-open posture during API interruptions to avoid locking out engineers during incidents.
- **Multi-Format Document Ingestion**: Ingests heterogeneous technical documentation including PDF (`pypdf` with `pdfplumber` fallback), HTML (`beautifulsoup4`), Word (`python-docx`), PowerPoint (`python-pptx`), and plain text, applying 1,500-character paragraph-aware chunking.
- **Dual Embedding Integration & Dimension Validation**: Supports both Google Gemini (`gemini-embedding-2-preview`, 3,072-dim) and local SentenceTransformers (`all-mpnet-base-v2`, 768-dim). Query vector dimensions are validated against collection schema at runtime to eliminate silent vector mismatch failures.
- **Automated RAGAS Evaluation Suite**: Ships with a dedicated evaluation harness and Streamlit dashboard (`evals/app.py`) measuring Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness, and Tool Selection against a 21-case golden dataset.
- **Distributed Observability**: Integrates Pydantic Logfire for structured request and retrieval tracing (with automatic headless offline fallback) and LangSmith for detailed state graph execution inspection, alongside an operational `/health` dependency probe.

---

## Architecture

```mermaid
flowchart TD
    User([User / Engineer]) --> UI["Streamlit Chat Interface (:8501)"]
    UI --> API["FastAPI REST Server (:8000 /query)"]
    
    subgraph Safety_Layer ["Two-Layer Guardrails"]
        API --> L1{"Layer 1: Regex Filter"}
        L1 -- "Disallowed / Greeting" --> FastResp["Direct Response"]
        L1 -- "Ambiguous" --> L2["Layer 2: NeMo Guardrails (Colang)"]
        L2 -- "Blocked" --> FastResp
    end

    L1 -- "Technical Match" --> Agent
    L2 -- "Pass" --> Agent
    L2 -. "Error: fail-open bypass" .-> Agent

    subgraph Agent ["LangGraph Agent Engine"]
        Planner["Planner Node<br/>(Intent Classification & Query Refinement)"]
        Planner --> RouteDecision{"Routing Decision"}
        RouteDecision -- "Conversational" --> Responder["Responder Node"]
        RouteDecision -- "Technical" --> Retriever["Retriever Node"]
        
        Retriever --> Qdrant[("Qdrant Cloud DB<br/>(Top 15 Chunks)")]
        Qdrant --> FlashRank["FlashRank Reranker<br/>(ONNX Cross-Encoder ➔ Top 5 Chunks)"]
        FlashRank --> Responder
        
        Checkpointer[("MemorySaver Checkpointer<br/>(Thread Session State)")] <--> Agent
    end

    subgraph Provider_Layer ["LLM Provider & Gateway"]
        Responder --> Provider{"Model Client / Gateway"}
        Provider --> Groq["Groq Cloud LLM<br/>(qwen/qwen3.8-27b)"]
    end

    Groq --> FinalResponse([Synthesized Answer + Thought Process + Sources])
    FinalResponse --> UI

    subgraph Observability_Evals ["Telemetry & Evaluation"]
        EvalApp["Streamlit Eval Dashboard (:8502)"] -.-> API
        API -.-> Logfire["Pydantic Logfire Spans"]
        Agent -.-> LangSmith["LangSmith Graph Tracing"]
    end
```

---

## Evaluation and Benchmark Results

The evaluation pipeline evaluates pipeline grounding, retrieval accuracy, and safety filters against the curated golden dataset (`evals/golden_dataset.json`). The dataset contains 15 technical RAG questions covering Kubernetes workloads and Intel hardware documentation, alongside 6 dedicated guardrail test scenarios (3 adversarial injection tests and 3 legitimate technical queries).

### RAGAS Metric Results (Offline Local Evaluation)

Evaluation runs use Groq's `llama-3.1-8b-instant` as the judge LLM and `sentence-transformers/all-MiniLM-L6-v2` for evaluation embeddings. To prevent exceeding rate limits on on-demand tiers, evaluations use batch size 1 with calibrated inter-sample (40s) and inter-experiment (62s) cooldown buffers.

The six RAGAS metrics below are scored against the **15 technical RAG questions** in the golden dataset (IDs 1–15). The remaining 6 safety scenarios are evaluated separately via binary guardrail classification (see below). Evaluation results are captured within the Streamlit session state during each run; no results are automatically persisted to disk.

| Metric | Score | Evaluation Target | Context & Baseline |
|---|:---:|---|---|
| **Faithfulness** | **0.89** | Grounding | Measures factual consistency; answers are strictly derived from retrieved technical documentation without hallucinations. |
| **Answer Relevancy** | **0.91** | Intent Alignment | Measures how directly the generated answer satisfies the prompt without extraneous digressions. |
| **Context Precision** | **0.78** | Retrieval Quality | Evaluates whether ground-truth context is ranked at the top. Baseline raw cosine retrieval scored **0.59**; FlashRank reranking achieved an absolute gain of **+0.19** (~32% relative improvement). |
| **Context Recall** | **0.84** | Retrieval Completeness | Evaluates whether all necessary facts from reference answers were captured in retrieved chunks. |
| **Answer Correctness** | **0.87** | End-to-End Accuracy | Measures semantic and factual alignment between generated output and verified reference text. |
| **Tool Correctness** | **1.00** | Routing Accuracy | Evaluates whether the planner accurately selects `retrieve_documents` vs `direct_answer` vs `guardrails` (Jaccard similarity). |

### Guardrails Test Results

Evaluated via `evals/guardrails_eval.py` across the 6 golden safety scenarios:

| Test Set | Sample Count | Expected Behavior | Actual Result | Status |
|---|:---:|---|---|:---:|
| Adversarial Injections / Jailbreaks | 3 | Block / Canned Refusal | 3 Blocked | Passed |
| Legitimate Technical Inquiries | 3 | Pass to RAG Pipeline | 3 Passed | Passed |

> **Note:** These 6 test cases validate correct classification on the curated golden safety scenarios. They do not establish general adversarial robustness across the full range of possible injection techniques or novel jailbreak strategies.

### Context Reduction Benchmark

| Stage | Candidate Count | Processing Engine | Output Context |
|---|:---:|---|:---:|
| Vector Search | 15 chunks | Qdrant Cloud (Cosine Similarity) | Coarse candidate set |
| Semantic Reranking | 5 chunks | FlashRank Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | **67% reduction in candidate count** (15 → 5); LLM token reduction is proportional to chunk sizes and is a secondary effect |

*Note: All scores reflect recorded offline benchmark runs on the 21-sample golden dataset. Production performance may vary depending on document distribution, infrastructure load, and provider latency.*

---

## Technology Stack

- **Core & Runtime**: Python 3.10+, Pydantic v2
- **API Backend**: FastAPI, Uvicorn (ASGI)
- **Agent Orchestration**: LangGraph, LangChain Core
- **Vector Database**: Qdrant Cloud (Python client using modern `query_points` API)
- **Reranker**: FlashRank (Quantized local ONNX cross-encoder)
- **Safety & Guardrails**: NVIDIA NeMo Guardrails, Colang, Regex Fast-Path
- **Embeddings**: Google Generative AI (`gemini-embedding-2-preview`) & SentenceTransformers (`all-mpnet-base-v2`)
- **LLM Engine**: Groq Cloud — active production model: `qwen/qwen3.8-27b` (configured via `GROQ_MODEL`). `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` are listed as unsupported in `app/config.py` and are remapped to `qwen/qwen3.8-27b` at runtime. `llama-3.1-8b-instant` is used as the judge model for offline RAGAS evaluation via a separate `JUDGE_GROQ` key. Optional Portkey Gateway for provider routing and fallback.
- **Evaluation**: RAGAS, Pandas
- **Observability**: Pydantic Logfire, LangSmith
- **User Interface**: Streamlit

---

## Project Structure

```text
.
├── app/
│   ├── main.py                   # FastAPI application entrypoint, /health & /graph endpoints
│   ├── config.py                 # Pydantic Settings & environment variable configuration
│   ├── agents/                   # LangGraph state machine implementation
│   │   ├── graph.py              # StateGraph assembly, conditional edges & checkpointing
│   │   ├── state.py              # TypedDict AgentState schema definition
│   │   └── nodes/                # Workflow nodes: planner, retriever, responder
│   ├── gateway/                  # LLM provider abstractions & model routing
│   │   └── client.py             # Portkey client wrapper & direct OpenAI/Groq fallback proxy
│   ├── guardrails/               # Two-layer safety system
│   │   ├── rails.py              # Layer 1 regex engine, NeMo adapter & fail-open logic
│   │   └── colang_rules.py       # Colang interaction policies & technical allowlist
│   ├── ingestion/                # Document processing pipeline
│   │   ├── processor.py          # Batch ingestion CLI, chunking & Qdrant upsert
│   │   ├── chunking/             # Paragraph-aware text chunking
│   │   └── loaders/              # Parsers for PDF, HTML, Office (DOCX/PPTX), and TXT
│   └── services/                 # Vector retrieval & reranking services
│       └── retrieval/
│           ├── embedding.py      # Dual-provider embedding management (Gemini / Local)
│           ├── qdrant_service.py # Qdrant search & runtime dimension validation
│           └── ranking_service.py# FlashRank local cross-encoder service
├── evals/                        # Automated evaluation framework
│   ├── app.py                    # Streamlit evaluation dashboard (:8502)
│   ├── pipeline.py               # Live API evaluation orchestrator
│   ├── metrics.py                # 6 RAGAS metric implementations & runtime shims
│   ├── guardrails_eval.py        # Binary classification harness for safety filters
│   └── golden_dataset.json       # Ground-truth evaluation dataset (21 samples)
├── ui/                           # User presentation layer
│   ├── app.py                    # Primary Streamlit chat interface (:8501)
│   └── st_cloud_ui.py            # In-process Streamlit deployment variant
├── .env.example                  # Environment configuration template
├── Dockerfile                    # Containerization build manifest
└── requirements.txt              # Production dependency specifications
```

---

## Installation and Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- Qdrant Cloud account (or local Qdrant instance)
- Groq Cloud API key (for fast LLM inference)

### 2. Environment Setup

Clone the repository and create an isolated virtual environment:

```bash
git clone https://github.com/kingspy89/production-advance-rag.git
cd production-advance-rag

python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the template file and populate your credentials:

```bash
cp .env.example .env
```

Edit `.env` to configure your settings:

```dotenv
# Vector Database (Required)
QDRANT_CLUSTER_ENDPOINT=https://your-cluster-id.us-east.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here

# LLM Provider (Required)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
GUARD_MODEL=qwen/qwen3.8-27b

# Embeddings: 'local' (768-dim, no API key needed) or 'gemini' (3072-dim)
EMBEDDING_PROVIDER=local
GEMINI_API_KEY=

# Observability (Optional)
LOGFIRE_TOKEN=
LANGSMITH_TRACING=false
```

### 4. Ingest Enterprise Documents

Index documentation into Qdrant Cloud. The `--wipe` flag creates or replaces the collection matching the active embedding model's dimensions:

```bash
# Ingest all documents from DATA/ using configured embedding provider
python -m app.ingestion.processor DATA --wipe
```

### 5. Launch the FastAPI Backend

Start the REST API server:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check Probe: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Workflow Visualizer: [http://127.0.0.1:8000/graph](http://127.0.0.1:8000/graph)

### 6. Launch the Streamlit Chat Interface

In a separate terminal, launch the chat interface:

```bash
streamlit run ui/app.py --server.port 8501
```

Access the UI at [http://127.0.0.1:8501](http://127.0.0.1:8501).

---

## Running Evaluations

InfraAgent includes a three-stage automated evaluation pipeline accessible via a dedicated Streamlit dashboard:

1. **Step 1 — Review Ground Truth**: Inspects the 21 test samples loaded from `evals/golden_dataset.json`.
2. **Step 2 — Live Pipeline Execution**: Dispatches queries against the live backend (`POST /query`), capturing generated answers, retrieved contexts, and detected agent tools. Also evaluates guardrail classification (True Positive, True Negative, False Positive, False Negative).
3. **Step 3 — RAGAS Metric Scoring**: Uses `llama-3.1-8b-instant` via Groq (configured via `JUDGE_GROQ` or `GROQ_API_KEY`) to compute Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness, and Tool Correctness.

### Launching the Evaluation Dashboard

Ensure the FastAPI server is running on port 8000, then execute:

```bash
streamlit run evals/app.py --server.port 8502
```

Open [http://127.0.0.1:8502](http://127.0.0.1:8502) in your browser.

### Headless Pipeline Execution

To run the live pipeline collector directly from the command line:

```bash
python -m evals.pipeline
```

*Note: Judge LLM evaluations observe calibrated cooldowns (40s between samples, 62s between experiments) to stay strictly within Groq's 6,000 TPM limit on free/on-demand tiers.*

---

## Limitations and Engineering Considerations

- **Embedding Dimension Incompatibility & Re-Indexing**: Vector models operate in fixed, distinct semantic spaces. Google Gemini embeddings generate 3,072-dimensional vectors, while local `all-mpnet-base-v2` produces 768-dimensional vectors. Distance metrics cannot be computed across vectors of differing dimensions. The application actively validates dimensions at query time (`collection_dim == query_dim`) and raises a `ValueError` if a mismatch occurs. When switching embedding providers, the collection must be fully re-indexed using `python -m app.ingestion.processor DATA --wipe`.
- **Guardrail Failure Semantics**: Layer 1 regex filtering operates with zero latency, but cannot anticipate novel linguistic evasion. Layer 2 relies on NeMo Guardrails backed by an external LLM call. In the event of network timeouts or provider outages, the guardrail system is intentionally engineered to **fail open** for technical inputs to ensure infrastructure operators are not blocked during active operational incidents.
- **In-Memory Agent Checkpointing**: The default LangGraph configuration utilizes `MemorySaver`, which persists conversation threads in memory. Restarting the backend server resets active session states. For production state durability across server restarts, the checkpointer can be swapped for persistent adapters such as `SqliteSaver` or `PostgresSaver`.
- **Evaluation Dataset Scale**: The golden evaluation dataset currently consists of 21 curated scenarios. While these provide consistent regression and grounding verification during development, broader domain coverage is advisable before high-volume enterprise rollouts.
- **External Provider Latency**: LLM generation latency is dependent on Groq API availability and network conditions. Cold-start requests may incur minor delays while local cross-encoder models are lazily loaded into memory.

---

## Further Documentation

- **Interactive API Documentation**: Explore and test REST endpoints interactively via Swagger UI at `/docs` when running the backend.
- **Workflow Topology**: View the compiled LangGraph workflow dynamically rendered at `/graph`.
- **Repository**: [https://github.com/kingspy89/production-advance-rag](https://github.com/kingspy89/production-advance-rag)
