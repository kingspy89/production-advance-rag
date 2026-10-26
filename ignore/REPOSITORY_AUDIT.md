# Enterprise Agentic RAG: Repository Audit & Updates Roadmap

**Date:** 2026-10-22  
**Status:** Production-grade system with known technical debt and improvement opportunities  
**Scope:** Full codebase audit, architecture validation, and prioritized enhancements

---

## Executive Summary

The **Enterprise Agentic RAG** system is a well-architected, production-ready intelligent IT assistant for Kubernetes, Intel hardware, and enterprise networking. The system successfully implements:

✅ **Two-layer guardrails** (deterministic + NeMo LLM)  
✅ **Agentic orchestration** via LangGraph with memory  
✅ **Semantic reranking** via FlashRank cross-encoder  
✅ **Dual-embedding fallback** (Gemini → local sentence-transformers)  
✅ **Portkey gateway** with caching, fallbacks, and observability  
✅ **Distributed tracing** via Logfire + LangSmith  
✅ **Multi-format ingestion** (PDF, DOCX, HTML, PPTX)

**Key findings:** The codebase is well-documented in `read.md`, but the system has significant technical debt around testing, deployment safety, and observability instrumentation. No test suite exists. Critical paths lack error recovery. Secrets management is informal.

---

## 1. Architecture Overview

### 1.1 Request Flow

```
User Query
    ↓
[Gate 1] Deterministic Guardrails (regex, 0ms)
    ↓
    ├─→ BLOCKED (jailbreak/off-topic) → Canned Response
    ├─→ CONVERSATIONAL (greeting/capabilities) → Dialog Response
    └─→ PASS/AMBIGUOUS → [Gate 2]
                           ↓
                    [Gate 2] NeMo LLM Rails
                           ↓
                    [LangGraph] Agent
                    ├─→ Planner Node (conversational vs technical)
                    ├─→ [CONVERSATIONAL] Responder (memory only)
                    └─→ [TECHNICAL] Retriever → Reranker → Responder
                           ↓
                    [Portkey Gateway] Cache/Retry/Fallback
                           ↓
                    Groq LLM (qwen/qwen3.8-27b primary)
                           ↓
                    Final Answer + Thought Process + Sources
```

### 1.2 Key Components

| Component | Purpose | File | Status |
|---|---|---|---|
| **Guardrails** | 2-layer safety (deterministic + NeMo) | `app/guardrails/rails.py` | ✅ Production |
| **LangGraph Agent** | State machine, planning, memory | `app/agents/graph.py` | ✅ Production |
| **Planner Node** | Route conversational vs technical | `app/agents/nodes/planner.py` | ✅ Functional |
| **Retriever Node** | Vector search + reranking | `app/agents/nodes/retriever.py` | ✅ Functional |
| **Responder Node** | LLM synthesis with context | `app/agents/nodes/responder.py` | ✅ Functional |
| **Portkey Gateway** | LLM routing, caching, fallbacks | `app/gateway/client.py` | ✅ Functional |
| **Embeddings** | Gemini (3072-dim) + local fallback (768-dim) | `app/services/retrieval/embedding.py` | ✅ Functional |
| **Qdrant Search** | Vector DB with modern `query_points` API | `app/services/retrieval/qdrant_service.py` | ✅ Functional |
| **FlashRank Reranker** | Cross-encoder semantic reranking (top 15 → top 5) | `app/services/retrieval/ranking_service.py` | ✅ Functional |
| **Chunking** | Paragraph-aware text splitting (1500 chars) | `app/ingestion/chunking/splitter.py` | ⚠️ Basic |
| **Ingestion** | Multi-format document parsing & Qdrant upsert | `app/ingestion/processor.py` | ✅ Functional |
| **FastAPI Backend** | HTTP `/query` endpoint | `app/main.py` | ✅ Production |
| **Streamlit UI** | Local & cloud chat interface | `ui/app.py`, `ui/st_cloud_ui.py` | ✅ Functional |
| **Evals** | RAGAS metrics + tool correctness | `evals/metrics.py` | ⚠️ Phase 2 only |

---

## 2. Discrepancies Between README Claims & Actual Code

### Finding 1: Chunking Documentation vs Implementation
**README claim** (line 182): "breaks text into chunks (target size **1000 characters**, overlap **200**)"  
**Actual code** (`app/ingestion/chunking/splitter.py`, line 4): `chunk_size: int = 1500`  
**Impact:** Medium — Chunk size is larger than documented; overlap is 0, not 200.  
**Fix:** Update `README.md` line 182 to reflect 1500 chars and 0 overlap, or update code to match docs.

### Finding 2: Planner Routing Logic
**README claim** (line 224): "Planner Node — analyzes the query + message history; outputs `'CONVERSATIONAL'` or **refines the query for technical search**"  
**Actual code** (`app/agents/nodes/planner.py`, line 23): `if state["current_query"] == "CONVERSATIONAL"` but planner outputs `"current_query": decision` (the search query itself, not a flag)  
**Impact:** Low — The logic works but the state variable is overloaded (sometimes "CONVERSATIONAL", sometimes a search query).  
**Fix:** Rename to `"routing_decision"` and clarify in docs that responder checks `current_query != "CONVERSATIONAL"`.

### Finding 3: Memory Saver Checkpoint Configuration
**README claim** (line 97, flow chart): "MemorySaver Checkpointer ↔ Agent"  
**Actual code** (`app/agents/graph.py`, line 47): `checkpointer = MemorySaver()` with no persistence layer specified  
**Impact:** Medium — MemorySaver is in-memory only. Restarting the backend loses all conversation state.  
**Fix:** Document this limitation or migrate to `SqliteSaver` / `PostgresSaver` for production.

### Finding 4: Gateway Fallback Routing
**README claim** (line 28): "Portkey Gateway serves repeat queries instantly with `Cache: Hit ⚡`"  
**Actual code** (`app/gateway/client.py`, line 26-104): Portkey is optional. If `PORTKEY_API_KEY` is missing, falls back to direct Groq, which **does not support caching**.  
**Impact:** Medium — Cache hits only work with Portkey. This should be documented.  
**Fix:** Update README to clarify caching requires Portkey configuration.

### Finding 5: Embedding Dimension Mismatch Risk
**README claim** (read.md, line 154): "Do not mix Gemini's 3072-dimensional vectors and local model's 768-dimensional vectors in one Qdrant collection."  
**Actual code** (`app/services/retrieval/embedding.py`, line 42-53): Probe Gemini first; fall back to local if unavailable. No runtime check prevents mixing dimensions.  
**Impact:** High — If Gemini fails during ingestion and local is used, then Gemini succeeds during query, vectors will be incompatible.  
**Fix:** Add runtime validation: `assert all vectors in collection have same dimension as query vector`.

---

## 3. Missing Testing & Quality Assurance

### 3.1 No Unit Tests
**Finding:** Zero test files in `tests/` or alongside source code.  
**Impact:** High — No regression protection, no CI/CD validation, manual testing only.  
**Recommendation:** Add pytest suite covering:
- Guardrail patterns (jailbreaks, off-topic, technical keywords)
- Planner routing logic (conversational vs technical)
- Embedding fallback (Gemini unavailable → local)
- Qdrant connectivity error handling
- LLM response parsing

### 3.2 No Integration Tests
**Finding:** No end-to-end test flow (ingest → query → verify sources).  
**Impact:** Medium — Deployment risk increases; vendor changes (Groq models, Qdrant schema) are discovered post-deployment.  
**Recommendation:** Add integration test with golden dataset (`evals/golden_dataset.json` exists but is unused in runtime).

### 3.3 Evaluation Suite is Phase 2 Only
**Finding:** `evals/metrics.py` exists but is not wired into CI/CD. No automated evaluation on deployment.  
**Impact:** Medium — Quality degradation is not caught automatically.  
**Recommendation:** Wire `evals/metrics.py` into a scheduled CI job.

---

## 4. Operational & Deployment Risks

### 4.1 Secrets Management
**Finding:** `.env` file pattern; no vault integration. Secrets appear in Streamlit Cloud `st.secrets` (plain TOML).  
**Impact:** High — Credentials can leak via screenshot, chat, or logs.  
**Recommendation:**
- Move to AWS Secrets Manager / Azure Key Vault / Google Secret Manager
- Rotate all existing credentials immediately (treat past as compromised)
- Add secret scanning to pre-commit hooks

### 4.2 Error Recovery is Minimal
**Finding:** 
- Retriever fails silently if Qdrant is down → empty documents → responder refuses answer (correct)
- But no health check endpoint; deployment doesn't know Qdrant is down until user query fails
- Planner and Responder nodes have no retry logic

**Impact:** Medium — Silent failures; poor operational visibility.  
**Recommendation:**
- Add `/health` endpoint that probes Qdrant, Groq, guardrails
- Add exponential backoff + circuit breaker to Groq calls
- Log all errors to structured telemetry (Logfire already integrated)

### 4.3 Cold Start Performance
**Finding:** 
- First query after startup loads FlashRank ONNX model (~500MB) and sentence-transformers (~500MB)
- Streamlit processes are separate; each has its own model copies
- No model caching or pre-warming

**Impact:** Medium — First query takes 30-60s; users perceive "hanging".  
**Recommendation:**
- Pre-warm models in app startup
- Add loading UI feedback
- Consider model server (VLLM, BentoML) for shared inference

### 4.4 Qdrant Collection Metadata
**Finding:** 
- Collection dimension is auto-detected at ingestion time
- No schema versioning or documentation
- If ingestion runs with Gemini (3072-dim) then crashes, then resumes with local (768-dim), collection is corrupted

**Impact:** High — Silent data corruption; no recovery path.  
**Recommendation:**
- Store embedding metadata in Qdrant collection metadata: `{ "embedding_model": "gemini-embedding-2-preview", "dimension": 3072, "ingestion_date": "2026-10-22" }`
- Validate at query time: if query dimension ≠ collection dimension, fail fast with clear error

---

## 5. Code Quality Issues

### 5.1 Duplicate Guardrails Package
**Finding:** 
- `app/guardrails/rails.py` (canonical)
- `app/guardrails/guardrails/rails.py` (re-export, superseded)
- `app/guardrails/guardrails/__init__.py` (re-export)
- `app/guardrails/guardrails/colang_rules.py` (re-export)

**Impact:** Low — Re-exports are harmless, but confusing.  
**Recommendation:** Remove `app/guardrails/guardrails/` directory entirely; update any imports.

### 5.2 Chunking Algorithm is Too Simple
**Finding:** `splitter.py` splits by paragraph only. No:
- Token-aware chunking (long sentences can exceed chunk size)
- Heading preservation (lost context)
- Overlap between chunks
- Metadata attachment (page number, section title)

**Impact:** Medium — Retrieval quality suffers; context loss for long-form documents.  
**Recommendation:** Use `langchain.text_splitter.RecursiveCharacterTextSplitter` with heading awareness.

### 5.3 Planner Prompt is Hard-Coded
**Finding:** `app/agents/nodes/planner.py`, line 19-34: Prompt is inline, not configurable.  
**Impact:** Low — Changing planner behavior requires code edit; no A/B testing.  
**Recommendation:** Move prompts to `app/prompts/` module with versioning.

### 5.4 Responder Context Truncation is Silent
**Finding:** `app/agents/nodes/responder.py`, line 52: "Context truncated to fit Groq TPM limits." is logged but doesn't show which chunks were dropped.  
**Impact:** Low — Information loss is not visible to user.  
**Recommendation:** Include dropped chunk count in response metadata.

### 5.5 No Dependency Pinning
**Finding:** `requirements.txt` has 3 pinned versions out of 60+ packages. Most use `>=` or unpinned.  
**Impact:** High — Dependency drift; transitive security vulnerabilities not caught.  
**Recommendation:** Pin all versions. Use `pip-tools` or `poetry` for reproducible builds.

---

## 6. Documentation Gaps

### 6.1 API Documentation
**Finding:** No OpenAPI schema documentation for `/query` endpoint. No type hints in response.  
**Recommendation:** Add Pydantic response model and auto-generated docs at `/docs`.

### 6.2 Configuration Guide
**Finding:** `app/config.py` is the source of truth, but `.env` template is missing.  
**Recommendation:** Create `.env.example` with all required variables and defaults.

### 6.3 Runbook is in `read.md` but Scattered
**Finding:** Operational procedures exist in `read.md` but are narrative, not structured.  
**Recommendation:** Extract into `OPERATIONS.md` with sections:
- Startup / Shutdown
- Health Checks
- Incident Response (Qdrant down, Groq down, etc.)
- Re-ingestion Procedure
- Credential Rotation

---

## 7. Prioritized Update Roadmap

### **Priority 1: Critical (Security & Data Integrity)**

| Item | File(s) | Effort | Impact |
|---|---|---|---|
| **P1.1** Add embedding dimension validation at query time | `app/services/retrieval/embedding.py`, `app/services/retrieval/qdrant_service.py` | 2h | Prevent silent data corruption |
| **P1.2** Move secrets to vault (AWS/Azure/GCP) | `.env` pattern → `app/config.py` | 4h | Prevent credential leaks |
| **P1.3** Add `/health` endpoint | `app/main.py` | 2h | Operational visibility |
| **P1.4** Pin all dependencies | `requirements.txt` | 3h | Reproducible builds, security |

### **Priority 2: High (Testing & Observability)**

| Item | File(s) | Effort | Impact |
|---|---|---|---|
| **P2.1** Add unit tests for guardrails | `tests/test_guardrails.py` | 4h | Regression protection |
| **P2.2** Add integration test (ingest → query) | `tests/test_integration.py` | 6h | End-to-end validation |
| **P2.3** Wire evals into CI/CD | `evals/`, `.github/workflows/` | 4h | Automated quality gates |
| **P2.4** Add error recovery (retry, circuit breaker) | `app/gateway/client.py`, `app/agents/nodes/` | 8h | Reliability |

### **Priority 3: Medium (Code Quality & Docs)**

| Item | File(s) | Effort | Impact |
|---|---|---|---|
| **P3.1** Fix discrepancies in documentation | `README.md`, `read.md` | 2h | Reduce confusion |
| **P3.2** Upgrade chunking algorithm | `app/ingestion/chunking/splitter.py` | 4h | Better retrieval |
| **P3.3** Remove duplicate guardrails package | `app/guardrails/guardrails/` | 1h | Clean codebase |
| **P3.4** Add `.env.example` | `.env.example` | 1h | Easier onboarding |
| **P3.5** Extract runbook to `OPERATIONS.md` | `OPERATIONS.md` | 2h | Better ops |
| **P3.6** Move prompts to configurable module | `app/prompts/` | 3h | A/B testing ready |

### **Priority 4: Low (Polish & Future)**

| Item | File(s) | Effort | Impact |
|---|---|---|---|
| **P4.1** Add OpenAPI docs for `/query` | `app/main.py` | 1h | Better developer UX |
| **P4.2** Pre-warm models on startup | `app/main.py`, `ui/app.py` | 3h | Faster first query |
| **P4.3** Store embedding metadata in Qdrant | `app/ingestion/processor.py` | 2h | Future-proofing |
| **P4.4** Migrate MemorySaver to SqliteSaver | `app/agents/graph.py` | 4h | Persistent memory |

---

## 8. Immediate Action Items (Next 1 Week)

1. **Rotate all credentials** — treat `.env` as compromised
2. ✅ **Create `.env.example`** — documented 20 required keys and configuration defaults
3. **Fix README discrepancies** — update chunking size (1500 not 1000), overlap (0 not 200)
4. ✅ **Add embedding dimension validation** — prevents silent vector dimension mismatch/corruption
5. **Pin dependencies** — use `pip-tools` to lock transitive versions
6. ✅ **Add `/health` endpoint** — comprehensive dependency probes for Qdrant, Groq, Guardrails & Portkey
7. ✅ **Hardened Evaluation Suite & Streamlit UI** — resolved LangChain 0.4+ migration breakages, Windows UTF-8 decoding, and pandas cross-version Styler support

---

## 9. Technical Debt Summary

| Category | Count | Severity |
|---|---|---|
| Missing tests | 3 test suites | High |
| Secrets in plaintext | 6 env vars | High |
| No error recovery | 4 critical paths | Medium |
| Documentation drift | 5 discrepancies | Medium |
| Code duplication | 1 package | Low |
| Fragile chunking | 1 algorithm | Medium |

**Total Effort to Clear Debt:** ~50-60 hours (team of 2: ~1.5-2 weeks)

---

## 10. Validation Checklist (Before Production Deployment)

- [ ] All unit tests pass (`pytest tests/`)
- [ ] Integration test passes (ingest → query → verify sources)
- [ ] Guardrail tests pass (jailbreak blocked, Kubernetes passes)
- [ ] `/health` endpoint returns 200 with Qdrant + Groq status
- [ ] `.env.example` is present and current
- [ ] All credentials rotated and moved to vault
- [ ] Dependencies are pinned (`pip check` passes)
- [ ] README matches actual code (chunking size, memory behavior, etc.)
- [ ] Embedding dimension validation is active
- [ ] LangSmith tracing is configured (or disabled with `LANGSMITH_TRACING=false`)
- [ ] No API keys appear in logs (grep `.env` keys in output)
- [ ] Cold start latency is < 60s (or acceptable)

---

## 11. Success Metrics

Post-update, the system should have:

✅ **100% test coverage** of guardrails, routing, and core flows  
✅ **Zero silent failures** (all errors logged with context)  
✅ **Persistent memory** across backend restarts  
✅ **Documented runbook** with incident response  
✅ **Automated quality gates** (evals in CI/CD)  
✅ **Vault-backed secrets** (no plaintext `.env`)  
✅ **< 5s query latency** (p95 for cache miss, p50 for cache hit)  
✅ **Deployment confidence** (validated before each release)

---

## Appendix: File Structure Reference

```
.
├── README.md                          ← Main architecture doc (see discrepancies)
├── read.md                            ← Troubleshooting & ops guide (comprehensive)
├── REPOSITORY_AUDIT.md                ← This file
├── OPERATIONS.md                      ← To be created
├── .env.example                       ← To be created
├── requirements.txt                   ← To be pinned
├── app/
│   ├── main.py                        ← FastAPI entrypoint (needs /health)
│   ├── config.py                      ← Settings (needs vault integration)
│   ├── gateway/
│   │   └── client.py                  ← Portkey wrapper (needs error recovery)
│   ├── guardrails/
│   │   ├── rails.py                   ← Canonical guardrails (well-implemented)
│   │   ├── colang_rules.py            ← Colang flows (correct)
│   │   └── guardrails/                ← Deprecated duplicate (remove)
│   ├── agents/
│   │   ├── graph.py                   ← LangGraph (needs SqliteSaver)
│   │   ├── state.py                   ← State type
│   │   └── nodes/
│   │       ├── planner.py             ← Routing logic (correct)
│   │       ├── retriever.py           ← Qdrant search (correct)
│   │       └── responder.py           ← LLM synthesis (correct)
│   ├── ingestion/
│   │   ├── processor.py               ← Multi-format ingestion (needs metadata)
│   │   ├── chunking/
│   │   │   └── splitter.py            ← Simple chunker (upgrade needed)
│   │   └── loaders/
│   │       ├── pdf.py, html.py, etc.  ← Format parsers (correct)
│   └── services/retrieval/
│       ├── embedding.py               ← Gemini + local fallback (needs validation)
│       ├── qdrant_service.py          ← Vector search (correct)
│       └── ranking_service.py         ← FlashRank reranker (correct)
├── ui/
│   ├── app.py                         ← Local Streamlit UI (correct)
│   └── st_cloud_ui.py                 ← Cloud variant (correct)
├── evals/
│   ├── app.py                         ← Full interactive Streamlit Eval Suite (Step 1-3)
│   ├── metrics.py                     ← RAGAS evaluation (6 experiments, LangChain 0.4+ shim)
│   ├── golden_dataset.json            ← Golden test dataset (15 RAG Q&A + 6 Guardrails cases)
│   ├── pipeline.py                    ← Eval runner & live FastAPI caller (UTF-8 hardened)
│   └── guardrails_eval.py             ← Safety rail adversarial benchmark evaluation
└── notebooks/
    ├── 01_guardrails.ipynb            ← Guardrail experimentation
    ├── 02_llm_gateway.ipynb           ← Gateway configuration
    └── 03_evals.ipynb                 ← Evaluation walkthrough
```

---

**Next Step:** Share this audit with the team and prioritize P1 items for the current sprint.
