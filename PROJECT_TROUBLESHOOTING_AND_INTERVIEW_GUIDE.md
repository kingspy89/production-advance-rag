# Enterprise Agentic RAG: Troubleshooting and Interview Guide

## 1. Project Summary

This project is an enterprise IT assistant for:

- Kubernetes and container platforms
- Intel hardware, CPUs, FPGAs, SR-IOV, NICs, and NUMA
- Enterprise networking, including VLAN, BGP, routing, SDN, TCP, UDP, and DNS

The main stack is:

- FastAPI and Uvicorn for the backend API
- Streamlit for the user interface
- LangGraph for planning, retrieval, response generation, and memory
- Qdrant Cloud for vector search
- Sentence Transformers for local embeddings
- Groq for language-model responses
- NeMo Guardrails plus deterministic checks for safety
- Logfire and optional LangSmith for observability

## 2. Request Flow

1. The client sends a question to `POST /query`.
2. The deterministic guardrail checks for jailbreaks, off-topic prompts, supported technical topics, and common spelling variations.
3. NeMo Guardrails performs the second safety and dialog check.
4. LangGraph plans the request as conversational or technical.
5. Technical questions are embedded and searched in Qdrant.
6. Retrieved chunks are reranked and passed to the responder.
7. The configured Groq model generates the answer.
8. The API returns the answer, status, thought process, and sources.

## 3. Problems and Solutions

### 3.1 Dependency installation used the wrong path

**Symptom:** The dependency installation command reported that `requirements.txt` could not be found.

**Cause:** The command was run from a directory different from the directory containing the requirements file in the earlier project layout.

**Solution:** Run the command from the project root and use the actual file location:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Lesson:** Confirm the current directory and inspect the repository before installing dependencies.

### 3.2 Settings import failure

**Symptom:** `ImportError: cannot import name 'settings' from 'app.config'`.

**Cause:** The settings instance name did not match imports used throughout the application.

**Solution:** The module now exposes the expected lowercase instance:

```python
settings = Settings()
```

### 3.3 Missing gateway module

**Symptom:** `ModuleNotFoundError: No module named 'app.gateway'`.

**Cause:** The gateway package was missing even though the application imported it.

**Solution:** Restored `app/gateway/` and its client abstraction for Portkey and direct Groq fallback routing.

### 3.4 Logfire startup prompt or hang

**Symptom:** Startup paused while asking for Logfire project credentials.

**Cause:** Logfire was configured interactively while no usable token was available.

**Solution:** The application checks for a token and uses local-only logging when one is unavailable. This keeps local startup non-interactive.

### 3.5 Chunking import failure

**Symptom:** `ModuleNotFoundError: No module named 'app.ingestion.chunking'`.

**Cause:** The directory name on disk did not match the import path.

**Solution:** Corrected the directory name to `chunking`.

### 3.6 Port 8000 already in use

**Symptom:** Uvicorn reported `Errno 10048` while binding to port 8000.

**Cause:** An old Python process was still listening on the port.

**Solution:** Find the process and stop only the process holding the port:

```powershell
Get-NetTCPConnection -LocalPort 8000
Stop-Process -Id <PID> -Force
```

### 3.7 Streamlit sent requests to an invalid URL

**Symptom:** The UI reported `Invalid URL '/query': No scheme supplied`.

**Cause:** `BACKEND_URL` existed but was empty. An empty environment variable does not trigger `os.getenv`'s default value.

**Solution:** The UI now treats an empty value as missing and defaults to the local backend URL:

```python
base_url = os.getenv("BACKEND_URL") or "http://localhost:8000"
```

### 3.8 Direct Groq fallback missed the model argument

**Symptom:** The UI returned an internal error and the client reported missing `messages` or `model` arguments.

**Cause:** Portkey supplies model routing through configuration, but the direct OpenAI-compatible Groq client requires the model in the request.

**Solution:** `app/gateway/client.py` injects `settings.GROQ_MODEL` when the direct fallback request does not specify a model.

### 3.9 Invalid or rate-limited Groq models

**Symptom:** Groq returned model-not-found errors or token rate-limit errors.

**Cause:** The configured model names were unavailable for the active account or exceeded the account quota.

**Solution:** The configuration was changed to account-supported models, including `openai/gpt-oss-120b` and a configured fallback model. Model availability should still be verified with the active account before deployment.

### 3.10 Qdrant connection reset

**Symptom:** Retrieval failed with `WinError 10054`, TLS handshake failures, and inability to obtain the Qdrant server version.

**Cause:** The configured remote Qdrant endpoint was unavailable, invalid, or refusing the connection.

**Solution:** Create or restart a Qdrant Cloud cluster, update `QDRANT_CLUSTER_ENDPOINT` and `QDRANT_API_KEY`, then verify connectivity:

```powershell
.\.venv\Scripts\python.exe -c "from app.config import settings; from qdrant_client import QdrantClient; c=QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=30); print(c.get_collections())"
```

### 3.11 Gemini embedding rate limits caused partial ingestion

**Symptom:** Gemini returned `RESOURCE_EXHAUSTED` and `429` errors. The ingestion job continued, but individual files were skipped after retries.

**Cause:** The ingestion code retried Gemini four times and then raised an exception. `process_file` logged the failure and moved to the next document, producing a partial index.

**Solution:** The project now defaults to local `all-mpnet-base-v2` embeddings through `EMBEDDING_PROVIDER=local` behavior in `app/config.py` and `app/services/retrieval/embedding.py`.

The collection was wiped and rebuilt with one consistent 768-dimensional embedding model:

```powershell
.\.venv\Scripts\python.exe -m app.ingestion.processor DATA --wipe
```

**Validation:** The rebuild ended with `Ingestion job completed.` and Qdrant reported `274` indexed vectors.

**Important design rule:** Do not mix Gemini's 3072-dimensional vectors and local model's 768-dimensional vectors in one Qdrant collection. Changing embedding models requires recreating and re-ingesting the collection.

### 3.12 Valid Kubernetes questions were blocked

**Symptom:** A request such as `teach me regarding kubernates` received the off-topic refusal.

**Cause:** The deterministic technical allowlist recognized `kubernetes` and `k8s`, but not the common misspelling `kubernates`. The guardrail is fail-closed when no supported technical topic matches.

**Solution:** The Kubernetes pattern now accepts both `kubernetes` and `kubernates`.

**Validation:**

- The Kubernetes request returned `blocked=False`.
- `tell me a joke` remained blocked.
- A live Kubernetes question returned `Response generated.` with deployment and pod content.

### 3.13 LangSmith authentication warnings

**Symptom:** LangSmith emitted repeated `401 Unauthorized` messages.

**Cause:** LangSmith tracing was enabled while `LANGSMITH_API_KEY` was empty.

**Solution:** Either provide a valid LangSmith key or disable hosted tracing locally:

```env
LANGSMITH_TRACING=false
```

This issue does not prevent the core RAG request from running.

## 4. Current Recovery Commands

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Start the UI in another terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8501
```

Rebuild the complete vector index after changing the embedding provider or Qdrant cluster:

```powershell
.\.venv\Scripts\python.exe -m app.ingestion.processor DATA --wipe
```

Check the indexed vector count:

```powershell
.\.venv\Scripts\python.exe -c "from app.config import settings; from qdrant_client import QdrantClient; c=QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=30); print(c.get_collection(settings.QDRANT_COLLECTION).points_count)"
```

## 5. Interview Questions and Suggested Answers

### Architecture

**Q: Explain the architecture of this project.**

A: FastAPI exposes the query API, Streamlit provides the UI, NeMo Guardrails filters unsafe or unsupported requests, LangGraph coordinates planning and memory, Qdrant performs vector search, a reranker improves relevance, and Groq generates the final grounded answer.

**Q: Why use LangGraph instead of one direct LLM call?**

A: LangGraph makes the workflow explicit and stateful. It separates planning, retrieval, response generation, and memory, which makes routing, debugging, retries, and future nodes easier to control.

**Q: What is the difference between conversational and technical routing?**

A: Greetings and simple capability questions can be answered without retrieval. Technical questions use the retrieval path so the answer is grounded in enterprise documents.

### RAG and Retrieval

**Q: What happens during ingestion?**

A: Documents are scanned, parsed by file type, split into chunks, saved as processed metadata, embedded, and upserted into Qdrant with text and source payloads.

**Q: Why must the embedding dimension stay consistent?**

A: Qdrant creates a collection with a fixed vector size. A 768-dimensional local vector cannot be inserted into a 3072-dimensional Gemini collection, and query vectors must use the same model as indexed vectors for meaningful similarity search.

**Q: Why was local embedding selected?**

A: It removes dependence on Gemini quotas, avoids ingestion failures from 429 responses, and makes local development more predictable. The tradeoff is local CPU or memory usage and model-download size.

**Q: How do you reduce hallucinations?**

A: The responder refuses unsupported technical answers when retrieval returns no documents. The prompt also receives retrieved context and source metadata instead of relying only on the model's general knowledge.

**Q: What would you improve in chunking?**

A: Use token-aware chunking, preserve document headings and page metadata, avoid oversized single chunks, and evaluate chunk size and overlap using retrieval recall and answer faithfulness metrics.

### Guardrails and Security

**Q: How does the guardrail system work?**

A: It uses deterministic checks for known jailbreak and off-topic patterns, then NeMo Guardrails for dialog and intent flows. The system fails closed if the safety layer is unavailable.

**Q: Why use deterministic checks in addition to an LLM guardrail?**

A: Deterministic rules provide predictable enforcement for high-confidence patterns and do not consume model quota. The LLM rail handles more flexible language, while the deterministic layer protects the boundary.

**Q: What is the risk of a fail-closed allowlist?**

A: Legitimate questions can be blocked because of spelling, phrasing, or new terminology. That happened with `kubernates`, so the allowlist was extended and validated. A production system should combine controlled topic classification with synonym handling and monitoring for false positives.

**Q: How would you protect the API in production?**

A: Use authentication and authorization, HTTPS, secret storage such as a managed vault, request limits, input and output logging with redaction, tenant isolation, dependency scanning, and separate service identities with least privilege.

### Reliability and Operations

**Q: How did you debug the Qdrant outage?**

A: I separated application errors from transport errors, tested the configured endpoint directly, observed TLS connection resets, verified the collection configuration was not the primary issue, then restored the cluster credentials and re-ingested the data.

**Q: What happens when Qdrant is unavailable?**

A: Retrieval returns no candidates, and the responder refuses to fabricate a technical answer. This is safer than returning an ungrounded response, although production code should also expose a clear dependency-health status and alert.

**Q: What happens when an embedding provider is rate-limited?**

A: The current default avoids the remote provider by using the local model. A more advanced design would use a provider circuit breaker and switch providers only after recreating or selecting a collection with the matching vector dimension.

**Q: How would you improve observability?**

A: Track request latency, guardrail decisions, retrieval hit count, reranker scores, token usage, model failures, Qdrant health, ingestion success and failure counts, and the percentage of answers with sources. Do not log API keys or sensitive document content.

### Testing and Tradeoffs

**Q: What did you test after the fixes?**

A: The modified Python files passed compilation, the exact misspelled Kubernetes request passed the deterministic gate, an off-topic joke remained blocked, the Qdrant collection contained 274 vectors, and a live Kubernetes request returned generated deployment and pod guidance.

**Q: What are the main tradeoffs in this project?**

A: Cloud embeddings can provide strong quality but introduce quota and network dependencies. Local embeddings improve availability and privacy but use local resources. Strict guardrails improve safety but can cause false positives. Retrieval grounding reduces hallucination but cannot answer topics absent from the knowledge base.

**Q: What would you do before production deployment?**

A: Remove exposed credentials and rotate them, move secrets to a vault, add automated tests for guardrails and retrieval outages, add health checks and alerts, pin and scan dependencies, configure authenticated tracing, add evaluation datasets, and document backup and re-index procedures.

## 6. Important Security Note

API keys must never be committed to Git, pasted into tickets, or shared in chat. Any credential that has been exposed should be revoked and replaced immediately. Use environment variables locally and a secret manager in deployment.

## 7. Current Status

- Backend API: working locally on port 8000
- Streamlit UI: configured for the local backend
- Qdrant collection: rebuilt and verified with 274 vectors
- Embeddings: local `all-mpnet-base-v2`, 768 dimensions
- Kubernetes questions: passing the guardrail, including the tested spelling variation
- Off-topic and jailbreak checks: still enforced
- LangSmith hosted tracing: requires a key or should be disabled locally
