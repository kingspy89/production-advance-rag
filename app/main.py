# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
from dotenv import load_dotenv

load_dotenv(override=True)
token = os.getenv("LOGFIRE_TOKEN")
if not token:
    logfire.configure(send_to_logfire=False)
else:
    logfire.configure(token=token)

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard
from app.config import settings

from pydantic import BaseModel
from typing import Optional


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_rails()
    yield

# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API", lifespan=lifespan)

from fastapi.staticfiles import StaticFiles

# Add CORS Middleware for Vercel and local UI cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI at http://localhost:8000/site/
if os.path.exists("public"):
    app.mount("/site", StaticFiles(directory="public", html=True), name="site")



class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"
    api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    
    
@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}
    
    
@app.post("/query")
def query(
    request: QueryRequest,
    x_api_key: Optional[str] = Header(None),
    x_groq_api_key: Optional[str] = Header(None),
    x_gemini_api_key: Optional[str] = Header(None)
):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id
    effective_api_key = request.api_key or x_api_key or x_groq_api_key or os.getenv("GROQ_API_KEY")
    effective_gemini_key = request.gemini_api_key or x_gemini_api_key or os.getenv("GEMINI_API_KEY")

    if not effective_api_key and not settings.PORTKEY_API_KEY and not settings.GROQ_API_KEY:
        return {
            "question": q,
            "answer": "🔑 API Key Required: Please enter your Groq or Gemini API key in the UI settings panel before asking questions.",
            "thought_process": ["Validation: Missing API Key"],
            "status": "API Key Missing",
            "sources": []
        }

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
        "api_key": effective_api_key,
        "gemini_api_key": effective_gemini_key
    }
    
    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": []
            }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)
        
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": f"I apologize, but I encountered an internal error: {e}",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }

