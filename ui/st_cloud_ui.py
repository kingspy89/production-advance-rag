import os
import sys
import streamlit as st
import time
import uuid
import logfire


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)


# Initialize Logfire
try:
    token = st.secrets.get("LOGFIRE_TOKEN", os.getenv("LOGFIRE_TOKEN"))
    if not token:
        logfire.configure(send_to_logfire=False)
    else:
        logfire.configure(token=token)
    logfire.instrument_requests()   # propagates trace context to the FastAPI backend
    LOGFIRE_STATUS = "Connected & Tracing"
except Exception:
    LOGFIRE_STATUS = "Standby (No Token)"


# Streamlit Cloud can run the agent in the same process, so no separate
# backend URL or Render service is required for this deployment.
for secret_name in (
    "GROQ_API_KEY",
    "GROQ_FALLBACK_API_KEY",
    "QDRANT_API_KEY",
    "QDRANT_CLUSTER_ENDPOINT",
    "GEMINI_API_KEY",
    "EMBEDDING_PROVIDER",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
):
    if secret_name in st.secrets:
        os.environ[secret_name] = str(st.secrets[secret_name])

from app.agents.graph import rag_agent
from app.config import settings
from app.guardrails import guard, initialize_rails


@st.cache_resource
def initialize_backend():
    initialize_rails()
    return True


def execute_query(query: str, thread_id: str, api_key: str | None) -> dict:
    """Run the guarded LangGraph directly inside the Streamlit process."""
    effective_api_key = api_key or settings.GROQ_API_KEY
    if not effective_api_key and not settings.PORTKEY_API_KEY:
        return {
            "answer": "API key required. Add GROQ_API_KEY to Streamlit Secrets.",
            "thought_process": ["Validation: Missing API Key"],
            "sources": [],
        }

    rail_fired, rail_response = guard(query)
    if rail_fired:
        return {
            "answer": rail_response,
            "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
            "sources": [],
        }

    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "current_query": query,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
        "api_key": effective_api_key,
        "gemini_api_key": settings.GEMINI_API_KEY,
    }
    final_output = rag_agent.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )
    return {
        "answer": final_output.get("final_answer"),
        "thought_process": final_output.get("plan"),
        "sources": final_output.get("documents", []),
    }

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)

# --- AVATARS ---
AI_AVATAR = "🤖"
USER_AVATAR = "👤"

# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Agent OS")
    st.markdown("---")

    st.subheader("🔑 API Key Configuration")
    user_api_key = st.text_input(
        "Enter Groq / Gemini API Key",
        type="password",
        value=st.session_state.get("user_api_key", ""),
        help="Paste your API key here to enable query processing.",
        key="cloud_api_key_input"
    )
    if user_api_key:
        st.session_state.user_api_key = user_api_key
        st.success("API Key set! Ready for queries.")
    else:
        st.warning("⚠️ Please paste your API key above to start.")

    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")
    
    if st.button("🗑️ Clear History & Memory", width="stretch", type="primary"):
        logfire.warning(f"🗑️ Memory Wipe Triggered for session: {st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

settings.GROQ_API_KEY = user_api_key or settings.GROQ_API_KEY
if settings.GROQ_API_KEY:
    initialize_backend()

# --- MAIN CHAT ---
st.title("🤖 Enterprise Agentic Assistant")

# Display history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about your documentation..."):
    # START TRACE: User Interaction
    with logfire.span("💬 User Chat Interaction", user_query=prompt, session_id=st.session_state.session_id):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            data = {}
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    with logfire.span("🧠 Running Agent Pipeline"):
                        data = execute_query(
                            prompt,
                            st.session_state.session_id,
                            st.session_state.get("user_api_key"),
                        )

                    steps = data.get("thought_process", [])
                    for step in steps:
                        st.markdown(f"⚙️ {step}", unsafe_allow_html=False)

                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)

                except Exception as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error("Backend Offline.")
                    st.stop()

            # Answer streaming — outside status so it's always visible
            answer_placeholder = st.empty()
            full_answer = data.get("answer", "No response.")

            curr_text = ""
            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(curr_text + "▌")
                time.sleep(0.005)
            answer_placeholder.markdown(full_answer)

            # Sources — outside status so they're visible after it collapses
            sources = data.get("sources", [])
            if sources:
                with st.expander(f"📄 Retrieved Context ({len(sources)} chunks)"):
                    for i, source in enumerate(sources):
                        st.caption(f"Chunk {i + 1}")
                        st.info(source)
            else:
                st.caption("ℹ️ No context retrieved — conversational response.")

            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            logfire.info("✅ Chat cycle completed successfully.")