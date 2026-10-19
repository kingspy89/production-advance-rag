"""
Hybrid Guardrail layer for the Enterprise RAG assistant.

Architecture:
    1. Layer 1 (Deterministic Gate):
       - Instant regex matching for known jailbreaks, off-topic requests,
         greetings, farewells, and unambiguous technical terms.
       - Runs in 0ms with 0 tokens and zero external API dependencies.

    2. Layer 2 (NeMo Guardrails with Free LLM):
       - If the query is ambiguous, passes through NeMo Guardrails using
         qwen/qwen3.8-27b on Groq (accessible and free).
       - Evaluates semantic intent, Colang dialog flows, and complex prompts.

    3. Fault Tolerance (Fail-Open):
       - If NeMo or the Groq API encounters a timeout or network issue,
         it safely falls open to the RAG pipeline so legitimate users are never locked out.
"""

import os
import re
import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.integrations.langchain.llm_adapter import LangChainLLMAdapter

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None

# ---------------------------------------------------------------------------
# Response templates
# ---------------------------------------------------------------------------

_JAILBREAK_RESPONSE = (
    "I can only help with Kubernetes, Intel hardware, and enterprise networking."
)
_OFF_TOPIC_RESPONSE = (
    "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. "
    "I can't help with that, but ask me a technical question."
)
_GREETING_RESPONSE = (
    "Hello! I'm your Enterprise IT Assistant. I specialise in Kubernetes, Intel hardware, "
    "and enterprise networking. What can I help you with today?"
)
_FAREWELL_RESPONSE = (
    "Goodbye! Feel free to return whenever you have more enterprise IT questions. Have a great day!"
)
_CAPABILITIES_RESPONSE = (
    "I'm an Enterprise AI Assistant with deep expertise in:\n"
    "- **Kubernetes** (deployment, scaling, networking, operators)\n"
    "- **Intel Hardware** (CPUs, FPGAs, SRIOV, NICs)\n"
    "- **Enterprise Networking** (SDN, VLANs, BGP, routing)\n\n"
    "Ask me anything in these areas!"
)

# ---------------------------------------------------------------------------
# Regex pattern banks (Layer 1)
# ---------------------------------------------------------------------------

_JAILBREAK_PATTERNS = (
    r"\b(ignore|disregard|forget|override)\b.{0,80}\b(previous|earlier|system|safety|security|developer|instructions?|rules?|guidelines?)\b",
    r"\b(reveal|show|print|repeat|disclose)\b.{0,50}\b(system prompt|developer message|hidden prompt|secret instructions?)\b",
    r"\b(bypass|disable|remove|circumvent)\b.{0,50}\b(guardrails?|safety|security|filters?|restrictions?)\b",
    r"\b(jailbreak|prompt injection|developer mode|dan mode|unrestricted mode)\b",
    r"\b(you are|act as|pretend to be)\b.{0,50}\b(unrestricted|dan|evil|unfiltered)\b",
)

_OFF_TOPIC_PATTERNS = (
    r"\btell me a joke\b",
    r"\b(capital of france|world history|weather today)\b",
    r"\b(write me a poem|recommend a movie|best restaurant)\b",
    r"\b(what should i eat|help me with math homework)\b",
)

_TECHNICAL_PATTERNS = (
    r"\b(kubern(?:e|a)tes|k8s|pod|deployment|statefulset|daemonset|service|ingress|helm|container|docker)\b",
    r"\b(autoscal(?:e|ing)|cluster|namespace|operator|kubelet|kubectl|cni|crd|rbac)\b",
    r"\b(intel|xeon|atom|cpu|fpga|sriov|sr-?iov|nic|numa|dpdk|ept|paging|processor)\b",
    r"\b(network|networking|vlan|bgp|routing|router|switch|firewall|sdn|tcp|udp|dns|ip address)\b",
)

_GREETING_PATTERNS = (
    r"^(hi+|hello+|hey+|good morning|good afternoon|what'?s up|howdy)[!?. ]*$",
)

_FAREWELL_PATTERNS = (
    r"^(bye+|goodbye|see you|thanks bye|that is all|i am done|see you later)[!?. ]*$",
)

_CAPABILITIES_PATTERNS = (
    r"^(what can you do|what do you know|help|what are you|what topics do you cover|what can i ask you|what are your capabilities)[?. ]*$",
)


def _deterministic_check(message: str) -> tuple[str, str | None]:
    """
    Layer 1 fast deterministic check.
    Returns:
        ("BLOCKED", response)    - rule fired; return response, skip RAG
        ("PASS", None)           - unambiguous technical query; skip NeMo, go straight to RAG
        ("CONVERSATIONAL", resp) - greeting/farewell/capabilities; return inline
        ("AMBIGUOUS", None)      - needs semantic check via NeMo Guardrails
    """
    normalized = re.sub(r"\s+", " ", message.casefold()).strip()

    if any(re.search(p, normalized) for p in _JAILBREAK_PATTERNS):
        return "BLOCKED", _JAILBREAK_RESPONSE

    if any(re.search(p, normalized) for p in _OFF_TOPIC_PATTERNS):
        return "BLOCKED", _OFF_TOPIC_RESPONSE

    if any(re.search(p, normalized) for p in _GREETING_PATTERNS):
        return "CONVERSATIONAL", _GREETING_RESPONSE

    if any(re.search(p, normalized) for p in _FAREWELL_PATTERNS):
        return "CONVERSATIONAL", _FAREWELL_RESPONSE

    if any(re.search(p, normalized) for p in _CAPABILITIES_PATTERNS):
        return "CONVERSATIONAL", _CAPABILITIES_RESPONSE

    if any(re.search(p, normalized) for p in _TECHNICAL_PATTERNS):
        return "PASS", None

    return "AMBIGUOUS", None


# ---------------------------------------------------------------------------
# NeMo LLM initialization
# ---------------------------------------------------------------------------

def initialize_rails(api_key: str | None = None) -> None:
    """
    Build the NeMo LLMRails singleton with the working free Groq model.
    """
    global _rails

    effective_key = api_key or settings.GROQ_API_KEY
    if not effective_key:
        logfire.warning("⚠️ No GROQ_API_KEY available for NeMo Guardrails init.")
        return

    try:
        model_name = settings.GUARD_MODEL or "qwen/qwen3.8-27b"
        if model_name in ("openai/gpt-oss-120b", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"):
            model_name = "qwen/qwen3.8-27b"

        guard_llm = ChatGroq(
            api_key=effective_key,
            model=model_name,
            temperature=0
        )
        adapter = LangChainLLMAdapter(guard_llm)

        config_yaml = f"""
models:
  - type: main
    engine: groq
    model: {model_name}
{YAML_CONTENT}
"""
        config = RailsConfig.from_content(
            colang_content=COLANG_CONTENT,
            yaml_content=config_yaml
        )

        _rails = LLMRails(config, llm=adapter)
        logfire.info(f"🛡️ NeMo Guardrails initialised successfully with model: {model_name}")
    except Exception as exc:
        logfire.error(f"⚠️ NeMo Guardrails initialization error: {exc}")
        _rails = None


# ---------------------------------------------------------------------------
# Guard API
# ---------------------------------------------------------------------------

def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the hybrid guardrail gate:
    1. Layer 1: Deterministic check (instant, 0ms)
       - If explicitly blocked or conversational -> return immediately
       - If explicitly technical -> pass directly to RAG
    2. Layer 2: NeMo Guardrails (semantic intent classification with free LLM)
       - Only evaluates ambiguous queries
    """
    # --- Layer 1: Fast deterministic checks ---
    status, response = _deterministic_check(message)

    if status == "BLOCKED":
        logfire.info(f"🛡️ Layer 1 Guardrail fired (Blocked) | query='{message[:80]}'")
        return True, response

    if status == "CONVERSATIONAL":
        logfire.info(f"🛡️ Layer 1 Conversational response | query='{message[:80]}'")
        return True, response

    if status == "PASS":
        logfire.info(f"✅ Technical domain match: passing directly to RAG | query='{message[:80]}'")
        return False, None

    # --- Layer 2: NeMo Guardrails Semantic Check (ambiguous queries only) ---
    if _rails is not None:
        try:
            with logfire.span("🛡️ NeMo Guardrails Check"):
                result = _rails.generate(messages=[{"role": "user", "content": message}])

            content = result.get("content", "") if isinstance(result, dict) else str(result)
            normalized_content = content.casefold()

            fired = any(indicator.casefold() in normalized_content for indicator in RAIL_INDICATORS)
            if fired:
                logfire.info(f"🛡️ NeMo Guardrail fired | query='{message[:80]}'")
                return True, content
        except Exception as error:
            # Fault tolerance: log warning and fail open to LangGraph
            logfire.warning(f"⚠️ NeMo Guardrail check exception (falling open): {error}")

    logfire.info("✅ Guardrail passed.")
    return False, None
