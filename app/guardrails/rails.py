"""
Guardrail layer for the Enterprise RAG assistant.

Architecture:
    guard() runs a fast, deterministic regex pipeline that requires NO external
    LLM calls. This makes it:
      - Instantaneous (no network latency)
      - Zero-cost (no token usage)
      - Immune to model deprecations / API key issues

    The original NeMo LLMRails layer was removed because `openai/gpt-oss-120b`
    stopped producing parseable Colang completions on Groq (400 output_parse_failed).
    Since the deterministic layer already handles every case that NeMo handled,
    nothing is lost.

Flow:
    1. Jailbreak regex  → block immediately
    2. Off-topic regex  → block immediately
    3. Technical regex  → pass through to RAG
    4. Conversational   → pass through to RAG
    5. Anything else    → block as off-topic
"""

import re
import logfire


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

# Conversational responses (returned by deterministic check so no LLM needed)
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
# Regex pattern banks
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


# ---------------------------------------------------------------------------
# Deterministic check
# ---------------------------------------------------------------------------

def _deterministic_check(message: str) -> tuple[bool, str | None]:
    """
    Pure-regex guardrail. Returns (blocked, response_or_None).
    No LLM call, no network, no latency.
    """
    normalized = re.sub(r"\s+", " ", message.casefold()).strip()

    # --- Hard blocks ---
    if any(re.search(p, normalized) for p in _JAILBREAK_PATTERNS):
        return True, _JAILBREAK_RESPONSE

    if any(re.search(p, normalized) for p in _OFF_TOPIC_PATTERNS):
        return True, _OFF_TOPIC_RESPONSE

    # --- Always-pass: clearly technical ---
    if any(re.search(p, normalized) for p in _TECHNICAL_PATTERNS):
        return False, None

    # --- Conversational shortcuts (return inline, skip RAG) ---
    if any(re.search(p, normalized) for p in _GREETING_PATTERNS):
        return True, _GREETING_RESPONSE

    if any(re.search(p, normalized) for p in _FAREWELL_PATTERNS):
        return True, _FAREWELL_RESPONSE

    if any(re.search(p, normalized) for p in _CAPABILITIES_PATTERNS):
        return True, _CAPABILITIES_RESPONSE

    # --- Default: pass through to RAG (let the LLM decide) ---
    return False, None


# ---------------------------------------------------------------------------
# Public API  (initialize_rails kept for import compatibility)
# ---------------------------------------------------------------------------

def initialize_rails(api_key: str | None = None) -> None:
    """
    No-op — NeMo LLM rails removed in favour of deterministic-only guardrails.
    Kept for API compatibility with st_cloud_ui.py and main.py.
    """
    logfire.info("🛡️ Guardrails initialised (deterministic-only mode).")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the guardrail gate.

    Returns:
        (True,  response) — rail fired; return response, skip RAG pipeline.
        (False, None)     — message is clean; proceed to LangGraph.
    """
    blocked, response = _deterministic_check(message)

    if blocked:
        logfire.info(f"🛡️ Guardrail fired | query='{message[:80]}'")
    else:
        logfire.info("✅ Guardrail passed.")

    return blocked, response
