import re

import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None

_JAILBREAK_PATTERNS = (
    r"\b(ignore|disregard|forget|override)\b.{0,80}\b(previous|earlier|system|safety|security|developer|instructions?|rules?|guidelines?)\b",
    r"\b(reveal|show|print|repeat| disclose)\b.{0,50}\b(system prompt|developer message|hidden prompt|secret instructions?)\b",
    r"\b(bypass|disable|remove|circumvent)\b.{0,50}\b(guardrails?|safety|security|filters?|restrictions?)\b",
    r"\b(jailbreak|prompt injection|developer mode|dan mode|unrestricted mode)\b",
    r"\b(you are|act as|pretend to be)\b.{0,50}\b(unrestricted|ไม่ได้|dan|evil|unfiltered)\b",
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

_ALLOWED_CONVERSATIONAL_PATTERNS = (
    r"^(hi|hello|hey|good morning|good afternoon|what's up|howdy)[!. ]*$",
    r"^(what can you do|what do you know|help|what are you|what topics do you cover|what can i ask you|what are your capabilities)[?. ]*$",
    r"^(bye|goodbye|see you|thanks bye|that is all|i am done|see you later)[!. ]*$",
)

_JAILBREAK_RESPONSE = (
    "I can only help with Kubernetes, Intel hardware, and enterprise networking."
)
_OFF_TOPIC_RESPONSE = (
    "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. "
    "I can't help with that, but ask me a technical question."
)
_UNAVAILABLE_RESPONSE = "The safety gate is unavailable, so this request cannot be processed."


def _deterministic_check(message: str) -> tuple[bool, str | None]:
    normalized = re.sub(r"\s+", " ", message.casefold()).strip()

    if any(re.search(pattern, normalized) for pattern in _JAILBREAK_PATTERNS):
        return True, _JAILBREAK_RESPONSE
    if any(re.search(pattern, normalized) for pattern in _OFF_TOPIC_PATTERNS):
        return True, _OFF_TOPIC_RESPONSE
    if any(re.search(pattern, normalized) for pattern in _TECHNICAL_PATTERNS):
        return False, None
    if any(re.search(pattern, normalized) for pattern in _ALLOWED_CONVERSATIONAL_PATTERNS):
        return False, None
    return True, _OFF_TOPIC_RESPONSE


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses llama-3.1-8b-instant for fast intent classification at the gate —
    the heavier llama-3.3-70b-versatile is reserved for the RAG pipeline.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised.")
    
    


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    blocked, response = _deterministic_check(message)
    if blocked:
        logfire.info(f"🛡️ Deterministic guardrail fired | query='{message[:80]}'")
        return True, response

    if _rails is None:
        logfire.error("⚠️ Guardrails not initialised — failing closed.")
        return True, _UNAVAILABLE_RESPONSE

    try:
        with logfire.span("🛡️ Guardrails Check"):
            result = _rails.generate(messages=[{"role": "user", "content": message}])

        content = result.get("content", "") if isinstance(result, dict) else str(result)
        normalized_content = content.casefold()

        fired = any(indicator.casefold() in normalized_content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
    except Exception as error:
        logfire.error(f"🛡️ Guardrails failed; rejecting request: {error}")
        return True, _UNAVAILABLE_RESPONSE
