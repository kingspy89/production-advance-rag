import logfire
from langchain_openai import ChatOpenAI
from app.config import settings

# Active Portkey configuration if API key is provided
if settings.PORTKEY_API_KEY:
    from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
    
    # If a saved config ID (slug starting with pc-) is provided, use it to avoid inline config blocking
    if getattr(settings, "PORTKEY_CONFIG_ID", None):
        gateway_config_param = settings.PORTKEY_CONFIG_ID
    else:
        gateway_config_param = {
            "strategy": {"mode": "fallback"},
            "cache": {"mode": "simple"},
            "retry": {
                "attempts": 2,
                "on_status_codes": [429, 503]
            },
            "targets": [
                {"override_params": {"model": f"@{settings.GROQ_SLUG}/llama-3.3-70b-versatile"}},
                {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/llama-3.1-8b-instant"}},
            ]
        }
        
    portkey_client = Portkey(
        api_key=settings.PORTKEY_API_KEY,
        config=gateway_config_param
    )
else:
    from openai import OpenAI
    # Fallback directly to Groq
    _raw_client = OpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
    
    class WrappedCompletions:
        def __init__(self, completions):
            self.completions = completions
            
        def create(self, *args, **kwargs):
            if "model" not in kwargs or not kwargs["model"]:
                kwargs["model"] = settings.GROQ_MODEL
            return self.completions.create(*args, **kwargs)
            
    class WrappedChat:
        def __init__(self, chat):
            self.completions = WrappedCompletions(chat.completions)
            
    class WrappedOpenAI:
        def __init__(self, client):
            self.client = client
            self.chat = WrappedChat(client.chat)
            
    portkey_client = WrappedOpenAI(_raw_client)

def get_langchain_llm(feature: str = "rag", api_key: str = None) -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance using a user-provided API key, Portkey, or Groq default.
    """
    effective_key = api_key or settings.GROQ_API_KEY
    if not api_key and settings.PORTKEY_API_KEY:
        from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
        
        # Determine headers based on whether saved config ID or inline config is used
        if getattr(settings, "PORTKEY_CONFIG_ID", None):
            config_headers = createHeaders(
                api_key=settings.PORTKEY_API_KEY,
                config=settings.PORTKEY_CONFIG_ID,
                metadata={
                    "feature": feature,
                    "_user": "rag-system",
                    "environment": "production"
                }
            )
        else:
            inline_config = {
                "strategy": {"mode": "fallback"},
                "cache": {"mode": "simple"},
                "retry": {
                    "attempts": 2,
                    "on_status_codes": [429, 503]
                },
                "targets": [
                    {"override_params": {"model": f"@{settings.GROQ_SLUG}/llama-3.3-70b-versatile"}},
                    {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/llama-3.1-8b-instant"}},
                ]
            }
            config_headers = createHeaders(
                api_key=settings.PORTKEY_API_KEY,
                config=inline_config,
                metadata={
                    "feature": feature,
                    "_user": "rag-system",
                    "environment": "production"
                }
            )
            
        return ChatOpenAI(
            api_key=settings.PORTKEY_API_KEY,
            base_url=PORTKEY_GATEWAY_URL,
            model=f"@{settings.GROQ_SLUG}/llama-3.3-70b-versatile",
            temperature=0,
            default_headers=config_headers
        )
    else:
        return ChatOpenAI(
            api_key=effective_key,
            base_url="https://api.groq.com/openai/v1",
            model=settings.GROQ_MODEL,
            temperature=0
        )

def get_llm_client(api_key: str = None):
    """
    Returns a client instance (wrapped OpenAI or Portkey) using user api_key or default.
    """
    effective_key = api_key or settings.GROQ_API_KEY
    if not api_key and settings.PORTKEY_API_KEY:
        return portkey_client
    
    from openai import OpenAI
    raw_client = OpenAI(
        api_key=effective_key,
        base_url="https://api.groq.com/openai/v1"
    )
    return WrappedOpenAI(raw_client)

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"

