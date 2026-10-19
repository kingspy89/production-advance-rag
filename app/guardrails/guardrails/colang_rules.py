# This inner package is superseded. All guardrail logic lives in app/guardrails/rails.py
# Re-export from canonical location so any stale import still works.
from app.guardrails.rails import initialize_rails, guard

__all__ = ["initialize_rails", "guard"]
