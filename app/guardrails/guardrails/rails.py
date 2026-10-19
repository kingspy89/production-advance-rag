# This file is superseded by app/guardrails/rails.py
# Re-export everything from the canonical location.
from app.guardrails.rails import initialize_rails, guard

__all__ = ["initialize_rails", "guard"]
